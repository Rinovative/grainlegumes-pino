"""
===============================================================================
learning_training_checkpoint.py
===============================================================================
Persist and restore exact completed-epoch training checkpoints.

Responsibilities:
  - Define the current checkpoint schema and run-identity binding
  - Capture model, optimizer, scheduler, scaler, loss, RNG, and loader state
  - Validate checkpoint identity before mutating runtime objects
  - Publish best and last checkpoints through atomic replacement

Design principles:
  - ``last_checkpoint.pt`` is the only continuation source
  - ``best_checkpoint.pt`` is the inference and artifact source
  - Resume is exact at a completed epoch boundary, never mid-epoch
  - Missing or mismatched state fails closed before runtime restoration

This module does NOT:
  - Allocate run directories, choose resume policy, or validate mutable run status
  - Construct models, optimizers, schedulers, scalers, losses, or dataloaders
  - Repair or reinterpret malformed checkpoints
===============================================================================
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from torch import nn

from src import common

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

CheckpointRole = Literal["best", "last"]
CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_IDENTITY_KEYS = frozenset(
    {
        "task",
        "task_contract_digest",
        "effective_config_digest",
        "resume_contract_digest",
        "dataset_fingerprints",
        "split_membership_digests",
        "objective",
    }
)
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_role",
        "identity",
        "completed_epoch",
        "next_epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "amp_enabled",
        "scaler_state_dict",
        "loss_state_dict",
        "best_metric",
        "best_epoch",
        "objective_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_loader_generator_state",
        "train_sampler_state_dict",
    }
)


def _config_identity_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Copy the config semantics that must remain fixed across exact resume.

    Resolved locations, requested device policy, and terminal epoch count are
    removed because resume may relocate execution metadata or extend duration.
    Every remaining scientific, data, optimizer, and lifecycle field retains
    identity significance.
    """
    view = copy.deepcopy(dict(config))
    view.pop("paths", None)
    run = view.get("run")
    if isinstance(run, dict):
        run.pop("device", None)
    training = view.get("training")
    if isinstance(training, dict):
        training.pop("epochs", None)
    return view


def config_digest(config: Mapping[str, Any]) -> str:
    """
    Return the scientific effective-config digest for a persisted config.

    The requested device policy is retained in ``config.yaml`` as operational
    provenance but excluded from scientific identity. Concrete runtime device
    facts never enter this function.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved saved config.

    Returns
    -------
    str
        Canonical SHA-256 digest excluding only the operational device policy.

    """
    view = copy.deepcopy(dict(config))
    run = view.get("run")
    if isinstance(run, dict):
        run.pop("device", None)
    return common.serialization.canonical_json_sha256(view)


def resume_contract_digest(config: Mapping[str, Any]) -> str:
    """
    Return the config digest excluding explicitly allowed runtime changes.

    The only excluded semantic duration field is ``training.epochs``. Resolved
    paths and ``run.device`` are runtime location/execution metadata.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved saved or resume-requested config.

    Returns
    -------
    str
        Canonical SHA-256 digest of the resume-fixed comparison view.

    """
    return common.serialization.canonical_json_sha256(_config_identity_view(config))


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one mapping value or raise with checkpoint context."""
    if not isinstance(value, Mapping):
        msg = f"Checkpoint {label} must be a mapping."
        raise TypeError(msg)
    return value


_OBJECTIVE_KEYS = frozenset({"id", "kind", "space", "fields", "reduction", "direction"})


def _validate_objective(value: Any, *, label: str) -> dict[str, Any]:
    """
    Validate and isolate one exact persisted objective identity.

    All six canonical fields are required, unknown fields are rejected, ordered
    output fields must be unique, and direction is limited to minimize/maximize.
    The returned deep copy prevents caller mutation from altering admitted state.
    """
    objective = _required_mapping(value, label=label)
    missing = sorted(_OBJECTIVE_KEYS.difference(objective))
    unknown = sorted(set(objective).difference(_OBJECTIVE_KEYS))
    if missing or unknown:
        msg = f"Checkpoint {label} objective keys do not match. Missing: {missing}. Unknown: {unknown}."
        raise ValueError(msg)
    for key in ("id", "kind", "space", "reduction"):
        if not isinstance(objective[key], str) or not objective[key]:
            msg = f"Checkpoint {label}.{key} must be a non-empty string."
            raise TypeError(msg)
    fields = objective["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) and field for field in fields):
        msg = f"Checkpoint {label}.fields must be a non-empty exact field list."
        raise TypeError(msg)
    if len(fields) != len(set(fields)):
        msg = f"Checkpoint {label}.fields contains duplicates."
        raise ValueError(msg)
    if objective["direction"] not in {"minimize", "maximize"}:
        msg = f"Checkpoint {label}.direction must be 'minimize' or 'maximize'."
        raise ValueError(msg)
    return copy.deepcopy(dict(objective))


def _validate_checkpoint_identity(value: Any, *, label: str) -> dict[str, Any]:
    """
    Validate and isolate the complete current checkpoint identity schema.

    Task/config digests, train/OOD fingerprints, all split membership digests,
    and the full objective must be present with no unknown keys. Validation is
    performed before any runtime component is restored.
    """
    identity = _required_mapping(value, label=label)
    missing = sorted(_CHECKPOINT_IDENTITY_KEYS.difference(identity))
    unknown = sorted(set(identity).difference(_CHECKPOINT_IDENTITY_KEYS))
    if missing or unknown:
        msg = f"Checkpoint {label} keys do not match. Missing: {missing}. Unknown: {unknown}."
        raise ValueError(msg)
    for key in (
        "task",
        "task_contract_digest",
        "effective_config_digest",
        "resume_contract_digest",
    ):
        if not isinstance(identity[key], str) or not identity[key]:
            msg = f"Checkpoint {label}.{key} must be a non-empty string."
            raise TypeError(msg)
    objective = _validate_objective(identity["objective"], label=f"{label}.objective")
    for key, roles in (
        ("dataset_fingerprints", {"train", "ood"}),
        ("split_membership_digests", {"train", "eval", "ood"}),
    ):
        values = _required_mapping(identity[key], label=f"{label}.{key}")
        if set(values) != roles or not all(isinstance(item, str) and item for item in values.values()):
            msg = f"Checkpoint {label}.{key} must contain exactly non-empty values for {sorted(roles)}."
            raise ValueError(msg)
    result = copy.deepcopy(dict(identity))
    result["objective"] = objective
    return result


def build_checkpoint_identity(
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    *,
    persisted_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the immutable run identity stored in every checkpoint.

    Parameters
    ----------
    config : Mapping[str, Any]
        Effective runtime config.
    split_indices : Mapping[str, Any]
        Validated saved split contract.
    persisted_config : Mapping[str, Any] | None, optional
        Immutable config.yaml payload. Resume may use a larger runtime epoch
        limit while retaining this original persisted identity.

    Returns
    -------
    dict[str, Any]
        Task, config, dataset, split, and objective identity.

    """
    task = config.get("task")
    task_contract = _required_mapping(config.get("task_contract"), label="config task_contract")
    evaluation = _required_mapping(config.get("evaluation"), label="config evaluation")
    objective = _validate_objective(
        evaluation.get("objective"),
        label="config evaluation.objective",
    )
    task_digest = task_contract.get("digest")
    if not isinstance(task, str) or not task:
        msg = "Config task must be a non-empty string for checkpoint identity."
        raise TypeError(msg)
    if not isinstance(task_digest, str) or not task_digest:
        msg = "Config task_contract.digest must be a non-empty string for checkpoint identity."
        raise TypeError(msg)
    metadata = _required_mapping(split_indices.get("metadata"), label="split metadata")
    datasets = _required_mapping(metadata.get("datasets"), label="split metadata.datasets")
    memberships = _required_mapping(metadata.get("membership_digests"), label="split metadata.membership_digests")

    fingerprints: dict[str, str] = {}
    for role in ("train", "ood"):
        dataset_identity = _required_mapping(datasets.get(role), label=f"split dataset {role}")
        fingerprint = dataset_identity.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            msg = f"Split dataset {role!r} must contain a fingerprint."
            raise TypeError(msg)
        fingerprints[role] = fingerprint

    membership_values: dict[str, str] = {}
    for role in ("train", "eval", "ood"):
        digest = memberships.get(role)
        if not isinstance(digest, str) or not digest:
            msg = f"Split membership {role!r} must contain a digest."
            raise TypeError(msg)
        membership_values[role] = digest

    persisted = dict(persisted_config or config)
    return _validate_checkpoint_identity(
        {
            "task": task,
            "task_contract_digest": task_digest,
            "effective_config_digest": config_digest(persisted),
            "resume_contract_digest": resume_contract_digest(persisted),
            "dataset_fingerprints": fingerprints,
            "split_membership_digests": membership_values,
            "objective": objective,
        },
        label="built identity",
    )


def _train_loader_generator(train_loader: DataLoader[Any]) -> torch.Generator:
    """Return the explicit generator controlling shuffled train membership."""
    generator = getattr(train_loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        sampler = getattr(train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
    if not isinstance(generator, torch.Generator):
        msg = "Exact resume requires an explicit torch.Generator on the training DataLoader."
        raise TypeError(msg)
    return generator


def _sampler_state(train_loader: DataLoader[Any]) -> Mapping[str, Any] | None:
    """Return optional explicit sampler state without inventing sampler state."""
    sampler = getattr(train_loader, "sampler", None)
    state_dict = getattr(sampler, "state_dict", None)
    if not callable(state_dict):
        return None
    state = state_dict()
    if not isinstance(state, Mapping):
        msg = "Training sampler state_dict() must return a mapping."
        raise TypeError(msg)
    return dict(state)


def _cuda_device_index(device: torch.device) -> int:
    """Return the already concrete CUDA index without querying runtime defaults."""
    if device.type != "cuda" or device.index is None:
        msg = f"Checkpoint CUDA state requires an indexed concrete CUDA device, got {device}."
        raise ValueError(msg)
    return device.index


def _active_cuda_devices(
    model: nn.Module,
    *,
    runtime_device: torch.device | None = None,
) -> tuple[torch.device, ...]:
    """
    Discover sorted concrete CUDA devices owned by this runtime or model.

    Only an explicitly supplied indexed runtime device and devices already
    owning model parameters or buffers participate. The helper never queries a
    process-default CUDA device or initializes availability-based fallback.
    """
    indices: set[int] = set()
    if runtime_device is not None:
        if not isinstance(runtime_device, torch.device):
            msg = f"runtime_device must be a concrete torch.device, got {runtime_device!r}."
            raise TypeError(msg)
        if runtime_device.type == "cuda":
            indices.add(_cuda_device_index(runtime_device))
    for tensors in (model.parameters(), model.buffers()):
        for tensor in tensors:
            if tensor.device.type == "cuda":
                indices.add(_cuda_device_index(tensor.device))
    return tuple(torch.device("cuda", index) for index in sorted(indices))


def _capture_cuda_rng_states(devices: tuple[torch.device, ...]) -> list[torch.Tensor]:
    """Capture RNG state only for CUDA devices used by this runtime."""
    return [torch.cuda.get_rng_state(device).clone() for device in devices]


def _set_cuda_rng_states(states: list[torch.Tensor], devices: tuple[torch.device, ...]) -> None:
    """Restore ordered CUDA RNG states onto the current active devices."""
    for state, device in zip(states, devices, strict=True):
        torch.cuda.set_rng_state(state, device=device)


def _attempt_rollback(
    label: str,
    action: Callable[[], Any],
    errors: list[str],
) -> None:
    """
    Attempt one transactional rollback action and retain any failure.

    Arbitrary component exceptions are converted to bounded diagnostic strings
    so later rollback actions still run. The original restore error remains the
    primary exception unless the collected failures make rollback incomplete.
    """
    try:
        action()
    except Exception as error:  # noqa: BLE001 -- rollback must retain arbitrary component failures
        errors.append(f"{label}: {type(error).__name__}: {error}")


def make_checkpoint(
    *,
    role: CheckpointRole,
    identity: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    loss: nn.Module,
    best_metric: float | None,
    best_epoch: int | None,
    objective_history: list[dict[str, Any]],
    train_loader: DataLoader[Any],
    runtime_device: torch.device,
) -> dict[str, Any]:
    """
    Capture a complete epoch-boundary checkpoint payload.

    Parameters
    ----------
    role : {"best", "last"}
        Enforced checkpoint lifecycle role.
    identity : Mapping[str, Any]
        Immutable run identity from :func:`build_checkpoint_identity`.
    completed_epoch : int
        Positive one-based epoch that completed fully.
    global_step : int
        Number of completed optimizer steps.
    model, optimizer, scheduler, scaler, loss : Any
        Stateful runtime objects.
    amp_enabled : bool
        Whether mixed-precision scaling is active.
    best_metric : float | None
        Best finite objective so far, if any.
    best_epoch : int | None
        One-based epoch owning ``best_metric``.
    objective_history : list[dict[str, Any]]
        Ordered finite evaluation history.
    train_loader : DataLoader
        Shuffled loader whose generator state controls the next epoch.
    runtime_device : torch.device
        Active concrete execution device. CUDA RNG is captured only for this device and
        any CUDA devices that actually own model parameters or buffers.

    Returns
    -------
    dict[str, Any]
        Complete validated checkpoint.

    """
    if role not in {"best", "last"}:
        msg = f"Unknown checkpoint role {role!r}."
        raise ValueError(msg)
    if completed_epoch <= 0:
        msg = f"completed_epoch must be positive, got {completed_epoch}."
        raise ValueError(msg)
    if global_step < 0:
        msg = f"global_step must be non-negative, got {global_step}."
        raise ValueError(msg)
    if (best_metric is None) != (best_epoch is None):
        msg = "best_metric and best_epoch must either both be present or both be absent."
        raise ValueError(msg)
    if best_metric is not None and not math.isfinite(float(best_metric)):
        msg = f"best_metric must be finite, got {best_metric}."
        raise ValueError(msg)
    if role == "best" and best_metric is None:
        msg = "A best checkpoint requires a finite best objective."
        raise ValueError(msg)
    if not isinstance(runtime_device, torch.device) or runtime_device.type not in {"cpu", "cuda"}:
        msg = f"Checkpoint capture requires one concrete CPU or CUDA torch.device, got {runtime_device!r}."
        raise TypeError(msg)
    if amp_enabled and scaler is None:
        msg = "AMP-enabled checkpoints require a scaler."
        raise ValueError(msg)
    if amp_enabled and runtime_device.type != "cuda":
        msg = "AMP-enabled checkpoints require a concrete CUDA runtime device."
        raise ValueError(msg)

    validated_identity = _validate_checkpoint_identity(identity, label="identity")
    generator = _train_loader_generator(train_loader)
    active_cuda_devices = _active_cuda_devices(model, runtime_device=runtime_device)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "identity": validated_identity,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "amp_enabled": bool(amp_enabled),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "loss_state_dict": loss.state_dict(),
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": best_epoch,
        "objective_history": copy.deepcopy(objective_history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),  # noqa: NPY002 -- exact process-global RNG state
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": _capture_cuda_rng_states(active_cuda_devices),
        "train_loader_generator_state": generator.get_state(),
        "train_sampler_state_dict": _sampler_state(train_loader),
    }


def validate_checkpoint(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_role: CheckpointRole,
    scheduler_expected: bool,
    amp_expected: bool,
    require_best: bool,
) -> dict[str, Any]:
    """
    Validate a checkpoint completely before runtime state restoration.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Untrusted loaded checkpoint mapping.
    expected_identity : Mapping[str, Any]
        Exact task/config/dataset/split/objective identity for the current run.
    expected_role : {"best", "last"}
        Required lifecycle role.
    scheduler_expected : bool
        Whether scheduler state must be present.
    amp_expected : bool
        Whether AMP and scaler state must be present.
    require_best : bool
        Whether finite best-objective state is mandatory.

    Returns
    -------
    dict[str, Any]
        Isolated shallow mapping admitted under the current schema.

    Raises
    ------
    TypeError
        If required mappings, scalar types, or RNG/sampler state shapes are invalid.
    ValueError
        If schema, identity, role, progress, best selection, or component presence
        contradicts the expected run contract.

    Notes
    -----
    Validation mutates no runtime object. RNG payloads are tested on temporary
    generators before the checkpoint can reach transactional restoration.

    """
    missing = sorted(_CHECKPOINT_KEYS.difference(payload))
    unknown = sorted(set(payload).difference(_CHECKPOINT_KEYS))
    if missing or unknown:
        msg = f"Checkpoint schema mismatch. Missing keys: {missing}. Unknown keys: {unknown}."
        raise ValueError(msg)
    schema_version = payload["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != CHECKPOINT_SCHEMA_VERSION:
        msg = f"Unsupported checkpoint schema_version {schema_version!r}. Expected integer {CHECKPOINT_SCHEMA_VERSION}."
        raise ValueError(msg)
    if payload["checkpoint_role"] != expected_role:
        msg = f"Checkpoint role mismatch: expected {expected_role!r}, got {payload['checkpoint_role']!r}."
        raise ValueError(msg)
    identity = _validate_checkpoint_identity(payload["identity"], label="identity")
    expected = _validate_checkpoint_identity(expected_identity, label="expected identity")
    if identity != expected:
        msg = "Checkpoint run identity is incompatible with config, task, dataset, split, or objective state."
        raise ValueError(msg)

    completed_epoch = payload["completed_epoch"]
    next_epoch = payload["next_epoch"]
    global_step = payload["global_step"]
    for label, value, minimum in (
        ("completed_epoch", completed_epoch, 1),
        ("next_epoch", next_epoch, 2),
        ("global_step", global_step, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            msg = f"Checkpoint {label} must be an integer >= {minimum}."
            raise TypeError(msg)
    if next_epoch != completed_epoch + 1:
        msg = "Checkpoint next_epoch must immediately follow completed_epoch."
        raise ValueError(msg)

    for key in ("model_state_dict", "optimizer_state_dict", "loss_state_dict"):
        _required_mapping(payload[key], label=key)
    scheduler_state = payload["scheduler_state_dict"]
    if scheduler_expected != isinstance(scheduler_state, Mapping):
        msg = "Checkpoint scheduler state is absent or present inconsistently with the resolved config."
        raise ValueError(msg)
    amp_enabled = payload["amp_enabled"]
    if not isinstance(amp_enabled, bool) or amp_enabled != amp_expected:
        msg = f"Checkpoint AMP mode mismatch: expected {amp_expected}, got {amp_enabled!r}."
        raise ValueError(msg)
    scaler_state = payload["scaler_state_dict"]
    if amp_expected != isinstance(scaler_state, Mapping):
        msg = "Checkpoint scaler state is absent or present inconsistently with active AMP."
        raise ValueError(msg)

    best_metric = payload["best_metric"]
    best_epoch = payload["best_epoch"]
    if (best_metric is None) != (best_epoch is None):
        msg = "Checkpoint best_metric/best_epoch presence is inconsistent."
        raise ValueError(msg)
    if require_best and best_metric is None:
        msg = "Checkpoint requires a finite best objective but none is stored."
        raise ValueError(msg)
    if best_metric is not None:
        if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)) or not math.isfinite(float(best_metric)):
            msg = "Checkpoint best_metric must be finite."
            raise ValueError(msg)
        if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 1 <= best_epoch <= completed_epoch:
            msg = "Checkpoint best_epoch must identify a completed epoch."
            raise ValueError(msg)

    history = payload["objective_history"]
    if not isinstance(history, list):
        msg = "Checkpoint objective_history must be a list."
        raise TypeError(msg)
    history_values: list[tuple[int, float]] = []
    previous_epoch = 0
    for entry in history:
        if not isinstance(entry, Mapping) or set(entry) != {"epoch", "objective_id", "value"}:
            msg = "Checkpoint objective_history entries must contain exactly epoch, objective_id, and value."
            raise ValueError(msg)
        value = entry["value"]
        epoch = entry["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not previous_epoch < epoch <= completed_epoch:
            msg = "Checkpoint objective_history epochs must be strictly increasing completed epochs."
            raise ValueError(msg)
        if entry["objective_id"] != identity["objective"]["id"]:
            msg = "Checkpoint objective_history id does not match checkpoint identity."
            raise ValueError(msg)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            msg = "Checkpoint objective_history values must be finite."
            raise ValueError(msg)
        history_values.append((epoch, float(value)))
        previous_epoch = epoch
    if (best_metric is None) != (not history_values):
        msg = "Checkpoint best state and objective_history presence are inconsistent."
        raise ValueError(msg)
    if best_metric is not None:
        selected = (
            min(history_values, key=lambda item: item[1])
            if identity["objective"]["direction"] == "minimize"
            else max(history_values, key=lambda item: item[1])
        )
        if selected != (best_epoch, float(best_metric)):
            msg = "Checkpoint best state does not match its direction-aware objective history."
            raise ValueError(msg)
        if expected_role == "best" and completed_epoch != best_epoch:
            msg = "A best checkpoint must be captured at its selected best epoch."
            raise ValueError(msg)

    python_rng_state = payload["python_rng_state"]
    numpy_rng_state = payload["numpy_rng_state"]
    if not isinstance(python_rng_state, tuple):
        msg = "Checkpoint Python RNG state must be a tuple."
        raise TypeError(msg)
    if not isinstance(numpy_rng_state, tuple):
        msg = "Checkpoint NumPy RNG state must be a tuple."
        raise TypeError(msg)
    try:
        random.Random().setstate(python_rng_state)  # noqa: S311 -- validate saved non-cryptographic RNG state
        np.random.RandomState().set_state(numpy_rng_state)
    except (TypeError, ValueError) as error:
        msg = "Checkpoint Python or NumPy RNG state is invalid."
        raise ValueError(msg) from error
    for key in ("torch_cpu_rng_state", "train_loader_generator_state"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
            msg = f"Checkpoint {key} must be a one-dimensional CPU byte tensor."
            raise TypeError(msg)
        try:
            torch.Generator().set_state(value)
        except RuntimeError as error:
            msg = f"Checkpoint {key} is not a valid Torch generator state."
            raise ValueError(msg) from error
    cuda_states = payload["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or not all(
        isinstance(state, torch.Tensor) and state.device.type == "cpu" and state.dtype == torch.uint8 and state.ndim == 1 for state in cuda_states
    ):
        msg = "Checkpoint CUDA RNG states must be one-dimensional CPU byte tensors."
        raise TypeError(msg)
    sampler_state = payload["train_sampler_state_dict"]
    if sampler_state is not None and not isinstance(sampler_state, Mapping):
        msg = "Checkpoint train_sampler_state_dict must be a mapping or None."
        raise TypeError(msg)
    return dict(payload)


def save_checkpoint(payload: Mapping[str, Any], path: Path | str) -> Path:
    """
    Atomically publish one already validated checkpoint payload.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Complete best- or last-checkpoint mapping.
    path : Path | str
        Exact role-qualified checkpoint path.

    Returns
    -------
    Path
        Atomically published checkpoint path.

    """
    return common.serialization.atomic_torch_save(dict(payload), path)


def load_checkpoint(
    path: Path | str,
    *,
    expected_identity: Mapping[str, Any],
    expected_role: CheckpointRole,
    scheduler_expected: bool,
    amp_expected: bool,
    require_best: bool,
) -> dict[str, Any]:
    """
    Load and validate one checkpoint under the strict saved schema.

    Parameters
    ----------
    path : Path | str
        Required checkpoint path.
    expected_identity : Mapping[str, Any]
        Exact task/config/dataset/split/objective identity.
    expected_role : {"best", "last"}
        Required lifecycle role.
    scheduler_expected : bool
        Whether scheduler state is required.
    amp_expected : bool
        Whether scaler state is required.
    require_best : bool
        Whether finite selected-best state must already exist.

    Returns
    -------
    dict[str, Any]
        Fully validated checkpoint mapping.

    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        msg = f"Required {expected_role} checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        msg = f"Checkpoint must contain a mapping: {checkpoint_path}"
        raise TypeError(msg)
    return validate_checkpoint(
        payload,
        expected_identity=expected_identity,
        expected_role=expected_role,
        scheduler_expected=scheduler_expected,
        amp_expected=amp_expected,
        require_best=require_best,
    )


def restore_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    loss: nn.Module,
    train_loader: DataLoader[Any],
) -> dict[str, Any]:
    """
    Restore a validated last checkpoint transactionally.

    Schema, identity, runtime capability, and rollback-state checks happen
    before mutation. If any component, loader, sampler, or RNG restore fails,
    every supplied runtime object and process RNG is restored to its exact
    pre-call state before the original failure is re-raised.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Candidate strict ``last`` checkpoint.
    expected_identity : Mapping[str, Any]
        Immutable identity required by the current resumed run.
    model, optimizer, scheduler, scaler, loss : Any
        Current runtime components whose state may be replaced.
    amp_enabled : bool
        Whether scaler state is required by the resumed runtime.
    train_loader : DataLoader
        Loader owning the explicit shuffle generator and optional sampler state.

    Returns
    -------
    dict[str, Any]
        Restored progress, best-objective state, and isolated objective history.

    Raises
    ------
    TypeError
        If checkpoint or current loader/sampler capabilities cannot support exact restore.
    RuntimeError
        If CUDA state cardinality differs or rollback after a restore failure is incomplete.

    Notes
    -----
    Process-global Python, NumPy, Torch CPU, and active-device CUDA RNG states
    are part of both the restore transaction and its rollback snapshot.

    """
    checkpoint = validate_checkpoint(
        payload,
        expected_identity=expected_identity,
        expected_role="last",
        scheduler_expected=scheduler is not None,
        amp_expected=amp_enabled,
        require_best=False,
    )
    generator = _train_loader_generator(train_loader)
    sampler_state = checkpoint["train_sampler_state_dict"]
    sampler = getattr(train_loader, "sampler", None)
    sampler_load_state = getattr(sampler, "load_state_dict", None)
    current_sampler_state = _sampler_state(train_loader)
    if sampler_state is not None:
        if not callable(sampler_load_state):
            msg = "Checkpoint contains sampler state but the current sampler cannot restore it."
            raise TypeError(msg)
        if current_sampler_state is None:
            msg = "Checkpoint sampler restore requires a snapshot-capable current sampler."
            raise TypeError(msg)

    active_cuda_devices = _active_cuda_devices(model)
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if active_cuda_devices and cuda_states and len(cuda_states) != len(active_cuda_devices):
        msg = "Checkpoint CUDA RNG state count does not match the model runtime devices."
        raise RuntimeError(msg)

    snapshot = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "scaler_state_dict": copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        "loss_state_dict": copy.deepcopy(loss.state_dict()),
        "train_loader_generator_state": generator.get_state().clone(),
        "train_sampler_state_dict": copy.deepcopy(current_sampler_state),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),  # noqa: NPY002 -- transactional process RNG snapshot
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": _capture_cuda_rng_states(active_cuda_devices),
    }

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        loss.load_state_dict(checkpoint["loss_state_dict"], strict=True)

        generator.set_state(checkpoint["train_loader_generator_state"])
        if sampler_state is not None and callable(sampler_load_state):
            sampler_load_state(sampler_state)

        random.setstate(cast("tuple[Any, ...]", checkpoint["python_rng_state"]))
        np.random.set_state(  # noqa: NPY002 -- restore the exact process-global RNG state
            cast("tuple[Any, ...]", checkpoint["numpy_rng_state"]),
        )
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if active_cuda_devices and cuda_states:
            _set_cuda_rng_states(cuda_states, active_cuda_devices)
    except BaseException as error:
        rollback_errors: list[str] = []
        _attempt_rollback(
            "model",
            lambda: model.load_state_dict(snapshot["model_state_dict"], strict=True),
            rollback_errors,
        )
        _attempt_rollback(
            "optimizer",
            lambda: optimizer.load_state_dict(snapshot["optimizer_state_dict"]),
            rollback_errors,
        )
        if scheduler is not None:
            _attempt_rollback(
                "scheduler",
                lambda: scheduler.load_state_dict(snapshot["scheduler_state_dict"]),
                rollback_errors,
            )
        if scaler is not None:
            _attempt_rollback(
                "scaler",
                lambda: scaler.load_state_dict(snapshot["scaler_state_dict"]),
                rollback_errors,
            )
        _attempt_rollback(
            "loss",
            lambda: loss.load_state_dict(snapshot["loss_state_dict"], strict=True),
            rollback_errors,
        )
        _attempt_rollback(
            "train loader generator",
            lambda: generator.set_state(snapshot["train_loader_generator_state"]),
            rollback_errors,
        )
        if current_sampler_state is not None and callable(sampler_load_state):
            _attempt_rollback(
                "train sampler",
                lambda: sampler_load_state(snapshot["train_sampler_state_dict"]),
                rollback_errors,
            )
        _attempt_rollback(
            "Python RNG",
            lambda: random.setstate(cast("tuple[Any, ...]", snapshot["python_rng_state"])),
            rollback_errors,
        )
        _attempt_rollback(
            "NumPy RNG",
            lambda: np.random.set_state(  # noqa: NPY002 -- transactional process RNG rollback
                cast("tuple[Any, ...]", snapshot["numpy_rng_state"]),
            ),
            rollback_errors,
        )
        _attempt_rollback(
            "Torch CPU RNG",
            lambda: torch.set_rng_state(snapshot["torch_cpu_rng_state"]),
            rollback_errors,
        )
        _attempt_rollback(
            "Torch CUDA RNG",
            lambda: _set_cuda_rng_states(snapshot["torch_cuda_rng_states"], active_cuda_devices),
            rollback_errors,
        )
        if rollback_errors:
            details = "; ".join(rollback_errors)
            msg = f"Checkpoint restore failed and runtime rollback was incomplete: {details}"
            raise RuntimeError(msg) from error
        raise

    return {
        "completed_epoch": checkpoint["completed_epoch"],
        "next_epoch": checkpoint["next_epoch"],
        "global_step": checkpoint["global_step"],
        "best_metric": checkpoint["best_metric"],
        "best_epoch": checkpoint["best_epoch"],
        "objective_history": copy.deepcopy(checkpoint["objective_history"]),
    }
