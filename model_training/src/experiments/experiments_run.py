"""
===============================================================================
experiments_run.py
===============================================================================
Allocate, initialize, execute, resume, and validate saved experiment runs.

Responsibilities:
  - Allocate fresh run leaves exclusively and transition explicit statuses
  - Derive stable labeled seeds and configure deterministic execution early
  - Persist immutable run inputs and mutable lifecycle outputs atomically
  - Enforce allowed resume config changes and exact last-checkpoint continuation
  - Validate the completed/loadable contract consumed by inference and artifacts

Design principles:
  - Only explicit resume may open an existing run directory
  - Config, split, and normalizer artifacts are immutable after fresh creation
  - Best and last checkpoints have distinct enforced lifecycle roles
  - Generic orchestration consumes task/config interfaces without field names

This module does NOT:
  - Define task physics, model architectures, loss formulas, or metric mathematics
  - Infer resume from an existing directory or treat observers as authoritative state
  - Render or publish post-training analysis artifacts
===============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src import common, datasets, learning

from . import experiments_tracking as tracking
from .config import experiments_config_loader as config_loader

RUN_SUMMARY_SCHEMA_VERSION = 1
RUN_STATUSES = frozenset(
    {
        "initializing",
        "running",
        "completed",
        "pruned",
        "nonfinite_pruned",
        "oom_pruned",
        "recoverable_failed",
        "failed",
        "interrupted",
    }
)
_TERMINAL_RUN_STATUSES = RUN_STATUSES.difference({"initializing", "running"})
_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"initializing"}),
    "initializing": frozenset({"running", "failed", "interrupted"}),
    "running": frozenset({"running", *_TERMINAL_RUN_STATUSES}),
    "interrupted": frozenset({"running", "failed"}),
    "completed": frozenset({"running"}),
    "failed": frozenset(),
    "pruned": frozenset(),
    "nonfinite_pruned": frozenset(),
    "oom_pruned": frozenset(),
    "recoverable_failed": frozenset(),
}
_SEED_LABELS = ("process", "model_init", "split", "loader", "worker", "tuner")
_MISSING = object()
_MAX_CONFIG_DIFFERENCES = 12
_RUN_WRITER_LOCKS_DIRNAME = ".run-writer-locks"


class RunLifecycleError(RuntimeError):
    """
    Represent a saved-run lifecycle or identity contract violation.

    Raised at allocation, transition, resume, and completed-run admission
    boundaries. It distinguishes invalid persisted run state from configuration
    schema errors and ordinary missing-file failures.
    """


def _run_writer_lock_path(run_dir: Path | str) -> Path:
    """Return the persistent sibling lock path for one canonical run leaf."""
    path = Path(run_dir).expanduser().resolve(strict=False)
    return path.parent / _RUN_WRITER_LOCKS_DIRNAME / f"{path.name}.lock"


@contextmanager
def run_writer_lease(
    run_dir: Path | str,
    *,
    blocking: bool = False,
) -> Iterator[Path]:
    """
    Hold the exclusive writer lease for one run lifecycle.

    The lock file is a sibling of the run directory so fresh allocation and
    resume prevalidation can use the same lease before touching run contents.
    Training fails fast by default; coordinated artifact readers may wait for
    the current run writer by setting ``blocking=True``.

    Parameters
    ----------
    run_dir : Path | str
        Canonical run leaf, which need not exist yet for fresh allocation.
    blocking : bool, optional
        Wait for the current owner when true; otherwise fail immediately.

    Yields
    ------
    pathlib.Path
        Expanded absolute run path protected by the lease.

    Raises
    ------
    RunLifecycleError
        If a non-blocking lease is already owned by another writer.

    Notes
    -----
    The persistent sibling lock is not run content and is not removed when the
    lease closes; the underlying OS lock, rather than file existence, owns exclusion.

    """
    path = Path(run_dir).expanduser().resolve(strict=False)
    try:
        with common.locking.exclusive_file_lock(
            _run_writer_lock_path(path),
            blocking=blocking,
        ):
            yield path
    except common.locking.FileLockUnavailableError as error:
        msg = f"Run already has an active writer lease: {path}"
        raise RunLifecycleError(msg) from error


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def derive_subseed(seed: int, label: str) -> int:
    """
    Derive a stable label-qualified non-negative Torch-compatible seed.

    Parameters
    ----------
    seed : int
        Base run seed.
    label : str
        Non-empty stream label.

    Returns
    -------
    int
        Deterministic 63-bit sub-seed independent of derivation call order.

    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        msg = f"seed must be a non-negative integer, got {seed!r}."
        raise ValueError(msg)
    if not isinstance(label, str) or not label:
        msg = "seed label must be a non-empty string."
        raise ValueError(msg)
    payload = f"run-subseed-v1\0{seed}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def build_seed_plan(seed: int) -> dict[str, int]:
    """
    Return the complete stable labeled seed plan for one run seed.

    Derivation is order-independent and covers process, model initialization,
    splitting, loader order, workers, and tuning. A cryptographic-prefix
    collision among the maintained labels raises ``RuntimeError``.
    """
    plan = {label: derive_subseed(seed, label) for label in _SEED_LABELS}
    if len(set(plan.values())) != len(plan):
        msg = "Stable labeled seed derivation produced a collision."
        raise RuntimeError(msg)
    return plan


def seed_process(seed: int, *, device: torch.device) -> None:
    """
    Seed process-global RNGs for one already resolved concrete device.

    Python, legacy NumPy, and Torch CPU state are always mutated. CUDA generators
    are seeded only for a concrete CUDA resolution, so CPU execution does not
    probe or initialize CUDA. Invalid device objects raise ``TypeError``.
    """
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Process seeding requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 -- exact process-global state is checkpointed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """
    Apply one exact deterministic policy to implemented Torch controls.

    The function mutates deterministic-algorithm and cuDNN process globals; when
    enabled it also sets the cuBLAS workspace environment contract. It performs
    no device selection and rejects non-boolean settings.
    """
    if not isinstance(enabled, bool):
        msg = f"run.deterministic must be boolean, got {enabled!r}."
        raise TypeError(msg)
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def configure_reproducibility(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, int]:
    """
    Apply reproducibility policy and seed the concrete process runtime.

    The resolved ``run`` section produces a stable labeled seed plan, then
    process-global deterministic controls and the ``process`` stream are applied.
    The full plan is returned for model, split, loader, worker, and tuner owners.
    """
    run = config.get("run")
    if not isinstance(run, Mapping):
        msg = "Resolved config must contain a run mapping."
        raise TypeError(msg)
    seed_plan = build_seed_plan(int(run["seed"]))
    configure_determinism(bool(run["deterministic"]))
    seed_process(seed_plan["process"], device=device)
    return seed_plan


def _validated_runtime_device(
    config: Mapping[str, Any],
    resolution: learning.device.DeviceResolution,
) -> torch.device:
    """
    Admit one service resolution against the requested config and AMP policy.

    Requested policy equality is checked without re-resolving availability.
    Mixed precision is then validated against the same concrete decision, which
    is returned as a ``torch.device`` for downstream factories.
    """
    run = config.get("run")
    if not isinstance(run, Mapping):
        msg = "Resolved config must contain a run mapping."
        raise TypeError(msg)
    requested = learning.device.validate_device_policy(run.get("device"), path="run.device")
    if resolution.requested_policy != requested:
        msg = f"Runtime device resolution does not match the requested config policy: {resolution.requested_policy!r} != {requested!r}."
        raise ValueError(msg)
    learning.device.validate_mixed_precision_device(
        config.get("training", {}).get("mixed_precision"),
        resolution,
    )
    return resolution.device


def runtime_session_updates(
    run_dir: Path,
    resolution: learning.device.DeviceResolution,
    *,
    started_at: datetime,
    session_id: str | None = None,
    tracking_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build summary updates that append one truthful runtime session.

    Prior sessions remain immutable. The optional nested tracking state records
    disabled or pending observer facts without importing the W&B SDK.

    Parameters
    ----------
    run_dir : pathlib.Path
        Existing run whose versioned summary supplies prior sessions.
    resolution : learning.device.DeviceResolution
        Truthful requested-versus-concrete runtime decision.
    started_at : datetime
        Session start instant persisted verbatim in ISO form.
    session_id : str | None, optional
        Caller identity, or a generated opaque ID when omitted.
    tracking_state : Mapping[str, Any] | None, optional
        Initial observer-only state copied into the new session.

    Returns
    -------
    dict[str, Any]
        Summary fields containing latest device facts and append-only sessions.

    """
    current = read_run_summary(run_dir)
    raw_sessions = current.get("runtime_sessions", [])
    if not isinstance(raw_sessions, list) or not all(isinstance(item, Mapping) for item in raw_sessions):
        msg = "Run summary runtime_sessions must be a list of mappings."
        raise RunLifecycleError(msg)
    resolved_session_id = session_id or uuid.uuid4().hex
    if not resolved_session_id:
        msg = "runtime session_id must be non-empty."
        raise ValueError(msg)
    device_metadata = resolution.as_dict()
    session: dict[str, Any] = {
        "session_id": resolved_session_id,
        "started_at": started_at.isoformat(),
        **device_metadata,
    }
    if tracking_state is not None:
        session["tracking"] = copy.deepcopy(dict(tracking_state))
    return {
        "runtime_device": device_metadata,
        "runtime_sessions": [*copy.deepcopy(raw_sessions), session],
    }


def _update_runtime_session_locked(
    run_dir: Path,
    session_id: str,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Update exactly one runtime session while the caller holds the writer lease.

    The session list is deep-copied, observer fields are merged, and the complete
    summary is atomically replaced. Missing/duplicate IDs or malformed tracking
    state raise ``RunLifecycleError`` before publication.
    """
    summary = read_run_summary(run_dir)
    raw_sessions = summary.get("runtime_sessions")
    if not isinstance(raw_sessions, list):
        msg = "Run summary runtime_sessions must be a list."
        raise RunLifecycleError(msg)
    sessions = copy.deepcopy(raw_sessions)
    matches = [index for index, session in enumerate(sessions) if isinstance(session, Mapping) and session.get("session_id") == session_id]
    if len(matches) != 1:
        msg = f"Expected one runtime session {session_id!r}, found {len(matches)}."
        raise RunLifecycleError(msg)
    index = matches[0]
    session = dict(cast("Mapping[str, Any]", sessions[index]))
    current_tracking = session.get("tracking", {})
    if not isinstance(current_tracking, Mapping):
        msg = f"Runtime session {session_id!r} tracking state must be a mapping."
        raise RunLifecycleError(msg)
    session["tracking"] = {
        **copy.deepcopy(dict(current_tracking)),
        **copy.deepcopy(dict(updates)),
    }
    sessions[index] = session
    summary["runtime_sessions"] = sessions
    summary["updated_at"] = _utc_now()
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        summary,
    )
    return summary


def update_runtime_session(
    run_dir: Path,
    session_id: str,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Merge tracking/runtime facts into one persisted session atomically.

    The exact ``session_id`` must already exist. The run writer lease prevents
    concurrent training, resume, artifact, or observer updates from losing
    summary state; immutable run identity and status are not changed here.
    """
    with run_writer_lease(run_dir):
        return _update_runtime_session_locked(run_dir, session_id, updates)


def initial_tracking_state(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build safe initial observer facts before any SDK import or initialization.

    The result records configured identity, mode, tags, and active/disabled
    status only. It performs no credential access, network operation, W&B import,
    or mutation of the resolved config.
    """
    settings = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    enabled = bool(settings["enabled"])
    raw_tags = settings.get("tags", [])
    tags = list(cast("list[str]", raw_tags)) if isinstance(raw_tags, list) else []
    return {
        "enabled": enabled,
        "requested_mode": settings.get("mode", "online"),
        "project": settings.get("project"),
        "entity": settings.get("entity"),
        "group": settings.get("group"),
        "tags": tags,
        "status": "active" if enabled else "disabled",
    }


def append_runtime_session(
    run_dir: Path,
    resolution: learning.device.DeviceResolution,
    *,
    started_at: datetime,
    session_id: str,
    tracking_state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Append one non-training runtime session to an existing run summary.

    Parameters identify the service-resolved device, start time, opaque session
    id, and initial tracking state. Publication is atomic under the run writer
    lease and preserves the authoritative local run status.
    """
    with run_writer_lease(run_dir):
        summary = read_run_summary(run_dir)
        updates = runtime_session_updates(
            run_dir,
            resolution,
            started_at=started_at,
            session_id=session_id,
            tracking_state=tracking_state,
        )
        payload = {**summary, **updates, "updated_at": _utc_now()}
        common.serialization.atomic_write_json(
            common.paths.resolve_run_summary_path(run_dir),
            payload,
        )
        return payload


def allocate_run_directory(run_dir: Path | str) -> Path:
    """
    Exclusively allocate one fresh run leaf.

    Existing leaves fail before any file inside them is read or written. Parent
    directories may be created as non-run containers.

    Parameters
    ----------
    run_dir : Path | str
        Exact fresh run leaf.

    Returns
    -------
    pathlib.Path
        Newly created absolute leaf.

    Raises
    ------
    FileExistsError
        If the leaf already exists; callers must use explicit resume instead.

    """
    path = Path(run_dir).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        msg = f"Fresh run directory already exists; use explicit --resume to open it: {path}"
        raise FileExistsError(msg) from error
    return path.resolve()


def read_run_summary(run_dir: Path | str) -> dict[str, Any]:
    """
    Load one current versioned run summary without mutation.

    Parameters
    ----------
    run_dir : Path | str
        Exact run leaf containing ``summary.json``.

    Returns
    -------
    dict[str, Any]
        Parsed schema-1 summary mapping.

    Raises
    ------
    FileNotFoundError
        If the summary is absent.
    RunLifecycleError
        If JSON or the schema version is invalid.

    """
    path = common.paths.resolve_run_summary_path(run_dir)
    if not path.is_file():
        msg = f"Run summary not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as error:
        msg = f"Run summary is invalid JSON: {path}: {error}"
        raise RunLifecycleError(msg) from error
    if not isinstance(payload, dict):
        msg = f"Run summary must contain a JSON object: {path}"
        raise RunLifecycleError(msg)
    if payload.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION:
        msg = f"Unsupported or missing run summary schema: {path}"
        raise RunLifecycleError(msg)
    return payload


def _transition_run_status_locked(
    run_dir: Path | str,
    status: str,
    *,
    updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Atomically transition summary.json through the explicit run state machine.

    Parameters
    ----------
    run_dir : Path | str
        Allocated run directory.
    status : str
        Target lifecycle status.
    updates : Mapping[str, Any] | None, optional
        Additional JSON-safe summary fields.

    Returns
    -------
    dict[str, Any]
        Newly published summary payload.

    """
    summary_path = common.paths.resolve_run_summary_path(run_dir)
    current: dict[str, Any] = {}
    current_status: str | None = None
    if summary_path.exists():
        current = read_run_summary(run_dir)
        raw_status = current.get("status")
        if not isinstance(raw_status, str):
            msg = f"Run summary has no valid status: {summary_path}"
            raise RunLifecycleError(msg)
        current_status = raw_status
    allowed = _TRANSITIONS.get(current_status, frozenset())
    if status not in allowed:
        msg = f"Invalid run status transition {current_status!r} -> {status!r} for {run_dir}."
        raise RunLifecycleError(msg)

    now = _utc_now()
    history = list(current.get("status_history", []))
    history.append({"status": status, "time": now})
    payload = {
        **current,
        **dict(updates or {}),
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "status": status,
        "status_history": history,
        "updated_at": now,
    }
    payload.setdefault("created_at", now)
    common.serialization.atomic_write_json(summary_path, payload)
    return payload


def transition_run_status(
    run_dir: Path | str,
    status: str,
    *,
    updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Publish one allowed summary-state transition under the writer lease.

    ``updates`` are merged with versioned status history and timestamps through
    atomic JSON replacement. Unknown or forbidden transitions raise
    ``RunLifecycleError`` without changing the summary.
    """
    with run_writer_lease(run_dir):
        return _transition_run_status_locked(run_dir, status, updates=updates)


def _load_mapping_artifact(path: Path, *, label: str) -> dict[str, Any]:
    """Load one local Torch mapping artifact."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        msg = f"Saved {label} must contain a mapping: {path}"
        raise TypeError(msg)
    return dict(payload)


def _validate_saved_data_contract(
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
) -> None:
    """
    Admit saved split and normalizer artifacts against the resolved task config.

    Task/digest identity, split ratios, stable split subseed, dataset membership,
    tensor layout, and input/output normalizer channel counts are validated on
    CPU before resume, inference, or artifact consumers may use the state.
    """
    task = config_loader.validate_resolved_task_contract(config)
    data_config = config.get("data")
    run_config = config.get("run")
    if not isinstance(data_config, Mapping) or not isinstance(run_config, Mapping):
        msg = "Completed run config must contain data and run mappings."
        raise RunLifecycleError(msg)
    if split_indices.get("task") != task.id or split_indices.get("task_contract_digest") != task.contract_digest:
        msg = "Saved split task identity does not match the resolved config task contract."
        raise RunLifecycleError(msg)
    datasets.base.validate_split_info(
        split_indices,
        expected_train_ratio=data_config.get("train_ratio"),
        expected_ood_fraction=data_config.get("ood_fraction"),
        expected_split_seed=derive_subseed(int(run_config["seed"]), "split"),
    )
    datasets.base.data_processor_from_state(normalizer_state, device="cpu")
    channel_axis = task.tensor_layout.index("channel")
    for prefix, expected_channels in (("in_normalizer", task.in_channels), ("out_normalizer", task.out_channels)):
        mean = normalizer_state[f"{prefix}.mean"]
        if not isinstance(mean, torch.Tensor) or mean.ndim != len(task.tensor_layout) or mean.shape[channel_axis] != expected_channels:
            actual_shape = tuple(mean.shape) if isinstance(mean, torch.Tensor) else type(mean).__name__
            msg = f"Saved {prefix} channel shape does not match task fields: {actual_shape}."
            raise RunLifecycleError(msg)


def _config_comparison_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return resume-fixed semantics after removing explicitly allowed fields."""
    view = copy.deepcopy(dict(config))
    view.pop("paths", None)
    run = view.get("run")
    if isinstance(run, dict):
        run.pop("device", None)
    training = view.get("training")
    if isinstance(training, dict):
        training.pop("epochs", None)
    return view


def _different_fields(left: Any, right: Any, *, prefix: str = "") -> list[str]:
    """Return dotted leaf paths whose values differ."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left).union(right), key=str):
            field = f"{prefix}.{key}" if prefix else str(key)
            differences.extend(_different_fields(left.get(key, _MISSING), right.get(key, _MISSING), prefix=field))
        return differences
    return [prefix or "<root>"] if left is _MISSING or right is _MISSING or left != right else []


def validate_resume_config(
    requested_config: Mapping[str, Any],
    saved_config: Mapping[str, Any],
) -> int:
    """
    Validate resume semantics and return the requested terminal epoch.

    ``training.epochs`` may remain equal or increase. Decreases and every
    task/data/model/loss/optimizer/scheduler change are rejected. ``run.device``
    and resolved paths are runtime metadata handled separately.
    """
    differences = _different_fields(
        _config_comparison_view(requested_config),
        _config_comparison_view(saved_config),
    )
    if differences:
        shown = ", ".join(differences[:_MAX_CONFIG_DIFFERENCES])
        suffix = " ..." if len(differences) > _MAX_CONFIG_DIFFERENCES else ""
        msg = f"Requested config is incompatible with the saved run. Differing field(s): {shown}{suffix}."
        raise ValueError(msg)
    requested_epochs = int(requested_config["training"]["epochs"])
    saved_epochs = int(saved_config["training"]["epochs"])
    if requested_epochs < saved_epochs:
        msg = f"Resume may only retain or increase training.epochs; requested {requested_epochs}, saved {saved_epochs}."
        raise ValueError(msg)
    return requested_epochs


def _validate_resume_output_root(
    run_dir: Path,
    saved_config: Mapping[str, Any],
    output_root: Path | str | None,
) -> None:
    """
    Reject a resume output override that resolves away from the saved run leaf.

    ``None`` preserves the existing location. An explicit root is combined with
    saved task/run identity and must resolve to the exact resumed directory.
    """
    if output_root is None:
        return
    task = str(saved_config["task"])
    run_name = str(saved_config["run"]["name"])
    expected = common.paths.resolve_run_output_dir(task, run_name, output_root=output_root).resolve()
    if expected != run_dir.resolve():
        msg = f"--output-root resolves to {expected}, but --resume identifies {run_dir}."
        raise ValueError(msg)


def _prepare_fresh_run_locked(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str | None = None,
    summary_extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Allocate a fresh leaf and publish its initial summary/config under a lease.

    This internal variant requires its caller to own the destination writer
    lease. If initialization fails after allocation, it best-effort publishes a
    terminal failure while preserving the original exception.
    """
    task = str(config["task"])
    run_name = str(config["run"]["name"])
    destination = (
        Path(run_dir)
        if run_dir is not None
        else common.paths.resolve_run_output_dir(
            task,
            run_name,
            output_root=Path(config["paths"]["output_root"]),
        )
    )
    allocated = allocate_run_directory(destination)
    initial = {
        "task": task,
        "run_name": run_name,
        "objective": dict(config["evaluation"]["objective"]),
        **dict(summary_extra or {}),
    }
    try:
        transition_run_status(allocated, "initializing", updates=initial)
        config_loader.save_yaml(dict(config), common.paths.resolve_run_config_path(allocated))
    except BaseException as error:
        with suppress(Exception):
            if common.paths.resolve_run_summary_path(allocated).is_file():
                transition_run_status(
                    allocated,
                    "failed",
                    updates={"error_type": type(error).__name__, "error": str(error)},
                )
        raise
    return allocated


def prepare_fresh_run(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str | None = None,
    summary_extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Allocate and initialize a fresh run while holding its writer lease.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved immutable experiment config.
    run_dir : Path | str | None, optional
        Exact fresh leaf override; otherwise resolve from task, run name, and
        output root.
    summary_extra : Mapping[str, Any] | None, optional
        Additional caller-owned lifecycle facts for the initial summary.

    Returns
    -------
    pathlib.Path
        Newly allocated run leaf containing an initializing summary and the
        authoritative immutable ``config.yaml``.

    Raises
    ------
    FileExistsError
        If the leaf already exists; only explicit resume may reopen it.
    RunLifecycleError
        If another writer owns the destination or initialization cannot publish
        an allowed summary transition.
    OSError
        If initial summary or config publication fails.

    Notes
    -----
    The destination is allocated before its initial files are written. A
    post-allocation failure is re-raised after best-effort publication of a
    terminal failed summary; the incomplete leaf is retained for diagnosis and
    never treated as loadable.

    """
    task = str(config["task"])
    run_name = str(config["run"]["name"])
    destination = (
        Path(run_dir)
        if run_dir is not None
        else common.paths.resolve_run_output_dir(
            task,
            run_name,
            output_root=Path(config["paths"]["output_root"]),
        )
    )
    with run_writer_lease(destination):
        return _prepare_fresh_run_locked(
            config,
            run_dir=destination,
            summary_extra=summary_extra,
        )


def _mark_failure(run_dir: Path, error: BaseException, *, interrupted: bool) -> None:
    """
    Best-effort publish failed or interrupted status without masking the cause.

    Transition and serialization errors are deliberately suppressed because
    this helper runs while propagating a primary lifecycle exception.
    """
    status = "interrupted" if interrupted else "failed"
    with suppress(Exception):
        transition_run_status(
            run_dir,
            status,
            updates={"error_type": type(error).__name__, "error": str(error)},
        )


def _validate_reused_data_state(
    *,
    data_processor: Any,
    restored_data_processor: Any,
    saved_split_indices: Mapping[str, Any] | None,
    rebuilt_split_indices: Mapping[str, Any],
) -> None:
    """
    Verify object-identical processor and tensor-identical split reuse on resume.

    Dataloader reconstruction must retain the admitted saved processor instance
    and exact train/eval/OOD membership; replacement or drift raises before model
    training or checkpoint restoration.
    """
    if data_processor is not restored_data_processor:
        msg = "Resume dataloader construction replaced the saved normalizer state."
        raise RuntimeError(msg)
    for key in ("train_indices", "eval_indices", "ood_indices"):
        saved = saved_split_indices[key] if saved_split_indices is not None else None
        rebuilt = rebuilt_split_indices.get(key)
        if not isinstance(saved, torch.Tensor) or not isinstance(rebuilt, torch.Tensor) or not torch.equal(saved, rebuilt):
            msg = f"Resume dataloader construction changed saved {key}."
            raise RuntimeError(msg)


def _validate_training_result_objective(
    result: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> None:
    """Require the training result to retain the resolved objective identity."""
    if result.get("objective") != objective:
        msg = "Training result objective does not match the resolved experiment objective."
        raise RunLifecycleError(msg)


def _execute_prepared_run_locked(
    config: dict[str, Any],
    *,
    run_dir: Path,
    persisted_config: Mapping[str, Any] | None = None,
    saved_split_indices: dict[str, Any] | None = None,
    restored_data_processor: Any | None = None,
    resume_from: Path | None = None,
    epoch_end_callback: Callable[[int, dict[str, float]], None] | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    device_resolution: learning.device.DeviceResolution,
) -> dict[str, Any]:
    """
    Build and execute one fresh or explicit-resume run in an allocated leaf.

    Authoritative data, split, normalizer, model, objective and device identity
    are admitted before optional W&B initialization. Local files and checkpoint
    state remain valid when an online observer degrades.

    Fresh execution atomically publishes fitted normalizer and split artifacts;
    resume requires object-identical processor reuse and unchanged membership.
    The helper owns status transitions, factories, checkpoint execution, terminal
    digests, observer finalization, and best-effort failure publication while its
    caller retains the exclusive writer lease.
    """
    start_time = datetime.now(UTC)
    runtime_session_id = uuid.uuid4().hex
    tracker: tracking.WandbSession | None = None
    tracking_status = "failed"
    tracking_result: Mapping[str, Any] | None = None
    tracking_error: BaseException | None = None
    tracking_initialization_attempted = False
    tracking_enabled = bool(config["tracking"]["wandb"]["enabled"])
    try:
        device = _validated_runtime_device(config, device_resolution)
        amp_enabled = bool(config["training"]["mixed_precision"])
        seed_plan = build_seed_plan(int(config["run"]["seed"]))
        transition_run_status(
            run_dir,
            "running",
            updates={
                "started_at": start_time.isoformat(),
                "target_epochs": int(config["training"]["epochs"]),
                "seed_plan": seed_plan,
                "deterministic": bool(config["run"]["deterministic"]),
                "amp_enabled": amp_enabled,
                **dict(summary_extra or {}),
                **runtime_session_updates(
                    run_dir,
                    device_resolution,
                    started_at=start_time,
                    session_id=runtime_session_id,
                    tracking_state=initial_tracking_state(config),
                ),
            },
        )
        configure_reproducibility(config, device=device)
        dataloaders = config_loader.create_dataloaders_from_config(
            config,
            split_indices=saved_split_indices,
            data_processor=restored_data_processor,
            seed_plan=seed_plan,
        )
        data_processor = dataloaders["data_processor"]
        split_indices = dataloaders["split_indices"]

        if resume_from is None:
            common.serialization.atomic_torch_save(
                data_processor.state_dict(),
                common.paths.resolve_normalizer_path(run_dir),
            )
            common.serialization.atomic_torch_save(
                split_indices,
                common.paths.resolve_split_indices_path(run_dir),
            )
        else:
            _validate_reused_data_state(
                data_processor=data_processor,
                restored_data_processor=restored_data_processor,
                saved_split_indices=saved_split_indices,
                rebuilt_split_indices=split_indices,
            )

        seed_process(seed_plan["model_init"], device=device)
        model = learning.models.factory.build_model(config, device=device)
        train_loss = learning.losses.factory.build_training_loss(config, device=device)
        set_normalizers = getattr(train_loss, "set_normalizers", None)
        if callable(set_normalizers):
            set_normalizers(
                in_normalizer=data_processor.in_normalizer,
                out_normalizer=data_processor.out_normalizer,
            )
        eval_metrics = learning.losses.factory.build_eval_metrics(config, device=device)
        optimizer = learning.training.optim.build_optimizer(model, config)
        scheduler = learning.training.optim.build_scheduler(optimizer, config)
        identity = learning.training.checkpoint.build_checkpoint_identity(
            config,
            split_indices,
            persisted_config=persisted_config,
        )

        def state_updater(updates: Mapping[str, Any]) -> None:
            """Persist W&B observer facts while the run writer lease is already held."""
            _update_runtime_session_locked(
                run_dir,
                runtime_session_id,
                updates,
            )

        monitor_membership = tracking.build_monitor_membership(config, split_indices)
        if monitor_membership is not None:
            state_updater({"monitor": monitor_membership})

        persisted_run_id: str | None = None
        previous_last_logged_epoch: int | None = None
        if tracking_enabled and resume_from is not None:
            persisted_run_id, previous_last_logged_epoch = tracking.persisted_wandb_identity(read_run_summary(run_dir))

        semantic_config: Mapping[str, Any] | None = None
        monitor_evaluator: tracking.MonitorEvaluator | None = None
        if tracking_enabled:
            semantic_config = tracking.build_semantic_config(
                config,
                split_indices=split_indices,
                normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
                checkpoint_identity=identity,
                model=model,
                device_metadata=device_resolution.as_dict(),
            )
            monitor_settings = config["tracking"]["wandb"]["monitor"]
            if bool(monitor_settings["enabled"]):
                max_monitor_cases = int(monitor_settings["max_cases"])

                def evaluate_monitor() -> Mapping[str, float]:
                    """Evaluate the fixed saved-ID prefix without changing training state."""
                    return learning.training.loop.evaluate_physics_monitor(
                        model,
                        dataloaders["eval"],
                        train_loss,
                        device,
                        data_processor,
                        max_cases=max_monitor_cases,
                    )

                monitor_evaluator = evaluate_monitor

        tracking_initialization_attempted = True
        tracker = tracking.initialize_wandb(
            config,
            run_dir=run_dir,
            semantic_config=semantic_config,
            resume=resume_from is not None,
            persisted_run_id=persisted_run_id,
            previous_last_logged_epoch=previous_last_logged_epoch,
            state_updater=state_updater,
            monitor_evaluator=monitor_evaluator,
        )

        result = learning.training.loop.train_loop(
            config=config,
            device=device,
            model=model,
            optimizer=optimizer,
            train_loader=dataloaders["train"],
            eval_loader=dataloaders["eval"],
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            data_processor=data_processor,
            scheduler=scheduler,
            save_dir=run_dir,
            use_amp=config["training"].get("mixed_precision", False),
            resume_from=resume_from,
            epoch_end_callback=tracking.combine_epoch_callbacks(
                epoch_end_callback,
                tracking.epoch_callback(tracker),
            ),
            checkpoint_identity=identity,
        )
        objective = config_loader.get_resolved_objective(config)
        _validate_training_result_objective(result, objective)
        end_time = datetime.now(UTC)
        completed_updates = {
            "task": config["task"],
            "run_name": config["run"]["name"],
            "model_kind": config["model"]["kind"],
            "model_parameter_counts": {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            },
            "objective": objective,
            "best_epoch": result["best_epoch"],
            "best_metric": result["best_metric"],
            "completed_epoch": result["completed_epoch"],
            "global_step": result["global_step"],
            "best_checkpoint": "best_checkpoint.pt",
            "last_checkpoint": "last_checkpoint.pt",
            "config_sha256": common.serialization.file_sha256(common.paths.resolve_run_config_path(run_dir)),
            "split_indices_sha256": common.serialization.file_sha256(common.paths.resolve_split_indices_path(run_dir)),
            "normalizer_sha256": common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
            "best_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_best_checkpoint_file(run_dir)),
            "last_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_last_checkpoint_file(run_dir)),
            "effective_config_digest": identity["effective_config_digest"],
            "elapsed_seconds": (end_time - start_time).total_seconds(),
            "ended_at": end_time.isoformat(),
            "error": None,
            "error_type": None,
            **dict(summary_extra or {}),
        }
        transition_run_status(run_dir, "completed", updates=completed_updates)
        tracking_status = "completed"
        tracking_result = result
    except KeyboardInterrupt as error:
        tracking_status = "interrupted"
        tracking_error = error
        if tracking_enabled and tracker is None and not tracking_initialization_attempted:
            _update_runtime_session_locked(
                run_dir,
                runtime_session_id,
                {
                    "status": "failed_before_start",
                    "failed_operation": "local_admission",
                    "error_class": type(error).__name__,
                    "error_message": "Local run was interrupted before tracking initialization.",
                },
            )
        _mark_failure(run_dir, error, interrupted=True)
        raise
    except BaseException as error:
        tracking_error = error
        if tracking_enabled and tracker is None and not tracking_initialization_attempted:
            _update_runtime_session_locked(
                run_dir,
                runtime_session_id,
                {
                    "status": "failed_before_start",
                    "failed_operation": "local_admission",
                    "error_class": type(error).__name__,
                    "error_message": "Local run admission failed before tracking initialization.",
                },
            )
        _mark_failure(run_dir, error, interrupted=False)
        raise
    finally:
        if tracker is not None:
            local_summary: Mapping[str, Any] | None = None
            with suppress(Exception):
                local_summary = read_run_summary(run_dir)
            try:
                tracker.finish(
                    status=tracking_status,
                    result=tracking_result,
                    local_summary=local_summary,
                    error=tracking_error,
                )
            except tracking.TrackingError:
                if tracking_error is None:
                    raise
    return result


def execute_prepared_run(
    config: dict[str, Any],
    *,
    run_dir: Path,
    persisted_config: Mapping[str, Any] | None = None,
    saved_split_indices: dict[str, Any] | None = None,
    restored_data_processor: Any | None = None,
    resume_from: Path | None = None,
    epoch_end_callback: Callable[[int, dict[str, float]], None] | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    device_resolution: learning.device.DeviceResolution,
) -> dict[str, Any]:
    """
    Execute a prepared fresh or resume run under its exclusive writer lease.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved runtime config. On resume this may extend duration and
        change runtime-only device policy while preserving saved science.
    run_dir : pathlib.Path
        Already initialized run leaf whose lifecycle this call exclusively owns.
    persisted_config : Mapping[str, Any] | None, optional
        Immutable saved config used for checkpoint identity; defaults to
        ``config`` for fresh execution.
    saved_split_indices : dict[str, Any] | None, optional
        Admitted saved membership reused during explicit resume.
    restored_data_processor : Any | None, optional
        Processor reconstructed from the saved normalizer for explicit resume.
    resume_from : pathlib.Path | None, optional
        Validated ``last_checkpoint.pt`` continuation source. ``None`` starts a
        fresh training state.
    epoch_end_callback : Callable[[int, dict[str, float]], None] | None, optional
        Local authoritative callback invoked at completed evaluation epochs
        before the optional tracking observer.
    summary_extra : Mapping[str, Any] | None, optional
        Caller-owned facts merged into lifecycle publications.
    device_resolution : learning.device.DeviceResolution
        Concrete service-resolved device and serializable runtime metadata.

    Returns
    -------
    dict[str, Any]
        Completed training result with objective, history, and checkpoint facts.

    Raises
    ------
    RunLifecycleError
        If the writer lease, saved data, checkpoint identity, objective result,
        or lifecycle transition violates the admitted run contract.
    BaseException
        Runtime construction, training, interruption, and required tracking
        failures are re-raised after best-effort failed/interrupted publication.

    Notes
    -----
    Fresh execution atomically publishes split and normalizer state before model
    construction. Resume reuses those immutable artifacts and restores only the
    last checkpoint; the best checkpoint remains the selected inference source.
    Local summary/checkpoint publication is authoritative. Online W&B transport
    may degrade without changing it, while requested offline record failures
    follow the tracking contract and propagate.

    This is a lower-level service for callers that already prepared lifecycle
    state. Normal CLI/notebook launches should prefer :func:`run_experiment`.

    """
    with run_writer_lease(run_dir):
        return _execute_prepared_run_locked(
            config,
            run_dir=run_dir,
            persisted_config=persisted_config,
            saved_split_indices=saved_split_indices,
            restored_data_processor=restored_data_processor,
            resume_from=resume_from,
            epoch_end_callback=epoch_end_callback,
            summary_extra=summary_extra,
            device_resolution=device_resolution,
        )


def validate_completed_run(run_dir: Path | str) -> dict[str, Any]:
    """
    Validate and return the completed/loadable run contract.

    Both best and last checkpoints are schema- and identity-validated. This is
    the common gate used by inference and normal artifact generation.

    Parameters
    ----------
    run_dir : Path | str
        Candidate completed run leaf.

    Returns
    -------
    dict[str, Any]
        Validated summary, config, split/normalizer state, checkpoint identity,
        and distinct best/last checkpoint payloads.

    Raises
    ------
    FileNotFoundError
        If a required current run artifact is absent.
    RunLifecycleError
        If status, schema, digests, identity, progress, or checkpoint roles disagree.

    Notes
    -----
    Validation is read-only and loads checkpoints on CPU. It does not acquire a
    writer lease, initialize a runtime device, or repair incompatible artifacts.

    """
    path = Path(run_dir)
    missing = common.paths.missing_current_run_files(path)
    if missing:
        names = ", ".join(item.name for item in missing)
        msg = f"Run is incomplete and not loadable: {path}. Missing: {names}."
        raise RunLifecycleError(msg)
    summary = read_run_summary(path)
    if summary.get("status") != "completed":
        msg = f"Run status must be 'completed' for inference/artifacts, got {summary.get('status')!r}: {path}"
        raise RunLifecycleError(msg)
    config_path = common.paths.resolve_run_config_path(path)
    split_path = common.paths.resolve_split_indices_path(path)
    normalizer_path = common.paths.resolve_normalizer_path(path)
    config = config_loader.load_yaml(config_path)
    split_indices = _load_mapping_artifact(split_path, label="split indices")
    normalizer_state = _load_mapping_artifact(normalizer_path, label="normalizer")
    _validate_saved_data_contract(config, split_indices, normalizer_state)
    identity = learning.training.checkpoint.build_checkpoint_identity(
        config,
        split_indices,
        persisted_config=config,
    )
    if summary.get("effective_config_digest") != identity["effective_config_digest"]:
        msg = "Completed run summary/config digest mismatch."
        raise RunLifecycleError(msg)
    expected_file_digests = {
        "config_sha256": common.serialization.file_sha256(config_path),
        "split_indices_sha256": common.serialization.file_sha256(split_path),
        "normalizer_sha256": common.serialization.file_sha256(normalizer_path),
    }
    for label, expected_digest in expected_file_digests.items():
        if summary.get(label) != expected_digest:
            msg = f"Completed run {label} mismatch."
            raise RunLifecycleError(msg)
    amp_enabled = summary.get("amp_enabled")
    if not isinstance(amp_enabled, bool):
        msg = "Completed run summary must record amp_enabled."
        raise RunLifecycleError(msg)
    scheduler_expected = config.get("scheduler") is not None
    best = learning.training.checkpoint.load_checkpoint(
        common.paths.resolve_best_checkpoint_file(path),
        expected_identity=identity,
        expected_role="best",
        scheduler_expected=scheduler_expected,
        amp_expected=amp_enabled,
        require_best=True,
    )
    last = learning.training.checkpoint.load_checkpoint(
        common.paths.resolve_last_checkpoint_file(path),
        expected_identity=identity,
        expected_role="last",
        scheduler_expected=scheduler_expected,
        amp_expected=amp_enabled,
        require_best=True,
    )
    if summary.get("best_checkpoint_sha256") != common.serialization.file_sha256(common.paths.resolve_best_checkpoint_file(path)):
        msg = "Completed run best checkpoint digest mismatch."
        raise RunLifecycleError(msg)
    if summary.get("last_checkpoint_sha256") != common.serialization.file_sha256(common.paths.resolve_last_checkpoint_file(path)):
        msg = "Completed run last checkpoint digest mismatch."
        raise RunLifecycleError(msg)
    if summary.get("best_metric") != best["best_metric"] or summary.get("best_epoch") != best["best_epoch"]:
        msg = "Completed run summary disagrees with best checkpoint objective state."
        raise RunLifecycleError(msg)
    if last["best_metric"] != best["best_metric"] or last["best_epoch"] != best["best_epoch"]:
        msg = "Completed run best and last checkpoints disagree."
        raise RunLifecycleError(msg)
    expected_summary_values = {
        "task": config.get("task"),
        "run_name": config.get("run", {}).get("name") if isinstance(config.get("run"), Mapping) else None,
        "objective": config.get("evaluation", {}).get("objective") if isinstance(config.get("evaluation"), Mapping) else None,
        "best_checkpoint": common.paths.RUN_BEST_CHECKPOINT_FILENAME,
        "last_checkpoint": common.paths.RUN_LAST_CHECKPOINT_FILENAME,
        "completed_epoch": last["completed_epoch"],
        "global_step": last["global_step"],
    }
    for label, expected_value in expected_summary_values.items():
        if summary.get(label) != expected_value:
            msg = f"Completed run summary {label!r} mismatch."
            raise RunLifecycleError(msg)
    return {
        "run_dir": path,
        "summary": summary,
        "config": config,
        "split_indices": split_indices,
        "normalizer_state": normalizer_state,
        "checkpoint_identity": identity,
        "best_checkpoint": best,
        "last_checkpoint": last,
    }


def run_experiment(
    config_path: Path | str,
    *,
    resume: Path | str | None = None,
    device: str | None = None,
    output_root: Path | str | None = None,
) -> dict[str, Any]:
    """
    Resolve and execute a fresh or explicit-resume experiment.

    Config, requested device, concrete device, and mixed-precision compatibility
    are validated before fresh output allocation. Resume acquires the run writer
    lease before reading mutable lifecycle artifacts and restores only the
    identity-validated last checkpoint.

    Parameters
    ----------
    config_path : pathlib.Path | str
        Semantic experiment YAML resolved under the strict current schema.
    resume : pathlib.Path | str | None, optional
        Existing run directory explicitly continued from ``last_checkpoint.pt``.
        ``None`` requires exclusive allocation of a new run leaf.
    device : {"auto", "cuda", "cpu"} | None, optional
        Runtime policy override. Strict ``cuda`` never falls back; ``None`` keeps
        the YAML policy. The concrete resolution is recorded per runtime session.
    output_root : pathlib.Path | str | None, optional
        Fresh-run destination override. On resume it must resolve to the exact
        saved task/run leaf; dataset roots remain unchanged.

    Returns
    -------
    dict[str, Any]
        ``run_dir`` and the completed training-loop ``result`` containing the
        resolved objective, selected best state, completed epoch, and history.

    Raises
    ------
    FileNotFoundError
        If the YAML, explicit resume leaf, or a required resume artifact is absent.
    FileExistsError
        If a fresh destination already exists; reopening requires ``resume``.
    config_loader.ConfigError
        If YAML or an override violates the semantic experiment schema.
    learning.device.DeviceResolutionError
        If runtime policy is invalid, strict CUDA is unusable, or mixed precision
        is incompatible with the resolved device.
    RunLifecycleError
        If a writer lease is active or saved status, data, checkpoint identity,
        best/last roles, or lifecycle history is not resumable.
    ValueError
        If requested science differs from the saved config, terminal duration
        does not extend progress, or an output-root override identifies another leaf.

    Notes
    -----
    Fresh execution allocates first, publishes immutable config/split/normalizer
    inputs, then trains under one writer lease. Resume preserves those inputs,
    permits only continuation-safe runtime/duration changes, and validates best
    versus last checkpoint roles before mutation. ``best_checkpoint.pt`` remains
    the inference/artifact source; ``last_checkpoint.pt`` is the sole continuation
    source.

    Training or admission failures publish failed/interrupted state best-effort
    before the original exception propagates. Local lifecycle files are
    authoritative: optional online W&B failures may degrade observation, whereas
    requested offline-record failures follow the tracking contract and can fail
    the operation after local evidence has been published.

    """
    raw_requested = config_loader.load_yaml(config_path)
    if device is not None:
        raw_run = raw_requested.get("run")
        if raw_run is None:
            raw_run = {}
            raw_requested["run"] = raw_run
        if not isinstance(raw_run, dict):
            msg = "run must be a mapping before applying --device."
            raise config_loader.ConfigError(msg)
        raw_run["device"] = device
    requested = config_loader.resolve_config(raw_requested)
    device_resolution = learning.device.resolve_device(
        requested["run"]["device"],
        path="run.device",
    )
    learning.device.validate_mixed_precision_device(
        requested["training"]["mixed_precision"],
        device_resolution,
    )
    if resume is None:
        config = requested
        if output_root is not None:
            config["paths"]["output_root"] = str(Path(output_root).expanduser())
        destination = common.paths.resolve_run_output_dir(
            str(config["task"]),
            str(config["run"]["name"]),
            output_root=Path(config["paths"]["output_root"]),
        )
        with run_writer_lease(destination):
            run_dir = _prepare_fresh_run_locked(config, run_dir=destination)
            result = _execute_prepared_run_locked(
                config,
                run_dir=run_dir,
                persisted_config=config,
                device_resolution=device_resolution,
            )
        return {"run_dir": run_dir, "result": result}

    run_dir = Path(resume).expanduser().resolve()
    if not run_dir.is_dir():
        msg = f"Resume run directory not found: {run_dir}"
        raise FileNotFoundError(msg)
    with run_writer_lease(run_dir):
        missing = common.paths.missing_resume_run_files(run_dir)
        if missing:
            names = ", ".join(path.name for path in missing)
            msg = f"Resume run is incomplete: {run_dir}. Missing: {names}."
            raise RunLifecycleError(msg)
        summary = read_run_summary(run_dir)
        if summary.get("status") not in {"running", "interrupted", "completed"}:
            msg = f"Run status {summary.get('status')!r} is not resumable: {run_dir}"
            raise RunLifecycleError(msg)
        saved_config = config_loader.load_yaml(common.paths.resolve_run_config_path(run_dir))
        target_epochs = validate_resume_config(requested, saved_config)
        _validate_resume_output_root(run_dir, saved_config, output_root)
        runtime_config = copy.deepcopy(saved_config)
        runtime_config["training"]["epochs"] = target_epochs
        runtime_config["run"]["device"] = requested["run"]["device"]

        split_indices = _load_mapping_artifact(common.paths.resolve_split_indices_path(run_dir), label="split indices")
        normalizer_state = _load_mapping_artifact(common.paths.resolve_normalizer_path(run_dir), label="normalizer")
        _validate_saved_data_contract(saved_config, split_indices, normalizer_state)
        data_processor = datasets.base.data_processor_from_state(normalizer_state, device="cpu")
        identity = learning.training.checkpoint.build_checkpoint_identity(
            runtime_config,
            split_indices,
            persisted_config=saved_config,
        )
        amp_enabled = bool(runtime_config["training"]["mixed_precision"])
        last = learning.training.checkpoint.load_checkpoint(
            common.paths.resolve_last_checkpoint_file(run_dir),
            expected_identity=identity,
            expected_role="last",
            scheduler_expected=runtime_config.get("scheduler") is not None,
            amp_expected=amp_enabled,
            require_best=False,
        )
        best_path = common.paths.resolve_best_checkpoint_file(run_dir)
        if last["best_metric"] is None:
            if best_path.exists():
                msg = "Resume run contains a best checkpoint that is inconsistent with last_checkpoint.pt."
                raise RunLifecycleError(msg)
        else:
            best = learning.training.checkpoint.load_checkpoint(
                best_path,
                expected_identity=identity,
                expected_role="best",
                scheduler_expected=runtime_config.get("scheduler") is not None,
                amp_expected=amp_enabled,
                require_best=True,
            )
            if best["best_metric"] != last["best_metric"] or best["best_epoch"] != last["best_epoch"]:
                msg = "Resume run best and last checkpoints disagree about selected objective state."
                raise RunLifecycleError(msg)
        if summary.get("status") == "completed" and target_epochs <= int(last["completed_epoch"]):
            msg = "A completed run may be resumed only with a deliberate increase beyond its completed epoch."
            raise ValueError(msg)
        if target_epochs <= int(last["completed_epoch"]):
            msg = f"training.epochs={target_epochs} does not extend beyond completed epoch {last['completed_epoch']}."
            raise ValueError(msg)

        result = _execute_prepared_run_locked(
            runtime_config,
            run_dir=run_dir,
            persisted_config=saved_config,
            saved_split_indices=split_indices,
            restored_data_processor=data_processor,
            resume_from=common.paths.resolve_last_checkpoint_file(run_dir),
            device_resolution=device_resolution,
        )
        return {"run_dir": run_dir, "result": result}
