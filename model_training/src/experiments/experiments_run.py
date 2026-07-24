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
===============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import common, datasets, learning

from . import experiments_tracking as tracking
from .config import experiments_config_loader as config_loader

RUN_SUMMARY_SCHEMA_VERSION = 1
RUN_STATUSES = frozenset({"initializing", "running", "completed", "failed", "interrupted", "pruned", "oom_pruned"})
_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"initializing"}),
    "initializing": frozenset({"running", "failed", "interrupted"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "pruned", "oom_pruned"}),
    "interrupted": frozenset({"running", "failed"}),
    "completed": frozenset({"running"}),
    "failed": frozenset(),
    "pruned": frozenset(),
    "oom_pruned": frozenset(),
}
_SEED_LABELS = ("process", "model_init", "split", "loader", "worker", "tuner")
_MISSING = object()
_MAX_CONFIG_DIFFERENCES = 12
_RUN_WRITER_LOCKS_DIRNAME = ".run-writer-locks"


class RunLifecycleError(RuntimeError):
    """Raised when a saved run violates the current lifecycle contract."""


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
    """Return all stable labeled run sub-seeds."""
    plan = {label: derive_subseed(seed, label) for label in _SEED_LABELS}
    if len(set(plan.values())) != len(plan):
        msg = "Stable labeled seed derivation produced a collision."
        raise RuntimeError(msg)
    return plan


def seed_process(seed: int) -> None:
    """Seed Python, process NumPy, Torch CPU, and every available CUDA device."""
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 -- exact process-global state is checkpointed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """Apply the resolved deterministic setting to implemented Torch controls."""
    if not isinstance(enabled, bool):
        msg = f"run.deterministic must be boolean, got {enabled!r}."
        raise TypeError(msg)
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def configure_reproducibility(config: Mapping[str, Any]) -> dict[str, int]:
    """Configure deterministic behavior and seed the initial process stream."""
    run = config.get("run")
    if not isinstance(run, Mapping):
        msg = "Resolved config must contain a run mapping."
        raise TypeError(msg)
    seed_plan = build_seed_plan(int(run["seed"]))
    configure_determinism(bool(run["deterministic"]))
    seed_process(seed_plan["process"])
    return seed_plan


def allocate_run_directory(run_dir: Path | str) -> Path:
    """
    Exclusively allocate one fresh run leaf.

    Existing leaves fail before any file inside them is read or written. Parent
    directories may be created as non-run containers.
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
    """Load one current summary JSON object."""
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
    """Atomically transition summary state while holding the run writer lease."""
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
    """Validate saved split membership and normalizer channels against config."""
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
    """Reject an explicit output override that points away from the saved run."""
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
    """Exclusively allocate a fresh run and atomically publish initial metadata."""
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
    """Allocate and initialize a fresh run while holding its writer lease."""
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
    """Best-effort atomic failed/interrupted status publication."""
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
    """Validate immutable normalizer and split reuse before resumed execution."""
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
) -> dict[str, Any]:
    """
    Build and execute one fresh or explicit-resume run in an allocated leaf.

    Fresh immutable artifacts are atomically published once. Resume validates
    and reuses config, split, and normalizer state without replacing them.
    """
    start_time = datetime.now(UTC)
    tracker: tracking.WandbSession | None = None
    tracking_status = "failed"
    tracking_result: Mapping[str, Any] | None = None
    tracking_error: str | None = None
    try:
        tracker = tracking.initialize_wandb(config, run_dir=run_dir)
        seed_plan = configure_reproducibility(config)
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

        seed_process(seed_plan["model_init"])
        model = learning.models.factory.build_model(config)
        train_loss = learning.losses.factory.build_training_loss(config)
        set_normalizers = getattr(train_loss, "set_normalizers", None)
        if callable(set_normalizers):
            set_normalizers(
                in_normalizer=data_processor.in_normalizer,
                out_normalizer=data_processor.out_normalizer,
            )
        eval_metrics = learning.losses.factory.build_eval_metrics(config)
        optimizer = learning.training.optim.build_optimizer(model, config)
        scheduler = learning.training.optim.build_scheduler(optimizer, config)
        identity = learning.training.checkpoint.build_checkpoint_identity(
            config,
            split_indices,
            persisted_config=persisted_config,
        )

        amp_enabled = bool(config["training"].get("mixed_precision", False) and torch.device(config["run"]["device"]).type == "cuda")
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
            },
        )
        result = learning.training.loop.train_loop(
            config=config,
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
                tracking.epoch_callback(tracker, optimizer),
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
        tracking_error = str(error)
        _mark_failure(run_dir, error, interrupted=True)
        raise
    except BaseException as error:
        tracking_error = str(error)
        _mark_failure(run_dir, error, interrupted=False)
        raise
    finally:
        if tracker is not None:
            with suppress(Exception):
                tracker.finish(
                    status=tracking_status,
                    result=tracking_result,
                    error=tracking_error,
                )
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
) -> dict[str, Any]:
    """Execute a prepared run under its exclusive writer lease."""
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
        )


def validate_completed_run(run_dir: Path | str) -> dict[str, Any]:
    """
    Validate and return the completed/loadable run contract.

    Both best and last checkpoints are schema- and identity-validated. This is
    the common gate used by inference and normal artifact generation.
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

    Parameters
    ----------
    config_path : Path | str
        Semantic experiment YAML.
    resume : Path | str | None, optional
        Existing run directory explicitly resumed from last checkpoint.
    device : str | None, optional
        Runtime execution-device override.
    output_root : Path | str | None, optional
        Output-only root override; dataset roots remain unchanged.

    Returns
    -------
    dict[str, Any]
        Canonical run directory and completed training-loop result.

    """
    requested = config_loader.load_and_resolve_config(config_path)
    if resume is None:
        config = requested
        if device is not None:
            config["run"]["device"] = device
        if output_root is not None:
            config["paths"]["output_root"] = str(Path(output_root).expanduser())
        destination = common.paths.resolve_run_output_dir(
            str(config["task"]),
            str(config["run"]["name"]),
            output_root=Path(config["paths"]["output_root"]),
        )
        with run_writer_lease(destination):
            run_dir = _prepare_fresh_run_locked(config, run_dir=destination)
            result = _execute_prepared_run_locked(config, run_dir=run_dir, persisted_config=config)
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
        if device is not None:
            runtime_config["run"]["device"] = device

        split_indices = _load_mapping_artifact(common.paths.resolve_split_indices_path(run_dir), label="split indices")
        normalizer_state = _load_mapping_artifact(common.paths.resolve_normalizer_path(run_dir), label="normalizer")
        _validate_saved_data_contract(saved_config, split_indices, normalizer_state)
        data_processor = datasets.base.data_processor_from_state(normalizer_state, device="cpu")
        identity = learning.training.checkpoint.build_checkpoint_identity(
            runtime_config,
            split_indices,
            persisted_config=saved_config,
        )
        amp_enabled = bool(runtime_config["training"].get("mixed_precision", False) and torch.device(runtime_config["run"]["device"]).type == "cuda")
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
        )
        return {"run_dir": run_dir, "result": result}
