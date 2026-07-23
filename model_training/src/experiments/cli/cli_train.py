"""
===============================================================================
cli_train.py
===============================================================================
Train a neural operator model from an experiment YAML.

Responsibilities:
  - Parse experiment config paths and runtime overrides
  - Resolve configs and create run output directories
  - Build data, model, loss, optimizer and scheduler components
  - Execute the custom training loop and write run summaries

Design principles:
  - CLI code stays as thin orchestration
  - Config resolution supplies defaults and run naming
  - Training state is saved in the run directory
  - Resume reuses one complete saved-run contract in place

Boundaries:
  - Model and loss construction belong to learning factories
  - Training execution belongs to learning.training.loop
  - Optuna orchestration belongs to cli_optuna and experiments.tuning
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from src import common, datasets, experiments, learning

_RESUME_SPLIT_KEYS = frozenset({"train_indices", "eval_indices", "ood_indices", "metadata"})
_RESUME_CHECKPOINT_STATE_KEYS = frozenset({"model_state_dict", "optimizer_state_dict"})
_RESUME_CHECKPOINT_PROGRESS_KEYS = frozenset({"epoch", "best_metric", "best_epoch"})
_RESUME_CHECKPOINT_KEYS = _RESUME_CHECKPOINT_STATE_KEYS | _RESUME_CHECKPOINT_PROGRESS_KEYS
_MAX_CONFIG_DIFFERENCES = 12
_MISSING = object()


def _resolve_resume_dir(path: str) -> Path:
    """Resolve an existing resume directory without creating it."""
    resume_dir = Path(path).expanduser()
    if not resume_dir.exists():
        msg = f"Resume run directory not found: {resume_dir}"
        raise FileNotFoundError(msg)
    if not resume_dir.is_dir():
        msg = f"Resume path is not a directory: {resume_dir}"
        raise NotADirectoryError(msg)
    return resume_dir.resolve()


def _require_resume_artifacts(run_dir: Path) -> None:
    """Require the four load-bearing files in the current run contract."""
    missing = common.paths.missing_current_run_files(run_dir)
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        msg = (
            f"Resume run is incomplete: {run_dir}. Missing required artifact(s): {missing_names}. "
            "Required files are config.yaml, normalizer.pt, split_indices.pt, and best_checkpoint.pt. "
            "summary.json is optional completion metadata."
        )
        raise FileNotFoundError(msg)


def _load_mapping_artifact(path: Path, *, label: str) -> dict[str, Any]:
    """Load a saved torch artifact and require a mapping payload."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        msg = f"Saved {label} must contain a mapping: {path}"
        raise TypeError(msg)
    return dict(payload)


def _load_saved_config(path: Path) -> dict[str, Any]:
    """Load the effective config stored in a run directory."""
    config = experiments.config.loader.load_yaml(path)
    if not isinstance(config, Mapping):
        msg = f"Saved run config must contain a mapping: {path}"
        raise TypeError(msg)
    return dict(config)


def _require_mapping_keys(payload: Mapping[str, Any], required: frozenset[str], *, label: str, path: Path) -> None:
    """Fail with artifact context when required mapping keys are absent."""
    missing = sorted(required.difference(payload))
    if missing:
        msg = f"Saved {label} is missing required keys {missing}: {path}"
        raise ValueError(msg)


def _run_identity(config: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    """Extract and validate the task/run-name identity from a config."""
    task = config.get("task")
    run = config.get("run")
    run_name = run.get("name") if isinstance(run, Mapping) else None
    if not isinstance(task, str) or not task:
        msg = f"{label} must contain a non-empty string task."
        raise ValueError(msg)
    if not isinstance(run_name, str) or not run_name:
        msg = f"{label} must contain a non-empty string run.name."
        raise ValueError(msg)
    return task, run_name


def _config_comparison_view(config: Mapping[str, Any], *, ignore_device: bool) -> dict[str, Any]:
    """Return semantic config fields used for strict resume compatibility."""
    view = dict(config)
    # Resolved roots are environment metadata. Resume always reuses the roots
    # stored in config.yaml and validates any explicit output override below.
    view.pop("paths", None)

    run = view.get("run")
    if isinstance(run, Mapping):
        run_view = dict(run)
        if ignore_device:
            run_view.pop("device", None)
        view["run"] = run_view
    return view


def _different_config_fields(left: Any, right: Any, *, prefix: str = "") -> list[str]:
    """Return dotted fields whose values differ between two config trees."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        keys = sorted(set(left).union(right), key=str)
        for key in keys:
            field = f"{prefix}.{key}" if prefix else str(key)
            differences.extend(
                _different_config_fields(
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                    prefix=field,
                )
            )
        return differences
    if left is _MISSING or right is _MISSING or left != right:
        return [prefix or "<root>"]
    return []


def _validate_resume_config(
    requested_config: Mapping[str, Any],
    saved_config: Mapping[str, Any],
    *,
    allow_device_override: bool,
) -> None:
    """Reject a positional config that conflicts with the saved run config."""
    requested_identity = _run_identity(requested_config, label="Requested config")
    saved_identity = _run_identity(saved_config, label="Saved config.yaml")
    if requested_identity != saved_identity:
        msg = (
            "Requested config identifies run "
            f"{requested_identity!r}, but --resume identifies saved run {saved_identity!r}."
        )
        raise ValueError(msg)

    requested_view = _config_comparison_view(requested_config, ignore_device=allow_device_override)
    saved_view = _config_comparison_view(saved_config, ignore_device=allow_device_override)
    differences = _different_config_fields(requested_view, saved_view)
    if differences:
        displayed = ", ".join(differences[:_MAX_CONFIG_DIFFERENCES])
        suffix = " ..." if len(differences) > _MAX_CONFIG_DIFFERENCES else ""
        msg = (
            "Requested config is incompatible with the saved resume config. "
            f"Differing field(s): {displayed}{suffix}. Resume reuses config.yaml; "
            "only an explicit --device runtime override is allowed."
        )
        raise ValueError(msg)


def _validate_resume_output_root(run_dir: Path, saved_config: Mapping[str, Any], output_root: str | None) -> None:
    """Reject an output-root override that points away from the resume run."""
    if output_root is None:
        return
    task, run_name = _run_identity(saved_config, label="Saved config.yaml")
    expected_run_dir = (Path(output_root).expanduser() / task / "runs" / run_name).resolve()
    if expected_run_dir != run_dir:
        msg = (
            f"--output-root resolves run output to {expected_run_dir}, but --resume identifies {run_dir}. "
            "A resumed run must continue writing to its supplied run directory."
        )
        raise ValueError(msg)


def _validate_reused_split_membership(
    saved_split_indices: Mapping[str, Any],
    rebuilt_split_indices: Mapping[str, Any],
) -> None:
    """Prove dataloader construction retained every saved split index."""
    for key in ("train_indices", "eval_indices", "ood_indices"):
        saved = saved_split_indices.get(key)
        rebuilt = rebuilt_split_indices.get(key)
        if not isinstance(saved, torch.Tensor) or not isinstance(rebuilt, torch.Tensor):
            msg = f"Saved and rebuilt {key!r} must both be tensors."
            raise TypeError(msg)
        if not torch.equal(saved.cpu(), rebuilt.cpu()):
            msg = f"Dataloader construction changed saved split membership for {key!r}."
            raise RuntimeError(msg)


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    path: Path,
) -> None:
    """Require state and progress needed to resume from the saved best checkpoint."""
    required_keys = set(_RESUME_CHECKPOINT_KEYS)
    if config.get("scheduler") is not None:
        required_keys.add("scheduler_state_dict")
    _require_mapping_keys(checkpoint, frozenset(required_keys), label="checkpoint", path=path)

    state_keys = set(_RESUME_CHECKPOINT_STATE_KEYS)
    if "scheduler_state_dict" in required_keys:
        state_keys.add("scheduler_state_dict")
    for state_key in state_keys:
        if not isinstance(checkpoint[state_key], Mapping):
            msg = f"Saved checkpoint {state_key!r} must contain a mapping: {path}"
            raise TypeError(msg)

    for progress_key in ("epoch", "best_epoch"):
        progress_value = checkpoint[progress_key]
        if isinstance(progress_value, bool) or not isinstance(progress_value, int):
            msg = f"Saved checkpoint {progress_key!r} must be an integer: {path}"
            raise TypeError(msg)
        if progress_value <= 0:
            msg = f"Saved checkpoint {progress_key!r} must be positive: {path}"
            raise ValueError(msg)
    if checkpoint["best_epoch"] > checkpoint["epoch"]:
        msg = f"Saved checkpoint best_epoch cannot exceed epoch: {path}"
        raise ValueError(msg)

    best_metric = checkpoint["best_metric"]
    if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)):
        msg = f"Saved checkpoint 'best_metric' must be numeric: {path}"
        raise TypeError(msg)


def main() -> int:
    """
    Run the training entry point.

    Returns
    -------
    int
        Exit code (0 on success)

    """
    parser = argparse.ArgumentParser(description="Train a neural operator model from config")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume in place from a run directory using its complete saved-run contract",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override training output root directory",
    )

    args = parser.parse_args()

    requested_config = experiments.config.loader.load_and_resolve_config(args.config_path)
    checkpoint_path: Path | None = None
    saved_split_indices: dict[str, Any] | None = None
    restored_data_processor: Any | None = None

    if args.resume is not None:
        run_dir = _resolve_resume_dir(args.resume)
        _require_resume_artifacts(run_dir)

        saved_config_path = common.paths.resolve_run_config_path(run_dir)
        normalizer_path = common.paths.resolve_normalizer_path(run_dir)
        split_indices_path = common.paths.resolve_split_indices_path(run_dir)
        checkpoint_path = common.paths.resolve_best_checkpoint_file(run_dir)

        config = _load_saved_config(saved_config_path)
        _validate_resume_config(
            requested_config,
            config,
            allow_device_override=args.device is not None,
        )
        _validate_resume_output_root(run_dir, config, args.output_root)

        saved_split_indices = _load_mapping_artifact(split_indices_path, label="split indices")
        _require_mapping_keys(
            saved_split_indices,
            _RESUME_SPLIT_KEYS,
            label="split indices",
            path=split_indices_path,
        )

        normalizer_state = _load_mapping_artifact(normalizer_path, label="normalizer state")
        restored_data_processor = datasets.base.data_processor_from_state(normalizer_state, device="cpu")

        checkpoint = _load_mapping_artifact(checkpoint_path, label="checkpoint")
        _validate_resume_checkpoint(checkpoint, config=config, path=checkpoint_path)
        del checkpoint

        if args.device is not None:
            config["run"]["device"] = args.device

        print(f"Run directory: {run_dir}")
        print(f"Saved config reused: {saved_config_path}")
        print(f"Saved normalizer restored: {normalizer_path}")
        print(f"Saved split membership reused: {split_indices_path}")
        print(f"Resuming from: {checkpoint_path}")
    else:
        config = requested_config
        if args.device is not None:
            config["run"]["device"] = args.device
        if args.output_root is not None:
            config["paths"]["train_root"] = args.output_root

        train_root = Path(config["paths"]["train_root"])
        task, run_name = _run_identity(config, label="Resolved config")
        run_dir = train_root / task / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Run directory: {run_dir}")

        config_path = common.paths.resolve_run_config_path(run_dir)
        experiments.config.loader.save_yaml(config, config_path)
        print(f"Config saved: {config_path}")

    task, _ = _run_identity(config, label="Effective config")

    print("Creating dataloaders...")
    dataloaders = experiments.config.loader.create_dataloaders_from_config(
        config,
        split_indices=saved_split_indices,
        data_processor=restored_data_processor,
    )
    train_loader = dataloaders["train"]
    eval_loader = dataloaders["eval"]
    data_processor = dataloaders["data_processor"]
    split_indices = dataloaders["split_indices"]

    if args.resume is not None:
        if data_processor is not restored_data_processor:
            msg = "Dataloader construction replaced the saved data processor during resume."
            raise RuntimeError(msg)
        if saved_split_indices is None:
            msg = "Internal error: resume split membership was not loaded."
            raise RuntimeError(msg)
        _validate_reused_split_membership(saved_split_indices, split_indices)
    else:
        normalizer_path = common.paths.resolve_normalizer_path(run_dir)
        torch.save(data_processor.state_dict(), normalizer_path)

        split_indices_path = common.paths.resolve_split_indices_path(run_dir)
        torch.save(split_indices, split_indices_path)

        print(f"Normalizer saved: {normalizer_path}")
        print(f"Split indices saved: {split_indices_path}")

    print(f"Building model: {config['model']['architecture']}")
    model = learning.models.factory.build_model(config)

    print("Building loss functions...")
    train_loss = learning.losses.factory.build_training_loss(config)
    set_normalizers = getattr(train_loss, "set_normalizers", None)
    if callable(set_normalizers):
        set_normalizers(
            in_normalizer=data_processor.in_normalizer,
            out_normalizer=data_processor.out_normalizer,
        )
    eval_losses = learning.losses.factory.build_eval_losses(config, out_normalizer=data_processor.out_normalizer)

    print("Building optimizer and scheduler...")
    optimizer = learning.training.optim.build_optimizer(model, config)
    scheduler = learning.training.optim.build_scheduler(optimizer, config)

    print("Starting training loop...")
    start_time = datetime.now(UTC)

    result = learning.training.loop.train_loop(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=train_loss,
        eval_losses=eval_losses,
        data_processor=data_processor,
        scheduler=scheduler,
        save_dir=run_dir,
        use_amp=config["training"].get("mixed_precision", False),
        resume_from=checkpoint_path,
    )

    end_time = datetime.now(UTC)
    elapsed_seconds = (end_time - start_time).total_seconds()

    summary = {
        "task": task,
        "model_architecture": config["model"]["architecture"],
        "best_epoch": result["best_epoch"],
        "best_metric": result["best_metric"],
        "metric_name": config["training"].get("save_best_metric", "eval_overall_rmse"),
        "checkpoint_path": result["checkpoint_path"],
        "status": result["status"],
        "elapsed_seconds": elapsed_seconds,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }

    # summary.json is deliberately not required for resume: interrupted runs may
    # not have one. An existing summary is replaced only after this loop returns.
    summary_path = common.paths.resolve_run_summary_path(run_dir)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved: {summary_path}")
    print("\nTraining complete!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best metric: {result['best_metric']:.6f}")
    print(f"  Elapsed time: {elapsed_seconds:.1f}s")
    print(f"  Status: {result['status']}")

    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
