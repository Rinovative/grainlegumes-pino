"""
===============================================================================
cli_train.py
===============================================================================
Parse training CLI arguments and delegate to the reusable run service.

Responsibilities:
  - Parse config, resume, device, output-root, and artifact opt-out arguments
  - Orchestrate completed-run artifact preparation after training ownership ends
  - Return a non-zero process result for training or post-processing failures

Design principles:
  - Parser and dispatch code stays thin and import-light
  - Runtime overrides are forwarded without semantic reinterpretation
  - Training completion and artifact post-processing remain separate facts
  - Material lifecycle failures remain visible to shell and queue callers

This module does NOT:
  - Allocate, seed, persist, resume, or train runs. ``experiments.run`` owns lifecycle
  - Discover artifact roles, rebuild incompatible targets, or upload to W&B
===============================================================================
"""

from __future__ import annotations

import argparse
import shlex
import sys

from . import cli_device


def _build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser."""
    parser = argparse.ArgumentParser(description="Train a neural operator model from config")
    parser.add_argument("config_path", type=str, help="Path to experiment YAML config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Explicitly resume in place from last_checkpoint.pt in an existing run directory",
    )
    cli_device.add_device_argument(parser, default=None, help_prefix="Override run.device")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override outputs only. Dataset lookup remains bound to dataset_root",
    )
    parser.add_argument(
        "--no-build-artifacts",
        action="store_true",
        help="Skip only the default post-training generation or reuse of run-bundle evaluation artifacts",
    )
    return parser


def _print_existing_run_admission(report: dict[str, object]) -> None:
    """Render one actionable, non-interactive fresh-run rejection."""
    completed = report.get("completed_epoch")
    target = report.get("target_epoch")
    compatibility = str(report.get("resume_compatibility"))
    active = report.get("active_lock")
    active_label = "present" if active is True else "absent" if active is False else "unknown"
    state = report.get("state")
    state_mapping = state if isinstance(state, dict) else {}
    state_label = ", ".join(f"{name}={'available' if available else 'missing'}" for name, available in state_mapping.items())

    print("Run admission rejected: existing run requires explicit resume.", file=sys.stderr)
    print(f"Requested canonical run name: {report.get('requested_run_name')}", file=sys.stderr)
    print(f"Canonical config path: {report.get('config_path')}", file=sys.stderr)
    print(f"Existing directory: {report.get('run_dir')}", file=sys.stderr)
    print(f"Lifecycle status: {report.get('status')}", file=sys.stderr)
    print(f"Completed epoch: {completed if completed is not None else 'unknown'}", file=sys.stderr)
    print(f"Requested target epoch: {target}", file=sys.stderr)
    print(
        f"Last checkpoint: {'available' if report.get('last_checkpoint_available') else 'missing'}",
        file=sys.stderr,
    )
    print(
        f"Best checkpoint: {'available' if report.get('best_checkpoint_available') else 'missing'}",
        file=sys.stderr,
    )
    print(f"Manifest/state: {state_label or 'unavailable'}", file=sys.stderr)
    print(f"Active lock: {active_label}", file=sys.stderr)
    print(f"Resume compatibility: {compatibility}", file=sys.stderr)
    print(f"Reason: {report.get('reason')}", file=sys.stderr)

    if compatibility == "completed":
        print("Run already completed. No training was started.", file=sys.stderr)
        print(f"Completed epoch: {completed if completed is not None else 'unknown'} / {target}", file=sys.stderr)
    elif compatibility == "compatible":
        config_path = shlex.quote(str(report.get("config_path")))
        run_dir = shlex.quote(str(report.get("run_dir")))
        print("Resume explicitly with:", file=sys.stderr)
        print(
            f"  ./scripts/docker_job.sh train {config_path} --resume {run_dir}",
            file=sys.stderr,
        )


def _artifact_action_label(role_actions: dict[str, str]) -> str:
    """Summarize ID and OOD artifact actions without role-level verbosity."""
    actions = set(role_actions.values())
    if actions == {"generated"}:
        return "generated"
    if actions == {"reused"}:
        return "reused"
    if actions and actions.issubset({"generated", "reused"}):
        return "mixed"
    msg = f"Unexpected automatic artifact actions: {sorted(actions)}."
    raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> int:
    """
    Execute a fresh or explicit-resume run and return its process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` after completed training and requested post-processing, ``1``
        for a caught training or artifact failure, and ``130`` when interrupted.

    Notes
    -----
    The reusable run service owns allocation, persistence, device resolution,
    training, and resume. The artifact service owns role discovery, validation,
    generation, and reuse. Parser usage errors raise ``SystemExit`` directly.

    """
    args = _build_parser().parse_args(argv)
    try:
        from src.experiments import experiments_run  # noqa: PLC0415

        outcome = experiments_run.run_experiment(
            args.config_path,
            resume=args.resume,
            device=args.device,
            output_root=args.output_root,
        )
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        from src.experiments import experiments_console  # noqa: PLC0415
        from src.experiments import experiments_run as run_service  # noqa: PLC0415

        if isinstance(error, run_service.ExistingRunAdmissionError):
            _print_existing_run_admission(error.report)
            return 1
        print("Training failed. Sanitized traceback follows.", file=sys.stderr, flush=True)
        experiments_console.print_sanitized_traceback(error)
        return 1

    result = outcome["result"]
    run_dir = outcome["run_dir"]
    device_resolution = outcome["device_resolution"]
    print("Training status: completed")
    print(f"Run directory: {run_dir}")
    print("Best checkpoint: best_checkpoint.pt")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best metric: {result['best_metric']:.6f}")
    if args.no_build_artifacts:
        print("Post-training artifacts: skipped by --no-build-artifacts")
        return 0

    try:
        from src import analysis  # noqa: PLC0415

        experiments_run.validate_completed_run(run_dir)
        analysis.artifacts.service.cleanup_runtime(device_resolution.device)
        prepared = analysis.artifacts.service.load_or_build_run_artifacts(
            run_dir,
            device_resolution=device_resolution,
        )
        artifact_status = _artifact_action_label(prepared.role_actions)
    except KeyboardInterrupt:
        print("Post-training artifacts: failed", file=sys.stderr)
        print(f"Run directory: {run_dir}", file=sys.stderr)
        print("Recovery: run the artifact CLI or evaluation notebook later", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        from src.experiments import experiments_console  # noqa: PLC0415

        print("Post-training artifacts: failed", file=sys.stderr)
        print(f"Artifact error: {type(error).__name__}: {experiments_console.sanitized_exception_message(error)}", file=sys.stderr)
        print(f"Run directory: {run_dir}", file=sys.stderr)
        print("Recovery: run the artifact CLI or evaluation notebook later", file=sys.stderr)
        return 1

    print(f"Post-training artifacts: {artifact_status}")
    print(f"Artifact device: {device_resolution.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
