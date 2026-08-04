"""
===============================================================================
cli_optuna.py
===============================================================================
Run or validate config-driven Optuna studies from the command line.

Responsibilities:
  - Parse Optuna YAML paths and runtime overrides
  - Print dry-run study summaries without starting training
  - Delegate study execution to experiments.tuning.optuna

Design principles:
  - CLI code stays thin and side-effect-light
  - Search spaces live in YAML files
  - Runtime overrides are explicit command-line options

This module does NOT:
  - Create, reopen, or optimize studies. ``experiments.tuning.optuna`` owns lifecycle
  - Parse trial search schemas. ``experiments.tuning.search_space`` owns admission
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys

from . import cli_device


def _build_parser() -> argparse.ArgumentParser:
    """Build the Optuna CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=("Run additional fresh Optuna trials with held-out objective reports at every actual completed epoch")
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to Optuna YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Validate config, search policy, pruning cadence, device policy, and semantic signature without creating files or importing Optuna"),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of additional fresh trials to run (existing trial history is never resumed)",
    )
    cli_device.add_device_argument(parser, default=None, help_prefix="Override experiment run.device for all trials")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Invocation-only study database and fresh trial output root (excluded from the signature)",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Show Optuna progress bar during study optimization",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Validate or execute an Optuna study and return its process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` for a successful dry run or study invocation, ``1`` for a caught
        study failure, and ``130`` for an interrupted study.

    Notes
    -----
    Dry-run validation creates no study files and imports no Optuna SDK. Parser
    usage errors still raise ``SystemExit`` through ``argparse``.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        from src.experiments.tuning import optuna  # noqa: PLC0415

        study_config = optuna.load_optuna_study_config(args.config_path)
        study_config = optuna.with_runtime_overrides(
            study_config,
            device=args.device,
            output_root=args.output_root,
        )
        if args.dry_run:
            print(json.dumps(optuna.describe_optuna_study_config(study_config), indent=2))
            return 0
        study = optuna.run_optuna_study(
            study_config,
            n_trials=args.n_trials,
            show_progress_bar=args.show_progress_bar,
        )
    except KeyboardInterrupt:
        print("Optuna study interrupted.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        print(f"Optuna study failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("Optuna study complete")
    print(f"  Study name: {study.study_name}")
    try:
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best value: {study.best_trial.value}")
    except ValueError:
        print("  Best trial: unavailable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
