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

Boundaries:
  - Study orchestration belongs to experiments.tuning.optuna
  - Trial search-space parsing belongs to experiments.tuning.search_space
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the Optuna CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run an Optuna study from a YAML config")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to Optuna YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the Optuna YAML without starting training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override study.n_trials",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override experiment run.device for all trials",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override only the study and trial run/output root",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Show Optuna progress bar during study optimization",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the delegated Optuna entry point and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        from src.experiments.tuning import experiments_tuning_optuna as optuna  # noqa: PLC0415

        study_config = optuna.load_optuna_study_config(args.config_path)
        if args.dry_run:
            print(json.dumps(optuna.describe_optuna_study_config(study_config), indent=2))
            return 0
        study = optuna.run_optuna_study(
            study_config,
            n_trials=args.n_trials,
            device=args.device,
            output_root=args.output_root,
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
