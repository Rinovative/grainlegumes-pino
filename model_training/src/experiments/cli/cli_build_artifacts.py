"""
===============================================================================
cli_build_artifacts.py
===============================================================================
Parse artifact-build arguments and delegate evaluable-run orchestration.

Responsibilities:
  - Validate task, run selection, paths, rebuild intent, and device policy
  - Report material discovery, inference, or publication failures as exit 1
  - Request atomic target replacement only through explicit ``--rebuild``

Design principles:
  - Parser and dispatch code stays thin, import-light, and service-agnostic
  - Runtime device policy is forwarded unchanged for service-owned resolution
  - Material service failures remain observable through the process exit code

This module does NOT:
  - Admit, generate, cache, or publish artifacts. ``analysis`` owns those services
  - Inspect run contents or render scientific outputs
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cli_device


def _build_parser() -> argparse.ArgumentParser:
    """Build the artifact-generation parser."""
    parser = argparse.ArgumentParser(
        description="Generate or validate split-aware artifacts for terminal evaluable runs.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--task",
        default=None,
        help="Registered task used to resolve the default run discovery root.",
    )
    selection.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory containing runs, or one current run directory.",
    )
    selection.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        type=Path,
        default=None,
        help="Exact current direct-run or Optuna-trial directory. May be repeated.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Current dataset root, independent from output/run paths.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Validated dataset metadata root. Defaults below MODEL_TRAINING_DATA_ROOT.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=None,
        help="Selected run name under the resolved runs root. May be repeated.",
    )
    cli_device.add_device_argument(parser, default="auto")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Deliberately regenerate and atomically replace existing selected artifact targets; not required for initial generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse artifact arguments and return the delegated process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` after all selected runs are validated, or ``1`` after a caught
        discovery, inference, generation, or publication failure.

    Notes
    -----
    ``argparse`` usage errors raise ``SystemExit`` before delegation. Runtime
    services are imported only after parsing succeeds.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    task = args.task
    runs_root = args.runs_root
    try:
        from src import analysis, common, domain  # noqa: PLC0415

        if task is not None:
            domain.tasks.registry.get_task(task)
            runs_root = common.paths.resolve_runs_root(task)
        if runs_root is None and args.run_dirs is None:
            parser.error("exactly one of --task, --runs-root, or --run-dir is required")
        dataset_root = args.dataset_root if args.dataset_root is not None else common.paths.get_training_raw_root()
        metadata_root = args.metadata_root if args.metadata_root is not None else common.paths.get_training_meta_root()
        if args.run_dirs is not None and args.run_names is not None:
            parser.error("--run-name selects storage aliases below --task/--runs-root and cannot accompany --run-dir")
        if args.run_dirs is not None:
            selected_roots = args.run_dirs
        elif runs_root is not None:
            selected_roots = [runs_root]
        else:
            parser.error("artifact selection unexpectedly lacks a run root")
        result_count = 0
        for selected_root in selected_roots:
            results = analysis.artifacts.service.build_artifacts(
                runs_root=selected_root,
                dataset_root=dataset_root,
                metadata_root=metadata_root,
                run_names=args.run_names,
                device_policy=args.device,
                rebuild=args.rebuild,
            )
            result_count += len(results)
    except Exception as error:  # noqa: BLE001
        print(f"Artifact generation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[DONE] Validated artifacts for {result_count} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
