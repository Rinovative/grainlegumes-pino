"""
===============================================================================
cli_build_artifacts.py
===============================================================================
Parse artifact-build arguments and delegate completed-run orchestration.

Responsibilities:
  - Validate task, run selection, case limits, batching, and device policy
  - Report material discovery, inference, or publication failures as exit 1
  - Request atomic target replacement only through explicit ``--rebuild``

Design principles:
  - Parser and dispatch code stays thin, import-light, and service-agnostic
  - Runtime device policy is forwarded unchanged for service-owned resolution
  - Material service failures remain observable through the process exit code

This module does NOT:
  - Admit, generate, cache, or publish artifacts; ``analysis`` owns those services
  - Inspect run contents or render scientific outputs
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cli_device


def _positive_int(value: str) -> int:
    """Parse one positive CLI integer."""
    parsed = int(value)
    if parsed <= 0:
        msg = f"Expected a positive integer, got {value!r}."
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build the artifact-generation parser."""
    parser = argparse.ArgumentParser(
        description="Generate or validate split-aware artifacts for completed runs.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Registered task used to resolve the default run discovery root.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory containing runs, or one completed run directory.",
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
        help="Validated dataset metadata root; defaults below MODEL_TRAINING_DATA_ROOT.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=None,
        help="Selected run name under --runs-root; may be repeated.",
    )
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        default=None,
        help="Optional positive saved-split case limit.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Positive inference batch size; artifacts remain one row and NPZ per case.",
    )
    cli_device.add_device_argument(parser, default="auto")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate and atomically replace only each selected artifact target.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse artifact arguments and return the delegated process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector; ``None`` uses the process arguments.

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
    if args.runs_root is None and task is None:
        parser.error("--task is required when --runs-root is not supplied")
    try:
        from src import analysis, common, domain  # noqa: PLC0415

        if task is not None:
            domain.tasks.registry.get_task(task)
        if args.runs_root is not None:
            runs_root = args.runs_root
        elif task is not None:
            runs_root = common.paths.resolve_runs_root(task)
        else:
            parser.error("--task is required when --runs-root is not supplied")
        dataset_root = args.dataset_root if args.dataset_root is not None else common.paths.get_dataset_root()
        metadata_root = args.metadata_root if args.metadata_root is not None else common.paths.get_training_meta_root()
        results = analysis.artifact_service.build_artifacts(
            runs_root=runs_root,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
            run_names=args.run_names,
            max_cases=args.max_cases,
            batch_size=args.batch_size,
            device_policy=args.device,
            rebuild=args.rebuild,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Artifact generation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[DONE] Validated artifacts for {len(results)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
