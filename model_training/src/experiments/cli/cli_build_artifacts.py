"""
===============================================================================
cli_build_artifacts.py
===============================================================================
Parse artifact CLI arguments and delegate to analysis.artifact_service.

Material run, inference, generation, and cache failures return a non-zero exit.
The explicit ``--rebuild`` flag removes only each selected artifact target.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        type=int,
        choices=(1,),
        default=1,
        help="Per-case inference batch size (currently exactly 1).",
    )
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA preference.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly remove and rebuild only each selected artifact target.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Delegate artifact orchestration and return a process exit code."""
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
        results = analysis.artifact_service.build_artifacts(
            runs_root=runs_root,
            dataset_root=dataset_root,
            run_names=args.run_names,
            max_cases=args.max_cases,
            batch_size=args.batch_size,
            prefer_cuda=not args.cpu,
            rebuild=args.rebuild,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Artifact generation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[DONE] Validated artifacts for {len(results)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
