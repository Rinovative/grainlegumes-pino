"""
===============================================================================
cli_train.py
===============================================================================
Parse training CLI arguments and delegate to the reusable run service.

Responsibilities:
  - Parse config, resume, device, and output-root arguments
  - Return a non-zero process result for lifecycle failures

Design principles:
  - Parser and dispatch code stays thin and import-light
  - Runtime overrides are forwarded without semantic reinterpretation
  - Material lifecycle failures remain visible to shell and queue callers

This module does NOT:
  - Allocate, seed, persist, resume, or train runs; ``experiments.run`` owns lifecycle
  - Eagerly import command modules through the import-free ``experiments.cli`` package
===============================================================================
"""

from __future__ import annotations

import argparse
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
        help="Override outputs only; dataset lookup remains bound to dataset_root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Execute a fresh or explicit-resume run and return its process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector; ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` after a completed run, ``1`` for a caught lifecycle failure, and
        ``130`` when training is interrupted.

    Notes
    -----
    The reusable run service owns allocation, persistence, device resolution,
    training, and resume. Parser usage errors raise ``SystemExit`` directly.

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
        print(f"Training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    result = outcome["result"]
    print(f"Run directory: {outcome['run_dir']}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best metric: {result['best_metric']:.6f}")
    print("Status: completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
