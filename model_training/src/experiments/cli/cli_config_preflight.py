"""Host-safe executable-config workflow preflight command."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the config-family preflight parser."""
    parser = argparse.ArgumentParser(
        description="Classify and strictly validate a train or Optuna config without runtime allocation",
    )
    parser.add_argument("workflow", choices=("train", "optuna"))
    parser.add_argument("config_path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate one requested workflow and emit a tab-delimited host summary."""
    args = _build_parser().parse_args(argv)
    try:
        from src.experiments.config import experiments_config_preflight as preflight  # noqa: PLC0415

        result = preflight.validate_workflow(
            args.config_path,
            requested_workflow=args.workflow,
        )
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 2
    values = (result.family, result.task, result.canonical_path)
    if any("\t" in value or "\n" in value for value in values):
        print("Configuration preflight fields contain unsupported control characters.", file=sys.stderr)
        return 2
    print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
