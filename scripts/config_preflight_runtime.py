#!/usr/bin/env python3
"""Guard the project runtime before importing the authoritative config preflight."""

import runpy
import sys
from typing import Any

MINIMUM_PYTHON = (3, 11)
PREFLIGHT_MODULE = "src.experiments.cli.cli_config_preflight"


def _version_text(version_info: Any) -> str:
    """Return a stable three-component runtime version."""
    return ".".join(str(component) for component in version_info[:3])


def main() -> int:
    """Reject an incompatible image runtime, then delegate without changing arguments."""
    if sys.version_info < MINIMUM_PYTHON:
        print("Configuration preflight could not start.", file=sys.stderr)
        print(
            f"Project runtime Python: {_version_text(sys.version_info)} ({sys.executable})",
            file=sys.stderr,
        )
        print("Required Python: >=3.11", file=sys.stderr)
        print(
            "The maintained project image must be rebuilt with the declared project runtime.",
            file=sys.stderr,
        )
        return 1
    runpy.run_module(PREFLIGHT_MODULE, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
