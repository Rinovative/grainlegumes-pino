"""
Legacy notebook helper for the current artifact CLI.

From the model_training directory, prefer running:
    python -m src.experiments.cli.cli_build_artifacts --task steady_flow

This wrapper is retained for configured notebook environments that already
expose the model_training package root on Python's import path.
"""

from __future__ import annotations

from src.experiments.cli.cli_build_artifacts import main

if __name__ == "__main__":
    raise SystemExit(main())
