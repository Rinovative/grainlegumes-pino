"""
===============================================================================
 experiments.cli
===============================================================================
Lazy-loaded executable entry points for experiments.

Responsibilities:
  - Expose CLI modules without import-time training side effects
  - Keep CLI discovery lightweight for help and package introspection

This package does NOT:
  - Execute CLI logic at import time
  - Re-export reusable training or tuning internals
===============================================================================
"""

from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load CLI modules to avoid import-time side effects."""
    if name == "cli_build_artifacts":
        return importlib.import_module(f"{__name__}.cli_build_artifacts")
    if name == "cli_train":
        return importlib.import_module(f"{__name__}.cli_train")
    if name == "cli_optuna":
        return importlib.import_module(f"{__name__}.cli_optuna")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return lazily available CLI module names."""
    return ["cli_build_artifacts", "cli_optuna", "cli_train"]


__all__ = []
