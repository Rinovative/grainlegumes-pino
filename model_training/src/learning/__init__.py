"""
Model learning: training, inference, losses and metrics.

Core learning abstractions for model training, inference and evaluation.
"""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load submodules to avoid circular imports and training dependencies."""
    if name == "inference":
        return importlib.import_module(f"{__name__}.inference")
    if name == "metrics":
        return importlib.import_module(f"{__name__}.metrics")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["inference", "metrics"]


__all__ = []
