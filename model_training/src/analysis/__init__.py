"""Analysis, evaluation and visualization."""

import importlib
from typing import Any

from . import eda, evaluation, ui


def __getattr__(name: str) -> Any:
    """Lazy-load analysis_artifacts to avoid Phase 7 training dependencies."""
    if name == "analysis_artifacts":
        return importlib.import_module(f"{__name__}.analysis_artifacts")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["analysis_artifacts", "eda", "evaluation", "ui"]


__all__ = ["analysis_artifacts", "eda", "evaluation", "ui"]
