"""
Scientific analysis, evaluation, presentation, and notebook interfaces.

Provides:
- artifacts: artifact schemas, generation, lifecycle, and runtime evidence
- eda: exploratory generated-data analysis and visualization
- evaluation: persisted prediction evaluation and scientific plots
- presentation: ordered and curated scientific presentation services
- ui: reusable notebook controls, panels, and viewers
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import artifacts, eda, evaluation, presentation, ui

_MODULES = {"artifacts": "artifacts", "eda": "eda", "evaluation": "evaluation", "presentation": "presentation", "ui": "ui"}
__all__ = ["artifacts", "eda", "evaluation", "presentation", "ui"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    if name not in _MODULES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{_MODULES[name]}")
    globals()[name] = module
    return module
