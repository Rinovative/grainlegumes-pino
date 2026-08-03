"""
Evaluation of persisted model predictions and scientific evidence.

Provides:
- case: case-level prediction artifact access
- dataframe: aggregate evaluation-table construction
- panel: interactive evaluation notebook panel
- plots: scientific evaluation visualizations
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import evaluation_case as case
    from . import evaluation_dataframe as dataframe
    from . import evaluation_panel as panel
    from . import evaluation_plot as plots

_MODULES = {
    "case": "evaluation_case",
    "dataframe": "evaluation_dataframe",
    "panel": "evaluation_panel",
    "plots": "evaluation_plot",
}
__all__ = ["case", "dataframe", "panel", "plots"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
