"""
Exploratory analysis of admitted generated batches.

Provides:
- dataframe: generated-batch materialization for analysis
- panel: interactive exploratory-analysis notebook panel
- plots: scientific generated-data visualizations
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import eda_dataframe as dataframe
    from . import eda_panel as panel
    from . import plots

_MODULES = {
    "dataframe": "eda_dataframe",
    "panel": "eda_panel",
    "plots": "plots",
}
__all__ = ["dataframe", "panel", "plots"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
