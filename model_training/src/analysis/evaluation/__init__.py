"""
Versioned artifact evaluation readers, panels, and scientific plots.

Provides:
- case: validated per-case NPZ loading and grid metadata
- dataframe: schema-aware table loading and comparison admission
- panel: numbered lazy evaluation-panel composition
- plots: consolidated scientific plot implementations
"""

from . import evaluation_case as case
from . import evaluation_dataframe as dataframe
from . import evaluation_panel as panel
from . import evaluation_plot as plots

__all__ = [
    "case",
    "dataframe",
    "panel",
    "plots",
]
