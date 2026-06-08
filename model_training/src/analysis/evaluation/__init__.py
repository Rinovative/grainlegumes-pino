"""
Evaluation tables, panels and plots.

Provides:
- dataframe: evaluation DataFrame construction
- panel: interactive evaluation panel assembly
- plots: evaluation plot modules
"""

from . import evaluation_dataframe as dataframe
from . import evaluation_panel as panel
from . import evaluation_plot as plots

__all__ = [
    "dataframe",
    "panel",
    "plots",
]
