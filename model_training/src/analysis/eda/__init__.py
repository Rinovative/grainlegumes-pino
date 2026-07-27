"""
Exploratory data analysis modules.

Provides:
- dataframe: EDA DataFrame construction
- panel: numbered lazy EDA notebook composition
- plots: EDA plot modules
"""

from . import eda_dataframe as dataframe
from . import eda_panel as panel
from . import plots

__all__ = [
    "dataframe",
    "panel",
    "plots",
]
