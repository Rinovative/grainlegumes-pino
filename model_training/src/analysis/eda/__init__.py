"""
Exploratory data analysis modules.

Provides:
- dataframe: EDA DataFrame construction
- plots: EDA plot modules
"""

from . import eda_dataframe as dataframe
from . import plots

__all__ = [
    "dataframe",
    "plots",
]
