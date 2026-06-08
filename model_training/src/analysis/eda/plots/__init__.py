"""
Exploratory data analysis plot modules.

Provides:
- case_statistics: case-level metadata and field statistics plots
- spectral: dataset spectral-analysis plots
"""

from . import eda_plot_case_statistics as case_statistics
from . import eda_plot_spectral_analysis as spectral

__all__ = [
    "case_statistics",
    "spectral",
]
