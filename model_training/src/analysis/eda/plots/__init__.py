"""
Scientific plots for exploratory generated-data analysis.

Provides:
- case_statistics: case-level parameter and field-statistic plots
- spectral: isotropic, directional, and evolution spectrum plots
"""

from . import eda_plot_case_statistics as case_statistics
from . import eda_plot_spectral_analysis as spectral

__all__ = [
    "case_statistics",
    "spectral",
]
