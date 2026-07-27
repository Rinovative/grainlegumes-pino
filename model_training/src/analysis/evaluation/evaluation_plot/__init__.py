"""
Consolidated scientific plots used by numbered evaluation sections.

Provides:
- error_behavior: predictive distributions, spatial errors, and boundary effects
- physical_consistency: momentum, continuity, and pressure diagnostics
- run_summary: authoritative accuracy and physics trade-off summaries
- samples_outliers: linked samples, ranked errors, and extreme inputs
- sensitivity_capacity: model capacity and metadata sensitivity views
- spectral_fidelity: reference, prediction, error, and transfer spectra
"""

from . import evaluation_plot_error_behavior as error_behavior
from . import evaluation_plot_physical_consistency as physical_consistency
from . import evaluation_plot_run_summary as run_summary
from . import evaluation_plot_samples_outliers as samples_outliers
from . import evaluation_plot_sensitivity_capacity as sensitivity_capacity
from . import evaluation_plot_spectral_fidelity as spectral_fidelity

__all__ = [
    "error_behavior",
    "physical_consistency",
    "run_summary",
    "samples_outliers",
    "sensitivity_capacity",
    "spectral_fidelity",
]
