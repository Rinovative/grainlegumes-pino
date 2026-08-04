"""
Scientific plots for model-evaluation sections.

Provides:
- error_behavior: predictive-error distributions, maps, and decompositions
- layout: shared physical-map sizing and colorbar contracts
- physical_consistency: residual and boundary-consistency plots
- run_summary: completed-run overview plots
- samples_outliers: sample, outlier, and extreme-case plots
- sensitivity_capacity: capacity and metadata-sensitivity plots
- spectral_fidelity: prediction-spectrum fidelity plots
"""

from . import evaluation_plot_error_behavior as error_behavior
from . import evaluation_plot_layout as layout
from . import evaluation_plot_physical_consistency as physical_consistency
from . import evaluation_plot_run_summary as run_summary
from . import evaluation_plot_samples_outliers as samples_outliers
from . import evaluation_plot_sensitivity_capacity as sensitivity_capacity
from . import evaluation_plot_spectral_fidelity as spectral_fidelity

__all__ = [
    "error_behavior",
    "layout",
    "physical_consistency",
    "run_summary",
    "samples_outliers",
    "sensitivity_capacity",
    "spectral_fidelity",
]
