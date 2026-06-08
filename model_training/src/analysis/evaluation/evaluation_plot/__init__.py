"""
Evaluation plot modules.

Provides:
- architecture: architecture sensitivity plots
- error_decomposition: spatial error decomposition plots
- global_error: aggregate error plots
- outliers: outlier and extreme-case plots
- overview: summary scoreboard plots
- parameter_sensitivity: parameter/error sensitivity plots
- physical_consistency: physics diagnostic plots
- sample_viewer: per-case field viewers
- spectral: spectral evaluation plots
"""

from . import (
    evaluation_plot_architecture_sensitivity as architecture,
)
from . import (
    evaluation_plot_error_decomposition as error_decomposition,
)
from . import (
    evaluation_plot_global_error_analysis as global_error,
)
from . import (
    evaluation_plot_outlier_analysis as outliers,
)
from . import (
    evaluation_plot_overview_scoreboard as overview,
)
from . import (
    evaluation_plot_parameter_sensitivity as parameter_sensitivity,
)
from . import (
    evaluation_plot_physical_consistency as physical_consistency,
)
from . import (
    evaluation_plot_sample_viewer as sample_viewer,
)
from . import (
    evaluation_plot_spectral_analysis as spectral,
)

__all__ = [
    "architecture",
    "error_decomposition",
    "global_error",
    "outliers",
    "overview",
    "parameter_sensitivity",
    "physical_consistency",
    "sample_viewer",
    "spectral",
]
