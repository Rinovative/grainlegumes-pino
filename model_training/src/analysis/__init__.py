"""
Analysis, evaluation, visualization and artifact generation.

Provides:
- artifacts: reusable artifact generation logic
- eda: exploratory data analysis modules
- evaluation: evaluation panels, tables and plots
- ui: notebook and widget helpers
"""

from . import analysis_artifact_service as artifact_service
from . import analysis_artifacts as artifacts
from . import eda, evaluation, ui

__all__ = [
    "artifact_service",
    "artifacts",
    "eda",
    "evaluation",
    "ui",
]
