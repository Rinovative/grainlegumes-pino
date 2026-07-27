"""
Artifact production and notebook-centered scientific analysis.

Provides:
- artifact_service: cache discovery, validation, generation, rebuilding, and upload
- artifacts: task-aware Parquet and NPZ artifact generation
- curated_renderer: fixed local scientific media rendering
- eda: exploratory dataset tables, panels, and plots
- evaluation: artifact readers, panels, tables, and scientific plots
- presentation: numbered EDA and evaluation display registries
- ui: reusable notebook widgets, panels, and viewers
"""

from . import analysis_artifact_service as artifact_service
from . import analysis_artifacts as artifacts
from . import analysis_curated_renderer as curated_renderer
from . import analysis_presentation as presentation
from . import analysis_timing as timing
from . import eda, evaluation, ui

__all__ = [
    "artifact_service",
    "artifacts",
    "curated_renderer",
    "eda",
    "evaluation",
    "presentation",
    "timing",
    "ui",
]
