"""
Evaluation of persisted model predictions and scientific evidence.

Provides:
- artifact_loader: strict path-based evaluable-run artifact access
- case: case-level prediction artifact access
- dataframe: aggregate evaluation-table construction
- panel: interactive evaluation notebook panel
- presentation: current-native field, grid, and metadata presentation data
- plots: scientific evaluation visualizations
- session: bounded evaluation-case and numerical-summary reuse
- workflow: portable run preparation, reporting, and panel composition
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import evaluation_artifact_loader as artifact_loader
    from . import evaluation_case as case
    from . import evaluation_dataframe as dataframe
    from . import evaluation_panel as panel
    from . import evaluation_plot as plots
    from . import evaluation_presentation as presentation
    from . import evaluation_session as session
    from . import evaluation_workflow as workflow

_MODULES = {
    "artifact_loader": "evaluation_artifact_loader",
    "case": "evaluation_case",
    "dataframe": "evaluation_dataframe",
    "panel": "evaluation_panel",
    "plots": "evaluation_plot",
    "presentation": "evaluation_presentation",
    "session": "evaluation_session",
    "workflow": "evaluation_workflow",
}
__all__ = ["artifact_loader", "case", "dataframe", "panel", "plots", "presentation", "session", "workflow"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
