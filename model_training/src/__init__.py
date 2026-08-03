"""
Public scientific, data, experiment, and learning services for the project.

Provides:
- analysis: artifact, EDA, evaluation, presentation, and UI services
- common: shared filesystem, locking, and serialization infrastructure
- datasets: dataset contracts, metadata, loaders, and simulation readers
- domain: scientific fields, physics, permeability, and task contracts
- experiments: configuration, execution, tracking, tuning, and validation services
- learning: device, inference, loss, metric, model, and training services
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis, common, datasets, domain, experiments, learning

__all__ = [
    "analysis",
    "common",
    "datasets",
    "domain",
    "experiments",
    "learning",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    if name not in __all__:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
