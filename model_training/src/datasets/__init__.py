"""
Dataset contracts, metadata, loaders, and simulation readers.

Provides:
- base: shared dataset interfaces
- identity: deterministic dataset identity and fingerprint contracts
- metadata: dataset metadata admission and summary services
- modules: model-ready task dataset modules
- simulation: persisted simulation-dataset access
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_base as base
    from . import dataset_identity as identity
    from . import dataset_metadata as metadata
    from . import dataset_modules as modules
    from . import dataset_simulation as simulation

_MODULES = {
    "base": "dataset_base",
    "identity": "dataset_identity",
    "metadata": "dataset_metadata",
    "modules": "dataset_modules",
    "simulation": "dataset_simulation",
}
__all__ = ["base", "identity", "metadata", "modules", "simulation"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
