"""
Lazy experiment configuration and execution services.

Provides:
- cli: import-free command modules and shared argument definitions
- config: strict defaults and semantic YAML resolution
- run: saved-run allocation, execution, resume, and validation
- tracking: optional local-authoritative W&B observability
- tuning: Optuna search-space and study orchestration

Services are imported on first attribute access so importing this package does
not initialize optional SDKs or runtime-heavy experiment modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cli, config, tuning
    from . import experiments_run as run
    from . import experiments_tracking as tracking

_MODULES = {
    "cli": "cli",
    "config": "config",
    "run": "experiments_run",
    "tracking": "experiments_tracking",
    "tuning": "tuning",
}
__all__ = ["cli", "config", "run", "tracking", "tuning"]


def __getattr__(name: str) -> object:
    """
    Resolve one declared public experiment service on first access.

    Parameters
    ----------
    name : str
        Public alias listed in :data:`__all__`.

    Returns
    -------
    object
        Imported service module, cached in this package namespace.

    Raises
    ------
    AttributeError
        If ``name`` is not a declared service alias.

    """
    if name not in _MODULES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{_MODULES[name]}")
    globals()[name] = module
    return module
