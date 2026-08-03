"""
Experiment configuration, execution, tracking, tuning, and validation services.

Provides:
- cli: command-line experiment workflows
- config: strict defaults and YAML configuration resolution
- notebook_support: read-only notebook context and presentation preparation
- run: experiment lifecycle, identity, persistence, and resume
- tracking: W&B observer lifecycle and metric publication
- tuning: search-space and Optuna-study services
- validation: explicit production data-pipeline validation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cli, config, tuning, validation
    from . import experiments_console as console
    from . import experiments_notebook_support as notebook_support
    from . import experiments_run as run
    from . import experiments_tracking as tracking

_MODULES = {
    "cli": "cli",
    "config": "config",
    "console": "experiments_console",
    "notebook_support": "experiments_notebook_support",
    "run": "experiments_run",
    "tracking": "experiments_tracking",
    "tuning": "tuning",
    "validation": "validation",
}
__all__ = [
    "cli",
    "config",
    "console",
    "notebook_support",
    "run",
    "tracking",
    "tuning",
    "validation",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    if name not in _MODULES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{_MODULES[name]}")
    globals()[name] = module
    return module
