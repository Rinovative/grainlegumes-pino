"""Experiment services exposed without eager runtime imports."""

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
    """Import an experiment service only when it is first requested."""
    if name not in _MODULES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{_MODULES[name]}")
    globals()[name] = module
    return module
