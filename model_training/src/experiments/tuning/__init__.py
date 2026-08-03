"""
Search-space contracts and Optuna-study orchestration.

Provides:
- optuna: study lifecycle and trial orchestration
- search_space: YAML search-space parsing and trial overrides
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from . import experiments_tuning_search_space as search_space

if TYPE_CHECKING:
    from . import experiments_tuning_optuna as optuna

_MODULES = {"optuna": "experiments_tuning_optuna"}
__all__ = ["optuna", "search_space"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
