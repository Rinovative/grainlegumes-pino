"""
Supervised and physics-informed loss services.

Provides:
- factory: semantic training-loss construction
- pino: physics-informed loss composition
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import learning_losses_factory as factory
    from . import learning_losses_pino as pino

_MODULES = {
    "factory": "learning_losses_factory",
    "pino": "learning_losses_pino",
}
__all__ = ["factory", "pino"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
