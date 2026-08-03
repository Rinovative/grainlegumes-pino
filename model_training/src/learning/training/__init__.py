"""
Completed-epoch training, optimization, and persistence services.

Provides:
- checkpoint: atomic checkpoint and continuation-state persistence
- loop: training and evaluation epoch execution
- optim: optimizer and scheduler construction
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import learning_training_checkpoint as checkpoint
    from . import learning_training_events as events
    from . import learning_training_loop as loop
    from . import learning_training_optim as optim

_MODULES = {
    "checkpoint": "learning_training_checkpoint",
    "events": "learning_training_events",
    "loop": "learning_training_loop",
    "optim": "learning_training_optim",
}
__all__ = ["checkpoint", "events", "loop", "optim"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
