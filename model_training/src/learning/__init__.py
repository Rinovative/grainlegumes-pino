"""
Neural-operator device, inference, loss, metric, model, and training services.

Provides:
- device: runtime device-policy resolution
- inference: saved-run inference reconstruction
- losses: supervised and physics-informed loss construction
- metrics: task-resolved training and evaluation metrics
- models: semantic neural-operator model construction
- training: optimization, checkpoint, and completed-epoch execution
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import inference, losses, metrics, models, training
    from . import learning_device as device

_MODULES = {
    "device": "learning_device",
    "inference": "inference",
    "losses": "losses",
    "metrics": "metrics",
    "models": "models",
    "training": "training",
}
__all__ = ["device", "inference", "losses", "metrics", "models", "training"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
