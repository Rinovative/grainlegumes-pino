"""
Semantic supervised and physics-informed loss composition.

Provides:
- factory: semantic loss registries and config-driven construction
- pino: weighted supervised/physics composition and warmup schedules
"""

from . import learning_losses_factory as factory
from . import learning_losses_pino as pino

__all__ = ["factory", "pino"]
