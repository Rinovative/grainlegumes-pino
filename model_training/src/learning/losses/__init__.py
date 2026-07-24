"""Semantic supervised and physics-informed loss composition."""

from . import learning_losses_factory as factory
from . import learning_losses_pino as pino

__all__ = ["factory", "pino"]
