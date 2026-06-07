"""
Loss functions for supervised and physics-informed neural operator training.

Provides:
  - PINOPhysicalLossEps: PINO Brinkman loss with physical derivatives and conservative continuity
  - PINOPhysicalLossDiv: PINO Brinkman loss with physical derivatives and plain continuity
  - PINOSpectralLossEps: PINO Brinkman loss with spectral derivatives and conservative continuity
  - PINOSpectralLossDiv: PINO Brinkman loss with spectral derivatives and plain continuity
  - build_supervised_loss: Build supervised loss (data loss only)
  - build_pino_loss: Build PINO loss (data + physics)
  - build_training_loss: Config-driven training loss factory
  - build_eval_losses: Build evaluation loss suite
"""

from .learning_losses_factory import (
    build_eval_losses,
    build_pino_loss,
    build_supervised_loss,
    build_training_loss,
)
from .learning_losses_pino import (
    PINOPhysicalLossDiv,
    PINOPhysicalLossEps,
    PINOSpectralLossDiv,
    PINOSpectralLossEps,
)

__all__ = [
    "PINOPhysicalLossDiv",
    "PINOPhysicalLossEps",
    "PINOSpectralLossDiv",
    "PINOSpectralLossEps",
    "build_eval_losses",
    "build_pino_loss",
    "build_supervised_loss",
    "build_training_loss",
]
