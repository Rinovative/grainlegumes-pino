"""
Custom training loop and training utilities.

Provides:
  - train_loop: Main training loop with checkpointing and logging
  - SpectralEnergyHook: Forward hook for spectral energy diagnostics
  - build_optimizer: Build AdamW optimizer from config
  - build_scheduler: Build ReduceLROnPlateau scheduler from config
"""

from .learning_training_hooks import SpectralEnergyHook
from .learning_training_loop import train_loop
from .learning_training_optim import build_optimizer, build_scheduler

__all__ = [
    "SpectralEnergyHook",
    "build_optimizer",
    "build_scheduler",
    "train_loop",
]
