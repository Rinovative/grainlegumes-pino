"""
Loss factory and PINO derivative/loss modules.

Provides:
- factory: config-driven training and evaluation loss construction
- physical: physical-space derivative operators
- pino: PINO Brinkman loss classes
- spectral: FFT-based derivative operators
"""

from . import learning_losses_factory as factory
from . import learning_losses_physical as physical
from . import learning_losses_pino as pino
from . import learning_losses_spectral as spectral

__all__ = [
    "factory",
    "physical",
    "pino",
    "spectral",
]
