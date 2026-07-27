"""
Completed-epoch training execution and persistence.

Provides:
- checkpoint: strict best/last checkpoint capture and restoration
- loop: training, explicit-space evaluation, and lifecycle callbacks
- optim: optimizer and objective-aware scheduler construction
"""

from . import learning_training_checkpoint as checkpoint
from . import learning_training_loop as loop
from . import learning_training_optim as optim

__all__ = [
    "checkpoint",
    "loop",
    "optim",
]
