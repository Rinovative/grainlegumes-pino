"""
Dataset abstractions and simulation dataset modules.

Provides:
- base: split validation, normalizer reconstruction, and dataloaders
- identity: strict case, merged-dataset and split identity contracts
- modules: sample-construction dataset modules
- simulation: simulation dataset class
"""

from . import dataset_base as base
from . import dataset_identity as identity
from . import dataset_modules as modules
from . import dataset_simulation as simulation

__all__ = [
    "base",
    "identity",
    "modules",
    "simulation",
]
