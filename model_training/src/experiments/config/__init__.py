"""
===============================================================================
 experiments.config
===============================================================================
Public API for experiment configuration infrastructure.

Responsibilities:
  - Export config default lookup helpers
  - Export YAML and inline config resolution helpers
  - Export dataloader construction from resolved configs
  - Export YAML serialization helpers

This package does NOT:
  - Execute training
  - Define model, loss, optimizer, or scheduler factories
===============================================================================
"""

from .experiments_config_defaults import get_task_defaults
from .experiments_config_loader import (
    create_dataloaders_from_config,
    load_and_resolve_config,
    resolve_config,
    save_yaml,
)

__all__ = [
    "create_dataloaders_from_config",
    "get_task_defaults",
    "load_and_resolve_config",
    "resolve_config",
    "save_yaml",
]
