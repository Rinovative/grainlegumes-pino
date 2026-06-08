"""
Experiment defaults and YAML configuration loading.

Provides:
- defaults: task, model, loss, optimizer and scheduler defaults
- loader: YAML loading, config resolution and dataloader construction
"""

from . import experiments_config_defaults as defaults
from . import experiments_config_loader as loader

__all__ = [
    "defaults",
    "loader",
]
