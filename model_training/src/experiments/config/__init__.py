"""
Strict experiment defaults and YAML configuration services.

Provides:
- defaults: canonical experiment defaults
- loader: YAML admission, resolution, and semantic validation
"""

from . import experiments_config_defaults as defaults
from . import experiments_config_loader as loader
from . import experiments_config_preflight as preflight

__all__ = [
    "defaults",
    "loader",
    "preflight",
]
