"""
Dataset modules for model-ready samples.

Provides:
- flow: strict task tensor construction from current dataset payloads
"""

from . import dataset_module_flow as flow

__all__ = [
    "flow",
]
