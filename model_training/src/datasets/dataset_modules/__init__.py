"""
Dataset modules for model-ready samples.

Provides:
- flow: steady-flow tensor construction from dataset dictionaries
"""

from . import dataset_module_flow as flow

__all__ = [
    "flow",
]
