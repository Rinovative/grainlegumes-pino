"""
Task-resolved metric modules for training and evaluation.

Provides:
- metrics: semantic metric registry and explicit-space accumulators
"""

from . import learning_metrics as metrics

__all__ = [
    "metrics",
]
