"""
Generated COMSOL batch admission and final training-dataset publication.

Provides:
- generated_batch: read-only generated-batch admission and materialization
- load_generated_batch: canonical generated-batch reader
- build_training_dataset: final tensor and metadata publication command
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import build_training_dataset, generated_batch
    from .generated_batch import load_generated_batch

__all__ = [
    "build_training_dataset",
    "generated_batch",
    "load_generated_batch",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    if name in {"build_training_dataset", "generated_batch"}:
        value = import_module(f"{__name__}.{name}")
    elif name == "load_generated_batch":
        value = import_module(f"{__name__}.generated_batch").load_generated_batch
    else:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    globals()[name] = value
    return value
