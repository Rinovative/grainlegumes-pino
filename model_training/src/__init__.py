"""Reusable project packages with lazy top-level module access."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis, common, datasets, domain, experiments, learning

__all__ = [
    "analysis",
    "common",
    "datasets",
    "domain",
    "experiments",
    "learning",
]


def __getattr__(name: str) -> object:
    """Import a public package only when it is first requested."""
    if name not in __all__:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
