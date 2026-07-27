"""
Expose the six public project packages through lazy top-level access.

The initializer publishes ``analysis``, ``common``, ``datasets``, ``domain``,
``experiments``, and ``learning`` without importing their optional or expensive
dependencies during ``import src``. Attribute access imports and caches only the
requested package; command execution and runtime initialization remain owned by
the package-specific services.
"""

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
    """
    Import and cache one declared public package on first attribute access.

    Parameters
    ----------
    name : str
        Attribute requested from ``src``.

    Returns
    -------
    object
        Imported package object for a name listed in ``__all__``.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the six declared public package aliases.

    """
    if name not in __all__:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
