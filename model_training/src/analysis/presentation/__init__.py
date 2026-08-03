"""
Scientific presentation services.

Provides:
- curated: fixed local scientific media rendering
- registry: numbered EDA and evaluation presentation definitions
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from . import analysis_presentation_registry as registry

if TYPE_CHECKING:
    from . import analysis_presentation_curated as curated

_MODULES = {"curated": "analysis_presentation_curated"}
__all__ = ["curated", "registry"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
