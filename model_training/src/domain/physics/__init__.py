"""
Physics contracts and numerical operators.

Provides:
- boundary: boundary-condition evaluation
- brinkman: Darcy-Brinkman residual computation
- contracts: semantic derivative and continuity policies
- derivatives: spatial derivative operators
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import domain_physics_boundary as boundary
    from . import domain_physics_brinkman as brinkman
    from . import domain_physics_contracts as contracts
    from . import domain_physics_derivatives as derivatives

_MODULES = {
    "boundary": "domain_physics_boundary",
    "brinkman": "domain_physics_brinkman",
    "contracts": "domain_physics_contracts",
    "derivatives": "domain_physics_derivatives",
}
__all__ = ["boundary", "brinkman", "contracts", "derivatives"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
