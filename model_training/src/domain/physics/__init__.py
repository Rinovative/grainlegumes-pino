"""
Reusable physical-field equations and numerical operators.

Provides:
- boundary: pressure inlet/outlet masks and diagnostics
- brinkman: steady Darcy-Brinkman residuals and evaluator registry
- derivatives: physical/FFT derivative operators and Cartesian-grid admission
"""

from . import domain_physics_boundary as boundary
from . import domain_physics_brinkman as brinkman
from . import domain_physics_derivatives as derivatives

__all__ = [
    "boundary",
    "brinkman",
    "derivatives",
]
