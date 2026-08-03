"""
===============================================================================
domain_physics_contracts.py
===============================================================================
Define dependency-light semantic identifiers for steady-flow physics.

Responsibilities:
  - Own canonical Brinkman equation, boundary, continuity, and derivative identifiers
  - Validate continuity and derivative selectors without tensor implementations
  - Provide the default and complete supported continuity contract

Design principles:
  - Task declarations and runtime equations share one identifier source
  - Configuration admission remains independent of Torch-backed computation
  - Exact identifiers fail closed without aliases or compatibility fallbacks

This module does NOT:
  - Implement residual equations, derivatives, masks, or scalar reductions
  - Import Torch or allocate tensors
  - Select experiment loss weights or diagnostic schedules
===============================================================================
"""

from __future__ import annotations

from typing import Final, Literal, cast

ContinuityKind = Literal["div_eps_velocity", "div_velocity"]
DerivativeKind = Literal["physical", "spectral"]
SpectralExtension = Literal["none", "reflect"]
STEADY_BRINKMAN_KIND: Final = "steady_2d_brinkman"
STEADY_BRINKMAN_EQUATION_SET: Final = "steady_two_dimensional_brinkman"
DEFAULT_CONTINUITY_KIND: Final[ContinuityKind] = "div_eps_velocity"
PRESSURE_BOUNDARY_KIND: Final = "pressure_inlet_zero_pressure_outlet"
_CONTINUITY_KINDS: Final = ("div_velocity", "div_eps_velocity")
_DERIVATIVE_KINDS: Final = ("physical", "spectral")
_DERIVATIVE_EXTENSIONS: Final = ("none", "reflect")

__all__ = [
    "DEFAULT_CONTINUITY_KIND",
    "PRESSURE_BOUNDARY_KIND",
    "STEADY_BRINKMAN_EQUATION_SET",
    "STEADY_BRINKMAN_KIND",
    "ContinuityKind",
    "DerivativeKind",
    "SpectralExtension",
    "available_continuity_kinds",
    "available_derivative_kinds",
    "validate_continuity_kind",
    "validate_derivative_kind",
]


def available_continuity_kinds() -> tuple[str, ...]:
    """Return supported semantic continuity formulations."""
    return _CONTINUITY_KINDS


def available_derivative_kinds() -> tuple[str, ...]:
    """Return supported semantic numerical derivative identifiers."""
    return _DERIVATIVE_KINDS


def validate_derivative_kind(kind: str, *, extension: str) -> tuple[DerivativeKind, SpectralExtension]:
    """Validate and return one exact derivative kind and extension pair."""
    if kind not in _DERIVATIVE_KINDS:
        available = ", ".join(_DERIVATIVE_KINDS)
        msg = f"Unknown derivative identifier {kind!r}. Available derivatives: {available}."
        raise ValueError(msg)
    if extension not in _DERIVATIVE_EXTENSIONS:
        available = ", ".join(_DERIVATIVE_EXTENSIONS)
        msg = f"Unknown derivative extension {extension!r}. Available extensions: {available}."
        raise ValueError(msg)
    if kind == "physical" and extension != "none":
        msg = "Physical derivatives require extension 'none'."
        raise ValueError(msg)
    return cast("DerivativeKind", kind), cast("SpectralExtension", extension)


def validate_continuity_kind(kind: str) -> ContinuityKind:
    """
    Validate and return one exact continuity formulation identifier.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"div_velocity"`` or ``"div_eps_velocity"``.

    """
    if kind not in _CONTINUITY_KINDS:
        available = ", ".join(_CONTINUITY_KINDS)
        msg = f"Unknown continuity identifier {kind!r}. Available continuity formulations: {available}."
        raise ValueError(msg)
    return cast("ContinuityKind", kind)
