"""
===============================================================================
domain_field_sets.py
===============================================================================
Define canonical input and output field sets for model tensors.

Responsibilities:
  - Declare default 2D and 3D model input fields
  - Declare default 2D and 3D model output fields
  - Return canonical field lists by problem dimension

Design principles:
  - Field sets are declarative and side-effect free
  - Tensor channel order is centralized here
  - Model architecture stays independent from field selection

Boundaries:
  - Field-name constants belong to domain.fields
  - Permeability component mappings belong to domain.permeability
===============================================================================
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Default training inputs
# ---------------------------------------------------------------------
# These are INTERNAL, CANONICAL field names
# (after domain.permeability mapping and batch dataset construction)
# ---------------------------------------------------------------------

DEFAULT_INPUTS_2D = [
    "x",
    "y",
    "kxx",
    "kyy",
    "kxy",
    "phi",
    "p_bc",
]

DEFAULT_INPUTS_3D = [
    "x",
    "y",
    "z",
    "kxx",
    "kyy",
    "kzz",
    "kxy",
    "kxz",
    "kyz",
    "phi",
    "p_bc",
]

# ---------------------------------------------------------------------
# Default training outputs
# ---------------------------------------------------------------------

DEFAULT_OUTPUTS_2D = [
    "p",
    "u",
    "v",
]

DEFAULT_OUTPUTS_3D = [
    "p",
    "u",
    "v",
    "w",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def default_training_inputs(dim: int) -> list[str]:
    """
    Return default input field names for the given problem dimension.

    Parameters
    ----------
    dim : int
        Problem dimension (2 or 3).

    Returns
    -------
    list[str]
        Canonical input field names for training.

    """
    if dim == 2:  # noqa: PLR2004
        return DEFAULT_INPUTS_2D
    if dim == 3:  # noqa: PLR2004
        return DEFAULT_INPUTS_3D
    msg = f"Unsupported dimension: {dim}"
    raise ValueError(msg)


def default_training_outputs(dim: int) -> list[str]:
    """
    Return default output field names for the given problem dimension.

    Parameters
    ----------
    dim : int
        Problem dimension (2 or 3).

    Returns
    -------
    list[str]
        Canonical output field names for training.

    """
    if dim == 2:  # noqa: PLR2004
        return DEFAULT_OUTPUTS_2D
    if dim == 3:  # noqa: PLR2004
        return DEFAULT_OUTPUTS_3D
    msg = f"Unsupported dimension: {dim}"
    raise ValueError(msg)
