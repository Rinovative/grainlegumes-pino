"""
===============================================================================
domain_permeability.py
===============================================================================
Define permeability tensor naming, ordering and mapping rules.

Responsibilities:
  - Declare COMSOL-exported permeability component names
  - Declare internal canonical permeability component names
  - Map source field names to internal tensor components
  - Detect permeability dimensionality from available names

Design principles:
  - Permeability contracts are declarative and deterministic
  - Numeric tensor conversion is left to dataset modules
  - Dimension detection is based only on field names

Boundaries:
  - Model input field sets belong to domain.field_sets
  - Physics residual calculations belong to learning.losses
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# =============================================================================
# COMSOL-exported full tensor components (3D, 9 components)
# Row-wise order:
#   [ kxx  kyx  kzx ]
#   [ kxy  kyy  kzy ]
#   [ kxz  kyz  kzz ]
# =============================================================================

COMSOL_KAPPA_3D_ORDER: list[str] = [
    "kappaxx",
    "kappayx",
    "kappazx",
    "kappaxy",
    "kappayy",
    "kappazy",
    "kappaxz",
    "kappayz",
    "kappazz",
]

# =============================================================================
# Internal canonical representation (symmetric, reduced)
# 2D: (xx, yy, xy)  where xy represents the symmetric shear component
# 3D: (xx, yy, zz, xy, xz, yz)
# =============================================================================

INTERNAL_KAPPA_2D_ORDER: list[str] = [
    "kxx",
    "kyy",
    "kxy",
]

INTERNAL_KAPPA_3D_ORDER: list[str] = [
    "kxx",
    "kyy",
    "kzz",
    "kxy",
    "kxz",
    "kyz",
]

# =============================================================================
# Mapping: internal channel -> acceptable COMSOL source fields
# The consumer decides HOW to combine if multiple are present:
#   - prefer first available
#   - or average symmetric pairs
# =============================================================================

INTERNAL_TO_COMSOL: dict[str, list[str]] = {
    "kxx": ["kappaxx"],
    "kyy": ["kappayy"],
    "kzz": ["kappazz"],
    "kxy": ["kappaxy", "kappayx"],
    "kxz": ["kappaxz", "kappazx"],
    "kyz": ["kappayz", "kappazy"],
}


# =============================================================================
# Helpers
# =============================================================================


def is_kappa_field(name: str) -> bool:
    """
    Decide if a given field name is a COMSOL kappa component.

    Parameters
    ----------
    name : str
        Field name to check.

    Returns
    -------
    bool
        True if the name corresponds to a kappa component.

    """
    return name.startswith("kappa")


def detect_comsol_kappa_fields(field_names: Iterable[str]) -> list[str]:
    """
    Detect the given field names, return those that are COMSOL kappa components.

    in COMSOL 3D order.
    filtered to those that are actually present.
    2D vs 3D is not determined here.

    Parameters
    ----------
    field_names : Iterable[str]
        List of available field names.

    Returns
    -------
    list[str]
        COMSOL kappa component names in 3D order, filtered to those present.

    """
    available = set(field_names)
    return [k for k in COMSOL_KAPPA_3D_ORDER if k in available]


def detect_dimension_from_kappa(comsol_kappa_fields: Iterable[str]) -> int:
    """
    Detect problem dimension (2D vs 3D) based on available COMSOL kappa fields.

    Parameters
    ----------
    comsol_kappa_fields : Iterable[str]
        List of available COMSOL kappa component names.

    Returns
    -------
    int
        2 for 2D, 3 for 3D.

    """
    s = set(comsol_kappa_fields)
    has_z = any(name in s for name in ("kappazx", "kappazy", "kappaxz", "kappayz", "kappazz"))
    return 3 if has_z else 2


def internal_kappa_order_for_fields(available_fields: Iterable[str]) -> list[str]:
    """
    Determine the internal kappa component order based on available COMSOL fields.

    Parameters
    ----------
    available_fields : Iterable[str]
        List of available COMSOL field names.

    Returns
    -------
    list[str]
        Internal kappa component names in correct order (2D or 3D).

    """
    comsol = detect_comsol_kappa_fields(available_fields)
    dim = detect_dimension_from_kappa(comsol)
    return INTERNAL_KAPPA_3D_ORDER if dim == 3 else INTERNAL_KAPPA_2D_ORDER  # noqa: PLR2004


def required_comsol_sources_for_internal(
    internal_name: str,
) -> list[str]:
    """
    Get the list of required COMSOL source fields for a given internal kappa component.

    Parameters
    ----------
    internal_name : str
        Internal kappa component name.

    Returns
    -------
    list[str]
        List of COMSOL source field names that can provide this component.

    Raises
    ------
    ValueError
        If the internal_name is unknown.

    """
    if internal_name not in INTERNAL_TO_COMSOL:
        msg = f"Unknown internal kappa component '{internal_name}'."
        raise ValueError(msg)
    return INTERNAL_TO_COMSOL[internal_name]


def resolve_internal_to_present_sources(
    available_fields: Iterable[str],
    nonzero_fields: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """
    Resolve internal kappa components to physically present COMSOL source fields.

    Parameters
    ----------
    available_fields : Iterable[str]
        List of available COMSOL field names.
    nonzero_fields : Iterable[str] or None
        Optional list of COMSOL field names that are known to be non-zero.
        If None, all available fields are considered non-zero.

    Returns
    -------
    dict[str, list[str]]
        Mapping from internal kappa component names to lists of present COMSOL source fields.

    Note:
    -----
    Only field names are inspected; numerical values are ignored.
    Decisions about physical activity (e.g. zero-valued fields)
    must be made by the data ingestion layer and passed explicitly
    via `nonzero_fields`.

    """
    available = set(available_fields)
    active = set(nonzero_fields) if nonzero_fields is not None else available

    # determine effective dimension from *active* fields
    comsol_active = [k for k in COMSOL_KAPPA_3D_ORDER if k in active]
    dim = detect_dimension_from_kappa(comsol_active)
    order = INTERNAL_KAPPA_3D_ORDER if dim == 3 else INTERNAL_KAPPA_2D_ORDER  # noqa: PLR2004

    out: dict[str, list[str]] = {}
    for internal in order:
        candidates = INTERNAL_TO_COMSOL[internal]
        present = [c for c in candidates if c in active]
        if present:
            out[internal] = present

    return out
