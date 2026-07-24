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
# Internal canonical representation primitives (symmetric, reduced).
# A TaskSpec decides which components become learned channels.
# 2D follows the steady-flow semantic order (xx, xy, yy).
# 3D groups components in upper-triangular row order.
# =============================================================================

INTERNAL_KAPPA_2D_ORDER: list[str] = [
    "kxx",
    "kxy",
    "kyy",
]

INTERNAL_KAPPA_3D_ORDER: list[str] = [
    "kxx",
    "kxy",
    "kxz",
    "kyy",
    "kyz",
    "kzz",
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


def resolve_internal_to_present_sources(
    available_fields: Iterable[str],
) -> dict[str, list[str]]:
    """
    Resolve internal permeability components to present COMSOL source fields.

    Parameters
    ----------
    available_fields : Iterable[str]
        COMSOL field names actually present in the dataset.

    Returns
    -------
    dict[str, list[str]]
        Internal component names mapped to their present source fields.

    """
    available = set(available_fields)
    has_z_component = any(name in available for name in ("kappazx", "kappazy", "kappaxz", "kappayz", "kappazz"))
    order = INTERNAL_KAPPA_3D_ORDER if has_z_component else INTERNAL_KAPPA_2D_ORDER

    return {
        internal: [source for source in INTERNAL_TO_COMSOL[internal] if source in available]
        for internal in order
        if any(source in available for source in INTERNAL_TO_COMSOL[internal])
    }
