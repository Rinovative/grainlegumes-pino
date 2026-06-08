"""
===============================================================================
domain_fields.py
===============================================================================
Define canonical field names and semantic field groups.

Responsibilities:
  - Declare coordinate, permeability, boundary and output field names
  - Group fields by physical role
  - Provide stable names for datasets, losses and analysis

Design principles:
  - Field names are declarative constants
  - Semantic groups avoid duplicated string lists
  - Naming stays independent of model architecture

Boundaries:
  - Model tensor field order belongs to domain.field_sets
  - Permeability tensor ordering belongs to domain.permeability

Notes:
  - Coordinate fields are always present and define the spatial domain
  - Kappa fields are dynamic based on the number of permeability components
  - The canonical input order is: [coords, kappa fields, scalar inputs]
===============================================================================

"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Coordinate fields (always present)
# ---------------------------------------------------------------------
COORD_FIELDS = ["x", "y"]

# ---------------------------------------------------------------------
# Scalar field inputs (material properties, boundary conditions, etc.)
# All are represented as volume fields in COMSOL
# ---------------------------------------------------------------------
SCALAR_INPUT_FIELDS = {
    "eps": "int4(x,y)",  # porosity (material field)
    "p_bc": "int5(x,y)",  # pressure boundary condition (volume-encoded)
}
# ---------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------
OUTPUT_FIELDS = ["p", "u", "v", "U"]


# ---------------------------------------------------------------------
# Canonical input field order (without kappa!)
# kappa is inserted dynamically between coords and scalars
# ---------------------------------------------------------------------
def canonical_input_order(kappa_fields: list[str]) -> list[str]:
    """
    Get the canonical input field order, given the kappa component names.

    Parameters
    ----------
    kappa_fields : list[str]
        List of kappa component names to include.

    Returns
    -------
    list[str]
        Canonical input field order.

    """
    return [
        *COORD_FIELDS,
        *kappa_fields,
        *SCALAR_INPUT_FIELDS.keys(),
    ]
