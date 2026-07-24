"""
===============================================================================
domain_fields.py
===============================================================================
Define reusable canonical field names and semantic field groups.

Responsibilities:
  - Declare coordinate, permeability, boundary and state field primitives
  - Group fields by physical role
  - Validate exact canonical names without aliases

Design principles:
  - Field names are declarative constants
  - Semantic groups avoid duplicated string lists
  - Naming stays independent of model architecture

Boundaries:
  - Task-specific tensor field order belongs to domain.tasks
  - Permeability tensor representation belongs to domain.permeability

Notes:
  - These groups are reusable vocabulary, not a learned tensor contract
  - The only authoritative learned ordering belongs to a registered TaskSpec

===============================================================================

"""

from __future__ import annotations

COORDINATE_FIELDS = ("x", "y")
PERMEABILITY_FIELDS = ("kxx", "kxy", "kyy")
POROSITY_FIELDS = ("eps",)
BOUNDARY_FIELDS = ("p_bc",)
STATE_FIELDS = ("p", "u", "v")
DERIVED_ANALYSIS_FIELDS = ("U",)
ANALYSIS_FIELDS = (*STATE_FIELDS, *DERIVED_ANALYSIS_FIELDS)
KNOWN_FIELDS = frozenset(
    (
        *COORDINATE_FIELDS,
        *PERMEABILITY_FIELDS,
        *POROSITY_FIELDS,
        *BOUNDARY_FIELDS,
        *STATE_FIELDS,
    )
)


def require_known_field(name: str) -> str:
    """
    Validate and return one canonical field name.

    Parameters
    ----------
    name : str
        Candidate machine-readable field name.

    Returns
    -------
    str
        The unchanged canonical field name.

    Raises
    ------
    ValueError
        If `name` is not part of the reusable canonical field vocabulary.

    """
    if name not in KNOWN_FIELDS:
        available = ", ".join(sorted(KNOWN_FIELDS))
        msg = f"Unknown field {name!r}. Available canonical fields: {available}."
        raise ValueError(msg)
    return name
