"""
Domain contracts for fields, permeability and physics.

Provides:
- field_sets: canonical model input and output field sets
- fields: field-name constants and semantic groups
- permeability: permeability tensor naming and ordering
- physics: physics helper modules
"""

from . import domain_field_sets as field_sets
from . import domain_fields as fields
from . import domain_permeability as permeability
from . import physics

__all__ = [
    "field_sets",
    "fields",
    "permeability",
    "physics",
]
