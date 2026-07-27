"""
Domain contracts for tasks, fields, permeability, and physics.

Provides:
- field_sets: strict ordered field-contract validation
- fields: reusable canonical field primitives
- permeability: permeability tensor naming and representation logic
- physics: derivative, boundary, and Brinkman residual contracts
- tasks: immutable task specifications and semantic registries
"""

from . import domain_field_sets as field_sets
from . import domain_fields as fields
from . import domain_permeability as permeability
from . import physics, tasks

__all__ = [
    "field_sets",
    "fields",
    "permeability",
    "physics",
    "tasks",
]
