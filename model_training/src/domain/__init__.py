"""
Scientific field, permeability, physics, and task contracts.

Provides:
- field_sets: named input and output field collections
- fields: field definitions, units, and tensor semantics
- permeability: permeability representations and validation
- physics: boundary, derivative, and Brinkman operator services
- tasks: registered scientific task specifications
"""

from . import domain_field_sets as field_sets
from . import domain_fields as fields
from . import domain_permeability as permeability
from . import physics, tasks

__all__ = ["field_sets", "fields", "permeability", "physics", "tasks"]
