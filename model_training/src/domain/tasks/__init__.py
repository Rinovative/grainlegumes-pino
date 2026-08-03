"""
Registered scientific task specifications.

Provides:
- registry: task registration and lookup
- spec: immutable task-schema definitions
- steady_flow: maintained steady-flow task contract
"""

from . import domain_task_registry as registry
from . import domain_task_spec as spec
from . import domain_task_steady_flow as steady_flow

__all__ = [
    "registry",
    "spec",
    "steady_flow",
]
