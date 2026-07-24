"""
Task contracts and registries.

Provides:
- registry: strict semantic task and physics lookup
- spec: immutable field, metric, physics, and task descriptors
- steady_flow: the canonical steady-flow task contract
"""

from . import domain_task_registry as registry
from . import domain_task_spec as spec
from . import domain_task_steady_flow as steady_flow

__all__ = [
    "registry",
    "spec",
    "steady_flow",
]
