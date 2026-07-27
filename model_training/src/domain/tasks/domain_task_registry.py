"""
===============================================================================
domain_task_registry.py
===============================================================================
Register and resolve semantic task and task-selected physics identifiers.

Responsibilities:
  - Expose the exact set of registered task identifiers
  - Resolve immutable TaskSpec objects without aliases or fallbacks
  - Resolve task-owned physics selectors by semantic identifier

Design principles:
  - Registry lookup is exact, deterministic, and read-only
  - Error messages enumerate the available semantic identifiers
  - Registration does not depend on model or training implementations

This module does NOT:
  - Define task structure or concrete steady-flow field declarations
  - Translate aliases, import plugins, or mutate registrations at runtime
  - Implement physics equations, losses, models, or training behavior
===============================================================================
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .domain_task_spec import PhysicsSpec, TaskSpec
from .domain_task_steady_flow import STEADY_FLOW

_TASKS: Final = MappingProxyType({STEADY_FLOW.id: STEADY_FLOW})
_PHYSICS: Final = MappingProxyType({STEADY_FLOW.physics.kind: STEADY_FLOW.physics})


def available_tasks() -> tuple[str, ...]:
    """
    Return registered task identifiers in deterministic order.

    Returns
    -------
    tuple[str, ...]
        Exact semantic task identifiers accepted by the registry.

    """
    return tuple(sorted(_TASKS))


def get_task(task_id: str) -> TaskSpec:
    """
    Resolve an exact semantic task identifier.

    Parameters
    ----------
    task_id : str
        Canonical task identifier.

    Returns
    -------
    TaskSpec
        Immutable registered task contract.

    Raises
    ------
    ValueError
        If `task_id` is not registered.

    """
    try:
        return _TASKS[task_id]
    except KeyError as error:
        available = ", ".join(available_tasks())
        msg = f"Unknown task {task_id!r}. Available tasks: {available}."
        raise ValueError(msg) from error


def available_physics() -> tuple[str, ...]:
    """
    Return registered semantic physics identifiers.

    Returns
    -------
    tuple[str, ...]
        Exact physics selectors owned by registered tasks.

    """
    return tuple(sorted(_PHYSICS))


def resolve_physics(kind: str) -> PhysicsSpec:
    """
    Resolve a task-selected semantic physics identifier.

    Parameters
    ----------
    kind : str
        Canonical physics selector.

    Returns
    -------
    PhysicsSpec
        Immutable registered physics descriptor.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    try:
        return _PHYSICS[kind]
    except KeyError as error:
        available = ", ".join(available_physics())
        msg = f"Unknown physics identifier {kind!r}. Available physics: {available}."
        raise ValueError(msg) from error
