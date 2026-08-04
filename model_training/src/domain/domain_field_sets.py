"""
===============================================================================
domain_field_sets.py
===============================================================================
Validate exact ordered field contracts.

Responsibilities:
  - Reject duplicate, missing, unexpected, or misordered fields
  - Compare producer/consumer declarations with a task-owned sequence

Design principles:
  - Validation is strict, order-sensitive, and side-effect free
  - Task specifications are the only source of tensor channel order

This module does NOT:
  - Define canonical field names or complete task contracts
  - Reorder, deduplicate, alias, or coerce a producer declaration
  - Inspect tensor values, shapes, storage, or normalization state
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def validate_ordered_fields(
    actual: Sequence[str],
    expected: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """
    Validate one exact, duplicate-free ordered field declaration.

    Parameters
    ----------
    actual : Sequence[str]
        Field order declared by a producer or consumer.
    expected : Sequence[str]
        Authoritative field order from the task contract.
    label : str
        Human-readable path or contract label used in errors.

    Returns
    -------
    tuple[str, ...]
        The validated `actual` declaration as an immutable tuple.

    Raises
    ------
    ValueError
        If fields are duplicated, missing, unexpected, or misordered.

    """
    actual_fields = tuple(actual)
    expected_fields = tuple(expected)

    duplicates = sorted({name for name in actual_fields if actual_fields.count(name) > 1})
    if duplicates:
        msg = f"{label} contains duplicate fields: {duplicates}."
        raise ValueError(msg)

    missing = [name for name in expected_fields if name not in actual_fields]
    unexpected = [name for name in actual_fields if name not in expected_fields]
    if missing or unexpected:
        msg = f"{label} does not match the task contract. Missing: {missing}. Unexpected: {unexpected}. Expected order: {list(expected_fields)}."
        raise ValueError(msg)

    if actual_fields != expected_fields:
        msg = f"{label} has the wrong channel order. Expected: {list(expected_fields)}. Received: {list(actual_fields)}."
        raise ValueError(msg)
    return actual_fields
