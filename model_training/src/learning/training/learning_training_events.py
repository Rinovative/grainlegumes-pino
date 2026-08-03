"""
Canonical completed-epoch event scheduling.

Every interval-based training phase uses :func:`is_completed_epoch_event`.
Normal histories begin only after a completed optimizer epoch, include the
terminal target exactly once, and never synthesize or backfill observations.
"""

from __future__ import annotations


def _positive_epoch(value: int, *, label: str) -> int:
    """Return one exact positive integer epoch value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{label} must be an integer >= 1, got {value!r}."
        raise ValueError(msg)
    return value


def is_completed_epoch_event(
    completed_epoch: int,
    *,
    interval: int,
    target_epoch: int,
) -> bool:
    """Return whether one genuine completed epoch owns an interval event."""
    completed = _positive_epoch(completed_epoch, label="completed_epoch")
    cadence = _positive_epoch(interval, label="interval")
    target = _positive_epoch(target_epoch, label="target_epoch")
    if completed > target:
        msg = f"completed_epoch {completed} exceeds target_epoch {target}."
        raise ValueError(msg)
    return completed % cadence == 0 or completed == target


def completed_epoch_events(*, interval: int, target_epoch: int) -> tuple[int, ...]:
    """Return the exact ordered event epochs for one target and cadence."""
    cadence = _positive_epoch(interval, label="interval")
    target = _positive_epoch(target_epoch, label="target_epoch")
    return tuple(epoch for epoch in range(1, target + 1) if is_completed_epoch_event(epoch, interval=cadence, target_epoch=target))


def first_completed_epoch_event(*, interval: int, target_epoch: int) -> int:
    """Return the first genuine event epoch under the canonical predicate."""
    return completed_epoch_events(interval=interval, target_epoch=target_epoch)[0]
