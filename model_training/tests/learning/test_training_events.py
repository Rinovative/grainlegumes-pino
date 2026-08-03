# ruff: noqa: S101
"""Protect the one canonical completed-epoch event predicate."""

from __future__ import annotations

import pytest
from src import learning


@pytest.mark.parametrize(
    ("interval", "target_epoch", "expected"),
    [
        (5, 10, (5, 10)),
        (5, 12, (5, 10, 12)),
        (5, 600, tuple(range(5, 601, 5))),
        (1, 3, (1, 2, 3)),
        (5, 3, (3,)),
    ],
)
def test_completed_epoch_events_are_terminal_inclusive_without_epoch_zero(
    interval: int,
    target_epoch: int,
    expected: tuple[int, ...],
) -> None:
    """Include regular cadence points and the genuine terminal epoch exactly once."""
    observed = learning.training.events.completed_epoch_events(
        interval=interval,
        target_epoch=target_epoch,
    )
    assert observed == expected
    assert 0 not in observed
    assert observed[-1] == target_epoch
    assert observed.count(target_epoch) == 1
    assert (
        learning.training.events.first_completed_epoch_event(
            interval=interval,
            target_epoch=target_epoch,
        )
        == expected[0]
    )


@pytest.mark.parametrize(
    ("completed_epoch", "interval", "target_epoch"),
    [
        (0, 1, 1),
        (True, 1, 1),
        (1, 0, 1),
        (1, True, 1),
        (1, 1, 0),
        (1, 1, True),
        (2, 1, 1),
    ],
)
def test_completed_epoch_event_rejects_invalid_or_out_of_range_epochs(
    completed_epoch: int,
    interval: int,
    target_epoch: int,
) -> None:
    """Reject epoch zero, boolean lookalikes, nonpositive cadence, and overshoot."""
    with pytest.raises(ValueError, match=r"must be|exceeds"):
        learning.training.events.is_completed_epoch_event(
            completed_epoch,
            interval=interval,
            target_epoch=target_epoch,
        )
