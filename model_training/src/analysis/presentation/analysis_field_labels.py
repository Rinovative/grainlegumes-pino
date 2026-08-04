"""Canonical user-facing labels for scientific analysis fields."""

from __future__ import annotations

from typing import Final

VELOCITY_MAGNITUDE_KEY: Final = "U"
VELOCITY_MAGNITUDE_LABEL: Final = "|u|"
VELOCITY_MAGNITUDE_MATHTEXT: Final = r"$|\mathbf{u}|$"


def field_label(field: str, *, mathtext: bool = False) -> str:
    """Return the canonical plain or Matplotlib label for one field key."""
    if field == VELOCITY_MAGNITUDE_KEY:
        return VELOCITY_MAGNITUDE_MATHTEXT if mathtext else VELOCITY_MAGNITUDE_LABEL
    return field
