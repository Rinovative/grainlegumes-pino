"""
===============================================================================
learning_device_policy.py
===============================================================================
Define the dependency-free runtime-device policy vocabulary.

Responsibilities:
  - Publish the exact auto, cuda, and cpu policy literals
  - Validate user-facing policy values with path-qualified errors
  - Keep parser construction and semantic config validation backend-free

Design principles:
  - Policy vocabulary is distinct from hardware resolution
  - Validation is exact, deterministic, and free of runtime probing
  - Torch is imported only by the concrete device resolver

This module does NOT:
  - Query CUDA or construct Torch devices; ``learning_device`` owns resolution
  - Parse CLI arguments or config files; their boundary modules own parsing
  - Persist runtime metadata or select mixed-precision behavior
===============================================================================
"""

from __future__ import annotations

from typing import Any, Literal, cast

DevicePolicy = Literal["auto", "cuda", "cpu"]
DEVICE_POLICIES: tuple[DevicePolicy, ...] = ("auto", "cuda", "cpu")


class DevicePolicyError(ValueError):
    """Represent an invalid user-facing runtime-device policy."""


def validate_device_policy(policy: Any, *, path: str = "device") -> DevicePolicy:
    """Validate and return one exact user-facing runtime-device policy."""
    if type(policy) is not str or policy not in DEVICE_POLICIES:
        allowed = ", ".join(DEVICE_POLICIES)
        msg = f"{path} must be exactly one of: {allowed}; got {policy!r}."
        raise DevicePolicyError(msg)
    return cast("DevicePolicy", policy)
