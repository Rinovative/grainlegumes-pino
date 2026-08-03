"""
===============================================================================
learning_device.py
===============================================================================
Resolve the exact runtime device policy shared by every execution service.

Responsibilities:
  - Consume the dependency-free auto, cuda, and cpu policy contract
  - Resolve one concrete indexed Torch device at a top-level service boundary
  - Fail strict CUDA requests without an availability-based CPU fallback
  - Return compact, immutable, serialization-safe runtime metadata

Design principles:
  - Requested policy and concrete runtime resolution remain distinct facts
  - Only this module decides whether auto selects CPU or CUDA
  - CPU resolution never queries CUDA availability or device properties
  - Physical CUDA indices are operational metadata, not scientific identity

This module does NOT:
  - Parse semantic config paths; ``experiments.config.loader`` owns config validation
  - Persist or forward runtime facts; top-level service boundaries own publication
  - Construct or execute models, losses, metrics, checkpoints, or training loops
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from . import learning_device_policy as _policy

__all__ = [
    "DeviceResolution",
    "DeviceResolutionError",
    "DeviceRuntimeMetadata",
    "resolve_device",
    "validate_mixed_precision_device",
]


class DeviceResolutionError(RuntimeError):
    """
    Represent failure to satisfy a valid runtime-device requirement.

    This covers unusable strict CUDA and the CUDA-only mixed-precision boundary.
    Unlike ``DevicePolicyError``, the requested vocabulary was already valid.
    """


@dataclass(frozen=True, slots=True)
class DeviceRuntimeMetadata:
    """
    Store compact safe facts about one concrete runtime device.

    Parameters
    ----------
    requested_policy : {"auto", "cuda", "cpu"}
        Exact user-requested policy before runtime resolution.
    resolved_device : str
        Concrete Torch device string, such as ``cpu`` or ``cuda:0``.
    device_type : str
        Concrete Torch device type.
    pytorch_version : str
        Active PyTorch version.
    cuda_index : int | None
        Concrete visible CUDA index when CUDA is selected.
    cuda_device_name : str | None
        CUDA device name when CUDA is selected.
    cuda_runtime_version : str | None
        PyTorch-reported CUDA runtime version when CUDA is selected.

    """

    requested_policy: _policy.DevicePolicy
    resolved_device: str
    device_type: str
    pytorch_version: str
    cuda_index: int | None = None
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None

    def as_dict(self) -> dict[str, str | int]:
        """
        Return JSON-compatible safe runtime metadata.

        Returns
        -------
        dict[str, str | int]
            Requested and resolved device facts. CUDA-only keys are omitted for
            CPU resolution rather than fabricated.

        """
        payload: dict[str, str | int] = {
            "requested_policy": self.requested_policy,
            "resolved_device": self.resolved_device,
            "device_type": self.device_type,
            "pytorch_version": self.pytorch_version,
        }
        if self.cuda_index is not None:
            payload["cuda_index"] = self.cuda_index
        if self.cuda_device_name is not None:
            payload["cuda_device_name"] = self.cuda_device_name
        if self.cuda_runtime_version is not None:
            payload["cuda_runtime_version"] = self.cuda_runtime_version
        return payload


@dataclass(frozen=True, slots=True)
class DeviceResolution:
    """
    Describe one immutable requested-versus-resolved runtime device decision.

    Parameters
    ----------
    requested_policy : {"auto", "cuda", "cpu"}
        Exact policy supplied by configuration or a CLI override.
    device : torch.device
        Concrete resolved Torch device.
    device_type : str
        Concrete device type, currently ``cpu`` or ``cuda``.
    cuda_index : int | None
        Concrete visible CUDA index when applicable.
    metadata : DeviceRuntimeMetadata
        Compact safe runtime facts suitable for session persistence.

    """

    requested_policy: _policy.DevicePolicy
    device: torch.device
    device_type: str
    cuda_index: int | None
    metadata: DeviceRuntimeMetadata

    def as_dict(self) -> dict[str, str | int]:
        """
        Return the serialization-compatible runtime record.

        Returns
        -------
        dict[str, str | int]
            Compact requested and resolved device metadata.

        """
        return self.metadata.as_dict()


def _cpu_resolution(policy: _policy.DevicePolicy) -> DeviceResolution:
    """
    Construct a concrete CPU decision without touching the CUDA runtime API.

    The original requested policy is retained, allowing both explicit ``cpu``
    and ``auto`` fallback to serialize truthfully while CUDA-only metadata stays
    absent rather than being fabricated.
    """
    device = torch.device("cpu")
    metadata = DeviceRuntimeMetadata(
        requested_policy=policy,
        resolved_device=str(device),
        device_type=device.type,
        pytorch_version=str(torch.__version__),
    )
    return DeviceResolution(
        requested_policy=policy,
        device=device,
        device_type=device.type,
        cuda_index=None,
        metadata=metadata,
    )


def _cuda_resolution(policy: _policy.DevicePolicy, *, path: str) -> DeviceResolution:
    """
    Construct an indexed CUDA decision or fail without device substitution.

    The active visible device index and name are queried together. Initialization
    errors or invalid metadata become ``DeviceResolutionError``; this helper
    never chooses CPU, including when the originating policy was ``auto``.
    """
    try:
        index = int(torch.cuda.current_device())
        device_name = str(torch.cuda.get_device_name(index))
    except Exception as error:
        msg = f"{path} requested {policy!r}, but no usable CUDA device could be initialized: {error}"
        raise DeviceResolutionError(msg) from error
    if index < 0 or not device_name:
        msg = f"{path} requested {policy!r}, but CUDA returned invalid device metadata."
        raise DeviceResolutionError(msg)
    device = torch.device("cuda", index)
    runtime_version = torch.version.cuda
    metadata = DeviceRuntimeMetadata(
        requested_policy=policy,
        resolved_device=str(device),
        device_type=device.type,
        pytorch_version=str(torch.__version__),
        cuda_index=index,
        cuda_device_name=device_name,
        cuda_runtime_version=None if runtime_version is None else str(runtime_version),
    )
    return DeviceResolution(
        requested_policy=policy,
        device=device,
        device_type=device.type,
        cuda_index=index,
        metadata=metadata,
    )


def resolve_device(policy: Any, *, path: str = "device") -> DeviceResolution:
    """
    Resolve one canonical runtime device policy exactly once.

    ``auto`` selects a usable CUDA device when available and otherwise CPU.
    ``cuda`` is strict and raises before callers allocate authoritative or
    expensive runtime state. ``cpu`` returns immediately without querying CUDA.

    Parameters
    ----------
    policy : Any
        Exact requested policy.
    path : str, optional
        Semantic source path included in errors.

    Returns
    -------
    DeviceResolution
        Immutable requested policy, concrete device, and safe runtime metadata.

    Raises
    ------
    DevicePolicyError
        If the policy is outside the exact vocabulary.
    DeviceResolutionError
        If strict CUDA is unavailable or unusable.

    """
    requested = _policy.validate_device_policy(policy, path=path)
    if requested == "cpu":
        return _cpu_resolution(requested)

    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:
        if requested == "cuda":
            msg = f"{path} requested 'cuda', but CUDA availability could not be established; strict CUDA never falls back to CPU: {error}"
            raise DeviceResolutionError(msg) from error
        return _cpu_resolution(requested)
    if not available:
        if requested == "cuda":
            msg = f"{path} requested 'cuda', but CUDA is unavailable; strict CUDA never falls back to CPU."
            raise DeviceResolutionError(msg)
        return _cpu_resolution(requested)

    try:
        return _cuda_resolution(requested, path=path)
    except DeviceResolutionError:
        if requested == "cuda":
            raise
        return _cpu_resolution(requested)


def validate_mixed_precision_device(
    enabled: Any,
    resolution: DeviceResolution,
    *,
    path: str = "training.mixed_precision",
) -> None:
    """
    Require the current CUDA-only mixed-precision contract.

    Parameters
    ----------
    enabled : Any
        Resolved mixed-precision setting.
    resolution : DeviceResolution
        Concrete runtime device decision.
    path : str, optional
        Semantic config path included in errors.

    Raises
    ------
    TypeError
        If the setting is not a boolean.
    DeviceResolutionError
        If mixed precision is requested for a non-CUDA resolution.

    """
    if type(enabled) is not bool:
        msg = f"{path} must be boolean, got {enabled!r}."
        raise TypeError(msg)
    if enabled and resolution.device_type != "cuda":
        msg = f"{path}=true requires a resolved CUDA device; CPU autocast is not supported by this runtime contract."
        raise DeviceResolutionError(msg)
