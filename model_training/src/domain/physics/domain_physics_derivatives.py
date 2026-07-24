"""
===============================================================================
domain_physics_derivatives.py
===============================================================================
Provide reusable spatial derivative operators for physical tensor fields.

Responsibilities:
  - Infer uniform-grid spacing from explicit coordinate tensors
  - Compute physical-space and FFT-based gradients and divergences
  - Apply explicit spectral extension, spatial-axis, and crop semantics

Design principles:
  - Operators are independent of task fields, normalizers, losses, and logging
  - Grid spacing and spatial axes are explicit at every numerical boundary
  - Caller dtype and device are preserved after stable internal FFT work
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional

DerivativeKind = Literal["physical", "spectral"]
SpectralExtension = Literal["none", "reflect"]
SpatialAxes = tuple[int, int]
_MIN_SPATIAL_POINTS = 2
_SPATIAL_AXIS_COUNT = 2
_DEFAULT_UNIFORM_TOLERANCE = 1e-5
_RECTILINEAR_EPSILON_FACTOR = 32.0


class DerivativeOperator(Protocol):
    """Define the reusable gradient/divergence interface used by equations."""

    @property
    def kind(self) -> DerivativeKind:
        """Return the semantic derivative identifier."""
        ...

    @property
    def extension(self) -> SpectralExtension:
        """Return the explicit boundary-extension identifier."""
        ...

    @property
    def axes(self) -> SpatialAxes:
        """Return the explicit ``(y, x)`` spatial axes."""
        ...

    def gradient(
        self,
        field: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(d/dx, d/dy)`` for one scalar field."""
        ...

    def divergence(
        self,
        field_x: Tensor,
        field_y: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> Tensor:
        """Return ``d(field_x)/dx + d(field_y)/dy``."""
        ...


def _normalized_axes(ndim: int, axes: SpatialAxes) -> SpatialAxes:
    """Return distinct non-negative spatial axes for a tensor rank."""
    normalized = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    if len(set(normalized)) != _SPATIAL_AXIS_COUNT or any(axis < 0 or axis >= ndim for axis in normalized):
        msg = f"Spatial axes {axes!r} are invalid for tensor rank {ndim}."
        raise ValueError(msg)
    return cast("SpatialAxes", normalized)


def _validate_field(field: Tensor, axes: SpatialAxes) -> SpatialAxes:
    """Validate a real floating field and its two spatial axes."""
    if not isinstance(field, Tensor):
        msg = f"Derivative fields must be torch.Tensor instances, got {type(field).__name__}."
        raise TypeError(msg)
    if not field.is_floating_point():
        msg = f"Derivative fields must use a floating dtype, got {field.dtype}."
        raise TypeError(msg)
    normalized = _normalized_axes(field.ndim, axes)
    if any(field.shape[axis] < _MIN_SPATIAL_POINTS for axis in normalized):
        msg = f"Derivative spatial axes require at least two points, got shape {tuple(field.shape)}."
        raise ValueError(msg)
    return normalized


def _spacing_tensor(spacing: float | Tensor, *, reference: Tensor, label: str) -> Tensor:
    """Return one positive finite scalar spacing on the reference device."""
    value = torch.as_tensor(spacing, dtype=reference.dtype, device=reference.device)
    if value.numel() != 1:
        msg = f"{label} must be scalar, got shape {tuple(value.shape)}."
        raise ValueError(msg)
    if not bool(torch.isfinite(value).item()) or not bool((value > 0).item()):
        msg = f"{label} must be finite and positive, got {float(value.detach().cpu().item())}."
        raise ValueError(msg)
    return value.reshape(())


def _spacing_float(spacing: float | Tensor, *, reference: Tensor, label: str) -> float:
    """Return one validated spacing as a Python float for ``torch.fft``."""
    value = _spacing_tensor(spacing, reference=reference, label=label)
    return float(value.detach().cpu().item())


def infer_uniform_spacing(
    x_coordinate: Tensor,
    y_coordinate: Tensor,
    *,
    axes: SpatialAxes = (-2, -1),
    uniform_tolerance: float = _DEFAULT_UNIFORM_TOLERANCE,
) -> tuple[Tensor, Tensor]:
    """
    Infer positive spacing from finite, increasing, uniform Cartesian grids.

    Parameters
    ----------
    x_coordinate, y_coordinate : torch.Tensor
        Rectilinear coordinate fields with matching shapes.
    axes : tuple[int, int], optional
        ``(y_axis, x_axis)`` in the coordinate tensors.
    uniform_tolerance : float, optional
        Maximum finite non-negative relative deviation from mean spacing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Scalar ``(dx, dy)`` tensors on the coordinate device.

    """
    y_axis, x_axis = _validate_field(x_coordinate, axes)
    _validate_field(y_coordinate, axes)
    if x_coordinate.shape != y_coordinate.shape:
        msg = f"Coordinate shapes must match, got {tuple(x_coordinate.shape)} and {tuple(y_coordinate.shape)}."
        raise ValueError(msg)
    if isinstance(uniform_tolerance, bool) or not isinstance(uniform_tolerance, (int, float)):
        msg = f"uniform_tolerance must be a real number, got {type(uniform_tolerance).__name__}."
        raise TypeError(msg)
    tolerance = float(uniform_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        msg = "uniform_tolerance must be finite and non-negative."
        raise ValueError(msg)
    if not bool(torch.isfinite(x_coordinate).all().item()) or not bool(torch.isfinite(y_coordinate).all().item()):
        msg = "Coordinate grids must contain only finite values."
        raise ValueError(msg)

    x_cross_differences = torch.diff(x_coordinate, dim=y_axis)
    y_cross_differences = torch.diff(y_coordinate, dim=x_axis)
    epsilon = torch.finfo(x_coordinate.dtype).eps
    x_cross_tolerance = _RECTILINEAR_EPSILON_FACTOR * epsilon * x_coordinate.abs().amax().clamp_min(1.0)
    y_cross_tolerance = _RECTILINEAR_EPSILON_FACTOR * epsilon * y_coordinate.abs().amax().clamp_min(1.0)
    if bool((x_cross_differences.abs().amax() > x_cross_tolerance).item()):
        msg = "x-coordinate must be constant along the y-axis on a Cartesian grid."
        raise ValueError(msg)
    if bool((y_cross_differences.abs().amax() > y_cross_tolerance).item()):
        msg = "y-coordinate must be constant along the x-axis on a Cartesian grid."
        raise ValueError(msg)

    x_differences = torch.diff(x_coordinate, dim=x_axis)
    y_differences = torch.diff(y_coordinate, dim=y_axis)
    if not bool((x_differences > 0).all().item()):
        msg = "x-coordinate must be strictly increasing along the x-axis."
        raise ValueError(msg)
    if not bool((y_differences > 0).all().item()):
        msg = "y-coordinate must be strictly increasing along the y-axis."
        raise ValueError(msg)
    dx = x_differences.mean()
    dy = y_differences.mean()
    for label, differences, mean in (("x", x_differences, dx), ("y", y_differences, dy)):
        relative_deviation = ((differences - mean).abs() / mean).amax()
        if float(relative_deviation.detach().cpu().item()) > tolerance:
            msg = f"{label}-coordinate spacing is not uniform within tolerance {tolerance}."
            raise ValueError(msg)
    return dx, dy


def crop_interior(field: Tensor, crop: int, *, axes: SpatialAxes = (-2, -1)) -> Tensor:
    """
    Crop an equal number of cells from every spatial boundary.

    Parameters
    ----------
    field : torch.Tensor
        Tensor containing the spatial axes.
    crop : int
        Non-negative number of cells removed from each side.
    axes : tuple[int, int], optional
        Spatial axes to crop.

    Returns
    -------
    torch.Tensor
        A view of the requested interior.

    """
    if isinstance(crop, bool) or not isinstance(crop, int):
        msg = f"crop must be an integer, got {type(crop).__name__}."
        raise TypeError(msg)
    if crop < 0:
        msg = f"crop must be non-negative, got {crop}."
        raise ValueError(msg)
    if crop == 0:
        return field
    normalized = _normalized_axes(field.ndim, axes)
    if any(2 * crop >= field.shape[axis] for axis in normalized):
        msg = f"crop={crop} removes the complete spatial domain from shape {tuple(field.shape)}."
        raise ValueError(msg)
    slices = [slice(None)] * field.ndim
    for axis in normalized:
        slices[axis] = slice(crop, -crop)
    return field[tuple(slices)]


@dataclass(frozen=True, slots=True)
class PhysicalDerivatives:
    """Compute physical-space derivatives with ``torch.gradient``."""

    axes: SpatialAxes = (-2, -1)
    kind: DerivativeKind = "physical"
    extension: SpectralExtension = "none"

    def __post_init__(self) -> None:
        """Reject meaningless extension settings for physical derivatives."""
        if self.extension != "none":
            msg = "Physical derivatives require extension 'none'."
            raise ValueError(msg)

    def gradient(
        self,
        field: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the physical-space x/y gradient of ``field``."""
        y_axis, x_axis = _validate_field(field, self.axes)
        dx = _spacing_tensor(spacing_x, reference=field, label="spacing_x")
        dy = _spacing_tensor(spacing_y, reference=field, label="spacing_y")
        derivative_x = torch.gradient(field, dim=x_axis)[0] / dx
        derivative_y = torch.gradient(field, dim=y_axis)[0] / dy
        return derivative_x, derivative_y

    def divergence(
        self,
        field_x: Tensor,
        field_y: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> Tensor:
        """Return the physical-space divergence of a vector field."""
        if field_x.shape != field_y.shape:
            msg = f"Vector component shapes must match, got {tuple(field_x.shape)} and {tuple(field_y.shape)}."
            raise ValueError(msg)
        derivative_x, _ = self.gradient(field_x, spacing_x, spacing_y)
        _, derivative_y = self.gradient(field_y, spacing_x, spacing_y)
        return derivative_x + derivative_y


@dataclass(frozen=True, slots=True)
class SpectralDerivatives:
    """Compute FFT derivatives with explicit periodic or reflected extension."""

    extension: SpectralExtension = "reflect"
    axes: SpatialAxes = (-2, -1)
    kind: DerivativeKind = "spectral"

    def __post_init__(self) -> None:
        """Validate the spectral extension identifier."""
        if self.extension not in {"none", "reflect"}:
            msg = f"Unknown spectral extension {self.extension!r}; expected 'none' or 'reflect'."
            raise ValueError(msg)

    @staticmethod
    def _gradient_last_axes(field: Tensor, dx: float, dy: float) -> tuple[Tensor, Tensor]:
        """Return a periodic FFT gradient for fields with trailing y/x axes."""
        height, width = field.shape[-2:]
        work_dtype = torch.float32 if field.dtype in {torch.float16, torch.bfloat16} else field.dtype
        work = field.to(work_dtype)
        frequencies_x = 2.0 * torch.pi * torch.fft.rfftfreq(width, d=dx, device=field.device, dtype=work_dtype)
        frequencies_y = 2.0 * torch.pi * torch.fft.fftfreq(height, d=dy, device=field.device, dtype=work_dtype)
        transformed = torch.fft.rfft2(work, dim=(-2, -1))
        shape_x = (1,) * (field.ndim - 1) + (frequencies_x.numel(),)
        shape_y = (1,) * (field.ndim - 2) + (frequencies_y.numel(), 1)
        derivative_x = torch.fft.irfft2(1j * frequencies_x.reshape(shape_x) * transformed, s=(height, width))
        derivative_y = torch.fft.irfft2(1j * frequencies_y.reshape(shape_y) * transformed, s=(height, width))
        return derivative_x.to(field.dtype), derivative_y.to(field.dtype)

    def gradient(
        self,
        field: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the spectral x/y gradient of ``field``."""
        normalized_axes = _validate_field(field, self.axes)
        dx = _spacing_float(spacing_x, reference=field, label="spacing_x")
        dy = _spacing_float(spacing_y, reference=field, label="spacing_y")
        moved = field.movedim(normalized_axes, (-2, -1))
        if self.extension == "none":
            derivative_x, derivative_y = self._gradient_last_axes(moved, dx, dy)
        else:
            height, width = moved.shape[-2:]
            padded = functional.pad(
                moved,
                (width - 1, width - 1, height - 1, height - 1),
                mode="reflect",
            )
            padded_x, padded_y = self._gradient_last_axes(padded, dx, dy)
            derivative_x = padded_x[..., height - 1 : 2 * height - 1, width - 1 : 2 * width - 1]
            derivative_y = padded_y[..., height - 1 : 2 * height - 1, width - 1 : 2 * width - 1]
        return (
            derivative_x.movedim((-2, -1), normalized_axes),
            derivative_y.movedim((-2, -1), normalized_axes),
        )

    def divergence(
        self,
        field_x: Tensor,
        field_y: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> Tensor:
        """Return the spectral divergence of a vector field."""
        if field_x.shape != field_y.shape:
            msg = f"Vector component shapes must match, got {tuple(field_x.shape)} and {tuple(field_y.shape)}."
            raise ValueError(msg)
        derivative_x, _ = self.gradient(field_x, spacing_x, spacing_y)
        _, derivative_y = self.gradient(field_y, spacing_x, spacing_y)
        return derivative_x + derivative_y


def available_derivative_kinds() -> tuple[str, ...]:
    """Return supported semantic derivative identifiers."""
    return ("physical", "spectral")


def build_derivative_operator(
    kind: str,
    *,
    extension: str,
    axes: SpatialAxes = (-2, -1),
) -> DerivativeOperator:
    """
    Build a derivative operator from semantic identifiers.

    Parameters
    ----------
    kind : str
        ``"physical"`` or ``"spectral"``.
    extension : str
        ``"none"`` or ``"reflect"``. Physical derivatives require ``"none"``.
    axes : tuple[int, int], optional
        ``(y_axis, x_axis)`` used by the operator.

    Returns
    -------
    DerivativeOperator
        Validated reusable derivative backend.

    """
    if kind == "physical":
        if extension != "none":
            msg = "Physical derivatives require extension 'none'."
            raise ValueError(msg)
        return PhysicalDerivatives(axes=axes)
    if kind == "spectral":
        if extension not in {"none", "reflect"}:
            msg = f"Unknown derivative extension {extension!r}; expected 'none' or 'reflect'."
            raise ValueError(msg)
        return SpectralDerivatives(extension=cast("SpectralExtension", extension), axes=axes)
    available = ", ".join(available_derivative_kinds())
    msg = f"Unknown derivative identifier {kind!r}. Available derivatives: {available}."
    raise ValueError(msg)
