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
  - Grid spacing and spatial axes are explicit at every numerical boundary
  - Cartesian-grid admission fails closed before an equation is evaluated
  - Caller shape, dtype, and device are restored after internal FFT work

This module does NOT:
  - Bind task field names, transform normalized data, or select loss weights
  - Impose equation-specific interior crops or scalar residual reductions
  - Infer nonuniform, curvilinear, or unstructured-grid derivatives
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional

from . import domain_physics_contracts as physics_contracts

SpatialAxes = tuple[int, int]
_MIN_SPATIAL_POINTS = 2
_SPATIAL_AXIS_COUNT = 2
_DEFAULT_UNIFORM_TOLERANCE = 1e-5
_RECTILINEAR_EPSILON_FACTOR = 32.0
_UNIFORM_ROUNDOFF_FACTOR = 4.0


class DerivativeOperator(Protocol):
    """
    Define the reusable physical-field derivative interface used by equations.

    Implementations accept real floating tensors with explicit ``(y, x)`` axes
    and positive physical spacing. ``gradient`` returns ``(d/dx, d/dy)``;
    ``divergence`` returns ``d(field_x)/dx + d(field_y)/dy`` without cropping or
    scalar reduction.
    """

    @property
    def kind(self) -> physics_contracts.DerivativeKind:
        """Return the semantic derivative identifier."""
        ...

    @property
    def extension(self) -> physics_contracts.SpectralExtension:
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
    """
    Normalize the declared ``(y, x)`` axes for a tensor rank.

    Negative indices are translated relative to ``ndim`` while order is
    preserved. Duplicate or out-of-range axes are rejected before tensor
    movement, differentiation, or cropping.

    Raises
    ------
    ValueError
        If the declaration does not resolve to two distinct in-range axes.

    """
    normalized = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    if len(set(normalized)) != _SPATIAL_AXIS_COUNT or any(axis < 0 or axis >= ndim for axis in normalized):
        msg = f"Spatial axes {axes!r} are invalid for tensor rank {ndim}."
        raise ValueError(msg)
    return cast("SpatialAxes", normalized)


def _validate_field(field: Tensor, axes: SpatialAxes) -> SpatialAxes:
    """
    Validate a differentiable real field and return normalized spatial axes.

    The function does not copy data. Each declared spatial dimension must have
    at least two points, which is the minimum shared by the physical stencil and
    reflected FFT extension.

    Raises
    ------
    TypeError
        If ``field`` is not a floating-point ``torch.Tensor``.
    ValueError
        If the axes are invalid or either spatial dimension is shorter than two.

    """
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
    """
    Convert one physical spacing to a scalar on the reference dtype and device.

    Raises
    ------
    ValueError
        If conversion does not produce exactly one finite, strictly positive
        value.

    """
    value = torch.as_tensor(spacing, dtype=reference.dtype, device=reference.device)
    if value.numel() != 1:
        msg = f"{label} must be scalar, got shape {tuple(value.shape)}."
        raise ValueError(msg)
    if not bool(torch.isfinite(value).item()) or not bool((value > 0).item()):
        msg = f"{label} must be finite and positive, got {float(value.detach().cpu().item())}."
        raise ValueError(msg)
    return value.reshape(())


def _spacing_float(spacing: float | Tensor, *, reference: Tensor, label: str) -> float:
    """
    Return validated physical spacing as the host scalar required by FFT grids.

    Validation and dtype conversion first follow ``reference``; transferring the
    scalar to the host does not move or copy the differentiated field.
    """
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
        Maximum finite non-negative scientific relative deviation from mean
        spacing, applied together with a dtype- and coordinate-scale-aware
        floating-point roundoff allowance.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean scalar ``(dx, dy)`` tensors on the corresponding coordinate dtype
        and device.

    Raises
    ------
    TypeError
        If coordinates are not floating tensors or the tolerance is not a real
        non-boolean number.
    ValueError
        If shapes/axes differ, values are non-finite, the grid is not increasing
        rectilinear Cartesian data, or spacing exceeds ``uniform_tolerance``.

    Notes
    -----
    Cross-axis coordinate variation is admitted only at a small dtype-scaled
    roundoff tolerance. Along each varying axis, strict monotonicity remains
    mandatory. Uniformity admits the configured relative deviation plus four
    source-dtype epsilons at the coordinate scale, which covers subtraction of
    independently rounded coordinate endpoints and mean-spacing reduction
    without treating materially nonuniform grids as Cartesian.

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

    # Validate in float64 so the admission decision measures quantization in the
    # persisted source coordinates rather than adding float32 reduction error.
    # The source dtype still determines the permitted roundoff allowance.
    x_work = x_coordinate.to(torch.float64)
    y_work = y_coordinate.to(torch.float64)
    x_differences = torch.diff(x_work, dim=x_axis)
    y_differences = torch.diff(y_work, dim=y_axis)
    if not bool((x_differences > 0).all().item()):
        msg = "x-coordinate must be strictly increasing along the x-axis."
        raise ValueError(msg)
    if not bool((y_differences > 0).all().item()):
        msg = "y-coordinate must be strictly increasing along the y-axis."
        raise ValueError(msg)
    dx_work = x_differences.mean()
    dy_work = y_differences.mean()
    for label, coordinate, differences, mean in (
        ("x", x_coordinate, x_differences, dx_work),
        ("y", y_coordinate, y_differences, dy_work),
    ):
        source_epsilon = torch.finfo(coordinate.dtype).eps
        coordinate_scale = coordinate.detach().to(torch.float64).abs().amax().clamp_min(mean.abs())
        roundoff_allowance = _UNIFORM_ROUNDOFF_FACTOR * source_epsilon * coordinate_scale
        relative_allowance = tolerance * mean.abs()
        maximum_deviation = (differences - mean).abs().amax()
        allowed_deviation = relative_allowance + roundoff_allowance
        if bool((maximum_deviation > allowed_deviation).item()):
            msg = (
                f"{label}-coordinate spacing is not uniform: maximum absolute deviation "
                f"{float(maximum_deviation.detach().cpu().item()):.12g} exceeds the combined "
                f"relative ({tolerance:.12g}) and {coordinate.dtype} roundoff allowance "
                f"{float(allowed_deviation.detach().cpu().item()):.12g}."
            )
            raise ValueError(msg)
    return dx_work.to(dtype=x_coordinate.dtype), dy_work.to(dtype=y_coordinate.dtype)


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
        A view of the requested interior; ``crop=0`` returns ``field`` itself.

    Raises
    ------
    TypeError
        If ``crop`` is not an integer or is a boolean.
    ValueError
        If ``crop`` is negative, axes are invalid, or cropping would remove a
        complete spatial dimension.

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
    """
    Compute finite-difference physical-space derivatives with ``torch.gradient``.

    The frozen operator preserves input shape, dtype, and device, uses explicit
    x/y spacing, and supports no boundary extension. Grid admission and interior
    cropping remain caller responsibilities.

    Attributes
    ----------
    axes : tuple[int, int]
        Declared ``(y, x)`` axes, normalized and validated when an operation runs.
    kind : Literal["physical", "spectral"]
        Semantic metadata identifier; the canonical factory value is
        ``"physical"``.
    extension : Literal["none", "reflect"]
        Must be ``"none"`` for this implementation.

    Raises
    ------
    ValueError
        If constructed with an extension other than ``"none"``.

    """

    axes: SpatialAxes = (-2, -1)
    kind: physics_contracts.DerivativeKind = "physical"
    extension: physics_contracts.SpectralExtension = "none"

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
        """
        Return the physical-space gradient in ``(d/dx, d/dy)`` order.

        Parameters
        ----------
        field : torch.Tensor
            Real floating scalar field with at least two points on each declared
            spatial axis and arbitrary leading dimensions.
        spacing_x, spacing_y : float or torch.Tensor
            Finite positive scalar physical spacings.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(d(field)/dx, d(field)/dy)`` with the input shape, dtype, and
            device. Derivative units are field units per spacing unit.

        Raises
        ------
        TypeError
            If ``field`` is not a real floating tensor.
        ValueError
            If axes, spatial lengths, or either spacing violate the operator
            contract.

        Notes
        -----
        ``torch.gradient`` supplies centered interior differences and its
        supported one-sided edge stencil; no boundary cells are cropped.

        """
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
        """
        Return the physical divergence ``d(field_x)/dx + d(field_y)/dy``.

        Parameters
        ----------
        field_x, field_y : torch.Tensor
            Real floating vector components with identical shapes and arbitrary
            leading dimensions.
        spacing_x, spacing_y : float or torch.Tensor
            Finite positive scalar physical spacings shared by both components.

        Returns
        -------
        torch.Tensor
            Divergence with the component shape, dtype, and device; units are
            component units per spacing unit when both components share units.

        Raises
        ------
        TypeError
            If either component is not a real floating tensor.
        ValueError
            If component shapes, axes, spatial lengths, or spacings are invalid.

        """
        if field_x.shape != field_y.shape:
            msg = f"Vector component shapes must match, got {tuple(field_x.shape)} and {tuple(field_y.shape)}."
            raise ValueError(msg)
        derivative_x, _ = self.gradient(field_x, spacing_x, spacing_y)
        _, derivative_y = self.gradient(field_y, spacing_x, spacing_y)
        return derivative_x + derivative_y


@dataclass(frozen=True, slots=True)
class SpectralDerivatives:
    """
    Compute FFT derivatives with explicit periodic or reflected extension.

    Frequencies are scaled by physical spacing. Reflected extension mirrors both
    spatial axes before differentiation and crops back to the original domain;
    stable internal FFT precision is cast back to the caller dtype/device.

    Attributes
    ----------
    extension : Literal["none", "reflect"]
        ``"none"`` assumes periodic input; ``"reflect"`` mirrors both axes.
    axes : tuple[int, int]
        Declared ``(y, x)`` axes, normalized and validated per operation.
    kind : Literal["physical", "spectral"]
        Semantic metadata identifier; the canonical factory value is
        ``"spectral"``.

    Raises
    ------
    ValueError
        If ``extension`` is not ``"none"`` or ``"reflect"``.

    """

    extension: physics_contracts.SpectralExtension = "reflect"
    axes: SpatialAxes = (-2, -1)
    kind: physics_contracts.DerivativeKind = "spectral"

    def __post_init__(self) -> None:
        """Validate the spectral extension identifier."""
        if self.extension not in {"none", "reflect"}:
            msg = f"Unknown spectral extension {self.extension!r}; expected 'none' or 'reflect'."
            raise ValueError(msg)

    @staticmethod
    def _gradient_last_axes(field: Tensor, dx: float, dy: float) -> tuple[Tensor, Tensor]:
        """
        Differentiate trailing periodic y/x axes with physically scaled FFT modes.

        Parameters
        ----------
        field : torch.Tensor
            Validated real field whose last two axes are y and x.
        dx, dy : float
            Positive physical spacings used to build angular wavenumbers.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Periodic ``(d/dx, d/dy)`` fields with the original shape and dtype.

        Notes
        -----
        Half and bfloat inputs use float32 FFT work arrays. Multiplication by
        ``1j*k`` occurs in frequency space and ``irfft2`` restores real fields.

        """
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
        """
        Return the spectral gradient in ``(d/dx, d/dy)`` order.

        Parameters
        ----------
        field : torch.Tensor
            Real floating scalar field with arbitrary leading dimensions.
        spacing_x, spacing_y : float or torch.Tensor
            Finite positive scalar physical spacings.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Physically scaled derivatives with the input shape, dtype, and device.

        Raises
        ------
        TypeError
            If ``field`` is not a real floating tensor.
        ValueError
            If axes, spatial lengths, or either spacing violate the spectral
            operator contract.

        Notes
        -----
        Arbitrary spatial axes move to trailing y/x positions internally and are
        restored on return. ``extension="none"`` assumes periodic data;
        ``"reflect"`` mirrors both axes and crops back to the original domain.

        """
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
        """
        Return the FFT divergence ``d(field_x)/dx + d(field_y)/dy``.

        Parameters
        ----------
        field_x, field_y : torch.Tensor
            Real floating vector components with identical shapes.
        spacing_x, spacing_y : float or torch.Tensor
            Finite positive scalar physical spacings shared by both components.

        Returns
        -------
        torch.Tensor
            Divergence with the component shape, dtype, and device.

        Raises
        ------
        TypeError
            If either component is not a real floating tensor.
        ValueError
            If component shapes, axes, spatial lengths, or spacings are invalid.

        """
        if field_x.shape != field_y.shape:
            msg = f"Vector component shapes must match, got {tuple(field_x.shape)} and {tuple(field_y.shape)}."
            raise ValueError(msg)
        derivative_x, _ = self.gradient(field_x, spacing_x, spacing_y)
        _, derivative_y = self.gradient(field_y, spacing_x, spacing_y)
        return derivative_x + derivative_y


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
        Frozen reusable derivative backend with the requested canonical policy.

    Raises
    ------
    ValueError
        If ``kind`` or ``extension`` is unsupported, including reflected
        extension requested for physical differences.

    """
    resolved_kind, resolved_extension = physics_contracts.validate_derivative_kind(kind, extension=extension)
    if resolved_kind == "physical":
        return PhysicalDerivatives(axes=axes)
    return SpectralDerivatives(extension=resolved_extension, axes=axes)
