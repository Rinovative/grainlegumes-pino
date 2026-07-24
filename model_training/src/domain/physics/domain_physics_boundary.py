"""
===============================================================================
domain_physics_boundary.py
===============================================================================
Provide reusable pressure-boundary masks, residuals, and diagnostics.

Responsibilities:
  - Identify inlet and outlet cells from explicit coordinates and grid spacing
  - Compute inlet pressure mismatch and outlet pressure-gauge residuals
  - Expose unreduced residual values and well-defined scalar diagnostics

Boundaries:
  - Task field lookup belongs to task-specific physics adapters
  - Loss weights, warmup, normalization, and logging belong to learning
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .domain_physics_derivatives import SpatialAxes

_SPATIAL_AXIS_COUNT = 2


@dataclass(frozen=True, slots=True)
class PressureBoundaryMasks:
    """Hold boolean inlet and outlet masks for a structured grid."""

    inlet: Tensor
    outlet: Tensor


@dataclass(frozen=True, slots=True)
class PressureBoundaryResiduals:
    """Hold pressure boundary residual values and per-sample diagnostics."""

    inlet_error: Tensor
    outlet_pressure: Tensor
    inlet_sample_mse: Tensor
    outlet_sample_mean: Tensor
    masks: PressureBoundaryMasks

    @property
    def inlet_mse(self) -> Tensor:
        """Return the batch mean of per-sample inlet pressure MSE values."""
        return self.inlet_sample_mse.mean()

    @property
    def outlet_mean_square(self) -> Tensor:
        """Return the batch mean of squared per-sample outlet gauges."""
        return self.outlet_sample_mean.square().mean()

    @property
    def mse(self) -> Tensor:
        """Return the complete inlet-plus-outlet boundary diagnostic."""
        return self.inlet_mse + self.outlet_mean_square


def _normalized_axes(ndim: int, axes: SpatialAxes) -> SpatialAxes:
    """Return valid non-negative spatial axes."""
    normalized = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    if len(set(normalized)) != _SPATIAL_AXIS_COUNT or any(axis < 0 or axis >= ndim for axis in normalized):
        msg = f"Spatial axes {axes!r} are invalid for tensor rank {ndim}."
        raise ValueError(msg)
    return normalized[0], normalized[1]


def pressure_boundary_masks(
    y_coordinate: Tensor,
    spacing_y: float | Tensor,
    *,
    spatial_axes: SpatialAxes = (-2, -1),
) -> PressureBoundaryMasks:
    """
    Build y-min inlet and y-max outlet masks.

    Parameters
    ----------
    y_coordinate : torch.Tensor
        Physical y-coordinate field.
    spacing_y : float or torch.Tensor
        Positive grid spacing along y.
    spatial_axes : tuple[int, int], optional
        Axes spanning the structured spatial domain.

    Returns
    -------
    PressureBoundaryMasks
        Boolean masks matching ``y_coordinate``.

    """
    if not y_coordinate.is_floating_point():
        msg = f"y_coordinate must use a floating dtype, got {y_coordinate.dtype}."
        raise TypeError(msg)
    axes = _normalized_axes(y_coordinate.ndim, spatial_axes)
    dy = torch.as_tensor(spacing_y, dtype=y_coordinate.dtype, device=y_coordinate.device)
    if dy.numel() != 1 or not bool(torch.isfinite(dy).item()) or not bool((dy > 0).item()):
        msg = "spacing_y must be a finite positive scalar."
        raise ValueError(msg)
    minimum = y_coordinate.amin(dim=axes, keepdim=True)
    maximum = y_coordinate.amax(dim=axes, keepdim=True)
    inlet = (y_coordinate - minimum).abs() <= 0.5 * dy
    outlet = (y_coordinate - maximum).abs() <= 0.5 * dy
    if not bool(inlet.any().item()) or not bool(outlet.any().item()):
        msg = "Pressure boundary masks must contain at least one inlet and outlet cell."
        raise ValueError(msg)
    return PressureBoundaryMasks(inlet=inlet, outlet=outlet)


def pressure_boundary_residuals(
    pressure: Tensor,
    prescribed_pressure: Tensor,
    y_coordinate: Tensor,
    spacing_y: float | Tensor,
    *,
    spatial_axes: SpatialAxes = (-2, -1),
) -> PressureBoundaryResiduals:
    """
    Compute pressure inlet and outlet-gauge residual values.

    Parameters
    ----------
    pressure : torch.Tensor
        Predicted physical pressure field.
    prescribed_pressure : torch.Tensor
        Physical pressure boundary field with the same shape.
    y_coordinate : torch.Tensor
        Physical y-coordinate field with the same shape.
    spacing_y : float or torch.Tensor
        Positive physical y-grid spacing.
    spatial_axes : tuple[int, int], optional
        Axes spanning the structured spatial domain.

    Returns
    -------
    PressureBoundaryResiduals
        Unreduced boundary values plus scalar MSE properties.

    """
    if pressure.shape != prescribed_pressure.shape or pressure.shape != y_coordinate.shape:
        msg = (
            "Pressure, prescribed pressure, and y-coordinate shapes must match; "
            f"got {tuple(pressure.shape)}, {tuple(prescribed_pressure.shape)}, and {tuple(y_coordinate.shape)}."
        )
        raise ValueError(msg)
    masks = pressure_boundary_masks(
        y_coordinate,
        spacing_y,
        spatial_axes=spatial_axes,
    )
    axes = _normalized_axes(pressure.ndim, spatial_axes)
    inlet_difference = pressure - prescribed_pressure
    inlet_count = masks.inlet.sum(dim=axes)
    outlet_count = masks.outlet.sum(dim=axes)
    if bool((inlet_count == 0).any().item()) or bool((outlet_count == 0).any().item()):
        msg = "Every sample must contain at least one inlet and outlet cell."
        raise ValueError(msg)
    inlet_sample_mse = (inlet_difference.square() * masks.inlet).sum(dim=axes) / inlet_count
    outlet_sample_mean = (pressure * masks.outlet).sum(dim=axes) / outlet_count
    return PressureBoundaryResiduals(
        inlet_error=inlet_difference[masks.inlet],
        outlet_pressure=pressure[masks.outlet],
        inlet_sample_mse=inlet_sample_mse,
        outlet_sample_mean=outlet_sample_mean,
        masks=masks,
    )
