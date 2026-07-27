"""
===============================================================================
domain_physics_boundary.py
===============================================================================
Provide reusable pressure-boundary masks, residuals, and diagnostics.

Responsibilities:
  - Identify inlet and outlet cells from explicit coordinates and grid spacing
  - Compute inlet pressure mismatch and zero-gauge outlet residuals
  - Expose boundary values and cancellation-safe per-sample diagnostics

Design principles:
  - Boundary masks use coordinate extrema within half a positive grid spacing
  - Spatial reductions occur per sample before aggregation across the batch
  - Tensor shape, dtype, and device follow the caller's physical fields

This module does NOT:
  - Bind task channel names or convert normalized tensors to physical units
  - Choose boundary-loss weights, warmup schedules, or logging names
  - Differentiate fields or evaluate interior momentum and continuity equations
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
    """
    Hold structured-grid inlet and outlet masks.

    Both frozen boolean tensors match the coordinate field shape. ``inlet``
    selects each sample's spatial y-min boundary and ``outlet`` selects y-max within half a
    validated grid spacing; no batch or spatial reduction is encoded.

    Attributes
    ----------
    inlet, outlet : torch.Tensor
        Boolean masks with the same sample/spatial layout as the coordinate
        field supplied to ``pressure_boundary_masks``.

    """

    inlet: Tensor
    outlet: Tensor


@dataclass(frozen=True, slots=True)
class PressureBoundaryResiduals:
    """
    Hold unreduced pressure-boundary values and per-sample diagnostics.

    ``inlet_error`` is ``p-p_bc`` flattened over selected inlet cells and
    ``outlet_pressure`` retains flattened outlet gauge values, both in Pa.
    Spatial inlet MSE and outlet mean are retained per sample before batch
    aggregation so opposite outlet gauges cannot cancel each other.

    Attributes
    ----------
    inlet_error, outlet_pressure : torch.Tensor
        One-dimensional selected physical boundary values in Pa.
    inlet_sample_mse : torch.Tensor
        Spatially reduced inlet pressure MSE per leading sample, in Pa².
    outlet_sample_mean : torch.Tensor
        Spatially reduced outlet gauge mean per leading sample, in Pa.
    masks : PressureBoundaryMasks
        Full-shape boolean masks used for the reductions.

    """

    inlet_error: Tensor
    outlet_pressure: Tensor
    inlet_sample_mse: Tensor
    outlet_sample_mean: Tensor
    masks: PressureBoundaryMasks

    @property
    def inlet_mse(self) -> Tensor:
        """Return the batch mean of per-sample inlet pressure MSE in Pa²."""
        return self.inlet_sample_mse.mean()

    @property
    def outlet_mean_square(self) -> Tensor:
        """Return the batch mean of squared per-sample outlet gauges in Pa²."""
        return self.outlet_sample_mean.square().mean()

    @property
    def mse(self) -> Tensor:
        """Return the inlet-plus-outlet pressure diagnostic in Pa²."""
        return self.inlet_mse + self.outlet_mean_square


def _normalized_axes(ndim: int, axes: SpatialAxes) -> SpatialAxes:
    """
    Normalize exactly two distinct spatial axes for a tensor rank.

    Negative axes are translated relative to ``ndim``. The returned order is
    preserved for downstream reductions; duplicates and out-of-range axes fail
    before any boundary mask is constructed.

    Raises
    ------
    ValueError
        If ``axes`` does not identify two distinct axes within the tensor rank.

    """
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
        Finite floating physical y-coordinate field, normally shaped
        ``[batch, y, x]`` and measured in the same length unit as ``spacing_y``.
    spacing_y : float or torch.Tensor
        Finite positive scalar grid spacing along y.
    spatial_axes : tuple[int, int], optional
        Axes spanning the structured spatial domain.

    Returns
    -------
    PressureBoundaryMasks
        Boolean masks matching ``y_coordinate`` in shape and device.

    Raises
    ------
    TypeError
        If ``y_coordinate`` is not floating point.
    ValueError
        If spatial axes are invalid, spacing is not one finite positive scalar,
        or either extremal boundary mask is empty.

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
        Predicted physical pressure field in Pa, normally ``[batch, y, x]``.
    prescribed_pressure : torch.Tensor
        Physical inlet-pressure field in Pa with the same shape.
    y_coordinate : torch.Tensor
        Physical y-coordinate field with the same shape.
    spacing_y : float or torch.Tensor
        Positive physical y-grid spacing.
    spatial_axes : tuple[int, int], optional
        Axes spanning the structured spatial domain.

    Returns
    -------
    PressureBoundaryResiduals
        Selected boundary values, per-sample reductions, and scalar batch-mean
        properties. ``mse`` is inlet MSE plus the mean squared per-sample outlet
        gauge; outlet means are squared before batch aggregation.

    Raises
    ------
    TypeError
        If the coordinate field is not floating point.
    ValueError
        If field shapes, axes, spacing, masks, or per-sample boundary membership
        violate the structured-grid boundary contract.

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
