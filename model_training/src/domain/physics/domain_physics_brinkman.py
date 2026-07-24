"""
===============================================================================
domain_physics_brinkman.py
===============================================================================
Provide reusable steady Darcy-Brinkman equations and physical diagnostics.

Responsibilities:
  - Compute deviatoric Brinkman momentum residuals
  - Compute conservative and plain continuity residuals
  - Bind the steady-flow task fields outside generic numerical kernels
  - Return reusable full-field and interior-cropped diagnostics

Design principles:
  - Numerical kernels accept named physical quantities, never channel indices
  - Semantic physics and continuity identifiers fail closed
  - Normalization, weights, warmup, and training logging remain outside domain
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import torch
from torch import Tensor

from .domain_physics_boundary import PressureBoundaryResiduals, pressure_boundary_residuals
from .domain_physics_derivatives import DerivativeOperator, SpatialAxes, crop_interior, infer_uniform_spacing

AIR_DYNAMIC_VISCOSITY = 1.8139e-5
ContinuityKind = Literal["div_eps_velocity", "div_velocity"]
STEADY_BRINKMAN_KIND = "steady_2d_brinkman"
PRESSURE_BOUNDARY_KIND = "pressure_inlet_zero_pressure_outlet"
_MIN_TASK_TENSOR_RANK = 3


@dataclass(frozen=True, slots=True)
class MomentumResiduals:
    """Hold x- and y-momentum residual fields."""

    x: Tensor
    y: Tensor


@dataclass(frozen=True, slots=True)
class ContinuityResiduals:
    """Hold both reusable continuity formulations."""

    selected: Tensor
    divergence_velocity: Tensor
    divergence_porosity_velocity: Tensor
    kind: ContinuityKind


@dataclass(frozen=True, slots=True)
class BrinkmanDiagnostics:
    """Hold full residual fields and well-defined scalar diagnostics."""

    momentum: MomentumResiduals
    continuity: ContinuityResiduals
    boundary: PressureBoundaryResiduals
    momentum_mse: Tensor
    continuity_mse: Tensor
    momentum_mse_full: Tensor
    continuity_mse_full: Tensor
    interior_crop: int

    @property
    def boundary_mse(self) -> Tensor:
        """Return the inlet-plus-outlet pressure boundary diagnostic."""
        return self.boundary.mse

    def as_dict(self) -> dict[str, Tensor]:
        """Return declared diagnostic names without exposing loss ownership."""
        return {
            "Rx": self.momentum.x.unsqueeze(1),
            "Ry": self.momentum.y.unsqueeze(1),
            "Rc": self.continuity.selected.unsqueeze(1),
            "div_u": self.continuity.divergence_velocity.unsqueeze(1),
            "div_eps_u": self.continuity.divergence_porosity_velocity.unsqueeze(1),
            "mom_mse": self.momentum_mse,
            "cont_mse": self.continuity_mse,
            "mom_mse_full": self.momentum_mse_full,
            "cont_mse_full": self.continuity_mse_full,
            "bc_mse": self.boundary_mse,
            "p_inlet_mse": self.boundary.inlet_mse,
            "p_outlet_mse": self.boundary.outlet_mean_square,
        }


def available_continuity_kinds() -> tuple[str, ...]:
    """Return supported semantic continuity formulations."""
    return ("div_eps_velocity", "div_velocity")


def validate_continuity_kind(kind: str) -> ContinuityKind:
    """Return one exact semantic continuity identifier."""
    if kind not in available_continuity_kinds():
        available = ", ".join(available_continuity_kinds())
        msg = f"Unknown continuity identifier {kind!r}. Available continuity formulations: {available}."
        raise ValueError(msg)
    return cast("ContinuityKind", kind)


def continuity_residuals(
    velocity_x: Tensor,
    velocity_y: Tensor,
    porosity: Tensor,
    derivatives: DerivativeOperator,
    spacing_x: float | Tensor,
    spacing_y: float | Tensor,
    *,
    kind: str,
) -> ContinuityResiduals:
    """
    Compute plain and porosity-weighted continuity residuals.

    Parameters
    ----------
    velocity_x, velocity_y : torch.Tensor
        Physical velocity component fields.
    porosity : torch.Tensor
        Dimensionless porosity field.
    derivatives : DerivativeOperator
        Explicit numerical derivative backend.
    spacing_x, spacing_y : float or torch.Tensor
        Positive physical grid spacing.
    kind : str
        ``"div_eps_velocity"`` or ``"div_velocity"``.

    Returns
    -------
    ContinuityResiduals
        Both formulations and the semantically selected residual.

    """
    if velocity_x.shape != velocity_y.shape or velocity_x.shape != porosity.shape:
        msg = "Velocity components and porosity must have identical shapes."
        raise ValueError(msg)
    resolved_kind = validate_continuity_kind(kind)
    divergence_velocity = derivatives.divergence(
        velocity_x,
        velocity_y,
        spacing_x,
        spacing_y,
    )
    divergence_porosity_velocity = derivatives.divergence(
        porosity * velocity_x,
        porosity * velocity_y,
        spacing_x,
        spacing_y,
    )
    selected = divergence_porosity_velocity if resolved_kind == "div_eps_velocity" else divergence_velocity
    return ContinuityResiduals(
        selected=selected,
        divergence_velocity=divergence_velocity,
        divergence_porosity_velocity=divergence_porosity_velocity,
        kind=resolved_kind,
    )


def _inverse_permeability_components(
    permeability_xx: Tensor,
    permeability_xy_ratio: Tensor,
    permeability_yy: Tensor,
    *,
    permeability_scale_floor: float,
    determinant_floor: float,
    cross_ratio_clip: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return stable physical inverse-permeability tensor components."""
    if permeability_xx.shape != permeability_yy.shape or permeability_xx.shape != permeability_xy_ratio.shape:
        msg = "Permeability component shapes must match."
        raise ValueError(msg)
    scale = torch.sqrt((permeability_xx * permeability_yy).clamp_min(permeability_scale_floor))
    normalized_xx = permeability_xx / scale
    normalized_yy = permeability_yy / scale
    normalized_xy = permeability_xy_ratio.clamp(-cross_ratio_clip, cross_ratio_clip)
    determinant = (normalized_xx * normalized_yy - normalized_xy.square()).clamp_min(determinant_floor)
    inverse_xx = normalized_yy / determinant / scale
    inverse_xy = -normalized_xy / determinant / scale
    inverse_yy = normalized_xx / determinant / scale
    return inverse_xx, inverse_xy, inverse_yy


def brinkman_momentum_residuals(
    pressure: Tensor,
    velocity_x: Tensor,
    velocity_y: Tensor,
    porosity: Tensor,
    permeability_xx: Tensor,
    permeability_xy_ratio: Tensor,
    permeability_yy: Tensor,
    derivatives: DerivativeOperator,
    spacing_x: float | Tensor,
    spacing_y: float | Tensor,
    *,
    viscosity: float = AIR_DYNAMIC_VISCOSITY,
    porosity_floor: float = 1e-6,
    permeability_scale_floor: float = 1e-30,
    determinant_floor: float = 1e-4,
    cross_ratio_clip: float = 0.999,
) -> MomentumResiduals:
    """
    Compute the steady two-dimensional Darcy-Brinkman momentum residual.

    The implemented convention is ``-grad(p) + div(tau) - mu K^-1 u`` with
    ``tau = (mu/eps) * (grad(u) + grad(u)^T - 2/3 div(u) I)``.

    Parameters
    ----------
    pressure, velocity_x, velocity_y : torch.Tensor
        Physical pressure and velocity fields.
    porosity : torch.Tensor
        Dimensionless porosity field.
    permeability_xx, permeability_yy : torch.Tensor
        Positive physical diagonal permeability components in m^2.
    permeability_xy_ratio : torch.Tensor
        Dimensionless cross component relative to the geometric mean.
    derivatives : DerivativeOperator
        Explicit numerical derivative backend.
    spacing_x, spacing_y : float or torch.Tensor
        Positive physical grid spacing.
    viscosity : float, optional
        Dynamic viscosity in Pa s.
    porosity_floor : float, optional
        Positive numerical floor for porosity.
    permeability_scale_floor : float, optional
        Positive numerical floor for the permeability geometric mean.
    determinant_floor : float, optional
        Positive numerical floor for normalized permeability inversion.
    cross_ratio_clip : float, optional
        Absolute clamp for the dimensionless cross-permeability ratio.

    Returns
    -------
    MomentumResiduals
        Physical x- and y-momentum residual fields.

    """
    fields = (
        pressure,
        velocity_x,
        velocity_y,
        porosity,
        permeability_xx,
        permeability_xy_ratio,
        permeability_yy,
    )
    if any(field.shape != pressure.shape for field in fields[1:]):
        msg = "All Brinkman physical fields must have identical shapes."
        raise ValueError(msg)
    if viscosity <= 0:
        msg = f"viscosity must be positive, got {viscosity}."
        raise ValueError(msg)

    safe_porosity = porosity.clamp_min(porosity_floor)
    pressure_x, pressure_y = derivatives.gradient(pressure, spacing_x, spacing_y)
    velocity_x_x, velocity_x_y = derivatives.gradient(velocity_x, spacing_x, spacing_y)
    velocity_y_x, velocity_y_y = derivatives.gradient(velocity_y, spacing_x, spacing_y)
    divergence_velocity = velocity_x_x + velocity_y_y

    coefficient = viscosity / safe_porosity
    stress_xx = coefficient * (2.0 * velocity_x_x - (2.0 / 3.0) * divergence_velocity)
    stress_yy = coefficient * (2.0 * velocity_y_y - (2.0 / 3.0) * divergence_velocity)
    stress_xy = coefficient * (velocity_x_y + velocity_y_x)
    stress_divergence_x = derivatives.divergence(stress_xx, stress_xy, spacing_x, spacing_y)
    stress_divergence_y = derivatives.divergence(stress_xy, stress_yy, spacing_x, spacing_y)

    inverse_xx, inverse_xy, inverse_yy = _inverse_permeability_components(
        permeability_xx,
        permeability_xy_ratio,
        permeability_yy,
        permeability_scale_floor=permeability_scale_floor,
        determinant_floor=determinant_floor,
        cross_ratio_clip=cross_ratio_clip,
    )
    drag_x = viscosity * (inverse_xx * velocity_x + inverse_xy * velocity_y)
    drag_y = viscosity * (inverse_xy * velocity_x + inverse_yy * velocity_y)
    return MomentumResiduals(
        x=-pressure_x + stress_divergence_x - drag_x,
        y=-pressure_y + stress_divergence_y - drag_y,
    )


def _field_mapping(
    tensor: Tensor,
    fields: Sequence[str],
    required: tuple[str, ...],
    *,
    label: str,
) -> dict[str, Tensor]:
    """Bind task-specific names to channel tensors outside numerical kernels."""
    if tensor.ndim < _MIN_TASK_TENSOR_RANK:
        msg = f"{label} tensor must have batch, channel, and spatial axes."
        raise ValueError(msg)
    if tensor.shape[1] != len(fields):
        msg = f"{label} tensor has {tensor.shape[1]} channels but {len(fields)} field names."
        raise ValueError(msg)
    if len(fields) != len(set(fields)):
        msg = f"{label} field declaration contains duplicate names: {list(fields)}."
        raise ValueError(msg)
    missing = [name for name in required if name not in fields]
    if missing:
        msg = f"{label} fields are missing required steady-flow roles: {missing}."
        raise ValueError(msg)
    indices = {name: index for index, name in enumerate(fields)}
    return {name: tensor[:, indices[name]] for name in required}


def evaluate_steady_2d_brinkman(
    inputs: Tensor,
    outputs: Tensor,
    *,
    input_fields: Sequence[str],
    output_fields: Sequence[str],
    derivatives: DerivativeOperator,
    continuity: str,
    boundary: str,
    interior_crop: int = 0,
    spatial_axes: SpatialAxes = (-2, -1),
) -> BrinkmanDiagnostics:
    """
    Evaluate task-bound steady-flow residuals on physical tensor views.

    Parameters
    ----------
    inputs, outputs : torch.Tensor
        Physical tensor views in caller-declared field order. Diagonal
        permeability channels retain the task's dimensionless log10 storage
        representation and are converted before entering the numerical kernel.
    input_fields, output_fields : Sequence[str]
        Exact task-owned field declarations used for name-based binding.
    derivatives : DerivativeOperator
        Explicit physical or spectral derivative backend.
    continuity : str
        Semantic continuity formulation.
    boundary : str
        Semantic pressure boundary formulation.
    interior_crop : int, optional
        Cells cropped before scalar momentum/continuity diagnostics.
    spatial_axes : tuple[int, int], optional
        Spatial axes after removing the channel dimension.

    Returns
    -------
    BrinkmanDiagnostics
        Full residual fields and scalar full/interior diagnostics.

    """
    if boundary != PRESSURE_BOUNDARY_KIND:
        msg = f"Unknown pressure boundary identifier {boundary!r}; expected {PRESSURE_BOUNDARY_KIND!r}."
        raise ValueError(msg)
    input_values = _field_mapping(
        inputs,
        input_fields,
        ("x", "y", "kxx", "kxy", "kyy", "eps", "p_bc"),
        label="input",
    )
    output_values = _field_mapping(
        outputs,
        output_fields,
        ("p", "u", "v"),
        label="output",
    )
    spacing_x, spacing_y = infer_uniform_spacing(
        input_values["x"],
        input_values["y"],
        axes=spatial_axes,
    )
    permeability_xx = torch.pow(10.0, input_values["kxx"])
    permeability_yy = torch.pow(10.0, input_values["kyy"])
    momentum = brinkman_momentum_residuals(
        output_values["p"],
        output_values["u"],
        output_values["v"],
        input_values["eps"],
        permeability_xx,
        input_values["kxy"],
        permeability_yy,
        derivatives,
        spacing_x,
        spacing_y,
    )
    continuity_residual = continuity_residuals(
        output_values["u"],
        output_values["v"],
        input_values["eps"].clamp_min(1e-6),
        derivatives,
        spacing_x,
        spacing_y,
        kind=continuity,
    )
    pressure_boundary = pressure_boundary_residuals(
        output_values["p"],
        input_values["p_bc"],
        input_values["y"],
        spacing_y,
        spatial_axes=spatial_axes,
    )
    momentum_x_interior = crop_interior(momentum.x, interior_crop, axes=spatial_axes)
    momentum_y_interior = crop_interior(momentum.y, interior_crop, axes=spatial_axes)
    continuity_interior = crop_interior(
        continuity_residual.selected,
        interior_crop,
        axes=spatial_axes,
    )
    return BrinkmanDiagnostics(
        momentum=momentum,
        continuity=continuity_residual,
        boundary=pressure_boundary,
        momentum_mse=(momentum_x_interior.square() + momentum_y_interior.square()).mean(),
        continuity_mse=continuity_interior.square().mean(),
        momentum_mse_full=(momentum.x.square() + momentum.y.square()).mean(),
        continuity_mse_full=continuity_residual.selected.square().mean(),
        interior_crop=interior_crop,
    )


PhysicsEvaluator = Callable[..., BrinkmanDiagnostics]
_PHYSICS_EVALUATORS = MappingProxyType({STEADY_BRINKMAN_KIND: evaluate_steady_2d_brinkman})


def available_physics_kinds() -> tuple[str, ...]:
    """Return domain physics equation-set identifiers with evaluators."""
    return tuple(sorted(_PHYSICS_EVALUATORS))


def resolve_physics_evaluator(kind: str) -> PhysicsEvaluator:
    """Resolve a task-selected reusable physics evaluator."""
    try:
        return _PHYSICS_EVALUATORS[kind]
    except KeyError as error:
        available = ", ".join(available_physics_kinds())
        msg = f"Unknown domain physics identifier {kind!r}. Available physics: {available}."
        raise ValueError(msg) from error
