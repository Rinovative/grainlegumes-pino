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
  - Semantic physics, continuity, and pressure-boundary identifiers fail closed
  - Full-grid fields remain available beside explicit interior scalar reductions

This module does NOT:
  - Normalize model tensors or choose physics-loss weights and warmup schedules
  - Own derivative discretization, checkpointing, W&B logging, or Optuna policy
  - Infer task channel order without explicit field-name declarations
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from . import domain_physics_contracts as physics_contracts
from .domain_physics_boundary import PressureBoundaryResiduals, pressure_boundary_residuals
from .domain_physics_derivatives import DerivativeOperator, SpatialAxes, crop_interior, infer_uniform_spacing

AIR_DYNAMIC_VISCOSITY = 1.8139e-5
POROSITY_FLOOR = 1e-6
PERMEABILITY_SCALE_FLOOR = 1e-30
PERMEABILITY_DETERMINANT_FLOOR = 1e-4
PERMEABILITY_CROSS_RATIO_CLIP = 0.999
_MIN_TASK_TENSOR_RANK = 3


@dataclass(frozen=True, slots=True)
class MomentumResiduals:
    """
    Hold physical x- and y-momentum residual fields.

    ``x`` and ``y`` retain the input tensor's batch/spatial shape and have units
    of pressure gradient, Pa/m. No crop or scalar reduction is implied.

    Attributes
    ----------
    x, y : torch.Tensor
        Full-grid x- and y-momentum equation residuals in Pa/m.

    """

    x: Tensor
    y: Tensor


@dataclass(frozen=True, slots=True)
class ContinuityResiduals:
    """
    Hold both continuity residual fields and the selected training formulation.

    ``divergence_velocity`` is ``div(u)`` and
    ``divergence_porosity_velocity`` is ``div(eps*u)``. Both have units 1/s.
    ``selected`` aliases exactly the field identified by ``kind`` without
    removing the other diagnostic.

    Attributes
    ----------
    selected : torch.Tensor
        Exact alias of the training-selected full-grid continuity field.
    divergence_velocity, divergence_porosity_velocity : torch.Tensor
        Full-grid plain and conservative residuals with matching shapes.
    kind : Literal["div_eps_velocity", "div_velocity"]
        Identifier determining the ``selected`` alias.

    """

    selected: Tensor
    divergence_velocity: Tensor
    divergence_porosity_velocity: Tensor
    kind: physics_contracts.ContinuityKind


@dataclass(frozen=True, slots=True)
class BrinkmanDiagnostics:
    """
    Hold full residual fields and formulation-explicit scalar diagnostics.

    Momentum and both continuity fields remain full-grid. Canonical scalar MSEs
    use the declared interior crop, parallel ``*_full_grid`` values support the
    fixed training monitor, and pressure boundary diagnostics use full-grid
    masks. The selected training continuity never replaces either retained
    formulation.

    Attributes
    ----------
    momentum, continuity, boundary
        Full-grid residual containers and pressure-boundary diagnostics.
    momentum_residual_mse : torch.Tensor
        Mean of ``Rx² + Ry²`` over the cropped domain, in ``(Pa/m)²``.
    div_velocity_mse, div_eps_velocity_mse : torch.Tensor
        Independent cropped mean-square continuity diagnostics, in ``1/s²``.
    momentum_residual_mse_full_grid : torch.Tensor
        Full-grid mean of ``Rx² + Ry²``.
    div_velocity_mse_full_grid, div_eps_velocity_mse_full_grid : torch.Tensor
        Independent full-grid mean-square continuity diagnostics.
    interior_crop : int
        Cells removed from every spatial edge for canonical scalar residuals.

    """

    momentum: MomentumResiduals
    continuity: ContinuityResiduals
    boundary: PressureBoundaryResiduals
    momentum_residual_mse: Tensor
    div_velocity_mse: Tensor
    div_eps_velocity_mse: Tensor
    momentum_residual_mse_full_grid: Tensor
    div_velocity_mse_full_grid: Tensor
    div_eps_velocity_mse_full_grid: Tensor
    interior_crop: int

    @property
    def momentum_mse(self) -> Tensor:
        """Return the canonical cropped training momentum MSE in ``(Pa/m)²``."""
        return self.momentum_residual_mse

    @property
    def continuity_mse(self) -> Tensor:
        """Return only the experiment-selected continuity value for training."""
        if self.continuity.kind == "div_eps_velocity":
            return self.div_eps_velocity_mse
        return self.div_velocity_mse

    @property
    def momentum_mse_full(self) -> Tensor:
        """Return the full-grid momentum MSE used by monitors, in ``(Pa/m)²``."""
        return self.momentum_residual_mse_full_grid

    @property
    def continuity_mse_full(self) -> Tensor:
        """Return the selected full-grid continuity value for training monitors."""
        if self.continuity.kind == "div_eps_velocity":
            return self.div_eps_velocity_mse_full_grid
        return self.div_velocity_mse_full_grid

    @property
    def boundary_mse(self) -> Tensor:
        """Return the inlet-plus-outlet pressure boundary diagnostic in Pa²."""
        return self.boundary.mse

    def as_dict(self) -> dict[str, Tensor]:
        """
        Return formulation-explicit residual arrays and scalar diagnostics.

        ``Rx`` and ``Ry`` have units Pa/m. ``div_u`` and ``div_eps_u`` have
        units 1/s. Canonical residual MSEs use ``interior_crop`` while pressure
        boundary diagnostics use inlet/outlet masks on the full grid.

        Returns
        -------
        dict[str, torch.Tensor]
            Full fields with a singleton channel axis plus formulation-explicit
            scalar tensors. The mapping contains no selected-continuity alias or
            loss weights, so consumers must use the declared key names.

        """
        return {
            "Rx": self.momentum.x.unsqueeze(1),
            "Ry": self.momentum.y.unsqueeze(1),
            "div_u": self.continuity.divergence_velocity.unsqueeze(1),
            "div_eps_u": self.continuity.divergence_porosity_velocity.unsqueeze(1),
            "momentum_residual_mse": self.momentum_residual_mse,
            "div_velocity_mse": self.div_velocity_mse,
            "div_eps_velocity_mse": self.div_eps_velocity_mse,
            "momentum_residual_mse_full_grid": self.momentum_residual_mse_full_grid,
            "div_velocity_mse_full_grid": self.div_velocity_mse_full_grid,
            "div_eps_velocity_mse_full_grid": self.div_eps_velocity_mse_full_grid,
            "pressure_boundary_mse": self.boundary_mse,
            "pressure_inlet_mse": self.boundary.inlet_mse,
            "pressure_outlet_mean_square": self.boundary.outlet_mean_square,
        }


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
        Physical velocity component fields in m/s with matching sample/spatial
        shapes, normally ``[batch, y, x]``.
    porosity : torch.Tensor
        Dimensionless porosity field with the same shape.
    derivatives : DerivativeOperator
        Explicit numerical derivative backend.
    spacing_x, spacing_y : float or torch.Tensor
        Positive physical grid spacing.
    kind : str
        ``"div_eps_velocity"`` or ``"div_velocity"``.

    Returns
    -------
    ContinuityResiduals
        Full-grid ``div(u)`` and ``div(eps*u)`` fields in 1/s plus the exact
        selected alias. No crop or scalar reduction is applied.

    Raises
    ------
    TypeError
        If the derivative backend rejects a field or spacing dtype.
    ValueError
        If field shapes differ, ``kind`` is unsupported, or the derivative
        backend rejects spacing, axes, or field geometry.

    """
    if velocity_x.shape != velocity_y.shape or velocity_x.shape != porosity.shape:
        msg = "Velocity components and porosity must have identical shapes."
        raise ValueError(msg)
    resolved_kind = physics_contracts.validate_continuity_kind(kind)
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
    """
    Invert the symmetric 2D permeability representation with explicit floors.

    Diagonal components are physical m² values. The stored cross channel is the
    dimensionless ratio ``Kxy / sqrt(Kxx*Kyy)``. It is clipped before inversion.
    The geometric scale and normalized determinant are floored independently,
    yielding ``(K^-1_xx, K^-1_xy, K^-1_yy)`` in 1/m² without materializing a
    matrix tensor.

    Raises
    ------
    ValueError
        If the three permeability component shapes differ.

    """
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
    porosity_floor: float = POROSITY_FLOOR,
    permeability_scale_floor: float = PERMEABILITY_SCALE_FLOOR,
    determinant_floor: float = PERMEABILITY_DETERMINANT_FLOOR,
    cross_ratio_clip: float = PERMEABILITY_CROSS_RATIO_CLIP,
) -> MomentumResiduals:
    """
    Compute the steady two-dimensional Darcy-Brinkman momentum residual.

    The implemented convention is ``-grad(p) + div(tau) - mu K^-1 u`` with
    ``tau = (mu/eps) * (grad(u) + grad(u)^T - 2/3 div(u) I)``.

    Parameters
    ----------
    pressure, velocity_x, velocity_y : torch.Tensor
        Physical pressure in Pa and velocity components in m/s with identical
        sample/spatial shapes, normally ``[batch, y, x]``.
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
        Lower clamp applied to dimensionless porosity. The canonical default is
        positive.
    permeability_scale_floor : float, optional
        Lower clamp for the permeability geometric mean in m².
    determinant_floor : float, optional
        Lower clamp for the normalized permeability determinant.
    cross_ratio_clip : float, optional
        Symmetric absolute clamp for the dimensionless cross-permeability ratio.

    Returns
    -------
    MomentumResiduals
        Full-grid x- and y-momentum residual fields in Pa/m.

    Raises
    ------
    TypeError
        If the derivative backend rejects a field or spacing dtype.
    ValueError
        If physical field shapes differ, viscosity is not positive, or the
        derivative backend rejects field geometry or spacing.

    Notes
    -----
    The deviatoric stress uses the three-dimensional ``2/3 div(u)`` trace
    removal convention on a two-dimensional flow slice. Stabilization clamps
    affect porosity and permeability inversion only. Inputs are not mutated.

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
    """
    Bind required task names to channel views without reordering the source.

    ``tensor`` must expose batch and channel as its first two axes. The complete
    declared ``fields`` sequence must match the channel count and be duplicate
    free. Every required steady-flow role is returned as ``tensor[:, index]``,
    preserving batch/spatial storage as a view.

    Raises
    ------
    ValueError
        If tensor rank/channel count, field uniqueness, or required membership
        violates the task-binding contract.

    """
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
        Task-ordered BCHW tensors. Output channels are physical ``p``/``u``/``v``
        values. Input ``x``, ``y``, ``eps``, and ``p_bc`` retain physical/stored
        values. Diagonal permeability channels are log10 ratios to 1 m² and are
        exponentiated, while ``kxy`` is the dimensionless geometric-mean ratio.
    input_fields, output_fields : Sequence[str]
        Exact task-owned field declarations used for name-based binding.
    derivatives : DerivativeOperator
        Explicit physical or spectral derivative backend.
    continuity : str
        Training-selected continuity identifier used for the selected aggregate.
        Both plain and porosity-weighted formulations are always evaluated.
    boundary : str
        Semantic pressure boundary formulation.
    interior_crop : int, optional
        Cells removed from every spatial edge before canonical scalar momentum
        and continuity mean-square reductions. Full-grid diagnostics are also
        retained.
    spatial_axes : tuple[int, int], optional
        Spatial axes after removing the channel dimension.

    Returns
    -------
    BrinkmanDiagnostics
        Full-grid fields, cropped and full-grid mean-square residuals, and
        pressure-boundary diagnostics. Momentum MSE has units ``(Pa/m)²``,
        continuity MSEs ``1/s²``, and boundary MSE ``Pa²``.

    Raises
    ------
    TypeError
        If tensor/coordinate dtypes violate derivative or boundary contracts.
    ValueError
        If semantic identifiers, channel declarations, BCHW geometry, Cartesian
        spacing, crop, or physical field shapes violate the evaluator contract.

    Notes
    -----
    Both continuity formulations are always computed. ``continuity`` selects
    only selected aggregate properties used by training. Formulation-explicit fields
    and scalars remain available regardless of that selection.

    """
    if boundary != physics_contracts.PRESSURE_BOUNDARY_KIND:
        msg = f"Unknown pressure boundary identifier {boundary!r}. Expected {physics_contracts.PRESSURE_BOUNDARY_KIND!r}."
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
        input_values["eps"].clamp_min(POROSITY_FLOOR),
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
    div_velocity_interior = crop_interior(
        continuity_residual.divergence_velocity,
        interior_crop,
        axes=spatial_axes,
    )
    div_eps_velocity_interior = crop_interior(
        continuity_residual.divergence_porosity_velocity,
        interior_crop,
        axes=spatial_axes,
    )
    return BrinkmanDiagnostics(
        momentum=momentum,
        continuity=continuity_residual,
        boundary=pressure_boundary,
        momentum_residual_mse=(momentum_x_interior.square() + momentum_y_interior.square()).mean(),
        div_velocity_mse=div_velocity_interior.square().mean(),
        div_eps_velocity_mse=div_eps_velocity_interior.square().mean(),
        momentum_residual_mse_full_grid=(momentum.x.square() + momentum.y.square()).mean(),
        div_velocity_mse_full_grid=continuity_residual.divergence_velocity.square().mean(),
        div_eps_velocity_mse_full_grid=continuity_residual.divergence_porosity_velocity.square().mean(),
        interior_crop=interior_crop,
    )


PhysicsEvaluator = Callable[..., BrinkmanDiagnostics]
_PHYSICS_EVALUATORS = MappingProxyType({physics_contracts.STEADY_BRINKMAN_KIND: evaluate_steady_2d_brinkman})


def available_physics_kinds() -> tuple[str, ...]:
    """Return domain physics equation-set identifiers with evaluators."""
    return tuple(sorted(_PHYSICS_EVALUATORS))


def resolve_physics_evaluator(kind: str) -> PhysicsEvaluator:
    """
    Resolve the exact task-selected equation-set evaluator.

    Parameters
    ----------
    kind : str
        Canonical domain physics identifier.

    Returns
    -------
    PhysicsEvaluator
        Registered evaluator callable. Resolution performs no computation.

    Raises
    ------
    ValueError
        If ``kind`` has no registered evaluator.

    """
    try:
        return _PHYSICS_EVALUATORS[kind]
    except KeyError as error:
        available = ", ".join(available_physics_kinds())
        msg = f"Unknown domain physics identifier {kind!r}. Available physics: {available}."
        raise ValueError(msg) from error
