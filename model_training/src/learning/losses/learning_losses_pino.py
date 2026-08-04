"""
===============================================================================
learning_losses_pino.py
===============================================================================
Compose supervised and task-selected physics-informed loss components.

Responsibilities:
  - Combine named data, momentum, continuity, and boundary contributions
  - Apply explicit component weights and deterministic epoch warmup
  - Construct physical tensor views once for domain-owned physics evaluators
  - Expose current named components and reusable domain diagnostics

Design principles:
  - Named loss components remain stable across supervised and physics-enabled runs
  - Normalized predictions are converted to physical views explicitly and once
  - Warmup is deterministic at completed-epoch boundaries

This module does NOT:
  - Define equations, derivatives, residuals, or boundaries. ``domain`` owns them
  - Parse semantic config or select implementations. ``losses.factory`` owns selection
  - Define or aggregate dataset metrics. ``learning.metrics`` owns evaluation
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from src import domain


class TensorNormalizer(Protocol):
    """Define the normalizer surface required by physics loss composition."""

    def inverse_transform(self, tensor: Tensor) -> Tensor:
        """Convert a normalized tensor to its physical/task representation."""
        ...


class SemanticDataLoss(nn.Module):
    """Wrap a relative norm with one explicit non-negative semantic weight."""

    def __init__(self, implementation: Any, *, weight: float) -> None:
        """Validate the weight and retain the callable relative norm."""
        super().__init__()
        if weight < 0:
            msg = f"Data-loss weight must be non-negative, got {weight}."
            raise ValueError(msg)
        self.implementation = implementation
        self.weight = float(weight)
        self.reduction = "sample_mean"

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return the explicitly weighted relative data loss."""
        return self.weight * self.implementation(pred, target)


@dataclass(frozen=True, slots=True)
class LinearWarmup:
    """
    Define an immutable deterministic zero-to-target epoch schedule.

    Parameters
    ----------
    target : float
        Non-negative terminal component weight.
    epochs : int
        Non-negative number of zero-based epoch positions needed to reach it.

    Raises
    ------
    ValueError
        If either schedule value is negative.

    """

    target: float
    epochs: int

    def __post_init__(self) -> None:
        """Validate non-negative schedule values."""
        if self.target < 0:
            msg = f"Warmup target must be non-negative, got {self.target}."
            raise ValueError(msg)
        if self.epochs < 0:
            msg = f"Warmup epochs must be non-negative, got {self.epochs}."
            raise ValueError(msg)

    def fraction(self, epoch: int) -> float:
        """
        Return the applied zero-to-one fraction at a zero-based epoch position.

        A zero-length schedule returns ``1.0`` immediately. Active schedules
        start at zero and clamp at one. Negative epoch positions raise
        ``ValueError``.
        """
        if epoch < 0:
            msg = f"Warmup epoch must be non-negative, got {epoch}."
            raise ValueError(msg)
        if self.epochs == 0:
            return 1.0
        return min(float(epoch) / float(self.epochs), 1.0)

    def value(self, epoch: int) -> float:
        """
        Return the scheduled weight for a zero-based epoch position.

        Epoch zero has zero physics weight when warmup is active. ``epochs``
        and later use the complete target. A zero-length warmup uses the target
        immediately.
        """
        return self.target * self.fraction(epoch)


class SemanticComposedLoss(nn.Module):
    """
    Compose one supervised or physics-informed semantic training objective.

    Named returned contributions are already weighted, so ``total`` is exactly
    the sum of ``data``, ``momentum``, ``boundary``, and the selected
    formulation-qualified continuity component. Disabled physics components
    remain present as scalar zeros, giving supervised and physics-informed
    training one stable interface.

    Parameters
    ----------
    data_loss : torch.nn.Module
        Unweighted normalized-space supervised loss.
    data_weight : float
        Explicit supervised contribution weight.
    physics_enabled : bool
        Whether to evaluate domain physics.
    physics_kind : str
        Task-owned domain physics identifier.
    input_fields, output_fields : tuple[str, ...]
        Exact task field declarations used for name-based domain binding.
    continuity : str
        Experiment-selected task-allowed continuity formulation.
    boundary : str
        Task-owned boundary formulation identifier.
    derivatives : domain.physics.derivatives.DerivativeOperator
        Explicit numerical derivative backend.
    residual_weight, boundary_weight : LinearWarmup
        Deterministic component schedules.
    interior_crop : int
        Interior crop applied by the domain diagnostic evaluator.

    """

    def __init__(
        self,
        *,
        data_loss: nn.Module,
        data_weight: float,
        physics_enabled: bool,
        physics_kind: str,
        input_fields: tuple[str, ...],
        output_fields: tuple[str, ...],
        continuity: str,
        boundary: str,
        derivatives: domain.physics.derivatives.DerivativeOperator,
        residual_weight: LinearWarmup,
        boundary_weight: LinearWarmup,
        interior_crop: int,
    ) -> None:
        """
        Validate composition inputs, resolve physics, and register epoch state.

        The zero-based warmup position becomes persistent module state, whereas
        fitted normalizers and latest detached components remain runtime-only.
        """
        super().__init__()
        if data_weight < 0:
            msg = f"data_weight must be non-negative, got {data_weight}."
            raise ValueError(msg)
        if interior_crop < 0:
            msg = f"interior_crop must be non-negative, got {interior_crop}."
            raise ValueError(msg)
        self.data_loss = data_loss
        self.data_weight = float(data_weight)
        self.physics_enabled = bool(physics_enabled)
        self.physics_kind = physics_kind
        self.input_fields = tuple(input_fields)
        self.output_fields = tuple(output_fields)
        self.continuity = domain.physics.contracts.validate_continuity_kind(continuity)
        if boundary != domain.physics.contracts.PRESSURE_BOUNDARY_KIND:
            msg = f"Unknown pressure boundary identifier {boundary!r}. Expected {domain.physics.contracts.PRESSURE_BOUNDARY_KIND!r}."
            raise ValueError(msg)
        self.boundary = boundary
        self.derivatives = derivatives
        self.residual_weight = residual_weight
        self.boundary_weight = boundary_weight
        self.interior_crop = int(interior_crop)
        self._physics_evaluator = domain.physics.brinkman.resolve_physics_evaluator(physics_kind)
        self.in_normalizer: TensorNormalizer | None = None
        self.out_normalizer: TensorNormalizer | None = None
        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long), persistent=True)
        self._last_components: dict[str, Tensor] = {}

    @property
    def continuity_component_name(self) -> str:
        """
        Return the selected formulation-qualified continuity component name.

        Returns
        -------
        str
            Stable selected continuity contribution identifier.

        """
        return f"continuity_{self.continuity}"

    @property
    def component_names(self) -> tuple[str, ...]:
        """
        Return the exact ordered named loss-component interface.

        Returns
        -------
        tuple[str, ...]
            Total, data, momentum, boundary, and selected continuity names.

        """
        return (
            "total",
            "data",
            "momentum",
            "boundary",
            self.continuity_component_name,
        )

    def set_normalizers(
        self,
        *,
        in_normalizer: TensorNormalizer,
        out_normalizer: TensorNormalizer,
    ) -> None:
        """
        Attach fitted normalizers used to construct physical physics views.

        Parameters
        ----------
        in_normalizer, out_normalizer : TensorNormalizer
            Fitted task input/output normalizers.

        """
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer

    def set_epoch(self, epoch: int) -> None:
        """
        Mutate the persistent zero-based warmup position.

        The scalar buffer is checkpointed with the loss module. Booleans and
        negative or non-integer positions raise ``ValueError``.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            msg = f"epoch must be a non-negative integer, got {epoch!r}."
            raise ValueError(msg)
        self.current_epoch.fill_(epoch)

    def component_weights(self, *, epoch: int | None = None) -> dict[str, float]:
        """
        Return explicit weights for every named non-total component.

        ``epoch`` overrides the checkpointed warmup position for this query.
        Disabled physics keeps its stable component keys with zero weights.
        """
        position = int(self.current_epoch.item()) if epoch is None else epoch
        return {
            "data": self.data_weight,
            "momentum": self.residual_weight.value(position) if self.physics_enabled else 0.0,
            "boundary": self.boundary_weight.value(position) if self.physics_enabled else 0.0,
            self.continuity_component_name: (self.residual_weight.value(position) if self.physics_enabled else 0.0),
        }

    def telemetry_state(self, *, epoch: int | None = None) -> dict[str, float]:
        """
        Return applied physics weights and warmup fractions for one epoch.

        Supervised configurations return no synthetic physics telemetry. The
        selected continuity contribution shares the declared residual weight.
        """
        if not self.physics_enabled:
            return {}
        position = int(self.current_epoch.item()) if epoch is None else epoch
        return {
            "weight_physics": self.residual_weight.value(position),
            "weight_boundary": self.boundary_weight.value(position),
            "warmup_physics_fraction": self.residual_weight.fraction(position),
            "warmup_boundary_fraction": self.boundary_weight.fraction(position),
        }

    def compute_physics_diagnostics(
        self,
        pred: Tensor,
        *,
        x: Tensor,
    ) -> domain.physics.brinkman.BrinkmanDiagnostics:
        """
        Evaluate domain-owned physics from normalized model tensors.

        Parameters
        ----------
        pred : torch.Tensor
            Normalized model outputs.
        x : torch.Tensor
            Normalized model inputs.

        Returns
        -------
        domain.physics.brinkman.BrinkmanDiagnostics
            Reusable full-field and scalar physical diagnostics.

        """
        if self.in_normalizer is None or self.out_normalizer is None:
            msg = "Physics diagnostics require fitted input and output normalizers."
            raise RuntimeError(msg)
        inputs_physical = self.in_normalizer.inverse_transform(x)
        outputs_physical = self.out_normalizer.inverse_transform(pred)
        return self._physics_evaluator(
            inputs_physical,
            outputs_physical,
            input_fields=self.input_fields,
            output_fields=self.output_fields,
            derivatives=self.derivatives,
            continuity=self.continuity,
            boundary=self.boundary,
            interior_crop=self.interior_crop,
        )

    @torch.no_grad()
    def compute_diagnostics(self, pred: Tensor, *, x: Tensor) -> dict[str, Tensor]:
        """Return declared diagnostic keys from the domain evaluator."""
        return self.compute_physics_diagnostics(pred, x=x).as_dict()

    def compute_components(
        self,
        pred: Tensor,
        *,
        x: Tensor | None,
        y: Tensor,
        epoch: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute named weighted loss components.

        Parameters
        ----------
        pred : torch.Tensor
            Normalized prediction tensor.
        x : torch.Tensor or None
            Normalized task inputs, required only when physics is enabled.
        y : torch.Tensor
            Normalized supervised target tensor.
        epoch : int or None, optional
            Explicit warmup position. Defaults to ``current_epoch``.

        Returns
        -------
        dict[str, torch.Tensor]
            Scalar ``total``, ``data``, ``momentum``, ``boundary``, and selected
            formulation-qualified continuity contributions.

        """
        if pred.shape != y.shape:
            msg = f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(y.shape)}."
            raise ValueError(msg)
        weights = self.component_weights(epoch=epoch)
        continuity_name = self.continuity_component_name
        data = weights["data"] * self.data_loss(pred, y)
        zero = pred.new_zeros(())
        momentum = zero
        boundary = zero
        continuity = zero
        if self.physics_enabled:
            if x is None:
                msg = "Physics-informed loss requires the normalized input tensor x."
                raise ValueError(msg)
            diagnostics = self.compute_physics_diagnostics(pred, x=x)
            momentum = weights["momentum"] * diagnostics.momentum_mse
            boundary = weights["boundary"] * diagnostics.boundary_mse
            continuity = weights[continuity_name] * diagnostics.continuity_mse
        total = data + momentum + boundary + continuity
        return {
            "total": total,
            "data": data,
            "momentum": momentum,
            "boundary": boundary,
            continuity_name: continuity,
        }

    @property
    def last_components(self) -> dict[str, Tensor]:
        """Return detached components from the most recent forward call."""
        return dict(self._last_components)

    def forward(
        self,
        pred: Tensor,
        y: Tensor | None = None,
        *,
        x: Tensor | None = None,
        epoch: int | None = None,
        **_kwargs: Any,
    ) -> Tensor:
        """
        Return the scalar total while caching detached named components.

        ``y`` is mandatory. Physics-enabled compositions additionally require
        ``x`` through :meth:`compute_components`. Extra keyword arguments are
        accepted because the generic training loop supplies shared loss arguments.
        """
        if y is None:
            msg = "SemanticComposedLoss requires a target tensor y."
            raise ValueError(msg)
        components = self.compute_components(pred, x=x, y=y, epoch=epoch)
        self._last_components = {name: value.detach() for name, value in components.items()}
        return components["total"]
