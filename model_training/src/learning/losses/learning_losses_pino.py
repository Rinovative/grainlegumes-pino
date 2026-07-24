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

Boundaries:
  - Equations, derivatives, residuals, and boundary calculations live in domain
  - Semantic config parsing and implementation selection live in losses.factory
  - Dataset metric definitions and aggregation live in learning.metrics
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


@dataclass(frozen=True, slots=True)
class LinearWarmup:
    """Define a deterministic zero-to-target epoch schedule."""

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

    def value(self, epoch: int) -> float:
        """
        Return the scheduled weight for a zero-based epoch position.

        Epoch zero has zero physics weight when warmup is active; ``epochs``
        and later use the complete target. A zero-length warmup uses the target
        immediately.
        """
        if epoch < 0:
            msg = f"Warmup epoch must be non-negative, got {epoch}."
            raise ValueError(msg)
        if self.epochs == 0:
            return self.target
        fraction = min(float(epoch) / float(self.epochs), 1.0)
        return self.target * fraction


class SemanticComposedLoss(nn.Module):
    """
    Compose one supervised or physics-informed semantic training objective.

    Named returned contributions are already weighted, so ``total`` is exactly
    ``data + momentum + continuity + boundary``. Disabled physics components
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
    continuity, boundary : str
        Task-owned physics formulation identifiers.
    derivatives : domain.physics.derivatives.DerivativeOperator
        Explicit numerical derivative backend.
    residual_weight, boundary_weight : LinearWarmup
        Deterministic component schedules.
    interior_crop : int
        Interior crop applied by the domain diagnostic evaluator.

    """

    component_names = ("total", "data", "momentum", "continuity", "boundary")

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
        """Initialize the semantic composition and resolve domain physics."""
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
        self.continuity = domain.physics.brinkman.validate_continuity_kind(continuity)
        if boundary != domain.physics.brinkman.PRESSURE_BOUNDARY_KIND:
            msg = f"Unknown pressure boundary identifier {boundary!r}; expected {domain.physics.brinkman.PRESSURE_BOUNDARY_KIND!r}."
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
        """Set the zero-based deterministic warmup position."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            msg = f"epoch must be a non-negative integer, got {epoch!r}."
            raise ValueError(msg)
        self.current_epoch.fill_(epoch)

    def component_weights(self, *, epoch: int | None = None) -> dict[str, float]:
        """Return explicit weights for every named non-total component."""
        position = int(self.current_epoch.item()) if epoch is None else epoch
        return {
            "data": self.data_weight,
            "momentum": self.residual_weight.value(position) if self.physics_enabled else 0.0,
            "continuity": self.residual_weight.value(position) if self.physics_enabled else 0.0,
            "boundary": self.boundary_weight.value(position) if self.physics_enabled else 0.0,
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
        if not self.physics_enabled:
            msg = "Physics diagnostics are unavailable when loss.physics.enabled is false."
            raise RuntimeError(msg)
        if self.in_normalizer is None or self.out_normalizer is None:
            msg = "Physics-informed loss requires fitted input and output normalizers."
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
            Explicit warmup position; defaults to ``current_epoch``.

        Returns
        -------
        dict[str, torch.Tensor]
            Scalar ``total``, ``data``, ``momentum``, ``continuity``, and
            ``boundary`` contributions.

        """
        if pred.shape != y.shape:
            msg = f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(y.shape)}."
            raise ValueError(msg)
        weights = self.component_weights(epoch=epoch)
        data = weights["data"] * self.data_loss(pred, y)
        zero = pred.new_zeros(())
        momentum = zero
        continuity = zero
        boundary = zero
        if self.physics_enabled:
            if x is None:
                msg = "Physics-informed loss requires the normalized input tensor x."
                raise ValueError(msg)
            diagnostics = self.compute_physics_diagnostics(pred, x=x)
            momentum = weights["momentum"] * diagnostics.momentum_mse
            continuity = weights["continuity"] * diagnostics.continuity_mse
            boundary = weights["boundary"] * diagnostics.boundary_mse
        total = data + momentum + continuity + boundary
        return {
            "total": total,
            "data": data,
            "momentum": momentum,
            "continuity": continuity,
            "boundary": boundary,
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
        """Return the scalar total semantic loss."""
        if y is None:
            msg = "SemanticComposedLoss requires a target tensor y."
            raise ValueError(msg)
        components = self.compute_components(pred, x=x, y=y, epoch=epoch)
        self._last_components = {name: value.detach() for name, value in components.items()}
        return components["total"]
