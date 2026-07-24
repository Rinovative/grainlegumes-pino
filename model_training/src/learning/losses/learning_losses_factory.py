"""
===============================================================================
learning_losses_factory.py
===============================================================================
Build semantic supervised and physics-informed loss compositions.

Responsibilities:
  - Resolve semantic data-loss, derivative, and task-physics identifiers
  - Build task-dimensional data losses with explicit reduction and weighting
  - Build one named composition interface for supervised and physics training
  - Delegate evaluation metric construction to learning.metrics

Boundaries:
  - Task contracts and physics selection belong to domain.tasks
  - Derivatives, equations, residuals, and diagnostics belong to domain.physics
  - Metric definitions and dataset aggregation belong to learning.metrics
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neuralop import H1Loss, LpLoss
from torch import Tensor, nn

from src import domain
from src.learning.metrics import learning_metrics

from . import learning_losses_pino as pino


@dataclass(frozen=True, slots=True)
class DataLossKindSpec:
    """Describe one semantic supervised loss identifier and valid spaces."""

    kind: str
    spaces: frozenset[str]


@dataclass(frozen=True, slots=True)
class PhysicsLossImplementation:
    """Describe one task-physics identifier backed by domain diagnostics."""

    kind: str
    evaluator: domain.physics.brinkman.PhysicsEvaluator


_DATA_LOSS_KINDS = MappingProxyType(
    {
        "relative_h1": DataLossKindSpec("relative_h1", frozenset({"normalized"})),
        "relative_l2": DataLossKindSpec("relative_l2", frozenset({"normalized"})),
    }
)
_PHYSICS_LOSS_IMPLEMENTATIONS = MappingProxyType(
    {
        domain.physics.brinkman.STEADY_BRINKMAN_KIND: PhysicsLossImplementation(
            kind=domain.physics.brinkman.STEADY_BRINKMAN_KIND,
            evaluator=domain.physics.brinkman.resolve_physics_evaluator(domain.physics.brinkman.STEADY_BRINKMAN_KIND),
        )
    }
)
_DERIVATIVE_EXTENSIONS = frozenset({"none", "reflect"})


class SemanticDataLoss(nn.Module):
    """Wrap a NeuralOp relative norm with explicit sample-mean reduction."""

    def __init__(self, implementation: Any, *, weight: float) -> None:
        """Store one callable relative norm and validate its scalar weight."""
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


def available_data_loss_kinds() -> tuple[str, ...]:
    """Return registered semantic supervised loss identifiers."""
    return tuple(sorted(_DATA_LOSS_KINDS))


def resolve_data_loss_kind(kind: str) -> DataLossKindSpec:
    """Resolve an exact semantic supervised loss identifier."""
    try:
        return _DATA_LOSS_KINDS[kind]
    except KeyError as error:
        available = ", ".join(available_data_loss_kinds())
        msg = f"Unknown loss identifier {kind!r}. Available losses: {available}."
        raise ValueError(msg) from error


def validate_data_loss_semantics(kind: str, *, space: str) -> DataLossKindSpec:
    """Validate the selected data-loss tensor space."""
    spec = resolve_data_loss_kind(kind)
    if space not in spec.spaces:
        msg = f"Loss {kind!r} does not support space {space!r}; expected one of {sorted(spec.spaces)}."
        raise ValueError(msg)
    return spec


def available_physics_loss_kinds() -> tuple[str, ...]:
    """Return task physics identifiers with semantic loss adapters."""
    return tuple(sorted(_PHYSICS_LOSS_IMPLEMENTATIONS))


def resolve_physics_loss_implementation(kind: str) -> PhysicsLossImplementation:
    """Resolve one task-selected domain physics evaluator for loss composition."""
    try:
        return _PHYSICS_LOSS_IMPLEMENTATIONS[kind]
    except KeyError as error:
        available = ", ".join(available_physics_loss_kinds())
        msg = f"Unknown physics loss identifier {kind!r}. Available physics losses: {available}."
        raise ValueError(msg) from error


def available_derivative_kinds() -> tuple[str, ...]:
    """Return semantic numerical derivative identifiers."""
    return domain.physics.derivatives.available_derivative_kinds()


def resolve_derivative_kind(kind: str, *, extension: str) -> tuple[str, str]:
    """Validate a derivative kind/extension combination without constructing it."""
    if kind not in available_derivative_kinds():
        available = ", ".join(available_derivative_kinds())
        msg = f"Unknown derivative identifier {kind!r}. Available derivatives: {available}."
        raise ValueError(msg)
    if extension not in _DERIVATIVE_EXTENSIONS:
        available = ", ".join(sorted(_DERIVATIVE_EXTENSIONS))
        msg = f"Unknown derivative extension {extension!r}. Available extensions: {available}."
        raise ValueError(msg)
    if kind == "physical" and extension != "none":
        msg = "Physical derivatives require extension 'none'."
        raise ValueError(msg)
    return kind, extension


def build_data_loss(
    kind: str,
    *,
    space: str,
    operator_dimensionality: int,
    weight: float = 1.0,
) -> nn.Module:
    """
    Build one semantic task-dimensional supervised loss.

    Parameters
    ----------
    kind : str
        ``"relative_h1"`` or ``"relative_l2"``.
    space : str
        Explicit tensor space.
    operator_dimensionality : int
        Task-owned number of spatial operator axes.
    weight : float, optional
        Explicit scalar component weight.

    Returns
    -------
    torch.nn.Module
        Sample-mean relative loss module.

    """
    validate_data_loss_semantics(kind, space=space)
    if operator_dimensionality <= 0:
        msg = f"operator_dimensionality must be positive, got {operator_dimensionality}."
        raise ValueError(msg)
    if kind == "relative_h1":
        implementation: Any = H1Loss(d=operator_dimensionality, reduction="mean")
    elif kind == "relative_l2":
        implementation = LpLoss(d=operator_dimensionality, p=2, reduction="mean")
    else:
        msg = f"No implementation registered for loss identifier {kind!r}."
        raise ValueError(msg)
    return SemanticDataLoss(implementation, weight=weight)


def _resolved_weight(config: dict[str, Any], name: str) -> pino.LinearWarmup:
    """Build one explicit linear warmup from resolved semantic config."""
    weight = config.get(name)
    if not isinstance(weight, dict):
        msg = f"loss.physics.{name} must be a mapping."
        raise TypeError(msg)
    warmup = weight.get("warmup")
    if not isinstance(warmup, dict):
        msg = f"loss.physics.{name}.warmup must be a mapping."
        raise TypeError(msg)
    if warmup.get("kind") != "linear":
        msg = f"Unknown warmup identifier {warmup.get('kind')!r}; expected 'linear'."
        raise ValueError(msg)
    return pino.LinearWarmup(
        target=float(weight["target"]),
        epochs=int(warmup["epochs"]),
    )


def build_training_loss(config: dict[str, Any]) -> pino.SemanticComposedLoss:
    """
    Build the unified semantic training loss from resolved configuration.

    Supervised and physics-informed configurations return the same type and
    named component interface. The task contract selects the equation,
    continuity, and boundary formulations; YAML selects numerical derivatives,
    explicit weights, and warmup.
    """
    task = domain.tasks.registry.get_task(str(config["task"]))
    task_physics = domain.tasks.registry.resolve_physics(task.physics.kind)
    implementation = resolve_physics_loss_implementation(task_physics.kind)
    if implementation.evaluator is not domain.physics.brinkman.resolve_physics_evaluator(task_physics.kind):
        msg = f"Physics loss registry drift for task physics {task_physics.kind!r}."
        raise RuntimeError(msg)

    loss_config = config.get("loss")
    if not isinstance(loss_config, dict):
        msg = "Resolved config must contain a loss mapping."
        raise TypeError(msg)
    data_config = loss_config.get("data")
    physics_config = loss_config.get("physics")
    if not isinstance(data_config, dict) or not isinstance(physics_config, dict):
        msg = "Resolved loss config must contain data and physics mappings."
        raise TypeError(msg)

    derivatives_config = physics_config.get("derivatives")
    if not isinstance(derivatives_config, dict):
        msg = "loss.physics.derivatives must be a mapping."
        raise TypeError(msg)
    derivative_kind, extension = resolve_derivative_kind(
        str(derivatives_config["kind"]),
        extension=str(derivatives_config["extension"]),
    )
    derivatives = domain.physics.derivatives.build_derivative_operator(
        derivative_kind,
        extension=extension,
    )
    data_loss = build_data_loss(
        str(data_config["kind"]),
        space=str(data_config["space"]),
        operator_dimensionality=task.operator_dimensionality,
        weight=1.0,
    )
    return pino.SemanticComposedLoss(
        data_loss=data_loss,
        data_weight=float(data_config["weight"]),
        physics_enabled=bool(physics_config["enabled"]),
        physics_kind=task_physics.kind,
        input_fields=task.input_names,
        output_fields=task.output_names,
        continuity=task_physics.continuity,
        boundary=task_physics.boundary,
        derivatives=derivatives,
        residual_weight=_resolved_weight(physics_config, "residual_weight"),
        boundary_weight=_resolved_weight(physics_config, "boundary_weight"),
        interior_crop=int(physics_config["interior_crop"]),
    )


def build_eval_metrics(config: dict[str, Any]) -> dict[str, learning_metrics.DatasetMetric]:
    """Delegate semantic evaluation metric construction to its owner."""
    return learning_metrics.build_evaluation_metrics(config)
