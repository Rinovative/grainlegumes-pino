"""
===============================================================================
learning_losses_factory.py
===============================================================================
Build semantic supervised and physics-informed loss compositions.

Responsibilities:
  - Resolve semantic supervised data-loss identifiers
  - Build task-dimensional data losses with explicit reduction and weighting
  - Build one named composition interface for supervised and physics training

Design principles:
  - Resolved semantic identifiers select implementations without exposing class names
  - Construction is explicit, task-aware, and placed on the caller-selected device
  - One composition interface serves supervised and physics-informed training

This module does NOT:
  - Define task contracts or allowed physics choices. ``domain.tasks`` owns them
  - Implement derivatives, equations, residuals, or diagnostics. ``domain.physics`` does
  - Define evaluation metrics or dataset aggregation. ``learning.metrics`` does
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from src import domain

if TYPE_CHECKING:
    import torch
    from torch import nn

    from . import learning_losses_pino as pino


@dataclass(frozen=True, slots=True)
class DataLossKindSpec:
    """
    Describe one immutable supervised-loss registry entry.

    Attributes
    ----------
    kind : str
        Canonical configuration identifier.
    spaces : frozenset[str]
        Explicit tensor spaces admitted for this loss.

    """

    kind: str
    spaces: frozenset[str]


_DATA_LOSS_KINDS = MappingProxyType(
    {
        "relative_h1": DataLossKindSpec("relative_h1", frozenset({"normalized"})),
        "relative_l2": DataLossKindSpec("relative_l2", frozenset({"normalized"})),
    }
)


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
        msg = f"Loss {kind!r} does not support space {space!r}. Expected one of {sorted(spec.spaces)}."
        raise ValueError(msg)
    return spec


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
    from neuralop import H1Loss, LpLoss  # noqa: PLC0415

    from . import learning_losses_pino as pino  # noqa: PLC0415

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
    return pino.SemanticDataLoss(implementation, weight=weight)


def _resolved_weight(config: dict[str, Any], name: str) -> pino.LinearWarmup:
    """
    Build one validated linear warmup from a resolved physics-weight node.

    Missing mappings, unknown warmup kinds, and malformed scalar values fail at
    the semantic weight name instead of silently producing a schedule. The
    returned object owns non-negative target and epoch validation.
    """
    from . import learning_losses_pino as pino  # noqa: PLC0415

    weight = config.get(name)
    if not isinstance(weight, dict):
        msg = f"loss.physics.{name} must be a mapping."
        raise TypeError(msg)
    warmup = weight.get("warmup")
    if not isinstance(warmup, dict):
        msg = f"loss.physics.{name}.warmup must be a mapping."
        raise TypeError(msg)
    if warmup.get("kind") != "linear":
        msg = f"Unknown warmup identifier {warmup.get('kind')!r}. Expected 'linear'."
        raise ValueError(msg)
    return pino.LinearWarmup(
        target=float(weight["target"]),
        epochs=int(warmup["epochs"]),
    )


def build_training_loss(config: dict[str, Any], *, device: torch.device) -> pino.SemanticComposedLoss:
    """
    Build the unified semantic training loss from resolved configuration.

    Supervised and physics-informed configurations return the same type and
    named component interface. The task contract selects the equation, allowed
    continuity formulations, default continuity, and boundary formulation. The
    effective experiment config selects continuity, numerical derivatives,
    explicit weights, and warmup. The caller supplies the already resolved
    concrete runtime device. This factory never performs availability fallback.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved task, loss, and physics configuration.
    device : torch.device
        Concrete CPU or indexed CUDA device selected by the service boundary.

    Returns
    -------
    SemanticComposedLoss
        Device-bound composition with a stable named-component interface.

    Raises
    ------
    TypeError
        If the device or required resolved mappings have invalid types.
    ValueError
        If task physics, continuity, derivatives, weights, or crop settings
        contradict their registered semantic contracts.

    """
    import torch  # noqa: PLC0415

    from . import learning_losses_pino as pino  # noqa: PLC0415

    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Loss construction requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    task = domain.tasks.registry.get_task(str(config["task"]))
    task_physics = domain.tasks.registry.resolve_physics(task.physics.kind)
    loss_config = config.get("loss")
    if not isinstance(loss_config, dict):
        msg = "Resolved config must contain a loss mapping."
        raise TypeError(msg)
    data_config = loss_config.get("data")
    physics_config = loss_config.get("physics")
    if not isinstance(data_config, dict) or not isinstance(physics_config, dict):
        msg = "Resolved loss config must contain data and physics mappings."
        raise TypeError(msg)

    continuity = physics_config.get("continuity")
    if not isinstance(continuity, str) or not continuity:
        msg = "loss.physics.continuity must be a non-empty semantic identifier."
        raise TypeError(msg)
    if continuity not in task_physics.allowed_continuities:
        available = ", ".join(task_physics.allowed_continuities)
        msg = f"Unknown continuity identifier {continuity!r} at loss.physics.continuity. Available continuity formulations: {available}."
        raise ValueError(msg)

    derivatives_config = physics_config.get("derivatives")
    if not isinstance(derivatives_config, dict):
        msg = "loss.physics.derivatives must be a mapping."
        raise TypeError(msg)
    derivative_kind, extension = domain.physics.contracts.validate_derivative_kind(
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
    loss = pino.SemanticComposedLoss(
        data_loss=data_loss,
        data_weight=float(data_config["weight"]),
        physics_enabled=bool(physics_config["enabled"]),
        physics_kind=task_physics.kind,
        input_fields=task.input_names,
        output_fields=task.output_names,
        continuity=continuity,
        boundary=task_physics.boundary,
        derivatives=derivatives,
        residual_weight=_resolved_weight(physics_config, "residual_weight"),
        boundary_weight=_resolved_weight(physics_config, "boundary_weight"),
        interior_crop=int(physics_config["interior_crop"]),
    )
    return loss.to(device)
