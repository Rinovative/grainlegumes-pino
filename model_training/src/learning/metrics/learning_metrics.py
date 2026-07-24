"""
===============================================================================
learning_metrics.py
===============================================================================
Compute task-resolved PyTorch metrics for training and evaluation.

Responsibilities:
  - Register and validate semantic metric identifiers
  - Resolve metric fields, spaces, reductions and directions
  - Accumulate dataset metrics from explicit normalized or physical views

Design principles:
  - Dataset accumulators use mathematically sufficient statistics
  - Metric implementations never own or apply normalizers
  - Semantic identifiers remain independent of metric implementation classes

Boundaries:
  - Tensor-view construction belongs to evaluation orchestration
  - Logging and persistence belong to callers
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import torch
from neuralop import H1Loss

from src import domain

MetricSpace = Literal["normalized", "physical"]
MetricReduction = Literal["sample_mean", "element_mean"]
MetricDirection = Literal["minimize", "maximize"]
_MIN_METRIC_TENSOR_RANK = 3


@dataclass(frozen=True, slots=True)
class MetricKindSpec:
    """
    Describe schema and optimization semantics for one metric identifier.

    Attributes
    ----------
    kind : str
        Canonical saved configuration identifier.
    spaces : frozenset[MetricSpace]
        Supported tensor spaces.
    reductions : frozenset[MetricReduction]
        Supported reduction semantics.
    direction : MetricDirection
        Required optimization direction.

    """

    kind: str
    spaces: frozenset[MetricSpace]
    reductions: frozenset[MetricReduction]
    direction: MetricDirection


_METRIC_KINDS = MappingProxyType(
    {
        "relative_h1": MetricKindSpec(
            kind="relative_h1",
            spaces=frozenset({"normalized"}),
            reductions=frozenset({"sample_mean"}),
            direction="minimize",
        ),
        "relative_l2": MetricKindSpec(
            kind="relative_l2",
            spaces=frozenset({"normalized"}),
            reductions=frozenset({"sample_mean"}),
            direction="minimize",
        ),
        "rmse": MetricKindSpec(
            kind="rmse",
            spaces=frozenset({"normalized", "physical"}),
            reductions=frozenset({"element_mean"}),
            direction="minimize",
        ),
    }
)


def available_metric_kinds() -> tuple[str, ...]:
    """
    Return registered semantic metric identifiers.

    Returns
    -------
    tuple[str, ...]
        Exact metric kinds accepted by the registry.

    """
    return tuple(sorted(_METRIC_KINDS))


def resolve_metric_kind(kind: str) -> MetricKindSpec:
    """
    Resolve an exact semantic metric identifier.

    Parameters
    ----------
    kind : str
        Canonical metric kind.

    Returns
    -------
    MetricKindSpec
        Immutable metric-space, reduction, and direction descriptor.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    try:
        return _METRIC_KINDS[kind]
    except KeyError as error:
        available = ", ".join(available_metric_kinds())
        msg = f"Unknown metric identifier {kind!r}. Available metrics: {available}."
        raise ValueError(msg) from error


def validate_metric_semantics(
    kind: str,
    *,
    space: str,
    reduction: str,
) -> MetricKindSpec:
    """
    Validate a metric space and reduction against its registry entry.

    Parameters
    ----------
    kind : str
        Canonical metric kind.
    space : str
        Requested normalized or physical tensor space.
    reduction : str
        Requested dataset/sample reduction identifier.

    Returns
    -------
    MetricKindSpec
        Validated semantic metric descriptor.

    Raises
    ------
    ValueError
        If the metric kind, space, or reduction is unsupported.

    """
    spec = resolve_metric_kind(kind)
    if space not in spec.spaces:
        msg = f"Metric {kind!r} does not support space {space!r}; expected one of {sorted(spec.spaces)}."
        raise ValueError(msg)
    if reduction not in spec.reductions:
        msg = f"Metric {kind!r} does not support reduction {reduction!r}; expected one of {sorted(spec.reductions)}."
        raise ValueError(msg)
    return spec


# ============================================================================
# Explicit-space dataset metric accumulators
# ============================================================================


@dataclass(frozen=True, slots=True)
class ResolvedMetric:
    """Describe one task-resolved evaluation metric and its physical unit."""

    id: str
    kind: str
    space: MetricSpace
    fields: tuple[str, ...]
    field_indices: tuple[int, ...]
    reduction: MetricReduction
    direction: MetricDirection
    unit: str
    operator_dimensionality: int


class DatasetMetric:
    """Accumulate one explicit-space dataset metric by sufficient statistics."""

    def __init__(self, definition: ResolvedMetric) -> None:
        """Store the immutable definition and clear sufficient statistics."""
        self.definition = definition
        self.reset()

    @property
    def id(self) -> str:
        """Return the configured metric identifier."""
        return self.definition.id

    @property
    def space(self) -> MetricSpace:
        """Return the tensor space this metric requires."""
        return self.definition.space

    @property
    def fields(self) -> tuple[str, ...]:
        """Return exact selected task output fields."""
        return self.definition.fields

    @property
    def unit(self) -> str:
        """Return the task-owned physical unit or dimensionless unit ``1``."""
        return self.definition.unit

    def reset(self) -> None:
        """Clear dataset sufficient statistics."""
        self._sum = 0.0
        self._count = 0

    def _validate_update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate space, shape, finiteness, and select task fields."""
        if space != self.space:
            msg = f"Metric {self.id!r} expects {self.space!r} tensors, got {space!r}."
            raise ValueError(msg)
        if pred.shape != target.shape:
            msg = f"Metric {self.id!r} prediction/target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}."
            raise ValueError(msg)
        if pred.ndim < _MIN_METRIC_TENSOR_RANK:
            msg = f"Metric {self.id!r} requires batch, channel, and spatial axes."
            raise ValueError(msg)
        if not bool(torch.isfinite(pred).all().item()) or not bool(torch.isfinite(target).all().item()):
            msg = f"Metric {self.id!r} received non-finite values in evaluation batch {batch_index}."
            raise FloatingPointError(msg)
        maximum_index = max(self.definition.field_indices)
        if pred.shape[1] <= maximum_index:
            msg = f"Metric {self.id!r} field index {maximum_index} exceeds {pred.shape[1]} channels."
            raise ValueError(msg)
        indices = torch.tensor(self.definition.field_indices, device=pred.device)
        return pred.index_select(1, indices), target.index_select(1, indices)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Accumulate one evaluation batch."""
        raise NotImplementedError

    def compute(self) -> float:
        """Finalize one dataset metric after all batches."""
        raise NotImplementedError


class RMSEMetric(DatasetMetric):
    """Accumulate global squared error and take one final square root."""

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Add squared-error sum and exact selected element count."""
        selected_pred, selected_target = self._validate_update(
            pred,
            target,
            space=space,
            batch_index=batch_index,
        )
        squared_error = (selected_pred.double() - selected_target.double()).square()
        batch_sum = float(squared_error.sum().detach().cpu().item())
        if not np.isfinite(batch_sum):
            msg = f"Metric {self.id!r} produced non-finite squared error in evaluation batch {batch_index}."
            raise FloatingPointError(msg)
        self._sum += batch_sum
        self._count += squared_error.numel()

    def compute(self) -> float:
        """Return ``sqrt(total squared error / total element count)``."""
        if self._count == 0:
            msg = f"Metric {self.id!r} cannot finalize without samples."
            raise RuntimeError(msg)
        value = float(np.sqrt(self._sum / self._count))
        if not np.isfinite(value):
            msg = f"Metric {self.id!r} finalized to a non-finite value."
            raise FloatingPointError(msg)
        return value


class RelativeL2Metric(DatasetMetric):
    """Accumulate one combined selected-field relative L2 value per sample."""

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Add finite per-sample relative L2 values."""
        selected_pred, selected_target = self._validate_update(
            pred,
            target,
            space=space,
            batch_index=batch_index,
        )
        flat_difference = (selected_pred.double() - selected_target.double()).flatten(start_dim=1)
        flat_target = selected_target.double().flatten(start_dim=1)
        values = torch.linalg.vector_norm(flat_difference, dim=1) / (torch.linalg.vector_norm(flat_target, dim=1) + 1e-8)
        self._accumulate_samples(values, batch_index=batch_index)

    def _accumulate_samples(self, values: torch.Tensor, *, batch_index: int) -> None:
        """Accumulate finite sample values with sample context on failure."""
        finite = torch.isfinite(values)
        if not bool(finite.all().item()):
            first = int((~finite).nonzero(as_tuple=False)[0, 0].item())
            msg = f"Metric {self.id!r} produced a non-finite value in evaluation batch {batch_index}, sample {first}."
            raise FloatingPointError(msg)
        self._sum += float(values.sum().detach().cpu().item())
        self._count += values.numel()

    def compute(self) -> float:
        """Return the arithmetic mean of defined per-sample values."""
        if self._count == 0:
            msg = f"Metric {self.id!r} cannot finalize without samples."
            raise RuntimeError(msg)
        value = self._sum / self._count
        if not np.isfinite(value):
            msg = f"Metric {self.id!r} finalized to a non-finite value."
            raise FloatingPointError(msg)
        return float(value)


class RelativeH1Metric(RelativeL2Metric):
    """Accumulate NeuralOp-compatible relative H1 values per sample."""

    def __init__(self, definition: ResolvedMetric) -> None:
        """Build the task-dimensional relative H1 implementation."""
        super().__init__(definition)
        self._implementation = H1Loss(
            d=definition.operator_dimensionality,
            reduction="sum",
        )

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Compute and accumulate one relative H1 value per sample."""
        selected_pred, selected_target = self._validate_update(
            pred,
            target,
            space=space,
            batch_index=batch_index,
        )
        values = torch.stack(
            [
                self._implementation(
                    selected_pred[index : index + 1],
                    selected_target[index : index + 1],
                ).reshape(())
                for index in range(selected_pred.shape[0])
            ]
        )
        self._accumulate_samples(values.double(), batch_index=batch_index)


def _resolved_metric_fields(
    raw_fields: Any,
    *,
    task_fields: tuple[str, ...],
    metric_id: str,
) -> tuple[str, ...]:
    """Return exact fields from an already resolved config metric."""
    if raw_fields == "all":
        return task_fields
    if not isinstance(raw_fields, list) or not raw_fields or not all(isinstance(field, str) for field in raw_fields):
        msg = f"Evaluation metric {metric_id!r} fields must be 'all' or a non-empty list of strings."
        raise TypeError(msg)
    fields = tuple(raw_fields)
    if len(fields) != len(set(fields)):
        msg = f"Evaluation metric {metric_id!r} contains duplicate fields: {list(fields)}."
        raise ValueError(msg)
    unknown = [field for field in fields if field not in task_fields]
    if unknown:
        msg = f"Evaluation metric {metric_id!r} references unknown output fields: {unknown}."
        raise ValueError(msg)
    return fields


def build_evaluation_metrics(config: dict[str, Any]) -> dict[str, DatasetMetric]:
    """
    Build explicit-space dataset accumulators from semantic config.

    Metric implementations do not receive or own normalizers. Physical units
    come from the resolved task field contract, and physical metrics select one
    field so incompatible units can never be silently combined.
    """
    task = domain.tasks.registry.get_task(str(config["task"]))
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        msg = "Resolved config must contain an evaluation mapping."
        raise TypeError(msg)
    raw_metrics = evaluation.get("metrics")
    if not isinstance(raw_metrics, list):
        msg = "evaluation.metrics must be a list."
        raise TypeError(msg)

    built: dict[str, DatasetMetric] = {}
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            msg = "Each evaluation metric must be a mapping."
            raise TypeError(msg)
        metric_id = str(raw_metric["id"])
        if metric_id in built:
            msg = f"Duplicate evaluation metric id {metric_id!r}."
            raise ValueError(msg)
        kind = str(raw_metric["kind"])
        space = str(raw_metric["space"])
        reduction = str(raw_metric["reduction"])
        kind_spec = validate_metric_semantics(kind, space=space, reduction=reduction)
        fields = _resolved_metric_fields(
            raw_metric["fields"],
            task_fields=task.output_names,
            metric_id=metric_id,
        )
        if space == "physical" and len(fields) != 1:
            units = sorted({task.field(field).unit for field in fields})
            msg = f"Physical metric {metric_id!r} must select exactly one field; selected units are {units}."
            raise ValueError(msg)
        direction = str(raw_metric.get("direction", kind_spec.direction))
        if direction != kind_spec.direction:
            msg = f"Metric {metric_id!r} direction {direction!r} contradicts {kind!r}."
            raise ValueError(msg)
        definition = ResolvedMetric(
            id=metric_id,
            kind=kind,
            space=cast("MetricSpace", space),
            fields=fields,
            field_indices=tuple(task.output_names.index(field) for field in fields),
            reduction=cast("MetricReduction", reduction),
            direction=cast("MetricDirection", direction),
            unit=task.field(fields[0]).unit if space == "physical" else "1",
            operator_dimensionality=task.operator_dimensionality,
        )
        if kind == "rmse":
            built[metric_id] = RMSEMetric(definition)
        elif kind == "relative_l2":
            built[metric_id] = RelativeL2Metric(definition)
        elif kind == "relative_h1":
            built[metric_id] = RelativeH1Metric(definition)
        else:
            msg = f"No dataset accumulator exists for metric identifier {kind!r}."
            raise ValueError(msg)
    return built


def reset_metrics(metrics: dict[str, DatasetMetric]) -> None:
    """Reset every configured dataset accumulator."""
    for metric in metrics.values():
        metric.reset()


def finalize_metrics(metrics: dict[str, DatasetMetric]) -> dict[str, float]:
    """Finalize every configured dataset accumulator exactly once."""
    return {metric_id: metric.compute() for metric_id, metric in metrics.items()}
