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

This module does NOT:
  - Construct normalized or physical tensor views; evaluation orchestration owns them
  - Log or persist metric results; callers own observer and storage side effects
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from src import domain

if TYPE_CHECKING:
    import torch

MetricSpace = Literal["normalized", "physical"]
MetricReduction = Literal["sample_mean", "element_mean", "field_macro_element_mean"]
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
        "macro_rmse": MetricKindSpec(
            kind="macro_rmse",
            spaces=frozenset({"normalized"}),
            reductions=frozenset({"field_macro_element_mean"}),
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
    """
    Describe one task-resolved evaluation metric and its reduction contract.

    Attributes
    ----------
    id, kind : str
        Stable config identifier and registered implementation kind.
    space : {"normalized", "physical"}
        Tensor representation required by accumulation.
    fields : tuple[str, ...]
        Exact TaskSpec outputs included in declared order.
    field_indices : tuple[int, ...]
        Corresponding channel indices in the task output tensor.
    reduction : str
        Sample, element, or field-macro sufficient-statistic reduction.
    direction : {"minimize", "maximize"}
        Selection direction when used as an objective.
    unit : str
        Task-owned physical unit or dimensionless ``1``.
    operator_dimensionality : int
        Spatial dimensionality used by derivative-aware metrics.

    """

    id: str
    kind: str
    space: MetricSpace
    fields: tuple[str, ...]
    field_indices: tuple[int, ...]
    reduction: MetricReduction
    direction: MetricDirection
    unit: str
    operator_dimensionality: int

    def __post_init__(self) -> None:
        """Reject ambiguous field declarations before tensor accumulation."""
        if not self.fields or len(self.fields) != len(set(self.fields)):
            msg = f"Metric {self.id!r} fields must be unique and non-empty."
            raise ValueError(msg)
        if len(self.field_indices) != len(self.fields):
            msg = f"Metric {self.id!r} field names and channel indices must have equal length."
            raise ValueError(msg)
        if len(self.field_indices) != len(set(self.field_indices)) or any(index < 0 for index in self.field_indices):
            msg = f"Metric {self.id!r} field indices must be unique non-negative integers."
            raise ValueError(msg)


class DatasetMetric:
    """
    Accumulate one explicit-space dataset metric by sufficient statistics.

    Implementations validate tensor space, shape ``(batch, channel, *spatial)``,
    concrete device, finiteness, and TaskSpec field selection on every update.
    Callers must reset once, update with every evaluation batch, and compute only
    after the complete dataset; batching must not change the final value.

    Parameters
    ----------
    definition : ResolvedMetric
        Immutable semantic fields, space, reduction, direction, and unit.
    device : torch.device
        Concrete device on which every update tensor must reside.

    Raises
    ------
    TypeError
        If ``device`` is not a concrete CPU or CUDA ``torch.device``.

    Notes
    -----
    Accumulators own only sufficient statistics. They never normalize tensors,
    transfer devices, persist values, or infer field/unit semantics.

    """

    def __init__(self, definition: ResolvedMetric, *, device: torch.device) -> None:
        """
        Validate device ownership and initialize empty sufficient statistics.

        Construction performs no tensor transfer or normalization; the concrete
        device becomes an invariant checked on every update.
        """
        import torch  # noqa: PLC0415

        if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
            msg = f"Metric construction requires one concrete CPU or CUDA torch.device, got {device!r}."
            raise TypeError(msg)
        self.definition = definition
        self.device = device
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
        """
        Validate one batch view and select the declared task channels.

        Prediction and target must share shape ``(batch, channel, *spatial)``,
        concrete device, requested tensor space, and finite values. Failures
        include metric and batch identity; returned tensors preserve batch and
        spatial axes while selecting fields in declaration order.
        """
        import torch  # noqa: PLC0415

        if space != self.space:
            msg = f"Metric {self.id!r} expects {self.space!r} tensors, got {space!r}."
            raise ValueError(msg)
        if pred.shape != target.shape:
            msg = f"Metric {self.id!r} prediction/target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}."
            raise ValueError(msg)
        if pred.device != self.device or target.device != self.device:
            msg = f"Metric {self.id!r} requires tensors on resolved device {self.device}, got prediction={pred.device} and target={target.device}."
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
    """
    Accumulate global squared error and take one final square root.

    The metric is ``sqrt(sum((pred-target)^2) / selected_element_count)`` over
    the complete dataset. It therefore does not average batch RMSE values.
    """

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """
        Add double-precision squared-error sum and selected element count.

        Batch RMSE is never computed. Non-finite intermediate sums raise with
        metric and batch identity before sufficient statistics are mutated.
        """
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


class MacroRMSEMetric(DatasetMetric):
    """
    Accumulate global per-field RMSE values and take their macro mean.

    Squared normalized errors and exact element counts are accumulated
    independently for every TaskSpec output field over the complete evaluation
    split. Finalization takes one global RMSE per field, then the unweighted
    arithmetic mean of those field RMSEs. This differs from pooled overall RMSE,
    whose final square root follows aggregation across fields.
    """

    def reset(self) -> None:
        """Clear independent per-field squared-error sums and element counts."""
        self._field_sums = [0.0] * len(self.fields)
        self._field_counts = [0] * len(self.fields)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """
        Add one batch to independent per-field sufficient statistics.

        Each selected channel accumulates a double-precision squared-error sum
        and exact element count; no field or batch receives an implicit weight.
        """
        selected_pred, selected_target = self._validate_update(
            pred,
            target,
            space=space,
            batch_index=batch_index,
        )
        squared_error = (selected_pred.double() - selected_target.double()).square()
        for field_index, field in enumerate(self.fields):
            field_error = squared_error[:, field_index]
            batch_sum = float(field_error.sum().detach().cpu().item())
            if not np.isfinite(batch_sum):
                msg = f"Metric {self.id!r} produced non-finite squared error for field {field!r} in evaluation batch {batch_index}."
                raise FloatingPointError(msg)
            self._field_sums[field_index] += batch_sum
            self._field_counts[field_index] += field_error.numel()

    def compute(self) -> float:
        """
        Finalize each global field RMSE and return their unweighted macro mean.

        Empty fields and non-finite field or aggregate values fail explicitly;
        fields are not pooled before their square roots are taken.
        """
        field_values: list[float] = []
        for field, squared_error_sum, element_count in zip(
            self.fields,
            self._field_sums,
            self._field_counts,
            strict=True,
        ):
            if element_count == 0:
                msg = f"Metric {self.id!r} cannot finalize field {field!r} without evaluation elements."
                raise RuntimeError(msg)
            field_value = float(np.sqrt(squared_error_sum / element_count))
            if not np.isfinite(field_value):
                msg = f"Metric {self.id!r} finalized field {field!r} to a non-finite value."
                raise FloatingPointError(msg)
            field_values.append(field_value)
        value = float(np.mean(field_values))
        if not np.isfinite(value):
            msg = f"Metric {self.id!r} finalized to a non-finite field-macro value."
            raise FloatingPointError(msg)
        return value


class RelativeL2Metric(DatasetMetric):
    """
    Accumulate one combined selected-field relative L2 value per sample.

    Each sample contributes ``||pred-target||_2 / (||target||_2 + 1e-8)`` after
    named-channel selection; finalization takes the arithmetic sample mean.
    """

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """
        Compute and accumulate one combined selected-field relative L2 per sample.

        Batch and selected channel/spatial axes are flattened separately, using
        the maintained ``1e-8`` target-norm stabilizer before sample-mean accumulation.
        """
        selected_pred, selected_target = self._validate_update(
            pred,
            target,
            space=space,
            batch_index=batch_index,
        )
        import torch  # noqa: PLC0415

        flat_difference = (selected_pred.double() - selected_target.double()).flatten(start_dim=1)
        flat_target = selected_target.double().flatten(start_dim=1)
        values = torch.linalg.vector_norm(flat_difference, dim=1) / (torch.linalg.vector_norm(flat_target, dim=1) + 1e-8)
        self._accumulate_samples(values, batch_index=batch_index)

    def _accumulate_samples(self, values: torch.Tensor, *, batch_index: int) -> None:
        """Accumulate finite sample values with sample context on failure."""
        import torch  # noqa: PLC0415

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
    """
    Accumulate NeuralOp-compatible relative H1 values per sample.

    The registered NeuralOp H1 implementation uses the TaskSpec operator
    dimensionality and is evaluated independently per sample before sample-mean
    accumulation, preserving the declared ``sample_mean`` reduction.
    """

    def __init__(self, definition: ResolvedMetric, *, device: torch.device) -> None:
        """Build the task-dimensional relative H1 implementation."""
        from neuralop import H1Loss  # noqa: PLC0415

        super().__init__(definition, device=device)
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
        """
        Compute and accumulate one task-dimensional relative H1 value per sample.

        Samples are evaluated independently with NeuralOp reduction ``sum`` so
        final accumulation preserves the declared dataset ``sample_mean`` contract.
        """
        selected_pred, selected_target = self._validate_update(
            pred,
            target,
            space=space,
            batch_index=batch_index,
        )
        import torch  # noqa: PLC0415

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
    """
    Return the exact ordered fields from an already resolved metric declaration.

    ``all`` expands to TaskSpec output order. Explicit lists must be non-empty,
    unique, and task-known; failures retain the metric ID for config diagnostics.
    """
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


def build_evaluation_metrics(config: dict[str, Any], *, device: torch.device) -> dict[str, DatasetMetric]:
    """
    Build explicit-space dataset accumulators from semantic config.

    Metric implementations do not receive or own normalizers. Physical units
    come from the resolved task field contract, and physical metrics select one
    field so incompatible units can never be silently combined.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved task and evaluation configuration.
    device : torch.device
        Concrete device required by all accumulator updates.

    Returns
    -------
    dict[str, DatasetMetric]
        Metric-ID keyed fresh accumulators in declaration order.

    Raises
    ------
    TypeError
        If required resolved sections or metric entries have invalid types.
    ValueError
        If IDs, field selections, units, directions, or semantic combinations
        contradict the registered task and metric contracts.

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
        if kind == "macro_rmse" and fields != task.output_names:
            msg = f"Metric {metric_id!r} with kind 'macro_rmse' must select every TaskSpec output field in declared order: {list(task.output_names)}."
            raise ValueError(msg)
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
            built[metric_id] = RMSEMetric(definition, device=device)
        elif kind == "macro_rmse":
            built[metric_id] = MacroRMSEMetric(definition, device=device)
        elif kind == "relative_l2":
            built[metric_id] = RelativeL2Metric(definition, device=device)
        elif kind == "relative_h1":
            built[metric_id] = RelativeH1Metric(definition, device=device)
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
