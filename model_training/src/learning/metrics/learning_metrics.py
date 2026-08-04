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
  - Construct normalized or physical tensor views. Evaluation orchestration owns them
  - Log or persist metric results. Callers own observer and storage side effects
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from src import domain

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

MetricSpace = Literal["normalized", "physical"]
MetricReduction = Literal[
    "sample_mean",
    "element_mean",
    "group_element_mean",
    "group_macro_element_mean",
    "vector_element_mean",
]
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
        "group_macro_rmse": MetricKindSpec(
            kind="group_macro_rmse",
            spaces=frozenset({"physical"}),
            reductions=frozenset({"group_macro_element_mean"}),
            direction="minimize",
        ),
        "group_rmse": MetricKindSpec(
            kind="group_rmse",
            spaces=frozenset({"physical"}),
            reductions=frozenset({"group_element_mean"}),
            direction="minimize",
        ),
        "vector_rmse": MetricKindSpec(
            kind="vector_rmse",
            spaces=frozenset({"physical"}),
            reductions=frozenset({"vector_element_mean"}),
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
        msg = f"Metric {kind!r} does not support space {space!r}. Expected one of {sorted(spec.spaces)}."
        raise ValueError(msg)
    if reduction not in spec.reductions:
        msg = f"Metric {kind!r} does not support reduction {reduction!r}. Expected one of {sorted(spec.reductions)}."
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
        Sample, element, vector, or physical-group sufficient-statistic reduction.
    direction : {"minimize", "maximize"}
        Selection direction when used as an objective.
    unit : str
        Task-owned physical unit or dimensionless ``1``.
    operator_dimensionality : int
        Spatial dimensionality used by derivative-aware metrics.
    groups : tuple[domain.tasks.spec.OutputGroupSpec, ...]
        Task-owned physical groups selected by group-aware metrics.
    field_standard_deviations : tuple[float, ...]
        Raw train-fitted output standard deviations aligned with ``fields``.

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
    groups: tuple[domain.tasks.spec.OutputGroupSpec, ...]
    field_standard_deviations: tuple[float, ...]

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
        if self.field_standard_deviations and len(self.field_standard_deviations) != len(self.fields):
            msg = f"Metric {self.id!r} standard deviations must align with selected fields."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GroupRMSEValues:
    """
    Hold physical and normalized output-group RMSE values.

    Attributes
    ----------
    physical : Mapping[str, float]
        Root-sum-square physical component RMSE for each group.
    normalized : Mapping[str, float]
        Physical group error divided by its shared train-fitted vector scale.
    normalized_macro : float
        Equal arithmetic mean of normalized group errors.

    """

    physical: Mapping[str, float]
    normalized: Mapping[str, float]
    normalized_macro: float


def finalize_group_rmse_statistics(
    groups: tuple[domain.tasks.spec.OutputGroupSpec, ...],
    *,
    squared_error_sums: Mapping[str, float],
    element_counts: Mapping[str, int],
    train_standard_deviations: Mapping[str, float],
) -> GroupRMSEValues:
    """
    Finalize global physical and shared-scale normalized group errors.

    For each physical output group, this function first sums fieldwise physical
    mean squared errors. Its square root is the physical vector error. Dividing
    the same sum by the sum of raw train-fitted field variances before the square
    root gives the dimensionless normalized group error. The macro value gives
    every physical group equal weight and is lower-is-better, not a percentage.

    Parameters
    ----------
    groups : tuple[domain.tasks.spec.OutputGroupSpec, ...]
        Ordered task-owned output groups.
    squared_error_sums : Mapping[str, float]
        Global physical squared-error sum for every grouped field.
    element_counts : Mapping[str, int]
        Validated physical element count for every grouped field.
    train_standard_deviations : Mapping[str, float]
        Raw train-fitted output standard deviation for every grouped field.

    Returns
    -------
    GroupRMSEValues
        Physical and normalized values by group plus their normalized macro mean.

    Raises
    ------
    TypeError
        If a squared-error sum or train-fitted scale is not a real number.
    ValueError
        If groups or field statistics are missing, duplicated, negative, empty,
        zero-scale, or non-finite.

    """
    if not groups:
        msg = "Group RMSE finalization requires at least one task-owned output group."
        raise ValueError(msg)
    grouped_fields = tuple(field for group in groups for field in group.fields)
    if len(grouped_fields) != len(set(grouped_fields)):
        msg = "Group RMSE finalization received duplicate field membership."
        raise ValueError(msg)

    physical: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for group in groups:
        physical_mse_sum = 0.0
        scale_squared_sum = 0.0
        for field in group.fields:
            try:
                squared_error_sum = squared_error_sums[field]
                element_count = element_counts[field]
                standard_deviation = train_standard_deviations[field]
            except KeyError as error:
                msg = f"Group {group.id!r} is missing sufficient statistics for field {field!r}."
                raise ValueError(msg) from error
            if isinstance(squared_error_sum, bool) or not isinstance(squared_error_sum, Real):
                msg = f"Field {field!r} squared-error sum must be a real number."
                raise TypeError(msg)
            squared_error_value = float(squared_error_sum)
            if not np.isfinite(squared_error_value) or squared_error_value < 0.0:
                msg = f"Field {field!r} squared-error sum must be finite and non-negative."
                raise ValueError(msg)
            if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0:
                msg = f"Field {field!r} element count must be a positive integer."
                raise ValueError(msg)
            if isinstance(standard_deviation, bool) or not isinstance(standard_deviation, Real):
                msg = f"Field {field!r} train-fitted standard deviation must be a real number."
                raise TypeError(msg)
            scale = float(standard_deviation)
            if not np.isfinite(scale) or scale <= 0.0:
                msg = f"Field {field!r} train-fitted standard deviation must be finite and positive."
                raise ValueError(msg)
            physical_mse_sum += squared_error_value / element_count
            scale_squared_sum += scale * scale

        physical_value = float(np.sqrt(physical_mse_sum))
        normalized_value = float(np.sqrt(physical_mse_sum / scale_squared_sum))
        if not np.isfinite(physical_value) or not np.isfinite(normalized_value):
            msg = f"Group {group.id!r} finalized to a non-finite RMSE value."
            raise ValueError(msg)
        physical[group.id] = physical_value
        normalized[group.id] = normalized_value

    normalized_macro = float(np.mean(tuple(normalized.values())))
    if not np.isfinite(normalized_macro):
        msg = "Normalized group-macro RMSE finalized to a non-finite value."
        raise ValueError(msg)
    return GroupRMSEValues(
        physical=MappingProxyType(physical),
        normalized=MappingProxyType(normalized),
        normalized_macro=normalized_macro,
    )


class DatasetMetric:
    """
    Accumulate one explicit-space dataset metric by sufficient statistics.

    Implementations validate tensor space, shape ``(batch, channel, *spatial)``,
    concrete device, finiteness, and TaskSpec field selection on every update.
    Callers must reset once, update with every evaluation batch, and compute only
    after the complete dataset. Batching must not change the final value.

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

        Construction performs no tensor transfer or normalization. The concrete
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
        include metric and batch identity. Returned tensors preserve batch and
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


class _PhysicalGroupSSEMetric(DatasetMetric):
    """Accumulate physical per-field SSE/count for task-owned output groups."""

    def reset(self) -> None:
        """Clear physical per-field squared-error sums and element counts."""
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
        Add one physical batch to independent per-field sufficient statistics.

        Every selected component retains its exact global element count. Group
        roots and weighting are deferred until complete-dataset finalization.
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

    def _finalize_groups(self) -> GroupRMSEValues:
        """Return shared physical and normalized group values from global state."""
        return finalize_group_rmse_statistics(
            self.definition.groups,
            squared_error_sums=dict(zip(self.fields, self._field_sums, strict=True)),
            element_counts=dict(zip(self.fields, self._field_counts, strict=True)),
            train_standard_deviations=dict(
                zip(
                    self.fields,
                    self.definition.field_standard_deviations,
                    strict=True,
                )
            ),
        )


class GroupMacroRMSEMetric(_PhysicalGroupSSEMetric):
    """
    Compute the equal macro mean over normalized physical output groups.

    Each group combines physical component mean squared errors before one square
    root and uses the root-sum-square of raw train-fitted component standard
    deviations as its shared scale. Group count, not stored channel count,
    determines final objective weighting.
    """

    def compute(self) -> float:
        """Return the equal mean of complete-dataset normalized group errors."""
        return self._finalize_groups().normalized_macro


class GroupRMSEMetric(_PhysicalGroupSSEMetric):
    """Compute one dimensionless physical-group error with a shared train scale."""

    def compute(self) -> float:
        """Return one complete-dataset normalized group error."""
        values = self._finalize_groups().normalized
        if len(values) != 1:
            msg = f"Metric {self.id!r} requires exactly one output group."
            raise RuntimeError(msg)
        return next(iter(values.values()))


class VectorRMSEMetric(_PhysicalGroupSSEMetric):
    """Compute one physical vector RMSE from component mean squared errors."""

    def compute(self) -> float:
        """Return the root-sum-square physical component RMSE for one group."""
        values = self._finalize_groups().physical
        if len(values) != 1:
            msg = f"Metric {self.id!r} requires exactly one output group."
            raise RuntimeError(msg)
        return next(iter(values.values()))


class RelativeL2Metric(DatasetMetric):
    """
    Accumulate one combined selected-field relative L2 value per sample.

    Each sample contributes ``||pred-target||_2 / (||target||_2 + 1e-8)`` after
    named-channel selection. Finalization takes the arithmetic sample mean.
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
    unique, and task-known. Failures retain the metric ID for config diagnostics.
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


def _resolved_output_standard_deviations(
    raw_scales: torch.Tensor | None,
    *,
    task_fields: tuple[str, ...],
) -> dict[str, float]:
    """Return raw fitted scale tensor values aligned with TaskSpec output order."""
    if raw_scales is None:
        return {}
    import torch  # noqa: PLC0415

    if not isinstance(raw_scales, torch.Tensor):
        msg = "Output standard deviations must be the fitted output-normalizer tensor."
        raise TypeError(msg)
    flattened = raw_scales.detach().reshape(-1).cpu()
    if flattened.numel() != len(task_fields):
        msg = f"Output standard deviations must contain {len(task_fields)} TaskSpec channels, got {flattened.numel()}."
        raise ValueError(msg)
    resolved: dict[str, float] = {}
    for index, field in enumerate(task_fields):
        scale = float(flattened[index].item())
        if not np.isfinite(scale) or scale <= 0.0:
            msg = f"Output field {field!r} train-fitted standard deviation must be finite and positive."
            raise ValueError(msg)
        resolved[field] = scale
    return resolved


def _resolved_output_groups(
    task: domain.tasks.spec.TaskSpec,
    *,
    kind: str,
    fields: tuple[str, ...],
    metric_id: str,
) -> tuple[domain.tasks.spec.OutputGroupSpec, ...]:
    """Bind one group-aware metric to exact task-owned output groups."""
    if kind not in {"group_macro_rmse", "group_rmse", "vector_rmse"}:
        return ()
    if not task.output_groups:
        msg = f"Metric {metric_id!r} requires task-owned output groups."
        raise ValueError(msg)
    if kind == "group_macro_rmse":
        if fields != task.output_names:
            msg = f"Metric {metric_id!r} must select every TaskSpec output field in declared order: {list(task.output_names)}."
            raise ValueError(msg)
        return task.output_groups
    matches = tuple(group for group in task.output_groups if group.fields == fields)
    if len(matches) != 1:
        available = {group.id: list(group.fields) for group in task.output_groups}
        msg = f"Metric {metric_id!r} fields must match one complete task output group. Available groups: {available}."
        raise ValueError(msg)
    return matches


def build_evaluation_metrics(
    config: dict[str, Any],
    *,
    device: torch.device,
    output_standard_deviations: torch.Tensor | None = None,
) -> dict[str, DatasetMetric]:
    """
    Build explicit-space dataset accumulators from semantic config.

    Metric implementations do not receive or own normalizers. Group-aware
    metrics receive only raw train-fitted output standard deviations and consume
    physical prediction/target views. Physical group membership comes from the
    resolved task contract rather than metric IDs or field-name conditionals.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved task and evaluation configuration.
    device : torch.device
        Concrete device required by all accumulator updates.
    output_standard_deviations : torch.Tensor | None, optional
        Raw fitted output-normalizer standard-deviation tensor in TaskSpec output
        order. Required when any configured metric uses physical output groups.

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
    resolved_scales = _resolved_output_standard_deviations(
        output_standard_deviations,
        task_fields=task.output_names,
    )

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
        groups = _resolved_output_groups(
            task,
            kind=kind,
            fields=fields,
            metric_id=metric_id,
        )
        if groups and not resolved_scales:
            msg = f"Metric {metric_id!r} requires raw train-fitted output standard deviations."
            raise ValueError(msg)
        if space == "physical" and len(fields) != 1 and not groups:
            selected_units = sorted({task.field(field).unit for field in fields})
            msg = f"Physical metric {metric_id!r} must select exactly one field. Selected units are {selected_units}."
            raise ValueError(msg)
        if kind == "vector_rmse":
            vector_units = {task.field(field).unit for field in fields}
            if len(vector_units) != 1:
                msg = f"Physical vector metric {metric_id!r} fields must share one unit, got {sorted(vector_units)}."
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
            unit=task.field(fields[0]).unit if kind == "vector_rmse" else (task.field(fields[0]).unit if space == "physical" and not groups else "1"),
            operator_dimensionality=task.operator_dimensionality,
            groups=groups,
            field_standard_deviations=tuple(resolved_scales[field] for field in fields) if groups else (),
        )
        if kind == "rmse":
            built[metric_id] = RMSEMetric(definition, device=device)
        elif kind == "group_macro_rmse":
            built[metric_id] = GroupMacroRMSEMetric(definition, device=device)
        elif kind == "group_rmse":
            built[metric_id] = GroupRMSEMetric(definition, device=device)
        elif kind == "vector_rmse":
            built[metric_id] = VectorRMSEMetric(definition, device=device)
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
