# ruff: noqa: S101
"""
Protect named field selection, normalized/physical tensor routing, and metric units.

Synthetic processors show physical RMSE uses inverse-transformed channels and
TaskSpec units while incompatible mixed-unit aggregates are rejected. Exact
macro sufficient-statistic algebra is covered by ``test_metric_reduction``.
these fixtures do not model training normalization quality.
"""

from __future__ import annotations

import copy

import pytest
import torch
from src import domain, experiments, learning
from support import configs

_TASK = domain.tasks.registry.get_task("steady_flow")
_GROUP_OBJECTIVE = next(metric for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_NORMALIZED_AGGREGATE = next(
    metric for metric in _TASK.default_metrics if metric.kind == "rmse" and metric.space == "normalized" and metric.fields == _TASK.output_names
)
_PHYSICAL_FIELD_METRICS = tuple(
    metric for metric in _TASK.default_metrics if metric.kind == "rmse" and metric.space == "physical" and len(metric.fields) == 1
)


class AffineNormalizer:
    """
    Apply a scalar affine transform for observable metric-space checks.

    Parameters
    ----------
    mean : float
        Shared physical offset for every synthetic output channel.
    standard_deviation : float
        Shared scale. Tests use a positive value and do not model fitted statistics.

    """

    def __init__(self, mean: float, standard_deviation: float) -> None:
        """Store affine normalization statistics."""
        self.mean = mean
        self.standard_deviation = standard_deviation

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize one tensor."""
        return (tensor - self.mean) / self.standard_deviation

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse-normalize one tensor."""
        return tensor * self.standard_deviation + self.mean


class SyntheticProcessor:
    """
    Provide the minimal evaluation data-processor surface.

    The helper preserves physical targets and exposes only an output normalizer.
    it is not a production preprocessor and owns no learned or serialized state.
    """

    def __init__(self, normalizer: AffineNormalizer) -> None:
        """Store the synthetic output normalizer."""
        self.out_normalizer = normalizer

    def eval(self) -> None:
        """Enter evaluation mode."""

    def preprocess(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preserve the physical target in evaluation mode."""
        return {"x": batch["x"], "y": batch["y"]}


class UnitErrorModel(torch.nn.Module):
    """Return a unit normalized error for every TaskSpec output channel."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return task-derived output channels of normalized ones."""
        return torch.ones(
            (inputs.shape[0], _TASK.out_channels, *inputs.shape[-2:]),
            device=inputs.device,
        )


def _metrics(*metric_ids: str) -> dict[str, learning.metrics.metrics.DatasetMetric]:
    """Build a selected subset of resolved default metrics."""
    config = experiments.config.loader.resolve_config(configs.direct_config())
    selected = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in metric_ids]
    config["evaluation"]["metrics"] = selected
    return learning.metrics.metrics.build_evaluation_metrics(
        config,
        device=torch.device("cpu"),
        output_standard_deviations=torch.full((_TASK.out_channels,), 2.0),
    )


def test_normalized_and_physical_rmse_are_space_correct_once() -> None:
    """
    Evaluate unit normalized error through a processor with physical scale two.

    Normalized metrics must remain one and each physical field RMSE must become
    two exactly once, protecting against missing or duplicate inverse transforms.
    """
    metrics = _metrics(
        _GROUP_OBJECTIVE.id,
        _NORMALIZED_AGGREGATE.id,
        *(metric.id for metric in _PHYSICAL_FIELD_METRICS),
    )
    normalizer = AffineNormalizer(mean=10.0, standard_deviation=2.0)
    processor = SyntheticProcessor(normalizer)
    raw_batch = {
        "x": torch.zeros((2, _TASK.in_channels, 3, 4)),
        "y": torch.full((2, _TASK.out_channels, 3, 4), 10.0),
    }
    values = learning.training.loop.eval_one_epoch(
        UnitErrorModel(),
        [raw_batch],  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
        processor,
    )

    assert values[_GROUP_OBJECTIVE.id] == pytest.approx(1.0)
    assert values[_NORMALIZED_AGGREGATE.id] == pytest.approx(1.0)
    assert {metric.id: values[metric.id] for metric in _PHYSICAL_FIELD_METRICS} == pytest.approx(
        {metric.id: 2.0 for metric in _PHYSICAL_FIELD_METRICS}
    )


def test_physical_units_and_named_channel_selection() -> None:
    """
    Give every task output a distinct constant physical error.

    Each TaskSpec-derived field metric must select the matching channel and unit,
    while a normalized-space update fails instead of mixing tensor spaces.
    """
    metrics = _metrics(*(metric.id for metric in _PHYSICAL_FIELD_METRICS))
    target = torch.zeros((1, _TASK.out_channels, 2, 2))
    errors = {field: float(2 + 3 * index) for index, field in enumerate(_TASK.output_names)}
    pred = torch.stack(
        tuple(torch.full((1, 2, 2), errors[field]) for field in _TASK.output_names),
        dim=1,
    )
    for metric in metrics.values():
        metric.update(pred, target, space="physical", batch_index=0)

    for definition in _PHYSICAL_FIELD_METRICS:
        field = definition.fields[0]
        metric = metrics[definition.id]
        assert metric.unit == _TASK.field(field).unit
        assert metric.compute() == pytest.approx(errors[field])

    first_metric = metrics[_PHYSICAL_FIELD_METRICS[0].id]
    with pytest.raises(ValueError, match="expects 'physical'"):
        first_metric.update(pred, target, space="normalized", batch_index=1)


def test_incompatible_physical_aggregate_is_rejected() -> None:
    """
    Resolve one physical aggregate spanning pressure and both velocity fields.

    Configuration must reject the mixed-unit reduction before runtime so a single
    scalar cannot falsely imply a coherent physical unit.
    """
    raw = configs.direct_config()
    defaults = experiments.config.defaults.get_task_defaults(str(raw["task"]))
    aggregate = copy.deepcopy(
        next(metric for metric in defaults["evaluation"]["metrics"] if metric["id"] == _PHYSICAL_FIELD_METRICS[0].id),
    )
    aggregate.pop("direction")
    aggregate["id"] = "physical_rmse_all"
    aggregate["fields"] = list(_TASK.output_names)
    raw["evaluation"] = {
        "metrics": [aggregate],
        "objective": {"id": "physical_rmse_all"},
    }

    with pytest.raises(ValueError, match="incompatible units"):
        experiments.config.loader.resolve_config(raw)
