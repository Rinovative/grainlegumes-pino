# ruff: noqa: S101
"""
Protect global physical-group RMSE reduction and raw train-scale semantics.

Two behavioral tests cover equal group weighting, component observability,
batch-partition invariance, task-owned groups, and contextual invalid-state
failures. Tensor-space routing remains covered by ``test_metric_spaces``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from src import domain, experiments, learning
from support import configs

_TASK = domain.tasks.registry.get_task("steady_flow")
_OBJECTIVE_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_NORMALIZED_VECTOR_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "group_rmse")
_PHYSICAL_VECTOR_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "vector_rmse")
_OBJECTIVE_ID = _OBJECTIVE_METRIC.id
_VECTOR_IDS = (_NORMALIZED_VECTOR_METRIC.id, _PHYSICAL_VECTOR_METRIC.id)


def _metric_config(*metric_ids: str) -> dict[str, Any]:
    """Return a resolved config containing only requested public metrics."""
    config = experiments.config.loader.resolve_config(configs.direct_config())
    requested = set(metric_ids)
    config["evaluation"]["metrics"] = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in requested]
    return config


def _evaluate_physical_partitions(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scales: torch.Tensor,
    partitions: tuple[int, ...],
) -> dict[str, float]:
    """Accumulate identical physical tensors under prescribed batch boundaries."""
    metrics = learning.metrics.metrics.build_evaluation_metrics(
        _metric_config(_OBJECTIVE_ID, *_VECTOR_IDS),
        device=torch.device("cpu"),
        output_standard_deviations=scales,
    )
    start = 0
    for batch_index, size in enumerate(partitions):
        stop = start + size
        for metric in metrics.values():
            metric.update(
                pred[start:stop],
                target[start:stop],
                space="physical",
                batch_index=batch_index,
            )
        start = stop
    assert start == pred.shape[0]
    return learning.metrics.metrics.finalize_metrics(metrics)


def test_group_objective_uses_equal_physical_groups_and_global_statistics() -> None:
    """
    Evaluate unequal pressure and velocity scales under three batch partitions.

    The objective must average the scalar and vector group errors equally. The
    velocity term combines physical component MSE before one root and divides by
    one root-sum-square raw train scale. A physically small transverse error stays
    visible without receiving independent unit-variance influence.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    groups = {group.id: group for group in task.output_groups}
    pressure = groups["pressure"]
    velocity = groups["velocity"]
    scales_by_field = {
        pressure.fields[0]: 4.0,
        velocity.fields[0]: 0.1,
        velocity.fields[1]: 10.0,
    }
    error_by_field = {
        pressure.fields[0]: 8.0,
        velocity.fields[0]: 0.1,
        velocity.fields[1]: 20.0,
    }
    scales = torch.tensor([scales_by_field[field] for field in task.output_names], dtype=torch.float64)
    errors = torch.tensor(
        [error_by_field[field] for field in task.output_names],
        dtype=torch.float64,
    ).reshape(1, -1, 1, 1)
    target = torch.zeros((5, task.out_channels, 2, 3), dtype=torch.float64)
    pred = target + errors

    partial = _evaluate_physical_partitions(pred, target, scales=scales, partitions=(2, 3))
    separate = _evaluate_physical_partitions(pred, target, scales=scales, partitions=(1, 1, 1, 1, 1))
    combined = _evaluate_physical_partitions(pred, target, scales=scales, partitions=(5,))

    pressure_error = 8.0 / 4.0
    physical_velocity = (0.1**2 + 20.0**2) ** 0.5
    normalized_velocity = physical_velocity / (0.1**2 + 10.0**2) ** 0.5
    expected_objective = 0.5 * pressure_error + 0.5 * normalized_velocity
    assert partial == pytest.approx(
        {
            _OBJECTIVE_ID: expected_objective,
            _NORMALIZED_VECTOR_METRIC.id: normalized_velocity,
            _PHYSICAL_VECTOR_METRIC.id: physical_velocity,
        },
        rel=1e-14,
        abs=1e-14,
    )
    assert partial == pytest.approx(separate, rel=1e-14, abs=1e-14)
    assert partial == pytest.approx(combined, rel=1e-14, abs=1e-14)
    independently_scaled_field_mean = (pressure_error + 1.0 + 2.0) / 3.0
    assert partial[_OBJECTIVE_ID] != pytest.approx(independently_scaled_field_mean)

    zero_sums = dict.fromkeys(task.output_names, 0.0)
    counts = dict.fromkeys(task.output_names, 1)
    u_only_sums = {**zero_sums, velocity.fields[0]: scales_by_field[velocity.fields[0]] ** 2}
    v_only_sums = {**zero_sums, velocity.fields[1]: scales_by_field[velocity.fields[1]] ** 2}
    u_only = learning.metrics.metrics.finalize_group_rmse_statistics(
        task.output_groups,
        squared_error_sums=u_only_sums,
        element_counts=counts,
        train_standard_deviations=scales_by_field,
    )
    v_only = learning.metrics.metrics.finalize_group_rmse_statistics(
        task.output_groups,
        squared_error_sums=v_only_sums,
        element_counts=counts,
        train_standard_deviations=scales_by_field,
    )
    shared_variance = scales_by_field[velocity.fields[0]] ** 2 + scales_by_field[velocity.fields[1]] ** 2
    expected_u_only = scales_by_field[velocity.fields[0]] / shared_variance**0.5
    expected_v_only = scales_by_field[velocity.fields[1]] / shared_variance**0.5
    assert u_only.normalized[velocity.id] == pytest.approx(expected_u_only)
    assert v_only.normalized[velocity.id] == pytest.approx(expected_v_only)
    assert v_only.normalized[velocity.id] / u_only.normalized[velocity.id] == pytest.approx(
        scales_by_field[velocity.fields[1]] / scales_by_field[velocity.fields[0]]
    )


def test_group_metric_is_task_owned_and_rejects_invalid_scales_or_statistics(
    synthetic_task: domain.tasks.spec.TaskSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve an alternate grouped task, then exercise scale and statistic failures.

    The generic builder must consume TaskSpec group membership without steady-flow
    field checks. Missing groups, partial groups, empty counts, and invalid raw
    fitted scales fail before an objective can be selected or compared.
    """
    monkeypatch.setattr(
        learning.metrics.metrics.domain.tasks.registry,
        "get_task",
        lambda _task_id: synthetic_task,
    )
    definition = next(
        metric.as_dict(all_fields=synthetic_task.output_names) for metric in synthetic_task.default_metrics if metric.kind == "group_macro_rmse"
    )
    objective_id = str(definition["id"])
    config = {"task": synthetic_task.id, "evaluation": {"metrics": [definition]}}
    scales = torch.tensor([2.0, 6.0])
    metric = learning.metrics.metrics.build_evaluation_metrics(
        config,
        device=torch.device("cpu"),
        output_standard_deviations=scales,
    )[objective_id]
    target = torch.zeros((2, synthetic_task.out_channels, 1, 2), dtype=torch.float64)
    errors = torch.tensor([2.0, 12.0], dtype=torch.float64).reshape(1, -1, 1, 1)
    metric.update(target + errors, target, space="physical", batch_index=0)

    assert metric.fields == synthetic_task.output_names
    assert metric.compute() == pytest.approx(1.5)

    partial_definition = dict(definition)
    partial_definition["fields"] = [synthetic_task.output_names[0]]
    config["evaluation"]["metrics"] = [partial_definition]
    with pytest.raises(ValueError, match="every TaskSpec output field"):
        learning.metrics.metrics.build_evaluation_metrics(
            config,
            device=torch.device("cpu"),
            output_standard_deviations=scales,
        )

    config["evaluation"]["metrics"] = [definition]
    for invalid_scales in (torch.tensor([0.0, 1.0]), torch.tensor([float("inf"), 1.0])):
        with pytest.raises(ValueError, match="finite and positive"):
            learning.metrics.metrics.build_evaluation_metrics(
                config,
                device=torch.device("cpu"),
                output_standard_deviations=invalid_scales,
            )
    with pytest.raises(ValueError, match="requires raw train-fitted"):
        learning.metrics.metrics.build_evaluation_metrics(config, device=torch.device("cpu"))

    group = synthetic_task.output_groups[0]
    with pytest.raises(ValueError, match="positive integer"):
        learning.metrics.metrics.finalize_group_rmse_statistics(
            (group,),
            squared_error_sums={group.fields[0]: 0.0},
            element_counts={group.fields[0]: 0},
            train_standard_deviations={group.fields[0]: 1.0},
        )
