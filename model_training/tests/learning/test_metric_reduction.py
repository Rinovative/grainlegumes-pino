# ruff: noqa: S101
"""
Protect dataset metrics whose sufficient statistics must be batch-partition invariant.

Unequal and partial batches establish global per-field SSE/count accumulation,
field-macro RMSE, task-driven synthetic outputs, and contextual empty/non-finite
failures. Normalized-versus-physical space routing is covered by
``test_metric_spaces``; training loss reduction is outside this module.
"""

from __future__ import annotations

import pytest
import torch
from src import domain, experiments, learning
from support import configs

_CONFIG = configs.acceptance_config_path()


def _metric_config() -> dict[str, object]:
    """Return the maintained normalized predictive metric declarations."""
    config = experiments.config.loader.load_and_resolve_config(_CONFIG)
    selected_ids = {
        "normalized_macro_rmse",
        "normalized_rmse_p",
        "normalized_rmse_u",
        "normalized_rmse_v",
        "normalized_rmse",
        "normalized_relative_l2",
    }
    config["evaluation"]["metrics"] = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in selected_ids]
    return config


class FirstThreeChannelsModel(torch.nn.Module):
    """Expose three synthetic input channels as normalized predictions."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the first three channels unchanged."""
        return inputs[:, :3]


def _evaluate_loop_partition(
    pred: torch.Tensor,
    target: torch.Tensor,
    partitions: tuple[int, ...],
) -> dict[str, float]:
    """
    Run the real evaluation loop with one prescribed batch partition.

    The helper exposes identical prediction/target content under different batch
    boundaries so tests can isolate dataset-level reduction invariance.
    """
    metrics = learning.metrics.metrics.build_evaluation_metrics(_metric_config(), device=torch.device("cpu"))
    batches: list[dict[str, torch.Tensor]] = []
    start = 0
    for size in partitions:
        stop = start + size
        inputs = torch.zeros((size, 7, *pred.shape[-2:]))
        inputs[:, :3] = pred[start:stop]
        batches.append({"x": inputs, "y": target[start:stop]})
        start = stop
    return learning.training.loop.eval_one_epoch(
        FirstThreeChannelsModel(),
        batches,  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
    )


def test_macro_rmse_uses_global_equal_field_statistics_across_partitions() -> None:
    """
    Partition unequal per-field errors into different complete/partial batches.

    Final macro RMSE must equal the arithmetic mean of fieldwise global SSE/count
    roots in every partition, rejecting pooled or mean-of-batch alternatives.
    """
    target = torch.tensor([1.0, 2.0, 4.0]).reshape(3, 1, 1, 1).expand(-1, 3, 2, 2)
    field_errors = torch.tensor([0.0, 1.0, 3.0]).reshape(1, 3, 1, 1)
    pred = target + field_errors

    partial = _evaluate_loop_partition(pred, target, (2, 1))
    separate = _evaluate_loop_partition(pred, target, (1, 1, 1))
    combined = _evaluate_loop_partition(pred, target, (3,))
    pooled_rmse = (10.0 / 3.0) ** 0.5
    expected = {
        "normalized_macro_rmse": 4.0 / 3.0,
        "normalized_rmse_p": 0.0,
        "normalized_rmse_u": 1.0,
        "normalized_rmse_v": 3.0,
        "normalized_rmse": pooled_rmse,
        "normalized_relative_l2": pooled_rmse * 7.0 / 12.0,
    }

    assert partial == pytest.approx(expected)
    assert partial == pytest.approx(separate, rel=1e-14, abs=1e-14)
    assert partial == pytest.approx(combined, rel=1e-14, abs=1e-14)
    assert partial["normalized_macro_rmse"] == pytest.approx(
        (partial["normalized_rmse_p"] + partial["normalized_rmse_u"] + partial["normalized_rmse_v"]) / 3.0
    )
    assert partial["normalized_macro_rmse"] != pytest.approx(pooled_rmse)

    for field_index in range(3):
        perturbed_errors = field_errors.clone()
        perturbed_errors[:, field_index] += 3.0
        perturbed = _evaluate_loop_partition(target + perturbed_errors, target, (2, 1))
        assert perturbed["normalized_macro_rmse"] - partial["normalized_macro_rmse"] == pytest.approx(1.0)


def test_macro_rmse_is_task_driven_for_alternate_outputs(
    synthetic_task: domain.tasks.spec.TaskSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replace the registered task with a valid two-output synthetic contract.

    Macro RMSE must average those two named field errors and reject an incomplete
    field list, proving the accumulator contains no steady-flow channel constants.
    """
    monkeypatch.setattr(
        learning.metrics.metrics.domain.tasks.registry,
        "get_task",
        lambda _task_id: synthetic_task,
    )
    metric_definition = {
        "id": "normalized_macro_rmse",
        "kind": "macro_rmse",
        "space": "normalized",
        "fields": list(synthetic_task.output_names),
        "reduction": "field_macro_element_mean",
        "direction": "minimize",
    }
    config = {
        "task": synthetic_task.id,
        "evaluation": {"metrics": [metric_definition]},
    }
    metric = learning.metrics.metrics.build_evaluation_metrics(config, device=torch.device("cpu"))["normalized_macro_rmse"]
    target = torch.zeros((2, len(synthetic_task.output_names), 1, 2))
    errors = torch.tensor([2.0, 6.0]).reshape(1, 2, 1, 1)

    metric.update(target + errors, target, space="normalized", batch_index=0)

    assert metric.fields == synthetic_task.output_names
    assert metric.compute() == pytest.approx(4.0)

    metric_definition["fields"] = [synthetic_task.output_names[0]]
    with pytest.raises(ValueError, match="every TaskSpec output field"):
        learning.metrics.metrics.build_evaluation_metrics(config, device=torch.device("cpu"))


def test_empty_and_nonfinite_macro_metric_report_objective_context() -> None:
    """
    Compute before any update, then update one batch containing a NaN prediction.

    Both failures must name the metric and the latter its batch index, making an
    invalid selection objective diagnosable rather than silently comparable.
    """
    metrics = learning.metrics.metrics.build_evaluation_metrics(_metric_config(), device=torch.device("cpu"))
    metric = metrics["normalized_macro_rmse"]
    with pytest.raises(RuntimeError, match=r"normalized_macro_rmse.*without evaluation elements"):
        metric.compute()

    pred = torch.zeros((1, 3, 2, 2))
    pred[0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match=r"normalized_macro_rmse.*batch 7"):
        metric.update(pred, torch.zeros_like(pred), space="normalized", batch_index=7)
