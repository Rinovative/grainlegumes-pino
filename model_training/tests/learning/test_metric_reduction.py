# ruff: noqa: S101
"""Verify dataset metrics use sufficient statistics independent of batches."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src import experiments, learning

_CONFIG = Path(__file__).parents[2] / "configs" / "experiments" / "steady_flow_fno.yaml"


def _metric_config() -> dict[str, object]:
    """Return resolved normalized RMSE and relative-L2 metric declarations."""
    config = experiments.config.loader.load_and_resolve_config(_CONFIG)
    rmse = next(metric for metric in config["evaluation"]["metrics"] if metric["id"] == "normalized_rmse")
    relative = {
        "id": "normalized_relative_l2",
        "kind": "relative_l2",
        "space": "normalized",
        "fields": "all",
        "reduction": "sample_mean",
        "direction": "minimize",
    }
    config["evaluation"]["metrics"] = [rmse, relative]
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
    """Run the real evaluation loop with one requested batch partition."""
    metrics = learning.metrics.metrics.build_evaluation_metrics(_metric_config())
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


def test_evaluation_metrics_use_global_statistics_independent_of_partition() -> None:
    """The real evaluation loop ignores batch boundaries and partial batches."""
    target = torch.tensor([1.0, 2.0, 4.0]).reshape(3, 1, 1, 1).expand(-1, 3, 2, 2)
    errors = torch.tensor([0.0, 1.0, 3.0]).reshape(3, 1, 1, 1)
    pred = target + errors

    partial = _evaluate_loop_partition(pred, target, (2, 1))
    separate = _evaluate_loop_partition(pred, target, (1, 1, 1))
    combined = _evaluate_loop_partition(pred, target, (3,))
    expected = {
        "normalized_rmse": (10.0 / 3.0) ** 0.5,
        "normalized_relative_l2": 5.0 / 12.0,
    }
    misleading_equal_batch_rmse = ((0.5**0.5) + 3.0) / 2.0

    assert partial == pytest.approx(expected)
    assert partial == pytest.approx(separate, rel=1e-14, abs=1e-14)
    assert partial == pytest.approx(combined, rel=1e-14, abs=1e-14)
    assert partial["normalized_rmse"] != pytest.approx(misleading_equal_batch_rmse)


def test_nonfinite_metric_reports_metric_and_batch_context() -> None:
    """Non-finite values surface the configured metric id and batch index."""
    metrics = learning.metrics.metrics.build_evaluation_metrics(_metric_config())
    metric = metrics["normalized_rmse"]
    pred = torch.zeros((1, 3, 2, 2))
    pred[0, 0, 0, 0] = torch.nan

    with pytest.raises(FloatingPointError, match=r"normalized_rmse.*batch 7"):
        metric.update(pred, torch.zeros_like(pred), space="normalized", batch_index=7)
