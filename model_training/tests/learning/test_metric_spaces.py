# ruff: noqa: S101
"""Verify normalized and physical metric spaces, units, and field selection."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from src import experiments, learning

_CONFIG = Path(__file__).parents[2] / "configs" / "experiments" / "steady_flow_fno.yaml"


class AffineNormalizer:
    """Apply an affine normalization for observable metric-space checks."""

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
    """Provide the minimal evaluation data-processor surface."""

    def __init__(self, normalizer: AffineNormalizer) -> None:
        """Store the synthetic output normalizer."""
        self.out_normalizer = normalizer

    def eval(self) -> None:
        """Enter evaluation mode."""

    def preprocess(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize only the synthetic output target."""
        return {"x": batch["x"], "y": self.out_normalizer.transform(batch["y"])}


class UnitErrorModel(torch.nn.Module):
    """Return a normalized prediction exactly one above a zero target."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return three output channels of normalized ones."""
        return torch.ones((inputs.shape[0], 3, *inputs.shape[-2:]), device=inputs.device)


def _metrics(*metric_ids: str) -> dict[str, learning.metrics.metrics.DatasetMetric]:
    """Build a selected subset of resolved default metrics."""
    config = experiments.config.loader.load_and_resolve_config(_CONFIG)
    selected = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in metric_ids]
    config["evaluation"]["metrics"] = selected
    return learning.metrics.metrics.build_evaluation_metrics(config)


def test_normalized_and_physical_rmse_are_space_correct_once() -> None:
    """A normalized error of one maps to physical error two, never four."""
    metrics = _metrics(
        "normalized_rmse",
        "physical_rmse_p",
        "physical_rmse_u",
        "physical_rmse_v",
    )
    normalizer = AffineNormalizer(mean=10.0, standard_deviation=2.0)
    processor = SyntheticProcessor(normalizer)
    raw_batch = {
        "x": torch.zeros((2, 7, 3, 4)),
        "y": torch.full((2, 3, 3, 4), 10.0),
    }
    values = learning.training.loop.eval_one_epoch(
        UnitErrorModel(),
        [raw_batch],  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
        processor,
    )

    assert values["normalized_rmse"] == pytest.approx(1.0)
    assert values["physical_rmse_p"] == pytest.approx(2.0)
    assert values["physical_rmse_u"] == pytest.approx(2.0)
    assert values["physical_rmse_v"] == pytest.approx(2.0)


def test_physical_units_and_named_channel_selection() -> None:
    """Physical metrics follow task field names, units, and channel indices."""
    metrics = _metrics("physical_rmse_p", "physical_rmse_u", "physical_rmse_v")
    target = torch.zeros((1, 3, 2, 2))
    pred = torch.stack(
        (
            torch.full((1, 2, 2), 2.0),
            torch.full((1, 2, 2), 5.0),
            torch.full((1, 2, 2), 7.0),
        ),
        dim=1,
    )
    for metric in metrics.values():
        metric.update(pred, target, space="physical", batch_index=0)

    assert metrics["physical_rmse_p"].unit == "Pa"
    assert metrics["physical_rmse_u"].unit == "m/s"
    assert metrics["physical_rmse_v"].unit == "m/s"
    assert metrics["physical_rmse_p"].compute() == pytest.approx(2.0)
    assert metrics["physical_rmse_u"].compute() == pytest.approx(5.0)
    assert metrics["physical_rmse_v"].compute() == pytest.approx(7.0)
    with pytest.raises(ValueError, match="expects 'physical'"):
        metrics["physical_rmse_p"].update(pred, target, space="normalized", batch_index=1)


def test_incompatible_physical_aggregate_is_rejected() -> None:
    """Pressure and velocity cannot form a misleading physical overall RMSE."""
    raw = experiments.config.loader.load_yaml(_CONFIG)
    aggregate = copy.deepcopy(raw["evaluation"]["metrics"][2])
    aggregate.pop("field")
    aggregate["id"] = "physical_rmse_all"
    aggregate["fields"] = ["p", "u", "v"]
    raw["evaluation"]["metrics"] = [aggregate]
    raw["evaluation"]["objective"] = {"id": "physical_rmse_all"}

    with pytest.raises(ValueError, match="incompatible units"):
        experiments.config.loader.resolve_config(raw)
