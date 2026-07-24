# ruff: noqa: S101
"""Verify semantic supervised and physics-loss composition and warm-up state."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src import experiments, learning

_MIDPOINT_EPOCH = 2
_CONFIG_ROOT = Path(__file__).parents[2] / "configs" / "experiments"


class IdentityNormalizer:
    """Return tensors unchanged for synthetic physical-space tests."""

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged."""
        return tensor


def _physics_config() -> dict[str, object]:
    """Return a resolved PI config with short deterministic warmups."""
    raw = experiments.config.loader.load_yaml(_CONFIG_ROOT / "steady_flow_pifno.yaml")
    raw["loss"]["physics"]["residual_weight"] = {  # type: ignore[index]
        "target": 2.0,
        "warmup": {"kind": "linear", "epochs": 4},
    }
    raw["loss"]["physics"]["boundary_weight"] = {  # type: ignore[index]
        "target": 3.0,
        "warmup": {"kind": "linear", "epochs": 4},
    }
    raw["loss"]["physics"]["interior_crop"] = 1  # type: ignore[index]
    return experiments.config.loader.resolve_config(raw)


def _manufactured_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized-as-physical task tensors with nonzero physics terms."""
    height = 9
    width = 11
    y_values = torch.linspace(0.0, 1.0, height)
    x_values = torch.linspace(0.0, 2.0, width)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    zeros = torch.zeros_like(x_grid)
    p_bc = zeros.clone()
    p_bc[:, 0, :] = 1.0
    inputs = torch.stack(
        (
            x_grid,
            y_grid,
            zeros,
            zeros,
            zeros,
            torch.full_like(x_grid, 0.5),
            p_bc,
        ),
        dim=1,
    )
    pred = torch.stack((zeros, 1e-5 * x_grid, zeros), dim=1)
    target = torch.zeros_like(pred)
    return inputs, pred, target


def test_disabled_physics_exposes_named_zero_components() -> None:
    """Disabled physics stays present as clean named zero components."""
    supervised = experiments.config.loader.load_and_resolve_config(_CONFIG_ROOT / "steady_flow_fno.yaml")
    supervised_loss = learning.losses.factory.build_training_loss(supervised)
    pred = torch.ones((2, 3, 8, 8))
    target = torch.zeros_like(pred)
    components = supervised_loss.compute_components(pred, x=None, y=target)

    assert tuple(components) == ("total", "data", "momentum", "continuity", "boundary")
    assert components["total"] == components["data"]
    assert components["momentum"].item() == 0.0
    assert components["continuity"].item() == 0.0
    assert components["boundary"].item() == 0.0


def test_physics_composition_weighting_and_linear_warmup() -> None:
    """Explicit epoch state produces exact start, midpoint, and end sums."""
    loss = learning.losses.factory.build_training_loss(_physics_config())
    normalizer = IdentityNormalizer()
    loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
    inputs, pred, target = _manufactured_batch()

    start = loss.compute_components(pred, x=inputs, y=target, epoch=0)
    midpoint = loss.compute_components(pred, x=inputs, y=target, epoch=2)
    end = loss.compute_components(pred, x=inputs, y=target, epoch=4)

    assert start["momentum"].item() == 0.0
    assert start["continuity"].item() == 0.0
    assert start["boundary"].item() == 0.0
    assert midpoint["momentum"] == pytest.approx(0.5 * end["momentum"])
    assert midpoint["continuity"] == pytest.approx(0.5 * end["continuity"])
    assert midpoint["boundary"] == pytest.approx(0.5 * end["boundary"])
    for components in (start, midpoint, end):
        assert components["total"] == pytest.approx(components["data"] + components["momentum"] + components["continuity"] + components["boundary"])
    assert loss.component_weights(epoch=2) == {
        "data": 1.0,
        "momentum": 1.0,
        "continuity": 1.0,
        "boundary": 1.5,
    }

    loss.set_epoch(_MIDPOINT_EPOCH)
    forward_total = loss(pred, x=inputs, y=target)
    assert forward_total == loss.last_components["total"]
    assert forward_total == pytest.approx(midpoint["total"])
    assert int(loss.state_dict()["current_epoch"].item()) == _MIDPOINT_EPOCH
    with pytest.raises(ValueError, match="non-negative integer"):
        loss.set_epoch(-1)


def test_invalid_physics_loss_settings_fail_through_public_factory() -> None:
    """Unknown derivative and warm-up settings fail during loss construction."""
    derivative_config = _physics_config()
    derivative_config["loss"]["physics"]["derivatives"]["kind"] = "finite_difference"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown derivative identifier"):
        learning.losses.factory.build_training_loss(derivative_config)

    warmup_config = _physics_config()
    warmup_config["loss"]["physics"]["residual_weight"]["warmup"]["kind"] = "cosine"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown warmup identifier"):
        learning.losses.factory.build_training_loss(warmup_config)
