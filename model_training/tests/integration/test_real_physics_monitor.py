# ruff: noqa: S101
"""Opt-in bounded read-only regression for the real full-resolution monitor path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src import common, experiments, learning
from support import real_data

_GPU_SMOKE_CONFIG = Path(__file__).parents[2] / "configs/acceptance/steady_flow_fno_gpu_smoke.yaml"
_EXPECTED_FIRST_ID_INDEX = 418
_EXPECTED_FIRST_OOD_INDEX = 57
_FULL_SPATIAL_SHAPE = (251, 401)
_RESIDUAL_KEYS = {
    "physics/id/momentum_residual_mse",
    "physics/id/continuity_div_velocity_mse",
    "physics/id/continuity_div_eps_velocity_mse",
    "physics/id/pressure_boundary_mse",
}


class _ZeroNormalizedModel(torch.nn.Module):
    """Avoid training while exercising prediction and inverse reconstruction."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.new_zeros((inputs.shape[0], 3, *inputs.shape[-2:]))


@pytest.mark.real_data
@pytest.mark.skipif(
    not real_data.real_data_tests_enabled(),
    reason="set RUN_REAL_DATA_TESTS=1 for strict mounted-package acceptance",
)
def test_real_full_resolution_id_and_ood_physics_monitors_are_finite() -> None:
    """Use the explicit real-data root for one bounded ID and OOD monitor case."""
    real_data.require_real_data_root()
    config = experiments.config.loader.load_and_resolve_config(_GPU_SMOKE_CONFIG)
    dataset_root = Path(config["paths"]["dataset_root"])
    assert common.paths.resolve_dataset_path("lhs_var80_seed3001", dataset_root=dataset_root).is_file()
    assert common.paths.resolve_dataset_path("lhs_var120_seed4001", dataset_root=dataset_root).is_file()
    seed_plan = experiments.run.build_seed_plan(int(config["run"]["seed"]))
    dataloaders = experiments.config.loader.create_dataloaders_from_config(
        config,
        seed_plan=seed_plan,
    )
    assert dataloaders["split_indices"]["eval_indices"][0].item() == _EXPECTED_FIRST_ID_INDEX
    assert dataloaders["split_indices"]["ood_indices"][0].item() == _EXPECTED_FIRST_OOD_INDEX

    loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"))
    processor = dataloaders["data_processor"]
    loss.set_normalizers(
        in_normalizer=processor.in_normalizer,
        out_normalizer=processor.out_normalizer,
    )
    model = _ZeroNormalizedModel()
    for role in ("eval", "ood"):
        values = learning.training.loop.evaluate_physics_monitor(
            model,
            dataloaders[role],
            loss,
            torch.device("cpu"),
            processor,
            max_cases=1,
        )
        assert set(values) == _RESIDUAL_KEYS
        assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
        raw_batch = next(iter(dataloaders[role]))
        assert tuple(raw_batch["x"].shape[-2:]) == _FULL_SPATIAL_SHAPE
