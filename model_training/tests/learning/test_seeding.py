# ruff: noqa: S101, N818, NPY002, S311, TC003
"""Verify stable labeled seeds and pre-construction reproducibility controls."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from src import experiments, learning


def test_labeled_subseeds_are_stable_distinct_and_reproducible() -> None:
    """Unrelated call order cannot change label-derived streams."""
    first = experiments.run.build_seed_plan(19)
    reverse = {label: experiments.run.derive_subseed(19, label) for label in reversed(tuple(first))}

    assert first == reverse
    assert len(first) == len(set(first.values()))


def test_process_seed_reproduces_python_numpy_torch_and_model_init() -> None:
    """Process seeding controls every maintained process-global RNG."""
    seed = experiments.run.derive_subseed(11, "model_init")
    experiments.run.seed_process(seed)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )
    experiments.run.seed_process(seed)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])


def test_run_deterministic_controls_torch_settings() -> None:
    """The config flag changes implemented deterministic behavior."""
    experiments.run.configure_determinism(True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    experiments.run.configure_determinism(False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark


class _Processor:
    """Minimal serializable data processor for construction-order testing."""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a small serializable state."""
        return {"value": torch.tensor(1)}


def test_model_subseed_is_applied_immediately_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run orchestration applies model_init after DataLoader construction."""
    config: dict[str, Any] = {
        "task": "synthetic",
        "run": {"name": "run", "seed": 23, "deterministic": True, "device": "cpu"},
        "training": {"epochs": 1, "mixed_precision": False},
        "tracking": {"wandb": {"enabled": False}},
        "evaluation": {
            "objective": {
                "id": "objective",
                "kind": "objective",
                "space": "normalized",
                "fields": ["value"],
                "reduction": "element_mean",
                "direction": "minimize",
            }
        },
        "model": {"kind": "synthetic"},
        "paths": {"output_root": str(tmp_path)},
    }
    run_dir = experiments.run.prepare_fresh_run(config, run_dir=tmp_path / "run")
    seed_calls: list[int] = []
    expected = experiments.run.build_seed_plan(23)

    monkeypatch.setattr(experiments.run, "seed_process", seed_calls.append)
    monkeypatch.setattr(
        experiments.run.config_loader,
        "create_dataloaders_from_config",
        lambda *_args, **_kwargs: {
            "data_processor": _Processor(),
            "split_indices": {"train_indices": torch.tensor([0]), "eval_indices": torch.tensor([1]), "ood_indices": torch.tensor([0])},
        },
    )

    class ConstructionReached(RuntimeError):
        """Stop the orchestration immediately after the ordering assertion."""

    def build_model(_config: dict[str, Any]) -> torch.nn.Module:
        assert seed_calls[-1] == expected["model_init"]
        raise ConstructionReached

    monkeypatch.setattr(learning.models.factory, "build_model", build_model)
    with pytest.raises(ConstructionReached):
        experiments.run.execute_prepared_run(config, run_dir=run_dir, persisted_config=config)

    assert seed_calls == [expected["process"], expected["model_init"]]
    assert experiments.run.read_run_summary(run_dir)["status"] == "failed"
