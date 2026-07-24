# ruff: noqa: RUF043, S101, TC003
"""Verify completed-run admission and strict inference/resume checkpoint roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from src import analysis, experiments, learning
from torch import nn
from torch.utils.data import Dataset

_REQUIRED_PAYLOAD_FILES = (
    "config.yaml",
    "normalizer.pt",
    "split_indices.pt",
    "best_checkpoint.pt",
    "last_checkpoint.pt",
)


class _SyntheticDataset(Dataset[dict[str, Any]]):
    """Minimal field-aware dataset for inference context reconstruction."""

    input_fields: ClassVar[list[str]] = ["source"]
    output_fields: ClassVar[list[str]] = ["target"]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return {
            "x": torch.zeros(1, 1, 1),
            "y": torch.zeros(1, 1, 1),
            "meta": {},
        }


def _status_run(tmp_path: Path, status: str, *, touch_payloads: bool) -> Path:
    """Create one synthetic run summary and optional placeholder payload set."""
    run_dir = experiments.run.allocate_run_directory(tmp_path / status)
    experiments.run.transition_run_status(run_dir, "initializing")
    if status == "running":
        experiments.run.transition_run_status(run_dir, "running")
    elif status == "failed":
        experiments.run.transition_run_status(run_dir, "failed")
    if touch_payloads:
        for filename in _REQUIRED_PAYLOAD_FILES:
            (run_dir / filename).touch()
    return run_dir


def test_inference_and_artifacts_reject_running_status(tmp_path: Path) -> None:
    """Present filenames cannot make a running run loadable."""
    run_dir = _status_run(tmp_path, "running", touch_payloads=True)

    with pytest.raises(experiments.run.RunLifecycleError, match="status must be 'completed'"):
        learning.inference.context.load_inference_context(run_dir=run_dir, prefer_cuda=False)
    with pytest.raises(experiments.run.RunLifecycleError, match="status must be 'completed'"):
        analysis.artifact_service.load_run_artifact_plan(run_dir)


def test_incomplete_run_is_rejected_before_reconstruction(tmp_path: Path) -> None:
    """An allocated but incomplete leaf cannot reach model reconstruction."""
    run_dir = _status_run(tmp_path, "initializing", touch_payloads=False)

    with pytest.raises(experiments.run.RunLifecycleError, match="incomplete and not loadable"):
        learning.inference.context.load_inference_context(run_dir=run_dir, prefer_cuda=False)
    with pytest.raises(experiments.run.RunLifecycleError, match="incomplete and not loadable"):
        analysis.artifact_service.load_run_artifact_plan(run_dir)


def test_resume_requires_last_checkpoint_even_when_other_files_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume never substitutes best_checkpoint.pt for missing continuation state."""
    run_dir = _status_run(tmp_path, "running", touch_payloads=True)
    (run_dir / "last_checkpoint.pt").unlink()
    monkeypatch.setattr(experiments.run.config_loader, "load_and_resolve_config", lambda _path: {})

    with pytest.raises(experiments.run.RunLifecycleError, match="last_checkpoint.pt"):
        experiments.run.run_experiment("unused.yaml", resume=run_dir)


def test_inference_uses_exact_normalizer_state_returned_by_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference reconstructs preprocessing without reopening normalizer.pt."""
    context = learning.inference.context
    model = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    normalizer_state = {
        "in_normalizer.mean": torch.zeros(1, 1, 1, 1),
        "in_normalizer.std": torch.ones(1, 1, 1, 1),
        "out_normalizer.mean": torch.zeros(1, 1, 1, 1),
        "out_normalizer.std": torch.ones(1, 1, 1, 1),
    }
    completed_run = {
        "config": {},
        "split_indices": {},
        "best_checkpoint": {"model_state_dict": model.state_dict()},
        "normalizer_state": normalizer_state,
    }
    selection = context.SplitSelection(
        role="eval",
        dataset_path=tmp_path / "unused.pt",
        indices=torch.tensor([0]),
    )
    dataset = _SyntheticDataset()

    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _run_dir: completed_run)
    monkeypatch.setattr(context, "_field_contract", lambda _config: (["source"], ["target"]))
    monkeypatch.setattr(experiments.run, "configure_reproducibility", lambda _config: {"model_init": 1})
    monkeypatch.setattr(experiments.run, "seed_process", lambda _seed: None)
    monkeypatch.setattr(context, "_select_split", lambda **_kwargs: selection)

    def build_model(_config: dict[str, Any], *, device: torch.device) -> nn.Module:
        del device
        return model

    def create_dataset(_path: Path, *, task: object) -> _SyntheticDataset:
        del task
        return dataset

    monkeypatch.setattr(context, "_build_model_from_config", build_model)
    monkeypatch.setattr(experiments.config.loader, "validate_resolved_task_contract", lambda _config: object())
    monkeypatch.setattr(context.datasets.simulation, "create_task_dataset", create_dataset)
    monkeypatch.setattr(context, "_validate_split_indices_for_dataset", lambda **_kwargs: None)

    def unexpected_torch_load(*_args: Any, **_kwargs: Any) -> Any:
        msg = "Inference reopened a validated Torch artifact"
        raise AssertionError(msg)

    monkeypatch.setattr(torch, "load", unexpected_torch_load)

    loaded_model, loader, processor, device = context.load_inference_context(
        run_dir=tmp_path / "run",
        prefer_cuda=False,
    )

    assert loaded_model is model
    selected_dataset = loader.dataset
    assert isinstance(selected_dataset, context.IndexedSubset)
    assert len(selected_dataset) == 1
    assert device.type == "cpu"
    in_normalizer = processor.in_normalizer
    out_normalizer = processor.out_normalizer
    assert in_normalizer is not None
    assert out_normalizer is not None
    assert torch.equal(in_normalizer.mean, normalizer_state["in_normalizer.mean"])
    assert torch.equal(out_normalizer.std, normalizer_state["out_normalizer.std"])
