# ruff: noqa: S101, PLR2004
"""
Exercise the explicit complete-data validator with production-shaped datasets.

A tiny in-memory fixture preserves the real DatasetIdentity, DataLoader sampler,
split-admission, and normalizer reconstruction contracts. Only configuration
resolution and mounted metadata admission are replaced at the I/O boundary.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import datasets, experiments
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

if TYPE_CHECKING:
    from pathlib import Path

validation = experiments.validation.data_pipeline


class _SyntheticDataset(Dataset[dict[str, Any]]):
    """Expose the production dataset surface over small in-memory BCHW tensors."""

    def __init__(
        self,
        *,
        dataset_id: str,
        inputs: Tensor,
        outputs: Tensor,
        task_id: str,
        data_digest: str,
        path: Path,
    ) -> None:
        """Store tensors, ordered field names, metadata, and exact identity."""
        sample_ids = tuple(f"{dataset_id}_case_{index}" for index in range(len(inputs)))
        self.data: dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "source_metadata": [{"case_id": case_id} for case_id in sample_ids],
        }
        self.input_fields = ["input"]
        self.output_fields = ["output"]
        self.path = path
        self.identity = datasets.identity.DatasetIdentity(
            dataset_id=dataset_id,
            task=task_id,
            data_contract_digest=data_digest,
            fingerprint=("a" if dataset_id == "id_data" else "b") * 64,
            sample_ids=sample_ids,
            sample_count=len(sample_ids),
            spatial_shape=tuple(inputs.shape[2:]),
        )

    def __len__(self) -> int:
        """Return the complete source sample count."""
        return len(self.identity.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one production-shaped raw sample with admitted case identity."""
        return {
            "x": self.data["inputs"][index],
            "y": self.data["outputs"][index],
            "meta": {"case_id": self.identity.sample_ids[index]},
        }


def _task() -> SimpleNamespace:
    """Return the TaskSpec attributes consumed by complete validation."""
    return SimpleNamespace(
        id="test_flow",
        contract_digest="d" * 64,
        data_contract_digest="d" * 64,
        tensor_layout=("batch", "channel", "y", "x"),
        normalization_axes=(0, 2, 3),
        preprocessing=SimpleNamespace(fit_split="train"),
        default_datasets=SimpleNamespace(train="id_data", ood=("ood_data",)),
        in_channels=1,
        out_channels=1,
        input_names=("input",),
        output_names=("output",),
    )


def _normalizer_state(inputs: Tensor, outputs: Tensor, indices: Tensor) -> dict[str, Tensor]:
    """Fit the exact sample-standard-deviation state on ID training membership."""
    selected_inputs = inputs[indices]
    selected_outputs = outputs[indices]
    return {
        "in_normalizer.mean": selected_inputs.mean(dim=(0, 2, 3), keepdim=True),
        "in_normalizer.std": selected_inputs.std(dim=(0, 2, 3), correction=1, keepdim=True),
        "out_normalizer.mean": selected_outputs.mean(dim=(0, 2, 3), keepdim=True),
        "out_normalizer.std": selected_outputs.std(dim=(0, 2, 3), correction=1, keepdim=True),
    }


def _split_info(
    *,
    task: SimpleNamespace,
    id_source: _SyntheticDataset,
    ood_source: _SyntheticDataset,
    split_seed: int,
) -> dict[str, Any]:
    """Build the current persisted split schema for a deterministic 2/2/2 split."""
    train_indices = torch.tensor([0, 1], dtype=torch.long)
    eval_indices = torch.tensor([2, 3], dtype=torch.long)
    ood_indices = torch.tensor([0, 2], dtype=torch.long)
    memberships = {
        "train": datasets.identity.membership_digest(
            role="train",
            dataset_fingerprint=id_source.identity.fingerprint,
            sample_ids=id_source.identity.sample_ids,
            indices=train_indices.tolist(),
        ),
        "eval": datasets.identity.membership_digest(
            role="eval",
            dataset_fingerprint=id_source.identity.fingerprint,
            sample_ids=id_source.identity.sample_ids,
            indices=eval_indices.tolist(),
        ),
        "ood": datasets.identity.membership_digest(
            role="ood",
            dataset_fingerprint=ood_source.identity.fingerprint,
            sample_ids=ood_source.identity.sample_ids,
            indices=ood_indices.tolist(),
        ),
    }
    return {
        "schema_version": datasets.identity.SPLIT_SCHEMA_VERSION,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "ood_indices": ood_indices,
        "metadata": {
            "datasets": {
                "train": id_source.identity.as_dict(),
                "ood": ood_source.identity.as_dict(),
            },
            "n_train_full": len(id_source),
            "n_train": len(train_indices),
            "n_eval": len(eval_indices),
            "n_ood_full": len(ood_source),
            "n_ood": len(ood_indices),
            "train_ratio": 0.5,
            "ood_fraction": 0.5,
            "split_seed": split_seed,
            "membership_digests": memberships,
        },
    }


def test_full_validator_checks_complete_metadata_split_normalizer_and_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return typed PASS evidence while using the maintained scientific owners."""
    task = _task()
    id_inputs = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    id_outputs = id_inputs.mul(2).add(3)
    ood_inputs = torch.arange(32, 48, dtype=torch.float32).reshape(4, 1, 2, 2)
    ood_outputs = ood_inputs.mul(2).add(3)
    id_source = _SyntheticDataset(
        dataset_id="id_data",
        inputs=id_inputs,
        outputs=id_outputs,
        task_id=task.id,
        data_digest=task.data_contract_digest,
        path=tmp_path / "id_data.pt",
    )
    ood_source = _SyntheticDataset(
        dataset_id="ood_data",
        inputs=ood_inputs,
        outputs=ood_outputs,
        task_id=task.id,
        data_digest=task.data_contract_digest,
        path=tmp_path / "ood_data.pt",
    )
    config = {
        "task": task.id,
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "ood_datasets": ["ood_data"],
            "train_ratio": 0.5,
            "ood_fraction": 0.5,
        },
        "paths": {"training_meta_root": str(tmp_path / "metadata")},
    }
    build_calls: list[dict[str, int]] = []

    def build_loaders(_config: dict[str, Any], *, seed_plan: dict[str, int]) -> dict[str, Any]:
        """Create fresh loaders, processor, and split state for each production call."""
        build_calls.append(seed_plan)
        split = _split_info(
            task=task,
            id_source=id_source,
            ood_source=ood_source,
            split_seed=seed_plan["split"],
        )
        state = _normalizer_state(id_inputs, id_outputs, split["train_indices"])
        processor = datasets.base.data_processor_from_state(state, device="cpu")
        train_generator = torch.Generator().manual_seed(seed_plan["loader"])
        return {
            "train": DataLoader(
                Subset(id_source, split["train_indices"].tolist()),
                batch_size=2,
                shuffle=True,
                generator=train_generator,
            ),
            "eval": DataLoader(
                Subset(id_source, split["eval_indices"].tolist()),
                batch_size=2,
                shuffle=False,
            ),
            "ood": DataLoader(
                Subset(ood_source, split["ood_indices"].tolist()),
                batch_size=2,
                shuffle=False,
            ),
            "data_processor": processor,
            "split_indices": split,
        }

    metadata_calls: list[tuple[str, Path, Path]] = []

    def load_metadata(
        dataset_id: str,
        *,
        dataset_identity: Any,
        metadata_root: Path,
        dataset_path: Path,
    ) -> dict[str, str]:
        """Record complete identity/path binding at the mounted metadata boundary."""
        assert dataset_identity.dataset_id == dataset_id
        metadata_calls.append((dataset_id, metadata_root, dataset_path))
        return {"dataset_id": dataset_id}

    monkeypatch.setattr(validation.config_loader, "validate_resolved_config", dict)
    monkeypatch.setattr(validation.config_loader, "validate_resolved_task_contract", lambda _value: task)
    monkeypatch.setattr(validation.config_loader, "create_dataloaders_from_config", build_loaders)
    monkeypatch.setattr(datasets.metadata, "load_dataset_metadata", load_metadata)

    result = validation.validate_full_data_pipeline(config)

    assert isinstance(result, validation.FullDataValidationResult)
    assert len(build_calls) == 2
    assert metadata_calls == [
        ("id_data", tmp_path / "metadata", tmp_path / "id_data.pt"),
        ("ood_data", tmp_path / "metadata", tmp_path / "ood_data.pt"),
    ]
    assert len(result.dataset_membership) == 5
    assert len(result.channels) == 2
    assert len(result.coverage) == 3
    assert {record.result for record in result.dataset_membership} == {"PASS"}
    assert {record.result for record in result.channels} == {"PASS"}
    assert {record.result for record in result.coverage} == {"PASS"}
    assert {record.loader for record in result.coverage} == {"ID train", "ID evaluation", "OOD"}
    assert all(record.inverse_checked for record in result.coverage)
    assert [record.result for record in result.overall[-2:]] == ["INFO", "INFO"]


def test_full_validator_delegates_resolved_recipe_datasets_without_reapplying_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass explicit resolved dataset selection to the production loader unchanged."""
    task = _task()
    config = {
        "task": task.id,
        "run": {"seed": 9},
        "data": {
            "train_dataset": "recipe_id",
            "ood_datasets": ["recipe_ood"],
            "train_ratio": 0.5,
            "ood_fraction": 0.5,
        },
        "paths": {"training_meta_root": "/unused"},
    }

    class LoaderReachedError(RuntimeError):
        """Mark successful delegation without constructing synthetic loaders."""

    def capture_loader_config(effective: dict[str, Any], *, seed_plan: dict[str, int]) -> None:
        assert effective["data"]["train_dataset"] == "recipe_id"
        assert effective["data"]["ood_datasets"] == ["recipe_ood"]
        assert seed_plan == experiments.run.build_seed_plan(9)
        raise LoaderReachedError

    monkeypatch.setattr(validation.config_loader, "validate_resolved_config", dict)
    monkeypatch.setattr(validation.config_loader, "validate_resolved_task_contract", lambda _value: task)
    monkeypatch.setattr(validation.config_loader, "create_dataloaders_from_config", capture_loader_config)

    with pytest.raises(LoaderReachedError):
        validation.validate_full_data_pipeline(config)
