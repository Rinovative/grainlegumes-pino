# ruff: noqa: S101, EM101, PT017, TC003, TRY003
"""Verify task-generic dataset/artifact fields, units, provenance, and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from src import analysis, datasets, domain
from torch import nn
from torch.utils.data import DataLoader

_FUTURE_SOURCE_VALUE = 4.0
_FUTURE_METADATA_VALUE = 3.25


class _IdentityNormalizer:
    """Leave synthetic tensors unchanged at the artifact processor boundary."""

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Projection(nn.Module):
    """Project three future-task inputs into two deterministic outputs."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 2.0 * value[:, :2]


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """Save one strict payload under its logical dataset id."""
    dataset_id = payload["dataset_id"]
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.pt"
    torch.save(payload, path)
    return path


def test_generic_artifacts_preserve_task_fields_units_and_provenance(
    tmp_path: Path,
    future_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Future fields flow through NPZ, Parquet, provenance, and metadata cleanly."""
    task = future_task
    source_indices = [4]
    provenance: dict[str, Any] = {
        "provenance_schema_version": analysis.artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
        "run": {"name": "future", "task": task.id, "best_checkpoint_sha256": "abc"},
        "selection": {
            "effective_case_count": 1,
            "effective_ordered_source_indices_sha256": analysis.artifacts.ordered_indices_sha256(source_indices),
        },
        "evaluator": {
            "input_fields": list(task.input_names),
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
        },
    }
    inputs = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    targets = torch.ones(1, 2, 2, 2)
    loader = [
        {
            "x": inputs,
            "y": targets,
            "source_index": torch.tensor([4]),
            "split_local_index": torch.tensor([0]),
            "meta": {"label": ["future-case"], "quality": torch.tensor([7])},
        }
    ]
    processor = SimpleNamespace(
        in_normalizer=_IdentityNormalizer(),
        out_normalizer=_IdentityNormalizer(),
    )
    save_root = tmp_path / "analysis" / "id"

    frame, parquet_path = analysis.artifacts.generate_artifacts(
        task=task,
        model=_Projection(),
        loader=loader,
        processor=processor,
        device=torch.device("cpu"),
        save_root=save_root,
        dataset_name="future_id",
        provenance=provenance,
    )

    assert parquet_path.is_file()
    assert frame.columns.is_unique
    assert {"rmse_transported_mass", "rmse_temperature"}.issubset(frame.columns)
    npz_path = save_root / "npz" / "case_0005.npz"
    with np.load(npz_path, allow_pickle=False) as payload:
        assert payload["input_fields"].tolist() == list(task.input_names)
        assert payload["output_fields"].tolist() == list(task.output_names)
        assert payload["output_units"].tolist() == ["kg", "K"]
        assert payload["artifact_fields"].tolist() == list(task.output_names)
        assert payload["artifact_units"].tolist() == ["kg", "K"]
        metadata = json.loads(str(payload["meta"].item()))
        assert not {"case_index", "source_index", "split_local_index"}.intersection(metadata)

    stored_provenance = json.loads((save_root / analysis.artifacts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    stored_outputs = stored_provenance.pop("outputs")
    assert stored_provenance == provenance
    assert stored_outputs == analysis.artifacts.artifact_output_manifest(save_root)


def test_physical_cross_permeability_is_reconstructed_from_its_ratio() -> None:
    """Artifact kxy is physical m², while its stored representation stays dimensionless."""
    encoded = torch.tensor([[[[-2.0]], [[0.25]], [[-4.0]]]])
    permeability = analysis.artifacts.extract_kappa(
        encoded,
        input_fields=["kxx", "kxy", "kyy"],
        kappa_names=["kxx", "kxy", "kyy"],
    )

    assert torch.equal(permeability["kappa_encoded"], encoded)
    assert permeability["kappa"][0, :, 0, 0].tolist() == pytest.approx([1e-2, 2.5e-4, 1e-4])


def test_generic_artifacts_reject_reserved_source_metadata(
    tmp_path: Path,
    future_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Source metadata may not duplicate authoritative artifact identity."""
    task = future_task
    provenance = {
        "selection": {
            "effective_case_count": 1,
            "effective_ordered_source_indices_sha256": analysis.artifacts.ordered_indices_sha256([0]),
        }
    }
    loader = [
        {
            "x": torch.zeros(1, 3, 2, 2),
            "y": torch.zeros(1, 2, 2, 2),
            "source_index": torch.tensor([0]),
            "split_local_index": torch.tensor([0]),
            "meta": {"case_index": 99},
        }
    ]
    processor = SimpleNamespace(
        in_normalizer=_IdentityNormalizer(),
        out_normalizer=_IdentityNormalizer(),
    )

    try:
        analysis.artifacts.generate_artifacts(
            task=task,
            model=_Projection(),
            loader=loader,
            processor=processor,
            device=torch.device("cpu"),
            save_root=tmp_path / "target",
            dataset_name="future_id",
            provenance=provenance,
        )
    except KeyError as error:
        assert "reserved artifact identity" in str(error)
    else:
        raise AssertionError("reserved identity metadata was accepted")


def test_future_task_flows_through_generic_dataset_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    future_task: domain.tasks.spec.TaskSpec,
) -> None:
    """A distinct valid task uses the same case, merge, and loader boundaries."""
    input_fields = {name: torch.full((2, 3), _FUTURE_SOURCE_VALUE + index) for index, name in enumerate(future_task.input_names)}
    output_fields = {name: torch.full((2, 3), 20.0 + index) for index, name in enumerate(future_task.output_names)}
    case = datasets.identity.build_case_payload(
        task=future_task,
        case_id="case_0000",
        input_fields=input_fields,
        output_fields=output_fields,
        source_identity={"token": "future"},
        source_metadata={
            "generator": {
                "parameters": {"source_strength": _FUTURE_METADATA_VALUE},
            },
        },
    )
    validated = datasets.identity.validate_case_payload(case, task=future_task)
    merged = datasets.identity.build_merged_dataset_payload(
        task=future_task,
        dataset_id="future_id",
        sample_ids=("case_0000",),
        source_identities=(validated.source_identity,),
        source_metadata=(validated.source_metadata,),
        case_fingerprints=(validated.fingerprint,),
        inputs=validated.inputs.unsqueeze(0),
        outputs=validated.outputs.unsqueeze(0),
    )
    path = _save_dataset(tmp_path, merged)
    loaded = datasets.simulation.create_task_dataset(path, task=future_task)
    expected_metadata = {
        "generator": {
            "parameters": {"source_strength": _FUTURE_METADATA_VALUE},
        },
    }
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    frame, logs = analysis.eda.dataframe.generate_eda_dataframe(
        "future_id",
        task=future_task,
        max_cases=1,
    )
    assert frame.index.tolist() == ["case_0000"]
    assert frame.columns.tolist() == [*future_task.input_names, *future_task.output_names, "meta"]
    assert np.all(frame.at["case_0000", "source_rate"] == _FUTURE_SOURCE_VALUE)
    assert frame.at["case_0000", "meta"] == expected_metadata
    assert str(path) in "\n".join(logs)

    sample = loaded[0]
    assert sample["meta"] == expected_metadata
    sample["meta"]["generator"]["parameters"]["source_strength"] = -1.0
    assert loaded[0]["meta"] == expected_metadata

    batch = next(iter(DataLoader(loaded, batch_size=1, shuffle=False)))
    artifact_metadata = analysis.artifacts.meta_to_jsonable(batch["meta"])
    flattened_metadata = analysis.evaluation.dataframe.flatten_meta_scalars(artifact_metadata)

    assert loaded.input_fields == list(future_task.input_names)
    assert loaded.output_fields == list(future_task.output_names)
    assert flattened_metadata["generator_parameters_source_strength"] == _FUTURE_METADATA_VALUE
    assert loaded[0]["x"].shape == (3, 2, 3)
    assert torch.all(loaded[0]["x"][0] == _FUTURE_SOURCE_VALUE)
    assert future_task.physics.kind == "none"
    assert [metric.fields for metric in future_task.default_metrics[1:]] == [
        ("transported_mass",),
        ("temperature",),
    ]
