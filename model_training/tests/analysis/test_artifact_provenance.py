# ruff: noqa: S101, EM101, PT017, SLF001, TRY003
"""
Verify task-generic and steady-flow artifact generation against current provenance.

The suite checks field/unit propagation, normalized SSE/count equivalence across
batch partitions, dual continuity, boundary naming, physical permeability, and
reserved metadata rejection. Cache locking and rebuild races belong to
``test_artifact_identity``; visualization behavior is outside this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from src import analysis, datasets, domain, learning
from torch import nn
from torch.utils.data import DataLoader, Dataset

_SYNTHETIC_INPUT_VALUE = 4.0
_SYNTHETIC_METADATA_VALUE = 3.25


class _MappingDataset(Dataset[dict[str, Any]]):
    """
    Expose ordered synthetic artifact samples through ``DataLoader``.

    The fixture deliberately implements only the map-style dataset boundary
    needed to compare artifact output across different batch partitions.
    """

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        """Store synthetic samples in deterministic order."""
        self.samples = samples

    def __len__(self) -> int:
        """Return the synthetic sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one synthetic sample by position."""
        return self.samples[index]


class _IdentityNormalizer:
    """
    Leave synthetic tensors unchanged at the processor boundary.

    Identity normalization makes runtime and stored SSE/count evidence directly
    comparable without introducing fitted-state or serialization concerns.
    """

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        """Return normalized-space input unchanged."""
        return value

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        """Return physical-space output unchanged."""
        return value


class _Projection(nn.Module):
    """
    Project synthetic-task inputs into two deterministic output fields.

    The fixed linear mapping gives the tests exact predictions while exercising
    the ordinary ``nn.Module`` inference boundary used by artifact generation.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map the leading two input channels to doubled predictions."""
        return 2.0 * value[:, :2]


class _SteadyProjection(nn.Module):
    """
    Return deterministic ``p=x``, ``u=x``, ``v=0`` steady-flow fields.

    This manufactured state yields distinct continuity definitions and stable
    pressure-boundary values without loading a trained checkpoint.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Construct the manufactured pressure and velocity channels."""
        zeros = torch.zeros_like(value[:, 0])
        return torch.stack((value[:, 0], value[:, 0], zeros), dim=1)


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """
    Save one strict payload at its canonical logical-dataset path.

    Returning the concrete ``.pt`` path lets the contract test exercise both
    environment-based discovery and direct task-dataset loading.
    """
    dataset_id = payload["dataset_id"]
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.pt"
    torch.save(payload, path)
    return path


def test_generic_artifacts_preserve_task_fields_units_and_provenance(
    tmp_path: Path,
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Preserve a synthetic task's fields, units, provenance, and metadata.

    A generic two-output fixture must produce current NPZ and Parquet evidence
    without steady-flow columns, proving the artifact path remains task-generic.
    """
    task = synthetic_task
    source_indices = [4]
    provenance: dict[str, Any] = {
        "provenance_schema_version": analysis.artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
        "run": {"name": "synthetic", "task": task.id, "best_checkpoint_sha256": "abc"},
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
            "meta": {"label": ["synthetic-case"], "quality": torch.tensor([7])},
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
        dataset_name="synthetic_train",
        provenance=provenance,
    )

    assert parquet_path.is_file()
    assert frame.columns.is_unique
    assert {
        "rel_l2",
        "rel_h1",
        "rmse_response_a",
        "rmse_response_b",
        "normalized_sse_response_a",
        "normalized_count_response_a",
        "normalized_rmse_response_a",
        "normalized_sse_response_b",
        "normalized_count_response_b",
        "normalized_rmse_response_b",
    }.issubset(frame.columns)
    assert not {
        "momentum_residual_mse",
        "div_velocity_mse",
        "div_eps_velocity_mse",
        "pressure_boundary_mse",
    }.intersection(frame.columns)
    npz_path = save_root / "npz" / "case_0005.npz"
    with np.load(npz_path, allow_pickle=False) as payload:
        assert payload["input_fields"].tolist() == list(task.input_names)
        assert payload["output_fields"].tolist() == list(task.output_names)
        assert payload["output_units"].tolist() == ["unit_out_a", "unit_out_b"]
        assert payload["artifact_fields"].tolist() == list(task.output_names)
        assert payload["artifact_units"].tolist() == ["unit_out_a", "unit_out_b"]
        metadata = json.loads(str(payload["meta"].item()))
        assert not {"case_index", "source_index", "split_local_index"}.intersection(metadata)

    stored_provenance = json.loads((save_root / analysis.artifacts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    stored_outputs = stored_provenance.pop("outputs")
    stored_aggregate = stored_provenance.pop("aggregate")
    assert stored_provenance == provenance
    assert stored_outputs == analysis.artifacts.artifact_output_manifest(save_root)
    assert stored_aggregate == analysis.artifacts.aggregate_normalized_macro_rmse(
        frame,
        output_fields=task.output_names,
    )


def test_normalized_macro_evidence_matches_runtime_and_artifacts_across_chunking(
    tmp_path: Path,
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Match macro-RMSE evidence across runtime and artifact chunking.

    Equivalent three-case inputs use two partition schemes and two loader batch
    sizes; SSE/count aggregation, row order, and stored provenance must agree.
    """
    fields = synthetic_task.output_names
    prediction = (
        torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 4.0],
                [0.0, 8.0],
            ],
            dtype=torch.float32,
        )
        .reshape(3, 2, 1, 1)
        .expand(-1, -1, 2, 2)
    )
    target = torch.zeros_like(prediction)
    definition = learning.metrics.metrics.ResolvedMetric(
        id="normalized_macro_rmse",
        kind="macro_rmse",
        space="normalized",
        fields=fields,
        field_indices=(0, 1),
        reduction="field_macro_element_mean",
        direction="minimize",
        unit="1",
        operator_dimensionality=2,
    )

    runtime_values: list[float] = []
    for chunks in ((2, 1), (1, 1, 1)):
        metric = learning.metrics.metrics.MacroRMSEMetric(
            definition,
            device=torch.device("cpu"),
        )
        start = 0
        for batch_index, size in enumerate(chunks):
            stop = start + size
            metric.update(
                prediction[start:stop],
                target[start:stop],
                space="normalized",
                batch_index=batch_index,
            )
            start = stop
        runtime_values.append(metric.compute())

    rows = analysis.artifacts.normalized_case_statistics(
        prediction,
        target,
        output_fields=fields,
    )
    aggregate = analysis.artifacts.aggregate_normalized_macro_rmse(
        pd.DataFrame(rows),
        output_fields=fields,
    )
    per_case_macro_mean = float(pd.DataFrame(rows)[[f"normalized_rmse_{field}" for field in fields]].mean(axis=1).mean())

    source_indices = [5, 1, 8]
    inputs = torch.zeros(3, len(synthetic_task.input_names), 2, 2)
    inputs[:, :2] = prediction / 2.0
    samples = [
        {
            "x": inputs[index],
            "y": target[index],
            "source_index": source_index,
            "split_local_index": index,
            "meta": {"label": f"case-{source_index}"},
        }
        for index, source_index in enumerate(source_indices)
    ]
    processor = SimpleNamespace(
        in_normalizer=_IdentityNormalizer(),
        out_normalizer=_IdentityNormalizer(),
    )
    artifact_frames: list[pd.DataFrame] = []
    artifact_values: list[float] = []
    for batch_size in (2, 1):
        root = tmp_path / f"batch-{batch_size}"
        provenance = {
            "provenance_schema_version": analysis.artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
            "run": {"name": "synthetic", "task": synthetic_task.id},
            "selection": {
                "effective_case_count": len(source_indices),
                "effective_ordered_source_indices_sha256": analysis.artifacts.ordered_indices_sha256(source_indices),
            },
            "evaluator": {
                "input_fields": list(synthetic_task.input_names),
                "output_fields": list(fields),
                "output_units": {field.name: field.unit for field in synthetic_task.outputs},
            },
        }
        frame, _ = analysis.artifacts.generate_artifacts(
            task=synthetic_task,
            model=_Projection(),
            loader=DataLoader(_MappingDataset(samples), batch_size=batch_size, shuffle=False),
            processor=processor,
            device=torch.device("cpu"),
            save_root=root,
            dataset_name="synthetic",
            provenance=provenance,
        )
        stored = json.loads((root / analysis.artifacts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        artifact_frames.append(frame)
        artifact_values.append(float(stored["aggregate"]["value"]))

    assert runtime_values[0] == pytest.approx(runtime_values[1], rel=0.0, abs=1e-15)
    assert aggregate["value"] == pytest.approx(runtime_values[0], rel=0.0, abs=1e-15)
    assert artifact_values == pytest.approx(runtime_values, rel=0.0, abs=1e-15)
    assert aggregate["value"] != pytest.approx(per_case_macro_mean)
    expected_element_count = prediction.shape[0] * prediction.shape[2] * prediction.shape[3]
    for field in fields:
        assert aggregate["field_statistics"][field]["normalized_element_count"] == expected_element_count
    pd.testing.assert_frame_equal(
        artifact_frames[0].drop(columns="npz_path"),
        artifact_frames[1].drop(columns="npz_path"),
    )
    for field in fields:
        sse_column, count_column, rmse_column = analysis.artifacts.normalized_statistic_columns(field)
        assert artifact_frames[0][sse_column].tolist() == pytest.approx([row[sse_column] for row in rows])
        assert artifact_frames[0][count_column].tolist() == [row[count_column] for row in rows]
        assert artifact_frames[0][rmse_column].tolist() == pytest.approx([row[rmse_column] for row in rows])
    assert artifact_frames[0]["source_index"].tolist() == source_indices


def test_steady_artifact_stores_dual_continuity_and_boundary_semantics(tmp_path: Path) -> None:
    """
    Keep dual continuity and pressure-boundary artifacts explicit.

    Both training selections must emit the same named scalar and residual-array
    contract, while retired ``Rc`` payloads are rejected as ambiguous.
    """
    task = domain.tasks.steady_flow.STEADY_FLOW
    height, width = 9, 11
    y_values = torch.linspace(0.0, 1.0, height)
    x_values = torch.linspace(0.0, 2.0, width)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    zeros = torch.zeros_like(x_grid)
    inputs = torch.stack(
        (
            x_grid,
            y_grid,
            torch.full_like(x_grid, -4.0),
            zeros,
            torch.full_like(x_grid, -4.0),
            0.25 + 0.25 * x_grid,
            zeros,
        ),
        dim=0,
    ).unsqueeze(0)
    targets = torch.zeros(1, 3, height, width)
    loader = [
        {
            "x": inputs,
            "y": targets,
            "source_index": torch.tensor([0]),
            "split_local_index": torch.tensor([0]),
            "meta": {"sample_id": ["manufactured"]},
        }
    ]
    processor = SimpleNamespace(
        in_normalizer=_IdentityNormalizer(),
        out_normalizer=_IdentityNormalizer(),
    )
    frames: list[pd.DataFrame] = []

    for continuity in ("div_velocity", "div_eps_velocity"):
        root = tmp_path / continuity
        provenance = {
            "provenance_schema_version": analysis.artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
            "run": {"name": continuity, "task": task.id},
            "selection": {
                "effective_case_count": 1,
                "effective_ordered_source_indices_sha256": analysis.artifacts.ordered_indices_sha256([0]),
            },
            "evaluator": {
                "input_fields": list(task.input_names),
                "output_fields": list(task.output_names),
                "output_units": {field.name: field.unit for field in task.outputs},
                "physics_kind": task.physics.kind,
            },
            "physics": {"selected_training_continuity": continuity},
        }
        frame, _parquet_path = analysis.artifacts.generate_artifacts(
            task=task,
            model=_SteadyProjection(),
            loader=loader,
            processor=processor,
            device=torch.device("cpu"),
            save_root=root,
            dataset_name="steady",
            provenance=provenance,
        )
        frames.append(frame)
        row = frame.iloc[0]
        required_scalars = {
            "momentum_residual_mse",
            "div_velocity_mse",
            "div_eps_velocity_mse",
            "pressure_boundary_mse",
            "pressure_inlet_mse",
            "pressure_outlet_mean_square",
        }
        assert required_scalars.issubset(frame.columns)
        assert not {"cont_mse", "continuity_mse", "mom_mse", "bc_mse"}.intersection(frame.columns)
        with np.load(Path(row["npz_path"]), allow_pickle=False) as payload:
            assert {"Rx", "Ry", "div_u", "div_eps_u", "coordinates"}.issubset(payload.files)
            assert "Rc" not in payload.files
            for name in ("Rx", "Ry", "div_u", "div_eps_u"):
                assert payload[name].shape == (height, width)
                assert np.issubdtype(payload[name].dtype, np.floating)
                assert np.isfinite(payload[name]).all()
            crop = analysis.artifacts.EVAL_PAD
            interior = np.s_[crop:-crop, crop:-crop]
            expected_momentum = float(np.mean(payload["Rx"][interior] ** 2 + payload["Ry"][interior] ** 2))
            assert row["momentum_residual_mse"] == pytest.approx(expected_momentum)
            assert row["div_velocity_mse"] == pytest.approx(float(np.mean(payload["div_u"][interior] ** 2)))
            assert row["div_eps_velocity_mse"] == pytest.approx(float(np.mean(payload["div_eps_u"][interior] ** 2)))
            assert row["div_velocity_mse"] != pytest.approx(row["div_eps_velocity_mse"])
            pressure = payload["pred"][0]
            pressure_boundary = payload["p_bc"][0]
            expected_inlet = float(np.mean((pressure[0] - pressure_boundary[0]) ** 2))
            expected_outlet_mean_square = float(np.mean(pressure[-1]) ** 2)
            outlet_pointwise_mse = float(np.mean(pressure[-1] ** 2))
        assert row["pressure_inlet_mse"] == pytest.approx(expected_inlet)
        assert row["pressure_outlet_mean_square"] == pytest.approx(expected_outlet_mean_square)
        assert row["pressure_outlet_mean_square"] != pytest.approx(outlet_pointwise_mse)
        assert row["pressure_boundary_mse"] == pytest.approx(row["pressure_inlet_mse"] + row["pressure_outlet_mean_square"])

    old_npz_path = Path(frames[0].iloc[0]["npz_path"])
    with np.load(old_npz_path, allow_pickle=False) as stored:
        old_payload = {name: np.asarray(stored[name]) for name in stored.files}
    old_payload["Rc"] = old_payload["div_eps_u"]
    with old_npz_path.open("wb") as stream:
        np.savez_compressed(stream, **old_payload)  # pyright: ignore[reportArgumentType]
    contract = analysis.artifact_service._EvaluatorArtifactContract(
        task_id=task.id,
        input_fields=task.input_names,
        output_fields=task.output_names,
        output_units=tuple(field.unit for field in task.outputs),
        physics_kind=task.physics.kind,
    )
    with pytest.raises(analysis.artifact_service.ArtifactCacheError, match=r"unexpected=\['Rc'\]"):
        analysis.artifact_service._validate_npz_payload(
            old_npz_path,
            case_index=1,
            source_index=0,
            split_local_index=0,
            contract=contract,
        )

    comparable = [
        "momentum_residual_mse",
        "div_velocity_mse",
        "div_eps_velocity_mse",
        "pressure_boundary_mse",
    ]
    assert frames[0].loc[:, comparable].iloc[0].tolist() == pytest.approx(frames[1].loc[:, comparable].iloc[0].tolist())


def test_physical_cross_permeability_is_reconstructed_from_its_ratio() -> None:
    """
    Reconstruct physical cross-permeability from its encoded ratio.

    The original tensor must remain unchanged while ``kxy`` is scaled by the
    diagonal permeability convention into square metres for artifact consumers.
    """
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
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Reject source metadata that duplicates artifact identity.

    A forged ``case_index`` must raise before publication so user metadata cannot
    replace the identity derived from ordered dataset membership.
    """
    task = synthetic_task
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
            dataset_name="synthetic_train",
            provenance=provenance,
        )
    except KeyError as error:
        assert "reserved artifact identity" in str(error)
    else:
        raise AssertionError("reserved identity metadata was accepted")


def test_synthetic_task_flows_through_generic_dataset_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Flow a distinct valid task through generic dataset boundaries.

    Case validation, merge, reload, EDA flattening, and batched metadata must all
    preserve synthetic-task fields and copy isolation without steady-flow coupling.
    """
    input_fields = {name: torch.full((2, 3), _SYNTHETIC_INPUT_VALUE + index) for index, name in enumerate(synthetic_task.input_names)}
    output_fields = {name: torch.full((2, 3), 20.0 + index) for index, name in enumerate(synthetic_task.output_names)}
    case = datasets.identity.build_case_payload(
        task=synthetic_task,
        case_id="case_0000",
        input_fields=input_fields,
        output_fields=output_fields,
        source_identity={"token": "synthetic"},
        source_metadata={
            "generator": {
                "parameters": {"scalar_parameter": _SYNTHETIC_METADATA_VALUE},
            },
        },
    )
    validated = datasets.identity.validate_case_payload(case, task=synthetic_task)
    merged = datasets.identity.build_merged_dataset_payload(
        task=synthetic_task,
        dataset_id="synthetic_train",
        sample_ids=("case_0000",),
        source_identities=(validated.source_identity,),
        source_metadata=(validated.source_metadata,),
        case_fingerprints=(validated.fingerprint,),
        inputs=validated.inputs.unsqueeze(0),
        outputs=validated.outputs.unsqueeze(0),
    )
    path = _save_dataset(tmp_path, merged)
    loaded = datasets.simulation.create_task_dataset(path, task=synthetic_task)
    expected_metadata = {
        "generator": {
            "parameters": {"scalar_parameter": _SYNTHETIC_METADATA_VALUE},
        },
    }
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    frame, logs = analysis.eda.dataframe.generate_eda_dataframe(
        "synthetic_train",
        task=synthetic_task,
        max_cases=1,
    )
    assert frame.index.tolist() == ["case_0000"]
    assert frame.columns.tolist() == [*synthetic_task.input_names, *synthetic_task.output_names, "meta"]
    assert np.all(frame.loc["case_0000", "feature_a"] == _SYNTHETIC_INPUT_VALUE)
    assert frame.loc["case_0000", "meta"] == expected_metadata
    assert str(path) in "\n".join(logs)

    sample = loaded[0]
    assert sample["meta"] == expected_metadata
    sample["meta"]["generator"]["parameters"]["scalar_parameter"] = -1.0
    assert loaded[0]["meta"] == expected_metadata

    batch = next(iter(DataLoader(loaded, batch_size=1, shuffle=False)))
    artifact_metadata = analysis.artifacts.meta_to_jsonable(batch["meta"])
    flattened_metadata = analysis.evaluation.dataframe.flatten_meta_scalars(artifact_metadata)

    assert loaded.input_fields == list(synthetic_task.input_names)
    assert loaded.output_fields == list(synthetic_task.output_names)
    assert flattened_metadata["generator_parameters_scalar_parameter"] == _SYNTHETIC_METADATA_VALUE
    assert loaded[0]["x"].shape == (3, 2, 3)
    assert torch.all(loaded[0]["x"][0] == _SYNTHETIC_INPUT_VALUE)
    assert synthetic_task.physics.kind == "none"
    assert [metric.fields for metric in synthetic_task.default_metrics[1:]] == [
        ("response_a",),
        ("response_b",),
    ]
