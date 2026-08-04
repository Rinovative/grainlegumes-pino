# ruff: noqa: S101, EM101, PT017, SLF001, TRY003
"""
Verify task-generic and steady-flow artifact generation against current provenance.

The suite checks field/unit propagation, normalized SSE/count equivalence across
batch partitions, dual continuity, boundary naming, physical permeability, and
reserved metadata rejection. Cache locking and rebuild races belong to
``test_artifact_identity``. Visualization behavior is outside this module.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import torch
from src import analysis, datasets, domain, learning
from support.synthetic_task import build_synthetic_generated_batch_identity
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from pathlib import Path

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
    """Leave tensors unchanged while exposing unit fitted scales."""

    def __init__(self, channels: int) -> None:
        """Store unit channel scales in the saved-normalizer layout."""
        self.std = torch.ones(1, channels, 1, 1)

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        """Return normalized-space input unchanged."""
        return value

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        """Return physical-space output unchanged."""
        return value


class _IdentityProcessor:
    """Model the fitted processor lifecycle used by online evaluation."""

    def __init__(self, task: domain.tasks.spec.TaskSpec) -> None:
        """Create task-shaped identity normalizers and call traces."""
        self.in_normalizer = _IdentityNormalizer(task.in_channels)
        self.out_normalizer = _IdentityNormalizer(task.out_channels)
        self.training = True
        self.preprocessed_batch_sizes: list[int] = []

    def eval(self) -> None:
        """Switch preprocessing to evaluation semantics."""
        self.training = False

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize complete-batch inputs while preserving physical targets."""
        if self.training:
            raise AssertionError("artifact preprocessing did not enter evaluation mode")
        inputs = batch["x"]
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("fixture inputs must be tensors")
        self.preprocessed_batch_sizes.append(int(inputs.shape[0]))
        return {**batch, "x": self.in_normalizer.transform(inputs)}


class _Projection(nn.Module):
    """
    Project synthetic-task inputs into two deterministic output fields.

    The fixed linear mapping gives the tests exact predictions while exercising
    the ordinary ``nn.Module`` inference boundary used by artifact generation.
    """

    def __init__(self) -> None:
        """Initialize the observed complete-batch forward sizes."""
        super().__init__()
        self.forward_batch_sizes: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map the leading two input channels to doubled predictions."""
        self.forward_batch_sizes.append(int(value.shape[0]))
        return 2.0 * value[:, :2]


class _SteadyProjection(nn.Module):
    """Return a deterministic task-ordered manufactured steady-flow state."""

    def __init__(self, task: domain.tasks.spec.TaskSpec) -> None:
        super().__init__()
        self.task = task

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Construct pressure and velocity values by TaskSpec field names."""
        coordinate = next(field.name for field in self.task.inputs if field.role == "coordinate")
        coordinate_values = value[:, self.task.input_names.index(coordinate)]
        zeros = torch.zeros_like(coordinate_values)
        pressure = next(group for group in self.task.output_groups if group.id == "pressure")
        velocity = next(group for group in self.task.output_groups if group.id == "velocity")
        output_by_field = dict.fromkeys(self.task.output_names, zeros)
        output_by_field[pressure.fields[0]] = coordinate_values
        output_by_field[velocity.fields[0]] = coordinate_values
        return torch.stack([output_by_field[field] for field in self.task.output_names], dim=1)


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


def _attach_objective_contract(
    provenance: dict[str, Any],
    *,
    task: domain.tasks.spec.TaskSpec,
) -> None:
    """Attach task-owned groups and unit saved-normalizer evidence in place."""
    selection = provenance.get("selection")
    if not isinstance(selection, dict):
        raise TypeError("fixture selection must be a dictionary")
    count = selection.get("effective_case_count")
    digest = selection.get("effective_ordered_source_indices_sha256")
    selection.update(
        {
            "full_selected_case_count": count,
            "generation_limit": None,
            "full_ordered_source_indices_sha256": digest,
        }
    )
    evaluator = provenance.setdefault("evaluator", {})
    if not isinstance(evaluator, dict):
        raise TypeError("fixture evaluator must be a dictionary")
    evaluator.update(
        {
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
            "objective": next(metric for metric in task.default_metrics if metric.kind == "group_macro_rmse").as_dict(all_fields=task.output_names),
            "output_groups": analysis.artifacts.contracts.output_group_payload(task.output_groups),
        }
    )
    provenance["normalizer"] = {
        "denominator_floor": 0.0,
        "output_standard_deviations": dict.fromkeys(task.output_names, 1.0),
    }


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
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "run": {"name": "synthetic", "task": task.id, "best_checkpoint_sha256": "abc"},
        "selection": {
            "effective_case_count": 1,
            "effective_ordered_source_indices_sha256": analysis.artifacts.contracts.ordered_indices_sha256(source_indices),
        },
        "evaluator": {
            "input_fields": list(task.input_names),
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
        },
    }
    _attach_objective_contract(provenance, task=task)
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
    processor = _IdentityProcessor(task)
    save_root = tmp_path / "analysis" / "id"

    frame, parquet_path = analysis.artifacts.generation.generate_artifacts(
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
        "physical_rmse_response_a",
        "physical_rmse_response_b",
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

    stored_provenance = json.loads((save_root / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    stored_outputs = stored_provenance.pop("outputs")
    stored_aggregate = stored_provenance.pop("aggregate")
    assert stored_provenance == provenance
    assert stored_outputs == analysis.artifacts.contracts.artifact_output_manifest(save_root)
    assert stored_aggregate == analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
        frame,
        output_groups=task.output_groups,
        train_standard_deviations=dict.fromkeys(task.output_names, 1.0),
        normalization_denominator_floor=0.0,
    )


def test_group_objective_matches_online_and_artifacts_across_partitions(
    tmp_path: Path,
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Match group-objective evidence across runtime and artifact chunking.

    Equivalent three-case inputs use two partition schemes and two loader batch
    sizes. SSE/count aggregation, row order, and stored provenance must agree.
    """
    task = synthetic_task
    fields = task.output_names
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
    objective_spec = next(metric for metric in task.default_metrics if metric.id == "normalized_group_macro_rmse")
    train_standard_deviations = dict.fromkeys(fields, 1.0)
    definition = learning.metrics.metrics.ResolvedMetric(
        id=objective_spec.id,
        kind=objective_spec.kind,
        space=objective_spec.space,
        fields=fields,
        field_indices=tuple(range(len(fields))),
        reduction=objective_spec.reduction,
        direction=objective_spec.direction,
        unit="1",
        operator_dimensionality=2,
        groups=task.output_groups,
        field_standard_deviations=tuple(train_standard_deviations[field] for field in fields),
    )

    runtime_values: list[float] = []
    for chunks in ((2, 1), (1, 1, 1)):
        metric = learning.metrics.metrics.GroupMacroRMSEMetric(
            definition,
            device=torch.device("cpu"),
        )
        start = 0
        for batch_index, size in enumerate(chunks):
            stop = start + size
            metric.update(
                prediction[start:stop],
                target[start:stop],
                space="physical",
                batch_index=batch_index,
            )
            start = stop
        runtime_values.append(metric.compute())

    physical_rows = analysis.artifacts.generation.physical_case_statistics(
        prediction,
        target,
        output_fields=fields,
    )
    normalized_rows = analysis.artifacts.generation.normalized_case_statistics(
        prediction,
        target,
        output_fields=fields,
    )
    rows = [{**physical_row, **normalized_row} for physical_row, normalized_row in zip(physical_rows, normalized_rows, strict=True)]
    aggregate = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
        pd.DataFrame(rows),
        output_groups=task.output_groups,
        train_standard_deviations=train_standard_deviations,
        normalization_denominator_floor=0.0,
    )
    per_case_macro_mean = float(pd.DataFrame(rows)[[f"normalized_rmse_{field}" for field in fields]].mean(axis=1).mean())

    source_indices = [5, 1, 8]
    inputs = torch.zeros(3, len(task.input_names), 2, 2)
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
    artifact_frames: list[pd.DataFrame] = []
    artifact_values: list[float] = []
    forward_batch_sizes: list[list[int]] = []
    preprocessed_batch_sizes: list[list[int]] = []
    for batch_size in (2, 1):
        root = tmp_path / f"batch-{batch_size}"
        provenance = {
            "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
            "run": {"name": "synthetic", "task": task.id},
            "selection": {
                "effective_case_count": len(source_indices),
                "effective_ordered_source_indices_sha256": analysis.artifacts.contracts.ordered_indices_sha256(source_indices),
            },
            "evaluator": {
                "input_fields": list(task.input_names),
                "output_fields": list(fields),
                "output_units": {field.name: field.unit for field in task.outputs},
            },
        }
        _attach_objective_contract(provenance, task=task)
        processor = _IdentityProcessor(task)
        model = _Projection()
        frame, _ = analysis.artifacts.generation.generate_artifacts(
            task=task,
            model=model,
            loader=DataLoader(_MappingDataset(samples), batch_size=batch_size, shuffle=False),
            processor=processor,
            device=torch.device("cpu"),
            save_root=root,
            dataset_name="synthetic",
            provenance=provenance,
        )
        stored = json.loads((root / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        artifact_frames.append(frame)
        artifact_values.append(float(stored["aggregate"]["value"]))
        forward_batch_sizes.append(model.forward_batch_sizes)
        preprocessed_batch_sizes.append(processor.preprocessed_batch_sizes)

    assert forward_batch_sizes == [[2, 1], [1, 1, 1]]
    assert preprocessed_batch_sizes == forward_batch_sizes
    assert runtime_values[0] == pytest.approx(runtime_values[1], rel=0.0, abs=1e-15)
    assert aggregate["value"] == pytest.approx(runtime_values[0], rel=0.0, abs=1e-15)
    assert artifact_values == pytest.approx(runtime_values, rel=0.0, abs=1e-15)
    assert aggregate["value"] != pytest.approx(per_case_macro_mean)
    expected_element_count = prediction.shape[0] * prediction.shape[2] * prediction.shape[3]
    for field in fields:
        assert aggregate["field_statistics"][field]["normalized_element_count"] == expected_element_count
    pd.testing.assert_frame_equal(
        artifact_frames[0].drop(columns=["npz_path", "inference_time_ms"]),
        artifact_frames[1].drop(columns=["npz_path", "inference_time_ms"]),
    )
    for field in fields:
        for columns in (
            analysis.artifacts.contracts.physical_statistic_columns(field),
            analysis.artifacts.contracts.normalized_statistic_columns(field),
        ):
            sse_column, count_column, rmse_column = columns
            assert artifact_frames[0][sse_column].tolist() == pytest.approx([row[sse_column] for row in rows])
            assert artifact_frames[0][count_column].tolist() == [row[count_column] for row in rows]
            assert artifact_frames[0][rmse_column].tolist() == pytest.approx([row[rmse_column] for row in rows])
    assert artifact_frames[0]["source_index"].tolist() == source_indices


def test_steady_artifact_stores_dual_continuity_and_boundary_semantics(tmp_path: Path) -> None:
    """
    Keep dual continuity and pressure-boundary artifacts explicit.

    Both training selections must emit the same named scalar and residual-array
    contract, while undeclared NPZ arrays fail the generic exact-schema check.
    """
    task = domain.tasks.steady_flow.STEADY_FLOW
    groups_by_id = {group.id: group for group in task.output_groups}
    (pressure_field,) = groups_by_id["pressure"].fields
    velocity_fields = groups_by_id["velocity"].fields
    pressure_index = task.output_names.index(pressure_field)
    velocity_indices = [task.output_names.index(field) for field in velocity_fields]
    velocity_units = {task.field(field).unit for field in velocity_fields}
    assert len(velocity_units) == 1
    velocity_unit = next(iter(velocity_units))
    height, width = 9, 11
    y_values = torch.linspace(0.0, 1.0, height)
    x_values = torch.linspace(0.0, 2.0, width)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    zeros = torch.zeros_like(x_grid)
    coordinate_values = iter((x_grid, y_grid))
    input_by_field: dict[str, torch.Tensor] = {}
    for field in task.inputs:
        if field.role == "coordinate":
            input_by_field[field.name] = next(coordinate_values)
        elif field.role == "permeability" and "cross_component" in field.representation:
            input_by_field[field.name] = zeros
        elif field.role == "permeability":
            input_by_field[field.name] = torch.full_like(x_grid, -4.0)
        elif field.role == "porosity":
            input_by_field[field.name] = 0.25 + 0.25 * x_grid
        elif field.role == "boundary":
            input_by_field[field.name] = zeros
        else:
            msg = f"unsupported steady-flow fixture role: {field.role}"
            raise AssertionError(msg)
    inputs = torch.stack([input_by_field[field] for field in task.input_names], dim=0).unsqueeze(0)
    targets = torch.zeros((1, task.out_channels, height, width))
    loader = [
        {
            "x": inputs,
            "y": targets,
            "source_index": torch.tensor([0]),
            "split_local_index": torch.tensor([0]),
            "meta": {"sample_id": ["manufactured"]},
        }
    ]
    processor = _IdentityProcessor(task)
    frames: list[pd.DataFrame] = []

    for continuity in ("div_velocity", "div_eps_velocity"):
        root = tmp_path / continuity
        provenance = {
            "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
            "run": {"name": continuity, "task": task.id},
            "selection": {
                "effective_case_count": 1,
                "effective_ordered_source_indices_sha256": analysis.artifacts.contracts.ordered_indices_sha256([0]),
            },
            "evaluator": {
                "input_fields": list(task.input_names),
                "output_fields": list(task.output_names),
                "output_units": {field.name: field.unit for field in task.outputs},
                "physics_kind": task.physics.kind,
            },
            "physics": {"selected_training_continuity": continuity},
        }
        _attach_objective_contract(provenance, task=task)
        frame, _parquet_path = analysis.artifacts.generation.generate_artifacts(
            task=task,
            model=_SteadyProjection(task),
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
        payload_path = analysis.artifacts.contracts.resolve_case_payload_path(
            root,
            row["npz_path"],
            expected_filename=f"case_{int(row['case_index']):04d}.npz",
        )
        with np.load(payload_path, allow_pickle=False) as payload:
            assert {"Rx", "Ry", "div_u", "div_eps_u", "coordinates"}.issubset(payload.files)
            for name in ("Rx", "Ry", "div_u", "div_eps_u"):
                assert payload[name].shape == (height, width)
                assert np.issubdtype(payload[name].dtype, np.floating)
                assert np.isfinite(payload[name]).all()
            crop = analysis.artifacts.contracts.EVAL_PAD
            interior = np.s_[crop:-crop, crop:-crop]
            expected_momentum = float(np.mean(payload["Rx"][interior] ** 2 + payload["Ry"][interior] ** 2))
            assert row["momentum_residual_mse"] == pytest.approx(expected_momentum)
            assert row["div_velocity_mse"] == pytest.approx(float(np.mean(payload["div_u"][interior] ** 2)))
            assert row["div_eps_velocity_mse"] == pytest.approx(float(np.mean(payload["div_eps_u"][interior] ** 2)))
            assert row["div_velocity_mse"] != pytest.approx(row["div_eps_velocity_mse"])
            expected_speed = np.sqrt(np.square(payload["pred"][velocity_indices]).sum(axis=0))
            np.testing.assert_allclose(payload["pred"][-1], expected_speed)
            assert payload["artifact_units"][-1] == velocity_unit
            pressure = payload["pred"][pressure_index]
            pressure_boundary = payload["p_bc"][0]
            expected_inlet = float(np.mean((pressure[0] - pressure_boundary[0]) ** 2))
            expected_outlet_mean_square = float(np.mean(pressure[-1]) ** 2)
            outlet_pointwise_mse = float(np.mean(pressure[-1] ** 2))
        assert row["pressure_inlet_mse"] == pytest.approx(expected_inlet)
        assert row["pressure_outlet_mean_square"] == pytest.approx(expected_outlet_mean_square)
        assert row["pressure_outlet_mean_square"] != pytest.approx(outlet_pointwise_mse)
        assert row["pressure_boundary_mse"] == pytest.approx(row["pressure_inlet_mse"] + row["pressure_outlet_mean_square"])

    invalid_row = frames[0].iloc[0]
    invalid_npz_path = analysis.artifacts.contracts.resolve_case_payload_path(
        tmp_path / "div_velocity",
        invalid_row["npz_path"],
        expected_filename=f"case_{int(invalid_row['case_index']):04d}.npz",
    )
    with np.load(invalid_npz_path, allow_pickle=False) as stored:
        unexpected_array = np.asarray(stored["div_eps_u"])
    with io.BytesIO() as stream:
        np.save(stream, unexpected_array, allow_pickle=False)
        unexpected_payload = stream.getvalue()
    with zipfile.ZipFile(invalid_npz_path, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("unexpected_array.npy", unexpected_payload)
    contract = analysis.artifacts.service._EvaluatorArtifactContract(
        task_id=task.id,
        input_fields=task.input_names,
        output_fields=task.output_names,
        output_units=tuple(field.unit for field in task.outputs),
        output_groups=tuple((group.id, group.fields) for group in task.output_groups),
        train_standard_deviations=dict.fromkeys(task.output_names, 1.0),
        normalization_denominator_floor=0.0,
        group_metric_columns=("normalized_velocity_vector_rmse", "physical_velocity_vector_rmse"),
        physics_kind=task.physics.kind,
    )
    with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match=r"unexpected=\['unexpected_array'\]"):
        analysis.artifacts.service._validate_npz_payload(
            invalid_npz_path,
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
    permeability = analysis.artifacts.generation.extract_kappa(
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
            "effective_ordered_source_indices_sha256": analysis.artifacts.contracts.ordered_indices_sha256([0]),
        }
    }
    _attach_objective_contract(provenance, task=task)
    loader = [
        {
            "x": torch.zeros(1, 3, 2, 2),
            "y": torch.zeros(1, 2, 2, 2),
            "source_index": torch.tensor([0]),
            "split_local_index": torch.tensor([0]),
            "meta": {"case_index": 99},
        }
    ]
    processor = _IdentityProcessor(task)

    try:
        analysis.artifacts.generation.generate_artifacts(
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


def test_synthetic_task_flows_through_final_dataset_contract(
    tmp_path: Path,
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Final dataset loading and metadata flattening remain task-generic."""
    inputs = torch.stack([torch.full((2, 3), _SYNTHETIC_INPUT_VALUE + index) for index in range(synthetic_task.in_channels)]).unsqueeze(0)
    outputs = torch.stack([torch.full((2, 3), 20.0 + index) for index in range(synthetic_task.out_channels)]).unsqueeze(0)
    source_identity = {"token": "synthetic"}
    expected_metadata = {
        "generator": {"parameters": {"scalar_parameter": _SYNTHETIC_METADATA_VALUE}},
    }
    fingerprint = datasets.identity.compute_case_fingerprint(
        task=synthetic_task,
        case_id="case_0000",
        source_identity=source_identity,
        source_metadata=expected_metadata,
        inputs=inputs[0],
        outputs=outputs[0],
    )
    payload = datasets.identity.build_training_dataset_payload(
        task=synthetic_task,
        dataset_id="synthetic_train",
        sample_ids=("case_0000",),
        generated_batch_identity=build_synthetic_generated_batch_identity(
            batch_name="synthetic_train",
            sample_ids=("case_0000",),
        ),
        source_identities=(source_identity,),
        source_metadata=(expected_metadata,),
        source_provenance={"batch_manifest_sha256": "2" * 64},
        case_fingerprints=(fingerprint,),
        inputs=inputs,
        outputs=outputs,
    )
    loaded = datasets.simulation.create_task_dataset(_save_dataset(tmp_path, payload), task=synthetic_task)
    sample = loaded[0]
    assert sample["meta"] == expected_metadata
    sample["meta"]["generator"]["parameters"]["scalar_parameter"] = -1.0
    assert loaded[0]["meta"] == expected_metadata

    batch = next(iter(DataLoader(loaded, batch_size=1, shuffle=False)))
    artifact_metadata = analysis.artifacts.generation.meta_to_jsonable(batch["meta"])
    flattened_metadata = analysis.evaluation.dataframe.flatten_meta_scalars(artifact_metadata)

    assert loaded.input_fields == list(synthetic_task.input_names)
    assert loaded.output_fields == list(synthetic_task.output_names)
    assert flattened_metadata["generator_parameters_scalar_parameter"] == _SYNTHETIC_METADATA_VALUE
    assert loaded[0]["x"].shape == (3, 2, 3)
    assert torch.all(loaded[0]["x"][0] == _SYNTHETIC_INPUT_VALUE)
    assert synthetic_task.physics.kind == "none"
    assert tuple(field for group in synthetic_task.output_groups for field in group.fields) == synthetic_task.output_names
