# ruff: noqa: S101
"""Verify bounded evaluation-session identity, reuse, and numerical evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src import analysis
from src.domain.tasks import domain_task_steady_flow

_TASK_SPEC = domain_task_steady_flow.STEADY_FLOW
_TRAIN_STANDARD_DEVIATIONS = {field: float(2**index) for index, field in enumerate(_TASK_SPEC.output_names)}
_NORMALIZATION_DENOMINATOR_FLOOR = 0.0


def _resolved_metrics() -> list[dict[str, object]]:
    """Return task-owned metric declarations with explicit fields."""
    metrics: list[dict[str, object]] = []
    for metric in _TASK_SPEC.default_metrics:
        payload = metric.as_dict(all_fields=_TASK_SPEC.output_names)
        if payload["fields"] == "all":
            payload["fields"] = list(_TASK_SPEC.output_names)
        metrics.append(payload)
    return metrics


def _field_names_and_units() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return ordered input/output semantics from the maintained task contract."""
    return (
        _TASK_SPEC.input_names,
        tuple(field.unit for field in _TASK_SPEC.inputs),
        _TASK_SPEC.output_names,
        tuple(field.unit for field in _TASK_SPEC.outputs),
    )


def _provenance(raw: pd.DataFrame, *, role: str, run_name: str) -> dict[str, object]:
    """Build complete current provenance for one deterministic fixture role."""
    input_fields, input_units, output_fields, output_units = _field_names_and_units()
    aggregate = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
        raw,
        output_groups=_TASK_SPEC.output_groups,
        train_standard_deviations=_TRAIN_STANDARD_DEVIATIONS,
        normalization_denominator_floor=_NORMALIZATION_DENOMINATOR_FLOOR,
    )
    return {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "split_role": role,
        "run": {
            "name": run_name,
            "task": _TASK_SPEC.id,
            "task_contract_digest": _TASK_SPEC.contract_digest,
            "effective_config_digest": f"config-{run_name}",
            "best_checkpoint_sha256": f"checkpoint-{run_name}",
            "normalizer_sha256": f"normalizer-{run_name}",
        },
        "model": {
            "kind": "fno",
            "architecture": {"hidden_channels": 8, "n_layers": 2, "n_modes": [4, 4]},
            "parameter_counts": {"total": 1300, "trainable": 1200},
            "physics_enabled": True,
        },
        "dataset": {
            "name": f"fixture-{role}",
            "fingerprint": f"fingerprint-{run_name}-{role}",
            "data_contract_digest": _TASK_SPEC.data_contract_digest,
            "saved_membership_digest": f"saved-{run_name}-{role}",
        },
        "selection": {
            "effective_case_count": len(raw),
            "effective_ordered_source_indices_sha256": f"effective-{run_name}-{role}",
        },
        "normalizer": {
            "denominator_floor": _NORMALIZATION_DENOMINATOR_FLOOR,
            "output_standard_deviations": dict(_TRAIN_STANDARD_DEVIATIONS),
        },
        "evaluator": {
            "metrics": _resolved_metrics(),
            "input_fields": list(input_fields),
            "input_units": dict(zip(input_fields, input_units, strict=True)),
            "output_fields": list(output_fields),
            "output_units": dict(zip(output_fields, output_units, strict=True)),
            "output_groups": analysis.artifacts.contracts.output_group_payload(_TASK_SPEC.output_groups),
            "objective": next(metric for metric in _resolved_metrics() if metric["id"] == analysis.evaluation.dataframe.PRIMARY_OBJECTIVE_ID),
            "predictive_metrics": ["rel_l2", "rel_h1"],
        },
        "aggregate": aggregate,
        "outputs": {"fixture_manifest_digest": f"manifest-{run_name}-{role}"},
        "physics": {
            "residual_schema_version": analysis.artifacts.contracts.RESIDUAL_SCHEMA_VERSION,
            "equation_kind": _TASK_SPEC.physics.kind,
            "boundary_condition_kind": _TASK_SPEC.physics.boundary,
            "derivatives": "central_difference",
            "interior_crop": 1,
            "scalar_definitions": {"dual_continuity": True},
            "array_definitions": {
                "Rx": "full_grid",
                "Ry": "full_grid",
                "div_u": "full_grid",
                "div_eps_u": "full_grid",
            },
            "residual_evaluation_region": "interior_crop",
            "selected_training_continuity": _TASK_SPEC.physics.continuity,
        },
    }


def _frame(
    root: Path,
    *,
    role: str,
    bias: float,
    run_name: str = "run-shared",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create schema-valid cases and return their exact q90 error map."""
    input_fields, _input_units, output_fields, output_units = _field_names_and_units()
    artifact_root = root / role
    npz_root = artifact_root / "npz"
    npz_root.mkdir(parents=True)
    y_grid, x_grid = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 2.0, 5), indexing="ij")
    rows: list[dict[str, object]] = []
    errors: list[np.ndarray] = []
    for position in range(2):
        source_index = position + (10 if role == "ood" else 0)
        pressure_group, velocity_group = _TASK_SPEC.output_groups
        pressure_field = pressure_group.fields[0]
        velocity_reference = (
            np.sin(np.pi * x_grid),
            0.5 * np.cos(np.pi * y_grid),
        )
        velocity_pattern = (
            np.sin(2.0 * np.pi * x_grid),
            np.cos(2.0 * np.pi * y_grid),
        )
        reference_by_field = {
            pressure_field: 2.0 - y_grid,
            **dict(zip(velocity_group.fields, velocity_reference, strict=True)),
        }
        pattern_by_field = {
            pressure_field: np.ones_like(x_grid),
            **dict(zip(velocity_group.fields, velocity_pattern, strict=True)),
        }
        reference = np.stack([reference_by_field[field] for field in output_fields])
        pattern = np.stack([pattern_by_field[field] for field in output_fields])
        error = bias * (position + 1) * pattern
        prediction = reference + error
        errors.append(error)
        p_bc = 2.0 - y_grid
        coordinate_values = iter((x_grid, y_grid))
        diagonal_permeability_values = iter((1e-4, 2e-4))
        input_by_field: dict[str, np.ndarray] = {}
        for field_spec in _TASK_SPEC.inputs:
            if field_spec.role == "coordinate":
                input_by_field[field_spec.name] = next(coordinate_values)
            elif field_spec.role == "permeability" and "cross_component" in field_spec.representation:
                input_by_field[field_spec.name] = np.zeros_like(x_grid)
            elif field_spec.role == "permeability":
                input_by_field[field_spec.name] = np.full_like(x_grid, next(diagonal_permeability_values))
            elif field_spec.role == "porosity":
                input_by_field[field_spec.name] = 0.35 + 0.05 * x_grid
            elif field_spec.role == "boundary":
                input_by_field[field_spec.name] = p_bc
            else:
                msg = f"unsupported fixture input role: {field_spec.role}"
                raise AssertionError(msg)
        inputs = np.stack([input_by_field[field] for field in input_fields])
        residual_scale = bias * (position + 1)
        residuals = {
            "Rx": residual_scale * np.sin(np.pi * x_grid),
            "Ry": residual_scale * np.cos(np.pi * y_grid),
            "div_u": residual_scale * np.sin(np.pi * y_grid),
            "div_eps_u": 1.5 * residual_scale * np.cos(np.pi * x_grid),
        }
        metadata = {
            "reynolds": float(position + 1),
            "conditions": {"roughness": float(0.2 + position)},
        }
        npz_path = npz_root / f"case_{source_index + 1:04d}.npz"
        permeability_fields = tuple(field.name for field in _TASK_SPEC.inputs if field.role == "permeability")
        permeability = np.stack([input_by_field[field] for field in permeability_fields])
        np.savez_compressed(
            npz_path,
            case_index=np.asarray(source_index + 1),
            source_index=np.asarray(source_index),
            split_local_index=np.asarray(position),
            pred=prediction,
            gt=reference,
            err=error,
            artifact_fields=np.asarray(output_fields),
            artifact_units=np.asarray(output_units),
            input_fields=np.asarray(input_fields),
            output_fields=np.asarray(output_fields),
            output_units=np.asarray(output_units),
            x_raw=inputs,
            y_raw=reference,
            meta=np.asarray(json.dumps(metadata)),
            kappa_encoded=permeability,
            kappa=permeability,
            kappa_names=np.asarray(permeability_fields),
            p_bc=p_bc[None],
            coordinates=np.stack((x_grid, y_grid)),
            Rx=residuals["Rx"],
            Ry=residuals["Ry"],
            div_u=residuals["div_u"],
            div_eps_u=residuals["div_eps_u"],
        )
        pressure_index = output_fields.index(pressure_field)
        velocity_indices = [output_fields.index(field) for field in velocity_group.fields]
        pressure_inlet_mse = float(np.mean(error[pressure_index, 0] ** 2))
        pressure_outlet_mean_square = float(np.mean(prediction[pressure_index, -1]) ** 2)
        row: dict[str, object] = {
            "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
            "task_id": _TASK_SPEC.id,
            "output_fields": list(output_fields),
            "output_units": list(output_units),
            "case_index": source_index + 1,
            "source_index": source_index,
            "split_local_index": position,
            "npz_path": str(npz_path),
            "meta": json.dumps(metadata),
            "inference_time_ms": 1.0,
            "rel_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
            "rel_h1": float(1.2 * np.linalg.norm(error) / np.linalg.norm(reference)),
            "physical_rmse_speed_magnitude": float(
                np.sqrt(
                    np.mean(
                        (np.linalg.vector_norm(prediction[velocity_indices], axis=0) - np.linalg.vector_norm(reference[velocity_indices], axis=0))
                        ** 2
                    )
                )
            ),
            "kappa_names": list(permeability_fields),
            "momentum_residual_mse": float(np.mean(residuals["Rx"] ** 2 + residuals["Ry"] ** 2)),
            "div_velocity_mse": float(np.mean(residuals["div_u"] ** 2)),
            "div_eps_velocity_mse": float(np.mean(residuals["div_eps_u"] ** 2)),
            "pressure_inlet_mse": pressure_inlet_mse,
            "pressure_outlet_mean_square": pressure_outlet_mean_square,
            "pressure_boundary_mse": pressure_inlet_mse + pressure_outlet_mean_square,
        }
        physical_mse: dict[str, float] = {}
        for field_index, field in enumerate(output_fields):
            squared_error_sum = float(np.sum(error[field_index] ** 2))
            count = int(error[field_index].size)
            physical_mse[field] = squared_error_sum / count
            physical_sse, physical_count, physical_rmse = analysis.artifacts.contracts.physical_statistic_columns(field)
            normalized_sse, normalized_count, normalized_rmse = analysis.artifacts.contracts.normalized_statistic_columns(field)
            row[physical_sse] = squared_error_sum
            row[physical_count] = count
            row[physical_rmse] = float(np.sqrt(physical_mse[field]))
            normalized_squared_error_sum = squared_error_sum / _TRAIN_STANDARD_DEVIATIONS[field] ** 2
            row[normalized_sse] = normalized_squared_error_sum
            row[normalized_count] = count
            row[normalized_rmse] = float(np.sqrt(normalized_squared_error_sum / count))
        for group in _TASK_SPEC.output_groups:
            group_mse = sum(physical_mse[field] for field in group.fields)
            group_variance = sum(_TRAIN_STANDARD_DEVIATIONS[field] ** 2 for field in group.fields)
            for metric in _TASK_SPEC.default_metrics:
                if metric.fields == group.fields and metric.kind == "group_rmse":
                    row[metric.id] = float(np.sqrt(group_mse / group_variance))
                elif metric.fields == group.fields and metric.kind == "vector_rmse":
                    row[metric.id] = float(np.sqrt(group_mse))
        rows.append(row)
    raw = pd.DataFrame(rows)
    raw.attrs["artifact_root"] = str(artifact_root)
    raw.attrs["artifact_provenance"] = _provenance(raw, role=role, run_name=run_name)
    return analysis.evaluation.dataframe.build_eval_df(raw), np.quantile(np.abs(np.stack(errors)), 0.9, axis=0)


@pytest.fixture
def session_fixture(tmp_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Return separate ID and OOD frames with exact spatial q90 evidence."""
    id_frame, id_q90 = _frame(tmp_path / "artifacts", role="eval", bias=0.03)
    ood_frame, ood_q90 = _frame(tmp_path / "artifacts", role="ood", bias=0.06)
    return {"Fixture ID": id_frame, "Fixture OOD": ood_frame}, {"Fixture ID": id_q90, "Fixture OOD": ood_q90}


def test_composite_colorbars_group_with_their_own_maps(
    session_fixture: tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]],
) -> None:
    """Keep each exact-height colorbar closer to its map than the next group."""
    datasets, _expected = session_fixture

    figure = analysis.evaluation.plots.physical_consistency.plot_velocity_divergence(
        datasets=datasets,
        max_cases=2,
    )
    figure.canvas.draw()
    map_axes = sorted(
        (axis for axis in figure.axes if axis.images),
        key=lambda axis: axis.get_position().x0,
    )
    assert len(map_axes) == len(datasets)
    first_map, next_map = map_axes[:2]
    line_axis = next(axis for axis in figure.axes if axis.lines and not axis.images)
    colorbar = first_map.images[0].colorbar
    assert colorbar is not None
    colorbar_axis = colorbar.ax
    map_to_colorbar = colorbar_axis.get_position().x0 - first_map.get_position().x1
    colorbar_to_next_map = next_map.get_position().x0 - colorbar_axis.get_position().x1

    assert map_to_colorbar > 0.0
    assert colorbar_to_next_map > 0.0
    assert map_to_colorbar < colorbar_to_next_map
    assert colorbar_axis.get_position().height == pytest.approx(first_map.get_position().height)
    vertical_gap = line_axis.get_position().y0 - first_map.get_position().y1
    assert 0.0 < vertical_gap < line_axis.get_position().height
    assert line_axis.get_position().height > first_map.get_position().height / 2.0
    assert 1.0 - line_axis.get_position().y1 < line_axis.get_position().height
    assert all(any(label.get_visible() for label in axis.get_xticklabels()) for axis in map_axes)
    assert any(label.get_visible() for label in first_map.get_yticklabels())
    assert not any(label.get_visible() for label in next_map.get_yticklabels())
    plt.close(figure)


def test_session_caches_bounded_summaries_and_cases(
    session_fixture: tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]],
) -> None:
    """Reuse full, prefix, and selected-case evidence without mutating inputs."""
    datasets, expected_q90 = session_fixture
    session = analysis.evaluation.session.EvaluationSession(datasets)
    assert session.counters["npz_opens"] == 0
    try:
        for label, frame in datasets.items():
            summary = session.full_summary(frame)
            np.testing.assert_array_equal(summary.absolute_error_q90, expected_q90[label])
        full_opens = session.counters["npz_opens"]
        assert full_opens == sum(len(frame) for frame in datasets.values())
        for frame in datasets.values():
            session.full_summary(frame)
        assert session.counters["npz_opens"] == full_opens

        id_frame = next(iter(datasets.values()))
        prefix = session.prefix_summary(id_frame, 2)
        prefix_opens = session.counters["npz_opens"]
        session.prefix_summary(id_frame, 2)
        assert session.counters["npz_opens"] == prefix_opens
        local = prefix.local_relative_error[session.full_summary(id_frame).grid.fields[0]]
        assert local.source_point_count == len(id_frame) * int(
            np.prod(expected_q90["Fixture ID"].shape[1:]),
        )
        assert local.probabilities[[0, -1]].tolist() == [0.0, 1.0]
        assert np.all(np.diff(local.quantiles) >= 0.0)

        selected = session.load_case(id_frame, 0)
        selected_opens = session.counters["npz_opens"]
        assert session.load_case(id_frame, 0) is selected
        assert session.counters["npz_opens"] == selected_opens
        assert 0 < session.counters["aggregate_bytes_current"] <= session.max_aggregate_bytes
        with pytest.raises(ValueError, match="read-only"):
            selected.prediction[0, 0, 0] = 0.0
        with pytest.raises(TypeError):
            selected.metadata["conditions"]["roughness"] = 0.0
    finally:
        session.close()

    assert session.closed
    assert all(analysis.evaluation.session.bound_session(frame) is None for frame in datasets.values())


def test_session_isolates_run_roles_bounds_cases_and_invalidates_changes(
    session_fixture: tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]],
    tmp_path: Path,
) -> None:
    """Keep run-role identities separate while enforcing reuse safety and bounds."""
    datasets, _expected = session_fixture
    other_frame, _other_q90 = _frame(tmp_path / "other-artifacts", role="eval", bias=0.09, run_name="run-other")
    compared = {**datasets, "Other run ID": other_frame}
    with analysis.evaluation.session.EvaluationSession(
        compared,
        max_case_entries=1,
        max_case_bytes=1024**2,
    ) as session:
        id_frame, ood_frame = tuple(datasets.values())
        assert set(session.canonical_frames) == {
            ("run-shared", "eval"),
            ("run-shared", "ood"),
            ("run-other", "eval"),
        }
        id_case = session.load_case(id_frame, 0)
        ood_case = session.load_case(ood_frame, 0)
        other_case = session.load_case(other_frame, 0)
        assert id_case is not ood_case
        assert id_case is not other_case
        assert not np.array_equal(id_case.error, ood_case.error)
        assert not np.array_equal(id_case.error, other_case.error)
        assert 0 < session.counters["case_bytes_current"] <= session.max_case_bytes
        before_reopen = session.counters["npz_opens"]
        assert session.load_case(id_frame, 0) is not id_case
        assert session.counters["npz_opens"] == before_reopen + 1

        session.load_case(other_frame, 0)
        changed = copy.deepcopy(other_frame.attrs["artifact_provenance"])
        changed["outputs"] = {"fixture_manifest_digest": "changed"}
        other_frame.attrs["artifact_provenance"] = changed
        with pytest.raises(analysis.evaluation.session.EvaluationArtifactChangedError, match="identity changed"):
            session.load_case(other_frame, 0)
        with pytest.raises(analysis.evaluation.session.EvaluationArtifactChangedError, match="changed"):
            session.load_case(other_frame, 0)
        assert session.load_case(ood_frame, 0).fields == _TASK_SPEC.output_names
        assert session.counters["case_bytes_current"] <= session.max_case_bytes


def test_loaded_run_session_owns_canonical_roles_for_one_explicit_lifecycle(
    session_fixture: tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]],
) -> None:
    """Build and release canonical role bindings from strict loader results."""
    datasets, _expected = session_fixture
    id_frame, ood_frame = tuple(datasets.values())
    id_root = Path(id_frame.attrs["artifact_root"])
    ood_root = Path(ood_frame.attrs["artifact_root"])
    loader = analysis.evaluation.artifact_loader
    loaded = loader.LoadedRunArtifacts(
        task=_TASK_SPEC.id,
        run_name="run-shared",
        run_dir=id_root.parent / "run-shared",
        id_artifact=loader.LoadedEvaluationArtifact("eval", "fixture-eval", id_root, id_frame, "id-digest"),
        ood_artifact=loader.LoadedEvaluationArtifact("ood", "fixture-ood", ood_root, ood_frame, "ood-digest"),
    )
    with analysis.evaluation.session.EvaluationSession.from_loaded_runs([loaded]) as session:
        assert tuple(session.canonical_frames) == (("run-shared", "eval"), ("run-shared", "ood"))
        assert session.canonical_frames[("run-shared", "eval")] is id_frame
        assert session.canonical_frames[("run-shared", "ood")] is ood_frame
        assert session.load_case(session.canonical_frames[("run-shared", "eval")], 0).fields == _TASK_SPEC.output_names
    assert all(analysis.evaluation.session.bound_session(frame) is None for frame in datasets.values())

    with pytest.raises(ValueError, match="at least one"):
        analysis.evaluation.session.EvaluationSession.from_loaded_runs([])
    with pytest.raises(ValueError, match="Duplicate canonical"):
        analysis.evaluation.session.EvaluationSession.from_loaded_runs([loaded, loaded])
    assert all(analysis.evaluation.session.bound_session(frame) is None for frame in datasets.values())
