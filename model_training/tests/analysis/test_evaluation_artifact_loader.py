# ruff: noqa: S101, SLF001
"""Protect the strict load-only completed-run evaluation artifact boundary."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import torch
from src import common, domain
from src.analysis.artifacts import contracts
from src.analysis.evaluation import evaluation_artifact_loader as loader
from src.analysis.evaluation import evaluation_dataframe

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.domain.tasks.domain_task_spec import TaskSpec

_TASK = "steady_flow"
_RUN_NAME = "fixture_run"
_ID_DATASET = "fixture_id"
_OOD_DATASET = "fixture_ood"


@dataclass(frozen=True, slots=True)
class _Fixture:
    """Compact completed-run evidence and its two persisted artifact roots."""

    output_root: Path
    run_dir: Path
    completed: dict[str, Any]
    validate_calls: list[Path]
    id_root: Path
    ood_root: Path


def _resolved_metrics(task_spec: TaskSpec) -> list[dict[str, Any]]:
    """Return task metrics with explicit resolved output fields."""
    metrics: list[dict[str, Any]] = []
    for metric in task_spec.default_metrics:
        payload = metric.as_dict(all_fields=task_spec.output_names)
        if payload["fields"] == "all":
            payload["fields"] = list(task_spec.output_names)
        metrics.append(payload)
    return metrics


def _objective(task_spec: TaskSpec) -> dict[str, Any]:
    """Return the task-owned primary objective with explicit fields."""
    return next(copy.deepcopy(metric) for metric in _resolved_metrics(task_spec) if metric["id"] == evaluation_dataframe.PRIMARY_OBJECTIVE_ID)


def _config(task_spec: TaskSpec) -> dict[str, Any]:
    """Build the resolved configuration fields consumed by the load-only binder."""
    objective = _objective(task_spec)
    metrics = _resolved_metrics(task_spec)
    return {
        "task": task_spec.id,
        "task_contract": task_spec.resolved_contract(),
        "run": {"name": _RUN_NAME},
        "data": {
            "train_dataset": _ID_DATASET,
            "ood_datasets": [_OOD_DATASET],
            "batch_size": 2,
        },
        "model": {
            "kind": "fno",
            "params": {
                "in_channels": task_spec.in_channels,
                "out_channels": task_spec.out_channels,
                "hidden_channels": 8,
                "n_layers": 2,
            },
        },
        "loss": {
            "physics": {
                "enabled": True,
                "continuity": "div_eps_velocity",
            }
        },
        "evaluation": {
            "metrics": copy.deepcopy(metrics),
            "objective": copy.deepcopy(objective),
        },
    }


def _summary() -> dict[str, Any]:
    """Return immutable run identities consumed by artifact provenance."""
    return {
        "task": _TASK,
        "run_name": _RUN_NAME,
        "effective_config_digest": "c" * 64,
        "best_checkpoint_sha256": "b" * 64,
        "normalizer_sha256": "n" * 64,
        "model_parameter_counts": {"total": 1200, "trainable": 1100},
    }


def _dataset_identity(
    task_spec: TaskSpec,
    *,
    dataset_name: str,
    fingerprint: str,
) -> dict[str, Any]:
    """Return one compact saved dataset identity."""
    return {
        "dataset_id": dataset_name,
        "task": task_spec.id,
        "data_contract_digest": task_spec.data_contract_digest,
        "fingerprint": fingerprint,
        "sample_ids": [f"case_{index + 1:04d}" for index in range(4)],
        "sample_count": 4,
        "spatial_shape": [2, 2],
    }


def _split(task_spec: TaskSpec) -> dict[str, Any]:
    """Build distinct ordered ID and OOD memberships with saved digests."""
    return {
        "schema_version": 1,
        "task": task_spec.id,
        "task_contract_digest": task_spec.contract_digest,
        "train_indices": torch.tensor([1, 3], dtype=torch.long),
        "eval_indices": torch.tensor([2, 0], dtype=torch.long),
        "ood_indices": torch.tensor([3, 1], dtype=torch.long),
        "metadata": {
            "datasets": {
                "train": _dataset_identity(task_spec, dataset_name=_ID_DATASET, fingerprint="i" * 64),
                "ood": _dataset_identity(task_spec, dataset_name=_OOD_DATASET, fingerprint="o" * 64),
            },
            "n_train_full": 4,
            "n_train": 2,
            "n_eval": 2,
            "n_ood_full": 4,
            "n_ood": 2,
            "train_ratio": 0.5,
            "ood_fraction": 0.5,
            "split_seed": 7,
            "membership_digests": {
                "train": "t" * 64,
                "eval": "e" * 64,
                "ood": "d" * 64,
            },
        },
    }


def _normalizer_state(task_spec: TaskSpec) -> dict[str, torch.Tensor]:
    """Return compact saved state with distinct task-aligned output scales."""
    output_scales = torch.tensor(
        [float(2**index) for index in range(task_spec.out_channels)],
        dtype=torch.float64,
    ).reshape(1, task_spec.out_channels, 1, 1)
    return {
        "in_normalizer.mean": torch.zeros(1, task_spec.in_channels, 1, 1, dtype=torch.float64),
        "in_normalizer.std": torch.ones(1, task_spec.in_channels, 1, 1, dtype=torch.float64),
        "out_normalizer.mean": torch.zeros(1, task_spec.out_channels, 1, 1, dtype=torch.float64),
        "out_normalizer.std": output_scales,
    }


def _train_standard_deviations(task_spec: TaskSpec) -> dict[str, float]:
    """Project task-aligned output scales from the saved fixture state."""
    return contracts.output_standard_deviations_from_state(
        _normalizer_state(task_spec),
        output_fields=task_spec.output_names,
    )


def _artifact_row(
    *,
    task_spec: TaskSpec,
    artifact_root: Path,
    source_index: int,
    split_local_index: int,
) -> dict[str, Any]:
    """Build one exact current-schema steady-flow Parquet row."""
    case_index = source_index + 1
    npz_path = artifact_root / "npz" / f"case_{case_index:04d}.npz"
    row: dict[str, Any] = {
        "artifact_schema_version": contracts.ARTIFACT_SCHEMA_VERSION,
        "task_id": task_spec.id,
        "output_fields": list(task_spec.output_names),
        "output_units": [field.unit for field in task_spec.outputs],
        "case_index": case_index,
        "source_index": source_index,
        "split_local_index": split_local_index,
        "npz_path": str(npz_path.resolve()),
        "meta": json.dumps({"fixture_parameter": float(source_index + 1)}),
        "inference_time_ms": 1.0,
        "rel_l2": 0.1,
        "rel_h1": 0.2,
        "physical_rmse_speed_magnitude": 0.3,
        "kappa_names": ["kxx", "kxy", "kyy"],
        "momentum_residual_mse": 0.4,
        "div_velocity_mse": 0.5,
        "div_eps_velocity_mse": 0.6,
        "pressure_inlet_mse": 0.7,
        "pressure_outlet_mean_square": 0.8,
        "pressure_boundary_mse": 1.5,
    }
    scales = _train_standard_deviations(task_spec)
    physical_mse: dict[str, float] = {}
    for field_index, field in enumerate(task_spec.output_names, start=1):
        squared_error_sum = float(field_index * (source_index + 1))
        element_count = 4
        physical_mse[field] = squared_error_sum / element_count
        physical_sse, physical_count, physical_rmse = contracts.physical_statistic_columns(field)
        normalized_sse, normalized_count, normalized_rmse = contracts.normalized_statistic_columns(field)
        row[physical_sse] = squared_error_sum
        row[physical_count] = element_count
        row[physical_rmse] = float(np.sqrt(physical_mse[field]))
        normalized_squared_error_sum = squared_error_sum / (scales[field] + 1e-7) ** 2
        row[normalized_sse] = normalized_squared_error_sum
        row[normalized_count] = element_count
        row[normalized_rmse] = float(np.sqrt(normalized_squared_error_sum / element_count))
    for group in task_spec.output_groups:
        group_mse = sum(physical_mse[field] for field in group.fields)
        group_scale = sum(scales[field] ** 2 for field in group.fields)
        for metric in task_spec.default_metrics:
            if metric.fields != group.fields:
                continue
            if metric.kind == "group_rmse":
                row[metric.id] = float(np.sqrt(group_mse / group_scale))
            elif metric.kind == "vector_rmse":
                row[metric.id] = float(np.sqrt(group_mse))
    return row


def _write_artifact(
    *,
    artifact_root: Path,
    dataset_name: str,
    split_role: loader.ArtifactRole,
    source_indices: tuple[int, ...],
    generation_limit: int | None,
    dataset_identity: dict[str, Any],
    saved_membership_digest: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    task_spec: TaskSpec,
) -> None:
    """Persist one compact manifest-valid artifact without using generation code."""
    effective_count = len(source_indices) if generation_limit is None else min(len(source_indices), generation_limit)
    effective_indices = source_indices[:effective_count]
    npz_root = artifact_root / "npz"
    npz_root.mkdir(parents=True)
    rows = []
    for split_local_index, source_index in enumerate(effective_indices):
        case_index = source_index + 1
        np.savez_compressed(
            npz_root / f"case_{case_index:04d}.npz",
            marker=np.asarray([case_index], dtype=np.int64),
        )
        rows.append(
            _artifact_row(
                task_spec=task_spec,
                artifact_root=artifact_root,
                source_index=source_index,
                split_local_index=split_local_index,
            )
        )
    raw = pd.DataFrame(rows)
    parquet_path = artifact_root / f"{dataset_name}.parquet"
    raw.to_parquet(parquet_path, index=False)
    provenance: dict[str, Any] = {
        "provenance_schema_version": contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": contracts.ARTIFACT_SCHEMA_VERSION,
        "run": loader._expected_run_provenance(
            summary=summary,
            task_spec=task_spec,
            run_name=_RUN_NAME,
        ),
        "model": loader._expected_model_provenance(config=config, summary=summary),
        "split_role": split_role,
        "dataset": {
            "name": dataset_name,
            "full_case_count": dataset_identity["sample_count"],
            "fingerprint": dataset_identity["fingerprint"],
            "data_contract_digest": dataset_identity["data_contract_digest"],
            "saved_membership_digest": saved_membership_digest,
        },
        "selection": {
            "index_key": "eval_indices" if split_role == "eval" else "ood_indices",
            "full_selected_case_count": len(source_indices),
            "effective_case_count": effective_count,
            "generation_limit": generation_limit,
            "full_ordered_source_indices_sha256": contracts.ordered_indices_sha256(source_indices),
            "effective_ordered_source_indices_sha256": contracts.ordered_indices_sha256(effective_indices),
        },
        "normalizer": loader._expected_normalizer_provenance(
            task_spec=task_spec,
            normalizer_sha256=summary["normalizer_sha256"],
            normalizer_state=_normalizer_state(task_spec),
        ),
        "evaluator": loader._expected_evaluator_provenance(config=config, task_spec=task_spec),
        "generation": {
            "effective_case_limit": generation_limit,
            "inference_batch_size": config["data"]["batch_size"],
            "compression": "numpy savez_compressed",
        },
        "runtime": {
            "requested_policy": "cpu",
            "resolved_device": "cpu",
            "device_type": "cpu",
            "pytorch_version": torch.__version__,
            "batch_size": config["data"]["batch_size"],
        },
        "aggregate": contracts.aggregate_normalized_group_macro_rmse(
            raw,
            output_groups=task_spec.output_groups,
            train_standard_deviations=_train_standard_deviations(task_spec),
            normalization_denominator_floor=1e-7,
        ),
    }
    physics = loader._expected_physics_provenance(config=config, task_spec=task_spec)
    assert physics is not None
    provenance["physics"] = physics
    provenance["outputs"] = contracts.artifact_output_manifest(artifact_root)
    contracts.artifact_provenance_path(artifact_root).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    roles: Iterable[loader.ArtifactRole] = ("eval", "ood"),
    generation_limit: int | None = None,
) -> _Fixture:
    """Build one completed-run fixture and install its read-only admission seam."""
    task_spec = domain.tasks.registry.get_task(_TASK)
    config = _config(task_spec)
    summary = _summary()
    split = _split(task_spec)
    output_root = tmp_path / "processed"
    run_dir = common.paths.resolve_run_output_dir(_TASK, _RUN_NAME, output_root=output_root).resolve()
    run_dir.mkdir(parents=True)
    completed = {
        "run_dir": run_dir,
        "config": config,
        "summary": summary,
        "split_indices": split,
        "normalizer_state": _normalizer_state(task_spec),
    }
    id_root = common.paths.resolve_id_analysis_dir(run_dir).resolve()
    ood_root = common.paths.resolve_ood_analysis_dir(run_dir, _OOD_DATASET).resolve()
    selected_roles = set(roles)
    if "eval" in selected_roles:
        _write_artifact(
            artifact_root=id_root,
            dataset_name=_ID_DATASET,
            split_role="eval",
            source_indices=tuple(int(value) for value in split["eval_indices"].tolist()),
            generation_limit=generation_limit,
            dataset_identity=split["metadata"]["datasets"]["train"],
            saved_membership_digest=split["metadata"]["membership_digests"]["eval"],
            config=config,
            summary=summary,
            task_spec=task_spec,
        )
    if "ood" in selected_roles:
        _write_artifact(
            artifact_root=ood_root,
            dataset_name=_OOD_DATASET,
            split_role="ood",
            source_indices=tuple(int(value) for value in split["ood_indices"].tolist()),
            generation_limit=generation_limit,
            dataset_identity=split["metadata"]["datasets"]["ood"],
            saved_membership_digest=split["metadata"]["membership_digests"]["ood"],
            config=config,
            summary=summary,
            task_spec=task_spec,
        )
    validate_calls: list[Path] = []

    def validate_completed_run(candidate: Path) -> dict[str, Any]:
        validate_calls.append(Path(candidate).resolve())
        return completed

    monkeypatch.setattr(loader.experiments.run, "validate_completed_run", validate_completed_run)
    return _Fixture(
        output_root=output_root,
        run_dir=run_dir,
        completed=completed,
        validate_calls=validate_calls,
        id_root=id_root,
        ood_root=ood_root,
    )


def _inventory(root: Path) -> dict[Path, tuple[str, int]]:
    """Return byte digests and mtimes for every persisted fixture file."""
    return {
        path.relative_to(root): (common.serialization.file_sha256(path), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mutate_provenance(root: Path, path: tuple[str, ...], value: Any) -> None:
    """Change one exact nested provenance field in a temporary fixture."""
    provenance_path = contracts.artifact_provenance_path(root)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_completed_run_admission_is_read_only_and_generation_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit complete artifacts read-only and reject bounded saved-membership prefixes."""
    fixture = _build_fixture(tmp_path / "complete", monkeypatch)
    before = _inventory(fixture.run_dir)

    def unexpected_npz_open(*_args: Any, **_kwargs: Any) -> Any:
        message = "load-only artifact admission opened an NPZ payload"
        raise AssertionError(message)

    monkeypatch.setattr(np, "load", unexpected_npz_open)
    loaded = loader.load_completed_run_artifacts(
        _TASK,
        _RUN_NAME,
        output_root=fixture.output_root,
    )

    assert loaded.task == _TASK
    assert loaded.run_name == _RUN_NAME
    assert loaded.run_dir == fixture.run_dir
    assert loaded.id_artifact is not None
    assert loaded.ood_artifact is not None
    assert (
        loaded.id_artifact.root,
        loaded.ood_artifact.root,
        loaded.id_artifact.split_role,
        loaded.ood_artifact.split_role,
    ) == (fixture.id_root, fixture.ood_root, "eval", "ood")
    assert loaded.id_artifact.dataset_name == fixture.completed["config"]["data"]["train_dataset"]
    assert loaded.ood_artifact.dataset_name == fixture.completed["config"]["data"]["ood_datasets"][0]
    split = fixture.completed["split_indices"]
    assert loaded.id_artifact.frame["source_index"].tolist() == split["eval_indices"].tolist()
    assert loaded.ood_artifact.frame["source_index"].tolist() == split["ood_indices"].tolist()
    for artifact in (loaded.id_artifact, loaded.ood_artifact):
        provenance = artifact.frame.attrs["artifact_provenance"]
        assert artifact.identity_sha256 == common.serialization.canonical_json_sha256(provenance)
    assert fixture.validate_calls == [fixture.run_dir]
    assert _inventory(fixture.run_dir) == before
    prefix_fixture = _build_fixture(
        tmp_path / "prefix",
        monkeypatch,
        generation_limit=1,
    )
    with pytest.raises(loader.IncompatibleEvaluationArtifactsError, match="complete saved membership"):
        loader.load_completed_run_artifacts(
            _TASK,
            _RUN_NAME,
            output_root=prefix_fixture.output_root,
        )


def test_artifact_boundary_rejects_missing_changed_and_aliased_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject absent or contradictory evidence with exact host-side recovery actions."""
    missing = _build_fixture(tmp_path / "missing", monkeypatch, roles=("eval",))
    build_command = f"./scripts/docker_job.sh --queue-gpu auto artifacts --run-dir {missing.run_dir}"
    with pytest.raises(loader.MissingEvaluationArtifactsError) as captured:
        loader.load_completed_run_artifacts(_TASK, _RUN_NAME, output_root=missing.output_root)
    assert "missing" in str(captured.value)
    assert build_command in str(captured.value)
    assert "--rebuild" not in str(captured.value)
    assert not missing.ood_root.exists()

    contradictions = (
        (("run", "best_checkpoint_sha256"), "x" * 64),
        (("dataset", "fingerprint"), "x" * 64),
        (("dataset", "data_contract_digest"), "x" * 64),
        (("dataset", "saved_membership_digest"), "x" * 64),
        (("selection", "effective_ordered_source_indices_sha256"), "x" * 64),
        (("physics", "selected_training_continuity"), "div_velocity"),
        (("split_role",), "ood"),
    )
    for index, (field_path, replacement) in enumerate(contradictions):
        fixture = _build_fixture(tmp_path / f"contradiction-{index}", monkeypatch)
        _mutate_provenance(fixture.id_root, field_path, replacement)
        with pytest.raises(loader.IncompatibleEvaluationArtifactsError) as captured:
            loader.load_completed_run_artifacts(
                _TASK,
                _RUN_NAME,
                output_root=fixture.output_root,
            )
        message = str(captured.value)
        assert "incompatible" in message
        rebuild_command = f"./scripts/docker_job.sh --queue-gpu auto artifacts --run-dir {fixture.run_dir} --rebuild"
        assert rebuild_command in message

    changed = _build_fixture(tmp_path / "changed-payload", monkeypatch)
    npz_path = next((changed.id_root / "npz").glob("*.npz"))
    npz_path.write_bytes(npz_path.read_bytes() + b"changed")

    def unexpected_npz_open(*_args: Any, **_kwargs: Any) -> Any:
        message = "digest admission decompressed an NPZ payload"
        raise AssertionError(message)

    monkeypatch.setattr(np, "load", unexpected_npz_open)
    with pytest.raises(loader.IncompatibleEvaluationArtifactsError, match="manifest"):
        loader.load_completed_run_artifacts(_TASK, _RUN_NAME, output_root=changed.output_root)

    aliased = _build_fixture(tmp_path / "aliased-root", monkeypatch)
    outside = tmp_path / "outside-id-artifact"
    aliased.id_root.rename(outside)
    aliased.id_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(loader.IncompatibleEvaluationArtifactsError, match="symbolic-link alias"):
        loader.load_completed_run_artifacts(_TASK, _RUN_NAME, output_root=aliased.output_root)
