# ruff: noqa: S101
"""
Exercise the complete steady-flow lifecycle on compact synthetic CPU data.

The integration path trains a tiny model, validates completed-run identity, reloads
the best checkpoint, and generates/reuses/rebuilds ID/OOD artifacts. Systematic
identity corruptions must fail before forward. Unit modules own exhaustive formula
and race coverage, and this fixture is not a performance benchmark.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import torch
from neuralop.models import FNO
from src import analysis, common, datasets, domain, experiments, learning
from support import configs
from support.synthetic_task import build_synthetic_generated_batch_identity

if TYPE_CHECKING:
    from collections.abc import Callable

_ID_DATASET = "tiny_steady_id"
_OOD_DATASET = "tiny_steady_ood_named"
_RUN_SUFFIX = "integration_smoke"
_SHAPE = (8, 8)
_ARTIFACT_CROP = 2


@dataclass(frozen=True)
class CompletedSmoke:
    """
    Retain the immutable outputs of the module-scoped one-epoch smoke lifecycle.

    Attributes
    ----------
    config : dict[str, Any]
        Resolved CPU experiment contract used for the run.
    dataset_root : pathlib.Path
        Temporary raw root owning the ID and named OOD final datasets.
    run_dir : pathlib.Path
        Completed saved-run leaf whose artifacts may be mutated only in copied tests.
    id_payload, ood_payload : dict[str, Any]
        Original strict final payloads used for split and normalizer assertions.
    completed : dict[str, Any]
        Result of strict completed-run validation after best/last roles diverge.

    Notes
    -----
    The dataclass is frozen, but contained dictionaries are test-owned mutable objects.

    """

    config: dict[str, Any]
    dataset_root: Path
    metadata_root: Path
    run_dir: Path
    id_payload: dict[str, Any]
    ood_payload: dict[str, Any]
    completed: dict[str, Any]


def _case_components(
    task: domain.tasks.spec.TaskSpec,
    *,
    case_id: str,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Build deterministic in-memory case tensors and their scientific identity."""
    y_axis = torch.linspace(0.0, 1.0, _SHAPE[0], dtype=torch.float32)
    x_axis = torch.linspace(0.0, 1.0, _SHAPE[1], dtype=torch.float32)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    input_fields = {
        "x": x,
        "y": y,
        "kxx": -12.0 + 0.02 * offset + 0.01 * x,
        "kxy": 0.01 * offset + 0.02 * x * y,
        "kyy": -11.8 + 0.015 * offset + 0.01 * y,
        "eps": 0.25 + 0.005 * offset + 0.02 * x + 0.01 * y,
        "p_bc": (1.0 - x) * (1.0 + 0.02 * offset) + 0.01 * y,
    }
    output_fields = {
        "p": (1.0 - x) * (1.0 + 0.01 * offset) + 0.02 * y,
        "u": 1.0e-4 * (1.0 + 0.03 * offset + x + 0.2 * y),
        "v": 1.0e-4 * (-0.5 + 0.02 * offset + 0.1 * x - y),
    }
    inputs = torch.stack([input_fields[name] for name in task.input_names])
    outputs = torch.stack([output_fields[name] for name in task.output_names])
    source_identity = {"generator": "synthetic-smoke", "case": case_id}
    source_metadata = {"offset": offset, "case_id": case_id, "parameters": {"synthetic_parameter": 1.0}}
    fingerprint = datasets.identity.compute_case_fingerprint(
        task=task,
        case_id=case_id,
        source_identity=source_identity,
        source_metadata=source_metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return inputs, outputs, source_identity, source_metadata, fingerprint


def _training_dataset_payload(
    task: domain.tasks.spec.TaskSpec,
    *,
    dataset_id: str,
    offsets: tuple[float, ...],
) -> dict[str, Any]:
    """Build one final version-1 synthetic dataset entirely in memory."""
    sample_ids = [f"case_{index + 1:04d}" for index in range(len(offsets))]
    cases = [_case_components(task, case_id=case_id, offset=offset) for case_id, offset in zip(sample_ids, offsets, strict=True)]
    return datasets.identity.build_training_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=sample_ids,
        generated_batch_identity=build_synthetic_generated_batch_identity(
            batch_name=dataset_id,
            sample_ids=sample_ids,
        ),
        source_identities=[case[2] for case in cases],
        source_metadata=[case[3] for case in cases],
        source_provenance={"batch_manifest_sha256": "2" * 64},
        case_fingerprints=[case[4] for case in cases],
        inputs=torch.stack([case[0] for case in cases]),
        outputs=torch.stack([case[1] for case in cases]),
    )


def _save_dataset(root: Path, metadata_root: Path, payload: dict[str, Any]) -> Path:
    """Publish one final dataset and its self-contained metadata test package."""
    dataset_id = str(payload["dataset_id"])
    task = domain.tasks.registry.get_task(str(payload["task"]))
    metadata_dir = metadata_root / dataset_id
    metadata_dir.mkdir(parents=True)
    sample_csv_path = metadata_dir / datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME
    sample_csv_path.write_text(
        "case_id;synthetic_parameter\n" + "\n".join(f"{int(str(sample_id).removeprefix('case_'))};1.0" for sample_id in payload["sample_ids"]) + "\n",
        encoding="utf-8",
    )
    generated_identity = payload["generated_batch_identity"]
    configuration = {
        **generated_identity["configuration"],
        "sample_sha256": common.serialization.file_sha256(sample_csv_path),
    }
    manifest_cases = []
    for source in generated_identity["scientific_case_sources"]:
        case_id = str(source["case_id"])
        manifest_cases.append(
            {
                "case_id": case_id,
                "status": "complete",
                "stage": "simulation",
                "message": "",
                "files": {
                    "raw_csv_sha256": source["raw_csv_sha256"],
                    "raw_json_sha256": common.serialization.canonical_json_sha256({"batch_name": dataset_id, "case_id": case_id}),
                    "solution_csv_sha256": source["solution_csv_sha256"],
                    "solution_model_sha256": source["solution_model_sha256"],
                },
            }
        )
    manifest = {
        "schema_kind": datasets.metadata.SOURCE_MANIFEST_SCHEMA_KIND,
        "schema_version": datasets.metadata.SOURCE_MANIFEST_SCHEMA_VERSION,
        "batch_name": dataset_id,
        "status": "complete",
        "configuration": configuration,
        "field_schema": generated_identity["field_schema"],
        "intended_case_ids": list(payload["sample_ids"]),
        "cases": manifest_cases,
    }
    common.serialization.atomic_write_json(metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME, manifest)
    manifest_path = metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME
    manifest_sha256 = common.serialization.file_sha256(manifest_path)
    payload["source_provenance"]["batch_manifest_sha256"] = manifest_sha256
    sample_json_path = metadata_dir / datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME
    common.serialization.atomic_write_json(
        sample_json_path,
        {
            "meta": {
                **generated_identity["sampling"],
                "timestamp": "synthetic-test-snapshot",
            },
            "n_cases": int(payload["sample_count"]),
        },
    )
    payload["source_provenance"]["source_sample_csv_sha256"] = common.serialization.file_sha256(sample_csv_path)
    payload["source_provenance"]["source_sample_json_sha256"] = common.serialization.file_sha256(sample_json_path)
    destination = root / dataset_id / f"{dataset_id}.pt"
    common.serialization.atomic_torch_save(payload, destination)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    timing_summary = {"status": "missing", "measured_case_count": 0, "intended_case_count": identity.sample_count}
    snapshot_roles = {
        datasets.metadata.SOURCE_MANIFEST_FILENAME: "validated_generation_manifest",
        datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME: "validated_parameter_sample_csv",
        datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME: "validated_parameter_sample_json",
    }
    snapshots = {
        filename: {
            "sha256": common.serialization.file_sha256(metadata_dir / filename),
            "size_bytes": (metadata_dir / filename).stat().st_size,
            "required": True,
            "role": role,
        }
        for filename, role in snapshot_roles.items()
    }
    common.serialization.atomic_write_json(
        metadata_dir / datasets.metadata.METADATA_FILENAME,
        {
            "schema_kind": datasets.metadata.METADATA_SCHEMA_KIND,
            "schema_version": datasets.metadata.METADATA_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "scientific_identity": {
                "dataset_schema_version": datasets.identity.TRAINING_DATASET_SCHEMA_VERSION,
                "dataset_fingerprint": identity.fingerprint,
                "task_id": task.id,
                "data_contract_digest": identity.data_contract_digest,
                "source_batch_id": dataset_id,
                "generated_batch_identity_sha256": str(payload["generated_batch_identity"]["batch_manifest_identity_sha256"]),
                "sample_count": identity.sample_count,
                "spatial_shape": list(identity.spatial_shape),
                "tensors": payload["tensor_metadata"],
            },
            "artifacts": {
                "dataset": {
                    "filename": destination.name,
                    "sha256": common.serialization.file_sha256(destination),
                    "size_bytes": destination.stat().st_size,
                },
                "snapshots": snapshots,
            },
            "operational_provenance": {
                "builder_module": datasets.metadata.BUILDER_MODULE,
                "publication_method": datasets.metadata.PUBLICATION_METHOD,
                "source_manifest_sha256": manifest_sha256,
                "timing": timing_summary,
            },
        },
    )
    datasets.metadata.validate_dataset_metadata_directory(
        metadata_dir,
        dataset_identity=identity,
        dataset_path=destination,
    )
    return destination


def _tiny_config(*, dataset_root: Path, output_root: Path) -> dict[str, Any]:
    """
    Resolve the smallest public one-epoch CPU FNO experiment used end to end.

    The recipe retains production semantic validation, splitting, normalization,
    metrics, checkpoints, and paths while reducing only model/data size and duration.
    """
    raw = configs.direct_config(model_kind="fno", physics_enabled=False)
    raw["run"].update(
        {
            "seed": 23,
            "device": "cpu",
            "suffix": _RUN_SUFFIX,
        }
    )
    raw["data"].update(
        {
            "train_dataset": _ID_DATASET,
            "ood_datasets": [_OOD_DATASET],
            "train_ratio": 0.5,
            "ood_fraction": 0.5,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )
    raw["model"] = {
        "kind": "fno",
        "params": {
            "n_modes": [2, 2],
            "hidden_channels": 4,
            "n_layers": 1,
        },
    }
    raw["loss"]["data"].update(
        {
            "kind": "relative_l2",
            "space": "normalized",
            "weight": 1.0,
        }
    )
    raw["loss"]["physics"] = {"enabled": False}
    raw["optimizer"].update(
        {
            "kind": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
        }
    )
    raw["scheduler"] = None
    raw["training"].update(
        {
            "epochs": 1,
            "evaluation_interval": 1,
            "ood_evaluation_interval": 1,
            "mixed_precision": False,
        }
    )
    raw["tracking"]["wandb"]["mode"] = "disabled"
    config = experiments.config.loader.resolve_config(raw)
    config["paths"]["dataset_root"] = str(dataset_root)
    config["paths"]["output_root"] = str(output_root)
    return config


def _refresh_summary_digest(
    run_dir: Path,
    *,
    summary_key: str,
    artifact_path: Path,
) -> None:
    """
    Republish one run-summary file digest after an intentional payload mutation.

    This keeps the outer file-integrity layer valid so a test can isolate the deeper
    task, config, split, or checkpoint identity boundary it intends to corrupt.
    """
    summary = experiments.run.read_run_summary(run_dir)
    summary[summary_key] = common.serialization.file_sha256(artifact_path)
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        summary,
    )


def _make_last_checkpoint_distinct(run_dir: Path) -> None:
    """
    Mutate one numeric ``last`` weight and republish its authoritative digest.

    The checkpoint schema and run identity remain valid. Only model state changes so
    inference can prove it loads selection-only ``best`` rather than continuation ``last``.
    """
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    payload = copy.deepcopy(torch.load(last_path, map_location="cpu", weights_only=False))
    state = payload["model_state_dict"]
    changed = False
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            replacement = value.detach().clone()
            replacement.reshape(-1)[0] += 1.0
            state[name] = replacement
            changed = True
            break
    if not changed:
        message = "The FNO checkpoint contained no mutable numeric parameter."
        raise AssertionError(message)
    common.serialization.atomic_torch_save(payload, last_path)
    _refresh_summary_digest(
        run_dir,
        summary_key="last_checkpoint_sha256",
        artifact_path=last_path,
    )


def _nested_state_equal(left: Any, right: Any) -> bool:
    """
    Return exact equality for nested tensor, mapping, and sequence checkpoint state.

    Tensors compare on CPU without tolerance. The helper intentionally supports only
    structures used by model state dictionaries in this smoke fixture.
    """
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_nested_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_state_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True))
    return bool(left == right)


def _state_dict_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Return exact equality for two model state mappings."""
    return _nested_state_equal(left, right)


def test_generic_smoke_config_explicitly_disables_wandb(tmp_path: Path) -> None:
    """Keep the bounded lifecycle fixture independent of keys, accounts, and network."""
    config = _tiny_config(
        dataset_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    assert config["tracking"]["wandb"]["mode"] == "disabled"
    assert experiments.config.defaults.TRACKING_DEFAULTS["wandb"]["mode"] == "online"


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory: pytest.TempPathFactory) -> CompletedSmoke:
    """
    Build ID/OOD datasets, train one tiny CPU epoch, and validate one completed run.

    The module-scoped fixture pays the bounded training cost once, verifies runtime
    session facts, then makes ``last`` observably distinct while preserving validity.
    It must not be treated as a performance or scientific-accuracy benchmark.
    """
    root = tmp_path_factory.mktemp("steady_flow_lifecycle")
    dataset_root = root / "raw"
    metadata_root = root / "meta"
    output_root = root / "processed"
    task = domain.tasks.registry.get_task("steady_flow")
    id_payload = _training_dataset_payload(
        task,
        dataset_id=_ID_DATASET,
        offsets=(0.0, 1.0, 4.0, 10.0),
    )
    ood_payload = _training_dataset_payload(
        task,
        dataset_id=_OOD_DATASET,
        offsets=(20.0, 21.0, 24.0, 30.0),
    )
    _save_dataset(dataset_root, metadata_root, id_payload)
    _save_dataset(dataset_root, metadata_root, ood_payload)

    config = _tiny_config(dataset_root=dataset_root, output_root=output_root)
    assert config["tracking"]["wandb"]["mode"] == "disabled"
    run_dir = experiments.run.prepare_fresh_run(
        config,
        run_dir=common.paths.resolve_run_output_dir(
            str(config["task"]),
            str(config["run"]["name"]),
            output_root=output_root,
        ),
    )
    environment = pytest.MonkeyPatch()
    environment.delenv("WANDB_API_KEY", raising=False)
    try:
        experiments.run.execute_prepared_run(
            config,
            run_dir=run_dir,
            persisted_config=config,
            device_resolution=learning.device.resolve_device("cpu"),
        )
    finally:
        environment.undo()
    assert not any(candidate.is_dir() for candidate in root.rglob("wandb"))
    experiments.run.validate_completed_run(run_dir)

    _make_last_checkpoint_distinct(run_dir)
    completed = experiments.run.validate_completed_run(run_dir)
    summary = completed["summary"]
    assert summary["runtime_device"]["requested_policy"] == "cpu"
    assert summary["runtime_device"]["resolved_device"] == "cpu"
    assert len(summary["runtime_sessions"]) == 1
    assert summary["runtime_sessions"][0]["requested_policy"] == "cpu"
    assert summary["runtime_sessions"][0]["resolved_device"] == "cpu"
    assert not _state_dict_equal(
        completed["best_checkpoint"]["model_state_dict"],
        completed["last_checkpoint"]["model_state_dict"],
    )
    return CompletedSmoke(
        config=config,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        run_dir=run_dir,
        id_payload=id_payload,
        ood_payload=ood_payload,
        completed=completed,
    )


def _artifact_inventory(targets: tuple[Path, ...]) -> dict[Path, tuple[str, int]]:
    """
    Return SHA-256 and nanosecond write-time identity for every file below targets.

    Tests use both values to prove cache reuse is mutation-free and rebuild changes
    only selected artifact roots.
    """
    return {
        path: (common.serialization.file_sha256(path), path.stat().st_mtime_ns)
        for target in targets
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def test_real_steady_flow_lifecycle_and_artifacts(  # noqa: PLR0915
    completed_smoke: CompletedSmoke,
) -> None:
    """
    Execute the bounded synthetic lifecycle from training through artifact rebuild.

    Saved splits and train-only normalizers must reconstruct exactly, inference must
    load ``best`` rather than the deliberately distinct ``last``, and ID/OOD artifacts
    must expose current metrics, physics arrays, provenance, and normalized evidence.
    Valid caches remain byte/time-identical. Corrupt provenance, Parquet, and NPZ
    content fail read-only. Explicit rebuild replaces only selected targets. This
    protects the integration seams without substituting for unit formula/race coverage.
    """
    smoke = completed_smoke
    split = smoke.completed["split_indices"]
    seed_plan = experiments.run.build_seed_plan(smoke.config["run"]["seed"])
    rebuilt = experiments.config.loader.create_dataloaders_from_config(
        smoke.config,
        seed_plan=seed_plan,
    )
    for key in ("train_indices", "eval_indices", "ood_indices"):
        assert torch.equal(rebuilt["split_indices"][key], split[key])
    assert split["metadata"]["split_seed"] == seed_plan["split"]

    train_indices = split["train_indices"]
    normalizer = torch.load(
        common.paths.resolve_normalizer_path(smoke.run_dir),
        map_location="cpu",
        weights_only=False,
    )
    expected_input_mean = smoke.id_payload["inputs"][train_indices].mean(
        dim=(0, 2, 3),
        keepdim=True,
    )
    expected_output_mean = smoke.id_payload["outputs"][train_indices].mean(
        dim=(0, 2, 3),
        keepdim=True,
    )
    task = domain.tasks.registry.get_task(str(smoke.config["task"]))
    output_units = tuple(field.unit for field in task.outputs)
    velocity_group = next(group for group in task.output_groups if group.id == "velocity")
    velocity_units = {task.field(field).unit for field in velocity_group.fields}
    assert len(velocity_units) == 1
    speed_unit = next(iter(velocity_units))
    objective_id = str(smoke.config["evaluation"]["objective"]["id"])
    train_standard_deviations = {field: float(normalizer["out_normalizer.std"][0, index, 0, 0]) for index, field in enumerate(task.output_names)}
    assert torch.allclose(normalizer["in_normalizer.mean"], expected_input_mean)
    assert torch.allclose(normalizer["out_normalizer.mean"], expected_output_mean)
    full_input_mean = smoke.id_payload["inputs"].mean(
        dim=(0, 2, 3),
        keepdim=True,
    )
    eps_index = domain.tasks.registry.get_task("steady_flow").input_names.index("eps")
    assert not torch.isclose(
        normalizer["in_normalizer.mean"][0, eps_index, 0, 0],
        full_input_mean[0, eps_index, 0, 0],
    )

    model, loader, processor, device = learning.inference.context.load_inference_context(
        run_dir=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        split="eval",
        batch_size=1,
        device_policy="cpu",
    )
    assert device.type == "cpu"
    selected_dataset = loader.dataset
    assert isinstance(selected_dataset, learning.inference.context.IndexedSubset)
    assert torch.equal(selected_dataset.source_indices, split["eval_indices"])
    in_normalizer = processor.in_normalizer
    assert in_normalizer is not None
    assert torch.equal(
        in_normalizer.mean.cpu(),
        normalizer["in_normalizer.mean"],
    )
    loaded_state = model.state_dict()
    best_state = smoke.completed["best_checkpoint"]["model_state_dict"]
    last_state = smoke.completed["last_checkpoint"]["model_state_dict"]
    assert _state_dict_equal(loaded_state, best_state)
    assert not _state_dict_equal(loaded_state, last_state)

    generated = analysis.artifacts.service.build_artifacts(
        runs_root=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        metadata_root=smoke.metadata_root,
        device_policy="cpu",
    )
    frames = generated[smoke.run_dir.name]
    assert set(frames) == {"eval", "ood"}
    for role, index_key in (("eval", "eval_indices"), ("ood", "ood_indices")):
        frame = frames[role]
        assert not frame.empty
        assert frame.columns.is_unique
        assert frame["source_index"].tolist() == split[index_key].tolist()
        assert all(Path(path).is_file() for path in frame["npz_path"])
        sufficient_statistics = {
            f"{space}_{statistic}_{field}"
            for field in task.output_names
            for space in ("normalized", "physical")
            for statistic in ("sse", "count", "rmse")
        }
        assert {
            "rel_l2",
            "rel_h1",
            "physical_rmse_speed_magnitude",
            "momentum_residual_mse",
            "div_velocity_mse",
            "div_eps_velocity_mse",
            "pressure_boundary_mse",
            "pressure_inlet_mse",
            "pressure_outlet_mean_square",
            *sufficient_statistics,
        }.issubset(frame.columns)
        with np.load(Path(frame.iloc[0]["npz_path"]), allow_pickle=False) as payload:
            assert payload["output_fields"].tolist() == list(task.output_names)
            assert payload["artifact_fields"].tolist() == [*task.output_names, "U"]
            assert payload["artifact_units"].tolist() == [*output_units, speed_unit]
            assert {"Rx", "Ry", "div_u", "div_eps_u", "coordinates"}.issubset(payload.files)
            div_u_interior = payload["div_u"][_ARTIFACT_CROP:-_ARTIFACT_CROP, _ARTIFACT_CROP:-_ARTIFACT_CROP]
            div_eps_u_interior = payload["div_eps_u"][_ARTIFACT_CROP:-_ARTIFACT_CROP, _ARTIFACT_CROP:-_ARTIFACT_CROP]
            assert frame.iloc[0]["div_velocity_mse"] == pytest.approx(float(np.mean(div_u_interior**2)))
            assert frame.iloc[0]["div_eps_velocity_mse"] == pytest.approx(float(np.mean(div_eps_u_interior**2)))
        enriched = analysis.evaluation.dataframe.build_eval_df(frame)
        assert enriched.attrs["output_units"] == output_units
        artifact_provenance = frame.attrs["artifact_provenance"]
        artifact_summary = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
            frame,
            output_groups=task.output_groups,
            train_standard_deviations=train_standard_deviations,
            normalization_denominator_floor=float(artifact_provenance["normalizer"]["denominator_floor"]),
        )
        assert enriched.attrs[objective_id] == artifact_summary
        online_role = "id" if role == "eval" else "ood"
        assert float(artifact_summary["value"]) == pytest.approx(
            smoke.completed["summary"]["selected_metrics"][f"selected/{online_role}/{objective_id}"],
            rel=analysis.artifacts.contracts.NORMALIZED_OBJECTIVE_TOLERANCE["rtol"],
            abs=analysis.artifacts.contracts.NORMALIZED_OBJECTIVE_TOLERANCE["atol"],
        )

    id_target = common.paths.resolve_id_analysis_dir(smoke.run_dir)
    ood_target = common.paths.resolve_ood_analysis_dir(
        smoke.run_dir,
        _OOD_DATASET,
    )
    targets = (id_target, ood_target)
    assert (id_target / f"{_ID_DATASET}.parquet").is_file()
    assert (ood_target / f"{_OOD_DATASET}.parquet").is_file()
    loaded_id = analysis.evaluation.dataframe.load_evaluation_artifact(id_target)
    assert loaded_id.attrs["provenance_complete"] is True
    assert loaded_id.attrs["artifact_root"] == str(id_target.resolve())
    assert loaded_id["source_index"].tolist() == split["eval_indices"].tolist()
    for role, target in zip(("eval", "ood"), targets, strict=True):
        stored_provenance = json.loads((target / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        assert stored_provenance["outputs"] == analysis.artifacts.contracts.artifact_output_manifest(target)
        assert stored_provenance["run"]["normalizer_sha256"] == smoke.completed["summary"]["normalizer_sha256"]
        assert stored_provenance["evaluator"]["objective"] == smoke.completed["config"]["evaluation"]["objective"]
        assert stored_provenance["physics"]["selected_training_continuity"] == "div_eps_velocity"
        assert stored_provenance["physics"]["evaluated_continuity_formulations"] == [
            "div_velocity",
            "div_eps_velocity",
        ]
        assert stored_provenance["runtime"]["requested_policy"] == "cpu"
        assert stored_provenance["runtime"]["resolved_device"] == "cpu"
        assert stored_provenance["run"]["best_checkpoint_sha256"] == smoke.completed["summary"]["best_checkpoint_sha256"]
        assert stored_provenance["dataset"]["saved_membership_digest"] == split["metadata"]["membership_digests"][role]
        assert stored_provenance["normalizer"]["sha256"] == smoke.completed["summary"]["normalizer_sha256"]
        assert stored_provenance["evaluator"]["group_objective_evidence"]["squared_error_accumulation_dtype"] == "float64"
        physics = stored_provenance["physics"]
        assert physics["residual_schema_version"] == analysis.artifacts.contracts.RESIDUAL_SCHEMA_VERSION
        assert physics["task_contract_digest"] == domain.tasks.registry.get_task("steady_flow").contract_digest
        assert physics["derivatives"] == {
            "kind": "spectral",
            "extension": "reflect",
            "operator_axes": [2, 3],
            "grid_axes": ["y", "x"],
        }
        assert physics["interior_crop"] == _ARTIFACT_CROP
        assert physics["constants"]["dynamic_viscosity_pa_s"] == domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
        assert physics["scalar_definitions"]["div_velocity_mse"] == {
            "formula": "mean(div(u)**2)",
            "unit": "1/s^2",
        }
        assert physics["scalar_definitions"]["pressure_outlet_mean_square"]["formula"] == "mean_outlet(p)**2"
        assert physics["residual_evaluation_region"]["residual_arrays"] == "full grid"
        runtime_comparison = analysis.artifacts.timing.load_runtime_comparison(target)
        assert runtime_comparison["split_role"] == role
        assert runtime_comparison["measurement"] == {
            "clock": "time.perf_counter_ns",
            "batch_size": smoke.config["data"]["batch_size"],
            "case_duration_attribution": analysis.artifacts.timing.CASE_DURATION_ATTRIBUTION,
            "warmup_passes": 1,
            "cuda_synchronized": False,
        }
        assert runtime_comparison["aggregates"]["neural_operator_forward_s"]["count"] == len(frames[role])

    id_provenance = json.loads((id_target / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert id_provenance["aggregate"]["value"] == pytest.approx(
        smoke.completed["summary"]["best_metric"],
        rel=analysis.artifacts.contracts.NORMALIZED_OBJECTIVE_TOLERANCE["rtol"],
        abs=analysis.artifacts.contracts.NORMALIZED_OBJECTIVE_TOLERANCE["atol"],
    )

    for target in targets:
        (target / "cache_marker.txt").write_text("preserve", encoding="utf-8")
    before_cache = _artifact_inventory(targets)
    cached = analysis.artifacts.service.build_artifacts(
        runs_root=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        metadata_root=smoke.metadata_root,
        device_policy="cpu",
    )
    pd.testing.assert_frame_equal(cached[smoke.run_dir.name]["eval"], frames["eval"])
    pd.testing.assert_frame_equal(cached[smoke.run_dir.name]["ood"], frames["ood"])
    assert _artifact_inventory(targets) == before_cache

    runtime_path = id_target / analysis.artifacts.timing.RUNTIME_COMPARISON_FILENAME
    valid_runtime = runtime_path.read_bytes()
    runtime_path.unlink()
    without_runtime = analysis.artifacts.service.run_or_load_artifacts(
        run_dir=smoke.run_dir,
        dataset_name=_ID_DATASET,
        split="eval",
        device_resolution=learning.device.resolve_device("cpu"),
        dataset_root=smoke.dataset_root,
        metadata_root=smoke.metadata_root,
    )
    pd.testing.assert_frame_equal(without_runtime, frames["eval"])
    assert not runtime_path.exists()
    runtime_path.write_bytes(valid_runtime)

    incompatible_runtime = json.loads(valid_runtime)
    incompatible_runtime["dataset_identity"]["fingerprint"] = "incompatible"
    common.serialization.atomic_write_json(runtime_path, incompatible_runtime)
    with_incompatible_runtime = analysis.artifacts.service.run_or_load_artifacts(
        run_dir=smoke.run_dir,
        dataset_name=_ID_DATASET,
        split="eval",
        device_resolution=learning.device.resolve_device("cpu"),
        dataset_root=smoke.dataset_root,
        metadata_root=smoke.metadata_root,
    )
    pd.testing.assert_frame_equal(with_incompatible_runtime, frames["eval"])
    assert runtime_path.read_bytes() != valid_runtime
    runtime_path.write_bytes(valid_runtime)

    provenance_path = id_target / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME
    valid_provenance = provenance_path.read_text(encoding="utf-8")
    provenance_path.write_text("{}\n", encoding="utf-8")
    incompatible_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifacts.service.ArtifactCacheError,
        match="provenance",
    ):
        analysis.artifacts.service.run_or_load_artifacts(
            run_dir=smoke.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=smoke.dataset_root,
            metadata_root=smoke.metadata_root,
        )
    assert _artifact_inventory((id_target,)) == incompatible_cache
    provenance_path.write_text(valid_provenance, encoding="utf-8")

    parquet_path = id_target / f"{_ID_DATASET}.parquet"
    valid_parquet = parquet_path.read_bytes()
    parquet_path.write_bytes(valid_parquet + b"corrupt")
    corrupted_parquet_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifacts.service.ArtifactCacheError,
        match="payload digest manifest mismatch",
    ):
        analysis.artifacts.service.run_or_load_artifacts(
            run_dir=smoke.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=smoke.dataset_root,
            metadata_root=smoke.metadata_root,
        )
    assert _artifact_inventory((id_target,)) == corrupted_parquet_cache
    parquet_path.write_bytes(valid_parquet)

    corrupted_npz = Path(frames["eval"].iloc[0]["npz_path"])
    corrupted_npz.write_bytes(corrupted_npz.read_bytes() + b"corrupt")
    corrupted_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifacts.service.ArtifactCacheError,
        match="payload digest manifest mismatch",
    ):
        analysis.artifacts.service.run_or_load_artifacts(
            run_dir=smoke.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=smoke.dataset_root,
            metadata_root=smoke.metadata_root,
        )
    assert _artifact_inventory((id_target,)) == corrupted_cache

    sibling_marker = common.paths.resolve_ood_analysis_dir(smoke.run_dir, "unselected_ood") / "keep.txt"
    sibling_marker.parent.mkdir(parents=True)
    sibling_marker.write_text("keep", encoding="utf-8")
    rebuilt_artifacts = analysis.artifacts.service.build_artifacts(
        runs_root=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        metadata_root=smoke.metadata_root,
        device_policy="cpu",
        rebuild=True,
    )
    for target in targets:
        assert not (target / "cache_marker.txt").exists()
    assert sibling_marker.read_text(encoding="utf-8") == "keep"
    pd.testing.assert_frame_equal(
        rebuilt_artifacts[smoke.run_dir.name]["eval"],
        frames["eval"],
    )
    pd.testing.assert_frame_equal(
        rebuilt_artifacts[smoke.run_dir.name]["ood"],
        frames["ood"],
    )


def _copy_without_analysis(source: Path, destination: Path) -> Path:
    """Copy one temporary run bundle without retaining prior artifact state."""
    shutil.copytree(source, destination)
    analysis_root = common.paths.resolve_analysis_root(destination)
    if analysis_root.exists():
        shutil.rmtree(analysis_root)
    return destination


def _rewrite_parquet_npz_paths(artifact_root: Path, values: list[str]) -> None:
    """Rewrite test-owned path rows and refresh their exact payload manifest."""
    provenance_path = analysis.artifacts.contracts.artifact_provenance_path(artifact_root)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    parquet_name = provenance["outputs"]["parquet"]["path"]
    parquet_path = artifact_root / parquet_name
    frame = pd.read_parquet(parquet_path)
    assert len(frame) == len(values)
    frame.loc[:, "npz_path"] = values
    frame.to_parquet(parquet_path, index=False)
    provenance["outputs"] = analysis.artifacts.contracts.artifact_output_manifest(artifact_root)
    common.serialization.atomic_write_json(provenance_path, provenance)


def test_completed_bundle_is_portable_and_legacy_paths_relocate_safely(
    completed_smoke: CompletedSmoke,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rename, generate, move, and reuse one completed bundle without identity drift."""
    original = _copy_without_analysis(completed_smoke.run_dir, tmp_path / "original")
    renamed = tmp_path / "paper_model_a"
    original.rename(renamed)
    evaluation = analysis.evaluation.workflow.prepare_evaluation_workflow(
        (analysis.evaluation.workflow.EvaluationRunSelection(run_dir=renamed, label="Paper model A"),),
        (
            analysis.evaluation.workflow.EvaluationContextSpec(key="id", label="ID", artifact_role="id"),
            analysis.evaluation.workflow.EvaluationContextSpec(key="ood", label="OOD", artifact_role="ood"),
        ),
        dataset_root=completed_smoke.dataset_root,
        metadata_root=completed_smoke.metadata_root,
        device_policy="cpu",
    )
    generated = evaluation.prepared_runs[0]
    first = generated.loaded_run
    assert generated.role_actions == {"id": "generated", "ood": "generated"}
    assert generated.artifact_device == "cpu"
    assert first.storage_alias == "paper_model_a"
    assert first.scientific_run_name == completed_smoke.config["run"]["name"]
    assert first.id_artifact is not None
    assert first.ood_artifact is not None
    assert first.is_completed
    assert not first.is_provisional
    assert evaluation.report[0]["artifact_device"] == "cpu"
    assert evaluation.report[0]["artifact_actions"] == {"id": "generated", "ood": "generated"}
    checkpoint_digest = first.selected_checkpoint_sha256
    evaluation.close()

    id_root = first.id_artifact.root
    raw_id = pd.read_parquet(id_root / f"{_ID_DATASET}.parquet")
    relative_paths = tuple(Path(value).parts for value in raw_id["npz_path"])
    expected_paths = tuple(("npz", f"case_{int(case_index):04d}.npz") for case_index in raw_id["case_index"])
    assert relative_paths == expected_paths
    legacy_values = [str((id_root / value).resolve()) for value in raw_id["npz_path"].tolist()]
    _rewrite_parquet_npz_paths(id_root, legacy_values)

    moved = tmp_path / "archive" / "promoted_model"
    moved.parent.mkdir()
    renamed.rename(moved)

    def unexpected_inference(**_kwargs: Any) -> Any:
        msg = "valid relocated artifacts must not reconstruct a model"
        raise AssertionError(msg)

    monkeypatch.setattr(learning.inference.context, "load_inference_context_with_resolution", unexpected_inference)
    reused = analysis.artifacts.service.load_or_build_run_artifacts(
        moved,
        dataset_root=completed_smoke.dataset_root,
        metadata_root=completed_smoke.metadata_root,
        device_policy="cpu",
    )
    loaded = reused.loaded_run
    assert reused.role_actions == {"id": "reused", "ood": "reused"}
    assert reused.artifact_device is None
    assert loaded.storage_alias == "promoted_model"
    assert loaded.scientific_run_name == first.scientific_run_name
    assert loaded.id_artifact is not None
    assert loaded.ood_artifact is not None
    assert loaded.selected_checkpoint_sha256 == checkpoint_digest
    assert all(Path(value).is_relative_to(loaded.id_artifact.root) for value in loaded.id_artifact.frame["npz_path"])

    raw_moved = pd.read_parquet(loaded.id_artifact.root / f"{_ID_DATASET}.parquet")
    escaping = raw_moved["npz_path"].tolist()
    escaping[0] = f"../npz/case_{int(raw_moved.iloc[0]['case_index']):04d}.npz"
    _rewrite_parquet_npz_paths(loaded.id_artifact.root, escaping)
    with pytest.raises(analysis.evaluation.artifact_loader.IncompatibleEvaluationArtifactsError, match=r"npz_path|relative NPZ"):
        analysis.artifacts.service.load_or_build_run_artifacts(
            moved,
            dataset_root=completed_smoke.dataset_root,
            metadata_root=completed_smoke.metadata_root,
            device_policy="cpu",
        )


def _rewrite_as_optuna_trial(run_dir: Path) -> dict[str, Any]:
    """Rewrite a test-owned completed copy with coherent Optuna-trial identity."""
    config_path = common.paths.resolve_run_config_path(run_dir)
    config = experiments.config.loader.load_yaml(config_path)
    config["tracking"]["wandb"]["workflow"] = "optuna_trial"
    config["tracking"]["wandb"]["study"] = "portable_fixture_study"
    config["tracking"]["wandb"].update(experiments.config.loader.derive_wandb_organization(config))
    config["run"]["name"] = experiments.config.loader.generate_run_name(config)
    config = experiments.config.loader.validate_resolved_config(config)
    experiments.config.loader.save_yaml(config, config_path)
    split = torch.load(common.paths.resolve_split_indices_path(run_dir), map_location="cpu", weights_only=False)
    identity = learning.training.checkpoint.build_checkpoint_identity(config, split, persisted_config=config)
    for checkpoint_path in (
        common.paths.resolve_best_checkpoint_file(run_dir),
        common.paths.resolve_last_checkpoint_file(run_dir),
    ):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["identity"] = copy.deepcopy(identity)
        common.serialization.atomic_torch_save(payload, checkpoint_path)
    summary = experiments.run.read_run_summary(run_dir)
    summary.update(
        {
            "run_name": config["run"]["name"],
            "effective_config_digest": identity["effective_config_digest"],
            "config_sha256": common.serialization.file_sha256(config_path),
            "best_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_best_checkpoint_file(run_dir)),
            "last_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_last_checkpoint_file(run_dir)),
            "study_name": "portable_fixture_study",
            "trial_number": 7,
            "sampled_parameters": {"model.params.hidden_channels": 4},
            "overrides": {"optimizer.lr": 1.0e-3},
            "study_role": "optimization",
        }
    )
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    return config


def test_completed_optuna_trial_moves_outside_study_without_origin_services(
    completed_smoke: CompletedSmoke,
    tmp_path: Path,
) -> None:
    """Move and rename a completed trial while preserving its origin metadata."""
    trial_dir = _copy_without_analysis(
        completed_smoke.run_dir,
        tmp_path / "study" / "trials" / "trial_000007",
    )
    config = _rewrite_as_optuna_trial(trial_dir)
    experiments.run.validate_completed_run(trial_dir)
    original_summary = experiments.run.read_run_summary(trial_dir)
    moved = tmp_path / "promoted" / "paper_trial_b"
    moved.parent.mkdir()
    trial_dir.rename(moved)

    prepared = analysis.artifacts.service.load_or_build_run_artifacts(
        moved,
        dataset_root=completed_smoke.dataset_root,
        metadata_root=completed_smoke.metadata_root,
        device_policy="cpu",
    )
    loaded = prepared.loaded_run
    assert loaded.scientific_run_name == config["run"]["name"]
    assert loaded.storage_alias == "paper_trial_b"
    assert loaded.run_dir == moved.resolve()
    assert loaded.is_completed
    assert prepared.role_actions == {"id": "generated", "ood": "generated"}
    assert not (tmp_path / "study" / "study.db").exists()
    assert loaded.summary is not None
    for key in ("study_name", "trial_number", "sampled_parameters", "overrides", "study_role"):
        assert loaded.summary[key] == original_summary[key]


def test_interrupted_bundle_generates_provisional_artifacts_and_reuses_after_move(
    completed_smoke: CompletedSmoke,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluate a renamed interrupted bundle without last state, then move and reuse it."""
    run_dir = _copy_without_analysis(completed_smoke.run_dir, tmp_path / "interrupted_source")
    summary = experiments.run.read_run_summary(run_dir)
    summary["status"] = "interrupted"
    summary["status_history"][-1]["status"] = "interrupted"
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    common.paths.resolve_last_checkpoint_file(run_dir).unlink()
    renamed = tmp_path / "diagnostic_interrupted"
    run_dir.rename(renamed)

    admitted = experiments.run.validate_evaluable_run(renamed)
    assert admitted["lifecycle_status"] == "interrupted"
    assert admitted["is_provisional"] is True
    assert admitted["selected_checkpoint_role"] == "best"
    prepared = analysis.artifacts.service.load_or_build_run_artifacts(
        renamed,
        dataset_root=completed_smoke.dataset_root,
        metadata_root=completed_smoke.metadata_root,
        device_policy="cpu",
    )
    loaded = prepared.loaded_run
    assert loaded.id_artifact is not None
    assert loaded.ood_artifact is not None
    assert loaded.is_provisional
    assert not loaded.is_completed
    assert prepared.role_actions == {"id": "generated", "ood": "generated"}
    for artifact in (loaded.id_artifact, loaded.ood_artifact):
        run_provenance = artifact.frame.attrs["artifact_provenance"]["run"]
        assert run_provenance["lifecycle_status"] == "interrupted"
        assert run_provenance["is_provisional"] is True
        assert run_provenance["selected_checkpoint_role"] == "best"
        assert run_provenance["best_checkpoint_sha256"] == loaded.selected_checkpoint_sha256

    moved = tmp_path / "diagnostics" / "moved_interrupted"
    moved.parent.mkdir()
    renamed.rename(moved)

    def unexpected_inference(**_kwargs: Any) -> Any:
        msg = "relocated provisional artifacts must be reused without inference"
        raise AssertionError(msg)

    monkeypatch.setattr(learning.inference.context, "load_inference_context_with_resolution", unexpected_inference)
    reused = analysis.artifacts.service.load_or_build_run_artifacts(
        moved,
        dataset_root=completed_smoke.dataset_root,
        metadata_root=completed_smoke.metadata_root,
        device_policy="cpu",
    )
    assert reused.role_actions == {"id": "reused", "ood": "reused"}
    assert reused.loaded_run.scientific_run_name == loaded.scientific_run_name
    assert reused.loaded_run.selected_checkpoint_sha256 == loaded.selected_checkpoint_sha256


def test_terminal_bundle_without_best_checkpoint_is_not_evaluable(
    completed_smoke: CompletedSmoke,
    tmp_path: Path,
) -> None:
    """Reject a terminal bundle whose selected best checkpoint is absent."""
    run_dir = _copy_without_analysis(completed_smoke.run_dir, tmp_path / "missing_best")
    summary = experiments.run.read_run_summary(run_dir)
    summary["status"] = "interrupted"
    summary["status_history"][-1]["status"] = "interrupted"
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    common.paths.resolve_best_checkpoint_file(run_dir).unlink()
    with pytest.raises(experiments.run.RunLifecycleError, match="no valid best checkpoint"):
        experiments.run.validate_evaluable_run(run_dir)


def test_interrupted_direct_run_resumes_from_renamed_directory(
    completed_smoke: CompletedSmoke,
    tmp_path: Path,
) -> None:
    """Resume one interrupted direct run by exact renamed path without redirection."""
    source = _copy_without_analysis(completed_smoke.run_dir, tmp_path / "resume_source")
    renamed = tmp_path / "renamed_resume_bundle"
    source.rename(renamed)
    summary = experiments.run.read_run_summary(renamed)
    summary["status"] = "interrupted"
    summary["status_history"][-1]["status"] = "interrupted"
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(renamed), summary)

    saved = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(renamed))
    raw = copy.deepcopy(saved)
    raw.pop("task_contract")
    raw.pop("paths")
    raw["run"].pop("name")
    for key in ("project", "entity", "tags"):
        raw["tracking"]["wandb"].pop(key)
    raw["model"]["params"].pop("in_channels")
    raw["model"]["params"].pop("out_channels")
    raw["evaluation"] = {"objective": {"id": saved["evaluation"]["objective"]["id"]}}
    resumed_epoch = 2
    raw["training"]["epochs"] = resumed_epoch
    request_path = configs.write_yaml(tmp_path / "resume_request.yaml", raw)

    resumed = experiments.run.run_experiment(
        request_path,
        resume=renamed,
        device="cpu",
    )
    assert resumed["run_dir"] == renamed.resolve()
    assert renamed.name == "renamed_resume_bundle"
    completed = experiments.run.validate_completed_run(renamed)
    assert completed["config"]["run"]["name"] == saved["run"]["name"]
    assert completed["last_checkpoint"]["completed_epoch"] == resumed_epoch


def _mutate_task_identity(run_dir: Path, dataset_root: Path) -> None:
    """Break only the saved split task digest."""
    del dataset_root
    path = common.paths.resolve_split_indices_path(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["task_contract_digest"] = "0" * 64
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="split_indices_sha256",
        artifact_path=path,
    )


def _mutate_config_identity(run_dir: Path, dataset_root: Path) -> None:
    """Break persisted config identity while retaining valid YAML."""
    del dataset_root
    path = common.paths.resolve_run_config_path(run_dir)
    config = experiments.config.loader.load_yaml(path)
    config["run"]["name"] = "different_run_identity"
    experiments.config.loader.save_yaml(config, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="config_sha256",
        artifact_path=path,
    )


def _mutate_dataset_identity(run_dir: Path, dataset_root: Path) -> None:
    """Change tensor content without changing its stored fingerprint."""
    del run_dir
    path = common.paths.resolve_dataset_path(
        _ID_DATASET,
        dataset_root=dataset_root,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["inputs"][0, 0, 0, 0] += 0.5
    common.serialization.atomic_torch_save(payload, path)


def _mutate_split_identity(run_dir: Path, dataset_root: Path) -> None:
    """Change ordered membership without changing its digest."""
    del dataset_root
    path = common.paths.resolve_split_indices_path(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["eval_indices"] = payload["eval_indices"].flip(0)
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="split_indices_sha256",
        artifact_path=path,
    )


def _mutate_checkpoint_identity(run_dir: Path, dataset_root: Path) -> None:
    """Break only the best checkpoint run identity."""
    del dataset_root
    path = common.paths.resolve_best_checkpoint_file(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["identity"]["task"] = "different_task"
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="best_checkpoint_sha256",
        artifact_path=path,
    )


def _mutate_checkpoint_schema(run_dir: Path, dataset_root: Path) -> None:
    """Replace the valid best-checkpoint schema version."""
    del dataset_root
    path = common.paths.resolve_best_checkpoint_file(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 2
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="best_checkpoint_sha256",
        artifact_path=path,
    )


def _remove_run_schema(run_dir: Path, dataset_root: Path) -> None:
    """Remove the run-summary schema marker."""
    del dataset_root
    summary = experiments.run.read_run_summary(run_dir)
    summary.pop("schema_version")
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        summary,
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_mutate_task_identity, "split task identity"),
        (_mutate_config_identity, "canonical generated leaf"),
        (_mutate_dataset_identity, "dataset fingerprint mismatch"),
        (_mutate_split_identity, "ordered membership digest mismatch"),
        (_mutate_checkpoint_identity, "Checkpoint run identity"),
        (_mutate_checkpoint_schema, "Unsupported checkpoint schema_version"),
        (_remove_run_schema, "Unsupported or missing run summary schema"),
    ],
    ids=(
        "task",
        "config",
        "dataset",
        "split",
        "checkpoint",
        "checkpoint-schema",
        "run-schema",
    ),
)
def test_saved_run_mismatches_are_rejected_before_forward(
    tmp_path: Path,
    completed_smoke: CompletedSmoke,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Path, Path], None],
    match: str,
) -> None:
    """
    Corrupt one of seven saved task/config/dataset/split/checkpoint/run-schema layers.

    Every parametrized mutation keeps unrelated outer evidence valid where needed
    but must fail its owning admission check before FNO forward, proving the complete
    saved-run identity chain is fail-closed.
    """
    run_dir = tmp_path / "run"
    dataset_root = tmp_path / "datasets"
    shutil.copytree(completed_smoke.run_dir, run_dir)
    shutil.copytree(completed_smoke.dataset_root, dataset_root)
    mutate(run_dir, dataset_root)

    def fail_forward(self: FNO, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        del self
        message = "Saved-run rejection reached model forward."
        raise AssertionError(message)

    monkeypatch.setattr(FNO, "forward", fail_forward)
    with pytest.raises((ValueError, RuntimeError), match=match):
        learning.inference.context.load_inference_context(
            run_dir=run_dir,
            dataset_root=dataset_root,
            split="eval",
            device_policy="cpu",
        )
