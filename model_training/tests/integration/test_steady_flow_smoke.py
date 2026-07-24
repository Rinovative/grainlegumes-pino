# ruff: noqa: S101
"""Exercise the complete steady-flow saved-run lifecycle on synthetic data."""

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

if TYPE_CHECKING:
    from collections.abc import Callable

_ID_DATASET = "tiny_steady_id"
_OOD_DATASET = "tiny_steady_ood_named"
_RUN_NAME = "tiny_steady_flow_smoke"
_SHAPE = (8, 8)
_ARTIFACT_CROP = 2


@dataclass(frozen=True)
class CompletedSmoke:
    """Paths and validated payloads produced by the one-epoch smoke run."""

    config: dict[str, Any]
    dataset_root: Path
    run_dir: Path
    id_payload: dict[str, Any]
    ood_payload: dict[str, Any]
    completed: dict[str, Any]


def _case_payload(
    task: domain.tasks.spec.TaskSpec,
    *,
    case_id: str,
    offset: float,
) -> dict[str, Any]:
    """Build one nonconstant steady-flow case with every canonical field."""
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
    return datasets.identity.build_case_payload(
        task=task,
        case_id=case_id,
        input_fields=input_fields,
        output_fields=output_fields,
        source_identity={"generator": "synthetic-smoke", "case": case_id},
        source_metadata={"offset": offset},
    )


def _merged_payload(
    task: domain.tasks.spec.TaskSpec,
    *,
    dataset_id: str,
    offsets: tuple[float, ...],
) -> dict[str, Any]:
    """Merge independently fingerprinted cases into one strict dataset."""
    cases = [
        _case_payload(
            task,
            case_id=f"{dataset_id}_case_{index:04d}",
            offset=offset,
        )
        for index, offset in enumerate(offsets)
    ]
    validated = [
        datasets.identity.validate_case_payload(
            case,
            task=task,
            verify_content=True,
        )
        for case in cases
    ]
    return datasets.identity.build_merged_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=[case.case_id for case in validated],
        source_identities=[case.source_identity for case in validated],
        source_metadata=[case.source_metadata for case in validated],
        case_fingerprints=[case.fingerprint for case in validated],
        inputs=torch.stack([case.inputs for case in validated]),
        outputs=torch.stack([case.outputs for case in validated]),
    )


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """Publish one merged payload under its logical dataset id."""
    dataset_id = str(payload["dataset_id"])
    destination = root / dataset_id / f"{dataset_id}.pt"
    common.serialization.atomic_torch_save(payload, destination)
    return destination


def _tiny_config(*, dataset_root: Path, output_root: Path) -> dict[str, Any]:
    """Resolve the smallest public one-epoch CPU FNO experiment."""
    raw = {
        "task": "steady_flow",
        "run": {
            "name": _RUN_NAME,
            "seed": 23,
            "deterministic": True,
            "device": "cpu",
            "prefix": None,
            "suffix": None,
        },
        "data": {
            "train_dataset": _ID_DATASET,
            "ood_datasets": [_OOD_DATASET],
            "train_ratio": 0.5,
            "ood_fraction": 0.5,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "model": {
            "kind": "fno",
            "params": {
                "n_modes": [2, 2],
                "hidden_channels": 4,
                "n_layers": 1,
            },
        },
        "loss": {
            "data": {
                "kind": "relative_l2",
                "space": "normalized",
                "weight": 1.0,
            },
            "physics": {"enabled": False},
        },
        "evaluation": {
            "metrics": [
                {
                    "id": "normalized_rmse",
                    "kind": "rmse",
                    "space": "normalized",
                    "fields": "all",
                    "reduction": "element_mean",
                }
            ],
            "objective": {"id": "normalized_rmse"},
        },
        "optimizer": {
            "kind": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
        },
        "scheduler": None,
        "training": {
            "epochs": 1,
            "evaluation_interval": 1,
            "mixed_precision": False,
        },
    }
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
    """Update one run-summary file digest after an intentional test mutation."""
    summary = experiments.run.read_run_summary(run_dir)
    summary[summary_key] = common.serialization.file_sha256(artifact_path)
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        summary,
    )


def _make_last_checkpoint_distinct(run_dir: Path) -> None:
    """Make valid last weights observably different from best weights."""
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
    """Return exact equality for nested tensor checkpoint state."""
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


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory: pytest.TempPathFactory) -> CompletedSmoke:
    """Train and validate exactly one tiny strict experiment."""
    root = tmp_path_factory.mktemp("steady_flow_lifecycle")
    dataset_root = root / "datasets"
    output_root = root / "runs"
    task = domain.tasks.registry.get_task("steady_flow")
    id_payload = _merged_payload(
        task,
        dataset_id=_ID_DATASET,
        offsets=(0.0, 1.0, 4.0, 10.0),
    )
    ood_payload = _merged_payload(
        task,
        dataset_id=_OOD_DATASET,
        offsets=(20.0, 21.0, 24.0, 30.0),
    )
    _save_dataset(dataset_root, id_payload)
    _save_dataset(dataset_root, ood_payload)

    config = _tiny_config(dataset_root=dataset_root, output_root=output_root)
    run_dir = experiments.run.prepare_fresh_run(
        config,
        run_dir=output_root / _RUN_NAME,
    )
    experiments.run.execute_prepared_run(
        config,
        run_dir=run_dir,
        persisted_config=config,
    )
    experiments.run.validate_completed_run(run_dir)

    _make_last_checkpoint_distinct(run_dir)
    completed = experiments.run.validate_completed_run(run_dir)
    assert not _state_dict_equal(
        completed["best_checkpoint"]["model_state_dict"],
        completed["last_checkpoint"]["model_state_dict"],
    )
    return CompletedSmoke(
        config=config,
        dataset_root=dataset_root,
        run_dir=run_dir,
        id_payload=id_payload,
        ood_payload=ood_payload,
        completed=completed,
    )


def _artifact_inventory(targets: tuple[Path, ...]) -> dict[Path, tuple[str, int]]:
    """Return content and write-time identity for every file below targets."""
    return {
        path: (common.serialization.file_sha256(path), path.stat().st_mtime_ns)
        for target in targets
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def test_real_steady_flow_lifecycle_and_artifacts(
    completed_smoke: CompletedSmoke,
) -> None:
    """Train, reload best, and generate/reuse/rebuild ID and named OOD outputs."""
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
        prefer_cuda=False,
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

    generated = analysis.artifact_service.build_artifacts(
        runs_root=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        batch_size=1,
        prefer_cuda=False,
    )
    frames = generated[smoke.run_dir.name]
    assert set(frames) == {"eval", "ood"}
    for role, index_key in (("eval", "eval_indices"), ("ood", "ood_indices")):
        frame = frames[role]
        assert not frame.empty
        assert frame.columns.is_unique
        assert frame["source_index"].tolist() == split[index_key].tolist()
        assert all(Path(path).is_file() for path in frame["npz_path"])
        assert "cont_mse_divepsu" not in frame
        assert {"rel_l2", "rel_h1", "rmse_p", "rmse_u", "rmse_v", "rmse_U"}.issubset(frame.columns)
        assert not {"l2", "h1", "phys_mse"}.intersection(frame.columns)
        with np.load(Path(frame.iloc[0]["npz_path"]), allow_pickle=False) as payload:
            assert payload["output_fields"].tolist() == ["p", "u", "v"]
            assert payload["artifact_fields"].tolist() == ["p", "u", "v", "U"]
            assert payload["artifact_units"].tolist() == ["Pa", "m/s", "m/s", "m/s"]
            assert np.array_equal(payload["Rc"], payload["div_eps_u"])
            selected_interior = payload["Rc"][_ARTIFACT_CROP:-_ARTIFACT_CROP, _ARTIFACT_CROP:-_ARTIFACT_CROP]
            assert frame.iloc[0]["cont_mse"] == pytest.approx(float(np.mean(selected_interior**2)))

    id_target = common.paths.resolve_id_analysis_dir(smoke.run_dir)
    ood_target = common.paths.resolve_ood_analysis_dir(
        smoke.run_dir,
        _OOD_DATASET,
    )
    targets = (id_target, ood_target)
    assert (id_target / f"{_ID_DATASET}.parquet").is_file()
    assert (ood_target / f"{_OOD_DATASET}.parquet").is_file()
    for target in targets:
        stored_provenance = json.loads((target / analysis.artifacts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        assert stored_provenance["outputs"] == analysis.artifacts.artifact_output_manifest(target)

    for target in targets:
        (target / "cache_marker.txt").write_text("preserve", encoding="utf-8")
    before_cache = _artifact_inventory(targets)
    cached = analysis.artifact_service.build_artifacts(
        runs_root=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        batch_size=1,
        prefer_cuda=False,
    )
    pd.testing.assert_frame_equal(cached[smoke.run_dir.name]["eval"], frames["eval"])
    pd.testing.assert_frame_equal(cached[smoke.run_dir.name]["ood"], frames["ood"])
    assert _artifact_inventory(targets) == before_cache

    provenance_path = id_target / analysis.artifacts.ARTIFACT_PROVENANCE_FILENAME
    valid_provenance = provenance_path.read_text(encoding="utf-8")
    provenance_path.write_text("{}\n", encoding="utf-8")
    incompatible_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifact_service.ArtifactCacheError,
        match="provenance is incompatible",
    ):
        analysis.artifact_service.run_or_load_artifacts(
            run_dir=smoke.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            max_cases=None,
            batch_size=1,
            prefer_cuda=False,
            dataset_root=smoke.dataset_root,
        )
    assert _artifact_inventory((id_target,)) == incompatible_cache
    provenance_path.write_text(valid_provenance, encoding="utf-8")

    corrupted_npz = Path(frames["eval"].iloc[0]["npz_path"])
    corrupted_npz.write_bytes(corrupted_npz.read_bytes() + b"corrupt")
    corrupted_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifact_service.ArtifactCacheError,
        match="payload digest manifest mismatch",
    ):
        analysis.artifact_service.run_or_load_artifacts(
            run_dir=smoke.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            max_cases=None,
            batch_size=1,
            prefer_cuda=False,
            dataset_root=smoke.dataset_root,
        )
    assert _artifact_inventory((id_target,)) == corrupted_cache

    sibling_marker = common.paths.resolve_ood_analysis_dir(smoke.run_dir, "unselected_ood") / "keep.txt"
    sibling_marker.parent.mkdir(parents=True)
    sibling_marker.write_text("keep", encoding="utf-8")
    rebuilt_artifacts = analysis.artifact_service.build_artifacts(
        runs_root=smoke.run_dir,
        dataset_root=smoke.dataset_root,
        batch_size=1,
        prefer_cuda=False,
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
    payload["schema_version"] = 0
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
        (_mutate_config_identity, "summary/config digest mismatch"),
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
    """Reject every saved identity layer before model execution."""
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
            prefer_cuda=False,
        )
