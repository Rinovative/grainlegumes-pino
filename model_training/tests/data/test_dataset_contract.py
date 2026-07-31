# ruff: noqa: S101, D103, EM101, EM102, PERF401, PLC0415, PLR2004, SLF001, TRY003
"""Protect direct generated-source to final training-dataset publication."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from src import common, datasets, domain

from data_generation import build_training_dataset as builder_module
from data_generation.build_training_dataset import build_batch_dataset

_SOLUTION_HEADER = (
    "% Length unit,m\n"
    "% x;y;br.kappaxx (m^2);br.kappayx (m^2);br.kappaxy (m^2);"
    "br.kappayy (m^2);int4(x,y) (1);int5(x,y) (Pa);p (Pa);u (m/s);v (m/s);br.U (m/s)"
)
_MANIFEST_FIELD_SCHEMA = {
    "input_columns": ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"],
    "solution_columns": ["x", "y", "kappaxx", "kappayx", "kappaxy", "kappayy", "eps", "p_bc", "p", "u", "v", "U"],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_rows(case_number: int) -> list[list[Any]]:
    offset = 10 * case_number
    return [
        ["0", "0", "1e-10", "0", "0", "2e-10", "0.4", "100", str(10 + offset), "1", "2", "3"],
        ["1", "0", "1e-10", "0", "0", "2e-10", "0.5", "101", str(11 + offset), "2", "3", "4"],
        ["0", "1", "1e-10", "0", "0", "2e-10", "0.6", "102", str(12 + offset), "3", "4", "5"],
        ["1", "1", "1e-10", "0", "0", "2e-10", "0.7", "103", str(13 + offset), "4", "5", "6"],
    ]


def _synthetic_generator_metadata() -> dict[str, Any]:
    statistics = {"min": 0.1, "max": 0.9, "mean": 0.5, "std": 0.1}
    return {
        "structure": {
            "parameters": {
                "background": {
                    "anisotropy": [1.0, 1.0],
                    "base_len_rel": 0.1,
                    "coupling": 0.0,
                    "ms_weight": [0.5, 0.5],
                    "smooth_len_rel": 0.1,
                },
                "noise": {"bias": 0.0, "granularity": 0.1, "level": 0.1},
                "rng_state": {"Seed": 1, "State": [1], "Type": "twister"},
                "seed": 1,
            },
            "statistics": {
                "noise": {"l2_norm": 1.0, "max_abs": 1.0},
                "structure": {
                    "z": dict(statistics),
                    "z_bg": {"mean": 0.5, "std": 0.1},
                    "z_noises": {"rms": 0.1},
                },
            },
        },
        "permeability": {
            "parameters": {
                "orientation": {"theta_jitter": 0.1, "theta_smooth_rel": 0.1},
                "permeability": {"k_mean": 1e-10, "s_logn": 0.1, "var_rel": 0.1},
                "tensor": {"a_gamma": 1.0, "a_max": 1.0, "tensor_strength": 0.1},
            },
            "statistics": {
                "kappa": dict(statistics),
                "tensor": {"det": {"mean": 1e-20}, "trace": {"mean": 2e-10}},
            },
        },
        "porosity": {
            "parameters": {
                "A_mat": 0.1,
                "A_rel": 0.1,
                "eps_max_global": 0.9,
                "eps_min_global": 0.1,
                "eps_ref": 0.5,
                "eps_smooth_rel": 0.1,
                "texture_amp": 0.1,
            },
            "statistics": {"eps": dict(statistics)},
        },
        "bc": {
            "parameters": {
                "a_gauss": 1.0,
                "a_lin": 1.0,
                "a_sin": 1.0,
                "f_sin": 1.0,
                "gauss_jitter": 0.1,
                "k_gauss": 1,
                "p_inlet_mean": 100.0,
                "sigma_gauss": 0.1,
            },
            "statistics": {"p_inlet": dict(statistics)},
        },
    }


def _write_case(raw_dir: Path, processed_dir: Path, case_number: int) -> str:
    case_id = f"case_{case_number:04d}"
    rows = _case_rows(case_number)
    raw_rows = [[row[index] for index in (0, 1, 2, 4, 5, 6, 7)] for row in rows]
    (raw_dir / f"{case_id}.csv").write_text(
        "\n".join(";".join(map(str, row)) for row in raw_rows) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "export": {
            "columns": _MANIFEST_FIELD_SCHEMA["input_columns"],
            "delimiter": ";",
            "file_base": case_id,
        },
        "fields_present": {"porosity": True, "pressure_bc": True, "tensor": True},
        "generator": _synthetic_generator_metadata(),
        "geometry": {"Lx": 1.0, "Ly": 1.0, "dx": 1.0, "dy": 1.0, "nx": 2, "ny": 2, "res": 1.0},
        "paths": {"csv": f"C:/Users/example/{case_id}.csv", "json": f"C:/Users/example/{case_id}.json"},
        "timestamp": "2026-01-01 00:00:00",
    }
    (raw_dir / f"{case_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (processed_dir / f"{case_id}_sol.csv").write_text(
        _SOLUTION_HEADER + "\n" + "\n".join(";".join(map(str, row)) for row in rows) + "\n",
        encoding="utf-8",
    )
    return case_id


def _write_generated_batch(
    root: Path,
    *,
    batch_name: str = "synthetic",
    case_numbers: tuple[int, ...] = (1, 2),
    timing_count: int = 1,
) -> tuple[Path, Path, Path]:
    meta_dir = root / "meta"
    raw_dir = root / "raw" / batch_name
    processed_dir = root / "processed" / batch_name
    meta_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    case_ids = [_write_case(raw_dir, processed_dir, number) for number in case_numbers]
    sample_frame = pd.DataFrame({"case_id": list(case_numbers), "alpha": [float(number) / 10 for number in case_numbers]})
    sample_csv = meta_dir / f"{batch_name}.csv"
    sample_frame.to_csv(sample_csv, sep=";", index=False, lineterminator="\n")
    sample_json = {
        "meta": {
            "method": "lhs",
            "variation": 0.2,
            "N": len(case_ids),
            "seed": 17,
            "base": {"alpha": 0.1},
            "param_names": ["alpha"],
            "timestamp": "2026-01-01 00:00:00",
        },
        "n_cases": len(case_ids),
    }
    (meta_dir / f"{batch_name}.json").write_text(json.dumps(sample_json), encoding="utf-8")
    records = []
    for case_id in case_ids:
        records.append(
            {
                "case_id": case_id,
                "status": "complete",
                "stage": "simulation",
                "message": "",
                "files": {
                    "raw_csv_sha256": _sha256(raw_dir / f"{case_id}.csv"),
                    "raw_json_sha256": _sha256(raw_dir / f"{case_id}.json"),
                    "solution_csv_sha256": _sha256(processed_dir / f"{case_id}_sol.csv"),
                    "solution_model_sha256": "",
                },
            }
        )
    manifest = {
        "schema_kind": "comsol_batch_manifest",
        "schema_version": 1,
        "batch_name": batch_name,
        "status": "complete",
        "configuration": {
            "method": "lhs",
            "variation": 0.2,
            "N": len(case_ids),
            "seed": 17,
            "Lx": 1.0,
            "Ly": 1.0,
            "res": 1.0,
            "save_model": False,
            "sample_sha256": _sha256(sample_csv),
            "template_name": "template_brinkman.mph",
            "template_sha256": "1" * 64,
        },
        "field_schema": _MANIFEST_FIELD_SCHEMA,
        "intended_case_ids": case_ids,
        "cases": records,
    }
    manifest_path = raw_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if timing_count >= 0:
        measured = case_ids[:timing_count]
        durations = np.asarray([float(index + 1) for index in range(len(measured))], dtype=np.float64)
        timing = {
            "schema_kind": "comsol_solve_timing",
            "schema_version": 1,
            "batch_name": batch_name,
            "batch_manifest_sha256": _sha256(manifest_path),
            "runtime": {
                "matlab_version": "test",
                "comsol_version": "test",
                "os": "test",
                "hostname": "test",
                "processor": "test",
                "case_execution": "sequential",
            },
            "cases": [{"case_id": case_id, "comsol_solve_s": float(index + 1)} for index, case_id in enumerate(measured)],
            "aggregates": {
                "measured_case_count": len(measured),
                "mean_s": [] if not measured else float(np.mean(durations)),
                "median_s": [] if not measured else float(np.percentile(durations, 50.0)),
                "p10_s": [] if not measured else float(np.percentile(durations, 10.0)),
                "p90_s": [] if not measured else float(np.percentile(durations, 90.0)),
            },
        }
        (processed_dir / "comsol_solve_timing.json").write_text(json.dumps(timing), encoding="utf-8")
    return meta_dir, raw_dir, processed_dir


def _write_private_batch_progress(raw_dir: Path) -> Path:
    manifest = json.loads((raw_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    progress = {
        **manifest,
        "schema_kind": "comsol_batch_progress",
        "schema_version": 1,
        "status": "in_progress",
    }
    progress_path = raw_dir / "batch_progress.json"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    return progress_path


def _build(tmp_path: Path, *, batch_name: str = "synthetic", **kwargs: Any) -> tuple[dict[str, Any], Path, Path]:
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root, batch_name=batch_name, **kwargs)
    training_root = tmp_path / "training"
    result = build_batch_dataset(
        batch_name,
        generated_data_root=generated_root,
        model_training_data_root=training_root,
    )
    return result, generated_root, training_root


def _rebind_case_sources(
    raw_dir: Path,
    processed_dir: Path,
    *,
    case_id: str = "case_0001",
) -> None:
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(record for record in manifest["cases"] if record["case_id"] == case_id)
    record["files"]["raw_csv_sha256"] = _sha256(raw_dir / f"{case_id}.csv")
    record["files"]["raw_json_sha256"] = _sha256(raw_dir / f"{case_id}.json")
    record["files"]["solution_csv_sha256"] = _sha256(processed_dir / f"{case_id}_sol.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    timing_path = processed_dir / "comsol_solve_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["batch_manifest_sha256"] = _sha256(manifest_path)
    timing_path.write_text(json.dumps(timing), encoding="utf-8")


def test_direct_builder_publishes_one_final_dataset_and_metadata(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, training_root = _build(tmp_path)
    dataset_path = training_root / "raw" / "synthetic" / "synthetic.pt"
    metadata_path = training_root / "meta" / "synthetic"
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    package = datasets.metadata.validate_dataset_metadata_directory(metadata_path, dataset_identity=identity)
    loaded = datasets.simulation.create_task_dataset(dataset_path, task=steady_task)

    assert result["dataset_path"] == dataset_path
    assert result["metadata_path"] == metadata_path
    assert payload["schema_version"] == datasets.identity.TRAINING_DATASET_SCHEMA_VERSION
    assert payload["schema_kind"] == datasets.identity.TRAINING_DATASET_SCHEMA_KIND
    assert payload["sample_ids"] == ["case_0001", "case_0002"]
    assert payload["inputs"].shape == (2, 7, 2, 2)
    assert payload["outputs"].shape == (2, 3, 2, 2)
    assert payload["source_metadata"][0]["parameters"] == {"alpha": 0.1}
    assert set(payload["source_metadata"][0]) == {"case_id", "geometry", "parameters"}
    assert "paths" not in payload["source_metadata"][0]
    assert "timestamp" not in payload["source_metadata"][0]
    assert loaded[0]["meta"] == payload["source_metadata"][0]
    assert package.timing is not None
    assert package.timing_summary == {"status": "partial", "measured_case_count": 1, "intended_case_count": 2}
    assert not list(training_root.rglob("case_*.pt"))
    assert not list(training_root.rglob("meta.pt"))
    lock_path = common.paths.resolve_dataset_build_lock_path(
        "synthetic",
        model_training_data_root=training_root,
    )
    transaction_path = common.paths.resolve_dataset_build_transaction_path(
        "synthetic",
        model_training_data_root=training_root,
    )
    assert lock_path.is_file()
    assert lock_path.stat().st_size == 0
    assert not transaction_path.exists()
    assert not list(transaction_path.parent.iterdir())
    assert not (training_root / "processed").exists()


def test_active_dataset_lock_rejects_builder_and_persistent_anchor_is_harmless(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root)
    training_root = tmp_path / "training"
    lock_path = common.paths.resolve_dataset_build_lock_path(
        "synthetic",
        model_training_data_root=training_root,
    )
    acquired = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with common.locking.exclusive_file_lock(lock_path, blocking=False):
                acquired.set()
                release.wait(timeout=10)
        except (OSError, common.locking.FileLockUnavailableError) as error:
            errors.append(error)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert acquired.wait(timeout=5)
        with pytest.raises(common.locking.FileLockUnavailableError, match="already held"):
            build_batch_dataset(
                "synthetic",
                generated_data_root=generated_root,
                model_training_data_root=training_root,
            )
    finally:
        release.set()
        holder.join(timeout=5)
    assert not holder.is_alive()
    assert not errors
    assert lock_path.is_file()
    assert lock_path.stat().st_size == 0

    result = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=training_root,
    )
    assert result["status"] == "complete"
    assert lock_path.is_file()


def test_manifest_order_drives_final_sample_order(tmp_path: Path) -> None:
    result, _generated_root, _training_root = _build(tmp_path, case_numbers=(2, 1), timing_count=2)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    assert payload["sample_ids"] == ["case_0002", "case_0001"]
    assert payload["source_metadata"][0]["case_id"] == "case_0002"
    assert payload["source_metadata"][1]["case_id"] == "case_0001"


def test_sample_csv_serialization_is_operational_not_scientific_identity(tmp_path: Path) -> None:
    first, _generated, _training = _build(tmp_path / "first")
    second_generated = tmp_path / "second" / "generated"
    meta_dir, raw_dir, processed_dir = _write_generated_batch(second_generated)
    sample_path = meta_dir / "synthetic.csv"
    sample_path.write_text(sample_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["sample_sha256"] = _sha256(sample_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    timing_path = processed_dir / "comsol_solve_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["batch_manifest_sha256"] = _sha256(manifest_path)
    timing_path.write_text(json.dumps(timing), encoding="utf-8")
    second = build_batch_dataset(
        "synthetic",
        generated_data_root=second_generated,
        model_training_data_root=tmp_path / "second" / "training",
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]


def test_sample_snapshots_remain_coherent_when_live_sidecars_change_mid_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    meta_dir, _raw_dir, _processed_dir = _write_generated_batch(generated_root)
    sample_csv_path = meta_dir / "synthetic.csv"
    sample_json_path = meta_dir / "synthetic.json"
    original_csv = sample_csv_path.read_bytes()
    original_json = sample_json_path.read_bytes()
    original_interpret = builder_module._interpret_generated_case
    call_count = 0

    def mutate_live_samples_after_first_case(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        interpreted = original_interpret(*args, **kwargs)
        call_count += 1
        if call_count == 1:
            sample_csv_path.write_bytes(original_csv + b"\n")
            sample_json_path.write_bytes(original_json + b"\n")
        return interpreted

    monkeypatch.setattr(builder_module, "_interpret_generated_case", mutate_live_samples_after_first_case)
    training_root = tmp_path / "training"
    result = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=training_root,
    )
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    snapshot_csv = result["metadata_path"] / datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME
    snapshot_json = result["metadata_path"] / datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME

    assert snapshot_csv.read_bytes() == original_csv
    assert snapshot_json.read_bytes() == original_json
    assert _sha256(snapshot_csv) == payload["source_provenance"]["source_sample_csv_sha256"]
    assert _sha256(snapshot_json) == payload["source_provenance"]["source_sample_json_sha256"]


@pytest.mark.parametrize("target", ["raw_csv", "raw_json", "solution_csv"])
def test_builder_rejects_manifest_bound_source_tampering(tmp_path: Path, target: str) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(generated_root)
    targets = {
        "raw_csv": raw_dir / "case_0001.csv",
        "raw_json": raw_dir / "case_0001.json",
        "solution_csv": processed_dir / "case_0001_sol.csv",
    }
    targets[target].write_bytes(targets[target].read_bytes() + b"tampered")
    training_root = tmp_path / "training"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_batch_dataset("synthetic", generated_data_root=generated_root, model_training_data_root=training_root)
    assert not (training_root / "raw" / "synthetic").exists()
    assert not (training_root / "meta" / "synthetic").exists()


def test_builder_rejects_extra_generated_membership(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, _processed = _write_generated_batch(generated_root)
    (raw_dir / "case_9999.csv").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly match"):
        build_batch_dataset("synthetic", generated_data_root=generated_root, model_training_data_root=tmp_path / "training")


@pytest.mark.parametrize("source_kind", ["raw", "solution"])
def test_builder_rejects_extra_csv_columns(tmp_path: Path, source_kind: str) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(
        generated_root,
        case_numbers=(1,),
        timing_count=1,
    )
    if source_kind == "raw":
        path = raw_dir / "case_0001.csv"
        lines = [f"{line};999" for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        path = processed_dir / "case_0001_sol.csv"
        source_lines = path.read_text(encoding="utf-8").splitlines()
        lines = [*source_lines[:2], *(f"{line};999" for line in source_lines[2:])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rebind_case_sources(raw_dir, processed_dir)

    expected_count = 7 if source_kind == "raw" else 12
    with pytest.raises(ValueError, match=rf"exactly {expected_count} columns"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("export_columns", "export contract"),
        ("delimiter", "export contract"),
        ("fields_present", "every generated field"),
        ("spacing", "dx/dy must equal res"),
        ("nested_generator_key", "generator.structure.parameters keys"),
    ],
)
def test_builder_rejects_raw_metadata_schema_drift(tmp_path: Path, corruption: str, message: str) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(
        generated_root,
        case_numbers=(1,),
        timing_count=1,
    )
    metadata_path = raw_dir / "case_0001.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if corruption == "export_columns":
        metadata["export"]["columns"] = [*metadata["export"]["columns"], "extra"]
    elif corruption == "delimiter":
        metadata["export"]["delimiter"] = ","
    elif corruption == "fields_present":
        metadata["fields_present"]["tensor"] = False
    elif corruption == "spacing":
        metadata["geometry"]["dx"] = 0.5
    else:
        metadata["generator"]["structure"]["parameters"]["source_path"] = "C:/private"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rebind_case_sources(raw_dir, processed_dir)

    with pytest.raises(ValueError, match=message):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


def test_builder_reverifies_manifest_hashes_after_case_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root, case_numbers=(1,), timing_count=1)
    original = builder_module._load_case_sources

    def mutate_after_read(csv_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        result = original(csv_path, metadata_path)
        metadata_path.write_text(metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return result

    monkeypatch.setattr(builder_module, "_load_case_sources", mutate_after_read)
    with pytest.raises(RuntimeError, match=r"SHA-256 mismatch.*raw JSON"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


@pytest.mark.parametrize("corruption", ["csv_hash", "json_seed", "json_columns"])
def test_builder_validates_parameter_sample_metadata(tmp_path: Path, corruption: str) -> None:
    generated_root = tmp_path / "generated"
    meta_dir, _raw, _processed = _write_generated_batch(generated_root)
    if corruption == "csv_hash":
        path = meta_dir / "synthetic.csv"
        path.write_text(path.read_text(encoding="utf-8") + "3;0.3\n", encoding="utf-8")
        match = "CSV SHA-256"
    else:
        path = meta_dir / "synthetic.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "json_seed":
            payload["meta"]["seed"] = 18
            match = "meta.seed"
        else:
            payload["meta"]["param_names"] = ["different"]
            match = "columns"
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match=match):
        build_batch_dataset("synthetic", generated_data_root=generated_root, model_training_data_root=tmp_path / "training")


def _source_sample_contract(root: Path) -> tuple[bytes, bytes, dict[str, Any]]:
    meta_dir, raw_dir, _processed_dir = _write_generated_batch(root)
    sample_csv = (meta_dir / "synthetic.csv").read_bytes()
    sample_json = (meta_dir / "synthetic.json").read_bytes()
    manifest = json.loads((raw_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    return sample_csv, sample_json, manifest


def test_public_source_sample_boundary_accepts_semantic_agreement(tmp_path: Path) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")

    semantics = datasets.metadata.validate_source_sample_semantics(
        sample_csv,
        sample_json,
        source_manifest=manifest,
    )

    assert semantics.case_ids == ("case_0001", "case_0002")
    assert semantics.parameter_names == ("alpha",)
    assert semantics.parameter_rows == ({"alpha": 0.1}, {"alpha": 0.2})
    assert semantics.sampling == {
        "method": "lhs",
        "variation": 0.2,
        "N": 2,
        "seed": 17,
        "base": {"alpha": 0.1},
        "param_names": ["alpha"],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 18, "meta.seed"),
        ("method", "uniform", "meta.method"),
        ("variation", 0.3, "meta.variation"),
    ],
)
def test_public_source_sample_boundary_rejects_generation_configuration_mismatch(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    payload = json.loads(sample_json)
    payload["meta"][field] = value

    with pytest.raises(ValueError, match=message):
        datasets.metadata.validate_source_sample_semantics(
            sample_csv,
            json.dumps(payload).encode(),
            source_manifest=manifest,
        )


@pytest.mark.parametrize(("field", "value"), [("N", True), ("seed", False), ("N", 2.0)])
def test_public_source_sample_boundary_rejects_malformed_integer_fields(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    payload = json.loads(sample_json)
    payload["meta"][field] = value

    with pytest.raises(ValueError, match=r"must be a (positive|non-negative) integer"):
        datasets.metadata.validate_source_sample_semantics(
            sample_csv,
            json.dumps(payload).encode(),
            source_manifest=manifest,
        )


def test_public_source_sample_boundary_rejects_case_count_mismatch(tmp_path: Path) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    payload = json.loads(sample_json)
    payload["n_cases"] = 3

    with pytest.raises(ValueError, match="n_cases"):
        datasets.metadata.validate_source_sample_semantics(
            sample_csv,
            json.dumps(payload).encode(),
            source_manifest=manifest,
        )


def test_public_source_sample_boundary_rejects_csv_json_row_count_mismatch(tmp_path: Path) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    shortened_csv = ("\n".join(sample_csv.decode().splitlines()[:-1]) + "\n").encode()
    manifest["configuration"]["sample_sha256"] = hashlib.sha256(shortened_csv).hexdigest()

    with pytest.raises(ValueError, match="row count"):
        datasets.metadata.validate_source_sample_semantics(
            shortened_csv,
            sample_json,
            source_manifest=manifest,
        )


def test_public_source_sample_boundary_rejects_ordered_sample_id_mismatch(tmp_path: Path) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    lines = sample_csv.decode().splitlines()
    reordered_csv = ("\n".join([lines[0], lines[2], lines[1]]) + "\n").encode()
    manifest["configuration"]["sample_sha256"] = hashlib.sha256(reordered_csv).hexdigest()

    with pytest.raises(ValueError, match="ordered sample IDs"):
        datasets.metadata.validate_source_sample_semantics(
            reordered_csv,
            sample_json,
            source_manifest=manifest,
        )


def test_public_source_sample_boundary_rejects_variable_name_mismatch(tmp_path: Path) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    payload = json.loads(sample_json)
    payload["meta"]["param_names"] = ["beta"]

    with pytest.raises(ValueError, match="columns"):
        datasets.metadata.validate_source_sample_semantics(
            sample_csv,
            json.dumps(payload).encode(),
            source_manifest=manifest,
        )


def test_source_sample_timestamp_is_operational_not_scientific_identity(tmp_path: Path) -> None:
    sample_csv, sample_json, manifest = _source_sample_contract(tmp_path / "generated")
    changed = json.loads(sample_json)
    changed["meta"]["timestamp"] = "2099-12-31 23:59:59"

    original = datasets.metadata.validate_source_sample_semantics(
        sample_csv,
        sample_json,
        source_manifest=manifest,
    )
    updated_semantics = datasets.metadata.validate_source_sample_semantics(
        sample_csv,
        json.dumps(changed).encode(),
        source_manifest=manifest,
    )

    assert original.generated_batch_identity == updated_semantics.generated_batch_identity
    assert original.sampling == updated_semantics.sampling


def test_metadata_rejects_hash_rebound_json_seed_mismatch(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    sample_path = result["metadata_path"] / datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["meta"]["seed"] = 18
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    metadata_path = result["metadata_path"] / datasets.metadata.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    snapshot = metadata["artifacts"]["snapshots"][datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME]
    snapshot["sha256"] = _sha256(sample_path)
    snapshot["size_bytes"] = sample_path.stat().st_size
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"meta\.seed"):
        datasets.metadata.validate_dataset_metadata_directory(
            result["metadata_path"],
            dataset_identity=identity,
        )


def test_metadata_rejects_hash_rebound_sampling_base_mismatch_with_final_identity(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    sample_path = result["metadata_path"] / datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["meta"]["base"]["alpha"] = 0.2
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    metadata_path = result["metadata_path"] / datasets.metadata.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    snapshot = metadata["artifacts"]["snapshots"][datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME]
    snapshot["sha256"] = _sha256(sample_path)
    snapshot["size_bytes"] = sample_path.stat().st_size
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="final generated-batch scientific identity"):
        datasets.metadata.validate_dataset_metadata_directory(
            result["metadata_path"],
            dataset_identity=identity,
        )


def test_source_sample_values_are_bound_to_final_payload_metadata(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    assert identity.source_metadata is not None
    source_metadata = [dict(value) for value in identity.source_metadata]
    source_metadata[0] = {**source_metadata[0], "parameters": {"alpha": 9.9}}
    inconsistent_identity = replace(identity, source_metadata=tuple(source_metadata))

    with pytest.raises(ValueError, match=r"parameters.alpha.*source-sample CSV"):
        datasets.metadata.validate_dataset_metadata_directory(
            result["metadata_path"],
            dataset_identity=inconsistent_identity,
        )


@pytest.mark.parametrize("dataset_id", ["lhs_var80_seed3001", "lhs_var120_seed4001"])
def test_current_production_metadata_package_passes_read_only_cross_binding(dataset_id: str) -> None:
    model_training_root = Path(__file__).parents[2]
    metadata_dir = model_training_root / "data/meta" / dataset_id
    metadata = json.loads((metadata_dir / datasets.metadata.METADATA_FILENAME).read_text(encoding="utf-8"))
    manifest = json.loads((metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    scientific = metadata["scientific_identity"]
    identity = datasets.identity.DatasetIdentity(
        dataset_id=dataset_id,
        task=scientific["task_id"],
        task_contract_digest=scientific["task_contract_digest"],
        fingerprint=scientific["dataset_fingerprint"],
        sample_ids=tuple(manifest["intended_case_ids"]),
        sample_count=scientific["sample_count"],
        spatial_shape=tuple(scientific["spatial_shape"]),
        generated_batch_identity_sha256=scientific["generated_batch_identity_sha256"],
    )

    package = datasets.metadata.validate_dataset_metadata_directory(
        metadata_dir,
        dataset_identity=identity,
    )

    assert package.source_manifest["intended_case_ids"] == manifest["intended_case_ids"]
    assert package.metadata["scientific_identity"]["generated_batch_identity_sha256"] == scientific["generated_batch_identity_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_length_unit", "Length unit"),
        ("wrong_pressure_unit", "field/unit header"),
        ("nonuniform_grid", "dimensions"),
        ("invalid_porosity", "Porosity"),
        ("non_spd", "positive definite"),
        ("nonfinite", "non-finite"),
    ],
)
def test_builder_rejects_invalid_units_grid_and_physics(tmp_path: Path, mutation: str, message: str) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(generated_root, case_numbers=(1,), timing_count=1)
    solution = processed_dir / "case_0001_sol.csv"
    lines = solution.read_text(encoding="utf-8").splitlines()
    if mutation == "missing_length_unit":
        lines.pop(0)
    elif mutation == "wrong_pressure_unit":
        lines[1] = lines[1].replace("p (Pa)", "p (bar)")
    elif mutation == "nonuniform_grid":
        # A 2-point axis is necessarily uniform; make the metadata/grid cardinality invalid.
        metadata_path = raw_dir / "case_0001.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["geometry"]["nx"] = 3
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        values = lines[2].split(";")
        if mutation == "invalid_porosity":
            values[6] = "0"
        elif mutation == "non_spd":
            values[3] = "2e-10"
            values[4] = "2e-10"
        elif mutation == "nonfinite":
            values[8] = "nan"
        lines[2] = ";".join(values)
        if mutation in {"invalid_porosity", "non_spd"}:
            raw_path = raw_dir / "case_0001.csv"
            raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
            raw_values = raw_lines[0].split(";")
            raw_values[5 if mutation == "invalid_porosity" else 3] = values[6 if mutation == "invalid_porosity" else 4]
            raw_lines[0] = ";".join(raw_values)
            raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    solution.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Rebind only the intentionally changed source so scientific validation is reached.
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["files"]["solution_csv_sha256"] = _sha256(solution)
    if mutation == "nonuniform_grid":
        manifest["cases"][0]["files"]["raw_json_sha256"] = _sha256(raw_dir / "case_0001.json")
    if mutation in {"invalid_porosity", "non_spd"}:
        manifest["cases"][0]["files"]["raw_csv_sha256"] = _sha256(raw_dir / "case_0001.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    timing_path = processed_dir / "comsol_solve_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["batch_manifest_sha256"] = _sha256(manifest_path)
    timing_path.write_text(json.dumps(timing), encoding="utf-8")

    with pytest.raises((ValueError, KeyError), match=message):
        build_batch_dataset("synthetic", generated_data_root=generated_root, model_training_data_root=tmp_path / "training")


def test_raw_solution_agreement_admits_only_scale_level_interpolation_noise(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(
        generated_root,
        case_numbers=(1,),
        timing_count=1,
    )
    raw_path = raw_dir / "case_0001.csv"
    raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
    raw_values = raw_lines[0].split(";")
    raw_values[6] = "0"
    raw_lines[0] = ";".join(raw_values)
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    solution_path = processed_dir / "case_0001_sol.csv"
    solution_lines = solution_path.read_text(encoding="utf-8").splitlines()
    solution_values = solution_lines[2].split(";")
    solution_values[3] = "1e-24"
    solution_values[4] = "1e-24"
    solution_values[7] = "7e-13"
    solution_lines[2] = ";".join(solution_values)
    solution_path.write_text("\n".join(solution_lines) + "\n", encoding="utf-8")
    _rebind_case_sources(raw_dir, processed_dir)

    result = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=tmp_path / "training",
    )
    assert result["status"] == "complete"


def test_raw_solution_agreement_rejects_material_difference(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(
        generated_root,
        case_numbers=(1,),
        timing_count=1,
    )
    solution_path = processed_dir / "case_0001_sol.csv"
    lines = solution_path.read_text(encoding="utf-8").splitlines()
    values = lines[2].split(";")
    values[7] = str(float(values[7]) + 0.01)
    lines[2] = ";".join(values)
    solution_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rebind_case_sources(raw_dir, processed_dir)

    with pytest.raises(ValueError, match="Raw input field 'p_bc' disagrees"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


def test_partial_or_missing_timing_does_not_invalidate_scientific_dataset(tmp_path: Path) -> None:
    partial, _generated, _training = _build(tmp_path / "partial", timing_count=1)
    missing, _generated, _training = _build(tmp_path / "missing", timing_count=-1)
    assert partial["timing_coverage"]["status"] == "partial"
    assert missing["timing_coverage"] == {"status": "missing", "measured_case_count": 0, "intended_case_count": 2}


def test_empty_timing_snapshot_is_valid_missing_coverage(tmp_path: Path) -> None:
    result, _generated, _training = _build(tmp_path, timing_count=0)
    assert result["timing_coverage"] == {
        "status": "missing",
        "measured_case_count": 0,
        "intended_case_count": 2,
    }


def test_direct_builder_rejects_timing_aggregate_corruption(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root, timing_count=1)
    timing_path = generated_root / "processed" / "synthetic" / "comsol_solve_timing.json"
    payload = json.loads(timing_path.read_text(encoding="utf-8"))
    payload["aggregates"]["mean_s"] += 0.25
    timing_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mean_s is not derived"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


def test_failed_build_and_overwrite_refusal_leave_authoritative_targets_intact(tmp_path: Path) -> None:
    result, generated_root, training_root = _build(tmp_path)
    original_dataset_hash = _sha256(result["dataset_path"])
    original_metadata_hash = _sha256(result["metadata_path"] / datasets.metadata.METADATA_FILENAME)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_batch_dataset("synthetic", generated_data_root=generated_root, model_training_data_root=training_root)
    assert _sha256(result["dataset_path"]) == original_dataset_hash
    assert _sha256(result["metadata_path"] / datasets.metadata.METADATA_FILENAME) == original_metadata_hash


def test_consolidated_metadata_package_has_exact_owned_membership(tmp_path: Path) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    assert {path.name for path in result["metadata_path"].iterdir()} == {
        datasets.metadata.METADATA_FILENAME,
        datasets.metadata.SOURCE_MANIFEST_FILENAME,
        datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME,
        datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME,
        datasets.metadata.COMSOL_TIMING_FILENAME,
    }
    metadata = json.loads((result["metadata_path"] / datasets.metadata.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["schema_kind"] == datasets.metadata.METADATA_SCHEMA_KIND
    assert metadata["schema_version"] == 1
    assert metadata["artifacts"]["dataset"] == {
        "filename": "synthetic.pt",
        "sha256": _sha256(result["dataset_path"]),
        "size_bytes": result["dataset_path"].stat().st_size,
    }
    assert set(metadata["artifacts"]["snapshots"]) == {
        datasets.metadata.SOURCE_MANIFEST_FILENAME,
        datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME,
        datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME,
        datasets.metadata.COMSOL_TIMING_FILENAME,
    }


def test_incomplete_metadata_package_is_rejected(tmp_path: Path, steady_task: domain.tasks.spec.TaskSpec) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    (result["metadata_path"] / datasets.metadata.METADATA_FILENAME).unlink()
    with pytest.raises(ValueError, match=r"missing=.*dataset_metadata\.json"):
        datasets.metadata.validate_dataset_metadata_directory(result["metadata_path"], dataset_identity=identity)


def test_metadata_generated_batch_digest_is_bound_to_final_dataset(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    metadata_path = result["metadata_path"] / datasets.metadata.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scientific_identity"]["generated_batch_identity_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific identity does not match"):
        datasets.metadata.validate_dataset_metadata_directory(result["metadata_path"], dataset_identity=identity)


def test_dataset_artifact_hash_binding_detects_file_mutation(tmp_path: Path, steady_task: domain.tasks.spec.TaskSpec) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    result["dataset_path"].write_bytes(result["dataset_path"].read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="does not match dataset metadata SHA-256 and size"):
        datasets.metadata.validate_dataset_metadata_directory(
            result["metadata_path"],
            dataset_identity=identity,
            dataset_path=result["dataset_path"],
        )


def test_dataset_metadata_detects_snapshot_tampering(tmp_path: Path, steady_task: domain.tasks.spec.TaskSpec) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    sample_snapshot = result["metadata_path"] / datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME
    sample_snapshot.write_bytes(sample_snapshot.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        datasets.metadata.validate_dataset_metadata_directory(
            result["metadata_path"],
            dataset_identity=identity,
            dataset_path=result["dataset_path"],
        )


def test_metadata_manifest_binding_rejects_rebound_sample_csv(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    sample_snapshot = result["metadata_path"] / datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME
    sample_snapshot.write_bytes(sample_snapshot.read_bytes() + b"tampered")

    metadata_path = result["metadata_path"] / datasets.metadata.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_entry = metadata["artifacts"]["snapshots"][datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME]
    sample_entry["sha256"] = _sha256(sample_snapshot)
    sample_entry["size_bytes"] = sample_snapshot.stat().st_size
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the source manifest SHA-256"):
        datasets.metadata.validate_dataset_metadata_directory(result["metadata_path"], dataset_identity=identity)


def test_metadata_rejects_mutated_dataset_schema_version(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    result, _generated_root, _training_root = _build(tmp_path)
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    identity = datasets.identity.validate_training_dataset_payload(payload, task=steady_task, verify_content=True)
    metadata_path = result["metadata_path"] / datasets.metadata.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scientific_identity"]["dataset_schema_version"] = datasets.identity.TRAINING_DATASET_SCHEMA_VERSION + 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_schema_version must be integer 1"):
        datasets.metadata.validate_dataset_metadata_directory(result["metadata_path"], dataset_identity=identity)


@pytest.mark.parametrize("schema_version", [True, 1.0, 2])
def test_publication_transaction_requires_integer_version_one(
    tmp_path: Path,
    schema_version: Any,
) -> None:
    training_root = tmp_path / "training"
    staging_root = training_root / ".synthetic.dataset-build.invalid-version.tmp"
    staging_root.mkdir(parents=True)
    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)
    record = builder_module._publication_transaction_record(
        dataset_id="synthetic",
        phase="building",
        staging_root=staging_root,
    )
    record["schema_version"] = schema_version
    common.serialization.atomic_write_json(transaction_path, record)

    with pytest.raises(RuntimeError, match="invalid identity or scalar fields"):
        builder_module._load_publication_transaction(
            transaction_path,
            training_root=training_root,
            dataset_id="synthetic",
        )


@pytest.mark.parametrize("failed_stage", ["metadata", "dataset"])
def test_direct_builder_publication_failure_leaves_no_authoritative_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    """A staged publication error must not expose either half of the build."""
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root)
    training_root = tmp_path / "training"
    dataset_target = training_root / "raw" / "synthetic"
    metadata_target = training_root / "meta" / "synthetic"
    transaction_path = common.paths.resolve_dataset_build_transaction_path(
        "synthetic",
        model_training_data_root=training_root,
    )
    original_replace = Path.replace

    def fail_selected_publication(source: Path, target: Path | str) -> Path:
        target_path = Path(target)
        should_fail = (failed_stage == "metadata" and target_path == metadata_target) or (failed_stage == "dataset" and target_path == dataset_target)
        if should_fail:
            raise OSError(f"injected {failed_stage} publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_selected_publication)
    with pytest.raises(OSError, match=f"injected {failed_stage} publication failure"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=training_root,
        )
    assert not dataset_target.exists()
    assert not metadata_target.exists()
    assert transaction_path.is_file()
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    assert transaction["phase"] == "ready"
    staging_root = Path(transaction["staging_root"])
    assert (staging_root / "raw" / "synthetic" / "synthetic.pt").is_file()
    assert (staging_root / "meta" / "synthetic" / datasets.metadata.METADATA_FILENAME).is_file()
    assert not (training_root / "processed").exists()


@pytest.mark.parametrize(
    "state",
    ["both_staged", "metadata_final", "dataset_final", "both_final"],
)
def test_ready_publication_transaction_recovers_every_rename_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    result, generated_root, training_root = _build(tmp_path)
    dataset_dir = result["dataset_path"].parent
    metadata_dir = result["metadata_path"]
    staging_root = training_root / ".synthetic.dataset-build.recovery.tmp"
    staging_root.mkdir()
    staged_dataset_dir = staging_root / "raw" / "synthetic"
    staged_metadata_dir = staging_root / "meta" / "synthetic"
    dataset_sha256 = _sha256(result["dataset_path"])
    dataset_size = result["dataset_path"].stat().st_size
    metadata_sha256 = _sha256(metadata_dir / datasets.metadata.METADATA_FILENAME)

    if state in {"both_staged", "metadata_final"}:
        staged_dataset_dir.parent.mkdir(parents=True)
        dataset_dir.replace(staged_dataset_dir)
    if state in {"both_staged", "dataset_final"}:
        staged_metadata_dir.parent.mkdir(parents=True)
        metadata_dir.replace(staged_metadata_dir)

    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)
    common.serialization.atomic_write_json(
        transaction_path,
        builder_module._publication_transaction_record(
            dataset_id="synthetic",
            phase="ready",
            staging_root=staging_root,
            dataset_sha256=dataset_sha256,
            dataset_size=dataset_size,
            dataset_metadata_sha256=metadata_sha256,
        ),
    )
    raw_source = generated_root / "raw" / "synthetic" / "case_0001.csv"
    raw_source.write_bytes(raw_source.read_bytes() + b"corrupted after ready transaction")

    def reject_rebuild(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ready publication recovery rebuilt generated cases")

    monkeypatch.setattr(builder_module, "_interpret_generated_case", reject_rebuild)
    recovered = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=training_root,
    )

    assert recovered["status"] == "complete"
    assert recovered["dataset_fingerprint"] == result["dataset_fingerprint"]
    assert result["dataset_path"].is_file()
    assert result["metadata_path"].is_dir()
    assert not transaction_path.exists()
    assert not staging_root.exists()


def test_private_progress_blocks_ready_transaction_recovery(tmp_path: Path) -> None:
    result, generated_root, training_root = _build(tmp_path)
    dataset_dir = result["dataset_path"].parent
    metadata_dir = result["metadata_path"]
    staging_root = training_root / ".synthetic.dataset-build.progress-recovery.tmp"
    staging_root.mkdir()
    staged_dataset_dir = staging_root / "raw" / "synthetic"
    staged_metadata_dir = staging_root / "meta" / "synthetic"
    staged_dataset_dir.parent.mkdir(parents=True)
    staged_metadata_dir.parent.mkdir(parents=True)
    dataset_dir.replace(staged_dataset_dir)
    metadata_dir.replace(staged_metadata_dir)
    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)
    common.serialization.atomic_write_json(
        transaction_path,
        builder_module._publication_transaction_record(
            dataset_id="synthetic",
            phase="ready",
            staging_root=staging_root,
            dataset_sha256=_sha256(staged_dataset_dir / "synthetic.pt"),
            dataset_size=(staged_dataset_dir / "synthetic.pt").stat().st_size,
            dataset_metadata_sha256=_sha256(staged_metadata_dir / datasets.metadata.METADATA_FILENAME),
        ),
    )
    _write_private_batch_progress(generated_root / "raw" / "synthetic")

    with pytest.raises(RuntimeError, match="active or interrupted COMSOL progress"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=training_root,
        )

    assert not dataset_dir.exists()
    assert not metadata_dir.exists()
    assert transaction_path.is_file()
    assert staged_dataset_dir.is_dir()
    assert staged_metadata_dir.is_dir()


@pytest.mark.parametrize(
    ("source_change", "message"),
    [
        ("progress", "active or interrupted COMSOL progress"),
        ("manifest", "Generation manifest changed"),
    ],
)
def test_source_change_during_ready_recovery_blocks_remaining_renames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_change: str,
    message: str,
) -> None:
    result, generated_root, training_root = _build(tmp_path)
    dataset_dir = result["dataset_path"].parent
    metadata_dir = result["metadata_path"]
    staging_root = training_root / ".synthetic.dataset-build.recovery-race.tmp"
    staging_root.mkdir()
    staged_dataset_dir = staging_root / "raw" / "synthetic"
    staged_metadata_dir = staging_root / "meta" / "synthetic"
    staged_dataset_dir.parent.mkdir(parents=True)
    staged_metadata_dir.parent.mkdir(parents=True)
    dataset_dir.replace(staged_dataset_dir)
    metadata_dir.replace(staged_metadata_dir)
    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)
    common.serialization.atomic_write_json(
        transaction_path,
        builder_module._publication_transaction_record(
            dataset_id="synthetic",
            phase="ready",
            staging_root=staging_root,
            dataset_sha256=_sha256(staged_dataset_dir / "synthetic.pt"),
            dataset_size=(staged_dataset_dir / "synthetic.pt").stat().st_size,
            dataset_metadata_sha256=_sha256(staged_metadata_dir / datasets.metadata.METADATA_FILENAME),
        ),
    )
    raw_dir = generated_root / "raw" / "synthetic"
    manifest_path = raw_dir / "batch_manifest.json"
    original_validate = datasets.metadata.validate_dataset_metadata_directory
    injected = False

    def inject_source_change(*args: Any, **kwargs: Any) -> Any:
        nonlocal injected
        package = original_validate(*args, **kwargs)
        if not injected:
            injected = True
            if source_change == "progress":
                _write_private_batch_progress(raw_dir)
            else:
                manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        return package

    monkeypatch.setattr(datasets.metadata, "validate_dataset_metadata_directory", inject_source_change)

    with pytest.raises(RuntimeError, match=message):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=training_root,
        )

    assert not dataset_dir.exists()
    assert not metadata_dir.exists()
    assert transaction_path.is_file()
    assert staged_dataset_dir.is_dir()
    assert staged_metadata_dir.is_dir()


def test_interrupted_build_retains_inspectable_transaction_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root)
    training_root = tmp_path / "training"
    transaction_path = common.paths.resolve_dataset_build_transaction_path(
        "synthetic",
        model_training_data_root=training_root,
    )
    original_interpret = builder_module._interpret_generated_case
    interrupted = False

    def interrupt_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("injected interruption")
        return original_interpret(*args, **kwargs)

    monkeypatch.setattr(builder_module, "_interpret_generated_case", interrupt_once)
    with pytest.raises(RuntimeError, match="injected interruption"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=training_root,
        )

    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    staging_root = Path(transaction["staging_root"])
    assert transaction["phase"] == "building"
    assert staging_root.is_dir()
    assert not (training_root / "raw/synthetic").exists()
    assert not (training_root / "meta/synthetic").exists()

    result = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=training_root,
    )
    assert result["status"] == "complete"
    assert not transaction_path.exists()
    assert not staging_root.exists()


def test_building_publication_transaction_is_discarded_before_retry(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root)
    training_root = tmp_path / "training"
    staging_root = training_root / ".synthetic.dataset-build.interrupted.tmp"
    staging_root.mkdir(parents=True)
    (staging_root / "partial.tmp").write_text("incomplete", encoding="utf-8")
    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)
    common.serialization.atomic_write_json(
        transaction_path,
        builder_module._publication_transaction_record(
            dataset_id="synthetic",
            phase="building",
            staging_root=staging_root,
        ),
    )

    result = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=training_root,
    )

    assert result["status"] == "complete"
    assert result["dataset_path"].is_file()
    assert not transaction_path.exists()
    assert not staging_root.exists()


def test_singleton_manifest_case_object_is_normalized_consistently(
    tmp_path: Path,
) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, processed_dir = _write_generated_batch(
        generated_root,
        case_numbers=(1,),
        timing_count=1,
    )
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"] = manifest["cases"][0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    timing_path = processed_dir / "comsol_solve_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["batch_manifest_sha256"] = _sha256(manifest_path)
    timing_path.write_text(json.dumps(timing), encoding="utf-8")

    result = build_batch_dataset(
        "synthetic",
        generated_data_root=generated_root,
        model_training_data_root=tmp_path / "training",
    )
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    assert payload["sample_ids"] == ["case_0001"]
    assert payload["generated_batch_identity"]["scientific_case_sources"][0]["case_id"] == "case_0001"


@pytest.mark.parametrize("schema_version", [0, 2, 3])
def test_batch_manifest_versions_other_than_one_are_rejected(
    tmp_path: Path,
    schema_version: int,
) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, _processed_dir = _write_generated_batch(generated_root)
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported batch manifest schema"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


def test_private_batch_progress_blocks_dataset_construction(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, _processed_dir = _write_generated_batch(generated_root)
    _write_private_batch_progress(raw_dir)

    with pytest.raises(RuntimeError, match="active or interrupted COMSOL progress"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=tmp_path / "training",
        )


@pytest.mark.parametrize("boundary", ["before_ready", "before_rename"])
def test_progress_appearing_during_build_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, _processed_dir = _write_generated_batch(generated_root)
    training_root = tmp_path / "training"
    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)

    if boundary == "before_ready":
        original_interpret = builder_module._interpret_generated_case
        injected = False

        def inject_after_case(*args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            interpreted = original_interpret(*args, **kwargs)
            if not injected:
                injected = True
                _write_private_batch_progress(raw_dir)
            return interpreted

        monkeypatch.setattr(builder_module, "_interpret_generated_case", inject_after_case)
    else:
        original_write_json = common.serialization.atomic_write_json

        def inject_after_ready(path: Path, payload: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            original_write_json(path, payload, *args, **kwargs)
            if Path(path) == transaction_path and payload.get("phase") == "ready":
                _write_private_batch_progress(raw_dir)

        monkeypatch.setattr(common.serialization, "atomic_write_json", inject_after_ready)

    with pytest.raises(RuntimeError, match="active or interrupted COMSOL progress"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=training_root,
        )

    assert not (training_root / "raw" / "synthetic").exists()
    assert not (training_root / "meta" / "synthetic").exists()
    assert transaction_path.is_file()
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    assert transaction["phase"] == ("building" if boundary == "before_ready" else "ready")
    assert Path(transaction["staging_root"]).is_dir()
    assert not (training_root / "processed").exists()


def test_manifest_change_after_ready_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    _meta, raw_dir, _processed_dir = _write_generated_batch(generated_root)
    training_root = tmp_path / "training"
    transaction_path = common.paths.resolve_dataset_build_transaction_path("synthetic", model_training_data_root=training_root)
    manifest_path = raw_dir / "batch_manifest.json"
    original_write_json = common.serialization.atomic_write_json

    def mutate_after_ready(path: Path, payload: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        original_write_json(path, payload, *args, **kwargs)
        if Path(path) == transaction_path and payload.get("phase") == "ready":
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    monkeypatch.setattr(common.serialization, "atomic_write_json", mutate_after_ready)

    with pytest.raises(RuntimeError, match="Generation manifest changed"):
        build_batch_dataset(
            "synthetic",
            generated_data_root=generated_root,
            model_training_data_root=training_root,
        )

    assert not (training_root / "raw" / "synthetic").exists()
    assert not (training_root / "meta" / "synthetic").exists()
    assert transaction_path.is_file()
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    assert transaction["phase"] == "ready"
    assert Path(transaction["staging_root"]).is_dir()
    assert not (training_root / "processed").exists()


def test_unversioned_payload_is_rejected(steady_task: domain.tasks.spec.TaskSpec) -> None:
    with pytest.raises(ValueError, match="Unsupported dataset schema"):
        datasets.modules.flow.FlowModule(
            {"inputs": torch.zeros((1, 7, 2, 3)), "outputs": torch.zeros((1, 3, 2, 3))},
            task=steady_task,
        )


def test_eda_reads_validated_generated_sources_only(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EDA must interpret generated sources without resolving final datasets."""
    from src import analysis, common

    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root, case_numbers=(2, 1), timing_count=1)

    def reject_training_dataset(*_args: Any, **_kwargs: Any) -> Path:
        raise AssertionError("EDA attempted to resolve a model-training dataset")

    monkeypatch.setattr(common.paths, "resolve_dataset_path", reject_training_dataset)
    frame, logs = analysis.eda.dataframe.generate_eda_dataframe(
        "synthetic",
        task=steady_task,
        generated_data_root=generated_root,
        max_cases=1,
    )

    assert list(frame.index) == ["case_0002"]
    assert list(frame.columns) == [*steady_task.input_names, *steady_task.output_names, "meta"]
    assert frame.attrs["loaded_case_count"] == 1
    assert frame.attrs["available_case_count"] == 2
    assert frame.attrs["generated_batch_identity"]["batch_name"] == "synthetic"
    assert "dataset_identity" not in frame.attrs
    assert frame.iloc[0]["meta"]["parameters"] == {"alpha": 0.2}
    assert any("Generated batch" in message for message in logs)
    assert not (tmp_path / "training").exists()


@pytest.mark.parametrize(("value", "error"), [(0, ValueError), (True, TypeError), (1.5, TypeError)])
def test_eda_rejects_invalid_case_limits(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    value: Any,
    error: type[Exception],
) -> None:
    """EDA validates prefix limits before interpreting generated cases."""
    from src import analysis

    generated_root = tmp_path / "generated"
    _write_generated_batch(generated_root)
    with pytest.raises(error, match="max_cases"):
        analysis.eda.dataframe.generate_eda_dataframe(
            "synthetic",
            task=steady_task,
            generated_data_root=generated_root,
            max_cases=value,
        )
