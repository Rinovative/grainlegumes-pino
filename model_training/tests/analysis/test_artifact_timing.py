# ruff: noqa: S101, D103, EM101, PLR2004, SLF001, TC003, TRY003
"""Verify direct timing, strict persistence, matching, and cache independence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from src import analysis, common
from torch import nn


class _IdentityNormalizer:
    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _RecordingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.inference_modes: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return value[:, :1] * self.weight


def _neural_runtime() -> dict[str, Any]:
    return {
        "requested_policy": "cpu",
        "resolved_device": "cpu",
        "device_type": "cpu",
        "pytorch_version": str(torch.__version__),
        "hostname": "test-host",
        "platform": "test-platform",
        "processor": "test-cpu",
        "python_version": "3.11",
        "inference_dtype": "float32",
        "torch_num_threads": 1,
    }


def _dataset_identity() -> dict[str, str]:
    return {
        "name": "batch-a",
        "fingerprint": "dataset-fingerprint",
        "task_contract_digest": "task-digest",
        "saved_membership_digest": "membership-digest",
        "effective_ordered_source_indices_sha256": "indices-digest",
    }


def _model_identity() -> dict[str, str]:
    return {
        "run_name": "run-a",
        "effective_config_digest": "config-digest",
        "best_checkpoint_sha256": "checkpoint-digest",
    }


def _neural_cases() -> list[dict[str, Any]]:
    return [
        {"case_id": "case_0001", "source_index": 0, "neural_operator_forward_s": 0.1},
        {"case_id": "case_0002", "source_index": 2, "neural_operator_forward_s": 0.2},
    ]


def _comsol_payload(*, digest: str = "a" * 64) -> dict[str, Any]:
    return {
        "schema_kind": analysis.timing.COMSOL_SOLVE_SCHEMA_KIND,
        "schema_version": 1,
        "batch_name": "batch-a",
        "batch_manifest_sha256": digest,
        "runtime": {
            "matlab_version": "test",
            "comsol_version": "6.4",
            "os": "test-os",
            "hostname": "comsol-host",
            "processor": "test-cpu",
            "case_execution": "sequential",
        },
        "cases": [
            {"case_id": "case_0001", "comsol_solve_s": 20.0},
            {"case_id": "case_0003", "comsol_solve_s": 30.0},
        ],
        "aggregates": {
            "measured_case_count": 2,
            "mean_s": 25.0,
            "median_s": 25.0,
            "p10_s": 21.0,
            "p90_s": 29.0,
        },
    }


def _comparison(*, comsol: bool = True) -> dict[str, Any]:
    return analysis.timing.build_runtime_comparison(
        split_role="eval",
        dataset_identity=_dataset_identity(),
        model_identity=_model_identity(),
        neural_runtime=_neural_runtime(),
        cases=_neural_cases(),
        comsol_timing=_comsol_payload() if comsol else None,
        batch_manifest_sha256="a" * 64 if comsol else None,
        unavailable_reason=None if comsol else "COMSOL timing is unavailable",
    )


def test_comsol_solve_timing_path_uses_processed_batch_stage(tmp_path: Path) -> None:
    expected = tmp_path / "processed" / "batch-a" / analysis.timing.COMSOL_SOLVE_TIMING_FILENAME
    assert analysis.timing.COMSOL_SOLVE_TIMING_STAGE == "processed"
    assert analysis.timing.comsol_solve_timing_path("batch-a", generated_data_root=tmp_path) == expected


def test_comsol_timing_resolution_ignores_legacy_raw_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "batch-a"
    processed_dir = generated_root / "processed" / "batch-a"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    monkeypatch.setenv("GENERATED_DATA_ROOT", str(generated_root))

    manifest = {"batch_name": "batch-a", "status": "complete"}
    manifest_path = raw_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    manifest_digest = common.serialization.file_sha256(manifest_path)
    comsol_payload = _comsol_payload(digest=manifest_digest)
    legacy_path = raw_dir / analysis.timing.COMSOL_SOLVE_TIMING_FILENAME
    legacy_path.write_text(json.dumps(comsol_payload) + "\n", encoding="utf-8")
    request = analysis.artifact_service.ArtifactRequest(
        provenance={},
        source_indices=(),
        source_batch_manifest=manifest,
    )

    payload, digest, reason = analysis.artifact_service._resolve_comsol_timing(request)
    assert payload is None
    assert digest is None
    assert reason == "authoritative raw COMSOL manifest or processed solve timing is unavailable"

    processed_path = analysis.timing.comsol_solve_timing_path("batch-a")
    processed_path.write_text(json.dumps(comsol_payload) + "\n", encoding="utf-8")
    payload, digest, reason = analysis.artifact_service._resolve_comsol_timing(request)
    assert payload == comsol_payload
    assert digest == manifest_digest
    assert reason is None


def test_cpu_forward_is_direct_inference_mode_and_never_uses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CPU timing accessed CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    model = _RecordingProjection()
    prediction, duration = analysis.timing.measure_forward(
        model=model,
        normalized_inputs=torch.ones(1, 2, 2, 2),
        device=torch.device("cpu"),
    )
    assert prediction.shape == (1, 1, 2, 2)
    assert duration > 0.0
    assert model.training is False
    assert model.inference_modes == [True]


def test_warmup_is_separate_from_authoritative_measurement() -> None:
    model = _RecordingProjection()
    processor = SimpleNamespace(in_normalizer=_IdentityNormalizer())
    batch = {"x": torch.ones(4, 2, 2, 2)}
    analysis.timing.warm_up_forward(
        representative_batch=batch,
        model=model,
        processor=processor,
        device=torch.device("cpu"),
        passes=1,
    )
    assert model.inference_modes == [True]
    _prediction, measured_s = analysis.timing.measure_forward(
        model=model,
        normalized_inputs=batch["x"][:1],
        device=torch.device("cpu"),
    )
    assert measured_s > 0.0
    assert model.inference_modes == [True, True]


def test_cuda_synchronizes_exact_resolved_device_around_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.device] = []

    def record(device: torch.device) -> None:
        calls.append(device)

    monkeypatch.setattr(torch.cuda, "synchronize", record)
    device = torch.device("cuda:2")
    analysis.timing.measure_forward(
        model=_RecordingProjection(),
        normalized_inputs=torch.ones(1, 2, 2, 2),
        device=device,
    )
    assert calls == [device, device]


def test_case_matching_uses_only_identical_ids_and_primary_speedup() -> None:
    payload = _comparison()
    first, second = payload["cases"]
    assert first["case_id"] == "case_0001"
    assert first["comsol_solve_s"] == 20.0
    assert first["speedup"] == 200.0
    assert second["case_id"] == "case_0002"
    assert second["comsol_solve_s"] is None
    assert second["speedup"] is None
    assert payload["aggregates"]["neural_operator_forward_s"]["count"] == 2
    assert payload["aggregates"]["comsol_solve_s"]["count"] == 1
    assert payload["aggregates"]["speedup"]["median"] == 200.0
    assert payload["comparison"]["status"] == "available"


def test_missing_comsol_timing_retains_neural_measurements_without_fabrication() -> None:
    payload = _comparison(comsol=False)
    assert payload["comparison"] == {"status": "unavailable", "reason": "COMSOL timing is unavailable"}
    assert all(case["comsol_solve_s"] is None and case["speedup"] is None for case in payload["cases"])
    assert payload["aggregates"]["neural_operator_forward_s"]["count"] == 2
    assert payload["aggregates"]["speedup"]["count"] == 0


def test_manifest_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch manifest"):
        analysis.timing.build_runtime_comparison(
            split_role="eval",
            dataset_identity=_dataset_identity(),
            model_identity=_model_identity(),
            neural_runtime=_neural_runtime(),
            cases=_neural_cases(),
            comsol_timing=_comsol_payload(digest="b" * 64),
            batch_manifest_sha256="a" * 64,
            unavailable_reason=None,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["cases"][0].__setitem__("neural_operator_forward_s", 0.0),
        lambda payload: payload["cases"][0].__setitem__("neural_operator_forward_s", -1.0),
        lambda payload: payload["cases"][0].__setitem__("neural_operator_forward_s", float("inf")),
        lambda payload: payload["cases"][0].__setitem__("speedup", 99.0),
        lambda payload: payload["cases"].append(copy.deepcopy(payload["cases"][0])),
        lambda payload: payload["cases"][0].pop("source_index"),
    ],
)
def test_runtime_comparison_rejects_invalid_or_duplicate_cases(mutation: Any) -> None:
    payload = _comparison()
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        analysis.timing.validate_runtime_comparison(payload)


def test_empty_comsol_sidecar_uses_matlab_empty_aggregates() -> None:
    payload = _comsol_payload()
    payload["cases"] = []
    payload["aggregates"] = {
        "measured_case_count": 0,
        "mean_s": [],
        "median_s": [],
        "p10_s": [],
        "p90_s": [],
    }
    assert analysis.timing.validate_comsol_solve_timing(payload) == payload


def test_comsol_timing_rejects_zero_nonfinite_malformed_and_duplicate_records() -> None:
    for value in (0.0, -1.0, float("inf")):
        payload = _comsol_payload()
        payload["cases"][0]["comsol_solve_s"] = value
        with pytest.raises((TypeError, ValueError)):
            analysis.timing.validate_comsol_solve_timing(payload)
    malformed = _comsol_payload()
    malformed["cases"][0].pop("case_id")
    with pytest.raises(ValueError, match="invalid fields"):
        analysis.timing.validate_comsol_solve_timing(malformed)
    duplicate = _comsol_payload()
    duplicate["cases"][1]["case_id"] = "case_0001"
    with pytest.raises(ValueError, match="unique"):
        analysis.timing.validate_comsol_solve_timing(duplicate)


def test_atomic_round_trip_and_scientific_manifest_exclusion(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    npz_root = artifact_root / "npz"
    npz_root.mkdir(parents=True)
    (artifact_root / "cases.parquet").write_bytes(b"scientific-table")
    (npz_root / "case_0001.npz").write_bytes(b"scientific-case")
    before = analysis.artifacts.artifact_output_manifest(artifact_root)
    path = analysis.timing.write_runtime_comparison(artifact_root, _comparison())
    assert path.name == analysis.timing.RUNTIME_COMPARISON_FILENAME
    assert not list(artifact_root.glob(".*.tmp"))
    assert analysis.timing.load_runtime_comparison(artifact_root) == _comparison()
    assert analysis.artifacts.artifact_output_manifest(artifact_root) == before
