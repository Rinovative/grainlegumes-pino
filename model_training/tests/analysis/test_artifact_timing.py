# ruff: noqa: S101, D103, EM101, PLR2004, SLF001, TRY003
"""Verify direct timing, strict persistence, matching, and cache independence."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from src import analysis
from src.datasets.dataset_metadata import DatasetMetadata, validate_comsol_timing_snapshot
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


def test_comsol_timing_resolution_uses_only_validated_training_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime comparison must not resolve or inspect generation storage."""
    digest = "a" * 64
    comsol_payload = _comsol_payload(digest=digest)
    package = DatasetMetadata(
        directory=tmp_path,
        provenance={
            "source_batch": {"batch_manifest_sha256": digest},
            "timing": {"status": "partial", "measured_case_count": 2, "intended_case_count": 3},
        },
        inventory={},
        source_manifest={},
        timing=comsol_payload,
    )
    request = analysis.artifact_service.ArtifactRequest(
        provenance={},
        source_indices=(),
        dataset_metadata=package,
    )

    generated_root = tmp_path / "must-not-be-opened"
    monkeypatch.setenv("GENERATED_DATA_ROOT", str(generated_root))
    payload, resolved_digest, reason = analysis.artifact_service._resolve_comsol_timing(request)
    assert payload == comsol_payload
    assert resolved_digest == digest
    assert reason is None
    assert not generated_root.exists()


def test_missing_training_timing_snapshot_is_nonfatal() -> None:
    package = DatasetMetadata(
        directory=Path(),
        provenance={
            "source_batch": {"batch_manifest_sha256": "a" * 64},
            "timing": {"status": "missing", "measured_case_count": 0, "intended_case_count": 3},
        },
        inventory={},
        source_manifest={},
        timing=None,
    )
    request = analysis.artifact_service.ArtifactRequest(
        provenance={},
        source_indices=(),
        dataset_metadata=package,
    )
    payload, digest, reason = analysis.artifact_service._resolve_comsol_timing(request)
    assert payload is None
    assert digest is None
    assert reason == "validated model-training COMSOL timing snapshot is missing"


def test_validated_zero_case_timing_snapshot_is_unavailable() -> None:
    comsol_payload = _comsol_payload()
    comsol_payload["cases"] = []
    comsol_payload["aggregates"] = {
        "measured_case_count": 0,
        "mean_s": [],
        "median_s": [],
        "p10_s": [],
        "p90_s": [],
    }
    package = DatasetMetadata(
        directory=Path(),
        provenance={
            "source_batch": {"batch_manifest_sha256": "a" * 64},
            "timing": {"status": "missing", "measured_case_count": 0, "intended_case_count": 3},
        },
        inventory={},
        source_manifest={},
        timing=comsol_payload,
    )
    request = analysis.artifact_service.ArtifactRequest(
        provenance={},
        source_indices=(),
        dataset_metadata=package,
    )

    payload, digest, reason = analysis.artifact_service._resolve_comsol_timing(request)

    assert payload is None
    assert digest is None
    assert reason == "validated model-training COMSOL timing snapshot is missing"


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


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_timing_payloads_require_integer_schema_version_one(schema_version: object) -> None:
    """Reject alternate representations in both persisted timing payloads."""
    comsol = _comsol_payload()
    comsol["schema_version"] = schema_version
    with pytest.raises(ValueError, match="invalid schema"):
        analysis.timing.validate_comsol_solve_timing(comsol)

    comparison = _comparison()
    comparison["schema_version"] = schema_version
    with pytest.raises(ValueError, match="invalid schema"):
        analysis.timing.validate_runtime_comparison(comparison)


def test_metadata_timing_normalizes_one_current_matlab_case_object() -> None:
    """Normalize MATLAB's scalar timing record to the maintained in-memory list."""
    payload = _comsol_payload()
    payload["cases"] = payload["cases"][0]
    payload["aggregates"] = {
        "measured_case_count": 1,
        "mean_s": 20.0,
        "median_s": 20.0,
        "p10_s": 20.0,
        "p90_s": 20.0,
    }

    validated = validate_comsol_timing_snapshot(
        payload,
        batch_name="batch-a",
        manifest_sha256="a" * 64,
        intended_case_ids=["case_0001"],
    )

    assert validated["cases"] == [{"case_id": "case_0001", "comsol_solve_s": 20.0}]


@pytest.mark.parametrize("measured_case_count", [True, 1.0], ids=("boolean-one", "floating-one"))
def test_metadata_timing_aggregate_count_requires_an_integer(measured_case_count: object) -> None:
    """Reject numeric values that compare equal to the one-case aggregate count."""
    payload = _comsol_payload()
    payload["cases"] = [payload["cases"][0]]
    payload["aggregates"] = {
        "measured_case_count": measured_case_count,
        "mean_s": 20.0,
        "median_s": 20.0,
        "p10_s": 20.0,
        "p90_s": 20.0,
    }

    with pytest.raises(ValueError, match="measured_case_count"):
        validate_comsol_timing_snapshot(
            payload,
            batch_name="batch-a",
            manifest_sha256="a" * 64,
            intended_case_ids=["case_0001"],
        )


def test_comsol_timing_accepts_nonlexicographic_manifest_order() -> None:
    payload = _comsol_payload()
    payload["cases"] = list(reversed(payload["cases"]))
    assert analysis.timing.validate_comsol_solve_timing(payload) == payload


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


def test_comsol_aggregates_allow_only_machine_roundoff() -> None:
    payload = _comsol_payload()
    payload["aggregates"]["mean_s"] += 5e-15
    assert analysis.timing.validate_comsol_solve_timing(payload) == payload
    payload["aggregates"]["mean_s"] += 1e-3
    with pytest.raises(ValueError, match="derived from valid case records"):
        analysis.timing.validate_comsol_solve_timing(payload)


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
