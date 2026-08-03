"""
===============================================================================
analysis_artifact_timing.py
===============================================================================
Measure and validate COMSOL and neural artifact runtime comparisons.

Responsibilities:
  - Measure synchronized neural forward passes after explicit warmup
  - Validate COMSOL timing snapshots and runtime comparison schemas
  - Build, publish, and load artifact-local runtime sidecars

Design principles:
  - Timing evidence is identity-bound and separate from scientific payload digests
  - Device synchronization and inference dtype are recorded explicitly
  - Unavailable COMSOL evidence remains an explicit validated state

This module does NOT:
  - Select runs, datasets, checkpoints, or artifact memberships
  - Generate Parquet or NPZ scientific payloads
  - Upload runtime evidence or mutate tracking workspaces
===============================================================================
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import common

COMSOL_SOLVE_TIMING_FILENAME = "comsol_solve_timing.json"
RUNTIME_COMPARISON_FILENAME = "runtime_comparison.json"
COMSOL_SOLVE_SCHEMA_KIND = "comsol_solve_timing"
RUNTIME_COMPARISON_SCHEMA_KIND = "comsol_neural_operator_runtime_comparison"
SCHEMA_VERSION = 1
WARMUP_PASSES = 1
MEASUREMENT_CLOCK = "time.perf_counter_ns"
_SHA256_HEX_LENGTH = 64
_SUMMARY_FIELDS = {"count", "mean", "median", "p10", "p90"}


def _is_schema_version(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == SCHEMA_VERSION


def _duration(value: Any, *, label: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real scalar."
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        relation = "positive" if positive else "non-negative"
        msg = f"{label} must be finite and {relation}."
        raise ValueError(msg)
    return result


def _require_sha256(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        msg = f"{label} must be text."
        raise TypeError(msg)
    if allow_empty and not value:
        return value
    if len(value) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in value):
        msg = f"{label} must be a lowercase SHA-256."
        raise ValueError(msg)
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{label} must be non-empty text."
        raise TypeError(msg)
    return value


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    checked = [_duration(value, label="summary duration") for value in values]
    if not checked:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    array = np.asarray(checked, dtype=np.float64)
    return {
        "count": len(checked),
        "mean": float(np.mean(array)),
        "median": float(np.percentile(array, 50.0)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _validate_summary(value: Any, *, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _SUMMARY_FIELDS or dict(value) != dict(expected):
        msg = f"{label} must be derived from the validated per-case values."
        raise ValueError(msg)


def synchronize_device(device: torch.device) -> None:
    """Synchronize one resolved CUDA device; CPU never accesses CUDA APIs."""
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Timing requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward(
    *,
    model: Any,
    normalized_inputs: torch.Tensor,
    device: torch.device,
) -> tuple[Any, float]:
    """Measure one direct batch-size-one model call on the resolved device."""
    if not isinstance(normalized_inputs, torch.Tensor) or normalized_inputs.ndim == 0 or normalized_inputs.shape[0] != 1:
        msg = "Authoritative neural-operator latency requires a size-one input batch."
        raise ValueError(msg)
    model.eval()
    with torch.inference_mode():
        synchronize_device(device)
        started_ns = time.perf_counter_ns()
        prediction = model(normalized_inputs)
        synchronize_device(device)
        elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    return prediction, _duration(elapsed_s, label="neural-operator forward duration")


def warm_up_forward(
    *,
    representative_batch: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: torch.device,
    passes: int = WARMUP_PASSES,
) -> None:
    """Warm up a size-one forward path without publishing its duration."""
    if isinstance(passes, bool) or not isinstance(passes, int) or passes <= 0:
        msg = "Warmup passes must be a positive integer."
        raise ValueError(msg)
    raw_inputs = representative_batch.get("x")
    if not isinstance(raw_inputs, torch.Tensor) or raw_inputs.ndim == 0 or raw_inputs.shape[0] < 1:
        msg = "Forward warmup requires a non-empty tensor batch under key 'x'."
        raise TypeError(msg)
    model.eval()
    with torch.inference_mode():
        normalized_inputs = processor.in_normalizer.transform(raw_inputs[:1].to(device))
        for _ in range(passes):
            synchronize_device(device)
            model(normalized_inputs)
            synchronize_device(device)


def model_inference_dtype(model: Any) -> str:
    """Return the first model-parameter dtype without allocating device state."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown"
    raw_parameters = parameters()
    if not isinstance(raw_parameters, Iterable):
        return "unknown"
    try:
        parameter = next(iter(raw_parameters))
    except StopIteration:
        return "unknown"
    return str(parameter.dtype).removeprefix("torch.")


def neural_runtime_metadata(
    *,
    device_metadata: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    """Extend the shared resolved-device record with compact benchmark context."""
    return {
        **dict(device_metadata),
        "hostname": platform.node() or "unknown",
        "platform": platform.platform() or "unknown",
        "processor": platform.processor() or platform.machine() or "unknown",
        "python_version": platform.python_version() or sys.version.split()[0],
        "inference_dtype": model_inference_dtype(model),
        "torch_num_threads": torch.get_num_threads(),
    }


def _validate_neural_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        msg = "neural_runtime must be a mapping."
        raise TypeError(msg)
    runtime = dict(value)
    base = {
        "requested_policy",
        "resolved_device",
        "device_type",
        "pytorch_version",
        "hostname",
        "platform",
        "processor",
        "python_version",
        "inference_dtype",
        "torch_num_threads",
    }
    cuda = {"cuda_index", "cuda_device_name", "cuda_runtime_version"}
    device_type = runtime.get("device_type")
    expected = base | (cuda if device_type == "cuda" else set())
    if device_type not in {"cpu", "cuda"} or set(runtime) != expected:
        msg = "neural_runtime has invalid fields for its resolved device type."
        raise ValueError(msg)
    if runtime.get("requested_policy") not in {"auto", "cpu", "cuda"}:
        msg = "neural_runtime.requested_policy is invalid."
        raise ValueError(msg)
    for field in (
        "resolved_device",
        "pytorch_version",
        "hostname",
        "platform",
        "processor",
        "python_version",
        "inference_dtype",
    ):
        _require_text(runtime.get(field), label=f"neural_runtime.{field}")
    threads = runtime.get("torch_num_threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        msg = "neural_runtime.torch_num_threads must be a positive integer."
        raise TypeError(msg)
    if device_type == "cuda":
        index = runtime.get("cuda_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            msg = "neural_runtime.cuda_index must be a non-negative integer."
            raise TypeError(msg)
        _require_text(runtime.get("cuda_device_name"), label="neural_runtime.cuda_device_name")
        _require_text(runtime.get("cuda_runtime_version"), label="neural_runtime.cuda_runtime_version")
    return runtime


def validate_comsol_solve_timing(value: Any) -> dict[str, Any]:
    """Validate the compact MATLAB COMSOL solve-timing sidecar."""
    if not isinstance(value, Mapping):
        msg = "COMSOL solve timing must be a mapping."
        raise TypeError(msg)
    payload = dict(value)
    required = {
        "schema_kind",
        "schema_version",
        "batch_name",
        "batch_manifest_sha256",
        "runtime",
        "cases",
        "aggregates",
    }
    if set(payload) != required or payload.get("schema_kind") != COMSOL_SOLVE_SCHEMA_KIND or not _is_schema_version(payload.get("schema_version")):
        msg = "COMSOL solve timing has an invalid schema."
        raise ValueError(msg)
    _require_text(payload.get("batch_name"), label="COMSOL batch_name")
    _require_sha256(payload.get("batch_manifest_sha256"), label="COMSOL batch_manifest_sha256", allow_empty=True)
    runtime = payload.get("runtime")
    runtime_fields = {"matlab_version", "comsol_version", "os", "hostname", "processor", "case_execution"}
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_fields:
        msg = "COMSOL runtime provenance has invalid fields."
        raise ValueError(msg)
    for field in runtime_fields:
        _require_text(runtime.get(field), label=f"COMSOL runtime.{field}")
    if runtime.get("case_execution") != "sequential":
        msg = "COMSOL runtime.case_execution must be sequential."
        raise ValueError(msg)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        msg = "COMSOL solve timing cases must be a JSON array."
        raise TypeError(msg)
    cases: list[dict[str, Any]] = []
    case_ids: list[str] = []
    for position, value_case in enumerate(raw_cases):
        if not isinstance(value_case, Mapping) or set(value_case) != {"case_id", "comsol_solve_s"}:
            msg = f"COMSOL solve timing case {position} has invalid fields."
            raise ValueError(msg)
        case_id = _require_text(value_case.get("case_id"), label=f"COMSOL case {position} case_id")
        duration = _duration(value_case.get("comsol_solve_s"), label=f"COMSOL case {position} solve duration")
        case_ids.append(case_id)
        cases.append({"case_id": case_id, "comsol_solve_s": duration})
    if len(case_ids) != len(set(case_ids)):
        msg = "COMSOL solve timing case IDs must be unique."
        raise ValueError(msg)
    summary = _summary([case["comsol_solve_s"] for case in cases])

    def matlab_optional(value: float | None) -> float | list[Any]:
        return [] if value is None else value

    expected_aggregates = {
        "measured_case_count": summary["count"],
        "mean_s": matlab_optional(summary["mean"]),
        "median_s": matlab_optional(summary["median"]),
        "p10_s": matlab_optional(summary["p10"]),
        "p90_s": matlab_optional(summary["p90"]),
    }
    aggregates = payload.get("aggregates")
    if not isinstance(aggregates, Mapping) or set(aggregates) != set(expected_aggregates):
        msg = "COMSOL solve aggregates must be derived from valid case records."
        raise ValueError(msg)
    if aggregates["measured_case_count"] != expected_aggregates["measured_case_count"]:
        msg = "COMSOL solve aggregates must be derived from valid case records."
        raise ValueError(msg)
    for field in ("mean_s", "median_s", "p10_s", "p90_s"):
        actual = aggregates[field]
        expected_value = expected_aggregates[field]
        if isinstance(expected_value, list):
            valid = actual == []
        else:
            valid = (
                isinstance(expected_value, Real)
                and not isinstance(actual, bool)
                and isinstance(actual, Real)
                and math.isfinite(float(actual))
                and math.isclose(float(actual), float(expected_value), rel_tol=1e-12, abs_tol=1e-12)
            )
        if not valid:
            msg = "COMSOL solve aggregates must be derived from valid case records."
            raise ValueError(msg)
    return payload


def _validate_identity(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        msg = f"{label} has invalid fields."
        raise ValueError(msg)
    identity = dict(value)
    for field in fields:
        _require_text(identity.get(field), label=f"{label}.{field}")
    return identity


def build_runtime_comparison(
    *,
    split_role: str,
    dataset_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    neural_runtime: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    comsol_timing: Mapping[str, Any] | None,
    batch_manifest_sha256: str | None,
    unavailable_reason: str | None,
    warmup_passes: int = WARMUP_PASSES,
) -> dict[str, Any]:
    """Build one direct, case-matched solver-to-forward comparison."""
    if split_role not in {"eval", "ood"}:
        msg = "Runtime comparison split_role must be eval or ood."
        raise ValueError(msg)
    if isinstance(warmup_passes, bool) or not isinstance(warmup_passes, int) or warmup_passes <= 0:
        msg = "Runtime comparison warmup_passes must be positive."
        raise ValueError(msg)
    neural_cases: list[dict[str, Any]] = []
    for position, raw_case in enumerate(cases):
        if not isinstance(raw_case, Mapping) or set(raw_case) != {"case_id", "source_index", "neural_operator_forward_s"}:
            msg = f"Neural timing case {position} has invalid fields."
            raise ValueError(msg)
        case_id = _require_text(raw_case.get("case_id"), label=f"Neural timing case {position} case_id")
        source_index = raw_case.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            msg = f"Neural timing case {position} source_index must be non-negative."
            raise TypeError(msg)
        neural_cases.append(
            {
                "case_id": case_id,
                "source_index": source_index,
                "neural_operator_forward_s": _duration(
                    raw_case.get("neural_operator_forward_s"),
                    label=f"Neural timing case {position} forward duration",
                ),
            }
        )
    if not neural_cases:
        msg = "Runtime comparison requires at least one measured neural case."
        raise ValueError(msg)
    if len({case["case_id"] for case in neural_cases}) != len(neural_cases) or len({case["source_index"] for case in neural_cases}) != len(
        neural_cases
    ):
        msg = "Neural timing case IDs and source indices must be unique."
        raise ValueError(msg)

    comsol_by_id: dict[str, float] = {}
    if comsol_timing is None:
        if batch_manifest_sha256 is not None or not isinstance(unavailable_reason, str) or not unavailable_reason:
            msg = "Unavailable comparison requires only a non-empty reason."
            raise ValueError(msg)
        comparison: dict[str, Any] = {"status": "unavailable", "reason": unavailable_reason}
    else:
        if unavailable_reason is not None or batch_manifest_sha256 is None:
            msg = "Available comparison requires a manifest digest and no unavailable reason."
            raise ValueError(msg)
        comsol = validate_comsol_solve_timing(comsol_timing)
        expected_digest = _require_sha256(batch_manifest_sha256, label="batch_manifest_sha256")
        if comsol["batch_manifest_sha256"] != expected_digest:
            msg = "COMSOL solve timing does not bind the dataset-authoritative batch manifest."
            raise ValueError(msg)
        comsol_by_id = {case["case_id"]: float(case["comsol_solve_s"]) for case in comsol["cases"]}
        comparison = {
            "status": "available",
            "batch_name": comsol["batch_name"],
            "batch_manifest_sha256": expected_digest,
            "runtime": dict(comsol["runtime"]),
        }

    records: list[dict[str, Any]] = []
    matched_comsol: list[float] = []
    speedups: list[float] = []
    forward_values: list[float] = []
    for case in neural_cases:
        forward_s = float(case["neural_operator_forward_s"])
        comsol_s = comsol_by_id.get(str(case["case_id"]))
        forward_values.append(forward_s)
        if comsol_s is None:
            speedup = None
        else:
            speedup = comsol_s / forward_s
            matched_comsol.append(comsol_s)
            speedups.append(speedup)
        records.append({**case, "comsol_solve_s": comsol_s, "speedup": speedup})

    payload = {
        "schema_kind": RUNTIME_COMPARISON_SCHEMA_KIND,
        "schema_version": SCHEMA_VERSION,
        "split_role": split_role,
        "dataset_identity": dict(dataset_identity),
        "model_identity": dict(model_identity),
        "neural_runtime": dict(neural_runtime),
        "measurement": {
            "clock": MEASUREMENT_CLOCK,
            "batch_size": 1,
            "warmup_passes": warmup_passes,
            "cuda_synchronized": neural_runtime.get("device_type") == "cuda",
        },
        "comparison": comparison,
        "cases": records,
        "aggregates": {
            "neural_operator_forward_s": _summary(forward_values),
            "comsol_solve_s": _summary(matched_comsol),
            "speedup": _summary(speedups),
        },
    }
    return validate_runtime_comparison(payload)


def validate_runtime_comparison(value: Any) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Validate identities, direct measurements, matches, and aggregates."""
    if not isinstance(value, Mapping):
        msg = "Runtime comparison must be a mapping."
        raise TypeError(msg)
    payload = dict(value)
    required = {
        "schema_kind",
        "schema_version",
        "split_role",
        "dataset_identity",
        "model_identity",
        "neural_runtime",
        "measurement",
        "comparison",
        "cases",
        "aggregates",
    }
    if (
        set(payload) != required
        or payload.get("schema_kind") != RUNTIME_COMPARISON_SCHEMA_KIND
        or not _is_schema_version(payload.get("schema_version"))
    ):
        msg = "Runtime comparison has an invalid schema."
        raise ValueError(msg)
    if payload.get("split_role") not in {"eval", "ood"}:
        msg = "Runtime comparison split_role must be eval or ood."
        raise ValueError(msg)
    _validate_identity(
        payload.get("dataset_identity"),
        fields={
            "name",
            "fingerprint",
            "task_contract_digest",
            "saved_membership_digest",
            "effective_ordered_source_indices_sha256",
        },
        label="dataset_identity",
    )
    _validate_identity(
        payload.get("model_identity"),
        fields={"run_name", "effective_config_digest", "best_checkpoint_sha256"},
        label="model_identity",
    )
    runtime = _validate_neural_runtime(payload.get("neural_runtime"))
    measurement = payload.get("measurement")
    if not isinstance(measurement, Mapping) or set(measurement) != {"clock", "batch_size", "warmup_passes", "cuda_synchronized"}:
        msg = "Runtime comparison measurement has invalid fields."
        raise ValueError(msg)
    if measurement.get("clock") != MEASUREMENT_CLOCK or measurement.get("batch_size") != 1:
        msg = "Runtime comparison must use perf_counter_ns and direct batch size one."
        raise ValueError(msg)
    passes = measurement.get("warmup_passes")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes <= 0:
        msg = "Runtime comparison warmup passes must be positive."
        raise TypeError(msg)
    if measurement.get("cuda_synchronized") is not (runtime["device_type"] == "cuda"):
        msg = "Runtime comparison CUDA synchronization metadata is inconsistent."
        raise ValueError(msg)

    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping):
        msg = "Runtime comparison descriptor must be a mapping."
        raise TypeError(msg)
    status = comparison.get("status")
    if status == "unavailable":
        if set(comparison) != {"status", "reason"}:
            msg = "Unavailable comparison descriptor has invalid fields."
            raise ValueError(msg)
        _require_text(comparison.get("reason"), label="comparison.reason")
    elif status == "available":
        if set(comparison) != {"status", "batch_name", "batch_manifest_sha256", "runtime"}:
            msg = "Available comparison descriptor has invalid fields."
            raise ValueError(msg)
        _require_text(comparison.get("batch_name"), label="comparison.batch_name")
        _require_sha256(comparison.get("batch_manifest_sha256"), label="comparison.batch_manifest_sha256")
        comsol_runtime = comparison.get("runtime")
        comsol_runtime_fields = {
            "matlab_version",
            "comsol_version",
            "os",
            "hostname",
            "processor",
            "case_execution",
        }
        if not isinstance(comsol_runtime, Mapping) or set(comsol_runtime) != comsol_runtime_fields:
            msg = "Available comparison COMSOL runtime has invalid fields."
            raise ValueError(msg)
        for field in comsol_runtime_fields:
            _require_text(comsol_runtime.get(field), label=f"comparison.runtime.{field}")
        if comsol_runtime.get("case_execution") != "sequential":
            msg = "Available comparison COMSOL runtime must describe sequential cases."
            raise ValueError(msg)
    else:
        msg = "Runtime comparison status must be available or unavailable."
        raise ValueError(msg)

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        msg = "Runtime comparison cases must be a non-empty JSON array."
        raise ValueError(msg)
    case_ids: list[str] = []
    source_indices: list[int] = []
    forwards: list[float] = []
    solves: list[float] = []
    speedups: list[float] = []
    for position, raw_case in enumerate(raw_cases):
        required_case = {"case_id", "source_index", "neural_operator_forward_s", "comsol_solve_s", "speedup"}
        if not isinstance(raw_case, Mapping) or set(raw_case) != required_case:
            msg = f"Runtime comparison case {position} has invalid fields."
            raise ValueError(msg)
        case_id = _require_text(raw_case.get("case_id"), label=f"cases[{position}].case_id")
        source_index = raw_case.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            msg = f"cases[{position}].source_index must be non-negative."
            raise TypeError(msg)
        forward_s = _duration(raw_case.get("neural_operator_forward_s"), label=f"cases[{position}].neural_operator_forward_s")
        solve_value = raw_case.get("comsol_solve_s")
        speedup_value = raw_case.get("speedup")
        if solve_value is None or speedup_value is None:
            if (solve_value is None) != (speedup_value is None):
                msg = f"Runtime comparison case {position} has an incomplete match."
                raise ValueError(msg)
            solve_s = None
            speedup = None
        else:
            if status != "available":
                msg = f"Unavailable comparison cannot contain matched case {position}."
                raise ValueError(msg)
            solve_s = _duration(solve_value, label=f"cases[{position}].comsol_solve_s")
            speedup = _duration(speedup_value, label=f"cases[{position}].speedup")
            if not math.isclose(speedup, solve_s / forward_s, rel_tol=1e-12, abs_tol=0.0):
                msg = f"Runtime comparison case {position} speedup is invalid."
                raise ValueError(msg)
            solves.append(solve_s)
            speedups.append(speedup)
        case_ids.append(case_id)
        source_indices.append(source_index)
        forwards.append(forward_s)
    if len(case_ids) != len(set(case_ids)) or len(source_indices) != len(set(source_indices)):
        msg = "Runtime comparison case IDs and source indices must be unique."
        raise ValueError(msg)
    aggregates = payload.get("aggregates")
    if not isinstance(aggregates, Mapping) or set(aggregates) != {"neural_operator_forward_s", "comsol_solve_s", "speedup"}:
        msg = "Runtime comparison aggregates have invalid fields."
        raise ValueError(msg)
    _validate_summary(aggregates["neural_operator_forward_s"], expected=_summary(forwards), label="forward aggregate")
    _validate_summary(aggregates["comsol_solve_s"], expected=_summary(solves), label="COMSOL aggregate")
    _validate_summary(aggregates["speedup"], expected=_summary(speedups), label="speedup aggregate")
    return payload


def runtime_comparison_path(save_root: str | Path) -> Path:
    """Return the operational sidecar path for one artifact target."""
    return Path(save_root) / RUNTIME_COMPARISON_FILENAME


def write_runtime_comparison(save_root: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish one validated operational comparison sidecar."""
    return common.serialization.atomic_write_json(
        runtime_comparison_path(save_root),
        validate_runtime_comparison(payload),
    )


def load_runtime_comparison(save_root: str | Path) -> dict[str, Any]:
    """Load one strictly validated operational comparison sidecar."""
    path = runtime_comparison_path(save_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Runtime comparison sidecar is missing or unreadable: {path}: {error}"
        raise RuntimeError(msg) from error
    return validate_runtime_comparison(payload)
