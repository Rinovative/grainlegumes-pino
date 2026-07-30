"""Validate self-contained model-training metadata snapshots."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src import common
from src.datasets.dataset_identity import TRAINING_DATASET_SCHEMA_VERSION

if TYPE_CHECKING:
    from src.datasets.dataset_identity import DatasetIdentity

PROVENANCE_FILENAME = "dataset_provenance.json"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
SOURCE_SAMPLE_CSV_FILENAME = "source_sample.csv"
SOURCE_SAMPLE_JSON_FILENAME = "source_sample.json"
COMSOL_TIMING_FILENAME = "comsol_solve_timing.json"
INVENTORY_FILENAME = "metadata_inventory.json"
PROVENANCE_SCHEMA_KIND = "training_dataset_provenance"
INVENTORY_SCHEMA_KIND = "training_dataset_metadata_inventory"
METADATA_SCHEMA_VERSION = 1
SOURCE_MANIFEST_SCHEMA_KIND = "comsol_batch_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_SPATIAL_DIMENSIONS = 2
_MAX_EXACT_MANIFEST_INTEGER = 2**53
_MAX_RANDOM_SEED = 2**32 - 1
_CASE_ID_PATTERN = re.compile(r"case_[0-9]{4,}")
_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "batch_name",
        "status",
        "configuration",
        "field_schema",
        "intended_case_ids",
        "cases",
    }
)
_SOURCE_MANIFEST_CONFIGURATION_KEYS = frozenset(
    {
        "method",
        "variation",
        "N",
        "seed",
        "Lx",
        "Ly",
        "res",
        "save_model",
        "sample_sha256",
        "template_name",
        "template_sha256",
    }
)
_SOURCE_MANIFEST_RECORD_KEYS = frozenset({"case_id", "status", "stage", "message", "files"})
_SOURCE_MANIFEST_FILE_KEYS = frozenset({"raw_csv_sha256", "raw_json_sha256", "solution_csv_sha256", "solution_model_sha256"})
_SOURCE_MANIFEST_FIELD_SCHEMA = {
    "input_columns": ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"],
    "solution_columns": [
        "x",
        "y",
        "kappaxx",
        "kappayx",
        "kappaxy",
        "kappayy",
        "eps",
        "p_bc",
        "p",
        "u",
        "v",
        "U",
    ],
}
_REQUIRED_SNAPSHOT_FILES = frozenset(
    {
        PROVENANCE_FILENAME,
        SOURCE_MANIFEST_FILENAME,
        SOURCE_SAMPLE_CSV_FILENAME,
        SOURCE_SAMPLE_JSON_FILENAME,
    }
)
_ALLOWED_SNAPSHOT_FILES = _REQUIRED_SNAPSHOT_FILES | {COMSOL_TIMING_FILENAME, INVENTORY_FILENAME}


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Validated metadata package bound to one final training dataset."""

    directory: Path
    provenance: dict[str, Any]
    inventory: dict[str, Any]
    source_manifest: dict[str, Any]
    timing: dict[str, Any] | None


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not load {label}: {path}"
        raise ValueError(msg) from error
    if not isinstance(value, dict):
        msg = f"{label} must contain a JSON object: {path}"
        raise TypeError(msg)
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str] | frozenset[str], *, label: str) -> None:
    missing = sorted(set(expected).difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        msg = f"{label} keys do not match: missing={missing}; unexpected={unexpected}."
        raise ValueError(msg)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        msg = f"{label} must be a lowercase SHA-256 digest."
        raise ValueError(msg)
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{label} must be a non-negative integer."
        raise ValueError(msg)
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{label} must be a positive integer."
        raise ValueError(msg)
    return value


def _require_spatial_shape(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != _SPATIAL_DIMENSIONS:
        msg = f"{label} must contain exactly {_SPATIAL_DIMENSIONS} dimensions."
        raise ValueError(msg)
    return [_require_positive_int(dimension, label=f"{label}[{index}]") for index, dimension in enumerate(value)]


def _require_schema_version(value: Any, *, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        msg = f"{label} must be integer {expected}."
        raise ValueError(msg)
    return value


def _require_manifest_real(value: Any, *, label: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real number."
        raise TypeError(msg)
    numeric = float(value)
    invalid = numeric <= 0.0 if positive else numeric < 0.0
    if not math.isfinite(numeric) or invalid:
        domain = "positive" if positive else "non-negative"
        msg = f"{label} must be finite and {domain}."
        raise ValueError(msg)
    return numeric


def _validate_source_manifest_configuration(value: Any) -> dict[str, Any]:
    """Validate the exact terminal COMSOL manifest configuration."""
    if not isinstance(value, dict):
        msg = "Source manifest snapshot configuration must be a mapping."
        raise TypeError(msg)
    configuration = value
    _require_exact_keys(
        configuration,
        _SOURCE_MANIFEST_CONFIGURATION_KEYS,
        label="Source manifest snapshot configuration",
    )
    if configuration["method"] not in {"uniform", "lhs", "sobol"}:
        msg = "Source manifest snapshot configuration.method is unsupported."
        raise ValueError(msg)
    count = configuration["N"]
    seed = configuration["seed"]
    if isinstance(count, bool) or not isinstance(count, int):
        msg = "Source manifest snapshot configuration.N must be an integer."
        raise TypeError(msg)
    if not 1 <= count <= _MAX_EXACT_MANIFEST_INTEGER:
        msg = "Source manifest snapshot configuration.N is outside its supported range."
        raise ValueError(msg)
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "Source manifest snapshot configuration.seed must be an integer."
        raise TypeError(msg)
    if not 0 <= seed <= _MAX_RANDOM_SEED:
        msg = "Source manifest snapshot configuration.seed is outside its supported range."
        raise ValueError(msg)
    _require_manifest_real(
        configuration["variation"],
        label="Source manifest snapshot configuration.variation",
        positive=False,
    )
    lengths = {
        name: _require_manifest_real(
            configuration[name],
            label=f"Source manifest snapshot configuration.{name}",
            positive=True,
        )
        for name in ("Lx", "Ly", "res")
    }
    if lengths["res"] > min(lengths["Lx"], lengths["Ly"]):
        msg = "Source manifest snapshot resolution exceeds the shorter domain length."
        raise ValueError(msg)
    if not isinstance(configuration["save_model"], bool):
        msg = "Source manifest snapshot configuration.save_model must be boolean."
        raise TypeError(msg)
    _require_sha256(
        configuration["sample_sha256"],
        label="Source manifest snapshot configuration.sample_sha256",
    )
    _require_sha256(
        configuration["template_sha256"],
        label="Source manifest snapshot configuration.template_sha256",
    )
    template_name = configuration["template_name"]
    if (
        not isinstance(template_name, str)
        or not template_name
        or Path(template_name).name != template_name
        or "/" in template_name
        or "\\" in template_name
        or not template_name.endswith(".mph")
    ):
        msg = "Source manifest snapshot configuration.template_name must be an .mph basename."
        raise ValueError(msg)
    return configuration


def _validate_source_manifest_membership(value: Any, *, count: int) -> list[str]:
    """Validate exact complete ordered manifest membership."""
    if not isinstance(value, list) or not all(isinstance(case_id, str) and _CASE_ID_PATTERN.fullmatch(case_id) for case_id in value):
        msg = "Source manifest snapshot intended_case_ids must contain canonical case identifiers."
        raise TypeError(msg)
    if len(value) != len(set(value)):
        msg = "Source manifest snapshot intended_case_ids must be unique."
        raise ValueError(msg)
    if len(value) != count:
        msg = "Terminal source manifest membership must contain exactly configuration.N cases."
        raise ValueError(msg)
    return value


def _validate_source_manifest_records(
    value: Any,
    *,
    intended: list[str],
    save_model: bool,
) -> list[dict[str, Any]]:
    """Validate complete terminal case records, normalizing MATLAB singleton JSON."""
    records = [value] if isinstance(value, dict) else value
    if not isinstance(records, list):
        msg = "Source manifest snapshot cases must be a list of mappings."
        raise TypeError(msg)
    if len(records) != len(intended):
        msg = "Source manifest snapshot cases must align one-to-one with intended_case_ids."
        raise ValueError(msg)
    normalized: list[dict[str, Any]] = []
    for index, (case_id, record) in enumerate(zip(intended, records, strict=True)):
        if not isinstance(record, dict):
            msg = f"Source manifest snapshot cases[{index}] must be a mapping."
            raise TypeError(msg)
        record_label = f"Source manifest snapshot cases[{index}]"
        _require_exact_keys(record, _SOURCE_MANIFEST_RECORD_KEYS, label=record_label)
        if record["case_id"] != case_id or record["status"] != "complete" or record["stage"] != "simulation" or record["message"] != "":
            msg = "Source manifest snapshot terminal case records do not match ordered membership."
            raise ValueError(msg)
        files = record["files"]
        if not isinstance(files, dict):
            msg = f"{record_label}.files must be a mapping."
            raise TypeError(msg)
        _require_exact_keys(files, _SOURCE_MANIFEST_FILE_KEYS, label=f"{record_label}.files")
        for filename in ("raw_csv_sha256", "raw_json_sha256", "solution_csv_sha256"):
            _require_sha256(files[filename], label=f"{record_label}.files.{filename}")
        model_digest = files["solution_model_sha256"]
        if save_model:
            _require_sha256(model_digest, label=f"{record_label}.files.solution_model_sha256")
        elif model_digest != "":
            msg = "Source manifest snapshot cannot bind solved models when save_model is false."
            raise ValueError(msg)
        normalized.append(record)
    return normalized


def _validate_source_manifest_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate one terminal version-1 COMSOL batch-manifest snapshot."""
    _require_exact_keys(manifest, _SOURCE_MANIFEST_KEYS, label="Source manifest snapshot")
    if manifest["schema_kind"] != SOURCE_MANIFEST_SCHEMA_KIND:
        msg = "Unsupported source manifest snapshot schema kind."
        raise ValueError(msg)
    _require_schema_version(
        manifest["schema_version"],
        expected=SOURCE_MANIFEST_SCHEMA_VERSION,
        label="Source manifest snapshot schema_version",
    )
    batch_name = manifest["batch_name"]
    if not isinstance(batch_name, str) or not batch_name:
        msg = "Source manifest snapshot batch_name must be a non-empty string."
        raise ValueError(msg)
    if manifest["status"] != "complete":
        msg = "Source manifest snapshot must be terminal with status 'complete'."
        raise ValueError(msg)
    configuration = _validate_source_manifest_configuration(manifest["configuration"])
    field_schema = manifest["field_schema"]
    if not isinstance(field_schema, dict) or field_schema != _SOURCE_MANIFEST_FIELD_SCHEMA:
        msg = "Source manifest snapshot field_schema does not match the maintained COMSOL contract."
        raise ValueError(msg)
    intended = _validate_source_manifest_membership(
        manifest["intended_case_ids"],
        count=configuration["N"],
    )
    normalized = dict(manifest)
    normalized["cases"] = _validate_source_manifest_records(
        manifest["cases"],
        intended=intended,
        save_model=configuration["save_model"],
    )
    return normalized


def _validate_timing_summary(value: Any, *, intended_count: int, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(value, {"status", "measured_case_count", "intended_case_count"}, label=label)
    status = value["status"]
    measured = _require_nonnegative_int(value["measured_case_count"], label=f"{label}.measured_case_count")
    intended = _require_nonnegative_int(value["intended_case_count"], label=f"{label}.intended_case_count")
    if intended != intended_count or measured > intended:
        msg = f"{label} coverage does not match intended membership."
        raise ValueError(msg)
    expected_status = "missing" if measured == 0 else "complete" if measured == intended else "partial"
    if status != expected_status:
        msg = f"{label}.status must be {expected_status!r}, got {status!r}."
        raise ValueError(msg)
    return value


def _derived_timing_aggregates(durations: list[float]) -> dict[str, Any]:
    """Derive MATLAB-compatible timing aggregates from admitted durations."""
    if not durations:
        return {
            "measured_case_count": 0,
            "mean_s": [],
            "median_s": [],
            "p10_s": [],
            "p90_s": [],
        }
    values = np.asarray(durations, dtype=np.float64)
    return {
        "measured_case_count": len(durations),
        "mean_s": float(np.mean(values)),
        "median_s": float(np.percentile(values, 50.0)),
        "p10_s": float(np.percentile(values, 10.0)),
        "p90_s": float(np.percentile(values, 90.0)),
    }


def _validate_timing_aggregates(value: Any, *, durations: list[float]) -> None:
    """Require aggregates derived from cases, allowing only float roundoff."""
    expected = _derived_timing_aggregates(durations)
    if not isinstance(value, Mapping) or set(value) != set(expected):
        msg = "COMSOL timing snapshot aggregates have invalid fields."
        raise ValueError(msg)
    measured_case_count = _require_nonnegative_int(
        value["measured_case_count"],
        label="COMSOL timing snapshot aggregates.measured_case_count",
    )
    if measured_case_count != expected["measured_case_count"]:
        msg = "COMSOL timing snapshot aggregate count does not match its cases."
        raise ValueError(msg)
    for field in ("mean_s", "median_s", "p10_s", "p90_s"):
        actual = value[field]
        expected_value = expected[field]
        if isinstance(expected_value, list):
            valid = actual == []
        else:
            valid = (
                not isinstance(actual, bool)
                and isinstance(actual, Real)
                and math.isfinite(float(actual))
                and math.isclose(float(actual), expected_value, rel_tol=1e-12, abs_tol=1e-12)
            )
        if not valid:
            msg = f"COMSOL timing snapshot {field} is not derived from its cases."
            raise ValueError(msg)


def validate_comsol_timing_snapshot(
    timing: dict[str, Any],
    *,
    batch_name: str,
    manifest_sha256: str,
    intended_case_ids: list[str],
) -> dict[str, Any]:
    """Validate one final, manifest-bound operational timing snapshot."""
    _require_exact_keys(
        timing,
        {"schema_kind", "schema_version", "batch_name", "batch_manifest_sha256", "runtime", "cases", "aggregates"},
        label="COMSOL timing snapshot",
    )
    if timing["schema_kind"] != "comsol_solve_timing":
        msg = "Unsupported COMSOL timing snapshot schema kind."
        raise ValueError(msg)
    _require_schema_version(
        timing["schema_version"],
        expected=1,
        label="COMSOL timing snapshot schema_version",
    )
    if timing["batch_name"] != batch_name or timing["batch_manifest_sha256"] != manifest_sha256:
        msg = "COMSOL timing snapshot source-batch identity does not match metadata provenance."
        raise ValueError(msg)
    runtime = timing["runtime"]
    runtime_fields = {"matlab_version", "comsol_version", "os", "hostname", "processor", "case_execution"}
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_fields:
        msg = "COMSOL timing snapshot runtime provenance has invalid fields."
        raise ValueError(msg)
    if any(not isinstance(runtime[field], str) or not runtime[field] for field in runtime_fields):
        msg = "COMSOL timing snapshot runtime provenance must contain non-empty text."
        raise ValueError(msg)
    if runtime["case_execution"] != "sequential":
        msg = "COMSOL timing snapshot runtime.case_execution must be sequential."
        raise ValueError(msg)
    raw_cases = timing["cases"]
    cases = [raw_cases] if isinstance(raw_cases, dict) else raw_cases
    if not isinstance(cases, list):
        msg = "COMSOL timing snapshot cases must be a list of mappings."
        raise TypeError(msg)
    intended = set(intended_case_ids)
    case_ids: list[str] = []
    durations: list[float] = []
    for index, record in enumerate(cases):
        if not isinstance(record, dict):
            msg = f"COMSOL timing cases[{index}] must be a mapping."
            raise TypeError(msg)
        _require_exact_keys(record, {"case_id", "comsol_solve_s"}, label=f"COMSOL timing cases[{index}]")
        case_id = record["case_id"]
        duration = record["comsol_solve_s"]
        if not isinstance(case_id, str) or case_id not in intended:
            msg = f"COMSOL timing cases[{index}] is outside authoritative membership."
            raise ValueError(msg)
        if isinstance(duration, bool) or not isinstance(duration, Real) or not math.isfinite(float(duration)) or float(duration) <= 0:
            msg = f"COMSOL timing cases[{index}].comsol_solve_s must be finite and positive."
            raise ValueError(msg)
        case_ids.append(case_id)
        durations.append(float(duration))
    if len(case_ids) != len(set(case_ids)):
        msg = "COMSOL timing snapshot contains duplicate case IDs."
        raise ValueError(msg)
    present = set(case_ids)
    expected_order = [case_id for case_id in intended_case_ids if case_id in present]
    if case_ids != expected_order:
        msg = "COMSOL timing snapshot cases must follow authoritative manifest order."
        raise ValueError(msg)
    _validate_timing_aggregates(timing["aggregates"], durations=durations)
    normalized = dict(timing)
    normalized["cases"] = cases
    return normalized


def _validate_inventory_files(
    root: Path,
    *,
    names: set[str],
    inventory: dict[str, Any],
) -> None:
    """Validate exact metadata file membership, hashes, sizes, and declared roles."""
    files = inventory["files"]
    if not isinstance(files, dict):
        msg = "Metadata inventory files must be a mapping."
        raise TypeError(msg)
    expected_inventory_names = names.difference({INVENTORY_FILENAME})
    if set(files) != expected_inventory_names:
        msg = "Metadata inventory file membership does not match the package."
        raise ValueError(msg)
    for filename, entry in files.items():
        if not isinstance(entry, dict):
            msg = f"Metadata inventory entry {filename!r} must be a mapping."
            raise TypeError(msg)
        _require_exact_keys(entry, {"sha256", "size_bytes", "required", "role"}, label=f"Metadata inventory {filename}")
        file_path = root / filename
        expected_sha256 = _require_sha256(entry["sha256"], label=f"Metadata inventory {filename}.sha256")
        expected_size = _require_nonnegative_int(
            entry["size_bytes"],
            label=f"Metadata inventory {filename}.size_bytes",
        )
        if file_path.stat().st_size != expected_size or common.serialization.file_sha256(file_path) != expected_sha256:
            msg = f"Metadata snapshot hash or size mismatch: {file_path}"
            raise ValueError(msg)
        if not isinstance(entry["required"], bool) or not isinstance(entry["role"], str) or not entry["role"]:
            msg = f"Metadata inventory {filename} has invalid required/role values."
            raise ValueError(msg)
    if not _REQUIRED_SNAPSHOT_FILES.issubset(files):
        msg = "Metadata inventory omits a required validated snapshot."
        raise ValueError(msg)


def validate_dataset_metadata_directory(
    directory: Path | str,
    *,
    dataset_identity: DatasetIdentity,
) -> DatasetMetadata:
    """Validate one complete metadata package without accessing generation data."""
    root = Path(directory)
    if not root.is_dir():
        msg = f"Dataset metadata directory does not exist: {root}"
        raise FileNotFoundError(msg)
    names = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(names.difference(_ALLOWED_SNAPSHOT_FILES))
    missing = sorted((_REQUIRED_SNAPSHOT_FILES | {INVENTORY_FILENAME}).difference(names))
    if missing or unexpected:
        msg = f"Dataset metadata package is incomplete or inconsistent: missing={missing}; unexpected={unexpected}."
        raise ValueError(msg)
    provenance = _load_json(root / PROVENANCE_FILENAME, label="dataset provenance")
    _require_exact_keys(
        provenance,
        {
            "schema_kind",
            "schema_version",
            "dataset_id",
            "dataset_schema_version",
            "dataset_fingerprint",
            "task",
            "task_contract_digest",
            "source_batch",
            "sample_count",
            "spatial_shape",
            "timing",
        },
        label="Dataset provenance",
    )
    if provenance["schema_kind"] != PROVENANCE_SCHEMA_KIND:
        msg = "Unsupported dataset provenance schema kind."
        raise ValueError(msg)
    _require_schema_version(
        provenance["schema_version"],
        expected=METADATA_SCHEMA_VERSION,
        label="Dataset provenance schema_version",
    )
    _require_schema_version(
        provenance["dataset_schema_version"],
        expected=TRAINING_DATASET_SCHEMA_VERSION,
        label="Dataset provenance dataset_schema_version",
    )
    provenance_sample_count = _require_positive_int(
        provenance["sample_count"],
        label="Dataset provenance sample_count",
    )
    provenance_spatial_shape = _require_spatial_shape(
        provenance["spatial_shape"],
        label="Dataset provenance spatial_shape",
    )
    if (
        provenance["dataset_id"] != dataset_identity.dataset_id
        or provenance["dataset_fingerprint"] != dataset_identity.fingerprint
        or provenance["task"] != dataset_identity.task
        or provenance["task_contract_digest"] != dataset_identity.task_contract_digest
        or provenance_sample_count != dataset_identity.sample_count
        or provenance_spatial_shape != list(dataset_identity.spatial_shape)
    ):
        msg = "Dataset provenance does not match the loaded final dataset identity."
        raise ValueError(msg)
    source_batch = provenance["source_batch"]
    if not isinstance(source_batch, dict):
        msg = "Dataset provenance source_batch must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(
        source_batch,
        {"batch_name", "batch_manifest_sha256", "batch_manifest_identity_sha256"},
        label="Dataset provenance source_batch",
    )
    manifest_sha256 = _require_sha256(source_batch["batch_manifest_sha256"], label="batch_manifest_sha256")
    _require_sha256(source_batch["batch_manifest_identity_sha256"], label="batch_manifest_identity_sha256")

    inventory = _load_json(root / INVENTORY_FILENAME, label="metadata inventory")
    _require_exact_keys(
        inventory,
        {
            "schema_kind",
            "schema_version",
            "dataset_id",
            "dataset_fingerprint",
            "task",
            "task_contract_digest",
            "sample_count",
            "spatial_shape",
            "source_batch_name",
            "source_manifest_sha256",
            "files",
            "timing",
        },
        label="Metadata inventory",
    )
    if inventory["schema_kind"] != INVENTORY_SCHEMA_KIND:
        msg = "Unsupported metadata inventory schema kind."
        raise ValueError(msg)
    _require_schema_version(
        inventory["schema_version"],
        expected=METADATA_SCHEMA_VERSION,
        label="Metadata inventory schema_version",
    )
    inventory_sample_count = _require_positive_int(
        inventory["sample_count"],
        label="Metadata inventory sample_count",
    )
    inventory_spatial_shape = _require_spatial_shape(
        inventory["spatial_shape"],
        label="Metadata inventory spatial_shape",
    )
    expected_bindings = (
        inventory["dataset_id"] == dataset_identity.dataset_id,
        inventory["dataset_fingerprint"] == dataset_identity.fingerprint,
        inventory["task"] == dataset_identity.task,
        inventory["task_contract_digest"] == dataset_identity.task_contract_digest,
        inventory_sample_count == dataset_identity.sample_count,
        inventory_spatial_shape == list(dataset_identity.spatial_shape),
        inventory["source_batch_name"] == source_batch["batch_name"],
        inventory["source_manifest_sha256"] == manifest_sha256,
    )
    if not all(expected_bindings):
        msg = "Metadata inventory does not match dataset or source-batch identity."
        raise ValueError(msg)
    _validate_inventory_files(root, names=names, inventory=inventory)

    manifest = _validate_source_manifest_snapshot(_load_json(root / SOURCE_MANIFEST_FILENAME, label="source manifest snapshot"))
    if common.serialization.file_sha256(root / SOURCE_MANIFEST_FILENAME) != manifest_sha256:
        msg = "Source manifest snapshot does not match its recorded exact SHA-256."
        raise ValueError(msg)
    intended = manifest["intended_case_ids"]
    if manifest["batch_name"] != source_batch["batch_name"] or intended != list(dataset_identity.sample_ids):
        msg = "Source manifest snapshot membership does not match the final dataset."
        raise ValueError(msg)
    configuration = manifest["configuration"]
    expected_sample_csv_sha256 = _require_sha256(
        configuration.get("sample_sha256"),
        label="source manifest configuration.sample_sha256",
    )
    if common.serialization.file_sha256(root / SOURCE_SAMPLE_CSV_FILENAME) != expected_sample_csv_sha256:
        msg = "Parameter-sample CSV snapshot does not match the source manifest SHA-256."
        raise ValueError(msg)
    _load_json(root / SOURCE_SAMPLE_JSON_FILENAME, label="parameter-sample JSON snapshot")
    timing_summary = _validate_timing_summary(provenance["timing"], intended_count=len(intended), label="Dataset provenance timing")
    inventory_summary = _validate_timing_summary(inventory["timing"], intended_count=len(intended), label="Metadata inventory timing")
    if timing_summary != inventory_summary:
        msg = "Dataset provenance and metadata inventory timing summaries disagree."
        raise ValueError(msg)
    timing_path = root / COMSOL_TIMING_FILENAME
    timing = _load_json(timing_path, label="COMSOL timing snapshot") if timing_path.is_file() else None
    if timing is None and timing_summary["measured_case_count"] != 0:
        msg = "Metadata declares measured COMSOL timing but has no timing snapshot."
        raise ValueError(msg)
    if timing is not None:
        timing = validate_comsol_timing_snapshot(
            timing,
            batch_name=source_batch["batch_name"],
            manifest_sha256=manifest_sha256,
            intended_case_ids=intended,
        )
        if len(timing["cases"]) != timing_summary["measured_case_count"]:
            msg = "COMSOL timing snapshot count disagrees with metadata coverage."
            raise ValueError(msg)
    return DatasetMetadata(root, provenance, inventory, manifest, timing)


def load_dataset_metadata(
    dataset_id: str,
    *,
    dataset_identity: DatasetIdentity,
    metadata_root: Path | str | None = None,
) -> DatasetMetadata:
    """Resolve and validate one model-training metadata package."""
    directory = common.paths.resolve_dataset_metadata_dir(dataset_id, metadata_root=metadata_root)
    return validate_dataset_metadata_directory(directory, dataset_identity=dataset_identity)
