"""Validate self-contained model-training metadata snapshots."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src import common, domain
from src.datasets.dataset_identity import TRAINING_DATASET_SCHEMA_VERSION, build_generated_batch_identity

if TYPE_CHECKING:
    from src.datasets.dataset_identity import DatasetIdentity

METADATA_FILENAME = "dataset_metadata.json"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
SOURCE_SAMPLE_CSV_FILENAME = "source_sample.csv"
SOURCE_SAMPLE_JSON_FILENAME = "source_sample.json"
COMSOL_TIMING_FILENAME = "comsol_solve_timing.json"
METADATA_SCHEMA_KIND = "training_dataset_metadata"
METADATA_SCHEMA_VERSION = 1
BUILDER_MODULE = "data_generation.build_training_dataset"
PUBLICATION_METHOD = "atomic_directory_rename"
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
        SOURCE_MANIFEST_FILENAME,
        SOURCE_SAMPLE_CSV_FILENAME,
        SOURCE_SAMPLE_JSON_FILENAME,
    }
)
_ALLOWED_PACKAGE_FILES = _REQUIRED_SNAPSHOT_FILES | {COMSOL_TIMING_FILENAME, METADATA_FILENAME}
_SOURCE_SAMPLE_JSON_KEYS = frozenset({"meta", "n_cases"})
_SOURCE_SAMPLE_META_KEYS = frozenset({"method", "variation", "N", "seed", "base", "param_names", "timestamp"})
_SNAPSHOT_ROLES = {
    SOURCE_MANIFEST_FILENAME: "validated_generation_manifest",
    SOURCE_SAMPLE_CSV_FILENAME: "validated_parameter_sample_csv",
    SOURCE_SAMPLE_JSON_FILENAME: "validated_parameter_sample_json",
    COMSOL_TIMING_FILENAME: "validated_operational_comsol_timing",
}


@dataclass(frozen=True, slots=True)
class SourceSampleSemantics:
    """Validated portable sampling identity and manifest-aligned CSV values."""

    sample_json: dict[str, Any]
    sampling: dict[str, Any]
    case_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_rows: tuple[dict[str, int | float], ...]
    generated_batch_identity: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Validated metadata package bound to one final training dataset."""

    directory: Path
    metadata: dict[str, Any]
    source_manifest: dict[str, Any]
    timing: dict[str, Any] | None

    @property
    def source_manifest_sha256(self) -> str:
        """Return the exact validated source-manifest snapshot digest."""
        snapshots = self.metadata["artifacts"]["snapshots"]
        return str(snapshots[SOURCE_MANIFEST_FILENAME]["sha256"])

    @property
    def timing_summary(self) -> dict[str, Any]:
        """Return the validated optional COMSOL timing coverage summary."""
        return dict(self.metadata["operational_provenance"]["timing"])


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


def _parse_source_sample_json(
    sample_json: bytes,
    *,
    configuration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], int]:
    """Validate the parameter JSON and return timestamp-free sampling facts."""
    try:
        decoded = json.loads(sample_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = "Parameter-sample JSON snapshot is not valid UTF-8 JSON."
        raise ValueError(msg) from error
    if not isinstance(decoded, dict):
        msg = "Parameter-sample JSON snapshot must contain an object."
        raise TypeError(msg)
    _require_exact_keys(decoded, _SOURCE_SAMPLE_JSON_KEYS, label="Parameter-sample JSON")
    sample_meta = decoded["meta"]
    if not isinstance(sample_meta, dict):
        msg = "Parameter-sample JSON meta must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(sample_meta, _SOURCE_SAMPLE_META_KEYS, label="Parameter-sample JSON meta")

    method = sample_meta["method"]
    if not isinstance(method, str) or not method:
        msg = "Parameter-sample JSON meta.method must be non-empty text."
        raise TypeError(msg)
    if method != configuration["method"]:
        msg = "Parameter-sample JSON meta.method does not match the source manifest."
        raise ValueError(msg)
    variation = _require_manifest_real(
        sample_meta["variation"],
        label="Parameter-sample JSON meta.variation",
        positive=False,
    )
    if variation != float(configuration["variation"]):
        msg = "Parameter-sample JSON meta.variation does not match the source manifest."
        raise ValueError(msg)
    count = _require_positive_int(sample_meta["N"], label="Parameter-sample JSON meta.N")
    seed = _require_nonnegative_int(sample_meta["seed"], label="Parameter-sample JSON meta.seed")
    if count != configuration["N"]:
        msg = "Parameter-sample JSON meta.N does not match the source manifest."
        raise ValueError(msg)
    if seed != configuration["seed"]:
        msg = "Parameter-sample JSON meta.seed does not match the source manifest."
        raise ValueError(msg)
    n_cases = _require_positive_int(decoded["n_cases"], label="Parameter-sample JSON n_cases")
    if n_cases != count:
        msg = "Parameter-sample JSON n_cases does not match meta.N."
        raise ValueError(msg)
    if not isinstance(sample_meta["base"], dict):
        msg = "Parameter-sample JSON meta.base must be a mapping."
        raise TypeError(msg)
    parameter_names_value = sample_meta["param_names"]
    if (
        not isinstance(parameter_names_value, list)
        or not parameter_names_value
        or not all(isinstance(name, str) and name for name in parameter_names_value)
    ):
        msg = "Parameter-sample JSON meta.param_names must be a non-empty list of names."
        raise TypeError(msg)
    parameter_names = tuple(parameter_names_value)
    if len(parameter_names) != len(set(parameter_names)) or "case_id" in parameter_names:
        msg = "Parameter-sample JSON meta.param_names must be unique and exclude case_id."
        raise ValueError(msg)
    timestamp = sample_meta["timestamp"]
    if not isinstance(timestamp, str) or not timestamp:
        msg = "Parameter-sample JSON meta.timestamp must be non-empty text."
        raise TypeError(msg)
    portable_sampling = {key: value for key, value in sample_meta.items() if key != "timestamp"}
    return decoded, portable_sampling, parameter_names, count


def _parse_source_sample_csv(
    sample_csv: bytes,
    *,
    parameter_names: tuple[str, ...],
    count: int,
    intended_case_ids: list[str],
) -> tuple[tuple[str, ...], tuple[dict[str, int | float], ...]]:
    """Validate exact CSV shape, finite values, and ordered manifest membership."""
    try:
        csv_text = sample_csv.decode("utf-8")
        rows = [row for row in csv.reader(io.StringIO(csv_text, newline=""), delimiter=";", strict=True) if row]
    except (UnicodeDecodeError, csv.Error) as error:
        msg = "Parameter-sample CSV snapshot is not valid strict UTF-8 semicolon CSV."
        raise ValueError(msg) from error
    expected_header = ["case_id", *parameter_names]
    if not rows or rows[0] != expected_header:
        msg = "Parameter-sample CSV columns do not match JSON meta.param_names in exact order."
        raise ValueError(msg)
    data_rows = rows[1:]
    if len(data_rows) != count:
        msg = "Parameter-sample CSV row count does not match JSON and manifest case count."
        raise ValueError(msg)

    case_ids: list[str] = []
    for row_index, row in enumerate(data_rows, start=1):
        if len(row) != len(expected_header):
            msg = f"Parameter-sample CSV row {row_index} has the wrong field count."
            raise ValueError(msg)
        try:
            numeric_case_id = float(row[0])
            values = [float(value) for value in row[1:]]
        except ValueError as error:
            msg = f"Parameter-sample CSV row {row_index} must contain only numeric values."
            raise ValueError(msg) from error
        if (
            not math.isfinite(numeric_case_id)
            or numeric_case_id != math.floor(numeric_case_id)
            or not 0 <= numeric_case_id <= _MAX_EXACT_MANIFEST_INTEGER
        ):
            msg = f"Parameter-sample CSV row {row_index} case_id must be an exact non-negative integer."
            raise ValueError(msg)
        if not all(math.isfinite(value) for value in values):
            msg = f"Parameter-sample CSV row {row_index} parameters must be finite."
            raise ValueError(msg)
        case_ids.append(f"case_{int(numeric_case_id):04d}")
    if len(case_ids) != len(set(case_ids)):
        msg = "Parameter-sample CSV case IDs must be unique."
        raise ValueError(msg)
    if case_ids != intended_case_ids:
        msg = "Parameter-sample CSV ordered sample IDs do not match the source manifest."
        raise ValueError(msg)

    try:
        numeric_frame = pd.read_csv(io.BytesIO(sample_csv), sep=";").apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        msg = "Parameter-sample CSV values are not valid maintained numeric serialization."
        raise ValueError(msg) from error
    if not np.isfinite(numeric_frame.to_numpy(dtype=np.float64)).all():
        msg = "Parameter-sample CSV must contain only finite numeric values."
        raise ValueError(msg)
    parameter_rows: list[dict[str, int | float]] = []
    for row_index in range(count):
        row = numeric_frame.iloc[row_index]
        parameter_rows.append({name: value.item() if isinstance((value := row[name]), np.generic) else value for name in parameter_names})
    return tuple(case_ids), tuple(parameter_rows)


def _validate_final_parameter_rows(
    semantics: SourceSampleSemantics,
    final_source_metadata: tuple[dict[str, Any], ...],
) -> None:
    """Bind sampled CSV values to duplicated final per-case metadata."""
    if len(final_source_metadata) != len(semantics.parameter_rows):
        msg = "Final dataset source metadata does not align with source-sample rows."
        raise ValueError(msg)
    for index, (case_id, parameters, final_value) in enumerate(zip(semantics.case_ids, semantics.parameter_rows, final_source_metadata, strict=True)):
        if not isinstance(final_value, Mapping) or final_value.get("case_id") != case_id:
            msg = f"Final dataset source_metadata[{index}] does not match source-sample membership."
            raise ValueError(msg)
        final_parameters = final_value.get("parameters")
        if not isinstance(final_parameters, Mapping) or set(final_parameters) != set(semantics.parameter_names):
            msg = f"Final dataset source_metadata[{index}].parameters does not match sampled variable names."
            raise ValueError(msg)
        for name, expected_value in parameters.items():
            actual = final_parameters[name]
            if isinstance(actual, bool) or not isinstance(actual, Real) or not math.isfinite(float(actual)) or float(actual) != float(expected_value):
                msg = f"Final dataset source_metadata[{index}].parameters.{name} does not match source-sample CSV."
                raise ValueError(msg)


def _validate_final_source_sample_bindings(
    semantics: SourceSampleSemantics,
    *,
    dataset_identity: DatasetIdentity,
    csv_sha256: str,
    json_sha256: str,
    manifest_sha256: str | None,
) -> None:
    """Bind normalized source facts to all facts retained from the final payload."""
    if tuple(dataset_identity.sample_ids) != semantics.case_ids or dataset_identity.sample_count != len(semantics.case_ids):
        msg = "Source-sample membership does not match the final dataset identity."
        raise ValueError(msg)
    generated_identity = semantics.generated_batch_identity
    if dataset_identity.generated_batch_identity_sha256 != generated_identity["batch_manifest_identity_sha256"]:
        msg = "Source-sample semantics do not match the final generated-batch scientific identity."
        raise ValueError(msg)
    if dataset_identity.generated_batch_identity is not None and dataset_identity.generated_batch_identity != generated_identity:
        msg = "Reconstructed generated-batch identity does not match the final dataset payload."
        raise ValueError(msg)
    provenance = dataset_identity.source_provenance
    if provenance is not None:
        expected_hashes = {
            "source_sample_csv_sha256": csv_sha256,
            "source_sample_json_sha256": json_sha256,
        }
        if manifest_sha256 is not None:
            expected_hashes["batch_manifest_sha256"] = manifest_sha256
        for key, expected_digest in expected_hashes.items():
            if provenance.get(key) != expected_digest:
                msg = f"Final dataset source provenance {key} does not match the metadata snapshot."
                raise ValueError(msg)
    if dataset_identity.source_metadata is not None:
        _validate_final_parameter_rows(semantics, dataset_identity.source_metadata)


def validate_source_sample_semantics(
    sample_csv: bytes,
    sample_json: bytes,
    *,
    source_manifest: Mapping[str, Any],
    dataset_identity: DatasetIdentity | None = None,
    source_manifest_sha256: str | None = None,
) -> SourceSampleSemantics:
    """Cross-bind source-sample snapshots, their manifest, and final identity."""
    if not isinstance(sample_csv, bytes) or not isinstance(sample_json, bytes):
        msg = "Source-sample snapshots must be exact bytes."
        raise TypeError(msg)
    manifest = _validate_source_manifest_snapshot(dict(source_manifest))
    csv_sha256 = hashlib.sha256(sample_csv).hexdigest()
    if csv_sha256 != manifest["configuration"]["sample_sha256"]:
        msg = "Parameter-sample CSV SHA-256 does not match the source manifest."
        raise ValueError(msg)
    decoded_json, sampling, parameter_names, count = _parse_source_sample_json(
        sample_json,
        configuration=manifest["configuration"],
    )
    case_ids, parameter_rows = _parse_source_sample_csv(
        sample_csv,
        parameter_names=parameter_names,
        count=count,
        intended_case_ids=manifest["intended_case_ids"],
    )
    result = SourceSampleSemantics(
        sample_json=decoded_json,
        sampling=sampling,
        case_ids=case_ids,
        parameter_names=parameter_names,
        parameter_rows=parameter_rows,
        generated_batch_identity=build_generated_batch_identity(manifest, sampling=sampling),
    )
    if dataset_identity is not None:
        _validate_final_source_sample_bindings(
            result,
            dataset_identity=dataset_identity,
            csv_sha256=csv_sha256,
            json_sha256=hashlib.sha256(sample_json).hexdigest(),
            manifest_sha256=source_manifest_sha256,
        )
    return result


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


def _validate_tensor_contract(
    value: Any,
    *,
    dataset_identity: DatasetIdentity,
) -> dict[str, Any]:
    """Validate compact tensor shapes and dtypes against the registered task."""
    if not isinstance(value, dict):
        msg = "Dataset metadata tensors must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(value, {"inputs", "outputs"}, label="Dataset metadata tensors")
    task = domain.tasks.registry.get_task(dataset_identity.task)
    expected = {
        "inputs": {
            "dtype": "float32",
            "shape": [dataset_identity.sample_count, task.in_channels, *dataset_identity.spatial_shape],
        },
        "outputs": {
            "dtype": "float32",
            "shape": [dataset_identity.sample_count, task.out_channels, *dataset_identity.spatial_shape],
        },
    }
    if value != expected:
        msg = "Dataset metadata tensor shapes or dtypes do not match the final dataset contract."
        raise ValueError(msg)
    return value


def _validate_file_artifact(value: Any, *, label: str) -> dict[str, Any]:
    """Validate one immutable file name, SHA-256, and size record."""
    if not isinstance(value, dict):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(value, {"filename", "sha256", "size_bytes"}, label=label)
    filename = value["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        msg = f"{label}.filename must be one non-empty basename."
        raise ValueError(msg)
    _require_sha256(value["sha256"], label=f"{label}.sha256")
    _require_positive_int(value["size_bytes"], label=f"{label}.size_bytes")
    return value


def _validate_snapshot_artifacts(
    root: Path,
    *,
    names: set[str],
    snapshots: Any,
) -> dict[str, Any]:
    """Validate exact snapshot membership, hashes, sizes, and fixed roles."""
    if not isinstance(snapshots, dict):
        msg = "Dataset metadata snapshots must be a mapping."
        raise TypeError(msg)
    expected_names = names.difference({METADATA_FILENAME})
    if set(snapshots) != expected_names:
        msg = "Dataset metadata snapshot membership does not match the package."
        raise ValueError(msg)
    for filename, entry in snapshots.items():
        if not isinstance(entry, dict):
            msg = f"Dataset metadata snapshot {filename!r} must be a mapping."
            raise TypeError(msg)
        _require_exact_keys(
            entry,
            {"sha256", "size_bytes", "required", "role"},
            label=f"Dataset metadata snapshot {filename}",
        )
        expected_required = filename in _REQUIRED_SNAPSHOT_FILES
        expected_role = _SNAPSHOT_ROLES.get(filename)
        if entry["required"] is not expected_required or entry["role"] != expected_role:
            msg = f"Dataset metadata snapshot {filename!r} has invalid ownership metadata."
            raise ValueError(msg)
        expected_sha256 = _require_sha256(
            entry["sha256"],
            label=f"Dataset metadata snapshot {filename}.sha256",
        )
        expected_size = _require_nonnegative_int(
            entry["size_bytes"],
            label=f"Dataset metadata snapshot {filename}.size_bytes",
        )
        file_path = root / filename
        if file_path.stat().st_size != expected_size or common.serialization.file_sha256(file_path) != expected_sha256:
            msg = f"Metadata snapshot hash or size mismatch: {file_path}"
            raise ValueError(msg)
    return snapshots


def _validate_generated_batch_digest(
    value: Any,
    *,
    dataset_identity: DatasetIdentity,
) -> None:
    """Bind metadata to the generated-batch digest validated inside the dataset."""
    digest = _require_sha256(
        value,
        label="Dataset metadata generated_batch_identity_sha256",
    )
    expected = dataset_identity.generated_batch_identity_sha256
    if expected is None:
        msg = "Dataset identity does not expose its validated generated-batch scientific digest."
        raise ValueError(msg)
    if digest != expected:
        msg = "Dataset metadata scientific identity does not match the loaded final dataset."
        raise ValueError(msg)


def validate_dataset_metadata_directory(
    directory: Path | str,
    *,
    dataset_identity: DatasetIdentity,
    dataset_path: Path | str | None = None,
) -> DatasetMetadata:
    """Validate one consolidated metadata package without generation access."""
    root = Path(directory)
    if not root.is_dir():
        msg = f"Dataset metadata directory does not exist: {root}"
        raise FileNotFoundError(msg)
    names = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(names.difference(_ALLOWED_PACKAGE_FILES))
    missing = sorted((_REQUIRED_SNAPSHOT_FILES | {METADATA_FILENAME}).difference(names))
    if missing or unexpected:
        msg = f"Dataset metadata package is incomplete or inconsistent: missing={missing}; unexpected={unexpected}."
        raise ValueError(msg)

    metadata = _load_json(root / METADATA_FILENAME, label="dataset metadata")
    _require_exact_keys(
        metadata,
        {"schema_kind", "schema_version", "dataset_id", "scientific_identity", "artifacts", "operational_provenance"},
        label="Dataset metadata",
    )
    if metadata["schema_kind"] != METADATA_SCHEMA_KIND:
        msg = "Unsupported dataset metadata schema kind."
        raise ValueError(msg)
    _require_schema_version(
        metadata["schema_version"],
        expected=METADATA_SCHEMA_VERSION,
        label="Dataset metadata schema_version",
    )
    if metadata["dataset_id"] != dataset_identity.dataset_id:
        msg = "Dataset metadata ID does not match the loaded final dataset."
        raise ValueError(msg)

    scientific = metadata["scientific_identity"]
    if not isinstance(scientific, dict):
        msg = "Dataset metadata scientific_identity must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(
        scientific,
        {
            "dataset_schema_version",
            "dataset_fingerprint",
            "task_id",
            "task_contract_digest",
            "source_batch_id",
            "generated_batch_identity_sha256",
            "sample_count",
            "spatial_shape",
            "tensors",
        },
        label="Dataset metadata scientific_identity",
    )
    _require_schema_version(
        scientific["dataset_schema_version"],
        expected=TRAINING_DATASET_SCHEMA_VERSION,
        label="Dataset metadata dataset_schema_version",
    )
    sample_count = _require_positive_int(scientific["sample_count"], label="Dataset metadata sample_count")
    spatial_shape = _require_spatial_shape(scientific["spatial_shape"], label="Dataset metadata spatial_shape")
    _require_sha256(scientific["dataset_fingerprint"], label="Dataset metadata dataset_fingerprint")
    _require_sha256(scientific["task_contract_digest"], label="Dataset metadata task_contract_digest")
    _validate_generated_batch_digest(
        scientific["generated_batch_identity_sha256"],
        dataset_identity=dataset_identity,
    )
    if (
        scientific["dataset_fingerprint"] != dataset_identity.fingerprint
        or scientific["task_id"] != dataset_identity.task
        or scientific["task_contract_digest"] != dataset_identity.task_contract_digest
        or scientific["source_batch_id"] != dataset_identity.dataset_id
        or sample_count != dataset_identity.sample_count
        or spatial_shape != list(dataset_identity.spatial_shape)
    ):
        msg = "Dataset metadata scientific identity does not match the loaded final dataset."
        raise ValueError(msg)
    _validate_tensor_contract(scientific["tensors"], dataset_identity=dataset_identity)

    artifacts = metadata["artifacts"]
    if not isinstance(artifacts, dict):
        msg = "Dataset metadata artifacts must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(artifacts, {"dataset", "snapshots"}, label="Dataset metadata artifacts")
    dataset_artifact = _validate_file_artifact(artifacts["dataset"], label="Dataset metadata dataset artifact")
    if dataset_artifact["filename"] != f"{dataset_identity.dataset_id}.pt":
        msg = "Dataset metadata artifact filename does not match dataset identity."
        raise ValueError(msg)
    if dataset_path is not None:
        resolved_dataset_path = Path(dataset_path)
        if not resolved_dataset_path.is_file() or resolved_dataset_path.is_symlink():
            msg = f"Final dataset artifact is not a regular file: {resolved_dataset_path}"
            raise FileNotFoundError(msg)
        if (
            resolved_dataset_path.name != dataset_artifact["filename"]
            or resolved_dataset_path.stat().st_size != dataset_artifact["size_bytes"]
            or common.serialization.file_sha256(resolved_dataset_path) != dataset_artifact["sha256"]
        ):
            msg = "Final dataset artifact does not match dataset metadata SHA-256 and size."
            raise ValueError(msg)
    snapshots = _validate_snapshot_artifacts(root, names=names, snapshots=artifacts["snapshots"])

    manifest = _validate_source_manifest_snapshot(_load_json(root / SOURCE_MANIFEST_FILENAME, label="source manifest snapshot"))
    manifest_sha256 = snapshots[SOURCE_MANIFEST_FILENAME]["sha256"]
    if manifest["batch_name"] != scientific["source_batch_id"] or manifest["intended_case_ids"] != list(dataset_identity.sample_ids):
        msg = "Source manifest snapshot membership does not match the final dataset."
        raise ValueError(msg)
    configuration = manifest["configuration"]
    expected_sample_csv_sha256 = _require_sha256(
        configuration.get("sample_sha256"),
        label="source manifest configuration.sample_sha256",
    )
    if snapshots[SOURCE_SAMPLE_CSV_FILENAME]["sha256"] != expected_sample_csv_sha256:
        msg = "Parameter-sample CSV snapshot does not match the source manifest SHA-256."
        raise ValueError(msg)
    validate_source_sample_semantics(
        (root / SOURCE_SAMPLE_CSV_FILENAME).read_bytes(),
        (root / SOURCE_SAMPLE_JSON_FILENAME).read_bytes(),
        source_manifest=manifest,
        dataset_identity=dataset_identity,
        source_manifest_sha256=manifest_sha256,
    )

    operational = metadata["operational_provenance"]
    if not isinstance(operational, dict):
        msg = "Dataset metadata operational_provenance must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(
        operational,
        {"builder_module", "publication_method", "source_manifest_sha256", "timing"},
        label="Dataset metadata operational_provenance",
    )
    if operational["builder_module"] != BUILDER_MODULE or operational["publication_method"] != PUBLICATION_METHOD:
        msg = "Dataset metadata builder or publication identity is unsupported."
        raise ValueError(msg)
    if operational["source_manifest_sha256"] != manifest_sha256:
        msg = "Dataset metadata source-manifest binding does not match its snapshot artifact."
        raise ValueError(msg)
    timing_summary = _validate_timing_summary(
        operational["timing"],
        intended_count=len(manifest["intended_case_ids"]),
        label="Dataset metadata timing",
    )
    timing_path = root / COMSOL_TIMING_FILENAME
    timing = _load_json(timing_path, label="COMSOL timing snapshot") if timing_path.is_file() else None
    if timing is None and timing_summary["measured_case_count"] != 0:
        msg = "Metadata declares measured COMSOL timing but has no timing snapshot."
        raise ValueError(msg)
    if timing is not None:
        timing = validate_comsol_timing_snapshot(
            timing,
            batch_name=scientific["source_batch_id"],
            manifest_sha256=manifest_sha256,
            intended_case_ids=manifest["intended_case_ids"],
        )
        if len(timing["cases"]) != timing_summary["measured_case_count"]:
            msg = "COMSOL timing snapshot count disagrees with metadata coverage."
            raise ValueError(msg)
    return DatasetMetadata(root, metadata, manifest, timing)


def load_dataset_metadata(
    dataset_id: str,
    *,
    dataset_identity: DatasetIdentity,
    metadata_root: Path | str | None = None,
    dataset_path: Path | str | None = None,
) -> DatasetMetadata:
    """Resolve and validate one model-training metadata package."""
    directory = common.paths.resolve_dataset_metadata_dir(dataset_id, metadata_root=metadata_root)
    return validate_dataset_metadata_directory(
        directory,
        dataset_identity=dataset_identity,
        dataset_path=dataset_path,
    )
