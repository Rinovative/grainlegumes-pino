"""
===============================================================================
build_batch_dataset.py
===============================================================================
Build strict task-aware case payloads from a completed COMSOL batch.

Responsibilities:
  - Verify the terminal producer manifest and every bound source-file digest
  - Canonicalize uniform Cartesian exports in task-declared field order
  - Convert permeability components to stored representations and validate physics
  - Publish fingerprinted case payloads atomically beneath the case-data root

Design principles:
  - TaskSpec alone owns learned field names, channel order, and source mappings
  - Source identity includes authoritative exports, metadata, and the batch manifest
  - A complete case directory appears only after every intended case validates

This module does NOT:
  - Run COMSOL, repair incomplete producer batches, or merge cases for training
  - Consume processed COMSOL solve timing, which is operational rather than scientific provenance
  - Overwrite an existing case target or choose fields from mapping insertion order
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from src import common, datasets, domain
from tqdm import tqdm

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import FieldSpec, TaskSpec

COMSOL_PREFIX = "br."
_COMSOL_UNIT_SUFFIX = re.compile(r"\s+\([^)]*\)\s*$")
_PERMEABILITY_SYMMETRY_RTOL = 1e-6
_SYMMETRY_EPSILON_FACTOR = 16.0
_MIN_AXIS_POINTS = 2
_GRID_UNIFORM_RTOL = 1e-8
_BATCH_MANIFEST_SCHEMA_VERSION = 1
_BATCH_MANIFEST_SCHEMA_KIND = "comsol_batch_manifest"
_MAX_EXACT_MANIFEST_INTEGER = 2**53
_MAX_RANDOM_SEED = 2**32 - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CASE_ID_PATTERN = re.compile(r"case_[0-9]{4,}")
_MANIFEST_KEYS = frozenset({"schema_kind", "schema_version", "batch_name", "status", "configuration", "field_schema", "intended_case_ids", "cases"})
_MANIFEST_CONFIGURATION_KEYS = frozenset(
    {"method", "variation", "N", "seed", "Lx", "Ly", "res", "save_model", "sample_sha256", "template_name", "template_sha256"}
)
_MANIFEST_RECORD_KEYS = frozenset({"case_id", "status", "stage", "message", "files"})
_MANIFEST_FILE_KEYS = frozenset({"raw_csv_sha256", "raw_json_sha256", "solution_csv_sha256", "solution_model_sha256"})
_MANIFEST_FIELD_SCHEMA = {
    "input_columns": ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"],
    "solution_columns": ["x", "y", "kappaxx", "kappayx", "kappaxy", "kappayy", "eps", "p_bc", "p", "u", "v", "U"],
}
_NONCANONICAL_FIELDS = frozenset({"phi", "pbc"})


def _source_column(field: FieldSpec) -> str:
    """Return the exact source column declared for a task field."""
    return field.source_name or field.name


def _load_case_sources(csv_path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load one COMSOL solution table and its reproducibility metadata.

    The final commented CSV header is authoritative. Unit suffixes and the COMSOL
    prefix are removed before rejecting duplicate or retired field spellings.

    Raises
    ------
    TypeError
        If the metadata JSON is not an object.
    ValueError
        If the CSV has no commented header or normalization creates duplicates.

    """
    with meta_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        msg = f"Case metadata must contain a JSON object: {meta_path}"
        raise TypeError(msg)

    with csv_path.open(encoding="utf-8") as file:
        comment_lines = [line for line in file if line.strip().startswith("%")]
    if not comment_lines:
        msg = f"COMSOL CSV has no commented header declaration: {csv_path}"
        raise ValueError(msg)
    header = [_COMSOL_UNIT_SUFFIX.sub("", item.strip()).removeprefix(COMSOL_PREFIX) for item in comment_lines[-1][1:].split(";")]
    if len(header) != len(set(header)):
        msg = f"COMSOL CSV header contains duplicate fields after unit normalization: {header}"
        raise ValueError(msg)

    dataframe = pd.read_csv(
        csv_path,
        comment="%",
        sep=";",
        names=header,
        index_col=False,
        skip_blank_lines=True,
    ).copy()
    noncanonical = sorted(_NONCANONICAL_FIELDS.intersection(dataframe.columns))
    if noncanonical:
        msg = f"Noncanonical learned field name(s) are invalid: {noncanonical}."
        raise ValueError(msg)
    return dataframe, metadata


def _require_exact_mapping_keys(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    """
    Return a persisted mapping only when its key set is exact.

    Raises
    ------
    TypeError
        If ``value`` is not a dictionary.
    ValueError
        If required keys are missing or undeclared keys are present.

    """
    if not isinstance(value, dict):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    missing = sorted(expected.difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        msg = f"{label} keys do not match: missing={missing}; unexpected={unexpected}."
        raise ValueError(msg)
    return value


def _require_sha256(value: Any, *, label: str, allow_empty: bool = False) -> str:
    """
    Return one validated lowercase SHA-256 digest.

    An empty string is admitted only when ``allow_empty`` explicitly represents
    an artifact that the producer was configured not to save.

    Raises
    ------
    TypeError
        If the value is not a string.
    ValueError
        If a non-empty value is not exactly 64 lowercase hexadecimal characters.

    """
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str):
        msg = f"{label} must be a lowercase hexadecimal SHA-256 string."
        raise TypeError(msg)
    if _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be a 64-character lowercase hexadecimal SHA-256 digest."
        raise ValueError(msg)
    return value


def _require_manifest_real(
    configuration: dict[str, Any],
    key: str,
    *,
    positive: bool,
) -> float:
    """
    Return one finite manifest real in its declared sign domain.

    Boolean values are rejected even though ``bool`` is an ``int`` subclass.
    ``positive`` selects a strictly positive rather than non-negative domain.

    Raises
    ------
    TypeError
        If the field is boolean or not a real scalar.
    ValueError
        If the value is non-finite or outside the selected sign domain.

    """
    value = configuration[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"Batch manifest configuration.{key} must be a real number."
        raise TypeError(msg)
    numeric = float(value)
    if not math.isfinite(numeric) or (numeric <= 0 if positive else numeric < 0):
        domain = "positive" if positive else "non-negative"
        msg = f"Batch manifest configuration.{key} must be finite and {domain}."
        raise ValueError(msg)
    return numeric


def _validate_manifest_configuration(value: Any) -> dict[str, Any]:
    """
    Validate and return the exact production batch-configuration mapping.

    The contract binds sampling method and count, seed range, domain geometry,
    grid resolution, optional model publication, and producer/template digests.
    No coercion or compatibility keys are accepted.

    Raises
    ------
    TypeError
        If a field has the wrong exact JSON type.
    ValueError
        If keys, ranges, digests, or the template basename violate the schema.

    """
    configuration = _require_exact_mapping_keys(value, _MANIFEST_CONFIGURATION_KEYS, label="Batch manifest configuration")
    method = configuration["method"]
    if not isinstance(method, str) or method not in {"uniform", "lhs", "sobol"}:
        msg = "Batch manifest configuration.method must be one of 'uniform', 'lhs', or 'sobol'."
        raise ValueError(msg)
    count = configuration["N"]
    seed = configuration["seed"]
    if isinstance(count, bool) or not isinstance(count, int):
        msg = "Batch manifest configuration.N must be an integer."
        raise TypeError(msg)
    if not 1 <= count <= _MAX_EXACT_MANIFEST_INTEGER:
        msg = f"Batch manifest configuration.N must be in [1, {_MAX_EXACT_MANIFEST_INTEGER}]."
        raise ValueError(msg)
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "Batch manifest configuration.seed must be an integer."
        raise TypeError(msg)
    if not 0 <= seed <= _MAX_RANDOM_SEED:
        msg = f"Batch manifest configuration.seed must be in [0, {_MAX_RANDOM_SEED}]."
        raise ValueError(msg)
    _require_manifest_real(configuration, "variation", positive=False)
    length_x = _require_manifest_real(configuration, "Lx", positive=True)
    length_y = _require_manifest_real(configuration, "Ly", positive=True)
    resolution = _require_manifest_real(configuration, "res", positive=True)
    if resolution > min(length_x, length_y):
        msg = "Batch manifest configuration.res cannot exceed the shorter domain length."
        raise ValueError(msg)
    if not isinstance(configuration["save_model"], bool):
        msg = "Batch manifest configuration.save_model must be boolean."
        raise TypeError(msg)
    _require_sha256(configuration["sample_sha256"], label="Batch manifest configuration.sample_sha256")
    _require_sha256(configuration["template_sha256"], label="Batch manifest configuration.template_sha256")
    template_name = configuration["template_name"]
    if (
        not isinstance(template_name, str)
        or not template_name
        or template_name != Path(template_name).name
        or "/" in template_name
        or "\\" in template_name
        or not template_name.endswith(".mph")
    ):
        msg = "Batch manifest configuration.template_name must be one basename ending in '.mph'."
        raise ValueError(msg)
    return configuration


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one authoritative source file."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_manifest_file(path: Path, expected_digest: str, *, label: str) -> None:
    """
    Require one authoritative producer file to match its manifest digest.

    Raises
    ------
    RuntimeError
        If the file is absent or its current bytes do not match ``expected_digest``.

    """
    if not path.is_file():
        msg = f"Batch manifest file integrity failure: missing {label} at {path}."
        raise RuntimeError(msg)
    actual_digest = _sha256_file(path)
    if actual_digest != expected_digest:
        msg = f"Batch manifest file integrity failure: SHA-256 mismatch for {label} at {path}."
        raise RuntimeError(msg)


def _load_batch_manifest(raw_dir: Path, processed_dir: Path, *, batch_name: str) -> dict[str, Any]:
    """
    Load and cryptographically validate one terminal producer manifest.

    Validation is fail-closed: the schema, batch identity, configuration, ordered
    intended membership, terminal case records, and every required file digest
    must agree. Solved-model presence follows the manifest's ``save_model`` flag.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Directory containing the manifest, raw exports, and metadata.
    processed_dir : pathlib.Path
        Directory containing solved exports, optional solved models, and the
        operational solve-timing sidecar excluded from scientific source identity.
    batch_name : str
        Logical batch identity that the manifest must declare exactly.

    Returns
    -------
    dict[str, Any]
        The validated manifest, preserving its declared case order.

    Raises
    ------
    FileNotFoundError
        If the terminal manifest is absent.
    TypeError
        If a persisted field has the wrong JSON container or scalar type.
    ValueError
        If the schema, identity, configuration, membership, or digest syntax is invalid.
    RuntimeError
        If the manifest is non-terminal or an authoritative file is absent or changed.

    """
    path = raw_dir / "batch_manifest.json"
    if not path.is_file():
        msg = f"Generated batch is missing its terminal completion manifest: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    manifest = _require_exact_mapping_keys(loaded, _MANIFEST_KEYS, label="Batch manifest")
    schema_version = manifest["schema_version"]
    if (
        not isinstance(manifest["schema_kind"], str)
        or manifest["schema_kind"] != _BATCH_MANIFEST_SCHEMA_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _BATCH_MANIFEST_SCHEMA_VERSION
    ):
        msg = f"Unsupported batch manifest schema: {path}"
        raise ValueError(msg)
    if not isinstance(manifest["batch_name"], str) or manifest["batch_name"] != batch_name:
        msg = f"Batch manifest identity {manifest['batch_name']!r} does not match {batch_name!r}."
        raise ValueError(msg)
    configuration = _validate_manifest_configuration(manifest["configuration"])
    field_schema = manifest["field_schema"]
    if not isinstance(field_schema, dict) or field_schema != _MANIFEST_FIELD_SCHEMA:
        msg = "Batch manifest field_schema must exactly match the maintained COMSOL producer contract."
        raise ValueError(msg)
    if not isinstance(manifest["status"], str) or manifest["status"] != "complete":
        msg = f"Batch manifest is not complete: status={manifest['status']!r}."
        raise RuntimeError(msg)

    intended = manifest["intended_case_ids"]
    if not isinstance(intended, list) or not all(isinstance(case_id, str) and _CASE_ID_PATTERN.fullmatch(case_id) for case_id in intended):
        msg = "Batch manifest intended_case_ids must be a list of canonical case identifiers."
        raise TypeError(msg)
    if len(intended) != len(set(intended)):
        msg = "Batch manifest intended_case_ids must be unique."
        raise ValueError(msg)
    if len(intended) > configuration["N"]:
        msg = "Batch manifest intended membership cannot exceed configuration.N."
        raise ValueError(msg)
    records = manifest["cases"]
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        msg = "Batch manifest cases must be a list of case-status mappings."
        raise TypeError(msg)
    if len(records) != len(intended):
        msg = "Batch manifest case records must exactly match intended_case_ids."
        raise RuntimeError(msg)

    save_model = configuration["save_model"]
    for index, (case_id, record_value) in enumerate(zip(intended, records, strict=True)):
        record = _require_exact_mapping_keys(record_value, _MANIFEST_RECORD_KEYS, label=f"Batch manifest cases[{index}]")
        if record["case_id"] != case_id or record["status"] != "complete" or record["stage"] != "simulation" or record["message"] != "":
            msg = "Batch manifest complete case records must exactly match intended_case_ids and the terminal record schema."
            raise RuntimeError(msg)
        files = _require_exact_mapping_keys(record["files"], _MANIFEST_FILE_KEYS, label=f"Batch manifest cases[{index}].files")
        raw_csv_digest = _require_sha256(files["raw_csv_sha256"], label=f"Batch manifest cases[{index}].files.raw_csv_sha256")
        raw_json_digest = _require_sha256(files["raw_json_sha256"], label=f"Batch manifest cases[{index}].files.raw_json_sha256")
        solution_csv_digest = _require_sha256(files["solution_csv_sha256"], label=f"Batch manifest cases[{index}].files.solution_csv_sha256")
        model_digest = _require_sha256(
            files["solution_model_sha256"],
            label=f"Batch manifest cases[{index}].files.solution_model_sha256",
            allow_empty=not save_model,
        )
        if save_model and not model_digest:
            msg = f"Batch manifest cases[{index}] must bind the configured solved model."
            raise ValueError(msg)
        if not save_model and model_digest:
            msg = f"Batch manifest cases[{index}] cannot bind a solved model when save_model is false."
            raise ValueError(msg)

        _verify_manifest_file(raw_dir / f"{case_id}.csv", raw_csv_digest, label=f"{case_id} raw CSV")
        _verify_manifest_file(raw_dir / f"{case_id}.json", raw_json_digest, label=f"{case_id} raw JSON")
        _verify_manifest_file(processed_dir / f"{case_id}_sol.csv", solution_csv_digest, label=f"{case_id} solution CSV")
        model_path = processed_dir / f"{case_id}_sol.mph"
        if save_model:
            _verify_manifest_file(model_path, model_digest, label=f"{case_id} solved model")
        elif model_path.exists():
            msg = f"Batch manifest file integrity failure: unexpected solved model at {model_path}."
            raise RuntimeError(msg)
    return manifest


def _validate_uniform_axis(values: np.ndarray, *, label: str) -> None:
    """
    Require a finite, strictly increasing, uniformly spaced coordinate axis.

    Uniformity uses a relative tolerance of ``_GRID_UNIFORM_RTOL`` plus a
    machine-precision absolute floor scaled to the coordinate magnitude.

    Raises
    ------
    ValueError
        If fewer than two points are present, ordering is not strict, or spacing
        is non-finite or non-uniform.

    """
    if values.size < _MIN_AXIS_POINTS:
        msg = f"{label}-coordinate axis must contain at least {_MIN_AXIS_POINTS} points."
        raise ValueError(msg)
    differences = np.diff(values)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        msg = f"{label}-coordinate axis must be finite and strictly increasing."
        raise ValueError(msg)
    mean_spacing = float(np.mean(differences))
    absolute_tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values))))
    if not np.allclose(differences, mean_spacing, rtol=_GRID_UNIFORM_RTOL, atol=absolute_tolerance):
        msg = f"{label}-coordinate axis must be uniform within relative tolerance {_GRID_UNIFORM_RTOL}."
        raise ValueError(msg)


def _numeric_field(
    dataframe: pd.DataFrame,
    column: str,
    *,
    spatial_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Return one finite real source column in canonical grid shape.

    Values are converted to float64 for validation and scientific transforms;
    their element count must exactly match ``spatial_shape``.

    Raises
    ------
    TypeError
        If the source dtype is non-numeric or complex.
    ValueError
        If the size is wrong or any value is non-finite.

    """
    values = dataframe[column].to_numpy()
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        msg = f"Source column {column!r} must contain real numeric values, got dtype {values.dtype}."
        raise TypeError(msg)
    expected_count = int(np.prod(spatial_shape))
    if values.size != expected_count:
        msg = f"Source column {column!r} has {values.size} values; expected {expected_count} for shape {spatial_shape}."
        raise ValueError(msg)
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        msg = f"Source column {column!r} contains non-finite values."
        raise ValueError(msg)
    return numeric.reshape(spatial_shape)


def _canonicalize_cartesian_grid(
    dataframe: pd.DataFrame,
    *,
    spatial_shape: tuple[int, int],
) -> pd.DataFrame:
    """
    Validate a complete Cartesian grid and return deterministic y/x row order.

    ``spatial_shape`` is ``(ny, nx)``. Duplicate coordinates, missing Cartesian
    products, non-uniform axes, or cardinalities inconsistent with metadata fail
    before any field tensor is built. Sorting is stable by y and then x.

    Raises
    ------
    KeyError
        If either coordinate column is absent.
    TypeError
        If coordinates are non-real or non-numeric.
    ValueError
        If coordinate values, spacing, uniqueness, or shape are invalid.

    """
    missing = [column for column in ("x", "y") if column not in dataframe.columns]
    if missing:
        msg = f"COMSOL export is missing coordinate column(s): {missing}."
        raise KeyError(msg)
    y_count, x_count = spatial_shape
    x_coordinates = _numeric_field(dataframe, "x", spatial_shape=spatial_shape).reshape(-1)
    y_coordinates = _numeric_field(dataframe, "y", spatial_shape=spatial_shape).reshape(-1)
    x_values = np.unique(x_coordinates)
    y_values = np.unique(y_coordinates)
    _validate_uniform_axis(x_values, label="x")
    _validate_uniform_axis(y_values, label="y")
    if x_values.size != x_count or y_values.size != y_count:
        msg = f"COMSOL coordinate cardinality does not match metadata geometry: x={x_values.size}/{x_count}, y={y_values.size}/{y_count}."
        raise ValueError(msg)
    coordinate_pairs = np.column_stack((x_coordinates, y_coordinates))
    if np.unique(coordinate_pairs, axis=0).shape[0] != coordinate_pairs.shape[0]:
        msg = "COMSOL coordinate grid contains duplicate (x, y) pairs."
        raise ValueError(msg)
    if coordinate_pairs.shape[0] != x_values.size * y_values.size:
        msg = "COMSOL coordinates do not form one complete Cartesian product."
        raise ValueError(msg)

    canonical = dataframe.copy()
    canonical["x"] = x_coordinates
    canonical["y"] = y_coordinates
    canonical = canonical.sort_values(["y", "x"], kind="mergesort").reset_index(drop=True)
    expected_x, expected_y = np.meshgrid(x_values, y_values, indexing="xy")
    if not np.array_equal(canonical["x"].to_numpy(), expected_x.reshape(-1)) or not np.array_equal(
        canonical["y"].to_numpy(),
        expected_y.reshape(-1),
    ):
        msg = "COMSOL coordinates do not cover the complete Cartesian product."
        raise ValueError(msg)
    return canonical


def _build_permeability_fields(
    dataframe: pd.DataFrame,
    *,
    task: TaskSpec,
    spatial_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """
    Build validated task-declared permeability representations.

    Symmetric off-diagonal COMSOL sources are averaged only after tolerance
    agreement. Every pointwise permeability tensor must be positive definite.
    Diagonal components are stored as ``log10(k_ii)``; cross components are
    stored as ``k_ij / sqrt(k_ii * k_jj)`` in task-declared field order.

    Returns
    -------
    dict[str, numpy.ndarray]
        Float64 stored-representation fields with ``spatial_shape``.

    Raises
    ------
    ValueError
        If sources are missing, symmetric exports disagree, a diagonal is not
        positive, or a pointwise tensor is not positive definite.

    """
    available = [column for column in dataframe.columns if column.startswith("kappa")]
    mapping = domain.permeability.resolve_internal_to_present_sources(available)
    expected = [field.name for field in task.inputs if field.role == "permeability"]
    missing = [name for name in expected if name not in mapping]
    if missing:
        msg = f"Missing task permeability source component(s): {missing}."
        raise ValueError(msg)

    raw_fields: dict[str, np.ndarray] = {}
    for name in expected:
        sources = mapping[name]
        tensors = [_numeric_field(dataframe, source, spatial_shape=spatial_shape) for source in sources]
        reference = tensors[0]
        for source, tensor in zip(sources[1:], tensors[1:], strict=True):
            magnitude = max(float(np.max(np.abs(reference))), float(np.max(np.abs(tensor))), np.finfo(np.float64).tiny)
            absolute_tolerance = _SYMMETRY_EPSILON_FACTOR * np.finfo(np.float64).eps * magnitude
            if not np.allclose(reference, tensor, rtol=_PERMEABILITY_SYMMETRY_RTOL, atol=absolute_tolerance):
                msg = f"Symmetric permeability sources {sources[0]!r} and {source!r} disagree for {name!r}."
                raise ValueError(msg)
        raw_fields[name] = np.mean(np.stack(tensors), axis=0)

    diagonal_names = [name for name in expected if name[1] == name[2]]
    for name in diagonal_names:
        if np.any(raw_fields[name] <= 0):
            msg = f"Permeability diagonal {name!r} must be strictly positive."
            raise ValueError(msg)

    axes = tuple(axis for axis in "xyz" if f"k{axis}{axis}" in raw_fields)
    axis_indices = {axis: index for index, axis in enumerate(axes)}
    permeability_tensor = np.zeros((*spatial_shape, len(axes), len(axes)), dtype=np.float64)
    for axis, index in axis_indices.items():
        permeability_tensor[..., index, index] = raw_fields[f"k{axis}{axis}"]
    for name, values in raw_fields.items():
        if name[1] == name[2]:
            continue
        first_axis, second_axis = name[1], name[2]
        if first_axis not in axis_indices or second_axis not in axis_indices:
            msg = f"Cross component {name!r} requires both task diagonal permeability components."
            raise ValueError(msg)
        first_index = axis_indices[first_axis]
        second_index = axis_indices[second_axis]
        permeability_tensor[..., first_index, second_index] = values
        permeability_tensor[..., second_index, first_index] = values
    if np.any(np.linalg.eigvalsh(permeability_tensor) <= 0):
        msg = "The symmetric permeability tensor must be positive definite at every grid point."
        raise ValueError(msg)

    fields = {name: np.log10(raw_fields[name]) for name in diagonal_names}
    for name in expected:
        if name[1] == name[2]:
            continue
        first_axis, second_axis = name[1], name[2]
        denominator = np.sqrt(raw_fields[f"k{first_axis}{first_axis}"] * raw_fields[f"k{second_axis}{second_axis}"])
        fields[name] = raw_fields[name] / denominator
    return fields


def _float32_field(value: np.ndarray, *, name: str) -> np.ndarray:
    """
    Convert one validated field to owned float32 storage.

    Raises
    ------
    ValueError
        If conversion overflows or otherwise produces a non-finite value.

    """
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(value, dtype=np.float32).copy()
    if not np.isfinite(converted).all():
        msg = f"Field {name!r} is non-finite after float32 conversion."
        raise ValueError(msg)
    return converted


def _build_fields(
    dataframe: pd.DataFrame,
    *,
    task: TaskSpec,
    spatial_shape: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Build exact finite input and output mappings for one task case.

    Field names, source mappings, and order come only from ``task``. Porosity is
    constrained to ``0 < eps <= 1``; permeability uses its declared stored
    representation; all returned arrays are finite owned float32 values.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]
        Ordered input and output mappings in the task contract's field order.

    Raises
    ------
    KeyError
        If a declared non-permeability source column is absent.
    ValueError
        If permeability, porosity, shape, finiteness, or float32 conversion fails.

    """
    input_fields: dict[str, np.ndarray] = {}
    permeability = _build_permeability_fields(
        dataframe,
        task=task,
        spatial_shape=spatial_shape,
    )
    for field in task.inputs:
        if field.role == "permeability":
            input_fields[field.name] = permeability[field.name]
            continue
        source = _source_column(field)
        if source not in dataframe.columns:
            msg = f"Missing required task input source column {source!r} for field {field.name!r}."
            raise KeyError(msg)
        values = _numeric_field(dataframe, source, spatial_shape=spatial_shape)
        if field.role == "porosity" and np.any((values <= 0) | (values > 1)):
            msg = f"Porosity field {field.name!r} must satisfy 0 < eps <= 1."
            raise ValueError(msg)
        input_fields[field.name] = values

    output_fields: dict[str, np.ndarray] = {}
    for field in task.outputs:
        source = _source_column(field)
        if source not in dataframe.columns:
            msg = f"Missing required task output source column {source!r} for field {field.name!r}."
            raise KeyError(msg)
        output_fields[field.name] = _numeric_field(dataframe, source, spatial_shape=spatial_shape)
    return (
        {name: _float32_field(value, name=name) for name, value in input_fields.items()},
        {name: _float32_field(value, name=name) for name, value in output_fields.items()},
    )


def build_batch_dataset(
    batch_name: str,
    verbose: bool = False,
    *,
    task_id: str = "steady_flow",
    generated_data_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    """
    Build strict case payloads from one complete generated COMSOL batch.

    Parameters
    ----------
    batch_name : str
        Logical batch and dataset identifier.
    verbose : bool, optional
        Show build progress and a first-case preview.
    task_id : str, optional
        Exact registered task identifier.
    generated_data_root : Path | str | None, optional
        Independent generated-data root.
    data_root : Path | str | None, optional
        Independent case-preparation root. Cases are published below its
        established ``raw`` stage.

    Returns
    -------
    dict[str, Any]
        Task identity, case count, published destination, ordered fields, and
        content fingerprints in manifest order.

    Raises
    ------
    FileNotFoundError
        If the terminal producer manifest is absent.
    FileExistsError
        If the authoritative ``cases`` target already exists or appears while
        publication is in progress.
    KeyError
        If a task-declared source column is absent.
    TypeError
        If persisted metadata, manifest fields, or source columns have invalid types.
    ValueError
        If the manifest, Cartesian grid, scientific fields, or case schema is invalid.
    RuntimeError
        If producer membership/files are incomplete, changed, or non-terminal.

    Notes
    -----
    Cases are written beneath a private staging directory and the directory is
    renamed to ``cases`` only after every intended case succeeds. On failure the
    staging directory is removed; an existing authoritative target is never replaced.

    """
    task = domain.tasks.registry.get_task(task_id)
    processed_dir = common.paths.resolve_generated_batch_dir(
        batch_name,
        stage="processed",
        generated_data_root=generated_data_root,
    )
    raw_dir = common.paths.resolve_generated_batch_dir(
        batch_name,
        stage="raw",
        generated_data_root=generated_data_root,
    )
    batch_dir = common.paths.resolve_case_dataset_dir(batch_name, data_root=data_root)
    cases_dir = batch_dir / "cases"
    batch_manifest = _load_batch_manifest(raw_dir, processed_dir, batch_name=batch_name)
    intended_case_ids = batch_manifest["intended_case_ids"]

    raw_csv_files = sorted(raw_dir.glob("case_*.csv"))
    json_files = sorted(raw_dir.glob("case_*.json"))
    csv_files = sorted(processed_dir.glob("case_*_sol.csv"))
    model_files = sorted(processed_dir.glob("case_*_sol.mph"))
    raw_csv_names = {path.stem for path in raw_csv_files}
    json_names = {path.stem for path in json_files}
    csv_names = {path.stem.removesuffix("_sol") for path in csv_files}
    model_names = {path.stem.removesuffix("_sol") for path in model_files}
    intended_names = set(intended_case_ids)
    expected_model_names = intended_names if batch_manifest["configuration"]["save_model"] else set()
    missing_raw = sorted(intended_names.difference(raw_csv_names))
    missing_solutions = sorted(intended_names.difference(csv_names))
    missing_metadata = sorted(intended_names.difference(json_names))
    missing_models = sorted(expected_model_names.difference(model_names))
    unexpected_raw = sorted(raw_csv_names.difference(intended_names))
    unexpected_solutions = sorted(csv_names.difference(intended_names))
    unexpected_metadata = sorted(json_names.difference(intended_names))
    unexpected_models = sorted(model_names.difference(expected_model_names))
    if any(
        (
            missing_raw,
            missing_solutions,
            missing_metadata,
            missing_models,
            unexpected_raw,
            unexpected_solutions,
            unexpected_metadata,
            unexpected_models,
        )
    ):
        msg = (
            f"Generated batch {batch_name!r} does not match its terminal manifest: "
            f"missing raw={missing_raw}; missing solutions={missing_solutions}; "
            f"missing metadata={missing_metadata}; missing models={missing_models}; "
            f"unexpected raw={unexpected_raw}; unexpected solutions={unexpected_solutions}; "
            f"unexpected metadata={unexpected_metadata}; unexpected models={unexpected_models}."
        )
        raise RuntimeError(msg)
    case_ids = list(intended_case_ids)
    if not case_ids:
        msg = f"No complete generated cases found for {batch_name!r}."
        raise RuntimeError(msg)
    if cases_dir.exists() or cases_dir.is_symlink():
        msg = f"Refusing to overwrite existing strict case target: {cases_dir}"
        raise FileExistsError(msg)

    batch_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(dir=batch_dir, prefix=".cases.", suffix=".tmp"))
    reference_shape: tuple[int, ...] | None = None
    fingerprints: list[str] = []
    try:
        for case_id in tqdm(
            case_ids,
            desc=f"Building {batch_name}",
            unit="case",
            disable=not verbose,
        ):
            csv_path = processed_dir / f"{case_id}_sol.csv"
            raw_csv_path = raw_dir / f"{case_id}.csv"
            metadata_path = raw_dir / f"{case_id}.json"
            dataframe, metadata = _load_case_sources(csv_path, metadata_path)
            geometry = metadata.get("geometry")
            if not isinstance(geometry, dict):
                msg = f"Case metadata is missing a geometry mapping: {metadata_path}"
                raise TypeError(msg)
            nx = geometry.get("nx")
            ny = geometry.get("ny")
            if isinstance(nx, bool) or not isinstance(nx, int) or isinstance(ny, bool) or not isinstance(ny, int):
                msg = f"Case geometry nx/ny must be integers: {metadata_path}"
                raise TypeError(msg)
            if nx < _MIN_AXIS_POINTS or ny < _MIN_AXIS_POINTS:
                msg = f"Case geometry nx/ny must each be at least {_MIN_AXIS_POINTS}: {metadata_path}"
                raise ValueError(msg)
            spatial_shape = (ny, nx)
            dataframe = _canonicalize_cartesian_grid(
                dataframe,
                spatial_shape=spatial_shape,
            )
            if reference_shape is None:
                reference_shape = spatial_shape
            elif spatial_shape != reference_shape:
                msg = f"Inconsistent case shape for {case_id!r}: {spatial_shape} != {reference_shape}."
                raise ValueError(msg)

            input_fields, output_fields = _build_fields(
                dataframe,
                task=task,
                spatial_shape=spatial_shape,
            )
            identity_metadata = {key: value for key, value in metadata.items() if key not in {"paths", "timestamp"}}
            source_identity = {
                "raw_export": datasets.identity.source_file_identity(raw_csv_path),
                "raw_metadata": datasets.identity.canonical_metadata_identity(
                    identity_metadata,
                ),
                "solution_export": datasets.identity.source_file_identity(csv_path),
                "batch_manifest": datasets.identity.canonical_metadata_identity(batch_manifest),
            }
            case_payload = datasets.identity.build_case_payload(
                task=task,
                case_id=case_id,
                input_fields=input_fields,
                output_fields=output_fields,
                source_identity=source_identity,
                source_metadata=metadata,
            )
            common.serialization.atomic_torch_save(
                case_payload,
                staging_dir / f"{case_id}.pt",
            )
            fingerprints.append(case_payload["dataset_fingerprint"])

            if verbose and len(fingerprints) == 1:
                print(f"Input fields: {list(task.input_names)}")
                print(f"Output fields: {list(task.output_names)}")
                print(f"Spatial shape: {spatial_shape}")
                print(f"First case fingerprint: {fingerprints[0]}")

        if cases_dir.exists() or cases_dir.is_symlink():
            msg = f"Strict case target appeared during build: {cases_dir}"
            raise FileExistsError(msg)
        staging_dir.replace(cases_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    return {
        "batch_name": batch_name,
        "task": task.id,
        "n_cases": len(case_ids),
        "cases_dir": cases_dir,
        "input_fields": list(task.input_names),
        "output_fields": list(task.output_names),
        "case_fingerprints": fingerprints,
    }
