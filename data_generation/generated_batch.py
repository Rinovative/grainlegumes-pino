"""
===============================================================================
generated_batch.py
===============================================================================
Admit and materialize completed generated COMSOL batches without mutation.

Responsibilities:
  - Read terminal manifests, parameter samples, raw cases, solutions, and timing
  - Validate hashes, fields, grids, tensor symmetry, units, and finite numerics
  - Materialize canonical generated-case tensors and bounded summaries

Design principles:
  - Source files remain read-only throughout admission
  - One strict admission path serves EDA and final-dataset construction
  - Manifest order and TaskSpec scientific contracts are authoritative

This module does NOT:
  - Build or publish final training datasets or metadata packages
  - Run COMSOL, MATLAB, simulation, training, or artifact generation
  - Create splits, normalizers, dataloaders, runs, or checkpoints
===============================================================================
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from src import common, datasets, domain
from tqdm import tqdm

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import FieldSpec, TaskSpec

__all__ = ["load_generated_batch"]

COMSOL_PREFIX = "br."
_COMSOL_HEADER_ITEM = re.compile(r"^(?P<name>.*?)(?:\s+\((?P<unit>[^()]*)\))?$")
_PERMEABILITY_SYMMETRY_RTOL = 1e-6
_RAW_SOLUTION_RTOL = 1e-12
_RAW_SOLUTION_SCALE_ATOL = 1e-12
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
_EXPECTED_SOLUTION_HEADER = (
    ("x", "m"),
    ("y", "m"),
    ("kappaxx", "m^2"),
    ("kappayx", "m^2"),
    ("kappaxy", "m^2"),
    ("kappayy", "m^2"),
    ("int4(x,y)", "1"),
    ("int5(x,y)", "Pa"),
    ("p", "Pa"),
    ("u", "m/s"),
    ("v", "m/s"),
    ("U", "m/s"),
)
_RAW_METADATA_KEYS = frozenset({"export", "fields_present", "generator", "geometry", "paths", "timestamp"})
_RAW_EXPORT_KEYS = frozenset({"columns", "delimiter", "file_base"})
_RAW_FIELDS_PRESENT_KEYS = frozenset({"porosity", "pressure_bc", "tensor"})
_RAW_GEOMETRY_KEYS = frozenset({"Lx", "Ly", "dx", "dy", "nx", "ny", "res"})
_RAW_PATH_KEYS = frozenset({"csv", "json"})
_GENERATOR_MAPPING_KEYS = {
    "generator": frozenset({"bc", "permeability", "porosity", "structure"}),
    "generator.structure": frozenset({"parameters", "statistics"}),
    "generator.structure.statistics": frozenset({"noise", "structure"}),
    "generator.structure.statistics.noise": frozenset({"l2_norm", "max_abs"}),
    "generator.structure.statistics.structure": frozenset({"z", "z_bg", "z_noises"}),
    "generator.structure.statistics.structure.z": frozenset({"max", "mean", "min", "std"}),
    "generator.structure.statistics.structure.z_bg": frozenset({"mean", "std"}),
    "generator.structure.statistics.structure.z_noises": frozenset({"rms"}),
    "generator.structure.parameters": frozenset({"background", "noise", "rng_state", "seed"}),
    "generator.structure.parameters.rng_state": frozenset({"Seed", "State", "Type"}),
    "generator.structure.parameters.background": frozenset({"anisotropy", "base_len_rel", "coupling", "ms_weight", "smooth_len_rel"}),
    "generator.structure.parameters.noise": frozenset({"bias", "granularity", "level"}),
    "generator.permeability": frozenset({"parameters", "statistics"}),
    "generator.permeability.statistics": frozenset({"kappa", "tensor"}),
    "generator.permeability.statistics.kappa": frozenset({"max", "mean", "min", "std"}),
    "generator.permeability.statistics.tensor": frozenset({"det", "trace"}),
    "generator.permeability.statistics.tensor.trace": frozenset({"mean"}),
    "generator.permeability.statistics.tensor.det": frozenset({"mean"}),
    "generator.permeability.parameters": frozenset({"orientation", "permeability", "tensor"}),
    "generator.permeability.parameters.permeability": frozenset({"k_mean", "s_logn", "var_rel"}),
    "generator.permeability.parameters.tensor": frozenset({"a_gamma", "a_max", "tensor_strength"}),
    "generator.permeability.parameters.orientation": frozenset({"theta_jitter", "theta_smooth_rel"}),
    "generator.porosity": frozenset({"parameters", "statistics"}),
    "generator.porosity.statistics": frozenset({"eps"}),
    "generator.porosity.statistics.eps": frozenset({"max", "mean", "min", "std"}),
    "generator.porosity.parameters": frozenset({"A_mat", "A_rel", "eps_max_global", "eps_min_global", "eps_ref", "eps_smooth_rel", "texture_amp"}),
    "generator.bc": frozenset({"parameters", "statistics"}),
    "generator.bc.statistics": frozenset({"p_inlet"}),
    "generator.bc.statistics.p_inlet": frozenset({"max", "mean", "min", "std"}),
    "generator.bc.parameters": frozenset({"a_gauss", "a_lin", "a_sin", "f_sin", "gauss_jitter", "k_gauss", "p_inlet_mean", "sigma_gauss"}),
}


def _source_column(field: FieldSpec) -> str:
    """Return the exact source column declared for a task field."""
    return field.source_name or field.name


def _read_exact_width_csv(
    path: Path,
    *,
    expected_columns: list[str],
    comment: str | None = None,
) -> pd.DataFrame:
    """Read a headerless delimited file and reject every width mismatch."""
    try:
        dataframe = pd.read_csv(
            path,
            comment=comment,
            sep=";",
            header=None,
            index_col=False,
            skip_blank_lines=True,
            on_bad_lines="error",
        )
    except pd.errors.ParserError as error:
        msg = f"Delimited source has inconsistent row widths: {path}"
        raise ValueError(msg) from error
    if dataframe.shape[1] != len(expected_columns):
        msg = f"Delimited source must contain exactly {len(expected_columns)} columns, got {dataframe.shape[1]}: {path}"
        raise ValueError(msg)
    dataframe.columns = expected_columns
    return dataframe.copy()


def _load_case_sources(csv_path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one case and enforce the exact unit-bearing COMSOL solution header."""
    with meta_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        msg = f"Case metadata must contain a JSON object: {meta_path}"
        raise TypeError(msg)

    comment_lines: list[str] = []
    with csv_path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("%"):
                comment_lines.append(stripped[1:].strip())
            elif stripped:
                break
    length_units = [line.split(",", 1)[1].strip() for line in comment_lines if line.startswith("Length unit,")]
    if length_units != ["m"]:
        msg = f"COMSOL CSV must declare exactly one '% Length unit,m' line: {csv_path}"
        raise ValueError(msg)
    header_lines = [line for line in comment_lines if ";" in line]
    if len(header_lines) != 1:
        msg = f"COMSOL CSV must contain exactly one semicolon-delimited header: {csv_path}"
        raise ValueError(msg)
    parsed: list[tuple[str, str]] = []
    for item in header_lines[0].split(";"):
        match = _COMSOL_HEADER_ITEM.fullmatch(item.strip())
        if match is None:
            msg = f"Malformed COMSOL CSV header item {item!r}: {csv_path}"
            raise ValueError(msg)
        original_name = match.group("name").strip()
        name = original_name.removeprefix(COMSOL_PREFIX)
        unit = match.group("unit") or ("m" if name in {"x", "y"} else "")
        parsed.append((name, unit))
    if tuple(parsed) != _EXPECTED_SOLUTION_HEADER:
        msg = f"COMSOL CSV field/unit header does not match the steady-flow source contract: {parsed}."
        raise ValueError(msg)
    header = [name for name, _unit in parsed]
    if len(header) != len(set(header)):
        msg = f"COMSOL CSV header contains duplicate fields: {header}"
        raise ValueError(msg)
    dataframe = _read_exact_width_csv(
        csv_path,
        expected_columns=header,
        comment="%",
    )
    noncanonical = sorted(_NONCANONICAL_FIELDS.intersection(dataframe.columns))
    if noncanonical:
        msg = f"Noncanonical learned field name(s) are invalid: {noncanonical}."
        raise ValueError(msg)
    return dataframe, metadata


def require_exact_mapping_keys(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
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


def require_sha256(value: Any, *, label: str, allow_empty: bool = False) -> str:
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
    No coercion or additional keys are accepted.

    Raises
    ------
    TypeError
        If a field has the wrong exact JSON type.
    ValueError
        If keys, ranges, digests, or the template basename violate the schema.

    """
    configuration = require_exact_mapping_keys(value, _MANIFEST_CONFIGURATION_KEYS, label="Batch manifest configuration")
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
    require_sha256(configuration["sample_sha256"], label="Batch manifest configuration.sample_sha256")
    require_sha256(configuration["template_sha256"], label="Batch manifest configuration.template_sha256")
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


def sha256_file(path: Path) -> str:
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
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        msg = f"Batch manifest file integrity failure: SHA-256 mismatch for {label} at {path}."
        raise RuntimeError(msg)


def _verified_source_file_identity(path: Path, expected_digest: str, *, label: str) -> dict[str, Any]:
    """Return portable file identity only after matching a manifest digest."""
    if not path.is_file():
        msg = f"Batch manifest file integrity failure: missing {label} at {path}."
        raise RuntimeError(msg)
    identity = datasets.identity.source_file_identity(path)
    if identity["sha256"] != expected_digest:
        msg = f"Batch manifest file integrity failure: SHA-256 mismatch for {label} at {path}."
        raise RuntimeError(msg)
    return identity


def _verify_case_sources_after_read(
    record: dict[str, Any],
    *,
    raw_dir: Path,
    processed_dir: Path,
    save_model: bool,
) -> dict[str, dict[str, Any]]:
    """Rebind files after parsing so long builds cannot mix source versions."""
    case_id = record["case_id"]
    files = record["files"]
    identities = {
        "raw_export": _verified_source_file_identity(
            raw_dir / f"{case_id}.csv",
            files["raw_csv_sha256"],
            label=f"{case_id} raw CSV",
        ),
        "solution_export": _verified_source_file_identity(
            processed_dir / f"{case_id}_sol.csv",
            files["solution_csv_sha256"],
            label=f"{case_id} solution CSV",
        ),
    }
    _verified_source_file_identity(
        raw_dir / f"{case_id}.json",
        files["raw_json_sha256"],
        label=f"{case_id} raw JSON",
    )
    model_path = processed_dir / f"{case_id}_sol.mph"
    if save_model:
        identities["solution_model"] = _verified_source_file_identity(
            model_path,
            files["solution_model_sha256"],
            label=f"{case_id} solved model",
        )
    elif model_path.exists():
        msg = f"Batch manifest file integrity failure: unexpected solved model at {model_path}."
        raise RuntimeError(msg)
    return identities


def assert_generation_batch_idle(raw_dir: Path) -> None:
    """Reject source admission while private MATLAB progress is present."""
    progress_path = raw_dir / "batch_progress.json"
    if progress_path.exists() or progress_path.is_symlink():
        msg = (
            "Generated batch has active or interrupted COMSOL progress; "
            f"resume or finish the MATLAB batch before dataset construction: {progress_path}"
        )
        raise RuntimeError(msg)


def assert_generation_snapshot_current(
    raw_dir: Path,
    manifest_path: Path,
    manifest_snapshot: bytes,
) -> None:
    """Fence final staging against a producer that advanced after admission."""
    assert_generation_batch_idle(raw_dir)
    try:
        current_snapshot = manifest_path.read_bytes()
    except OSError as error:
        msg = f"Could not revalidate the admitted generation manifest: {manifest_path}"
        raise RuntimeError(msg) from error
    if current_snapshot != manifest_snapshot:
        msg = "Generation manifest changed while the final dataset was being built."
        raise RuntimeError(msg)
    assert_generation_batch_idle(raw_dir)


def load_batch_manifest(raw_dir: Path, processed_dir: Path, *, batch_name: str) -> dict[str, Any]:
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
        If private batch progress exists, the manifest is non-terminal, or an
        authoritative file is absent or changed.

    """
    assert_generation_batch_idle(raw_dir)
    path = raw_dir / "batch_manifest.json"
    if not path.is_file():
        msg = f"Generated batch is missing its terminal completion manifest: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    manifest = require_exact_mapping_keys(loaded, _MANIFEST_KEYS, label="Batch manifest")
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
    manifest["cases"] = records
    if len(records) != len(intended):
        msg = "Batch manifest case records must exactly match intended_case_ids."
        raise RuntimeError(msg)

    save_model = configuration["save_model"]
    for index, (case_id, record_value) in enumerate(zip(intended, records, strict=True)):
        record = require_exact_mapping_keys(record_value, _MANIFEST_RECORD_KEYS, label=f"Batch manifest cases[{index}]")
        if record["case_id"] != case_id or record["status"] != "complete" or record["stage"] != "simulation" or record["message"] != "":
            msg = "Batch manifest complete case records must exactly match intended_case_ids and the terminal record schema."
            raise RuntimeError(msg)
        files = require_exact_mapping_keys(record["files"], _MANIFEST_FILE_KEYS, label=f"Batch manifest cases[{index}].files")
        raw_csv_digest = require_sha256(files["raw_csv_sha256"], label=f"Batch manifest cases[{index}].files.raw_csv_sha256")
        raw_json_digest = require_sha256(files["raw_json_sha256"], label=f"Batch manifest cases[{index}].files.raw_json_sha256")
        solution_csv_digest = require_sha256(files["solution_csv_sha256"], label=f"Batch manifest cases[{index}].files.solution_csv_sha256")
        model_digest = require_sha256(
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


def load_generation_metadata(
    meta_dir: Path,
    batch_name: str,
    manifest: dict[str, Any],
) -> tuple[Path, Path, pd.DataFrame, dict[str, Any], dict[str, Any], bytes, bytes]:
    """Validate parameter snapshots through the shared public metadata boundary."""
    csv_path = meta_dir / f"{batch_name}.csv"
    json_path = meta_dir / f"{batch_name}.json"
    if not csv_path.is_file() or not json_path.is_file():
        msg = f"Generated batch is missing parameter-sample metadata under {meta_dir}."
        raise FileNotFoundError(msg)
    csv_snapshot = csv_path.read_bytes()
    json_snapshot = json_path.read_bytes()
    semantics = datasets.metadata.validate_source_sample_semantics(
        csv_snapshot,
        json_snapshot,
        source_manifest=manifest,
    )
    sample_frame = pd.read_csv(io.BytesIO(csv_snapshot), sep=";")
    sample_frame.index = list(semantics.case_ids)
    return (
        csv_path,
        json_path,
        sample_frame,
        semantics.sample_json,
        semantics.sampling,
        csv_snapshot,
        json_snapshot,
    )


def validate_exact_source_membership(
    raw_dir: Path,
    processed_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Reject every missing or unexpected generated case artifact."""
    case_ids = manifest["intended_case_ids"]
    intended = set(case_ids)
    save_model = manifest["configuration"]["save_model"]
    actual = {
        "raw": {path.stem for path in raw_dir.glob("case_*.csv")},
        "metadata": {path.stem for path in raw_dir.glob("case_*.json")},
        "solutions": {path.stem.removesuffix("_sol") for path in processed_dir.glob("case_*_sol.csv")},
        "models": {path.stem.removesuffix("_sol") for path in processed_dir.glob("case_*_sol.mph")},
    }
    expected = {
        "raw": intended,
        "metadata": intended,
        "solutions": intended,
        "models": intended if save_model else set(),
    }
    failures = {
        name: {
            "missing": sorted(expected[name].difference(actual[name])),
            "unexpected": sorted(actual[name].difference(expected[name])),
        }
        for name in expected
        if actual[name] != expected[name]
    }
    if failures:
        msg = f"Generated batch does not exactly match terminal manifest membership: {failures}."
        raise RuntimeError(msg)


def _python_scalar(value: Any) -> Any:
    """Convert a NumPy/pandas scalar to a JSON-compatible Python scalar."""
    return value.item() if isinstance(value, np.generic) else value


def _validate_generator_metadata(value: Any) -> dict[str, Any]:
    """Validate the exact nested generator mapping and finite JSON leaves."""
    generator = require_exact_mapping_keys(value, _GENERATOR_MAPPING_KEYS["generator"], label="Raw metadata generator")
    for path, expected_keys in _GENERATOR_MAPPING_KEYS.items():
        if path == "generator":
            continue
        current: Any = generator
        for component in path.split(".")[1:]:
            if not isinstance(current, dict) or component not in current:
                msg = f"Raw metadata {path} is missing."
                raise ValueError(msg)
            current = current[component]
        require_exact_mapping_keys(current, expected_keys, label=f"Raw metadata {path}")

    def validate_leaf(item: Any, *, label: str) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                validate_leaf(nested, label=f"{label}.{key}")
            return
        if isinstance(item, list):
            if not item:
                msg = f"{label} must not be an empty sequence."
                raise ValueError(msg)
            for index, nested in enumerate(item):
                validate_leaf(nested, label=f"{label}[{index}]")
            return
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            if not math.isfinite(float(item)):
                msg = f"{label} must be finite."
                raise ValueError(msg)
            return
        if isinstance(item, str) and item:
            return
        msg = f"{label} must be a finite JSON scientific value."
        raise TypeError(msg)

    validate_leaf(generator, label="generator")
    return generator


def _normalize_case_metadata(
    metadata: dict[str, Any],
    *,
    case_id: str,
    sample_row: pd.Series,
    manifest: dict[str, Any],
    metadata_path: Path,
    include_generation_details: bool = False,
) -> dict[str, Any]:
    """Validate raw metadata and retain only path-independent values."""
    metadata = require_exact_mapping_keys(metadata, _RAW_METADATA_KEYS, label=f"Raw case metadata {metadata_path}")
    export = require_exact_mapping_keys(metadata["export"], _RAW_EXPORT_KEYS, label="Raw metadata export")
    if export["columns"] != _MANIFEST_FIELD_SCHEMA["input_columns"] or export["delimiter"] != ";" or export["file_base"] != case_id:
        msg = f"Case metadata export contract does not match {case_id!r}: {metadata_path}"
        raise ValueError(msg)
    fields_present = require_exact_mapping_keys(
        metadata["fields_present"],
        _RAW_FIELDS_PRESENT_KEYS,
        label="Raw metadata fields_present",
    )
    if any(value is not True for value in fields_present.values()):
        msg = f"Case metadata must declare every generated field present: {metadata_path}"
        raise ValueError(msg)
    geometry = require_exact_mapping_keys(metadata["geometry"], _RAW_GEOMETRY_KEYS, label="Raw metadata geometry")
    nx = geometry["nx"]
    ny = geometry["ny"]
    if (
        isinstance(nx, bool)
        or not isinstance(nx, int)
        or isinstance(ny, bool)
        or not isinstance(ny, int)
        or nx < _MIN_AXIS_POINTS
        or ny < _MIN_AXIS_POINTS
    ):
        msg = f"Case geometry nx/ny must be integers of at least {_MIN_AXIS_POINTS}: {metadata_path}"
        raise ValueError(msg)
    for key in ("Lx", "Ly", "dx", "dy", "res"):
        value = geometry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            msg = f"Case geometry {key} must be finite and positive: {metadata_path}"
            raise ValueError(msg)
    configuration = manifest["configuration"]
    for key, expected in (("Lx", configuration["Lx"]), ("Ly", configuration["Ly"]), ("res", configuration["res"])):
        if not math.isclose(float(geometry[key]), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            msg = f"Case geometry {key} does not match the batch manifest: {metadata_path}"
            raise ValueError(msg)
    if not math.isclose(float(geometry["dx"]), float(geometry["res"]), rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(
        float(geometry["dy"]),
        float(geometry["res"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        msg = f"Case geometry dx/dy must equal res: {metadata_path}"
        raise ValueError(msg)
    if not math.isclose((nx - 1) * float(geometry["dx"]), float(geometry["Lx"]), rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(
        (ny - 1) * float(geometry["dy"]),
        float(geometry["Ly"]),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        msg = f"Case geometry dimensions do not match nx/ny and spacing: {metadata_path}"
        raise ValueError(msg)
    paths = require_exact_mapping_keys(metadata["paths"], _RAW_PATH_KEYS, label="Raw metadata paths")
    for key, suffix in (("csv", f"{case_id}.csv"), ("json", f"{case_id}.json")):
        value = paths[key]
        if not isinstance(value, str) or not value or value.replace(chr(92), "/").rsplit("/", 1)[-1] != suffix:
            msg = f"Raw metadata paths.{key} does not name {suffix}: {metadata_path}"
            raise ValueError(msg)
    if not isinstance(metadata["timestamp"], str) or not metadata["timestamp"]:
        msg = f"Raw metadata timestamp must be non-empty text: {metadata_path}"
        raise ValueError(msg)
    generator = _validate_generator_metadata(metadata["generator"])
    normalized = {
        "case_id": case_id,
        "geometry": geometry,
        "parameters": {name: _python_scalar(value) for name, value in sample_row.items() if name != "case_id"},
    }
    if include_generation_details:
        normalized.update(
            {
                "export": export,
                "fields_present": fields_present,
                "generator": generator,
            }
        )
    datasets.identity.canonical_metadata_identity(normalized)
    return normalized


def _load_raw_export(raw_csv_path: Path, *, spatial_shape: tuple[int, int]) -> pd.DataFrame:
    """Load the headerless seven-column generated input export."""
    frame = _read_exact_width_csv(
        raw_csv_path,
        expected_columns=_MANIFEST_FIELD_SCHEMA["input_columns"],
    )
    return _canonicalize_cartesian_grid(frame, spatial_shape=spatial_shape)


def _validate_raw_solution_agreement(
    raw_frame: pd.DataFrame,
    solution_frame: pd.DataFrame,
    *,
    spatial_shape: tuple[int, int],
) -> None:
    """Require agreement up to scale-aware COMSOL interpolation roundoff."""
    comparisons = {
        "x": "x",
        "y": "y",
        "Kxx": "kappaxx",
        "Kxy": "kappaxy",
        "Kyy": "kappayy",
        "eps": "int4(x,y)",
        "p_bc": "int5(x,y)",
    }
    raw_values = {raw_name: _numeric_field(raw_frame, raw_name, spatial_shape=spatial_shape) for raw_name in comparisons}
    solution_values = {
        raw_name: _numeric_field(solution_frame, solution_name, spatial_shape=spatial_shape) for raw_name, solution_name in comparisons.items()
    }
    permeability_scale = max(float(np.max(np.abs(raw_values[name]))) for name in ("Kxx", "Kxy", "Kyy"))
    permeability_scale = max(
        permeability_scale,
        *(float(np.max(np.abs(solution_values[name]))) for name in ("Kxx", "Kxy", "Kyy")),
    )
    scale_floors = {
        "x": 1.0,
        "y": 1.0,
        "Kxx": permeability_scale,
        "Kxy": permeability_scale,
        "Kyy": permeability_scale,
        "eps": 1.0,
        "p_bc": 1.0,
    }
    for raw_name, solution_name in comparisons.items():
        raw_field = raw_values[raw_name]
        solution_field = solution_values[raw_name]
        field_scale = max(
            scale_floors[raw_name],
            float(np.max(np.abs(raw_field))),
            float(np.max(np.abs(solution_field))),
        )
        absolute_tolerance = _RAW_SOLUTION_SCALE_ATOL * field_scale
        if not np.allclose(
            raw_field,
            solution_field,
            rtol=_RAW_SOLUTION_RTOL,
            atol=absolute_tolerance,
        ):
            maximum_error = float(np.max(np.abs(raw_field - solution_field)))
            msg = (
                f"Raw input field {raw_name!r} disagrees with COMSOL solution field {solution_name!r}: "
                f"max_abs_error={maximum_error:.6g}, atol={absolute_tolerance:.6g}."
            )
            raise ValueError(msg)
    first = _numeric_field(solution_frame, "kappayx", spatial_shape=spatial_shape)
    second = _numeric_field(solution_frame, "kappaxy", spatial_shape=spatial_shape)
    if not np.allclose(first, second, rtol=_PERMEABILITY_SYMMETRY_RTOL, atol=0.0):
        msg = "COMSOL symmetric permeability cross-component exports disagree."
        raise ValueError(msg)


def load_timing_snapshot(
    processed_dir: Path,
    *,
    batch_name: str,
    manifest_sha256: str,
    intended_case_ids: list[str],
) -> tuple[bytes | None, dict[str, Any] | None, dict[str, Any]]:
    """Validate and snapshot optional operational timing with partial coverage."""
    path = processed_dir / datasets.metadata.COMSOL_TIMING_FILENAME
    if not path.is_file():
        return (
            None,
            None,
            {
                "status": "missing",
                "measured_case_count": 0,
                "intended_case_count": len(intended_case_ids),
            },
        )
    try:
        timing_snapshot = path.read_bytes()
        value = json.loads(timing_snapshot.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Could not load COMSOL solve timing: {path}"
        raise ValueError(msg) from error
    if not isinstance(value, dict):
        msg = f"COMSOL solve timing must contain a JSON object: {path}"
        raise TypeError(msg)
    validated = datasets.metadata.validate_comsol_timing_snapshot(
        value,
        batch_name=batch_name,
        manifest_sha256=manifest_sha256,
        intended_case_ids=intended_case_ids,
    )
    measured_count = len(validated["cases"])
    if measured_count == 0:
        status = "missing"
    elif measured_count == len(intended_case_ids):
        status = "complete"
    else:
        status = "partial"
    return (
        timing_snapshot,
        validated,
        {
            "status": status,
            "measured_case_count": measured_count,
            "intended_case_count": len(intended_case_ids),
        },
    )


def source_provenance(manifest: dict[str, Any], *, manifest_sha256: str, sample_json_sha256: str) -> dict[str, Any]:
    """Retain exact operational source hashes outside scientific identity."""
    return {
        "batch_manifest_sha256": manifest_sha256,
        "source_sample_csv_sha256": manifest["configuration"]["sample_sha256"],
        "source_sample_json_sha256": sample_json_sha256,
        "cases": [{"case_id": record["case_id"], **record["files"]} for record in manifest["cases"]],
    }


def _load_generated_case(
    case_id: str,
    *,
    task: TaskSpec,
    manifest: dict[str, Any],
    manifest_record: dict[str, Any],
    sample_row: pd.Series,
    raw_dir: Path,
    processed_dir: Path,
    include_generation_details: bool,
) -> tuple[tuple[int, int], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, Any]]]:
    """Interpret one manifest-bound generated case through the canonical reader."""
    raw_csv_path = raw_dir / f"{case_id}.csv"
    metadata_path = raw_dir / f"{case_id}.json"
    solution_path = processed_dir / f"{case_id}_sol.csv"
    solution_frame, raw_metadata = _load_case_sources(solution_path, metadata_path)
    normalized_metadata = _normalize_case_metadata(
        raw_metadata,
        case_id=case_id,
        sample_row=sample_row,
        manifest=manifest,
        metadata_path=metadata_path,
        include_generation_details=include_generation_details,
    )
    geometry = normalized_metadata["geometry"]
    spatial_shape = (geometry["ny"], geometry["nx"])
    solution_frame = _canonicalize_cartesian_grid(solution_frame, spatial_shape=spatial_shape)
    raw_frame = _load_raw_export(raw_csv_path, spatial_shape=spatial_shape)
    _validate_raw_solution_agreement(raw_frame, solution_frame, spatial_shape=spatial_shape)
    input_fields, output_fields = _build_fields(solution_frame, task=task, spatial_shape=spatial_shape)
    verified_files = _verify_case_sources_after_read(
        manifest_record,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        save_model=manifest["configuration"]["save_model"],
    )
    return spatial_shape, input_fields, output_fields, normalized_metadata, verified_files


def load_generated_batch(
    batch_name: str,
    *,
    task_id: str = "steady_flow",
    generated_data_root: Path | str | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """
    Load a validated generated-batch prefix without publishing training data.

    This is the canonical read-only interpretation used by EDA. It enforces the
    same manifest, hash, metadata, unit, grid, and physical-field contracts as
    the direct final-dataset builder while never resolving model-training data.
    """
    task = domain.tasks.registry.get_task(task_id)
    if task.id != "steady_flow":
        msg = f"The generated COMSOL reader supports only the current steady_flow task, got {task.id!r}."
        raise ValueError(msg)
    batch_name = common.paths.validate_logical_name(batch_name, label="batch_name")
    if max_cases is not None:
        if isinstance(max_cases, bool) or not isinstance(max_cases, int):
            msg = f"max_cases must be a positive integer or None, got {max_cases!r}."
            raise TypeError(msg)
        if max_cases <= 0:
            msg = f"max_cases must be positive, got {max_cases}."
            raise ValueError(msg)
    generated_root = Path(generated_data_root).expanduser() if generated_data_root is not None else common.paths.get_generated_data_root()
    meta_dir = generated_root / "meta"
    raw_dir = generated_root / "raw" / batch_name
    processed_dir = generated_root / "processed" / batch_name
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = load_batch_manifest(raw_dir, processed_dir, batch_name=batch_name)
    validate_exact_source_membership(raw_dir, processed_dir, manifest)
    (
        _sample_csv_path,
        _sample_json_path,
        sample_frame,
        _sample_json,
        portable_sampling,
        _sample_csv_snapshot,
        _sample_json_snapshot,
    ) = load_generation_metadata(
        meta_dir,
        batch_name,
        manifest,
    )
    all_case_ids = list(manifest["intended_case_ids"])
    if len(all_case_ids) != manifest["configuration"]["N"]:
        msg = "A complete batch manifest must contain exactly configuration.N intended cases."
        raise ValueError(msg)
    selected_case_ids = all_case_ids if max_cases is None else all_case_ids[:max_cases]
    generated_identity = datasets.identity.build_generated_batch_identity(
        manifest,
        sampling=portable_sampling,
    )
    rows: list[dict[str, Any]] = []
    records_by_id = {record["case_id"]: record for record in manifest["cases"]}
    iterator = tqdm(
        selected_case_ids,
        desc=f"Loading {batch_name}",
        unit="case",
        disable=not show_progress,
    )
    for case_id in iterator:
        _spatial_shape, input_fields, output_fields, normalized_metadata, _verified_files = _load_generated_case(
            case_id,
            task=task,
            manifest=manifest,
            manifest_record=records_by_id[case_id],
            sample_row=sample_frame.loc[case_id],
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            include_generation_details=True,
        )
        rows.append(
            {
                **input_fields,
                **output_fields,
                "meta": normalized_metadata,
            }
        )
    return {
        "batch_name": batch_name,
        "generated_data_root": generated_root,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "generated_batch_identity": generated_identity,
        "sample_ids": selected_case_ids,
        "available_case_count": len(all_case_ids),
        "rows": rows,
        "task": task,
    }


def interpret_generated_case(
    case_id: str,
    *,
    task: TaskSpec,
    manifest: dict[str, Any],
    manifest_record: dict[str, Any],
    sample_row: pd.Series,
    raw_dir: Path,
    processed_dir: Path,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Interpret and fingerprint one manifest-bound generated case in memory."""
    spatial_shape, input_fields, output_fields, normalized_metadata, verified_files = _load_generated_case(
        case_id,
        task=task,
        manifest=manifest,
        manifest_record=manifest_record,
        sample_row=sample_row,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        include_generation_details=False,
    )
    case_inputs = torch.stack([torch.from_numpy(input_fields[name]) for name in task.input_names])
    case_outputs = torch.stack([torch.from_numpy(output_fields[name]) for name in task.output_names])
    stable_source = {
        "case_id": case_id,
        **verified_files,
        "raw_metadata": datasets.identity.canonical_metadata_identity(normalized_metadata),
        "sample_parameters": datasets.identity.canonical_metadata_identity(normalized_metadata["parameters"]),
    }
    fingerprint = datasets.identity.compute_case_fingerprint(
        task=task,
        case_id=case_id,
        source_identity=stable_source,
        source_metadata=normalized_metadata,
        inputs=case_inputs,
        outputs=case_outputs,
    )
    return spatial_shape, case_inputs, case_outputs, normalized_metadata, stable_source, fingerprint
