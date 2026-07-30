"""
Build one final training dataset directly from a completed COMSOL batch.

The module owns the only maintained bridge from generated scientific sources to
model-training data. It validates the terminal producer contract, interprets one
case at a time, builds in preallocated tensors, and publishes the final dataset
with a self-contained validated metadata snapshot. No per-case PyTorch payloads
are persisted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from src import common, datasets, domain
from tqdm import tqdm

if TYPE_CHECKING:
    from src.datasets.dataset_identity import DatasetIdentity
    from src.datasets.dataset_metadata import DatasetMetadata
    from src.domain.tasks.domain_task_spec import FieldSpec, TaskSpec

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
_PUBLICATION_TRANSACTION_SCHEMA_KIND = "training_dataset_publication_transaction"
_PUBLICATION_TRANSACTION_SCHEMA_VERSION = 1
_PUBLICATION_TRANSACTION_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_id",
        "phase",
        "staging_root",
        "dataset_sha256",
        "dataset_size",
        "metadata_inventory_sha256",
    }
)
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
_SAMPLE_JSON_KEYS = frozenset({"meta", "n_cases"})
_SAMPLE_META_KEYS = frozenset({"method", "variation", "N", "seed", "base", "param_names", "timestamp"})
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
    No coercion or additional keys are accepted.

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


def _assert_generation_batch_idle(raw_dir: Path) -> None:
    """Reject source admission while private MATLAB progress is present."""
    progress_path = raw_dir / "batch_progress.json"
    if progress_path.exists() or progress_path.is_symlink():
        msg = (
            "Generated batch has active or interrupted COMSOL progress; "
            f"resume or finish the MATLAB batch before dataset construction: {progress_path}"
        )
        raise RuntimeError(msg)


def _assert_generation_snapshot_current(
    raw_dir: Path,
    manifest_path: Path,
    manifest_snapshot: bytes,
) -> None:
    """Fence final staging against a producer that advanced after admission."""
    _assert_generation_batch_idle(raw_dir)
    try:
        current_snapshot = manifest_path.read_bytes()
    except OSError as error:
        msg = f"Could not revalidate the admitted generation manifest: {manifest_path}"
        raise RuntimeError(msg) from error
    if current_snapshot != manifest_snapshot:
        msg = "Generation manifest changed while the final dataset was being built."
        raise RuntimeError(msg)
    _assert_generation_batch_idle(raw_dir)


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
        If private batch progress exists, the manifest is non-terminal, or an
        authoritative file is absent or changed.

    """
    _assert_generation_batch_idle(raw_dir)
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
    manifest["cases"] = records
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


def _load_generation_metadata(
    meta_dir: Path,
    batch_name: str,
    manifest: dict[str, Any],
) -> tuple[Path, Path, pd.DataFrame, dict[str, Any], dict[str, Any], bytes, bytes]:
    """Validate and snapshot parameter metadata into manifest-aligned rows."""
    csv_path = meta_dir / f"{batch_name}.csv"
    json_path = meta_dir / f"{batch_name}.json"
    if not csv_path.is_file() or not json_path.is_file():
        msg = f"Generated batch is missing parameter-sample metadata under {meta_dir}."
        raise FileNotFoundError(msg)
    csv_snapshot = csv_path.read_bytes()
    json_snapshot = json_path.read_bytes()
    expected_sample_sha = manifest["configuration"]["sample_sha256"]
    if hashlib.sha256(csv_snapshot).hexdigest() != expected_sample_sha:
        msg = f"Parameter-sample CSV SHA-256 does not match the batch manifest: {csv_path}"
        raise RuntimeError(msg)
    try:
        sample_json = json.loads(json_snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Parameter-sample JSON is invalid: {json_path}"
        raise ValueError(msg) from error
    sample_json = _require_exact_mapping_keys(sample_json, _SAMPLE_JSON_KEYS, label="Parameter-sample JSON")
    sample_meta = _require_exact_mapping_keys(sample_json["meta"], _SAMPLE_META_KEYS, label="Parameter-sample JSON meta")
    configuration = manifest["configuration"]
    for key in ("method", "N", "seed"):
        if sample_meta[key] != configuration[key]:
            msg = f"Parameter-sample JSON meta.{key} does not match the batch manifest."
            raise ValueError(msg)
    if not math.isclose(float(sample_meta["variation"]), float(configuration["variation"]), rel_tol=0.0, abs_tol=0.0):
        msg = "Parameter-sample JSON meta.variation does not match the batch manifest."
        raise ValueError(msg)
    if sample_json["n_cases"] != configuration["N"]:
        msg = "Parameter-sample JSON n_cases does not match the batch manifest."
        raise ValueError(msg)
    param_names = sample_meta["param_names"]
    if not isinstance(param_names, list) or not param_names or not all(isinstance(name, str) and name for name in param_names):
        msg = "Parameter-sample JSON meta.param_names must be a non-empty list of names."
        raise TypeError(msg)
    if len(param_names) != len(set(param_names)):
        msg = "Parameter-sample JSON meta.param_names must be unique."
        raise ValueError(msg)
    if not isinstance(sample_meta["base"], dict) or not isinstance(sample_meta["timestamp"], str):
        msg = "Parameter-sample JSON meta.base/timestamp have invalid types."
        raise TypeError(msg)
    sample_frame = pd.read_csv(io.BytesIO(csv_snapshot), sep=";")
    if list(sample_frame.columns) != ["case_id", *param_names]:
        msg = "Parameter-sample CSV columns do not match meta.param_names in exact order."
        raise ValueError(msg)
    if len(sample_frame) != configuration["N"]:
        msg = "Parameter-sample CSV row count does not match manifest configuration.N."
        raise ValueError(msg)
    numeric = sample_frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        msg = "Parameter-sample CSV must contain only finite numeric values."
        raise ValueError(msg)
    raw_case_ids = numeric["case_id"].to_numpy(dtype=np.float64)
    if not np.equal(raw_case_ids, np.floor(raw_case_ids)).all():
        msg = "Parameter-sample CSV case_id values must be integers."
        raise ValueError(msg)
    case_ids = [f"case_{int(value):04d}" for value in raw_case_ids]
    if case_ids != manifest["intended_case_ids"]:
        msg = "Parameter-sample CSV membership/order does not match the terminal manifest."
        raise ValueError(msg)
    numeric = numeric.copy()
    numeric.index = case_ids
    portable_sampling = {key: value for key, value in sample_meta.items() if key != "timestamp"}
    return csv_path, json_path, numeric, sample_json, portable_sampling, csv_snapshot, json_snapshot


def _validate_exact_source_membership(
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
    generator = _require_exact_mapping_keys(value, _GENERATOR_MAPPING_KEYS["generator"], label="Raw metadata generator")
    for path, expected_keys in _GENERATOR_MAPPING_KEYS.items():
        if path == "generator":
            continue
        current: Any = generator
        for component in path.split(".")[1:]:
            if not isinstance(current, dict) or component not in current:
                msg = f"Raw metadata {path} is missing."
                raise ValueError(msg)
            current = current[component]
        _require_exact_mapping_keys(current, expected_keys, label=f"Raw metadata {path}")

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
    metadata = _require_exact_mapping_keys(metadata, _RAW_METADATA_KEYS, label=f"Raw case metadata {metadata_path}")
    export = _require_exact_mapping_keys(metadata["export"], _RAW_EXPORT_KEYS, label="Raw metadata export")
    if export["columns"] != _MANIFEST_FIELD_SCHEMA["input_columns"] or export["delimiter"] != ";" or export["file_base"] != case_id:
        msg = f"Case metadata export contract does not match {case_id!r}: {metadata_path}"
        raise ValueError(msg)
    fields_present = _require_exact_mapping_keys(
        metadata["fields_present"],
        _RAW_FIELDS_PRESENT_KEYS,
        label="Raw metadata fields_present",
    )
    if any(value is not True for value in fields_present.values()):
        msg = f"Case metadata must declare every generated field present: {metadata_path}"
        raise ValueError(msg)
    geometry = _require_exact_mapping_keys(metadata["geometry"], _RAW_GEOMETRY_KEYS, label="Raw metadata geometry")
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
    paths = _require_exact_mapping_keys(metadata["paths"], _RAW_PATH_KEYS, label="Raw metadata paths")
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


def _timing_snapshot(
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


def _generated_batch_identity(
    manifest: dict[str, Any],
    *,
    portable_sampling: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the stable scientific identity of one generated batch."""
    scientific_records = []
    for record in manifest["cases"]:
        files = record["files"]
        scientific_records.append(
            {
                "case_id": record["case_id"],
                "raw_csv_sha256": files["raw_csv_sha256"],
                "solution_csv_sha256": files["solution_csv_sha256"],
                "solution_model_sha256": files["solution_model_sha256"],
            }
        )
    scientific_configuration = {key: value for key, value in manifest["configuration"].items() if key != "sample_sha256"}
    content = {
        "schema_version": manifest["schema_version"],
        "batch_name": manifest["batch_name"],
        "configuration": scientific_configuration,
        "field_schema": manifest["field_schema"],
        "intended_case_ids": manifest["intended_case_ids"],
        "scientific_case_sources": scientific_records,
        "sampling": portable_sampling,
    }
    digest = common.serialization.canonical_json_sha256(content)
    identity = dict(content)
    identity["batch_manifest_identity_sha256"] = digest
    return identity, digest


def _source_provenance(manifest: dict[str, Any], *, manifest_sha256: str, sample_json_sha256: str) -> dict[str, Any]:
    """Retain exact operational source hashes outside scientific identity."""
    return {
        "batch_manifest_sha256": manifest_sha256,
        "source_sample_csv_sha256": manifest["configuration"]["sample_sha256"],
        "source_sample_json_sha256": sample_json_sha256,
        "cases": [{"case_id": record["case_id"], **record["files"]} for record in manifest["cases"]],
    }


def _metadata_inventory_entry(path: Path, *, required: bool, role: str) -> dict[str, Any]:
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "required": required,
        "role": role,
    }


def _stage_metadata_package(
    destination: Path,
    *,
    dataset_identity: datasets.identity.DatasetIdentity,
    task: TaskSpec,
    manifest_snapshot: bytes,
    manifest_sha256: str,
    manifest_identity_sha256: str,
    sample_csv_snapshot: bytes,
    sample_json_snapshot: bytes,
    sample_csv_sha256: str,
    sample_json_sha256: str,
    timing_snapshot: bytes | None,
    timing_summary: dict[str, Any],
) -> None:
    """Stage one coherent set of validated small model-training snapshots."""
    destination.mkdir(parents=True)
    snapshots = {
        datasets.metadata.SOURCE_MANIFEST_FILENAME: (manifest_snapshot, True, "validated_generation_manifest"),
        datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME: (sample_csv_snapshot, True, "validated_parameter_sample_csv"),
        datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME: (sample_json_snapshot, True, "validated_parameter_sample_json"),
    }
    if timing_snapshot is not None:
        snapshots[datasets.metadata.COMSOL_TIMING_FILENAME] = (timing_snapshot, False, "validated_operational_comsol_timing")
    for filename, (snapshot, _required, _role) in snapshots.items():
        common.serialization.atomic_write_bytes(destination / filename, snapshot)
    if _sha256_file(destination / datasets.metadata.SOURCE_MANIFEST_FILENAME) != manifest_sha256:
        msg = "Staged generation manifest does not match its admitted snapshot."
        raise RuntimeError(msg)
    if _sha256_file(destination / datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME) != sample_csv_sha256:
        msg = "Staged parameter-sample CSV does not match its admitted snapshot."
        raise RuntimeError(msg)
    if _sha256_file(destination / datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME) != sample_json_sha256:
        msg = "Staged parameter-sample JSON does not match its admitted snapshot."
        raise RuntimeError(msg)
    source_batch = {
        "batch_name": dataset_identity.dataset_id,
        "batch_manifest_sha256": manifest_sha256,
        "batch_manifest_identity_sha256": manifest_identity_sha256,
    }
    provenance = {
        "schema_kind": datasets.metadata.PROVENANCE_SCHEMA_KIND,
        "schema_version": datasets.metadata.METADATA_SCHEMA_VERSION,
        "dataset_id": dataset_identity.dataset_id,
        "dataset_schema_version": datasets.identity.TRAINING_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_identity.fingerprint,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "source_batch": source_batch,
        "sample_count": dataset_identity.sample_count,
        "spatial_shape": list(dataset_identity.spatial_shape),
        "timing": timing_summary,
    }
    common.serialization.atomic_write_json(destination / datasets.metadata.PROVENANCE_FILENAME, provenance)
    roles = {filename: (required, role) for filename, (_snapshot, required, role) in snapshots.items()}
    roles[datasets.metadata.PROVENANCE_FILENAME] = (True, "normalized_dataset_provenance")
    files = {
        filename: _metadata_inventory_entry(destination / filename, required=required, role=role) for filename, (required, role) in roles.items()
    }
    inventory = {
        "schema_kind": datasets.metadata.INVENTORY_SCHEMA_KIND,
        "schema_version": datasets.metadata.METADATA_SCHEMA_VERSION,
        "dataset_id": dataset_identity.dataset_id,
        "dataset_fingerprint": dataset_identity.fingerprint,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "sample_count": dataset_identity.sample_count,
        "spatial_shape": list(dataset_identity.spatial_shape),
        "source_batch_name": dataset_identity.dataset_id,
        "source_manifest_sha256": manifest_sha256,
        "files": files,
        "timing": timing_summary,
    }
    common.serialization.atomic_write_json(destination / datasets.metadata.INVENTORY_FILENAME, inventory)


def load_generated_batch_for_eda(
    batch_name: str,
    *,
    task_id: str = "steady_flow",
    generated_data_root: Path | str | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """
    Load a validated generated-batch prefix without publishing training data.

    The reader reuses the direct builder's manifest, hash, metadata, unit, grid,
    and physical-field validation. It materializes only the requested prefix and
    never resolves or opens the model-training domain.
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
    manifest = _load_batch_manifest(raw_dir, processed_dir, batch_name=batch_name)
    _validate_exact_source_membership(raw_dir, processed_dir, manifest)
    (
        _sample_csv_path,
        _sample_json_path,
        sample_frame,
        _sample_json,
        portable_sampling,
        _sample_csv_snapshot,
        _sample_json_snapshot,
    ) = _load_generation_metadata(
        meta_dir,
        batch_name,
        manifest,
    )
    all_case_ids = list(manifest["intended_case_ids"])
    if len(all_case_ids) != manifest["configuration"]["N"]:
        msg = "A complete batch manifest must contain exactly configuration.N intended cases."
        raise ValueError(msg)
    selected_case_ids = all_case_ids if max_cases is None else all_case_ids[:max_cases]
    generated_identity, _manifest_identity_sha256 = _generated_batch_identity(
        manifest,
        portable_sampling=portable_sampling,
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
        raw_csv_path = raw_dir / f"{case_id}.csv"
        metadata_path = raw_dir / f"{case_id}.json"
        solution_path = processed_dir / f"{case_id}_sol.csv"
        solution_frame, raw_metadata = _load_case_sources(solution_path, metadata_path)
        normalized_metadata = _normalize_case_metadata(
            raw_metadata,
            case_id=case_id,
            sample_row=sample_frame.loc[case_id],
            manifest=manifest,
            metadata_path=metadata_path,
            include_generation_details=True,
        )
        geometry = normalized_metadata["geometry"]
        spatial_shape = (geometry["ny"], geometry["nx"])
        solution_frame = _canonicalize_cartesian_grid(solution_frame, spatial_shape=spatial_shape)
        raw_frame = _load_raw_export(raw_csv_path, spatial_shape=spatial_shape)
        _validate_raw_solution_agreement(raw_frame, solution_frame, spatial_shape=spatial_shape)
        input_fields, output_fields = _build_fields(solution_frame, task=task, spatial_shape=spatial_shape)
        _verify_case_sources_after_read(
            records_by_id[case_id],
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            save_model=manifest["configuration"]["save_model"],
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
        "manifest_sha256": _sha256_file(manifest_path),
        "generated_batch_identity": generated_identity,
        "sample_ids": selected_case_ids,
        "available_case_count": len(all_case_ids),
        "rows": rows,
        "task": task,
    }


def _interpret_generated_case(
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
    )
    geometry = normalized_metadata["geometry"]
    spatial_shape = (geometry["ny"], geometry["nx"])
    solution_frame = _canonicalize_cartesian_grid(solution_frame, spatial_shape=spatial_shape)
    raw_frame = _load_raw_export(raw_csv_path, spatial_shape=spatial_shape)
    _validate_raw_solution_agreement(raw_frame, solution_frame, spatial_shape=spatial_shape)
    input_fields, output_fields = _build_fields(solution_frame, task=task, spatial_shape=spatial_shape)
    case_inputs = torch.stack([torch.from_numpy(input_fields[name]) for name in task.input_names])
    case_outputs = torch.stack([torch.from_numpy(output_fields[name]) for name in task.output_names])
    verified_files = _verify_case_sources_after_read(
        manifest_record,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        save_model=manifest["configuration"]["save_model"],
    )
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


def _publication_transaction_path(training_processed_root: Path, dataset_id: str) -> Path:
    """Return the fixed recovery marker for one logical dataset publication."""
    return training_processed_root / ".transactions" / f"dataset-{dataset_id}.json"


def _publication_transaction_record(
    *,
    dataset_id: str,
    phase: str,
    staging_root: Path,
    dataset_sha256: str = "",
    dataset_size: int = 0,
    metadata_inventory_sha256: str = "",
) -> dict[str, Any]:
    """Build one exact operational transaction marker."""
    if phase not in {"building", "ready"}:
        msg = f"Unsupported dataset publication phase: {phase!r}."
        raise ValueError(msg)
    return {
        "schema_kind": _PUBLICATION_TRANSACTION_SCHEMA_KIND,
        "schema_version": _PUBLICATION_TRANSACTION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "phase": phase,
        "staging_root": str(staging_root.resolve(strict=False)),
        "dataset_sha256": dataset_sha256,
        "dataset_size": dataset_size,
        "metadata_inventory_sha256": metadata_inventory_sha256,
    }


def _load_publication_transaction(
    transaction_path: Path,
    *,
    training_processed_root: Path,
    dataset_id: str,
) -> tuple[dict[str, Any], Path]:
    """Load and constrain a recovery marker to this builder's staging area."""
    try:
        loaded = json.loads(transaction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Dataset publication transaction is unreadable: {transaction_path}"
        raise RuntimeError(msg) from error
    record = _require_exact_mapping_keys(loaded, _PUBLICATION_TRANSACTION_KEYS, label="Dataset publication transaction")
    if (
        not isinstance(record["schema_kind"], str)
        or record["schema_kind"] != _PUBLICATION_TRANSACTION_SCHEMA_KIND
        or isinstance(record["schema_version"], bool)
        or not isinstance(record["schema_version"], int)
        or record["schema_version"] != _PUBLICATION_TRANSACTION_SCHEMA_VERSION
        or not isinstance(record["dataset_id"], str)
        or record["dataset_id"] != dataset_id
        or not isinstance(record["phase"], str)
        or record["phase"] not in {"building", "ready"}
        or not isinstance(record["staging_root"], str)
        or not record["staging_root"]
        or isinstance(record["dataset_size"], bool)
        or not isinstance(record["dataset_size"], int)
        or record["dataset_size"] < 0
        or not isinstance(record["dataset_sha256"], str)
        or not isinstance(record["metadata_inventory_sha256"], str)
    ):
        msg = f"Dataset publication transaction has invalid identity or scalar fields: {transaction_path}"
        raise RuntimeError(msg)
    staging_root = Path(record["staging_root"])
    expected_parent = training_processed_root.resolve(strict=False)
    if (
        not staging_root.is_absolute()
        or staging_root.parent.resolve(strict=False) != expected_parent
        or not staging_root.name.startswith(f".{dataset_id}.dataset-build.")
        or not staging_root.name.endswith(".tmp")
        or staging_root.is_symlink()
    ):
        msg = f"Dataset publication transaction names an unsafe staging root: {staging_root}"
        raise RuntimeError(msg)
    if record["phase"] == "building":
        if record["dataset_sha256"] or record["dataset_size"] or record["metadata_inventory_sha256"]:
            msg = "Building publication transaction cannot claim completed staged content."
            raise RuntimeError(msg)
    else:
        _require_sha256(record["dataset_sha256"], label="Dataset publication transaction dataset_sha256")
        _require_sha256(
            record["metadata_inventory_sha256"],
            label="Dataset publication transaction metadata_inventory_sha256",
        )
        if record["dataset_size"] <= 0:
            msg = "Ready publication transaction dataset_size must be positive."
            raise RuntimeError(msg)
    return record, staging_root


def _single_publication_component(
    staged: Path,
    final: Path,
    *,
    label: str,
) -> tuple[Path, bool]:
    """Resolve exactly one staged-or-final transaction component."""
    for candidate in (staged, final):
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            msg = f"Dataset publication {label} target has an invalid filesystem type: {candidate}"
            raise RuntimeError(msg)
    present = [candidate for candidate in (staged, final) if candidate.is_dir()]
    if len(present) != 1:
        msg = f"Ready dataset publication must have exactly one staged-or-final {label} directory."
        raise RuntimeError(msg)
    return present[0], present[0] == final


def _recover_interrupted_publication(
    transaction_path: Path,
    *,
    training_processed_root: Path,
    destination_dir: Path,
    metadata_destination: Path,
    raw_dir: Path,
    dataset_id: str,
    task: TaskSpec,
) -> tuple[DatasetIdentity, DatasetMetadata] | None:
    """Discard an incomplete build or finish an exact validated ready publication."""
    if transaction_path.is_symlink():
        msg = f"Dataset publication marker cannot be a symlink: {transaction_path}"
        raise RuntimeError(msg)
    if not transaction_path.is_file():
        if transaction_path.exists():
            msg = f"Dataset publication marker is not a regular file: {transaction_path}"
            raise RuntimeError(msg)
        return None
    record, staging_root = _load_publication_transaction(
        transaction_path,
        training_processed_root=training_processed_root,
        dataset_id=dataset_id,
    )
    if record["phase"] == "building":
        if destination_dir.exists() or destination_dir.is_symlink() or metadata_destination.exists() or metadata_destination.is_symlink():
            msg = "Incomplete building transaction unexpectedly has an authoritative target."
            raise RuntimeError(msg)
        if staging_root.exists():
            shutil.rmtree(staging_root)
        transaction_path.unlink()
        return None

    staged_dataset_dir = staging_root / "raw" / dataset_id
    staged_metadata_dir = staging_root / "meta" / dataset_id
    dataset_dir, dataset_is_final = _single_publication_component(
        staged_dataset_dir,
        destination_dir,
        label="dataset",
    )
    metadata_dir, metadata_is_final = _single_publication_component(
        staged_metadata_dir,
        metadata_destination,
        label="metadata",
    )
    dataset_path = dataset_dir / f"{dataset_id}.pt"
    if not dataset_path.is_file() or dataset_path.is_symlink() or set(dataset_dir.iterdir()) != {dataset_path}:
        msg = f"Recovered dataset directory does not contain exactly one regular payload: {dataset_dir}"
        raise RuntimeError(msg)
    if dataset_path.stat().st_size != record["dataset_size"] or _sha256_file(dataset_path) != record["dataset_sha256"]:
        msg = "Recovered staged/final dataset does not match its ready transaction digest and size."
        raise RuntimeError(msg)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    try:
        dataset_identity = datasets.identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    finally:
        del payload
        gc.collect()
    if dataset_identity.dataset_id != dataset_id:
        msg = "Recovered dataset identity does not match its publication transaction."
        raise RuntimeError(msg)
    inventory_path = metadata_dir / datasets.metadata.INVENTORY_FILENAME
    if not inventory_path.is_file() or _sha256_file(inventory_path) != record["metadata_inventory_sha256"]:
        msg = "Recovered metadata inventory does not match its ready transaction digest."
        raise RuntimeError(msg)
    datasets.metadata.validate_dataset_metadata_directory(metadata_dir, dataset_identity=dataset_identity)
    if not (metadata_is_final and dataset_is_final):
        source_manifest_snapshot = (metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME).read_bytes()
        _assert_generation_snapshot_current(
            raw_dir,
            raw_dir / "batch_manifest.json",
            source_manifest_snapshot,
        )

    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    moved_metadata = False
    try:
        if not metadata_is_final:
            metadata_dir.replace(metadata_destination)
            moved_metadata = True
        if not dataset_is_final:
            dataset_dir.replace(destination_dir)
    except BaseException:
        if moved_metadata and metadata_destination.is_dir() and not staged_metadata_dir.exists():
            metadata_destination.replace(staged_metadata_dir)
        raise

    final_dataset_path = destination_dir / f"{dataset_id}.pt"
    if final_dataset_path.stat().st_size != record["dataset_size"] or _sha256_file(final_dataset_path) != record["dataset_sha256"]:
        msg = "Recovered final dataset changed during publication."
        raise RuntimeError(msg)
    package = datasets.metadata.validate_dataset_metadata_directory(
        metadata_destination,
        dataset_identity=dataset_identity,
    )
    transaction_path.unlink()
    if staging_root.exists():
        shutil.rmtree(staging_root)
    return dataset_identity, package


def build_batch_dataset(  # noqa: C901, PLR0912, PLR0915
    batch_name: str,
    verbose: bool = False,
    *,
    dataset_id: str | None = None,
    task_id: str = "steady_flow",
    generated_data_root: Path | str | None = None,
    model_training_data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one final dataset plus metadata package."""
    task = domain.tasks.registry.get_task(task_id)
    if task.id != "steady_flow":
        msg = f"The COMSOL batch builder supports only the current steady_flow task, got {task.id!r}."
        raise ValueError(msg)
    batch_name = common.paths.validate_logical_name(batch_name, label="batch_name")
    resolved_dataset_id = common.paths.validate_logical_name(dataset_id or batch_name, label="dataset_id")
    if resolved_dataset_id != batch_name:
        msg = "Current one-batch datasets must use the source batch name as dataset_id."
        raise ValueError(msg)
    generated_root = Path(generated_data_root).expanduser() if generated_data_root is not None else common.paths.get_generated_data_root()
    training_root = (
        Path(model_training_data_root).expanduser() if model_training_data_root is not None else common.paths.get_model_training_data_root()
    )
    meta_dir = generated_root / "meta"
    raw_dir = generated_root / "raw" / batch_name
    processed_dir = generated_root / "processed" / batch_name
    manifest_path = raw_dir / "batch_manifest.json"
    training_meta_root = training_root / "meta"
    training_raw_root = training_root / "raw"
    training_processed_root = training_root / "processed"
    destination_dir = training_raw_root / resolved_dataset_id
    destination = destination_dir / f"{resolved_dataset_id}.pt"
    metadata_destination = training_meta_root / resolved_dataset_id
    lock_path = training_processed_root / ".locks" / f"dataset-{resolved_dataset_id}.lock"
    transaction_path = _publication_transaction_path(training_processed_root, resolved_dataset_id)
    training_processed_root.mkdir(parents=True, exist_ok=True)

    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        _assert_generation_batch_idle(raw_dir)
        recovered = _recover_interrupted_publication(
            transaction_path,
            training_processed_root=training_processed_root,
            destination_dir=destination_dir,
            metadata_destination=metadata_destination,
            raw_dir=raw_dir,
            dataset_id=resolved_dataset_id,
            task=task,
        )
        if recovered is not None:
            recovered_identity, recovered_metadata = recovered
            result = {
                "source_batch": batch_name,
                "generated_data_root": generated_root,
                "dataset_path": destination,
                "metadata_path": metadata_destination,
                "case_count": recovered_identity.sample_count,
                "task": task.id,
                "task_contract_digest": task.contract_digest,
                "timing_coverage": recovered_metadata.provenance["timing"],
                "dataset_fingerprint": recovered_identity.fingerprint,
                "status": "complete",
            }
            if verbose:
                for key, value in result.items():
                    print(f"{key}: {value}")
            return result
        if destination_dir.exists() or destination_dir.is_symlink():
            msg = f"Refusing to overwrite existing final dataset: {destination_dir}"
            raise FileExistsError(msg)
        if metadata_destination.exists() or metadata_destination.is_symlink():
            msg = f"Refusing to overwrite existing dataset metadata: {metadata_destination}"
            raise FileExistsError(msg)
        manifest = _load_batch_manifest(raw_dir, processed_dir, batch_name=batch_name)
        try:
            manifest_snapshot = manifest_path.read_bytes()
            snapshot_manifest = json.loads(manifest_snapshot.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            msg = f"Could not capture validated generation manifest: {manifest_path}"
            raise RuntimeError(msg) from error
        if isinstance(snapshot_manifest, dict) and isinstance(snapshot_manifest.get("cases"), dict):
            snapshot_manifest["cases"] = [snapshot_manifest["cases"]]
        if snapshot_manifest != manifest:
            msg = "Generation manifest changed while source admission was in progress."
            raise RuntimeError(msg)
        manifest_sha256 = hashlib.sha256(manifest_snapshot).hexdigest()
        _validate_exact_source_membership(raw_dir, processed_dir, manifest)
        (
            _sample_csv_path,
            _sample_json_path,
            sample_frame,
            _sample_json,
            portable_sampling,
            sample_csv_snapshot,
            sample_json_snapshot,
        ) = _load_generation_metadata(
            meta_dir,
            batch_name,
            manifest,
        )
        case_ids = list(manifest["intended_case_ids"])
        if len(case_ids) != manifest["configuration"]["N"]:
            msg = "A complete batch manifest must contain exactly configuration.N intended cases."
            raise ValueError(msg)
        timing_snapshot, _timing_payload, timing_summary = _timing_snapshot(
            processed_dir,
            batch_name=batch_name,
            manifest_sha256=manifest_sha256,
            intended_case_ids=case_ids,
        )
        generated_identity, manifest_identity_sha256 = _generated_batch_identity(
            manifest,
            portable_sampling=portable_sampling,
        )
        sample_json_sha256 = hashlib.sha256(sample_json_snapshot).hexdigest()
        provenance = _source_provenance(
            manifest,
            manifest_sha256=manifest_sha256,
            sample_json_sha256=sample_json_sha256,
        )
        records_by_id = {record["case_id"]: record for record in manifest["cases"]}

        staging_root = Path(tempfile.mkdtemp(dir=training_processed_root, prefix=f".{resolved_dataset_id}.dataset-build.", suffix=".tmp"))
        stage_dataset_dir = staging_root / "raw" / resolved_dataset_id
        stage_metadata_dir = staging_root / "meta" / resolved_dataset_id
        staged_dataset_path = stage_dataset_dir / f"{resolved_dataset_id}.pt"
        metadata_published = False
        dataset_published = False
        publication_complete = False
        transaction_active = False
        inputs: torch.Tensor | None = None
        outputs: torch.Tensor | None = None
        source_identities: list[dict[str, Any]] = []
        source_metadata: list[dict[str, Any]] = []
        fingerprints: list[str] = []
        reference_shape: tuple[int, int] | None = None
        try:
            common.serialization.atomic_write_json(
                transaction_path,
                _publication_transaction_record(
                    dataset_id=resolved_dataset_id,
                    phase="building",
                    staging_root=staging_root,
                ),
            )
            transaction_active = True
            stage_dataset_dir.mkdir(parents=True)
            for index, case_id in enumerate(tqdm(case_ids, desc=f"Building {batch_name}", unit="case", disable=not verbose)):
                spatial_shape, case_inputs, case_outputs, normalized_metadata, stable_source, fingerprint = _interpret_generated_case(
                    case_id,
                    task=task,
                    manifest=manifest,
                    manifest_record=records_by_id[case_id],
                    sample_row=sample_frame.loc[case_id],
                    raw_dir=raw_dir,
                    processed_dir=processed_dir,
                )
                if reference_shape is None:
                    reference_shape = spatial_shape
                    inputs = torch.empty((len(case_ids), task.in_channels, *spatial_shape), dtype=torch.float32)
                    outputs = torch.empty((len(case_ids), task.out_channels, *spatial_shape), dtype=torch.float32)
                elif spatial_shape != reference_shape:
                    msg = f"Inconsistent case shape for {case_id!r}: {spatial_shape} != {reference_shape}."
                    raise ValueError(msg)
                if inputs is None or outputs is None:
                    msg = "Final tensors were not allocated."
                    raise RuntimeError(msg)
                inputs[index].copy_(case_inputs)
                outputs[index].copy_(case_outputs)
                source_identities.append(stable_source)
                source_metadata.append(normalized_metadata)
                fingerprints.append(fingerprint)
                del case_inputs, case_outputs
                if verbose and index == 0:
                    print(f"Input fields: {list(task.input_names)}")
                    print(f"Output fields: {list(task.output_names)}")
                    print(f"Spatial shape: {spatial_shape}")
                    print(f"First case fingerprint: {fingerprint}")
            if inputs is None or outputs is None:
                msg = f"No complete generated cases found for {batch_name!r}."
                raise RuntimeError(msg)
            payload = datasets.identity.build_training_dataset_payload(
                task=task,
                dataset_id=resolved_dataset_id,
                sample_ids=case_ids,
                generated_batch_identity=generated_identity,
                source_identities=source_identities,
                source_metadata=source_metadata,
                source_provenance=provenance,
                case_fingerprints=fingerprints,
                inputs=inputs,
                outputs=outputs,
            )
            fingerprint = payload["dataset_fingerprint"]
            common.serialization.atomic_torch_save(payload, staged_dataset_path)
            del payload, inputs, outputs
            inputs = None
            outputs = None
            gc.collect()
            staged_payload = torch.load(staged_dataset_path, map_location="cpu", weights_only=False)
            staged_identity = datasets.identity.validate_training_dataset_payload(staged_payload, task=task, verify_content=True)
            del staged_payload
            gc.collect()
            _stage_metadata_package(
                stage_metadata_dir,
                dataset_identity=staged_identity,
                task=task,
                manifest_snapshot=manifest_snapshot,
                manifest_sha256=manifest_sha256,
                manifest_identity_sha256=manifest_identity_sha256,
                sample_csv_snapshot=sample_csv_snapshot,
                sample_json_snapshot=sample_json_snapshot,
                sample_csv_sha256=manifest["configuration"]["sample_sha256"],
                sample_json_sha256=sample_json_sha256,
                timing_snapshot=timing_snapshot,
                timing_summary=timing_summary,
            )
            datasets.metadata.validate_dataset_metadata_directory(stage_metadata_dir, dataset_identity=staged_identity)
            staged_dataset_sha256 = _sha256_file(staged_dataset_path)
            staged_dataset_size = staged_dataset_path.stat().st_size
            staged_inventory_sha256 = _sha256_file(stage_metadata_dir / datasets.metadata.INVENTORY_FILENAME)
            _assert_generation_snapshot_current(raw_dir, manifest_path, manifest_snapshot)
            common.serialization.atomic_write_json(
                transaction_path,
                _publication_transaction_record(
                    dataset_id=resolved_dataset_id,
                    phase="ready",
                    staging_root=staging_root,
                    dataset_sha256=staged_dataset_sha256,
                    dataset_size=staged_dataset_size,
                    metadata_inventory_sha256=staged_inventory_sha256,
                ),
            )
            training_meta_root.mkdir(parents=True, exist_ok=True)
            training_raw_root.mkdir(parents=True, exist_ok=True)
            if destination_dir.exists() or destination_dir.is_symlink() or metadata_destination.exists() or metadata_destination.is_symlink():
                msg = "Final dataset or metadata target appeared during the build transaction."
                raise FileExistsError(msg)
            _assert_generation_snapshot_current(raw_dir, manifest_path, manifest_snapshot)
            stage_metadata_dir.replace(metadata_destination)
            metadata_published = True
            stage_dataset_dir.replace(destination_dir)
            dataset_published = True
            datasets.metadata.validate_dataset_metadata_directory(
                metadata_destination,
                dataset_identity=staged_identity,
            )
            publication_complete = True
            transaction_path.unlink()
            transaction_active = False
        finally:
            del inputs, outputs
            if not publication_complete:
                if dataset_published and destination_dir.is_dir():
                    shutil.rmtree(destination_dir)
                if metadata_published and metadata_destination.is_dir():
                    shutil.rmtree(metadata_destination)
                if transaction_active:
                    transaction_path.unlink(missing_ok=True)
            if staging_root.exists():
                shutil.rmtree(staging_root)

    result = {
        "source_batch": batch_name,
        "generated_data_root": generated_root,
        "dataset_path": destination,
        "metadata_path": metadata_destination,
        "case_count": len(case_ids),
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "timing_coverage": timing_summary,
        "dataset_fingerprint": fingerprint,
        "status": "complete",
    }
    if verbose:
        for key, value in result.items():
            print(f"{key}: {value}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one final training dataset directly from a completed COMSOL batch.")
    parser.add_argument("batch_id", help="Completed generated batch and final dataset identifier")
    parser.add_argument("--task", default="steady_flow", help="Registered task identifier")
    parser.add_argument("--generated-data-root", type=Path, default=None, help="Override GENERATED_DATA_ROOT for this invocation")
    parser.add_argument("--model-training-data-root", type=Path, default=None, help="Override MODEL_TRAINING_DATA_ROOT for this invocation")
    parser.add_argument("--verbose", action="store_true", help="Show bounded progress and final identity")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the maintained direct final-dataset build command."""
    args = _build_parser().parse_args(argv)
    try:
        result = build_batch_dataset(
            args.batch_id,
            task_id=args.task,
            generated_data_root=args.generated_data_root,
            model_training_data_root=args.model_training_data_root,
            verbose=args.verbose,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Dataset build failed: {type(error).__name__}: {error}")
        return 1
    print(f"Source batch: {result['source_batch']}")
    print(f"Generated data root: {result['generated_data_root']}")
    print(f"Destination dataset: {result['dataset_path']}")
    print(f"Metadata destination: {result['metadata_path']}")
    print(f"Case count: {result['case_count']}")
    print(f"Task identity: {result['task']} ({result['task_contract_digest']})")
    coverage = result["timing_coverage"]
    print(f"COMSOL timing coverage: {coverage['measured_case_count']}/{coverage['intended_case_count']} ({coverage['status']})")
    print(f"Dataset fingerprint: {result['dataset_fingerprint']}")
    print(f"Status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
