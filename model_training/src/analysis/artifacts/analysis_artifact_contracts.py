"""
===============================================================================
analysis_artifact_contracts.py
===============================================================================
Define lightweight persisted contracts for generated analysis artifacts.

Responsibilities:
  - Declare artifact, provenance, residual, and derivative schema constants
  - Name physical and normalized sufficient-statistic columns
  - Finalize the equal physical-output-group selection objective
  - Build ordered membership digests, completion paths, and payload manifests

Design principles:
  - Persisted names and schema values have one authoritative owner
  - Objective aggregation delegates to the online group-metric finalizer
  - Physical SSE, counts, and train-fitted scales remain explicit evidence

This module does NOT:
  - Import Torch or run trained-model inference
  - Write artifact payloads or publish completion markers
  - Admit, rebuild, lock, time, render, or upload artifact caches
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.domain.tasks.domain_task_spec import OutputGroupSpec
from src.learning.metrics.learning_metrics import finalize_group_rmse_statistics

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 1
RESIDUAL_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"
NORMALIZED_OBJECTIVE_TOLERANCE = {"rtol": 1e-12, "atol": 1e-12}
NORMALIZED_DIAGNOSTIC_TOLERANCE = {"rtol": 2e-5, "atol": 1e-12}
EVAL_PAD = 2
ARTIFACT_DERIVATIVE_KIND = "spectral"
ARTIFACT_DERIVATIVE_EXTENSION = "reflect"


def resolve_case_payload_path(
    artifact_root: Path | str,
    stored_path: object,
    *,
    expected_filename: str,
) -> Path:
    """
    Resolve one relative or safely relocatable legacy NPZ reference.

    New rows must store exactly ``npz/<expected_filename>``. Legacy absolute
    paths are accepted only when they end in the same unambiguous bundle-local
    suffix below an ``analysis`` path. The historical absolute prefix is never
    followed. Existing symbolic-link substitutions below the current artifact
    root are rejected.
    """
    if not isinstance(stored_path, str) or not stored_path:
        msg = "Artifact Parquet npz_path must be a non-empty string."
        raise TypeError(msg)
    if not isinstance(expected_filename, str) or not expected_filename.endswith(".npz") or Path(expected_filename).name != expected_filename:
        msg = f"Expected NPZ filename is invalid: {expected_filename!r}."
        raise ValueError(msg)
    raw = Path(stored_path)
    expected_relative = Path("npz") / expected_filename
    if raw.is_absolute():
        if raw.name != expected_filename or raw.parent.name != "npz" or "analysis" not in raw.parts[:-2]:
            msg = f"Legacy absolute NPZ path has no safe bundle-local interpretation: {stored_path!r}."
            raise ValueError(msg)
    elif raw != expected_relative or ".." in raw.parts:
        msg = f"Artifact relative NPZ path must be exactly {expected_relative.as_posix()!r}, got {stored_path!r}."
        raise ValueError(msg)

    root = Path(artifact_root).expanduser().resolve()
    candidate = root / expected_relative
    if not candidate.is_relative_to(root) or candidate.parent.is_symlink() or candidate.is_symlink():
        msg = f"Artifact NPZ path uses an unsafe symbolic-link substitution: {candidate}"
        raise ValueError(msg)
    if not candidate.is_file() or candidate.resolve() != candidate:
        msg = f"Artifact NPZ payload is missing or escapes its current root: {candidate}"
        raise FileNotFoundError(msg)
    return candidate


def _file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one artifact payload file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_statistic_columns(field_name: str) -> tuple[str, str, str]:
    """
    Return task-derived per-case normalized SSE, count, and RMSE columns.

    Parameters
    ----------
    field_name : str
        Exact non-empty TaskSpec output field name.

    Returns
    -------
    tuple[str, str, str]
        Ordered sufficient-statistic and convenience-RMSE column names.

    Raises
    ------
    ValueError
        If the field name is empty or not text.

    """
    if not isinstance(field_name, str) or not field_name:
        msg = "Artifact output field names must be non-empty strings."
        raise ValueError(msg)
    return (
        f"normalized_sse_{field_name}",
        f"normalized_count_{field_name}",
        f"normalized_rmse_{field_name}",
    )


def physical_statistic_columns(field_name: str) -> tuple[str, str, str]:
    """
    Return task-derived per-case physical SSE, count, and RMSE columns.

    Parameters
    ----------
    field_name : str
        Exact non-empty TaskSpec output field name.

    Returns
    -------
    tuple[str, str, str]
        Ordered physical sufficient-statistic and convenience-RMSE columns.

    Raises
    ------
    ValueError
        If the field name is empty or not text.

    """
    if not isinstance(field_name, str) or not field_name:
        msg = "Artifact output field names must be non-empty strings."
        raise ValueError(msg)
    return (
        f"physical_sse_{field_name}",
        f"physical_count_{field_name}",
        f"physical_rmse_{field_name}",
    )


def output_group_payload(output_groups: Iterable[Any]) -> list[dict[str, Any]]:
    """
    Normalize task-owned output groups to an ordered JSON-safe declaration.

    Parameters
    ----------
    output_groups : Iterable[Any]
        Task ``OutputGroupSpec`` objects or equivalent mappings containing
        ``id`` and ``fields``.

    Returns
    -------
    list[dict[str, Any]]
        Ordered group identifiers and their ordered member fields.

    Raises
    ------
    TypeError, ValueError
        If group declarations are malformed, duplicated, or overlap.

    """
    payload: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_fields: set[str] = set()
    for index, raw_group in enumerate(output_groups):
        if isinstance(raw_group, Mapping):
            group_id = raw_group.get("id")
            raw_fields = raw_group.get("fields")
        else:
            group_id = getattr(raw_group, "id", None)
            raw_fields = getattr(raw_group, "fields", None)
        if not isinstance(group_id, str) or not group_id:
            msg = f"Artifact output group {index} must have a non-empty string id."
            raise TypeError(msg)
        if isinstance(raw_fields, np.ndarray):
            raw_fields = raw_fields.tolist()
        if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
            msg = f"Artifact output group {group_id!r} must contain fields."
            raise TypeError(msg)
        fields = tuple(raw_fields)
        if any(not isinstance(field, str) or not field for field in fields):
            msg = f"Artifact output group {group_id!r} fields must be non-empty strings."
            raise TypeError(msg)
        if group_id in seen_group_ids or len(fields) != len(set(fields)):
            msg = f"Artifact output group {group_id!r} is duplicated or contains duplicate fields."
            raise ValueError(msg)
        overlap = seen_fields.intersection(fields)
        if overlap:
            msg = f"Artifact output groups overlap on fields: {sorted(overlap)}."
            raise ValueError(msg)
        payload.append({"id": group_id, "fields": list(fields)})
        seen_group_ids.add(group_id)
        seen_fields.update(fields)
    if not payload:
        msg = "Artifact group objective requires at least one output group."
        raise ValueError(msg)
    return payload


def output_standard_deviations_from_state(
    normalizer_state: Mapping[str, Any],
    *,
    output_fields: Iterable[str],
) -> dict[str, float]:
    """
    Extract positive train-fitted output scales from saved normalizer state.

    Parameters
    ----------
    normalizer_state : collections.abc.Mapping[str, Any]
        Persisted normalizer mapping containing ``out_normalizer.std`` with the
        authoritative ``[1, channels, 1, 1]`` layout.
    output_fields : Iterable[str]
        Ordered unique TaskSpec output names aligned with the channel axis.

    Returns
    -------
    dict[str, float]
        Output field to exact finite positive training standard deviation.

    Raises
    ------
    KeyError, TypeError, ValueError
        If state, layout, fields, or fitted scales are invalid.

    """
    if not isinstance(normalizer_state, Mapping):
        msg = "Artifact normalizer state must be a mapping."
        raise TypeError(msg)
    fields = tuple(output_fields)
    if not fields or len(fields) != len(set(fields)) or any(not isinstance(field, str) or not field for field in fields):
        msg = "Artifact output fields must be unique non-empty strings."
        raise ValueError(msg)
    if "out_normalizer.std" not in normalizer_state:
        msg = "Artifact normalizer state is missing 'out_normalizer.std'."
        raise KeyError(msg)
    raw_scales = normalizer_state["out_normalizer.std"]
    detach = getattr(raw_scales, "detach", None)
    if callable(detach):
        raw_scales = detach()
    cpu = getattr(raw_scales, "cpu", None)
    if callable(cpu):
        raw_scales = cpu()
    to_numpy = getattr(raw_scales, "numpy", None)
    if callable(to_numpy):
        raw_scales = to_numpy()
    array = np.asarray(raw_scales)
    expected_shape = (1, len(fields), 1, 1)
    if array.shape != expected_shape or not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        msg = f"Artifact output normalizer standard deviations must have real shape {expected_shape}, got {array.shape}."
        raise TypeError(msg)
    values = array.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        msg = "Artifact output normalizer standard deviations must be finite and strictly positive."
        raise ValueError(msg)
    return {field: float(value) for field, value in zip(fields, values, strict=True)}


def _validated_train_scales(
    train_standard_deviations: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
) -> dict[str, float]:
    """Return an exact finite positive train-scale mapping for the fields."""
    if not isinstance(train_standard_deviations, Mapping) or set(train_standard_deviations) != set(fields):
        msg = "Artifact train standard deviations must map exactly the grouped output fields."
        raise ValueError(msg)
    result: dict[str, float] = {}
    for field in fields:
        raw_value = train_standard_deviations[field]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
            msg = f"Artifact train standard deviation for {field!r} must be a real scalar."
            raise TypeError(msg)
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            msg = f"Artifact train standard deviation for {field!r} must be finite and strictly positive."
            raise ValueError(msg)
        result[field] = value
    return result


def _row_statistic(
    *,
    raw_sse: Any,
    raw_count: Any,
    raw_rmse: Any,
    sse_column: str,
    count_column: str,
    rmse_column: str,
    row_index: int,
) -> tuple[float, int, float]:
    """Validate one row-local SSE/count/RMSE triplet and return native values."""
    if isinstance(raw_sse, bool) or not isinstance(raw_sse, (int, float, np.integer, np.floating)):
        msg = f"{sse_column} row {row_index} must be a real scalar."
        raise TypeError(msg)
    if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)) or int(raw_count) <= 0:
        msg = f"{count_column} row {row_index} must be a positive integer."
        raise TypeError(msg)
    if isinstance(raw_rmse, bool) or not isinstance(raw_rmse, (int, float, np.integer, np.floating)):
        msg = f"{rmse_column} row {row_index} must be a real scalar."
        raise TypeError(msg)
    sse_value = float(raw_sse)
    count_value = int(raw_count)
    rmse_value = float(raw_rmse)
    if not math.isfinite(sse_value) or sse_value < 0.0 or not math.isfinite(rmse_value) or rmse_value < 0.0:
        msg = f"Artifact evidence in {sse_column!r} row {row_index} must be finite and non-negative."
        raise ValueError(msg)
    expected_rmse = math.sqrt(sse_value / count_value)
    if not math.isclose(
        rmse_value,
        expected_rmse,
        rel_tol=NORMALIZED_OBJECTIVE_TOLERANCE["rtol"],
        abs_tol=NORMALIZED_OBJECTIVE_TOLERANCE["atol"],
    ):
        msg = f"{rmse_column} row {row_index} does not match its SSE/count evidence."
        raise ValueError(msg)
    return sse_value, count_value, rmse_value


def aggregate_normalized_group_macro_rmse(
    frame: pd.DataFrame,
    *,
    output_groups: Iterable[Any],
    train_standard_deviations: Mapping[str, Any],
    normalization_denominator_floor: float,
) -> dict[str, Any]:
    """
    Finalize the global equal macro mean over physical output groups.

    Physical SSE and element counts are summed before any square root. Each
    group's physical component MSEs are divided by the sum of its train-fitted
    component variances, and finalized group errors receive equal macro weight.
    Per-case or per-batch RMSE values never enter the objective reduction.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema per-case table containing physical and normalized
        sufficient-statistic columns.
    output_groups : Iterable[Any]
        Ordered task-owned output groups covering the complete output contract.
    train_standard_deviations : collections.abc.Mapping[str, Any]
        Exact positive per-field standard deviations fitted on training data.
    normalization_denominator_floor : float
        Persisted non-negative denominator addition used only to verify retained
        component-normalized diagnostics. It does not enter the objective.

    Returns
    -------
    dict[str, Any]
        Objective semantics, global field and group statistics, equal group
        weights, the dimensionless objective value, and agreement tolerance.

    Raises
    ------
    KeyError
        If any required evidence column is absent.
    TypeError, ValueError
        If groups, scales, evidence, counts, or diagnostic consistency fail.
    RuntimeError, FloatingPointError
        If no elements can be finalized or the aggregate is non-finite.

    """
    groups = output_group_payload(output_groups)
    fields = tuple(field for group in groups for field in group["fields"])
    scales = _validated_train_scales(train_standard_deviations, fields=fields)
    if (
        isinstance(normalization_denominator_floor, bool)
        or not isinstance(normalization_denominator_floor, (int, float, np.integer, np.floating))
        or not math.isfinite(float(normalization_denominator_floor))
        or float(normalization_denominator_floor) < 0.0
    ):
        msg = "Artifact normalization denominator floor must be finite and non-negative."
        raise ValueError(msg)
    denominator_floor = float(normalization_denominator_floor)
    if not frame.columns.is_unique:
        msg = "Artifact group macro RMSE cannot consume duplicate DataFrame columns."
        raise ValueError(msg)

    field_summary: dict[str, dict[str, float | int]] = {}
    for field_name in fields:
        physical_columns = physical_statistic_columns(field_name)
        normalized_columns = normalized_statistic_columns(field_name)
        missing = [name for name in (*physical_columns, *normalized_columns) if name not in frame.columns]
        if missing:
            msg = f"Artifact group macro RMSE is missing sufficient-statistic columns: {missing}."
            raise KeyError(msg)
        physical_sse = 0.0
        physical_count = 0
        normalized_sse = 0.0
        normalized_count = 0
        for row_index, values in enumerate(
            zip(
                *(frame[column].tolist() for column in (*physical_columns, *normalized_columns)),
                strict=True,
            )
        ):
            row_physical_sse, row_physical_count, _ = _row_statistic(
                raw_sse=values[0],
                raw_count=values[1],
                raw_rmse=values[2],
                sse_column=physical_columns[0],
                count_column=physical_columns[1],
                rmse_column=physical_columns[2],
                row_index=row_index,
            )
            row_normalized_sse, row_normalized_count, _ = _row_statistic(
                raw_sse=values[3],
                raw_count=values[4],
                raw_rmse=values[5],
                sse_column=normalized_columns[0],
                count_column=normalized_columns[1],
                rmse_column=normalized_columns[2],
                row_index=row_index,
            )
            expected_normalized_sse = row_physical_sse / (scales[field_name] + denominator_floor) ** 2
            if row_physical_count != row_normalized_count or not math.isclose(
                row_normalized_sse,
                expected_normalized_sse,
                rel_tol=NORMALIZED_DIAGNOSTIC_TOLERANCE["rtol"],
                abs_tol=NORMALIZED_DIAGNOSTIC_TOLERANCE["atol"],
            ):
                msg = (
                    f"Normalized evidence for field {field_name!r}, row {row_index} contradicts physical evidence "
                    "and the saved normalizer denominator."
                )
                raise ValueError(msg)
            physical_sse += row_physical_sse
            physical_count += row_physical_count
            normalized_sse += row_normalized_sse
            normalized_count += row_normalized_count
        if physical_count <= 0 or normalized_count <= 0:
            msg = f"Artifact group macro RMSE cannot finalize field {field_name!r} without elements."
            raise RuntimeError(msg)
        physical_mse = physical_sse / physical_count
        normalized_rmse = math.sqrt(normalized_sse / normalized_count)
        field_summary[field_name] = {
            "physical_squared_error_sum": physical_sse,
            "physical_element_count": physical_count,
            "physical_rmse": math.sqrt(physical_mse),
            "train_standard_deviation": scales[field_name],
            "normalized_squared_error_sum": normalized_sse,
            "normalized_element_count": normalized_count,
            "normalized_rmse": normalized_rmse,
        }

    resolved_groups = tuple(OutputGroupSpec(id=str(group["id"]), fields=tuple(group["fields"])) for group in groups)
    squared_error_sums = {field: float(field_summary[field]["physical_squared_error_sum"]) for field in fields}
    element_counts = {field: int(field_summary[field]["physical_element_count"]) for field in fields}
    finalized = finalize_group_rmse_statistics(
        resolved_groups,
        squared_error_sums=squared_error_sums,
        element_counts=element_counts,
        train_standard_deviations=scales,
    )

    group_summary: dict[str, dict[str, Any]] = {}
    group_weight = 1.0 / len(groups)
    for group in groups:
        group_id = str(group["id"])
        group_fields = tuple(group["fields"])
        physical_mse_sum = sum(squared_error_sums[field] / element_counts[field] for field in group_fields)
        train_variance_sum = sum(scales[field] ** 2 for field in group_fields)
        normalized_group_rmse = finalized.normalized[group_id]
        physical_group_rmse = finalized.physical[group_id]
        group_summary[group_id] = {
            "fields": list(group_fields),
            "weight": group_weight,
            "physical_component_mse_sum": physical_mse_sum,
            "train_variance_sum": train_variance_sum,
            "normalized_rmse": normalized_group_rmse,
            "physical_rmse": physical_group_rmse,
            "objective_contribution": group_weight * normalized_group_rmse,
        }

    value = finalized.normalized_macro
    if not math.isfinite(value):
        msg = "Artifact normalized_group_macro_rmse finalized to a non-finite value."
        raise FloatingPointError(msg)
    return {
        "objective_id": "normalized_group_macro_rmse",
        "kind": "group_macro_rmse",
        "reduction": "group_macro_element_mean",
        "space": "physical",
        "groups": groups,
        "direction": "minimize",
        "unit": "1",
        "value": value,
        "field_statistics": field_summary,
        "group_statistics": group_summary,
        "agreement_tolerance": dict(NORMALIZED_OBJECTIVE_TOLERANCE),
    }


def ordered_indices_sha256(indices: Iterable[int]) -> str:
    """
    Return the canonical digest for an ordered integer membership.

    Parameters
    ----------
    indices : Iterable[int]
        Source indices in saved membership order. Order and duplicates, if any,
        participate in the canonical compact-JSON byte representation.

    Returns
    -------
    str
        Lowercase SHA-256 digest of that exact ordered representation.

    """
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_provenance_path(save_root: str | Path) -> Path:
    """Return the versioned provenance sidecar path for an artifact root."""
    return Path(save_root) / ARTIFACT_PROVENANCE_FILENAME


def artifact_output_manifest(save_root: str | Path) -> dict[str, Any]:
    """
    Build the exact digest manifest for one complete artifact payload.

    Parameters
    ----------
    save_root : str | pathlib.Path
        Artifact target containing exactly one Parquet table and a non-empty
        ``npz`` directory.

    Returns
    -------
    dict[str, Any]
        Artifact-relative paths and SHA-256 digests in deterministic name order.

    Raises
    ------
    RuntimeError
        If the target lacks exactly one Parquet file or any NPZ case payload.

    """
    root = Path(save_root)
    parquet_files = sorted(root.glob("*.parquet"))
    npz_files = sorted((root / "npz").glob("*.npz"))
    if len(parquet_files) != 1:
        msg = f"Artifact payload must contain exactly one Parquet file, found {len(parquet_files)} in {root}."
        raise RuntimeError(msg)
    if not npz_files:
        msg = f"Artifact payload contains no NPZ files: {root / 'npz'}"
        raise RuntimeError(msg)

    def entry(payload_path: Path) -> dict[str, Any]:
        """Bind one artifact-relative payload path to its complete-file digest."""
        return {
            "path": payload_path.relative_to(root).as_posix(),
            "sha256": _file_sha256(payload_path),
        }

    return {
        "parquet": entry(parquet_files[0]),
        "npz": [entry(payload_path) for payload_path in npz_files],
    }
