"""
===============================================================================
analysis_artifact_contracts.py
===============================================================================
Define lightweight persisted contracts for generated analysis artifacts.

Responsibilities:
  - Declare artifact, provenance, residual, and derivative schema constants
  - Name normalized sufficient-statistic columns and finalize the macro objective
  - Build ordered membership digests, completion paths, and payload manifests

Design principles:
  - Reader contracts depend only on NumPy, pandas, and the standard library
  - Persisted names and schema values have one authoritative owner
  - Objective aggregation uses global per-field SSE and element counts

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 1
RESIDUAL_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"
NORMALIZED_OBJECTIVE_TOLERANCE = {"rtol": 1e-12, "atol": 1e-12}
EVAL_PAD = 2
ARTIFACT_DERIVATIVE_KIND = "spectral"
ARTIFACT_DERIVATIVE_EXTENSION = "reflect"


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


def aggregate_normalized_macro_rmse(
    frame: pd.DataFrame,
    *,
    output_fields: Iterable[str],
) -> dict[str, Any]:
    """
    Finalize global field RMSEs and their unweighted arithmetic macro mean.

    Per-field SSE and element counts are summed in exact membership order.
    Per-case or per-batch RMSE values never enter the aggregate.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema per-case table containing normalized evidence columns.
    output_fields : Iterable[str]
        Unique TaskSpec output names in declared field order.

    Returns
    -------
    dict[str, Any]
        Objective semantics, global per-field SSE/count/RMSE values, their
        unweighted arithmetic RMSE mean, and the declared agreement tolerance.

    Raises
    ------
    KeyError
        If any required evidence column is absent.
    TypeError, ValueError
        If fields, evidence values, counts, or per-case RMSE consistency fail.
    RuntimeError, FloatingPointError
        If no elements can be finalized or the aggregate is non-finite.

    """
    fields = tuple(output_fields)
    if not fields or len(fields) != len(set(fields)):
        msg = "Artifact macro RMSE requires unique non-empty output fields."
        raise ValueError(msg)
    if not frame.columns.is_unique:
        msg = "Artifact macro RMSE cannot consume duplicate DataFrame columns."
        raise ValueError(msg)

    field_summary: dict[str, dict[str, float | int]] = {}
    field_rmse_values: list[float] = []
    for field_name in fields:
        sse_column, count_column, rmse_column = normalized_statistic_columns(field_name)
        missing = [name for name in (sse_column, count_column, rmse_column) if name not in frame.columns]
        if missing:
            msg = f"Artifact macro RMSE is missing normalized evidence columns: {missing}."
            raise KeyError(msg)
        squared_error_sum = 0.0
        element_count = 0
        for row_index, (raw_sse, raw_count, raw_rmse) in enumerate(
            zip(frame[sse_column].tolist(), frame[count_column].tolist(), frame[rmse_column].tolist(), strict=True)
        ):
            if isinstance(raw_sse, bool) or not isinstance(raw_sse, (int, float, np.integer, np.floating)):
                msg = f"{sse_column} row {row_index} must be a real scalar."
                raise TypeError(msg)
            if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)) or int(raw_count) <= 0:
                msg = f"{count_column} row {row_index} must be a positive integer."
                raise TypeError(msg)
            sse_value = float(raw_sse)
            rmse_value = float(raw_rmse)
            if not math.isfinite(sse_value) or sse_value < 0.0 or not math.isfinite(rmse_value) or rmse_value < 0.0:
                msg = f"Normalized evidence for field {field_name!r}, row {row_index} must be finite and non-negative."
                raise ValueError(msg)
            expected_case_rmse = math.sqrt(sse_value / int(raw_count))
            if not math.isclose(
                rmse_value,
                expected_case_rmse,
                rel_tol=NORMALIZED_OBJECTIVE_TOLERANCE["rtol"],
                abs_tol=NORMALIZED_OBJECTIVE_TOLERANCE["atol"],
            ):
                msg = f"{rmse_column} row {row_index} does not match its SSE/count evidence."
                raise ValueError(msg)
            squared_error_sum += sse_value
            element_count += int(raw_count)
        if element_count <= 0:
            msg = f"Artifact macro RMSE cannot finalize field {field_name!r} without elements."
            raise RuntimeError(msg)
        field_rmse = math.sqrt(squared_error_sum / element_count)
        field_summary[field_name] = {
            "normalized_squared_error_sum": squared_error_sum,
            "normalized_element_count": element_count,
            "normalized_rmse": field_rmse,
        }
        field_rmse_values.append(field_rmse)

    value = float(np.mean(np.asarray(field_rmse_values, dtype=np.float64)))
    if not math.isfinite(value):
        msg = "Artifact normalized_macro_rmse finalized to a non-finite value."
        raise FloatingPointError(msg)
    return {
        "objective_id": "normalized_macro_rmse",
        "reduction": "field_macro_element_mean",
        "space": "normalized",
        "fields": list(fields),
        "direction": "minimize",
        "value": value,
        "field_statistics": field_summary,
        "agreement_tolerance": dict(NORMALIZED_OBJECTIVE_TOLERANCE),
    }


def ordered_indices_sha256(indices: Iterable[int]) -> str:
    """
    Return the canonical digest for an ordered integer membership.

    Parameters
    ----------
    indices : Iterable[int]
        Source indices in saved membership order; order and duplicates, if any,
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
