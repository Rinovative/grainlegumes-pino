"""
===============================================================================
evaluation_dataframe.py
===============================================================================
Build evaluation-ready DataFrames from artifact tables.

Responsibilities:
  - Load raw Parquet artifact tables
  - Flatten JSON metadata into scalar columns
  - Preserve case-level metric columns for downstream plots

Design principles:
  - Artifact parsing is deterministic and side-effect free
  - Metadata expansion is explicit and shallow
  - Plot modules receive normalized tabular inputs

Boundaries:
  - Artifact generation belongs to analysis.artifacts
  - Visualization belongs to analysis.evaluation.plots

Notes:
  Expected raw Parquet columns include authoritative identity, artifact path,
  dimensionless relative metrics, per-field physical RMSE values, diagnostics,
  and ``meta`` as a JSON object string.
===============================================================================

"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================
def _parse_meta(value: Any) -> dict[str, Any]:
    """Return one validated JSON metadata object."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        message = f"Artifact meta must be a JSON object or mapping, got {type(value).__name__}."
        raise TypeError(message)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        message = "Artifact meta must contain valid JSON."
        raise ValueError(message) from error
    if not isinstance(parsed, dict):
        message = f"Artifact meta JSON must decode to an object, got {type(parsed).__name__}."
        raise TypeError(message)
    return parsed


def _to_scalar(val: Any) -> Any:
    """
    Convert numpy scalar arrays to native Python scalars.

    Parameters
    ----------
    val : Any
        Input value.

    Returns
    -------
    Any
        Native Python scalar if input was a numpy scalar array; otherwise unchanged.

    """
    if isinstance(val, np.ndarray):
        if val.ndim == 0 or val.size == 1:
            return val.item()
        return val  # keep non-scalar arrays intact
    return val


def flatten_meta_scalars(
    obj: Any,
    *,
    prefix: str = "",
    out: dict[str, float | int | bool | str] | None = None,
) -> dict[str, float | int | bool | str]:
    """
    Recursively flatten a nested metadata structure and extract scalar values.

    Rules
    -----
    - dict            -> recurse
    - list of len 1   -> unwrap and recurse
    - scalar          -> stored as DataFrame column
    - list of len > 1 -> ignored (not suitable for tabular form)

    Parameters
    ----------
    obj : Any
        Arbitrary metadata object (dict, list, scalar).
    prefix : str
        Current key prefix used to construct column names.
    out : dict, optional
        Output dictionary used internally during recursion.

    Returns
    -------
    dict[str, float | int | bool | str]
        Flat mapping: column_name -> scalar value.

    """
    if out is None:
        out = {}

    # unwrap numpy 0-d arrays
    obj = _to_scalar(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}_{k}" if prefix else k
            flatten_meta_scalars(v, prefix=new_prefix, out=out)

    elif isinstance(obj, list):
        if len(obj) == 1:
            flatten_meta_scalars(obj[0], prefix=prefix, out=out)

        elif 2 <= len(obj) <= 4:  # noqa: PLR2004
            for i, v in enumerate(obj):
                flatten_meta_scalars(
                    v,
                    prefix=f"{prefix}_{i}",
                    out=out,
                )

        # lists longer than this are ignored on purpose

    elif isinstance(obj, (int, float, bool, str)) and prefix:
        out[prefix] = obj

    return out


# =============================================================================
# DataFrame builders
# =============================================================================


def load_and_build_eval_df(parquet_path: str | Path) -> pd.DataFrame:
    """
    Load a raw evaluation Parquet file and build an enriched DataFrame.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the raw evaluation Parquet file.

    Returns
    -------
    pd.DataFrame
        Enriched evaluation DataFrame with flattened metadata.

    """
    parquet_path = Path(parquet_path)
    df_raw = pd.read_parquet(parquet_path)
    return build_eval_df(df_raw)


def build_eval_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build an enriched evaluation DataFrame.

    This function:
    - keeps all existing scalar Parquet columns
    - flattens all scalar entries found in `meta`
    - appends them as additional DataFrame columns
    - drops the raw `meta` column afterwards

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw Parquet DataFrame produced by the artifact generator.

    Returns
    -------
    pd.DataFrame
        Evaluation DataFrame with flattened scalar metadata.

    """
    df = df_raw.copy()

    if "meta" in df.columns:
        meta_features = df["meta"].apply(lambda m: flatten_meta_scalars(_parse_meta(m)))
        meta_df = pd.DataFrame(meta_features.tolist(), index=df.index)
        authoritative_columns = set(df.columns)
        collisions = sorted(authoritative_columns.intersection(meta_df.columns))
        if collisions:
            msg = f"Artifact metadata collides with authoritative table columns: {collisions}."
            raise ValueError(msg)

        df = pd.concat([df, meta_df], axis=1)

        # Drop raw meta to keep table lightweight and analysis-friendly
        df = df.drop(columns=["meta"], errors="ignore")

    if not df.columns.is_unique:
        duplicates = sorted(set(df.columns[df.columns.duplicated()].tolist()))
        msg = f"Evaluation DataFrame contains duplicate columns: {duplicates}."
        raise ValueError(msg)
    return df
