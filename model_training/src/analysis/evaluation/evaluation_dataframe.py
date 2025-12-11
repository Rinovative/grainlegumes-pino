"""
Utility functions for constructing evaluation-ready DataFrames.

This module converts the raw Parquet files produced by the artifact
generator into lightweight evaluation DataFrames. All heavy numerical
fields (predictions, ground truth, error tensors) remain stored in
external NPZ files, while the DataFrame contains only scalar metrics
and flattened meta-information for analysis.

Supports both plog-space and physical kappa-space metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANISOTROPY_DIM = 2
MS_WEIGHT_DIM = 2
N_QUANTILES = 3


def _to_scalar(val: Any) -> Any:
    """
    Convert nested numpy arrays or lists into Python scalar values.

    Handles metadata fields that are often stored as np.ndarray([value]).
    Arrays with a single element are converted to float(value). Larger
    arrays are recursively converted to lists of scalars.

    Parameters
    ----------
    val : Any
        Input metadata value.

    Returns
    -------
    Any
        Scalar value or recursively processed list of scalars.

    """
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        return [_to_scalar(x) for x in val]
    return val


def extract_meta_features(meta: dict[str, Any]) -> dict[str, float | int | bool]:
    """
    Flatten a nested metadata dictionary into scalar evaluation features.

    Parameters
    ----------
    meta : dict
        Metadata dictionary parsed from JSON or extracted via artifact generation.
        Expected keys include:
            - "statistics": mean, std, min, max, coeff_var, skewness, kurtosis, quantiles
            - "geometry": domain and grid resolution parameters
            - "parameters": kappa-generation parameters such as k_mean, anisotropy, ms_scale, etc.

    Returns
    -------
    dict
        A flat dictionary mapping feature names to scalar values suitable
        for inclusion in an evaluation DataFrame.

    """
    out: dict[str, float | int | bool] = {}

    stats = meta.get("statistics", {})
    out["stat_mean"] = float(_to_scalar(stats.get("mean", np.nan)))
    out["stat_std"] = float(_to_scalar(stats.get("std", np.nan)))
    out["stat_min"] = float(_to_scalar(stats.get("min", np.nan)))
    out["stat_max"] = float(_to_scalar(stats.get("max", np.nan)))
    out["stat_cv"] = float(_to_scalar(stats.get("coeff_var", np.nan)))
    out["stat_skew"] = float(_to_scalar(stats.get("skewness", np.nan)))
    out["stat_kurt"] = float(_to_scalar(stats.get("kurtosis", np.nan)))

    quantiles = _to_scalar(stats.get("quantiles", []))
    if isinstance(quantiles, (list, tuple)) and len(quantiles) == N_QUANTILES:
        out["stat_q25"] = float(quantiles[0])
        out["stat_q50"] = float(quantiles[1])
        out["stat_q75"] = float(quantiles[2])

    geom = meta.get("geometry", {})
    for key in ["Lx", "Ly", "dx", "dy", "nx", "ny", "res"]:
        val = _to_scalar(geom.get(key))
        if val is not None:
            out[f"geom_{key}"] = float(val)

    par = meta.get("parameters", {})
    for key in ["k_mean", "var_rel", "corr_len_rel", "seed", "ms_scale", "coupling", "volume_fraction", "lognormal"]:
        val = _to_scalar(par.get(key))
        if isinstance(val, (float, int, bool)):
            out[f"par_{key}"] = val

    aniso = _to_scalar(par.get("anisotropy"))
    if isinstance(aniso, (list, tuple)) and len(aniso) == ANISOTROPY_DIM:
        major = float(aniso[0])
        minor = float(aniso[1])
        out["par_aniso_major"] = major
        out["par_aniso_minor"] = minor
        out["par_aniso_ratio"] = major / (minor + 1e-12)

    weights = _to_scalar(par.get("ms_weight"))
    if isinstance(weights, (list, tuple)) and len(weights) == MS_WEIGHT_DIM:
        out["par_ms_w1"] = float(weights[0])
        out["par_ms_w2"] = float(weights[1])

    return out


def load_and_build_eval_df(parquet_path: str | Path) -> pd.DataFrame:
    """
    Load a Parquet artifact file and construct the corresponding evaluation DataFrame.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the Parquet file created by the artifact generator.

    Returns
    -------
    DataFrame
        Evaluation-ready DataFrame with per-case metrics and flattened metadata.

    """
    parquet_path = Path(parquet_path)
    df_raw = pd.read_parquet(parquet_path)
    return build_eval_df(df_raw)


def build_eval_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build an enriched evaluation DataFrame.

    Raw parquet files created by the artifact generator contain:
        - case_index
        - npz_path
        - l2_plog
        - rel_l2_plog
        - l2_kappa
        - rel_l2_kappa
        - meta

    The returned DataFrame contains:
        - the same base fields
        - flattened meta-information (statistical features, geometry, parameters)
        - optional additional derived metrics

    Parameters
    ----------
    df_raw : DataFrame
        Raw DataFrame loaded from Parquet.

    Returns
    -------
    DataFrame
        Enriched DataFrame containing scalar evaluation features.

    """
    df = df_raw.copy()
    meta_features = df["meta"].apply(extract_meta_features)
    meta_df = pd.DataFrame(meta_features.tolist())
    df_out = pd.concat([df, meta_df], axis=1)
    return df_out.drop(columns=["meta"], errors="ignore")
