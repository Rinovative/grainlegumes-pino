"""
Utility functions for constructing evaluation-ready DataFrames.

This module converts the raw Parquet produced by the artifact generator
into a lightweight evaluation DataFrame.

Important:
    - Heavy arrays (pred, gt, err) remain in NPZ files.
    - The DataFrame contains only metrics + flattened meta-information.
    - Now supports BOTH plog-space metrics and physical kappa-space metrics.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ======================================================================
# CONSTANTS
# ======================================================================

ANISOTROPY_DIM = 2
MS_WEIGHT_DIM = 2
N_QUANTILES = 3


# ======================================================================
# META FEATURE EXTRACTION
# ======================================================================


def extract_meta_features(meta: dict[str, Any]) -> dict[str, float | int | bool]:
    """Flatten the nested meta.json structure into scalar evaluation features."""
    out: dict[str, float | int | bool] = {}

    # ----------------------------------------------------
    # 1) statistics
    # ----------------------------------------------------
    stats = meta.get("statistics", {})
    out["stat_mean"] = float(stats.get("mean", np.nan))
    out["stat_std"] = float(stats.get("std", np.nan))
    out["stat_min"] = float(stats.get("min", np.nan))
    out["stat_max"] = float(stats.get("max", np.nan))
    out["stat_cv"] = float(stats.get("coeff_var", np.nan))
    out["stat_skew"] = float(stats.get("skewness", np.nan))
    out["stat_kurt"] = float(stats.get("kurtosis", np.nan))

    quantiles = stats.get("quantiles", [])
    if isinstance(quantiles, (list, tuple)) and len(quantiles) == N_QUANTILES:
        out["stat_q25"] = float(quantiles[0])
        out["stat_q50"] = float(quantiles[1])
        out["stat_q75"] = float(quantiles[2])

    # ----------------------------------------------------
    # 2) geometry
    # ----------------------------------------------------
    geom = meta.get("geometry", {})
    for key in ["Lx", "Ly", "dx", "dy", "nx", "ny", "res"]:
        val = geom.get(key)
        if val is not None:
            out[f"geom_{key}"] = float(val)

    # ----------------------------------------------------
    # 3) parameters
    # ----------------------------------------------------
    par = meta.get("parameters", {})
    for key in ["k_mean", "var_rel", "corr_len_rel", "seed", "ms_scale", "coupling", "volume_fraction", "lognormal"]:
        val = par.get(key)
        if isinstance(val, (float, int, bool)):
            out[f"par_{key}"] = val

    # anisotropy
    aniso = par.get("anisotropy")
    if isinstance(aniso, (list, tuple)) and len(aniso) == ANISOTROPY_DIM:
        major, minor = float(aniso[0]), float(aniso[1])
        out["par_aniso_major"] = major
        out["par_aniso_minor"] = minor
        out["par_aniso_ratio"] = major / (minor + 1e-12)

    # ms weights
    weights = par.get("ms_weight")
    if isinstance(weights, (list, tuple)) and len(weights) == MS_WEIGHT_DIM:
        out["par_ms_w1"] = float(weights[0])
        out["par_ms_w2"] = float(weights[1])

    return out


# ======================================================================
# MAIN DATAFRAME BUILDING
# ======================================================================


def load_and_build_eval_df(parquet_path: str | Path) -> pd.DataFrame:
    """Load the artifact parquet and convert it to evaluation DataFrame."""
    parquet_path = Path(parquet_path)
    df_raw = pd.read_parquet(parquet_path)
    return build_eval_df(df_raw)


def build_eval_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build an enriched evaluation DataFrame.

    Raw parquet now contains:
        - case_index
        - npz_path
        - l2_plog
        - rel_l2_plog
        - l2_kappa
        - rel_l2_kappa
        - meta

    Output DF:
        - same as above
        - expanded meta columns
        - (optional) derived error metrics
    """
    df = df_raw.copy()

    # Extract scalar features from meta
    meta_features = df["meta"].apply(extract_meta_features)
    meta_df = pd.DataFrame(meta_features.tolist())

    # Merge everything
    df_out = pd.concat([df, meta_df], axis=1)

    # Remove heavy meta dictionary
    return df_out.drop(columns=["meta"], errors="ignore")

    # ------------------------------------------------------------------
    # OPTIONAL: Additional derived metrics
    # ------------------------------------------------------------------
    # df_out["kappa_vs_plog_ratio"] = df_out["l2_kappa"] / (df_out["l2_plog"] + 1e-12)
    # df_out["is_good_physics"] = df_out["rel_l2_kappa"] < 0.1
