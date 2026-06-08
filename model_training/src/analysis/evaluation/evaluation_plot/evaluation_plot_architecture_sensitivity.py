"""
===============================================================================
evaluation_plot_architecture_sensitivity.py
===============================================================================
Plot architecture sensitivity for evaluated model runs.

Responsibilities:
  - Extract architecture parameters from saved run configs
  - Aggregate case-level errors to model-level points
  - Plot capacity, parameter efficiency and architecture/error trends

Design principles:
  - One model run maps to one architecture point
  - Evaluation DataFrames remain case-level inputs
  - Architecture metadata is loaded only when needed

Boundaries:
  - Per-case field viewing belongs to evaluation_plot_sample_viewer
  - Model construction belongs to learning.models.factory
===============================================================================
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


# ======================================================================
# Helpers: architecture extraction
# ======================================================================


def _load_arch_from_npz_path(npz_path: str | Path) -> dict[str, Any]:
    """
    Load architecture and physics parameters from config.json via npz_path.

    Parameters
    ----------
    npz_path : str | Path
        Path to a single evaluation npz file.

    Returns
    -------
    dict[str, Any]
        Architecture and physics parameters.

    """
    npz_path_p = Path(npz_path)

    # npz -> cases -> analysis -> (id|ood) -> run_dir
    run_dir = None
    for parent in npz_path_p.parents:
        if parent.name == "analysis":
            run_dir = parent.parent
            break

    if run_dir is None:
        msg = f"Could not determine run directory from npz_path: {npz_path}"
        raise FileNotFoundError(msg)

    cfg_path = run_dir / "config.json"

    if not cfg_path.exists():
        msg = f"config.json not found for run: {run_dir}"
        raise FileNotFoundError(msg)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    params = model_cfg.get("model_params", {})
    physics = cfg.get("physics", {})

    arch: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Common parameters (ALL architectures)
    # ------------------------------------------------------------------
    arch["n_layers"] = params.get("n_layers")
    arch["hidden_channels"] = params.get("hidden_channels")

    # ------------------------------------------------------------------
    # Spectral capacity (FNO / PI-FNO / UNO unified)
    # ------------------------------------------------------------------
    if "n_modes" in params and isinstance(params["n_modes"], list):
        # FNO-style
        mx, my = params["n_modes"]
        arch["n_modes"] = 0.5 * (mx + my)

    elif "uno_n_modes" in params and isinstance(params["uno_n_modes"], list):
        # UNO: all layers have same modes → take first
        mx, my = params["uno_n_modes"][0]
        arch["n_modes"] = 0.5 * (mx + my)

    # ------------------------------------------------------------------
    # UNO-specific: U-shape scaling (REAL differentiator)
    # ------------------------------------------------------------------
    if "uno_scalings" in params and isinstance(params["uno_scalings"], list):
        scales = [float(s[0]) for s in params["uno_scalings"]]
        arch["uno_scale_mean"] = float(sum(scales) / len(scales))
        arch["uno_scale_max"] = float(max(scales))

    # ------------------------------------------------------------------
    # Physics weights (PI models only, otherwise None)
    # ------------------------------------------------------------------
    arch["lambda_phys"] = physics.get("lambda_phys")
    arch["lambda_p"] = physics.get("lambda_p")

    return arch


def _summarise_model(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise model performance and architecture from evaluation DataFrame.

    One model = one architecture point.
    1. Aggregate error as median relative L2 over all cases.
    2. Load architecture parameters from config.json via npz_path.
    3. Return combined summary dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation DataFrame for a single model.

    Returns
    -------
    dict[str, Any]
        Summary dictionary with keys:
            - "rel_l2_median": median relative L2 error over all cases
            - architecture parameters (e.g. "n_layers", "hidden_channels", "n_modes", etc.)

    """
    if df.empty:
        msg = "Empty evaluation DataFrame"
        raise ValueError(msg)

    npz_path = df.iloc[0]["npz_path"]
    arch = _load_arch_from_npz_path(npz_path)

    return {
        "rel_l2_median": float(df["rel_l2"].median()),
        **arch,
    }


# ======================================================================
# Plots: architecture sensitivity
# ======================================================================


def plot_error_vs_architecture_parameters(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse how architecture parameters influence model error.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of evaluation DataFrames per model.

    Returns
    -------
    Figure
        Matplotlib Figure object.

    """
    summaries = {name: _summarise_model(df) for name, df in datasets.items()}

    arch_params = sorted(
        {
            key
            for summary in summaries.values()
            for key, value in summary.items()
            if key != "rel_l2_median" and value is not None and isinstance(value, (int, float))
        }
    )

    n_cols = 3
    n_rows = math.ceil(len(arch_params) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.5 * n_cols, 4.5 * n_rows),
        sharey=True,
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax, param in zip(axes_flat, arch_params, strict=False):
        for name, s in summaries.items():
            if param not in s or s[param] is None:
                continue

            x = float(s[param])
            y = float(s["rel_l2_median"])

            ax.scatter(x, y, s=60)
            ax.annotate(name, (x, y), fontsize=8)

        ax.set_xlabel(param)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    axes_flat[0].set_ylabel("Median relative L2")

    for ax in axes_flat[len(arch_params) :]:
        ax.remove()

    fig.suptitle("Error vs architecture parameters")
    fig.tight_layout()
    return fig


# ======================================================================
# Plots: capacity vs performance
# ======================================================================


def plot_capacity_vs_performance(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse the tradeoff between model capacity and predictive performance.

    Capacity proxy:
        hidden_channels x n_layers x n_modes

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of evaluation DataFrames per model.

    """
    fig, ax = plt.subplots(figsize=(18, 8))

    for name, df in datasets.items():
        s = _summarise_model(df)

        if not {"hidden_channels", "n_layers", "n_modes"}.issubset(s):
            continue

        capacity = float(s["hidden_channels"] * s["n_layers"] * s["n_modes"])
        error = float(s["rel_l2_median"])

        ax.scatter(capacity, error, s=60)
        ax.annotate(name, (capacity, error), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Capacity proxy")
    ax.set_ylabel("Median relative L2")
    ax.set_title("Capacity vs performance")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


# ======================================================================
# Plots: parameter efficiency
# ======================================================================


def plot_parameter_efficiency(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse parameter efficiency across architectures.

    Parameter efficiency proxy:
        relative L2 x (hidden_channels x n_layers x n_modes)

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of evaluation DataFrames per model.

    Returns
    -------
    Figure
        Matplotlib Figure object.

    """
    fig, ax = plt.subplots(figsize=(18, 8))

    for name, df in datasets.items():
        s = _summarise_model(df)

        if not {"hidden_channels", "n_layers", "n_modes"}.issubset(s):
            continue

        capacity = float(s["hidden_channels"] * s["n_layers"] * s["n_modes"])
        efficiency = float(s["rel_l2_median"] * capacity)

        ax.scatter(capacity, efficiency, s=60)
        ax.annotate(name, (capacity, efficiency), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Capacity proxy")
    ax.set_ylabel("Relative L2 x capacity")
    ax.set_title("Parameter efficiency")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig
