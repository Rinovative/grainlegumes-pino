"""
===============================================================================
evaluation_plot_architecture_sensitivity.py
===============================================================================
Plot architecture sensitivity for evaluated model runs.

Responsibilities:
  - Extract architecture parameters from current saved-run config.yaml files
  - Aggregate case-level errors to model-level points
  - Plot capacity, parameter efficiency and architecture/error trends

Design principles:
  - One model run maps to one architecture point
  - Evaluation DataFrames remain case-level inputs
  - Architecture metadata comes from model.kind and model.params

Boundaries:
  - Per-case field viewing belongs to evaluation_plot_sample_viewer
  - Model construction belongs to learning.models.factory
===============================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from src import common, experiments

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


# ======================================================================
# Helpers: architecture extraction
# ======================================================================
def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return a required config section mapping."""
    if not isinstance(value, Mapping):
        msg = f"Expected {label} to be a mapping in config.yaml."
        raise TypeError(msg)
    return value


def _infer_run_dir_from_npz_path(npz_path: Path) -> Path:
    """Infer the current saved-run directory that owns an NPZ artifact."""
    for parent in [npz_path.parent, *npz_path.parents]:
        if common.paths.resolve_run_config_path(parent).is_file():
            return parent

    msg = f"Could not find current run config.yaml for npz_path: {npz_path}"
    raise FileNotFoundError(msg)


def _load_current_run_config_from_npz_path(npz_path: str | Path) -> dict[str, Any]:
    """Load the owning run's current config.yaml for an NPZ artifact."""
    run_dir = _infer_run_dir_from_npz_path(Path(npz_path).expanduser())
    return experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(run_dir))


def _current_config_sections(cfg: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return the model, model.params, and loss sections."""
    model_cfg = _as_mapping(cfg.get("model"), label='config["model"]')
    params = _as_mapping(model_cfg.get("params"), label='config["model"]["params"]')
    loss_cfg = _as_mapping(cfg.get("loss"), label='config["loss"]')

    # Validate all required run sections even though this plot only
    # consumes model and loss metadata.
    _as_mapping(cfg.get("data"), label='config["data"]')
    _as_mapping(cfg.get("run"), label='config["run"]')

    return model_cfg, params, loss_cfg


def _mean_pair(values: Any) -> float | None:
    """Return the mean of a two-entry mode pair when available."""
    if not isinstance(values, (list, tuple)) or len(values) != 2:  # noqa: PLR2004
        return None
    return 0.5 * (float(values[0]) + float(values[1]))


def _load_arch_from_npz_path(npz_path: str | Path) -> dict[str, Any]:
    """
    Load architecture and loss parameters from config.yaml via npz_path.

    Parameters
    ----------
    npz_path : str | Path
        Path to a single evaluation npz file.

    Returns
    -------
    dict[str, Any]
        Architecture and loss parameters.

    """
    cfg = _load_current_run_config_from_npz_path(npz_path)
    _, params, loss_cfg = _current_config_sections(cfg)

    arch: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Common parameters (ALL architectures)
    # ------------------------------------------------------------------
    arch["n_layers"] = params.get("n_layers")
    arch["hidden_channels"] = params.get("hidden_channels")

    # ------------------------------------------------------------------
    # Spectral capacity across FNO and UNO with either loss composition
    # ------------------------------------------------------------------
    modes_mean = _mean_pair(params.get("n_modes"))
    if modes_mean is not None:
        # FNO-style
        arch["n_modes"] = modes_mean

    elif "modes_x" in params and "modes_y" in params:
        # UNO current config stores base modes directly.
        arch["n_modes"] = 0.5 * (float(params["modes_x"]) + float(params["modes_y"]))

    # ------------------------------------------------------------------
    # UNO-specific capacity metadata
    # ------------------------------------------------------------------
    mode_ratio = params.get("mode_ratio")
    if mode_ratio is not None:
        arch["mode_ratio"] = float(mode_ratio)

    # ------------------------------------------------------------------
    # Physics weights (PI models only, otherwise None)
    # ------------------------------------------------------------------
    physics_cfg = loss_cfg.get("physics")
    physics_cfg = physics_cfg if isinstance(physics_cfg, Mapping) else {}
    residual_weight = physics_cfg.get("residual_weight")
    boundary_weight = physics_cfg.get("boundary_weight")
    arch["lambda_phys"] = residual_weight.get("target") if isinstance(residual_weight, Mapping) else None
    arch["lambda_p"] = boundary_weight.get("target") if isinstance(boundary_weight, Mapping) else None

    return arch


def _summarise_model(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise model performance and architecture from evaluation DataFrame.

    One model = one architecture point.
    1. Aggregate error as median relative L2 over all cases.
    2. Load architecture parameters from config.yaml via npz_path.
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
