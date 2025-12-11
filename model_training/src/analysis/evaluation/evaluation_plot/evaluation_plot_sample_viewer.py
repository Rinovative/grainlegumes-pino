"""
Interactive 4x4 PINO/FNO evaluation viewer with physical coordinates and independent color scales for each subplot.

This viewer displays, per output field (p, u, v, U):
    - prediction
    - ground truth
    - error map (switchable: MAE or relative error)
    - aggregated permeability field kappa

IMPORTANT — RELATIVE ERROR HANDLING
Relative error is defined as:

    rel = |pred - true| / (|true| + eps) * 100

Regions where |true| < mask_threshold (default: 1e-4) are masked (set to NaN),
because relative error becomes mathematically meaningless there and would
produce misleading artefacts. This is standard practice in CFD and operator-
learning visualization.

Supported kappa aggregation logic:
    • 1 component       → return it directly
    • 2 diagonals       → mean(kxx, kyy)
    • 4 components      → (kxx + kyy) / 2
    • 9 components      → (kxx + kyy + kzz) / 3
    • fallback          → mean across all components

The module supports:
    • multiple datasets via util.util_plot.make_interactive_plot_dropdown
    • quantile-based contour-level computation with rounding
    • consistent physical axes across all fields
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# =============================================================================
# NPZ LOADING
# =============================================================================

_npz_cache: dict[
    str,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
] = {}


def _load_npz(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load prediction, ground truth, error and permeability tensor from NPZ file.

    Parameters
    ----------
    row : pandas.Series
        Must contain 'npz_path'.

    Returns
    -------
    pred, gt, err, kappa : np.ndarray
        Arrays of shape (C, H, W).
    kappa_names : list[str]
        Component names for the permeability tensor.

    """
    key = str(Path(row["npz_path"]))

    if key in _npz_cache:
        return _npz_cache[key]

    data = np.load(key, allow_pickle=True)
    pred = data["pred"][0]
    gt = data["gt"][0]
    err = data["err"][0]
    kappa = data["kappa"][0]
    kappa_names = list(data["kappa_names"])

    _npz_cache[key] = (pred, gt, err, kappa, kappa_names)
    return pred, gt, err, kappa, kappa_names


# =============================================================================
# KAPPA AGGREGATION
# =============================================================================


def aggregate_kappa(kappa: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Construct a scalar permeability field from arbitrary tensor components.

    Parameters
    ----------
    kappa : np.ndarray
        Tensor components of shape (K, H, W).
    names : list[str]
        Component names such as ['kappaxx', 'kappayy', ...].

    Returns
    -------
    np.ndarray
        Scalar permeability field of shape (H, W).

    """
    K = kappa.shape[0]
    name_to_idx = {name.lower(): i for i, name in enumerate(names)}

    diag_keys = ["kappaxx", "kappayy", "kappazz"]
    diag_indices = [name_to_idx[k] for k in diag_keys if k in name_to_idx]

    if K == 1:
        return kappa[0]

    if K == 2 and len(diag_indices) == 2:  # noqa: PLR2004
        ixx, iyy = diag_indices
        return (kappa[ixx] + kappa[iyy]) / 2.0

    if K == 4 and len(diag_indices) >= 2:  # noqa: PLR2004
        ixx = name_to_idx.get("kappaxx", diag_indices[0])
        iyy = name_to_idx.get("kappayy", diag_indices[1])
        return (kappa[ixx] + kappa[iyy]) / 2.0

    if K == 9 and len(diag_indices) == 3:  # noqa: PLR2004
        ixx = name_to_idx["kappaxx"]
        iyy = name_to_idx["kappayy"]
        izz = name_to_idx["kappazz"]
        return (kappa[ixx] + kappa[iyy] + kappa[izz]) / 3.0

    return kappa.mean(axis=0)


# =============================================================================
# LEVEL COMPUTATION
# =============================================================================

_MIN_LEVEL_COUNT = 2


def compute_levels(arr: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute contour levels based on quantiles with rounding to two significant figures.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    n : int
        Number of levels.

    Returns
    -------
    np.ndarray
        Array of contour levels.

    """
    # Remove NaNs/Infs immediately
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Raw quantile levels
    raw = np.quantile(arr, np.linspace(0, 1, n))

    vmin, vmax = float(arr.min()), float(arr.max())

    # Constant field → return safe linear levels
    if vmin == vmax:
        eps = 1e-12
        return np.linspace(vmin - eps, vmax + eps, n)

    # Replace exact zeros before log10
    raw_safe = np.where(raw == 0.0, 1e-30, raw)

    # -------------------------------
    # Round to two significant figures
    # -------------------------------
    x = raw_safe.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.floor(np.log10(np.abs(x)))
        scale = np.power(10.0, exp - 1)
        rounded = np.round(x / scale) * scale

    # Remove any NaNs produced by rounding
    rounded = np.nan_to_num(rounded, nan=vmin, posinf=vmax, neginf=vmin)

    # Remove duplicates
    levels = np.unique(rounded)

    # Too few levels → fallback to linear spacing
    if len(levels) < _MIN_LEVEL_COUNT:
        return np.linspace(vmin, vmax, n)

    # Enforce monotonicity
    if not np.all(np.diff(levels) > 0):
        return np.linspace(levels[0], levels[-1], n)

    return levels


# =============================================================================
# PLOTTING
# =============================================================================


def _apply_axis_labels(ax: Axes, row: int, col: int, Lx: float, Ly: float) -> None:
    """
    Apply consistent axis labels and enforce explicit y-ticks including 0.75.

    Behaviour:
        - Left column: show y-axis with ticks at [0, 0.25, 0.5, 0.75]
        - Bottom row: show x-axis normally
        - All other axes: hide ticklabels
        - Axis limits always full domain
    """
    # Set limits
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    # ---- Y-AXIS ----
    yticks = [0.0, 0.25, 0.5, 0.75]

    if col == 0:
        ax.set_yticks(yticks)
        ax.set_ylabel("y [m]")
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelleft=False)

    # ---- X-AXIS ----
    if row == 3:  # noqa: PLR2004
        ax.set_xlabel("x [m]")
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def choose_colorbar_formatter(vmin: float, vmax: float) -> mticker.Formatter:
    """
    Choose a colorbar formatter based on value range.

    Parameters
    ----------
    vmin : float
        Minimum value.
    vmax : float
        Maximum value.

    Returns
    -------
    matplotlib.ticker.Formatter
        Appropriate formatter instance.

    """
    vr = max(abs(vmin), abs(vmax))

    # Very small values → scientific notation
    if vr < 1e-3:  # noqa: PLR2004
        return mticker.FormatStrFormatter("%.2e")

    # Small values < 0.1 → 4 decimals
    if vr < 0.1:  # noqa: PLR2004
        return mticker.FormatStrFormatter("%.4f")

    # Medium values < 1 → 3 decimals
    if vr < 1:
        return mticker.FormatStrFormatter("%.2f")

    # Normal values < 100 → 2 decimals
    if vr < 100:  # noqa: PLR2004
        return mticker.FormatStrFormatter("%.2f")

    # Large values → no decimals
    return mticker.FormatStrFormatter("%.0f")


# =============================================================================
# MAIN VIEWER
# =============================================================================


def plot_sample_prediction_overview(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: PLR0915
    """
    Build an interactive 4x4 evaluation viewer for PINO/FNO predictions and permeability fields.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → DataFrame with columns:
            - npz_path
            - geom_Lx
            - geom_Ly

    Returns
    -------
    widgets.VBox
        Interactive UI container with dropdown.

    """
    channels = ["p", "u", "v", "U"]
    unit_map = {"p": "Pa", "u": "m/s", "v": "m/s", "U": "m/s"}

    cmap_pred_true = "turbo"
    cmap_error = "Blues"
    cmap_kappa = "viridis"
    n_levels = 10
    mask_threshold = 1e-4

    # -------------------------------------------------------------
    # Error mode selector: MAE (default) vs. relative [%]
    # -------------------------------------------------------------
    error_selector = util.util_plot_components.ui_error_mode_selector()

    def _plot(  # noqa: PLR0915
        idx: int,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        error_mode: widgets.RadioButtons,
    ) -> Figure:
        """
        Plot a single evaluation case.

        Parameters
        ----------
        idx : int
            Row index in DataFrame.
        df : pandas.DataFrame
            Dataset-level table.
        dataset_name : str
            Name of dataset.
        error_mode : widgets.RadioButtons
            RadioButtons selecting 'MAE' or 'Relative [%]'.

        Returns
        -------
        matplotlib.figure.Figure
            Complete 4x4 subplot figure.

        """
        df = df.reset_index(drop=True)
        row = df.iloc[idx]
        pred, gt, err, kappa, kappa_names = _load_npz(row)

        Lx, Ly = float(row["geom_Lx"]), float(row["geom_Ly"])
        ny, nx = pred[0].shape

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(4, 4, figsize=(20, 9))

        # Precompute aggregated kappa
        kappa_field = aggregate_kappa(kappa, kappa_names)
        kappa_levels = compute_levels(kappa_field, n_levels)

        # Precompute log10(kappa)
        kappa_log_field = np.log10(np.maximum(kappa_field, 1e-30))
        kappa_log_levels = compute_levels(kappa_log_field, n_levels)

        for r, label in enumerate(channels):
            # Prediction
            levels_pred = compute_levels(pred[r], n_levels)
            ax = axes[r, 0]
            im = ax.contourf(X, Y, pred[r], levels=levels_pred, cmap=cmap_pred_true)
            ax.set_ylim(0, Ly)
            ax.set_title(f"{label} pred [{unit_map[label]}]")
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            vmin, vmax = im.get_clim()
            formatter = choose_colorbar_formatter(vmin, vmax)
            cb.ax.yaxis.set_major_formatter(formatter)
            _apply_axis_labels(ax, r, 0, Lx, Ly)

            # Ground truth
            levels_true = compute_levels(gt[r], n_levels)
            ax = axes[r, 1]
            im = ax.contourf(X, Y, gt[r], levels=levels_true, cmap=cmap_pred_true)
            ax.set_ylim(0, Ly)
            ax.set_title(f"{label} true [{unit_map[label]}]")
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            vmin, vmax = im.get_clim()
            formatter = choose_colorbar_formatter(vmin, vmax)
            cb.ax.yaxis.set_major_formatter(formatter)
            _apply_axis_labels(ax, r, 1, Lx, Ly)

            # -----------------------------------------------------------------
            # Error panel: MAE or Relative error depending on error_mode.value
            # -----------------------------------------------------------------
            mode = error_mode.value

            if mode == "MAE":
                err_field = np.abs(err[r])
                err_field = np.nan_to_num(err_field, nan=0.0, posinf=0.0, neginf=0.0)
                vmax_err = float(np.nanquantile(err_field, 0.99))
                vmax_err = max(vmax_err, 1e-12)
                levels_err = np.linspace(err_field.min(), err_field.max(), n_levels)
                err_title = f"{label} MAE [{unit_map[label]}]"
            else:  # "Relative [%]"
                abs_err = np.abs(err[r])
                true_abs = np.abs(gt[r])
                rel = abs_err / (true_abs + 1e-12) * 100.0
                rel[true_abs < mask_threshold] = np.nan
                err_field = rel
                vmax_err = float(np.nanquantile(err_field, 0.99))
                vmax_err = max(vmax_err, 1e-6)
                levels_err = np.linspace(0.0, vmax_err, n_levels)
                err_title = f"{label} rel err [%]"

            ax = axes[r, 2]
            im = ax.contourf(X, Y, err_field, levels=levels_err, cmap=cmap_error)
            ax.set_ylim(0, Ly)
            ax.set_title(err_title)
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            vmin_cb, vmax_cb = im.get_clim()
            formatter = choose_colorbar_formatter(vmin_cb, vmax_cb)
            cb.ax.yaxis.set_major_formatter(formatter)
            _apply_axis_labels(ax, r, 2, Lx, Ly)

            # Rightmost column: permeability panels
            ax = axes[r, 3]

            if r == 0:
                # ---- Physical permeability ----
                im = ax.contourf(X, Y, kappa_field, levels=kappa_levels, cmap=cmap_kappa)
                ax.set_title("kappa [m²]")
                fig.colorbar(im, ax=ax, fraction=0.04)
                _apply_axis_labels(ax, r, 3, Lx, Ly)

            elif r == 1:
                # ---- Log10 permeability ----
                im = ax.contourf(X, Y, kappa_log_field, levels=kappa_log_levels, cmap=cmap_kappa)
                ax.set_title("log10(kappa) [m²]")
                fig.colorbar(im, ax=ax, fraction=0.04)
                _apply_axis_labels(ax, r, 3, Lx, Ly)

            else:
                # Empty for rows 2 and 3
                ax.axis("off")

        fig.suptitle(f"{dataset_name} — Case {idx + 1}", fontsize=14)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Use global navigator with dropdown; pass error_selector into plot_kwargs
    # -------------------------------------------------------------------------
    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
        enable_prev_next=True,
        extra_widgets=[error_selector],
        error_mode=error_selector,
    )
