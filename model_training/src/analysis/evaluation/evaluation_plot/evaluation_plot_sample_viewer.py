"""
Interactive 4x3 PINO/FNO evaluation viewer with fully independent scales.

Each subplot (prediction, ground truth, absolute error) receives:
    • its own colormap
    • its own contour value levels
    • quantile-based level spacing for smooth interpretation

This avoids shared scaling between pred and true and makes qualitative
differences deutlich erkennbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure


# =============================================================================
# NPZ CACHE
# =============================================================================

_npz_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_npz(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load prediction, ground truth and error for a single case (cached).

    Parameters
    ----------
    row : pandas.Series
        Single row with "npz_path" field.

    Returns
    -------
    tuple of np.ndarray
        (pred, gt, err) each with shape (4, H, W)

    """
    key = str(Path(row["npz_path"]))

    if key in _npz_cache:
        return _npz_cache[key]

    data = np.load(key, allow_pickle=True)
    pred = data["pred_plog"][0]
    gt = data["gt_plog"][0]
    err = data["err_plog"][0]

    _npz_cache[key] = (pred, gt, err)
    return pred, gt, err


# =============================================================================
# LEVEL COMPUTATION
# =============================================================================


_MIN_LEVEL_COUNT = 2


def compute_levels(arr: np.ndarray, n: int = 12) -> np.ndarray:
    """
    Compute contour levels based on quantiles of the input array.

    Parameters
    ----------
    arr : np.ndarray
        Input array from which to compute levels.
    n : int, optional
        Number of desired levels, by default 12.

    Returns
    -------
    np.ndarray
        Array of contour levels.

    """
    levels = np.round(np.quantile(arr, np.linspace(0, 1, n)), 2)
    levels = np.unique(levels)

    # Fallback falls kaum Variabilitaet vorhanden ist
    if len(levels) < _MIN_LEVEL_COUNT:
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmin == vmax:
            vmin -= 1e-3
            vmax += 1e-3
        levels = np.linspace(vmin, vmax, n)

    # Sicherheit: strictly increasing
    if not np.all(np.diff(levels) > 0):
        levels = np.linspace(levels[0], levels[-1], n)

    return levels


# =============================================================================
# MAIN VIEWER
# =============================================================================


def plot_sample_prediction_overview(df: pd.DataFrame, dataset_name: str) -> widgets.VBox:
    """
    Interactive multi-case viewer for PINO/FNO predictions (4x3 grid).

    Rows  : p, u, v, U
    Cols  : prediction | ground truth | absolute error

    Every subplot receives:
        • its own contour levels (quantile-based)
        • its own color scale
        • its own colorbar

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain column "npz_path" with artifact file paths.

    dataset_name : str
        Name shown in viewer title.

    Returns
    -------
    widgets.VBox
        Interactive next/previous widget and plot.

    """
    df = df.reset_index(drop=True)
    channels = ["p", "u", "v", "U"]

    # Colormaps (pred & true consistent, error separate)
    cmap_pred_true = "viridis"
    cmap_error = "Blues"

    n_levels = 12

    def plot_case(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Render a 4x3 grid for a single selected case."""
        row = df.iloc[idx]
        pred, gt, err = _load_npz(row)

        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 11))

        for i, label in enumerate(channels):
            # ------------------------------
            # Prediction
            # ------------------------------
            pred_levels = compute_levels(pred[i], n_levels)

            ax = axes[i, 0]
            im = ax.contourf(pred[i], levels=pred_levels, cmap=cmap_pred_true)
            ax.set_title(f"{label} pred")
            fig.colorbar(im, ax=ax, fraction=0.04)

            # ------------------------------
            # Ground truth
            # ------------------------------
            true_levels = compute_levels(gt[i], n_levels)

            ax = axes[i, 1]
            im = ax.contourf(gt[i], levels=true_levels, cmap=cmap_pred_true)
            ax.set_title(f"{label} true")
            fig.colorbar(im, ax=ax, fraction=0.04)

            # ------------------------------
            # Absolute error
            # ------------------------------
            err_abs = np.abs(err[i])
            err_levels = compute_levels(err_abs, n_levels)

            ax = axes[i, 2]
            im = ax.contourf(err_abs, levels=err_levels, cmap=cmap_error)
            ax.set_title(f"{label} abs(err)")
            fig.colorbar(im, ax=ax, fraction=0.04)

        fig.suptitle(f"{dataset_name} - Case {idx + 1}", fontsize=13)
        fig.tight_layout()
        return fig

    # interactive viewer
    return util.util_plot.make_interactive_plot(
        n_cases=len(df),
        plot_func=plot_case,
        df=df,
        dataset_name=dataset_name,
    )
