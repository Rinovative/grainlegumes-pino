"""Interactive 4x3 PINO/FNO evaluation viewer with physical coordinates and idependent color scales for each subplot."""

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
    Load model prediction, ground truth and error arrays from an NPZ artifact.

    Parameters
    ----------
    row : pandas.Series
        Must contain a field ``npz_path`` pointing to the artifact file.

    Returns
    -------
    tuple of np.ndarray
        (pred, gt, err) each shaped (4, H, W).

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


def compute_levels(arr: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute contour levels using quantiles for stable and consistent plots.

    Parameters
    ----------
    arr : np.ndarray
        Field for which contour levels are computed.
    n : int, optional
        Number of levels, default is 10.

    Returns
    -------
    np.ndarray
        Sorted level boundaries for contourf.

    """
    levels = np.round(np.quantile(arr, np.linspace(0, 1, n)), 2)
    levels = np.unique(levels)

    if len(levels) < _MIN_LEVEL_COUNT:
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmin == vmax:
            vmin -= 1e-3
            vmax += 1e-3
        levels = np.linspace(vmin, vmax, n)

    if not np.all(np.diff(levels) > 0):
        levels = np.linspace(levels[0], levels[-1], n)

    return levels


# =============================================================================
# MAIN VIEWER
# =============================================================================


def plot_sample_prediction_overview(df: pd.DataFrame, dataset_name: str) -> widgets.VBox:  # noqa: PLR0915
    """
    Build an interactive viewer for PINO/FNO predictions, comparing prediction, ground truth and relative error for each field.

    The viewer displays:
        • p, u, v, U predictions
        • corresponding ground truth fields
        • relative error maps in percent

    IMPORTANT — RELATIVE ERROR HANDLING:
        Relative error is defined as

            rel = |pred - true| / (|true| + eps) * 100

        Regions where |true| < mask_threshold (default: 1e-4) are masked
        (set to NaN) because relative error becomes mathematically meaningless
        there and would produce misleading artefacts. This is standard practice
        in CFD and operator-learning visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
            - ``npz_path`` : artifact locations
            - ``geom_Lx``, ``geom_Ly`` : domain size in meters
    dataset_name : str
        Name shown in figure titles.

    Returns
    -------
    widgets.VBox
        A widget that allows interactive case navigation.

    """
    df = df.reset_index(drop=True)
    channels = ["p", "u", "v", "U"]

    unit_map = {
        "p": "Pa",
        "u": "m/s",
        "v": "m/s",
        "U": "m/s",
    }

    cmap_pred_true = "turbo"
    cmap_error = "Blues"

    n_levels = 10
    mask_threshold = 1e-4  # threshold for masking small |true|

    def plot_case(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """
        Render a single case of the interactive viewer.

        Parameters
        ----------
        idx : int
            Index of the selected simulation case.
        df : pandas.DataFrame
            Dataset table containing geometry and NPZ paths.
        dataset_name : str
            Name displayed as figure title.

        Returns
        -------
        Figure
            A matplotlib figure with 12 subplots.

        """
        row = df.iloc[idx]
        pred, gt, err = _load_npz(row)

        # Physical coordinates
        Lx = float(row["geom_Lx"])
        Ly = float(row["geom_Ly"])

        ny, nx = pred[0].shape
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 11))

        for i, label in enumerate(channels):
            # ----------------------------------------------------------
            # Prediction
            # ----------------------------------------------------------
            pred_levels = compute_levels(pred[i], n_levels)
            ax = axes[i, 0]
            im = ax.contourf(X, Y, pred[i], levels=pred_levels, cmap=cmap_pred_true)
            ax.set_title(f"{label} pred [{unit_map[label]}]")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_yticks(np.linspace(0, Ly, 6))
            fig.colorbar(im, ax=ax, fraction=0.04)

            # ----------------------------------------------------------
            # Ground truth
            # ----------------------------------------------------------
            true_levels = compute_levels(gt[i], n_levels)
            ax = axes[i, 1]
            im = ax.contourf(X, Y, gt[i], levels=true_levels, cmap=cmap_pred_true)
            ax.set_title(f"{label} true [{unit_map[label]}]")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_yticks(np.linspace(0, Ly, 6))
            fig.colorbar(im, ax=ax, fraction=0.04)

            # ----------------------------------------------------------
            # Relative error (%) — masked
            # ----------------------------------------------------------
            true_val = gt[i]
            err_abs = np.abs(err[i])

            rel = err_abs / (np.abs(true_val) + 1e-12)
            rel *= 100.0

            # Mask small |true|
            mask = np.abs(true_val) < mask_threshold
            rel = rel.astype(float)
            rel[mask] = np.nan

            error_threshold = 1e-6

            vmax = float(np.nanquantile(rel, 0.99))
            if vmax <= error_threshold:
                vmax = float(np.nanmax(rel) + error_threshold)

            levels = np.round(np.linspace(0.0, vmax, n_levels), 2)

            ax = axes[i, 2]
            im = ax.contourf(X, Y, rel, levels=levels, cmap=cmap_error)
            ax.set_title(f"{label} rel err (%)")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_yticks(np.linspace(0, Ly, 6))
            fig.colorbar(im, ax=ax, fraction=0.04)

        fig.suptitle(f"{dataset_name} - Case {idx + 1}", fontsize=13)
        fig.tight_layout()
        return fig

    return util.util_plot.make_interactive_plot(
        n_cases=len(df),
        plot_func=plot_case,
        df=df,
        dataset_name=dataset_name,
    )
