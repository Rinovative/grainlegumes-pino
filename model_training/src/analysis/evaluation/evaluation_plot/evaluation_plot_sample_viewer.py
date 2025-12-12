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


def _aggregate_kappa(kappa: np.ndarray, names: list[str]) -> np.ndarray:
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


def _compute_levels(arr: np.ndarray, n: int = 10) -> np.ndarray:
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


def _apply_axis_labels(
    ax: Axes,
    _row: int,
    col: int,
    Lx: float,
    Ly: float,
    *,
    is_last_row: bool,
) -> None:
    """
    Apply consistent axis labels and enforce explicit y-ticks including 0.75.

    Behaviour:
        - Left column: show y-axis with ticks at [0, 0.25, 0.5, 0.75]
        - Bottom row (dynamic): show x-axis
        - All other axes: hide tick labels
        - Axis limits always full domain
    """
    # ---- Limits ----
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
    if is_last_row:
        ax.set_xlabel("x [m]")
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _choose_colorbar_formatter(vmin: float, vmax: float) -> mticker.Formatter:
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


def _overlay_streamlines(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> None:
    """
    Overlay streamlines on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    X : np.ndarray
        Meshgrid X coordinates.
    Y : np.ndarray
        Meshgrid Y coordinates.
    u : np.ndarray
        Velocity field in x-direction.
    v : np.ndarray
        Velocity field in y-direction.

    Returns
    -------
    None
        Plots streamlines directly on the provided axis.

    """
    ax.streamplot(
        X,
        Y,
        u,
        v,
        color=(0, 0, 0, 0.6),
        density=1.0,
        linewidth=0.6,
        arrowsize=0.6,
        minlength=0.1,
        integration_direction="both",
    )


# =============================================================================
# Viewer: 4x4 Prediction/GT/Error/Kappa
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
    error_selector = util.util_plot_components.ui_radio_error_mode()

    def _plot(  # noqa: PLR0915
        idx: int,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        error_mode: widgets.ValueWidget,
    ) -> Figure:
        """
        Plot a single evaluation case with prediction, ground truth, error and kappa.

        Parameters
        ----------
        idx : int
            Row index in DataFrame.
        df : pandas.DataFrame
            Dataset-level table.
        dataset_name : str
            Name of dataset.
        error_mode : widgets.Widget
            Error mode selector widget.

        Returns
        -------
        matplotlib.figure.Figure
            Complete figure with 4x4 subplots.

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

        # ---- Kappa fields ----
        kappa_field = _aggregate_kappa(kappa, kappa_names)
        kappa_levels = _compute_levels(kappa_field, n_levels)

        kappa_log_field = np.log10(np.maximum(kappa_field, 1e-30))
        kappa_log_levels = _compute_levels(kappa_log_field, n_levels)

        nrows = 4  # fixed layout

        for r, label in enumerate(channels):
            is_last_row = r == nrows - 1

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------
            ax = axes[r, 0]
            im = ax.contourf(
                X,
                Y,
                pred[r],
                levels=_compute_levels(pred[r], n_levels),
                cmap=cmap_pred_true,
            )
            if label in {"u", "v", "U"}:
                _overlay_streamlines(ax, X, Y, pred[1], pred[2])

            ax.set_title(f"{label} pred [{unit_map[label]}]")
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            cb.ax.yaxis.set_major_formatter(_choose_colorbar_formatter(*im.get_clim()))
            _apply_axis_labels(ax, r, 0, Lx, Ly, is_last_row=is_last_row)

            # -------------------------------------------------
            # Ground truth
            # -------------------------------------------------
            ax = axes[r, 1]
            im = ax.contourf(
                X,
                Y,
                gt[r],
                levels=_compute_levels(gt[r], n_levels),
                cmap=cmap_pred_true,
            )
            if label in {"u", "v", "U"}:
                _overlay_streamlines(ax, X, Y, gt[1], gt[2])

            ax.set_title(f"{label} true [{unit_map[label]}]")
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            cb.ax.yaxis.set_major_formatter(_choose_colorbar_formatter(*im.get_clim()))
            _apply_axis_labels(ax, r, 1, Lx, Ly, is_last_row=is_last_row)

            # -------------------------------------------------
            # Error
            # -------------------------------------------------
            ax = axes[r, 2]
            if error_mode.value == "MAE":
                err_field = np.abs(err[r])
                err_field = np.nan_to_num(err_field, nan=0.0)
                levels_err = np.linspace(0.0, np.nanquantile(err_field, 0.99), n_levels)
                err_title = f"{label} MAE [{unit_map[label]}]"
            else:
                abs_err = np.abs(err[r])
                true_abs = np.abs(gt[r])
                err_field = abs_err / (true_abs + 1e-12) * 100.0
                err_field[true_abs < mask_threshold] = np.nan
                levels_err = np.linspace(0.0, np.nanquantile(err_field, 0.99), n_levels)
                err_title = f"{label} rel err [%]"

            im = ax.contourf(X, Y, err_field, levels=levels_err, cmap=cmap_error)
            ax.set_title(err_title)
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            cb.ax.yaxis.set_major_formatter(_choose_colorbar_formatter(*im.get_clim()))
            _apply_axis_labels(ax, r, 2, Lx, Ly, is_last_row=is_last_row)

            # -------------------------------------------------
            # Kappa panels
            # -------------------------------------------------
            ax = axes[r, 3]
            if r == 0:
                im = ax.contourf(X, Y, kappa_field, levels=kappa_levels, cmap=cmap_kappa)
                ax.set_title("kappa [m²]")
                fig.colorbar(im, ax=ax, fraction=0.04)
                _apply_axis_labels(ax, r, 3, Lx, Ly, is_last_row=is_last_row)

            elif r == 1:
                im = ax.contourf(
                    X,
                    Y,
                    kappa_log_field,
                    levels=kappa_log_levels,
                    cmap=cmap_kappa,
                )
                ax.set_title("log10(kappa) [m²]")
                fig.colorbar(im, ax=ax, fraction=0.04)
                _apply_axis_labels(ax, r, 3, Lx, Ly, is_last_row=is_last_row)

            else:
                ax.axis("off")

        fig.suptitle(f"{dataset_name} — Case {idx + 1}", fontsize=14)
        fig.tight_layout()
        return fig

    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
        extra_widgets=[error_selector],
        error_mode=error_selector,
    )


# =============================================================================
# Viewer: Kappa Tensor Components with Error Overlay
# =============================================================================


def plot_sample_kappa_tensor_with_overlay(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: C901, PLR0915
    """
    Build an interactive evaluation viewer for permeability tensor components with error contour overlays.

    For each permeability tensor component, the physical field (kappa or log10(kappa))
    is shown together with error contour lines of a selected output channel (p, u, v, U).

    Error contours correspond to fixed global quantiles (50 %, 75 %, 90 %) of the
    selected error metric (MAE or Relative [%]).

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
        Interactive UI container with navigation and selectors.

    """
    channels = ["p", "u", "v", "U"]
    unit_map = {"p": "Pa", "u": "m/s", "v": "m/s", "U": "m/s"}

    cmap_kappa = "viridis"
    cmap_error = "Reds"
    n_kappa_levels = 10
    mask_threshold = 1e-4

    channel_selector = util.util_plot_components.ui_dropdown_channel()
    error_selector = util.util_plot_components.ui_radio_error_mode()
    kappa_scale_selector = util.util_plot_components.ui_radio_kappa_scale()

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------
    def _compute_error_field(
        *,
        err: np.ndarray,
        gt: np.ndarray,
        channel_idx: int,
        mode: str,
    ) -> np.ndarray:
        """
        Compute error field for a given channel and error mode.

        Returns
        -------
        np.ndarray
            Error field of shape (H, W).

        """
        if mode == "MAE":
            return np.nan_to_num(np.abs(err[channel_idx]), nan=0.0)

        abs_err = np.abs(err[channel_idx])
        true_abs = np.abs(gt[channel_idx])
        rel = abs_err / (true_abs + 1e-12) * 100.0
        rel[true_abs < mask_threshold] = np.nan
        return rel

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _plot(  # noqa: PLR0915
        idx: int,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        channel: widgets.ValueWidget,
        error_mode: widgets.ValueWidget,
        kappa_scale: widgets.ValueWidget,
    ) -> Figure:
        """
        Plot a single evaluation case with kappa tensor components and error overlays.

        Parameters
        ----------
        idx : int
            Row index in DataFrame.
        df : pandas.DataFrame
            Dataset-level table.
        dataset_name : str
            Name of dataset.
        channel : widgets.Widget
            Channel selector widget.
        error_mode : widgets.Widget
            Error mode selector widget.
        kappa_scale : widgets.Widget
            Kappa scale selector widget.

        Returns
        -------
        matplotlib.figure.Figure
            Complete figure with kappa components and error overlays.

        """
        df = df.reset_index(drop=True)
        row = df.iloc[idx]
        _, gt, err, kappa, kappa_names = _load_npz(row)

        Lx, Ly = float(row["geom_Lx"]), float(row["geom_Ly"])
        ny, nx = kappa.shape[1:]

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        channel_name = channel.value
        channel_idx = channels.index(channel_name)

        err_field = _compute_error_field(
            err=err,
            gt=gt,
            channel_idx=channel_idx,
            mode=error_mode.value,
        )

        # --------------------------------------------------
        # Kappa components (only existing ones)
        # --------------------------------------------------
        name_to_idx = {n.lower(): i for i, n in enumerate(kappa_names)}
        comps = [(name, kappa[i]) for name, i in name_to_idx.items()]

        # Apply kappa scaling
        if kappa_scale.value == "log10(kappa)":
            comps = [(name, np.log10(np.maximum(field, 1e-30))) for name, field in comps]
            kappa_unit = "log10(m²)"
        else:
            kappa_unit = "m²"

        n_comp = len(comps)
        ncols = int(np.ceil(np.sqrt(n_comp)))
        nrows = int(np.ceil(n_comp / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols + 1,
            figsize=(4.8 * (ncols + 1), 3.6 * nrows),
        )

        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)

        # --------------------------------------------------
        # Levels
        # --------------------------------------------------
        kappa_levels = _compute_levels(
            np.concatenate([c.ravel() for _, c in comps]),
            n_kappa_levels,
        )

        valid_err = err_field[np.isfinite(err_field)]
        if valid_err.size > 0:
            err_levels = np.quantile(valid_err, [0.75, 0.95])
            err_levels = np.unique(err_levels)
        else:
            err_levels = []

        # --------------------------------------------------
        # Kappa panels + error contours
        # --------------------------------------------------
        for i, (name, field) in enumerate(comps):
            r, c = divmod(i, ncols)
            ax = axes[r, c]

            im = ax.contourf(
                X,
                Y,
                field,
                levels=kappa_levels,
                cmap=cmap_kappa,
            )

            if len(err_levels) > 0:
                ax.contour(
                    X,
                    Y,
                    err_field,
                    levels=err_levels,
                    cmap=cmap_error,
                    linewidths=1.0,
                )

            ax.set_title(f"{name} [{kappa_unit}]")

            _apply_axis_labels(
                ax,
                r,
                c,
                Lx,
                Ly,
                is_last_row=(r == nrows - 1),
            )

            cb = fig.colorbar(im, ax=ax, fraction=0.045)
            formatter = _choose_colorbar_formatter(*im.get_clim())
            cb.ax.yaxis.set_major_formatter(formatter)

        # --------------------------------------------------
        # Right column: channel GT + error contours
        # --------------------------------------------------
        ax_gt = axes[0, -1]
        gt_levels = _compute_levels(gt[channel_idx], n_kappa_levels)

        im = ax_gt.contourf(
            X,
            Y,
            gt[channel_idx],
            levels=gt_levels,
            cmap="turbo",
        )

        if len(err_levels) > 0:
            ax_gt.contour(
                X,
                Y,
                err_field,
                levels=err_levels,
                cmap=cmap_error,
                linewidths=1.0,
            )

        ax_gt.set_title(f"{channel_name} true [{unit_map[channel_name]}]")

        cb = fig.colorbar(im, ax=ax_gt, fraction=0.045)
        formatter = _choose_colorbar_formatter(*im.get_clim())
        cb.ax.yaxis.set_major_formatter(formatter)

        _apply_axis_labels(
            ax_gt,
            0,
            ncols,
            Lx,
            Ly,
            is_last_row=(nrows == 1),
        )

        # --------------------------------------------------
        # Right column: error reference
        # --------------------------------------------------
        if nrows > 1:
            ax_err = axes[1, -1]

            im = ax_err.contourf(
                X,
                Y,
                err_field,
                levels=err_levels if len(err_levels) > 0 else _compute_levels(err_field),
                cmap=cmap_error,
            )

            unit = "MAE" if error_mode.value == "MAE" else "rel [%]"
            ax_err.set_title(f"{channel_name} error [{unit}]")

            cb = fig.colorbar(im, ax=ax_err, fraction=0.045)
            formatter = _choose_colorbar_formatter(*im.get_clim())
            cb.ax.yaxis.set_major_formatter(formatter)

            _apply_axis_labels(
                ax_err,
                1,
                ncols,
                Lx,
                Ly,
                is_last_row=(nrows - 1 == 1),
            )

        # --------------------------------------------------
        # Supertitle
        # --------------------------------------------------
        err_label = "MAE" if error_mode.value == "MAE" else "Relative error [%]"
        fig.suptitle(
            f"{dataset_name} — Case {idx + 1} — Error contours: 75 %, 95 % ({err_label}, channel {channel_name}, kappa scale: {kappa_scale.value})",
            fontsize=14,
        )

        fig.tight_layout()
        return fig

    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
        extra_widgets=[
            kappa_scale_selector,
            channel_selector,
            error_selector,
        ],
        kappa_scale=kappa_scale_selector,
        channel=channel_selector,
        error_mode=error_selector,
    )
