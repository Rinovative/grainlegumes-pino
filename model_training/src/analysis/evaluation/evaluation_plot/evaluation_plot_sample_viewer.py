"""
===============================================================================
evaluation_plot_sample_viewer.py
===============================================================================
Build per-case viewers for fields, errors and permeability context.

Responsibilities:
  - Display prediction, target and error fields per case
  - Provide field-specific color scaling and error modes
  - Visualize permeability context alongside model errors

Design principles:
  - Viewers preserve physical coordinates
  - Error masking is explicit for near-zero targets
  - Permeability aggregation is visualization-only

Boundaries:
  - Global summary plots belong to evaluation_plot_global_error_analysis
  - Physical tensor ordering belongs to domain.permeability

Notes:
  Relative error handling:
    - Relative error is defined as: rel = |pred - true| / (|true| + denominator floor) * 100
    - relative error is masked (set to NaN) where ground-truth magnitude is below the threshold (default: 1e-4)
    - masking avoids misleading percentages in near-zero regions

  Kappa aggregation:
    - diagonal tensor components are aggregated only for visualization
    - single-component fields pass through unchanged
===============================================================================

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from src import analysis, domain

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# =============================================================================
# CHANNELS AND UNITS
# =============================================================================
CHANNELS = domain.fields.ANALYSIS_FIELDS
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}
UNIT_MAP = {
    "p": "Pa",
    "u": "m/s",
    "v": "m/s",
    "U": "m/s",
}

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
    key = str(Path(row.loc["npz_path"]))

    if key in _npz_cache:
        return _npz_cache[key]

    data = np.load(key, allow_pickle=False)
    pred = np.asarray(data["pred"])
    gt = np.asarray(data["gt"])
    err = np.asarray(data["err"])
    kappa = np.asarray(data["kappa"])
    kappa_names = [str(n) for n in data["kappa_names"]]

    _npz_cache[key] = (pred, gt, err, kappa, kappa_names)
    return pred, gt, err, kappa, kappa_names


# =============================================================================
# KAPPA AGGREGATION
# =============================================================================


def _aggregate_kappa(kappa: np.ndarray, kappa_names: list[str]) -> np.ndarray:
    """
    Aggregate permeability tensor to a scalar field using domain-field indexing.

    Parameters
    ----------
    kappa : np.ndarray
        Permeability tensor components of shape (C, H, W).
    kappa_names : list[str]
        Component names for the permeability tensor.

    Returns
    -------
    np.ndarray
        Aggregated permeability field of shape (H, W).

    """
    idx = {name: i for i, name in enumerate(kappa_names)}

    # ------------------------------------------------------------------
    # Diagonal-only cases (explicit, domain-contract driven)
    # ------------------------------------------------------------------
    if {"kxx", "kyy", "kzz"}.issubset(idx):
        return (kappa[idx["kxx"]] + kappa[idx["kyy"]] + kappa[idx["kzz"]]) / 3.0

    if {"kxx", "kyy"}.issubset(idx):
        return 0.5 * (kappa[idx["kxx"]] + kappa[idx["kyy"]])

    # ------------------------------------------------------------------
    # Single component (e.g. isotropic kappa)
    # ------------------------------------------------------------------
    if len(idx) == 1:
        return kappa[next(iter(idx.values()))]

    # ------------------------------------------------------------------
    # Generic summary: explicit mean over provided components
    # (visualisation only, no physics implied)
    # ------------------------------------------------------------------
    return np.mean(
        np.stack([kappa[i] for i in idx.values()], axis=0),
        axis=0,
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
    cmap_pred_true = "turbo"
    cmap_error = "Blues"
    cmap_kappa = "viridis"
    n_levels = 10
    mask_threshold = 1e-4

    # -------------------------------------------------------------
    # Widgets
    # -------------------------------------------------------------
    error_selector = analysis.ui.components.ui_radio_error_mode()
    pred_scale_selector = analysis.ui.components.ui_radio_pred_scale_mode()

    def _plot(  # noqa: PLR0915
        idx: int,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        error_mode: widgets.ValueWidget,
        pred_scale_mode: widgets.ValueWidget,
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
        pred_scale_mode : widgets.Widget
            Prediction scale mode selector widget.

        Returns
        -------
        matplotlib.figure.Figure
            Complete figure with 4x4 subplots.

        """
        df = df.reset_index(drop=True)
        row = df.iloc[idx]
        pred, gt, err, kappa, kappa_names = _load_npz(row)

        Lx = float(row.loc["geometry_Lx"])
        Ly = float(row.loc["geometry_Ly"])
        ny, nx = pred[CHANNEL_INDICES[CHANNELS[0]]].shape

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(4, 4, figsize=(20, 9))

        # ---- Kappa fields ----
        kappa_field = _aggregate_kappa(kappa, kappa_names)
        kappa_levels = analysis.ui.components.compute_levels(kappa_field, n_levels)

        kappa_log_field = np.log10(np.maximum(kappa_field, 1e-30))
        kappa_log_levels = analysis.ui.components.compute_levels(kappa_log_field, n_levels)

        nrows = 4  # fixed layout
        shared_mode = pred_scale_mode.value == "Shared (GT)"

        for r, label in enumerate(CHANNELS):
            is_last_row = r == nrows - 1
            k = CHANNEL_INDICES[label]

            # =================================================
            # SCALE SELECTION
            # =================================================
            if shared_mode:
                # Shared scaling driven by GT
                gt_field = gt[k]
                shared_levels = analysis.ui.components.compute_levels(gt_field, n_levels)
                vmin = float(shared_levels.min())
                vmax = float(shared_levels.max())
            else:
                # Independent scales
                gt_field = gt[k]
                gt_levels = analysis.ui.components.compute_levels(gt_field, n_levels)

                pred_field = pred[k]
                pred_levels = analysis.ui.components.compute_levels(pred_field, n_levels)

            # =================================================
            # Prediction
            # =================================================
            ax = axes[r, 0]

            if shared_mode:
                pred_field = pred[k]
                pred_plot = np.ma.masked_outside(pred_field, vmin, vmax)

                cmap_obj = plt.get_cmap(cmap_pred_true).copy()
                cmap_obj.set_bad("white")

                im = ax.contourf(
                    X,
                    Y,
                    pred_plot,
                    levels=shared_levels,
                    cmap=cmap_obj,
                )

                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=shared_levels)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    vmin,
                    vmax,
                    ticks=shared_levels,
                )
                cb.update_ticks()
            else:
                pred_field = pred[k]

                im = ax.contourf(
                    X,
                    Y,
                    pred_field,
                    levels=pred_levels,
                    cmap=cmap_pred_true,
                )

                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=pred_levels)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    float(pred_levels.min()),
                    float(pred_levels.max()),
                    ticks=pred_levels,
                )
                cb.update_ticks()

            if label in {"u", "v", "U"}:
                u = pred[CHANNEL_INDICES["u"]]
                v = pred[CHANNEL_INDICES["v"]]
                analysis.ui.components.overlay_streamlines(ax, X, Y, u, v)

            ax.set_title(f"{label} pred [{UNIT_MAP[label]}]")
            analysis.ui.components.apply_axis_labels(ax, 0, Lx, Ly, is_last_row=is_last_row)

            # =================================================
            # Ground truth
            # =================================================
            ax = axes[r, 1]

            if shared_mode:
                im = ax.contourf(
                    X,
                    Y,
                    gt_field,
                    levels=shared_levels,
                    cmap=cmap_pred_true,
                )

                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=shared_levels)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    vmin,
                    vmax,
                    ticks=shared_levels,
                )
                cb.update_ticks()
            else:
                im = ax.contourf(
                    X,
                    Y,
                    gt_field,
                    levels=gt_levels,
                    cmap=cmap_pred_true,
                )

                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=gt_levels)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    float(gt_levels.min()),
                    float(gt_levels.max()),
                    ticks=gt_levels,
                )
                cb.update_ticks()

            if label in {"u", "v", "U"}:
                u = gt[CHANNEL_INDICES["u"]]
                v = gt[CHANNEL_INDICES["v"]]
                analysis.ui.components.overlay_streamlines(ax, X, Y, u, v)

            ax.set_title(f"{label} true [{UNIT_MAP[label]}]")
            analysis.ui.components.apply_axis_labels(ax, 1, Lx, Ly, is_last_row=is_last_row)

            # =================================================
            # Error
            # =================================================
            ax = axes[r, 2]

            if error_mode.value == "MAE":
                err_field = np.abs(err[k]).astype(float)
                err_field = np.nan_to_num(err_field, nan=0.0)
                err_title = f"{label} MAE [{UNIT_MAP[label]}]"
            else:
                abs_err = np.abs(err[k]).astype(float)
                true_abs = np.abs(gt[k]).astype(float)
                err_field = abs_err / (true_abs + 1e-12) * 100.0
                err_field[true_abs < mask_threshold] = np.nan
                err_title = f"{label} rel err [%]"

            valid = err_field[np.isfinite(err_field)]
            if valid.size == 0:
                ax.set_title(err_title)
                ax.axis("off")
            else:
                clip_q = 0.99
                vmax_err = float(np.nanquantile(valid, clip_q))
                vmax_err = max(vmax_err, 1e-12)

                levels_err = np.linspace(0.0, vmax_err, n_levels, dtype=np.float64)

                err_plot = np.ma.masked_greater(err_field, vmax_err)
                cmap_obj = plt.get_cmap(cmap_error).copy()
                cmap_obj.set_bad("white")

                im = ax.contourf(X, Y, err_plot, levels=levels_err, cmap=cmap_obj)

                ax.set_title(err_title)
                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=levels_err)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    0.0,
                    vmax_err,
                    ticks=levels_err,
                )
                cb.update_ticks()

                analysis.ui.components.apply_axis_labels(ax, 2, Lx, Ly, is_last_row=is_last_row)

            # =================================================
            # Kappa panels
            # =================================================
            ax = axes[r, 3]
            if r == 0:
                im = ax.contourf(X, Y, kappa_field, levels=kappa_levels, cmap=cmap_kappa)
                ax.set_title("kappa [m²]")
                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=kappa_levels)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    float(kappa_levels.min()),
                    float(kappa_levels.max()),
                    ticks=kappa_levels,
                )
                cb.update_ticks()
                analysis.ui.components.apply_axis_labels(ax, 3, Lx, Ly, is_last_row=is_last_row)

            elif r == 1:
                im = ax.contourf(
                    X,
                    Y,
                    kappa_log_field,
                    levels=kappa_log_levels,
                    cmap=cmap_kappa,
                )
                ax.set_title("log10(kappa) [m²]")
                cb = fig.colorbar(im, ax=ax, fraction=0.04, ticks=kappa_log_levels)
                cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                    float(kappa_log_levels.min()),
                    float(kappa_log_levels.max()),
                    ticks=kappa_log_levels,
                )
                cb.update_ticks()
                analysis.ui.components.apply_axis_labels(ax, 3, Lx, Ly, is_last_row=is_last_row)

            else:
                ax.axis("off")

        fig.suptitle(
            f"{dataset_name} - Case {idx + 1} - Scale: {pred_scale_mode.value}",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
        extra_widgets=[pred_scale_selector, error_selector],
        error_mode=error_selector,
        pred_scale_mode=pred_scale_selector,
    )


# =============================================================================
# Viewer: Kappa Tensor Components with Error Overlay
# =============================================================================


def plot_sample_kappa_tensor_with_overlay(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: PLR0915
    """
    Build an interactive viewer for permeability tensor components with error overlay.

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
    cmap_kappa = "viridis"
    cmap_error = "Reds"
    cmap_offdiag = "viridis"
    n_kappa_levels = 11
    mask_threshold = 1e-4

    channel_selector = analysis.ui.components.ui_dropdown_channel()
    error_selector = analysis.ui.components.ui_radio_error_mode()
    kappa_scale_selector = analysis.ui.components.ui_radio_kappa_scale()

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------
    def _compute_error_field(*, err: np.ndarray, gt: np.ndarray, channel_idx: int, mode: str) -> np.ndarray:
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
    def _plot(
        idx: int,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        channel: widgets.ValueWidget,
        error_mode: widgets.ValueWidget,
        kappa_scale: widgets.ValueWidget,
    ) -> Figure:
        """
        Plot a single evaluation case with kappa tensor components and error overlay.

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
            Complete figure with kappa tensor subplots.

        """
        df = df.reset_index(drop=True)
        row = df.iloc[idx]
        _, gt, err, kappa, kappa_names = _load_npz(row)

        Lx = float(row.loc["geometry_Lx"])
        Ly = float(row.loc["geometry_Ly"])
        ny, nx = kappa.shape[1:]

        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        channel_name = channel.value
        channel_idx = CHANNEL_INDICES[channel_name]

        err_field = _compute_error_field(
            err=err,
            gt=gt,
            channel_idx=channel_idx,
            mode=error_mode.value,
        )

        # --------------------------------------------------
        # Tensor layout
        # --------------------------------------------------
        tensor_layout = [
            ("kxx", "kxx"),
            ("kxy", "kxy"),
            ("kyx", "kxy"),
            ("kyy", "kyy"),
        ]

        name_to_idx = {name: i for i, name in enumerate(kappa_names)}
        comps: list[tuple[str, np.ndarray]] = []

        for display, source in tensor_layout:
            if source in name_to_idx:
                comps.append((display, kappa[name_to_idx[source]]))
            else:
                comps.append((display, np.full_like(kappa[0], np.nan)))

        nrows, ncols = 2, 2
        use_log = kappa_scale.value == "log10(kappa)"

        fig, axes = plt.subplots(
            nrows,
            ncols + 1,
            figsize=(4.8 * (ncols + 1), 3.6 * nrows),
        )

        # --------------------------------------------------
        # Collect values for scaling
        # --------------------------------------------------
        diag_vals = np.concatenate(
            [f.ravel() for n, f in comps if n in {"kxx", "kyy"}],
            axis=0,
        )

        offdiag_vals = np.concatenate(
            [f.ravel() for n, f in comps if n in {"kxy", "kyx"}],
            axis=0,
        )

        # --------------------------------------------------
        # Contour levels for diagonal permeability components
        # --------------------------------------------------
        if use_log and diag_vals.size > 0:
            lo = np.nanpercentile(diag_vals, 5.0)
            hi = np.nanpercentile(diag_vals, 95.0)
            if lo > 0.0 and hi > lo:
                levels_diag = np.logspace(np.log10(lo), np.log10(hi), n_kappa_levels)
            else:
                levels_diag = analysis.ui.components.compute_levels(diag_vals, n_kappa_levels)
        else:
            levels_diag = analysis.ui.components.compute_levels(diag_vals, n_kappa_levels)

        # --------------------------------------------------
        # Quantile-bounded contour levels for off-diagonal components
        # use their observed range rather than a symmetric interval.
        # --------------------------------------------------
        if offdiag_vals.size > 0:
            vals = offdiag_vals[np.isfinite(offdiag_vals)]

            if vals.size > 0:
                lo = float(np.nanpercentile(vals, 5.0))  # z.B. 5 %
                hi = float(np.nanpercentile(vals, 95.0))  # z.B. 95 %

                levels_offdiag = np.linspace(lo, hi, n_kappa_levels) if hi > lo else analysis.ui.components.compute_levels(vals, n_kappa_levels)
            else:
                levels_offdiag = analysis.ui.components.compute_levels(
                    offdiag_vals,
                    n_kappa_levels,
                )
        else:
            levels_offdiag = analysis.ui.components.compute_levels(
                offdiag_vals,
                n_kappa_levels,
            )

        # --------------------------------------------------
        # Error contour levels
        # --------------------------------------------------
        valid_err = err_field[np.isfinite(err_field)]
        err_levels = np.unique(np.quantile(valid_err, [0.75, 0.95])) if valid_err.size > 0 else np.empty(0, dtype=float)

        # --------------------------------------------------
        # Kappa panels
        # --------------------------------------------------
        for i, (name, field) in enumerate(comps):
            r, c = divmod(i, ncols)
            ax = axes[r, c]

            if name in {"kxx", "kyy"}:
                levels = levels_diag
                cmap = cmap_kappa
            else:
                levels = levels_offdiag
                cmap = cmap_offdiag

            im = ax.contourf(X, Y, field, levels=levels, cmap=cmap)

            if err_levels.size > 0:
                ax.contour(
                    X,
                    Y,
                    err_field,
                    levels=err_levels,
                    cmap=cmap_error,
                    linewidths=1.0,
                )

            if name in {"kxx", "kyy"} and use_log:
                ax.set_title(f"{name} [m², log scale]")
            else:
                ax.set_title(f"{name} [m²]")

            analysis.ui.components.apply_axis_labels(
                ax,
                c,
                Lx,
                Ly,
                is_last_row=(r == nrows - 1),
            )

            cb = fig.colorbar(im, ax=ax, fraction=0.045, ticks=levels)
            cb.formatter = analysis.ui.components.choose_colorbar_formatter(float(levels.min()), float(levels.max()))
            cb.update_ticks()

        # --------------------------------------------------
        # Right column: GT
        # --------------------------------------------------
        ax_gt = axes[0, -1]
        gt_levels = analysis.ui.components.compute_levels(gt[channel_idx], n_kappa_levels)

        im = ax_gt.contourf(X, Y, gt[channel_idx], levels=gt_levels, cmap="turbo")

        if err_levels.size > 0:
            ax_gt.contour(X, Y, err_field, levels=err_levels, cmap=cmap_error, linewidths=1.0)

        ax_gt.set_title(f"{channel_name} true [{UNIT_MAP[channel_name]}]")
        cb = fig.colorbar(im, ax=ax_gt, fraction=0.045, ticks=gt_levels)
        cb.formatter = analysis.ui.components.choose_colorbar_formatter(float(gt_levels.min()), float(gt_levels.max()))
        cb.update_ticks()

        analysis.ui.components.apply_axis_labels(ax_gt, ncols, Lx, Ly, is_last_row=False)

        # --------------------------------------------------
        # Right column: error field
        # --------------------------------------------------
        ax_err = axes[1, -1]

        valid = err_field[np.isfinite(err_field)]
        unit = "MAE" if error_mode.value == "MAE" else "rel [%]"
        ax_err.set_title(f"{channel_name} error [{unit}]")

        if valid.size == 0:
            ax_err.axis("off")
        else:
            clip_q = 0.99
            vmax = float(np.nanquantile(valid, clip_q))
            vmax = max(vmax, 1e-12)

            levels = np.linspace(0.0, vmax, n_kappa_levels, dtype=np.float64)

            err_plot = np.ma.masked_greater(err_field, vmax)
            cmap_obj = plt.get_cmap(cmap_error).copy()
            cmap_obj.set_bad("white")

            im = ax_err.contourf(X, Y, err_plot, levels=levels, cmap=cmap_obj)

            cb = fig.colorbar(im, ax=ax_err, fraction=0.045, ticks=levels)
            cb.formatter = analysis.ui.components.choose_colorbar_formatter(0.0, vmax)
            cb.update_ticks()

        analysis.ui.components.apply_axis_labels(ax_err, ncols, Lx, Ly, is_last_row=True)

        # --------------------------------------------------
        # Supertitle
        # --------------------------------------------------
        err_label = "MAE" if error_mode.value == "MAE" else "Relative error [%]"
        fig.suptitle(
            f"{dataset_name} — Case {idx + 1} — Error contours: 75 %, 95 % "
            f"({err_label}, channel {channel_name}, "
            f"kappa scale: diag={kappa_scale.value}, offdiag=linear)",
            fontsize=14,
        )

        fig.tight_layout()
        return fig

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
        extra_widgets=[kappa_scale_selector, channel_selector, error_selector],
        kappa_scale=kappa_scale_selector,
        channel=channel_selector,
        error_mode=error_selector,
    )


# =============================================================================
# Viewer: p/U Two-Model Comparison (GT + 2 Predictions)
# =============================================================================


def plot_pu_two_model_comparison(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: PLR0915
    """
    Build an interactive 3x2 viewer to compare p and U between two selectable models.

    Layout
    ------
    Rows (top -> bottom):
        - GT
        - Model 1 prediction
        - Model 2 prediction

    Columns (left -> right):
        - p
        - U

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping model_name -> DataFrame with columns:
            - npz_path
            - geometry_Lx
            - geometry_Ly
        Optional:
            - case_index (used to align cases across models)

    Returns
    -------
    widgets.VBox
        Interactive UI container.

    """
    cmap_pred_true = "turbo"
    n_levels = 10

    model_names = list(datasets.keys())
    if len(model_names) == 0:
        msg = "datasets is empty. Expected model_name -> DataFrame."
        raise ValueError(msg)

    # Use a safe base length so the case slider never goes out of range
    min_len = min(len(df) for df in datasets.values())
    if min_len <= 0:
        msg = "All model DataFrames are empty."
        raise ValueError(msg)

    base_df = next(iter(datasets.values())).reset_index(drop=True).iloc[:min_len].copy()
    base_datasets = {"Model comparison": base_df}

    dd_model_1 = widgets.Dropdown(
        options=model_names,
        value=model_names[0],
        description="Model 1:",
        layout=widgets.Layout(width="340px"),
    )
    dd_model_2 = widgets.Dropdown(
        options=model_names,
        value=(model_names[1] if len(model_names) > 1 else model_names[0]),
        description="Model 2:",
        layout=widgets.Layout(width="340px"),
    )

    # Scale toggle (shared vs independent)
    pu_scale_selector = analysis.ui.components.ui_radio_pred_scale_mode()

    def _row_by_case_index(df: pd.DataFrame, case_index: int) -> pd.Series | None:
        if "case_index" not in df.columns:
            return None
        hit = df.loc[df.loc[:, "case_index"] == case_index]
        if hit.shape[0] == 0:
            return None
        return hit.iloc[0]

    def _plot(
        idx: int,
        *,
        df: pd.DataFrame,
        dataset_name: str,
        model_1: widgets.ValueWidget,
        model_2: widgets.ValueWidget,
        pu_scale_mode: widgets.ValueWidget,
    ) -> Figure:
        """
        Plot p and U for GT and two selected models, aligned by case_index if available.

        Parameters
        ----------
        idx : int
            Row index in the base DataFrame (used for case alignment).
        df : pandas.DataFrame
            Base DataFrame for case selection (used for indexing and case_index reference).
        dataset_name : str
            Name of dataset (for title).
        model_1 : widgets.Widget
            Dropdown for selecting the first model.
        model_2 : widgets.Widget
            Dropdown for selecting the second model.
        pu_scale_mode : widgets.Widget
            Radio button for selecting shared vs independent color scales for p and U.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with 3x2 subplots comparing GT, Model 1 and Model 2 for p and U.

        """
        df = df.reset_index(drop=True)
        idx = int(idx)

        name_1 = str(model_1.value)
        name_2 = str(model_2.value)

        df1 = datasets[name_1].reset_index(drop=True)
        df2 = datasets[name_2].reset_index(drop=True)

        # Reference row: always by positional idx on the base df
        row_ref = df.iloc[idx]

        # Align rows for both models: try case_index (if available), else positional idx
        if ("case_index" in row_ref.index) and ("case_index" in df1.columns) and ("case_index" in df2.columns):
            case_id = int(row_ref.loc["case_index"])

            row1 = _row_by_case_index(df1, case_id)
            if row1 is None:
                row1 = df1.iloc[min(idx, len(df1) - 1)]

            row2 = _row_by_case_index(df2, case_id)
            if row2 is None:
                row2 = df2.iloc[min(idx, len(df2) - 1)]

            _dataset_name = f"{dataset_name}"
            _case_label = f"Case {case_id}"
        else:
            row1 = df1.iloc[min(idx, len(df1) - 1)]
            row2 = df2.iloc[min(idx, len(df2) - 1)]
            _dataset_name = f"{dataset_name}"
            _case_label = f"Index {idx + 1}"

        pred1, gt1, _, _, _ = _load_npz(row1)
        pred2, _gt2, _, _, _ = _load_npz(row2)

        # Geometry
        Lx = float(row1.loc["geometry_Lx"])
        Ly = float(row1.loc["geometry_Ly"])

        # Coordinates
        ny, nx = gt1[CHANNEL_INDICES["p"]].shape
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(3, 2, figsize=(15, 10))

        # -----------------------------
        # Helpers: one panel like "rest"
        # -----------------------------
        def _panel(
            ax: Axes,
            field: np.ndarray,
            *,
            levels: np.ndarray,
            title: str,
            col: int,
            is_last_row: bool,
            mask_outside: tuple[float, float] | None = None,
        ) -> None:
            if mask_outside is not None:
                vmin, vmax = mask_outside
                field_plot = np.ma.masked_outside(field, vmin, vmax)

                cmap_obj = plt.get_cmap(cmap_pred_true).copy()
                cmap_obj.set_bad("white")
                im = ax.contourf(X, Y, field_plot, levels=levels, cmap=cmap_obj)
            else:
                im = ax.contourf(X, Y, field, levels=levels, cmap=cmap_pred_true)

            cb = fig.colorbar(im, ax=ax, fraction=0.045, ticks=levels)
            cb.formatter = analysis.ui.components.choose_colorbar_formatter(
                float(levels.min()),
                float(levels.max()),
                ticks=levels,
            )
            cb.update_ticks()
            ax.set_title(title)
            analysis.ui.components.apply_axis_labels(ax, col, Lx, Ly, is_last_row=is_last_row)

        shared_mode = pu_scale_mode.value == "Shared (GT)"
        # -----------------------------
        # p column
        # -----------------------------
        p_gt = gt1[CHANNEL_INDICES["p"]]
        p_m1 = pred1[CHANNEL_INDICES["p"]]
        p_m2 = pred2[CHANNEL_INDICES["p"]]

        if shared_mode:
            p_levels = analysis.ui.components.compute_levels(p_gt, n_levels)
            p_mask = (float(p_levels.min()), float(p_levels.max()))
            p_levels_gt = p_levels_m1 = p_levels_m2 = p_levels
        else:
            p_levels_gt = analysis.ui.components.compute_levels(p_gt, n_levels)
            p_levels_m1 = analysis.ui.components.compute_levels(p_m1, n_levels)
            p_levels_m2 = analysis.ui.components.compute_levels(p_m2, n_levels)
            p_mask = None

        _panel(axes[0, 0], p_gt, levels=p_levels_gt, title=f"pressure ground truth [{UNIT_MAP['p']}]", col=0, is_last_row=False)
        _panel(
            axes[1, 0],
            p_m1,
            levels=p_levels_m1,
            title=f"pressure prediction ({name_1}) [{UNIT_MAP['p']}]",
            col=0,
            is_last_row=False,
            mask_outside=p_mask,
        )
        _panel(
            axes[2, 0],
            p_m2,
            levels=p_levels_m2,
            title=f"pressure prediction ({name_2}) [{UNIT_MAP['p']}]",
            col=0,
            is_last_row=True,
            mask_outside=p_mask,
        )

        # -----------------------------
        # U column (+ streamlines)
        # -----------------------------
        U_gt = gt1[CHANNEL_INDICES["U"]]
        U_m1 = pred1[CHANNEL_INDICES["U"]]
        U_m2 = pred2[CHANNEL_INDICES["U"]]

        if shared_mode:
            U_levels = analysis.ui.components.compute_levels(U_gt, n_levels)
            U_mask = (float(U_levels.min()), float(U_levels.max()))
            U_levels_gt = U_levels_m1 = U_levels_m2 = U_levels
        else:
            U_levels_gt = analysis.ui.components.compute_levels(U_gt, n_levels)
            U_levels_m1 = analysis.ui.components.compute_levels(U_m1, n_levels)
            U_levels_m2 = analysis.ui.components.compute_levels(U_m2, n_levels)
            U_mask = None

        _panel(axes[0, 1], U_gt, levels=U_levels_gt, title=f"Velocity ground truth [{UNIT_MAP['U']}]", col=1, is_last_row=False)
        _panel(
            axes[1, 1],
            U_m1,
            levels=U_levels_m1,
            title=f"Velocity prediction ({name_1}) [{UNIT_MAP['U']}]",
            col=1,
            is_last_row=False,
            mask_outside=U_mask,
        )
        _panel(
            axes[2, 1],
            U_m2,
            levels=U_levels_m2,
            title=f"Velocity prediction ({name_2}) [{UNIT_MAP['U']}]",
            col=1,
            is_last_row=True,
            mask_outside=U_mask,
        )

        u_gt = gt1[CHANNEL_INDICES["u"]]
        v_gt = gt1[CHANNEL_INDICES["v"]]
        analysis.ui.components.overlay_streamlines(axes[0, 1], X, Y, u_gt, v_gt)

        u_1 = pred1[CHANNEL_INDICES["u"]]
        v_1 = pred1[CHANNEL_INDICES["v"]]
        analysis.ui.components.overlay_streamlines(axes[1, 1], X, Y, u_1, v_1)

        u_2 = pred2[CHANNEL_INDICES["u"]]
        v_2 = pred2[CHANNEL_INDICES["v"]]
        analysis.ui.components.overlay_streamlines(axes[2, 1], X, Y, u_2, v_2)

        # Row labels (links)
        axes[0, 0].set_ylabel("y [m]")
        axes[1, 0].set_ylabel("y [m]")
        axes[2, 0].set_ylabel("y [m]")

        fig.tight_layout()
        return fig

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=base_datasets,
        start_idx=0,
        enable_dataset_dropdown=False,
        extra_widgets=[dd_model_1, dd_model_2, pu_scale_selector],
        model_1=dd_model_1,
        model_2=dd_model_2,
        pu_scale_mode=pu_scale_selector,
    )
