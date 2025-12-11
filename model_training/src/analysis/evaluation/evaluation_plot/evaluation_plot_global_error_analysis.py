"""
Global error analysis plots for PINO/FNO evaluation.

This module provides high-level comparative plots across multiple datasets
(ID and OOD) based on the aggregated evaluation DataFrames.

Functions in this module are designed to be called through the notebook toggle
shortcut, which automatically injects a dictionary of datasets:

    dfs : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation_df

Each function returns either:
    - a Matplotlib Figure (for static plots), or
    - an ipywidgets.VBox (for interactive dropdown-based plots).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src import util

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# =============================================================================
# 1-1. GLOBAL ERROR METRICS
# =============================================================================


def plot_global_error_metrics(*, datasets: dict[str, pd.DataFrame]) -> Figure:  # noqa: PLR0915
    """
    Comprehensive global error comparison using three complementary views.

        1. Violinplot       - distribution shape + median
        2. KDE curves       - comparative density across datasets
        3. CDF curves       - dominance and cumulative behaviour

    All results are shown side-by-side for:
        • global L2
        • relative L2

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.
        Must contain:
            - l2
            - rel_l2

    Returns
    -------
    matplotlib.figure.Figure
        Multi-panel figure with violinplots, KDE and CDF curves.

    """
    names = list(datasets.keys())
    l2_vals = [datasets[n]["l2"].astype(float).to_numpy() for n in names]
    rel_vals = [datasets[n]["rel_l2"].astype(float).to_numpy() for n in names]

    palette = sns.color_palette("tab10", len(names))

    # ------------------------------------------------------------------
    # SMART GRID: 3 ROWS x 3 COLUMNS, RIGHT COLUMN FOR LEGEND
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(21, 10))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1, 1, 0.35],
        height_ratios=[1, 1, 1],
        wspace=0.25,
        hspace=0.35,
    )

    ax_vio_l2 = fig.add_subplot(gs[0, 0])
    ax_vio_rel = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])

    ax_kde_l2 = fig.add_subplot(gs[1, 0])
    ax_kde_rel = fig.add_subplot(gs[1, 1])

    ax_cdf_l2 = fig.add_subplot(gs[2, 0])
    ax_cdf_rel = fig.add_subplot(gs[2, 1])

    # ============================================================
    # 1. VIOLINPLOTS
    # ============================================================
    parts = ax_vio_l2.violinplot(l2_vals, showmeans=True, showextrema=False)
    bodies = cast("Iterable", parts["bodies"])
    for body, color in zip(bodies, palette, strict=False):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    ax_vio_l2.set_xticks([])
    ax_vio_l2.set_title("L2 - Violinplot")
    ax_vio_l2.set_ylabel("L2")
    ax_vio_l2.set_yscale("log")
    ax_vio_l2.grid(True, which="both", linestyle="--", alpha=0.3)

    parts = ax_vio_rel.violinplot(rel_vals, showmeans=True, showextrema=False)
    bodies = cast("Iterable", parts["bodies"])
    for body, color in zip(bodies, palette, strict=False):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    ax_vio_rel.set_xticks([])
    ax_vio_rel.set_title("Relative L2 - Violinplot")
    ax_vio_rel.set_ylabel("Relative L2")
    ax_vio_rel.set_yscale("log")
    ax_vio_rel.grid(True, which="both", linestyle="--", alpha=0.3)

    # ============================================================
    # 2. KDE CURVES
    # ============================================================
    for arr, name in zip(l2_vals, names, strict=False):
        kde = gaussian_kde(arr)
        xs = np.linspace(arr.min(), arr.max(), 400)
        ax_kde_l2.plot(xs, kde(xs), label=name)
    ax_kde_l2.set_xscale("log")
    ax_kde_l2.set_title("L2 - KDE Density Comparison")
    ax_kde_l2.set_ylabel("Density")
    ax_kde_l2.grid(True, which="both", linestyle="--", alpha=0.3)

    for arr, name in zip(rel_vals, names, strict=False):
        kde = gaussian_kde(arr)
        xs = np.linspace(arr.min(), arr.max(), 400)
        ax_kde_rel.plot(xs, kde(xs), label=name)
    ax_kde_rel.set_xscale("log")
    ax_kde_rel.set_title("Relative L2 - KDE Density Comparison")
    ax_kde_rel.set_ylabel("Density")
    ax_kde_rel.grid(True, which="both", linestyle="--", alpha=0.3)

    # ============================================================
    # 3. CDF CURVES
    # ============================================================
    for arr, name in zip(l2_vals, names, strict=False):
        s = np.sort(arr)
        y = np.linspace(0, 1, len(s))
        ax_cdf_l2.plot(s, y, label=name)
    ax_cdf_l2.set_xscale("log")
    ax_cdf_l2.set_title("L2 - CDF")
    ax_cdf_l2.set_xlabel("L2")
    ax_cdf_l2.set_ylabel("CDF")
    ax_cdf_l2.grid(True, which="both", linestyle="--", alpha=0.3)

    for arr, name in zip(rel_vals, names, strict=False):
        s = np.sort(arr)
        y = np.linspace(0, 1, len(s))
        ax_cdf_rel.plot(s, y, label=name)
    ax_cdf_rel.set_xscale("log")
    ax_cdf_rel.set_title("Relative L2 - CDF")
    ax_cdf_rel.set_xlabel("Relative L2")
    ax_cdf_rel.set_ylabel("CDF")
    ax_cdf_rel.grid(True, which="both", linestyle="--", alpha=0.3)

    # ============================================================
    # GLOBAL LEGEND (SEPARATE COLUMN)
    # ============================================================
    ax_legend.axis("off")
    handles = [Line2D([0], [0], color=c, lw=8) for c in palette]
    ax_legend.legend(handles, names, title="Dataset", loc="upper left")

    return fig


class CacheEntry(TypedDict):
    """
    Strongly typed cache entry for incremental loading.

    Attributes
    ----------
    loaded_until : int
        Number of cases loaded so far.
    global_l2_vals : list[float]
        Global L2 values for loaded cases.
    local_l2 : list[np.ndarray]
        Local L2 arrays for loaded cases.
    local_rel : list[np.ndarray]
        Local relative L2 arrays for loaded cases.

    """

    loaded_until: int
    global_l2_vals: list[float]
    local_l2: list[np.ndarray]
    local_rel: list[np.ndarray]


def plot_error_distribution(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: C901, PLR0915
    """
    Interactive global error distribution analysis across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.
        Must contain:
            - 'npz_path' : str (path to .npz with 'gt' and 'err' arrays)
            - 'l2' : float (global L2 value per case)

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with case count slider and error distribution plots.

    """
    names = list(datasets.keys())
    palette = sns.color_palette("tab10", len(names))

    # -------------------------------------------------------------------------
    # STRONGLY TYPED CACHE (MYPY + PYLANCE CLEAN)
    # -------------------------------------------------------------------------
    cache: dict[str, CacheEntry] = {
        name: CacheEntry(
            loaded_until=0,
            global_l2_vals=[],
            local_l2=[],
            local_rel=[],
        )
        for name in names
    }

    eps = 1e-8
    max_points = 20000
    clip_percentile = 99.5

    # -------------------------------------------------------------------------
    # INTERNAL PLOT FUNCTION
    # -------------------------------------------------------------------------
    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:  # noqa: C901, PLR0912, PLR0915
        """
        Plot global and local error distributions for the first `max_cases` samples.

        Parameters
        ----------
        max_cases : int
            Number of cases to include from each dataset.
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset_name → evaluation DataFrame.
            Must contain:
                - 'npz_path' : str (path to .npz with 'gt' and 'err' arrays)
                - 'l2' : float (global L2 value per case)

        Returns
        -------
        matplotlib.figure.Figure
            Multi-panel figure with global and local error distribution plots.

        """
        for name, df in datasets.items():
            entry = cache[name]

            loaded = entry["loaded_until"]
            _global_l2_vals = entry["global_l2_vals"]
            local_l2 = entry["local_l2"]
            local_rel = entry["local_rel"]

            # Load ONLY new cases
            if max_cases > loaded:
                df_new = df.iloc[loaded:max_cases]

                for path in df_new["npz_path"]:
                    data = np.load(path)

                    err = data["err"][0]
                    gt = data["gt"][0]

                    e = np.linalg.norm(err, axis=0)
                    g = np.linalg.norm(gt, axis=0)

                    mask = g > eps
                    rel = np.zeros_like(e)
                    rel[mask] = e[mask] / g[mask]

                    local_l2.append(e.ravel())
                    local_rel.append(rel.ravel())

                # update global values
                entry["global_l2_vals"] = list(df["l2"].iloc[:max_cases])
                entry["loaded_until"] = max_cases

        # ---------------------------------------------------------------------
        # Compute statistics
        # ---------------------------------------------------------------------
        global_l2_stats = {}
        local_rel_qtiles = {}
        local_l2_arrays: dict[str, np.ndarray] = {}

        for name in names:
            entry = cache[name]

            gvals = np.array(entry["global_l2_vals"], dtype=float)

            global_l2_stats[name] = {
                "median": float(np.median(gvals)),
                "mean": float(np.mean(gvals)),
                "q90": float(np.quantile(gvals, 0.90)),
                "q95": float(np.quantile(gvals, 0.95)),
            }

            arr_l2 = np.concatenate(entry["local_l2"]) if entry["local_l2"] else np.array([])
            arr_rel = np.concatenate(entry["local_rel"]) if entry["local_rel"] else np.array([])

            arr_rel = arr_rel[~np.isnan(arr_rel)]

            if arr_l2.size > 0:
                cutoff = float(np.percentile(arr_l2, clip_percentile))
                arr_l2 = np.clip(arr_l2, 0, cutoff)

            if arr_rel.size > 0:
                cutoff = float(np.percentile(arr_rel, clip_percentile))
                arr_rel = np.clip(arr_rel, 0, cutoff)

            rng = np.random.default_rng()

            if arr_l2.size > max_points:
                arr_l2 = arr_l2[rng.choice(arr_l2.size, max_points, replace=False)]
            if arr_rel.size > max_points:
                arr_rel = arr_rel[rng.choice(arr_rel.size, max_points, replace=False)]

            local_l2_arrays[name] = arr_l2

            if arr_rel.size > 0:
                local_rel_qtiles[name] = {
                    "median": float(np.median(arr_rel)),
                    "q75": float(np.quantile(arr_rel, 0.75)),
                    "q90": float(np.quantile(arr_rel, 0.90)),
                    "q95": float(np.quantile(arr_rel, 0.95)),
                }
            else:
                local_rel_qtiles[name] = {"median": 0.0, "q75": 0.0, "q90": 0.0, "q95": 0.0}

        # ---------------------------------------------------------------------
        # PLOTS
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        ax_global = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_local_l2 = fig.add_subplot(gs[1, 0])
        ax_local_rel = fig.add_subplot(gs[1, 1])
        ax_legend.axis("off")

        stats = ["median", "mean", "q90", "q95"]
        xpos = np.arange(len(stats))

        for idx, name in enumerate(names):
            vals = [global_l2_stats[name][s] for s in stats]
            ax_global.plot(xpos, vals, marker="o", lw=2, color=palette[idx])

        ax_global.set_xticks(xpos)
        ax_global.set_xticklabels(stats)
        ax_global.set_yscale("log")
        ax_global.set_title(f"Global L2 Summary (first {max_cases} cases)")
        ax_global.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

        # KDE
        for name, color in zip(names, palette, strict=False):
            arr = local_l2_arrays[name]
            if arr.size > 0:
                sns.kdeplot(arr, ax=ax_local_l2, lw=2, color=color, log_scale=True)

        ax_local_l2.set_title(f"Local L2 Distribution (first {max_cases} cases)")
        ax_local_l2.grid(True, linestyle="--", alpha=0.3)

        # rel quantiles
        qstats = ["median", "q75", "q90", "q95"]
        xpos = np.arange(len(qstats))

        for idx, name in enumerate(names):
            vals = [local_rel_qtiles[name][s] for s in qstats]
            ax_local_rel.plot(xpos, vals, marker="o", lw=2, color=palette[idx])

        ax_local_rel.set_yscale("log")
        ax_local_rel.set_title(f"Local Relative L2 Quantiles (first {max_cases} cases)")
        ax_local_rel.grid(True, axis="y", linestyle="--", alpha=0.3)

        handles = [Line2D([0], [0], color=c, lw=6) for c in palette]
        ax_legend.legend(handles, names, title="Dataset", loc="upper center")

        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 1-3. GLOBAL GT VS PRED (incremental cached)
# =============================================================================


class GTCacheEntry(TypedDict):
    """
    Strongly typed cache entry for incremental GT vs Prediction mean comparison.

    Attributes
    ----------
    loaded_until : int
        Number of cases loaded so far.
    gt_means : dict[str, list[float]]
        GT means per channel for loaded cases.
    pred_means : dict[str, list[float]]
        Prediction means per channel for loaded cases.

    """

    loaded_until: int
    gt_means: dict[str, list[float]]
    pred_means: dict[str, list[float]]


def plot_global_gt_vs_pred(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: C901, PLR0915
    """
    Interactive global GT vs Prediction mean comparison across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.
        Must contain:
            - 'npz_path' : str (path to .npz with 'gt' and 'pred' arrays)

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with case count slider and GT vs Prediction plots.

    """
    names = list(datasets.keys())
    channels = ["p", "u", "v", "U"]
    channel_indices = {"p": 0, "u": 1, "v": 2, "U": 3}

    # =========================================================================
    # Strongly typed cache (mypy + pylance clean)
    # =========================================================================
    cache: dict[str, GTCacheEntry] = {
        name: GTCacheEntry(
            loaded_until=0,
            gt_means={ch: [] for ch in channels},
            pred_means={ch: [] for ch in channels},
        )
        for name in names
    }

    # =========================================================================
    # INTERNAL PLOT FUNCTION
    # =========================================================================
    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:  # noqa: C901
        # ---------------------------------------------------------------------
        # Incremental NPZ loading
        # ---------------------------------------------------------------------
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]
            gt_means = entry["gt_means"]
            pred_means = entry["pred_means"]

            if max_cases > loaded:
                df_new = df.iloc[loaded:max_cases]

                for path in df_new["npz_path"]:
                    data = np.load(path)

                    gt = data["gt"][0]
                    pred = data["pred"][0]
                    C = gt.shape[0]

                    for ch in channels:
                        idx = channel_indices[ch]
                        if idx < C:
                            gt_means[ch].append(float(gt[idx].mean()))
                            pred_means[ch].append(float(pred[idx].mean()))

                entry["loaded_until"] = max_cases

        # ---------------------------------------------------------------------
        # Prepare figure
        # ---------------------------------------------------------------------
        num_datasets = len(names)
        num_channels = len(channels)

        fig = plt.figure(figsize=(6 * num_datasets, 9))
        gs = fig.add_gridspec(
            num_channels,
            num_datasets,
            wspace=0.25,
            hspace=0.35,
        )

        axes: list[list[Axes]] = []

        # ---------------------------------------------------------------------
        # Plot per dataset and per channel
        # ---------------------------------------------------------------------
        for row_idx, ch in enumerate(channels):
            row_axes: list[Axes] = []

            for col_idx, name in enumerate(names):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                row_axes.append(ax)

                entry = cache[name]

                gt_arr = np.array(entry["gt_means"][ch], dtype=float)
                pred_arr = np.array(entry["pred_means"][ch], dtype=float)

                if gt_arr.size == 0:
                    ax.text(0.5, 0.5, "Channel missing", ha="center", va="center")
                    ax.axis("off")
                    continue

                rmse = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
                ss_res = float(np.sum((pred_arr - gt_arr) ** 2))
                ss_tot = float(np.sum((gt_arr - gt_arr.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                vmin = float(min(gt_arr.min(), pred_arr.min()))
                vmax = float(max(gt_arr.max(), pred_arr.max()))

                ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, alpha=0.7)
                ax.scatter(gt_arr, pred_arr, s=18, alpha=0.45)

                ax.set_title(f"{ch}: RMSE={rmse:.4f}, R2={r2:.3f}", fontsize=11)
                ax.grid(alpha=0.3)

            axes.append(row_axes)

        # ---------------------------------------------------------------------
        # Add dataset titles
        # ---------------------------------------------------------------------
        for col_idx, name in enumerate(names):
            axes[0][col_idx].set_title(
                f"Dataset: {name}\n" + axes[0][col_idx].get_title(),
                fontsize=12,
                pad=20,
            )

        # Row & column labels
        for row_idx in range(num_channels):
            axes[row_idx][0].set_ylabel("Prediction mean")
        for ax in axes[-1]:
            ax.set_xlabel("GT mean")

        fig.subplots_adjust(
            top=0.92,
            bottom=0.07,
            left=0.07,
            right=0.98,
            hspace=0.35,
            wspace=0.25,
        )

        return fig

    # =========================================================================
    # Return viewer
    # =========================================================================
    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 1-4. MEAN ERROR MAPS
# =============================================================================


# =============================================================================
# 1-4. MEAN ERROR MAPS — FULLY CACHED, MAE + REL SEPARATE
# =============================================================================


def plot_mean_error_maps(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: C901, PLR0915
    """
    Interactive mean error maps across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.
        Must contain:
            - 'npz_path' : str (path to .npz with 'gt' and 'pred' arrays)

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with case count slider and mean error maps.

    """
    channels = ["p", "u", "v", "U"]
    channel_idx = {"p": 0, "u": 1, "v": 2, "U": 3}
    mask_threshold = 1e-4

    # -------------------------------------------------------
    # UI: MAE / Rel [%]
    # -------------------------------------------------------
    error_selector = util.util_plot_components.ui_error_mode_selector()

    # -------------------------------------------------------
    # Cache structure (clean, mypy-safe)
    # -------------------------------------------------------
    # Each dataset gets:
    #   - geom
    #   - loaded_until
    #   - count
    #   - sum_mae[ch]
    #   - sum_rel[ch]
    #
    cache: dict[str, dict[str, Any]] = {}

    for name in datasets:
        cache[name] = {
            "geom": None,  # tuple[float, float]
            "loaded_until": 0,  # int
            "count": 0,  # int
            # running sums
            "sum_mae": dict.fromkeys(channels),  # np.ndarray | None
            "sum_rel": dict.fromkeys(channels),  # np.ndarray | None
        }

    # -------------------------------------------------------
    # INTERNAL PLOT FUNCTION
    # -------------------------------------------------------
    def _plot(  # noqa: C901, PLR0912, PLR0915
        *,
        datasets: dict[str, pd.DataFrame],
        max_cases: int,
        error_mode: widgets.RadioButtons,
    ) -> Figure:
        """
        Plot mean error maps across datasets for the first `max_cases` samples.

        Parameters
        ----------
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset_name → evaluation DataFrame.
            Must contain:
                - 'npz_path' : str (path to .npz with 'gt' and 'pred' arrays)
        max_cases : int
            Number of cases to include from each dataset.
        error_mode : widgets.RadioButtons
            Radio button widget indicating error mode ("MAE" or "Relative [%]").

        Returns
        -------
        matplotlib.figure.Figure
            Multi-panel figure with mean error maps.

        """
        mode = error_mode.value  # "MAE" or "Relative [%]"
        names = list(datasets.keys())
        num_datasets = len(names)
        num_channels = len(channels)

        # ===================================================
        # LOAD NEW CASES INTO CACHE
        # ===================================================
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            df_i = df.reset_index(drop=True)

            # Set geometry once
            if entry["geom"] is None:
                entry["geom"] = (
                    float(df_i["geom_Lx"].iloc[0]),
                    float(df_i["geom_Ly"].iloc[0]),
                )

            # Nothing new to load
            if max_cases <= loaded:
                continue

            # New rows that need loading
            df_new = df_i.iloc[loaded:max_cases]

            for path in df_new["npz_path"]:
                data = np.load(path)
                pred = data["pred"][0]
                gt = data["gt"][0]

                # Compute MAE + REL both at load time.
                # This allows switching modes instantly without reloading NPZ.
                for ch in channels:
                    k = channel_idx[ch]

                    # ============ MAE ============
                    mae = np.abs(pred[k] - gt[k])

                    if entry["sum_mae"][ch] is None:
                        entry["sum_mae"][ch] = mae.astype(float)
                    else:
                        entry["sum_mae"][ch] += mae

                    # ============ REL ============
                    abs_err = np.abs(pred[k] - gt[k])
                    true_abs = np.abs(gt[k])
                    rel = abs_err / (true_abs + 1e-12) * 100.0
                    rel[true_abs < mask_threshold] = np.nan
                    vmax_rel = float(np.nanquantile(rel, 0.99))
                    vmax_rel = max(vmax_rel, 1e-6)
                    rel = np.clip(rel, 0.0, vmax_rel)

                    if entry["sum_rel"][ch] is None:
                        entry["sum_rel"][ch] = rel.astype(float)
                    else:
                        entry["sum_rel"][ch] += rel

                entry["count"] += 1

            entry["loaded_until"] = max_cases

        # ===================================================
        # BUILD FIGURE
        # ===================================================
        fig = plt.figure(figsize=(6 * num_datasets, 9))
        gs = fig.add_gridspec(num_channels, num_datasets, wspace=0.25, hspace=0.35)

        for r, ch in enumerate(channels):
            for c, name in enumerate(names):
                ax = fig.add_subplot(gs[r, c])
                entry = cache[name]

                geom = entry["geom"]
                if geom is None:
                    msg = f"Geometry for dataset '{name}' was not initialised."
                    raise RuntimeError(msg)
                Lx, Ly = geom

                sum_arr = entry["sum_mae"][ch] if mode == "MAE" else entry["sum_rel"][ch]

                mean_map = np.zeros((10, 10)) if sum_arr is None or entry["count"] == 0 else sum_arr / entry["count"]

                ny, nx = mean_map.shape
                x = np.linspace(0, Lx, nx)
                y = np.linspace(0, Ly, ny)
                X, Y = np.meshgrid(x, y)

                im = ax.contourf(X, Y, mean_map, levels=10, cmap="coolwarm")

                metric = "MAE" if mode == "MAE" else "rel err [%]"

                if r == 0:
                    ax.set_title(f"Dataset: {name}\n{ch} {metric}", fontsize=12, pad=20)
                else:
                    ax.set_title(f"{ch} {metric}", fontsize=11)

                if c == 0:
                    ax.set_ylabel("y [m]")
                    ax.set_yticks([0, Ly / 2, Ly])
                else:
                    ax.set_yticks([])

                if r == num_channels - 1:
                    ax.set_xlabel("x [m]")
                    ax.set_xticks([0, Lx / 2, Lx])
                else:
                    ax.set_xticks([])

                fig.colorbar(im, ax=ax, fraction=0.045)

        fig.subplots_adjust(
            top=0.92,
            bottom=0.07,
            left=0.07,
            right=0.98,
            hspace=0.35,
            wspace=0.25,
        )
        return fig

    # -------------------------------------------------------
    # Connect to CASECOUNT viewer
    # -------------------------------------------------------

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[error_selector],
        error_mode=error_selector,
    )


# =============================================================================
# 1-5. STD ERROR MAPS (INTERACTIVE DROPDOWN)
# =============================================================================


def plot_std_error_maps(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive standard deviation error map viewer across datasets.

    Each evaluation dataframe must provide the column:
        - 'err_std_map' : ndarray (H, W)

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping of dataset_name → evaluation dataframe.

    Returns
    -------
    widgets.VBox
        Interactive widget with dropdown + std error visualisation.

    """

    def _plot_single_dataset(*, dfs: dict[str, pd.DataFrame], dataset_name: str) -> Figure:
        df = dfs[dataset_name]

        if "err_std_map" not in df.columns:
            msg = f"Dataset '{dataset_name}' is missing 'err_std_map'."
            raise KeyError(msg)

        std_map = df["err_std_map"].iloc[0]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(std_map, cmap="viridis")

        ax.set_title(f"Std Error Map — {dataset_name}")
        ax.set_xticks([])
        ax.set_yticks([])

        fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout()
        return fig

    return util.util_plot.make_interactive_global_plot_dropdown(
        plot_func=_plot_single_dataset,
        datasets=datasets,
    )
