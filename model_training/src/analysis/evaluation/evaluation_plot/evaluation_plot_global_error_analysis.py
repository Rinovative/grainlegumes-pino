"""
===============================================================================
evaluation_plot_global_error_analysis.py
===============================================================================
Plot aggregate model errors across datasets and model variants.

Responsibilities:
  - Plot comparative global error metrics
  - Plot error distributions and mean error maps
  - Support multi-dataset aggregate comparisons

Design principles:
  - Errors are aggregated consistently across cases
  - Group-level plots support model comparison
  - Styling is shared across evaluation plots

Boundaries:
  - Per-case spatial decomposition belongs to evaluation_plot_error_decomposition
  - Outlier inspection belongs to evaluation_plot_outlier_analysis
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src import analysis, domain

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# =============================================================================
# GLOBAL OUTPUT CHANNEL CONFIGURATION (domain-derived)
# =============================================================================

CHANNELS = domain.fields.ANALYSIS_FIELDS
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}

# =============================================================================
# GLOBAL ERROR METRICS
# =============================================================================


def plot_global_error_metrics(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """Compare dimensionless channel-balanced relative L2 and relative H1."""
    names = list(datasets)
    if not names:
        msg = "At least one artifact DataFrame is required."
        raise ValueError(msg)

    metric_specs = (("Relative L2", "rel_l2"), ("Relative H1", "rel_h1"))
    minimum_kde_samples = 2
    metric_values: dict[str, list[np.ndarray]] = {}
    for _label, column in metric_specs:
        arrays: list[np.ndarray] = []
        for name in names:
            if column not in datasets[name]:
                msg = f"Artifact DataFrame {name!r} is missing required column {column!r}."
                raise KeyError(msg)
            values = datasets[name][column].astype(float).to_numpy()
            if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
                msg = f"Artifact column {column!r} for {name!r} must contain finite non-negative values."
                raise ValueError(msg)
            arrays.append(values)
        metric_values[column] = arrays

    palette = sns.color_palette("tab10", len(names))
    fig = plt.figure(figsize=(21, 10))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[1, 1, 0.35],
        height_ratios=[1, 1, 1],
        wspace=0.25,
        hspace=0.35,
    )

    for column_index, (label, column) in enumerate(metric_specs):
        arrays = metric_values[column]
        box_axis = fig.add_subplot(grid[0, column_index])
        density_axis = fig.add_subplot(grid[1, column_index])
        cdf_axis = fig.add_subplot(grid[2, column_index])

        boxplot = box_axis.boxplot(
            arrays,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 2},
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, color in zip(boxplot["boxes"], palette, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        box_axis.set_xticks([])
        box_axis.set_title(f"{label} - Boxplot")
        box_axis.set_ylabel(f"{label} (dimensionless)")
        box_axis.grid(True, which="both", linestyle="--", alpha=0.3)

        for values, name, color in zip(arrays, names, palette, strict=True):
            if values.size < minimum_kde_samples or np.allclose(values, values[0]):
                density_axis.axvline(values[0], color=color, label=name)
            else:
                density = gaussian_kde(values)
                coordinates = np.linspace(values.min(), values.max(), 400)
                density_axis.plot(coordinates, density(coordinates), color=color, label=name)
        density_axis.set_title(f"{label} - KDE Density")
        density_axis.set_xlabel(f"{label} (dimensionless)")
        density_axis.set_ylabel("Density")
        density_axis.grid(True, which="both", linestyle="--", alpha=0.3)

        for values, name, color in zip(arrays, names, palette, strict=True):
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
            cdf_axis.plot(ordered, cumulative, color=color, label=name)
        cdf_axis.set_title(f"{label} - CDF")
        cdf_axis.set_xlabel(f"{label} (dimensionless)")
        cdf_axis.set_ylabel("CDF")
        cdf_axis.grid(True, which="both", linestyle="--", alpha=0.3)

        if all(np.all(values > 0.0) for values in arrays):
            box_axis.set_yscale("log")
            density_axis.set_xscale("log")
            cdf_axis.set_xscale("log")

    legend_axis = fig.add_subplot(grid[:, 2])
    legend_axis.axis("off")
    handles = [Line2D([0], [0], color=color, lw=8) for color in palette]
    legend_axis.legend(handles, names, loc="upper left")
    return fig


class CacheEntry(TypedDict):
    """Incremental dimensionless error-distribution cache."""

    loaded_until: int
    global_relative: list[float]
    local_relative: list[np.ndarray]


def plot_error_distribution(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """Interactively compare global and local dimensionless relative errors."""
    names = list(datasets)
    cache: dict[str, CacheEntry] = {
        name: CacheEntry(
            loaded_until=0,
            global_relative=[],
            local_relative=[],
        )
        for name in names
    }
    palette = sns.color_palette("tab10", len(names))
    denominator_floor = 1e-12
    max_points = 20000
    clip_percentile = 99.5

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        for name, frame in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]
            if max_cases > loaded:
                selected = frame.iloc[loaded:max_cases]
                entry["global_relative"].extend(selected["rel_l2"].astype(float).tolist())
                for artifact_path in selected["npz_path"]:
                    with np.load(artifact_path, allow_pickle=False) as artifact:
                        output_fields = artifact["output_fields"].tolist()
                        if not isinstance(output_fields, list) or not output_fields:
                            msg = f"Artifact output_fields must be a non-empty string vector: {artifact_path}"
                            raise ValueError(msg)
                        learned_channels = len(output_fields)
                        error = np.asarray(artifact["err"][:learned_channels], dtype=float)
                        target = np.asarray(artifact["gt"][:learned_channels], dtype=float)
                    field_rms = np.sqrt(np.mean(target**2, axis=(1, 2), keepdims=True))
                    local_relative = np.abs(error) / np.maximum(field_rms, denominator_floor)
                    entry["local_relative"].append(local_relative.ravel())
                entry["loaded_until"] = min(max_cases, len(frame))

        global_stats: dict[str, dict[str, float]] = {}
        local_arrays: dict[str, np.ndarray] = {}
        local_quantiles: dict[str, dict[str, float]] = {}
        for name in names:
            entry = cache[name]
            case_count = min(max_cases, len(entry["global_relative"]))
            global_values = np.asarray(entry["global_relative"][:case_count], dtype=float)
            if global_values.size == 0:
                msg = f"Artifact DataFrame {name!r} contains no selected cases."
                raise ValueError(msg)
            global_stats[name] = {
                "median": float(np.median(global_values)),
                "mean": float(np.mean(global_values)),
                "q90": float(np.quantile(global_values, 0.90)),
                "q95": float(np.quantile(global_values, 0.95)),
            }

            local_values = np.concatenate(entry["local_relative"][:case_count])
            local_values = local_values[np.isfinite(local_values)]
            if local_values.size:
                cutoff = float(np.percentile(local_values, clip_percentile))
                local_values = np.clip(local_values, 0.0, cutoff)
            if local_values.size > max_points:
                random = np.random.default_rng(0)
                local_values = local_values[random.choice(local_values.size, max_points, replace=False)]
            local_arrays[name] = local_values
            local_quantiles[name] = {
                quantile: float(np.quantile(local_values, probability)) if local_values.size else 0.0
                for quantile, probability in (("median", 0.50), ("q75", 0.75), ("q90", 0.90), ("q95", 0.95))
            }

        figure = plt.figure(figsize=(20, 8))
        grid = figure.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        global_axis = figure.add_subplot(grid[0, 0])
        legend_axis = figure.add_subplot(grid[0, 1])
        local_density_axis = figure.add_subplot(grid[1, 0])
        local_quantile_axis = figure.add_subplot(grid[1, 1])
        legend_axis.axis("off")

        statistic_names = ("median", "mean", "q90", "q95")
        positions = np.arange(len(statistic_names))
        for index, name in enumerate(names):
            global_stat_values = [global_stats[name][statistic] for statistic in statistic_names]
            global_axis.plot(positions, global_stat_values, marker="o", lw=2, color=palette[index])
        global_axis.set_xticks(positions)
        global_axis.set_xticklabels(statistic_names)
        global_axis.set_yscale("log")
        global_axis.set_title("Global Relative L2 Summary")
        global_axis.set_ylabel("Relative L2 (dimensionless)")
        global_axis.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

        for name, color in zip(names, palette, strict=True):
            local_values = local_arrays[name]
            if local_values.size > 1 and not np.allclose(local_values, local_values[0]):
                sns.kdeplot(local_values, ax=local_density_axis, lw=2, color=color, log_scale=True)
            elif local_values.size:
                local_density_axis.axvline(local_values[0], color=color)
        local_density_axis.set_title("Local Field-Normalized Absolute Error")
        local_density_axis.set_xlabel("Absolute error / field RMS (dimensionless)")
        local_density_axis.grid(True, linestyle="--", alpha=0.3)

        quantile_names = ("median", "q75", "q90", "q95")
        positions = np.arange(len(quantile_names))
        for index, name in enumerate(names):
            quantile_values = [local_quantiles[name][quantile] for quantile in quantile_names]
            local_quantile_axis.plot(positions, quantile_values, marker="o", lw=2, color=palette[index])
        local_quantile_axis.set_yscale("log")
        local_quantile_axis.set_title("Local Field-Normalized Error Quantiles")
        local_quantile_axis.set_ylabel("Absolute error / field RMS (dimensionless)")
        local_quantile_axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        local_quantile_axis.set_xticks(positions)
        local_quantile_axis.set_xticklabels(quantile_names)

        handles = [Line2D([0], [0], color=color, lw=6) for color in palette]
        legend_axis.legend(handles, names, loc="upper center")
        return figure

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# GLOBAL GT VS PRED
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


def plot_global_gt_vs_pred(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
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
    # Cache each dataset independently so slider updates load only new cases.
    cache: dict[str, GTCacheEntry] = {
        name: GTCacheEntry(
            loaded_until=0,
            gt_means={ch: [] for ch in CHANNELS},
            pred_means={ch: [] for ch in CHANNELS},
        )
        for name in names
    }

    # =========================================================================
    # INTERNAL PLOT FUNCTION
    # =========================================================================
    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
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

                    gt = data["gt"]
                    pred = data["pred"]
                    C = gt.shape[0]

                    for ch in CHANNELS:
                        idx = CHANNEL_INDICES[ch]
                        if idx < C:
                            gt_means[ch].append(float(gt[idx].mean()))
                            pred_means[ch].append(float(pred[idx].mean()))

                entry["loaded_until"] = max_cases

        # ---------------------------------------------------------------------
        # Prepare figure
        # ---------------------------------------------------------------------
        num_datasets = len(names)
        num_channels = len(CHANNELS)

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
        for row_idx, ch in enumerate(CHANNELS):
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

                ax.set_title(f"{ch}: RMSE={rmse:.3g}, R2={r2:.3g}", fontsize=11)
                ax.grid(alpha=0.3)

            axes.append(row_axes)

        # ---------------------------------------------------------------------
        # Add dataset titles
        # ---------------------------------------------------------------------
        for col_idx, name in enumerate(names):
            axes[0][col_idx].set_title(
                f"{name}\n" + axes[0][col_idx].get_title(),
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
    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# MEAN ERROR MAPS
# =============================================================================


def plot_mean_error_maps(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
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
    mask_threshold = 1e-4

    # -------------------------------------------------------
    # UI: MAE / Rel [%]
    # -------------------------------------------------------
    error_selector = analysis.ui.components.ui_radio_error_mode()

    # -------------------------------------------------------
    # Cached aggregation state
    # -------------------------------------------------------
    # Each dataset records its geometry, loaded prefix, sample count, and
    # per-channel absolute and relative error totals.
    cache: dict[str, dict[str, Any]] = {}

    for name in datasets:
        cache[name] = {
            "geom": None,  # tuple[float, float]
            "loaded_until": 0,  # int
            "count": 0,  # int
            # running sums
            "sum_mae": dict.fromkeys(CHANNELS),  # np.ndarray | None
            "sum_rel": dict.fromkeys(CHANNELS),  # np.ndarray | None
        }

    # -------------------------------------------------------
    # INTERNAL PLOT FUNCTION
    # -------------------------------------------------------
    def _plot(
        *,
        datasets: dict[str, pd.DataFrame],
        max_cases: int,
        error_mode: widgets.ValueWidget,
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
        error_mode : widgets.ValueWidget
            Error mode selector widget.

        Returns
        -------
        matplotlib.figure.Figure
            Multi-panel figure with mean error maps.

        """
        mode = error_mode.value  # "MAE" or "Relative [%]"
        names = list(datasets.keys())
        num_datasets = len(names)
        num_channels = len(CHANNELS)

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
                    float(df_i["geometry_Lx"].iloc[0]),
                    float(df_i["geometry_Ly"].iloc[0]),
                )

            # Nothing new to load
            if max_cases <= loaded:
                continue

            # New rows that need loading
            df_new = df_i.iloc[loaded:max_cases]

            for path in df_new["npz_path"]:
                data = np.load(path)
                pred = data["pred"]
                gt = data["gt"]

                # Compute MAE + REL both at load time.
                # This allows switching modes instantly without reloading NPZ.
                for ch in CHANNELS:
                    k = CHANNEL_INDICES[ch]

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

        for r, ch in enumerate(CHANNELS):
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

                # robust clip (Ausreisser werden weiss)
                clip_q = 99.5
                vmax = float(np.nanpercentile(mean_map, clip_q)) if np.isfinite(mean_map).any() else 0.0
                vmax = max(vmax, 1e-12)

                mean_plot = np.ma.masked_greater(mean_map, vmax)

                cmap = plt.get_cmap("magma").copy()
                cmap.set_bad("white")

                levels = np.linspace(0.0, vmax, 11)
                im = ax.contourf(X, Y, mean_plot, levels=levels, cmap=cmap)

                metric = "MAE" if mode == "MAE" else "rel err [%]"

                if r == 0:
                    ax.set_title(f"{name}\n{ch} {metric}", fontsize=12, pad=20)
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

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[error_selector],
        error_mode=error_selector,
    )


# =============================================================================
# STD ERROR MAPS
# =============================================================================


def plot_std_error_maps(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive standard deviation error maps across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.
        Must contain:
            - 'npz_path' : str (path to .npz with 'gt' and 'pred' arrays)

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with case count slider and standard deviation error maps.

    """
    mask_threshold = 1e-4

    # -------------------------------------------------------
    # Streaming Welford statistics
    # -------------------------------------------------------
    cache: dict[str, dict[str, Any]] = {}

    for name in datasets:
        cache[name] = {
            "geom": None,
            "loaded_until": 0,
            "count": 0,
            "mean": dict.fromkeys(CHANNELS),  # running mean
            "M2": dict.fromkeys(CHANNELS),  # running sum of squares
        }

    # -------------------------------------------------------
    # INTERNAL PLOT FUNCTION
    # -------------------------------------------------------
    def _plot(
        *,
        datasets: dict[str, pd.DataFrame],
        max_cases: int,
    ) -> Figure:
        names = list(datasets.keys())
        num_datasets = len(names)
        num_channels = len(CHANNELS)

        # ===================================================
        # LOAD NEW CASES
        # ===================================================
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]
            df_i = df.reset_index(drop=True)

            if entry["geom"] is None:
                entry["geom"] = (
                    float(df_i["geometry_Lx"].iloc[0]),
                    float(df_i["geometry_Ly"].iloc[0]),
                )

            if max_cases <= loaded:
                continue

            df_new = df_i.iloc[loaded:max_cases]

            for path in df_new["npz_path"]:
                data = np.load(path)
                pred = data["pred"]
                gt = data["gt"]

                entry["count"] += 1
                n = entry["count"]

                for ch in CHANNELS:
                    k = CHANNEL_INDICES[ch]

                    abs_err = np.abs(pred[k] - gt[k])
                    true_abs = np.abs(gt[k])
                    abs_err[true_abs < mask_threshold] = np.nan

                    if entry["mean"][ch] is None:
                        entry["mean"][ch] = abs_err.astype(float)
                        entry["M2"][ch] = np.zeros_like(abs_err, dtype=float)
                    else:
                        delta = abs_err - entry["mean"][ch]
                        entry["mean"][ch] += delta / n
                        delta2 = abs_err - entry["mean"][ch]
                        entry["M2"][ch] += delta * delta2

            entry["loaded_until"] = max_cases

        # ===================================================
        # BUILD FIGURE (identical layout to 1-4)
        # ===================================================
        fig = plt.figure(figsize=(6 * num_datasets, 9))
        gs = fig.add_gridspec(num_channels, num_datasets, wspace=0.25, hspace=0.35)

        for r, ch in enumerate(CHANNELS):
            for c, name in enumerate(names):
                ax = fig.add_subplot(gs[r, c])
                entry = cache[name]

                geom = entry["geom"]
                Lx, Ly = geom
                std_map = np.sqrt(entry["M2"][ch] / (entry["count"] - 1))

                ny, nx = std_map.shape
                x = np.linspace(0, Lx, nx)
                y = np.linspace(0, Ly, ny)
                X, Y = np.meshgrid(x, y)

                clip_q = 99.5
                vmax = float(np.nanpercentile(std_map, clip_q)) if np.isfinite(std_map).any() else 0.0
                vmax = max(vmax, 1e-12)

                std_plot = np.ma.masked_greater(std_map, vmax)

                cmap = plt.get_cmap("magma").copy()
                cmap.set_bad("white")

                levels = np.linspace(0.0, vmax, 11)
                im = ax.contourf(X, Y, std_plot, levels=levels, cmap=cmap)

                if r == 0:
                    ax.set_title(f"{name}\n{ch} STD error", fontsize=12, pad=20)
                else:
                    ax.set_title(f"{ch} STD error", fontsize=11)

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
    # CASECOUNT VIEWER
    # -------------------------------------------------------
    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )
