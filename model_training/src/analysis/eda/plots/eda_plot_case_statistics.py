"""
===============================================================================
eda_plot_case_statistics.py
===============================================================================
Case-level statistical plots and metadata visualizations for EDA.

Provides:
  - metadata statistics distributions (count, range, value histograms)
  - generator parameter distributions across cases
  - field value distributions with case-level aggregates
  - interactive case viewers for statistical exploration

Responsibilities:
  - create publication-quality statistical plots
  - display metadata and parameter ranges across dataset
  - integrate with interactive analysis_ui_viewers for case navigation

Design principles:
  - statistics are aggregated per case (not per sample/grid point)
  - distributions are plotted with normalized scaling
  - metadata and generator parameters are treated as categorical/numeric distributions

This module does NOT:
  - compute derivatives or physics-specific metrics (use analysis modules)
  - handle field-level statistics across space (use eda_plot_spectral_analysis)
  - manage normalizers or unit conversion
===============================================================================
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from src import analysis

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.ui.analysis_ui_components import CheckboxGroup


# ============================================================================
# TYPES
# ============================================================================


class _StatCache(TypedDict):
    loaded: int
    cols: dict[str, list[float]]


class _ParamCache(TypedDict):
    loaded: int
    cols: dict[str, list[float]]


class _FieldCache(TypedDict):
    loaded: int
    data: dict[str, dict[str, list[float]]]


# ============================================================================
# HELPERS
# ============================================================================


def _as_float(x: Any) -> float | None:
    """
    Convert x to float if possible.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    float or None
        Converted float value, or None if conversion is not possible.

    """
    if x is None:
        return None

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
            return float(arr.item())
        return None

    # everything else (str, dict, enum, ...)
    return None


def _flatten_dict_raw(dct: dict[str, Any]) -> dict[str, float]:
    """
    Flatten nested dictionary into a single-level dictionary with float values.

    Parameters
    ----------
    dct : dict[str, Any]
        Input nested dictionary.

    Returns
    -------
    dict[str, float]
        Flattened dictionary with float values.

    """
    out: dict[str, float] = {}

    def _rec(key: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _rec(f"{key}_{k}", v)
            return

        if isinstance(obj, (list, tuple, np.ndarray)) and not np.isscalar(obj):
            for i, v in enumerate(obj):
                _rec(f"{key}_{i}", v)
            return

        val = _as_float(obj)
        if val is not None:
            out[key] = val

    for k, v in dct.items():
        _rec(k, v)

    return out


def _selected_datasets(dataset_selector: CheckboxGroup) -> list[str]:
    """
    Get list of selected dataset names from the dataset selector widget.

    Parameters    ----------
    dataset_selector : CheckboxGroup
        Dataset selector widget.

    Returns
    -------
    list[str]
        List of selected dataset names.

    Raises
    ------
    ValueError
        If no dataset is selected.

    """
    active = [n for n, cb in dataset_selector.boxes.items() if cb.value]
    if not active:
        msg = "Select at least one dataset."
        raise ValueError(msg)
    return active


def _clip_for_plot(vals: np.ndarray, *, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    """
    Clip extreme values based on percentiles for better plotting.

    Parameters
    ----------
    vals : np.ndarray
        Input values.
    q_low : float
        Lower percentile threshold.
    q_high : float
        Upper percentile threshold.

    Returns
    -------
    np.ndarray
        Clipped values.

    """
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:  # noqa: PLR2004
        return vals

    lo, hi = np.percentile(vals, [q_low, q_high])
    if lo >= hi:
        return vals

    return vals[(vals >= lo) & (vals <= hi)]


# ============================================================================
# DATA-DRIVEN BINNING + LAYOUT
# ============================================================================


def _infer_bins(vals: np.ndarray, *, min_bins: int = 10, max_bins: int = 80) -> int:
    """
    Infer number of histogram bins using Freedman-Diaconis rule.

    Parameters
    ----------
    vals : np.ndarray
        Input values.
    min_bins : int
        Minimum number of bins.
    max_bins : int
        Maximum number of bins.

    Returns
    -------
    int
        Inferred number of bins.

    """
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:  # noqa: PLR2004
        return 1

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 1
    if vmin == vmax:
        return 1

    if np.isclose(vmin, vmax, rtol=0.0, atol=1e-14):
        return 1

    q25, q75 = np.percentile(vals, [25, 75])
    iqr = float(q75 - q25)

    if iqr <= 0.0:
        return int(np.clip(min_bins, 1, max_bins))

    bw = 2.0 * iqr * vals.size ** (-1.0 / 3.0)
    if not np.isfinite(bw) or bw <= 0.0:
        return int(np.clip(min_bins, 1, max_bins))

    bins = int(np.ceil((vmax - vmin) / bw))
    return int(np.clip(bins, 1, max_bins))


def _infer_ncols(n_items: int, *, max_cols: int = 5) -> int:
    """
    Infer number of columns for subplot grid layout.

    Parameters
    ----------
    n_items : int
        Number of items to plot.
    max_cols : int
        Maximum number of columns.

    Returns
    -------
    int
        Inferred number of columns.

    """
    return min(max_cols, max(1, int(np.ceil(np.sqrt(n_items)))))


# ============================================================================
# HIST GRID (GENERIC)
# ============================================================================


def _hist_grid(
    *,
    data_by_dataset: dict[str, dict[str, np.ndarray]],
    active_datasets: list[str],
    columns: list[str],
    title: str,
) -> Figure:
    """
    Create a grid of histogram plots for multiple datasets and columns.

    Parameters
    ----------
    data_by_dataset : dict[str, dict[str, np.ndarray]]
        Data organized by dataset name and column name.
    active_datasets : list[str]
        List of active dataset names to plot.
    columns : list[str]
        List of column names to plot.
    title : str
        Title for the entire figure.

    Returns
    -------
    Figure
        Matplotlib Figure containing the histogram grid.

    """
    cmap = plt.get_cmap("tab10")
    dataset_colors = {name: cmap(i % 10) for i, name in enumerate(active_datasets)}

    ncols = _infer_ncols(len(columns))
    nrows = math.ceil(len(columns) / ncols)

    fig = plt.figure(figsize=(4.0 * ncols + 2.0, 2.8 * nrows))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.35],
        hspace=0.35,
        wspace=0.25,
    )

    axes: list[Axes] = []
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            if idx < len(columns):
                axes.append(fig.add_subplot(gs[r, c]))

    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.axis("off")

    legend_handles = [Line2D([], [], lw=6, color=dataset_colors[name], alpha=0.6) for name in active_datasets]

    for ax, key in zip(axes, columns, strict=False):
        for name in active_datasets:
            vals = data_by_dataset[name].get(key, np.array([]))
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:  # noqa: PLR2004
                continue

            color = dataset_colors[name]
            bins = _infer_bins(vals)

            if bins == 1:
                x0 = float(np.mean(vals))
                width = 0.02 * max(1.0, abs(x0))

                ax.bar(
                    x0,
                    vals.size,
                    width=width,
                    color=color,
                    alpha=0.35,
                    align="center",
                )

            else:
                _, bin_edges, _ = ax.hist(vals, bins=bins, color=color, alpha=0.35)

                if np.nanstd(vals) > 0:
                    try:
                        kde = gaussian_kde(vals)
                        x = np.linspace(vals.min(), vals.max(), 300)
                        bw = bin_edges[1] - bin_edges[0]
                        ax.plot(x, kde(x) * vals.size * bw, color=color, lw=1.6)
                    except LinAlgError:
                        pass

        ax.set_title(key, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.25)

    ax_leg.legend(legend_handles, active_datasets, title="Dataset", loc="upper left")
    fig.suptitle(title)
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.06, right=0.98)

    return fig


# ============================================================================
# META STATISTICS
# ============================================================================


def plot_meta_statistics(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot meta statistics distributions from datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of dataset names to DataFrames.

    Returns
    -------
        widgets.VBox
            VBox containing the plot and controls.

    """
    names = list(datasets.keys())
    cache: dict[str, _StatCache] = {n: {"loaded": 0, "cols": {}} for n in names}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame], dataset_selector: CheckboxGroup) -> Figure:
        """
        Plot function for case count viewer.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Dictionary of dataset names to DataFrames.
        dataset_selector : CheckboxGroup
            Dataset selector widget.

        Returns
        -------
        Figure
            Matplotlib Figure containing the histogram grid.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for m in df.iloc[entry["loaded"] : max_cases]["meta"]:
                gen = m.get("generator", {})
                for block in gen.values():
                    stats = block.get("statistics", {})
                    if isinstance(stats, dict) and stats:
                        flat = _flatten_dict_raw(stats)
                        for k, v in flat.items():
                            entry["cols"].setdefault(k, []).append(v)

            entry["loaded"] = max_cases

        active = _selected_datasets(dataset_selector)
        keys = list(dict.fromkeys(k for n in active for k in cache[n]["cols"]))

        data = {
            n: {
                k: np.asarray(
                    [v for v in cache[n]["cols"].get(k, []) if isinstance(v, (int, float, np.floating))],
                    dtype=float,
                )
                for k in keys
            }
            for n in active
        }

        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=keys,
            title=f"Meta statistics distributions (first {max_cases} cases)",
        )

    ds = analysis.ui.components.ui_checkbox_datasets(dataset_names=names)
    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=250,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )


# ============================================================================
# 1-2. META PARAMETERS (AUTO)
# ============================================================================


def plot_meta_parameters(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot meta parameters distributions from datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of dataset names to DataFrames.

    Returns
    -------
        widgets.VBox
            VBox containing the plot and controls.

    """
    names = list(datasets.keys())
    cache: dict[str, _ParamCache] = {n: {"loaded": 0, "cols": {}} for n in names}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame], dataset_selector: CheckboxGroup) -> Figure:
        """
        Plot function for case count viewer.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Dictionary of dataset names to DataFrames.
        dataset_selector : CheckboxGroup
            Dataset selector widget.

        Returns
        -------
        Figure
            Matplotlib Figure containing the histogram grid.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for m in df.iloc[entry["loaded"] : max_cases]["meta"]:
                gen = m.get("generator", {})
                for block in gen.values():
                    params = block.get("parameters", {})
                    if isinstance(params, dict) and params:
                        flat = _flatten_dict_raw(params)
                        for k, v in flat.items():
                            entry["cols"].setdefault(k, []).append(v)

            entry["loaded"] = max_cases

        active = _selected_datasets(dataset_selector)
        keys = list(dict.fromkeys(k for n in active for k in cache[n]["cols"]))

        data = {
            n: {
                k: np.asarray(
                    [v for v in cache[n]["cols"].get(k, []) if isinstance(v, (int, float, np.floating))],
                    dtype=float,
                )
                for k in keys
            }
            for n in active
        }

        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=keys,
            title=f"Meta parameter distributions (first {max_cases} cases)",
        )

    ds = analysis.ui.components.ui_checkbox_datasets(dataset_names=names)
    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=250,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )


# ============================================================================
# 1-3. FIELD VALUE DISTRIBUTIONS (AUTO)
# ============================================================================


def plot_field_value_distributions(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot field value distributions from datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of dataset names to DataFrames.

    Returns
    -------
        widgets.VBox
            VBox containing the plot and controls.

    """
    names = list(datasets.keys())

    # infer fields dynamically from first dataset
    sample_df = next(iter(datasets.values()))
    field_names = [c for c in sample_df.columns if c not in {"meta", "x", "y"}]
    cache: dict[str, _FieldCache] = {
        n: {
            "loaded": 0,
            "data": {f: {"min": [], "mean": [], "max": []} for f in field_names},
        }
        for n in names
    }

    cmap = plt.get_cmap("tab10")
    dataset_colors = {name: cmap(i % 10) for i, name in enumerate(names)}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame], dataset_selector: CheckboxGroup) -> Figure:
        """
        Plot function for case count viewer.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Dictionary of dataset names to DataFrames.
        dataset_selector : CheckboxGroup
            Dataset selector widget.

        Returns
        -------
        Figure
            Matplotlib Figure containing the histogram grid.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for _, row in df.iloc[entry["loaded"] : max_cases].iterrows():
                for f in field_names:
                    arr = np.asarray(row[f])
                    if arr.size == 0:
                        continue
                    entry["data"][f]["min"].append(float(np.nanmin(arr)))
                    entry["data"][f]["mean"].append(float(np.nanmean(arr)))
                    entry["data"][f]["max"].append(float(np.nanmax(arr)))

            entry["loaded"] = max_cases

        active = _selected_datasets(dataset_selector)

        nrows = len(field_names)
        ncols = 3

        fig = plt.figure(figsize=(12.5, 2.4 * nrows))
        gs = fig.add_gridspec(
            nrows,
            ncols + 1,
            width_ratios=[1.0, 1.0, 1.0, 0.35],
            hspace=0.35,
            wspace=0.25,
        )

        axes = [[fig.add_subplot(gs[i, j]) for j in range(ncols)] for i in range(nrows)]
        ax_leg = fig.add_subplot(gs[:, -1])
        ax_leg.axis("off")

        legend_handles = [Line2D([], [], lw=6, color=dataset_colors[name], alpha=0.6) for name in active]

        for i, f in enumerate(field_names):
            for j, stat in enumerate(["min", "mean", "max"]):
                ax = axes[i][j]

                for name in active:
                    vals = np.asarray(cache[name]["data"][f][stat])
                    vals = _clip_for_plot(vals)
                    if vals.size < 2:  # noqa: PLR2004
                        continue

                    color = dataset_colors[name]
                    bins = _infer_bins(vals)

                    if bins == 1:
                        # --- single-value spike instead of full-width histogram ---
                        x0 = float(np.mean(vals))
                        width = 0.02 * max(1.0, abs(x0))

                        ax.bar(
                            x0,
                            vals.size,
                            width=width,
                            color=color,
                            alpha=0.35,
                            align="center",
                        )

                    else:
                        _, bin_edges, _ = ax.hist(vals, bins=bins, color=color, alpha=0.35)

                        if np.nanstd(vals) > 0:
                            try:
                                kde = gaussian_kde(vals)
                                x = np.linspace(vals.min(), vals.max(), 300)
                                bw = bin_edges[1] - bin_edges[0]
                                ax.plot(x, kde(x) * vals.size * bw, color=color, lw=1.6)
                            except LinAlgError:
                                pass
                FIELD_LABELS: dict[str, str] = {
                    "kxx": r"log10($k_{xx}$)",
                    "kyy": r"log10($k_{yy}$)",
                    "kxy": r"$\hat{k}_{xy}$ [-]",
                    "eps": r"$\varepsilon$ [-]",
                    "p_bc": r"$p_{\mathrm{bc}}$ [Pa]",
                    "p": r"$p$ [Pa]",
                    "u": r"$u$ [m/s]",
                    "v": r"$v$ [m/s]",
                    "U": r"$|\mathbf{u}|$ [m/s]",
                }

                if i == 0:
                    ax.set_title(stat)
                if j == 0:
                    ax.set_ylabel("Count")
                    ax.annotate(
                        FIELD_LABELS.get(f, f),
                        xy=(-0.28, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=11,
                        fontweight="bold",
                    )

                ax.set_xlabel("Value")
                ax.grid(True, linestyle="--", alpha=0.25)

        ax_leg.legend(legend_handles, active, title="Dataset", loc="upper left")
        fig.suptitle(f"Field value distributions per channel (first {max_cases} cases)")
        fig.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.97)

        return fig

    ds = analysis.ui.components.ui_checkbox_datasets(dataset_names=names)
    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=250,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )
