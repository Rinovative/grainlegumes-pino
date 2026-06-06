"""
===============================================================================
evaluation_plot_outlier_analysis.py
===============================================================================
Outlier and extreme case analysis for model evaluation.

Provides:
  - identification and visualization of worst-performing cases
  - extreme error case analysis and grouping
  - per-case and multi-case outlier exploration
  - interactive viewers for outlier investigation

Responsibilities:
  - identify cases with anomalous or extreme errors
  - support visual inspection of outlier cases
  - enable root-cause analysis for performance issues

Design principles:
  - outliers are ranked by error magnitude or statistical deviation
  - plots support interactive case selection and comparison
  - visual feedback identifies extreme performance regions

This module does NOT:
  - perform statistical outlier filtering or removal
  - compute error decomposition (use evaluation_plot_error_decomposition)
  - handle model retraining or refinement
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib import cm

from src import analysis, domain

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# =============================================================================
# CHANNELS AND UNITS
# =============================================================================
CHANNELS = domain.fields.OUTPUT_FIELDS
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}
UNIT_MAP = {
    "p": "Pa",
    "u": "m/s",
    "v": "m/s",
    "U": "m/s",
}

# =============================================================================
# CONSTANTS
# =============================================================================
N_LEVELS = 10
MASK_THRESHOLD = 1e-4

# ============================================================================
# HELPERS
# ============================================================================


def _as_series(row: pd.Series | pd.DataFrame) -> pd.Series:
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _scalar(x: Any) -> float:
    if isinstance(x, pd.Series):
        x = x.iloc[0]

    if x is None or pd.isna(x):
        return float("nan")

    try:
        xf = float(x)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(xf):
        return float("nan")

    return xf


def _fmt_num(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(xf):
        return ""
    if 1e-4 <= abs(xf) < 1e4:  # noqa: PLR2004
        return f"{xf:.3g}".rstrip("0").rstrip(".")
    return f"{xf:.2e}"


def _is_parameter_column(c: str) -> bool:
    """
    Check if a column name corresponds to a parameter column.

    Parameters
    ----------
    c : str
        Column name.

    Returns
    -------
    bool
        True if the column is a parameter column, False otherwise.

    """
    return c.startswith("par_") or (c.startswith("generator_") and "_parameters_" in c)


# =============================================================================
# GLOBAL CACHES
# =============================================================================

_npz_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
_case_map_cache: dict[str, dict[str, list[int]]] = {}
_ref_case_cache: dict[str, int] = {}


# =============================================================================
# NPZ LOADING
# =============================================================================


def _load_npz(row: pd.Series | pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load prediction, ground truth, error, and kappa from NPZ file.

    Uses a global cache to avoid redundant loading.

    Parameters
    ----------
    row : pd.Series
        DataFrame row containing the 'npz_path' column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
        Tuple containing prediction, ground truth, error, kappa, and kappa names.

    """
    row = _as_series(row)
    key = str(Path(row.loc["npz_path"]))
    if key in _npz_cache:
        return _npz_cache[key]

    data = np.load(key, allow_pickle=True)
    pred = np.asarray(data["pred"])
    gt = np.asarray(data["gt"])
    err = np.asarray(data["err"])
    kappa = np.asarray(data["kappa"])
    kappa_names = [str(n) for n in data["kappa_names"]]

    # -------------------------------------------------
    # Normalize shapes to (C, H, W)
    # Accept common storage variants:
    #   (1, C, H, W) -> (C, H, W)
    #   (C, H, W)    -> (C, H, W)
    #   (H, W)       -> (1, H, W)
    # -------------------------------------------------
    def _to_chw(a: np.ndarray) -> np.ndarray:
        if a.ndim == 4 and a.shape[0] == 1:  # noqa: PLR2004
            return a[0]
        if a.ndim == 3:  # noqa: PLR2004
            return a
        if a.ndim == 2:  # noqa: PLR2004
            return a[np.newaxis, ...]
        msg = f"Unsupported array shape: {a.shape}"
        raise ValueError(msg)

    pred = _to_chw(pred)
    gt = _to_chw(gt)
    err = _to_chw(err)

    # kappa: commonly (1, K, H, W) or (K, H, W)
    if kappa.ndim == 4 and kappa.shape[0] == 1:  # noqa: PLR2004
        kappa = kappa[0]
    elif kappa.ndim != 3:  # noqa: PLR2004
        msg = f"Unsupported kappa shape: {kappa.shape}"
        raise ValueError(msg)

    _npz_cache[key] = (pred, gt, err, kappa, kappa_names)
    return pred, gt, err, kappa, kappa_names


# =============================================================================
# METRICS
# =============================================================================


def _rel_l2(err: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute relative L2 error.

    Parameters
    ----------
    err : np.ndarray
        Error array.
    gt : np.ndarray
        Ground truth array.

    Returns
    -------
    float
        Relative L2 error.

    """
    return float(np.linalg.norm(err) / (np.linalg.norm(gt) + 1e-12))


def _rel_l2_per_channel(err: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Compute relative L2 error per channel.

    Parameters
    ----------
    err : np.ndarray
        Error array.
    gt : np.ndarray
        Ground truth array.

    Returns
    -------
    dict[str, float]
        Relative L2 error per channel.

    """
    out = {}
    for ch in CHANNELS:
        k = CHANNEL_INDICES[ch]
        if k < err.shape[0] and k < gt.shape[0]:
            out[ch] = _rel_l2(err[k], gt[k])
    return out


# =============================================================================
# REFERENCE CASE (PARAMETER CENTER)
# =============================================================================


def _select_reference_case(df: pd.DataFrame) -> int:
    """
    Select the reference case based on the parameter center (median).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cases.

    Returns
    -------
    int
        Index of the reference case.

    """
    par_cols = [c for c in df.columns if _is_parameter_column(c) and c != "par_seed"]

    if not par_cols:
        return int(df.index[len(df) // 2])

    P = df[par_cols].astype(float)
    center = P.median()
    dist = ((P - center) / (P.std() + 1e-12)).pow(2).sum(axis=1)
    return int(dist.idxmin())


# =============================================================================
# WORST-CASE SELECTION (PER CHANNEL)
# =============================================================================


def _select_topk_per_channel(*, datasets: dict[str, pd.DataFrame], k: int) -> dict[str, dict[str, list[int]]]:
    """
    Select the top-k worst cases per channel based on relative L2 error.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets.
    k : int
        Number of top cases to select per channel.

    Returns
    -------
    dict[str, dict[str, list[int]]]
        Dictionary mapping dataset names to channel names to lists of case indices.

    """
    out: dict[str, dict[str, list[int]]] = {}
    k_top = int(k)

    for name, df in datasets.items():
        out[name] = {}
        for ch in CHANNELS:
            ch_idx = CHANNEL_INDICES[ch]
            scores = []
            for idx, row in df.iterrows():
                _, gt, err, _, _ = _load_npz(row)
                if ch_idx < err.shape[0] and ch_idx < gt.shape[0]:
                    scores.append((idx, _rel_l2(err[ch_idx], gt[ch_idx])))
            scores.sort(key=lambda x: x[1], reverse=True)
            out[name][ch] = [i for i, _ in scores[:k_top]]

    return out


# =============================================================================
# KAPPA AGGREGATION
# =============================================================================


def _aggregate_kappa(kappa: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Aggregate permeability tensor to a scalar field using ONLY diagonal components.

    Supported:
        - kxx, kyy          (2D)
        - kxx, kyy, kzz     (3D)
    Everything else raises to avoid wrong visualisation.

    Parameters
    ----------
    kappa : np.ndarray
        Kappa tensor array.
    names : list[str]
        Names of the kappa components.

    Returns
    -------
    np.ndarray
        Aggregated scalar kappa field.

    """
    name_to_idx = {str(name).lower(): i for i, name in enumerate(names)}

    has_2d = {"kxx", "kyy"}.issubset(name_to_idx)
    has_3d = {"kxx", "kyy", "kzz"}.issubset(name_to_idx)

    if has_3d:
        return (kappa[name_to_idx["kxx"]] + kappa[name_to_idx["kyy"]] + kappa[name_to_idx["kzz"]]) / 3.0

    if has_2d:
        return 0.5 * (kappa[name_to_idx["kxx"]] + kappa[name_to_idx["kyy"]])

    msg = f"kappa aggregation expects diagonal components {{kxx, kyy}} (and optional kzz). Got: {sorted(name_to_idx.keys())}"
    raise ValueError(msg)


# =============================================================================
# SHARED 4x4 PLOT KERNEL
# =============================================================================


def _plot_prediction_overview_case(
    *,
    row: pd.Series,
    dataset_name: str,
    case_label: str,
    error_mode: widgets.ValueWidget,
) -> Figure:
    """
    Plot a 4x4 overview of prediction, ground truth, error, and kappa for a given case.

    Parameters
    ----------
    row : pd.Series
        The DataFrame row containing the case data.
    dataset_name : str
        The name of the dataset.
    case_label : str
        A label for the case (e.g., "Case 42 | channel=u").
    error_mode : widgets.ValueWidget
        A widget indicating the error mode ("MAE" or "RelErr").

    Returns
    -------
        Figure
            The generated matplotlib Figure.

    """
    pred, gt, err, kappa, kappa_names = _load_npz(row)

    Lx = float(row.loc["geometry_Lx"])
    Ly = float(row.loc["geometry_Ly"])
    # -------------------------------------------------
    # Robust shape handling (C,H,W) vs (H,W)
    # -------------------------------------------------
    if pred.ndim == 3:  # noqa: PLR2004
        ny, nx = pred[CHANNEL_INDICES[CHANNELS[0]]].shape
    elif pred.ndim == 2:  # noqa: PLR2004
        ny, nx = pred.shape
    else:
        msg = f"Unsupported pred shape: {pred.shape}"
        raise ValueError(msg)

    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    fig, axes = plt.subplots(4, 4, figsize=(20, 9))

    kappa_field = _aggregate_kappa(kappa, kappa_names)
    kappa_levels = analysis.ui.components.compute_levels(kappa_field, N_LEVELS)

    kappa_log_field = np.log10(np.maximum(kappa_field, 1e-30))
    kappa_log_levels = analysis.ui.components.compute_levels(kappa_log_field, N_LEVELS)

    nrows = 4  # fixed layout

    for r, ch in enumerate(CHANNELS):
        is_last = r == nrows - 1

        # Prediction
        ax = axes[r, 0]
        k = CHANNEL_INDICES[ch]
        if k >= pred.shape[0]:
            ax.axis("off")
            continue
        field = pred[k]

        im = ax.contourf(
            X,
            Y,
            field,
            levels=analysis.ui.components.compute_levels(field),
            cmap="turbo",
        )
        if ch in {"u", "v", "U"}:
            u = pred[CHANNEL_INDICES["u"]]
            v = pred[CHANNEL_INDICES["v"]]
            analysis.ui.components.overlay_streamlines(ax, X, Y, u, v)
        ax.set_title(f"{ch} pred [{UNIT_MAP[ch]}]")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(*im.get_clim()))
        analysis.ui.components.apply_axis_labels(ax, 0, Lx, Ly, is_last_row=is_last)

        # Ground truth
        ax = axes[r, 1]
        im = ax.contourf(X, Y, gt[k], levels=analysis.ui.components.compute_levels(gt[k]), cmap="turbo")
        if ch in {"u", "v", "U"}:
            u = gt[CHANNEL_INDICES["u"]]
            v = gt[CHANNEL_INDICES["v"]]
            analysis.ui.components.overlay_streamlines(ax, X, Y, u, v)
        ax.set_title(f"{ch} true [{UNIT_MAP[ch]}]")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(*im.get_clim()))
        analysis.ui.components.apply_axis_labels(ax, 1, Lx, Ly, is_last_row=is_last)

        # Error
        ax = axes[r, 2]
        if error_mode.value == "MAE":
            field = np.abs(err[k])
            field = np.nan_to_num(field, nan=0.0)
            levels = np.linspace(
                0.0,
                np.nanquantile(field, 0.99),
                N_LEVELS,
            )
            title = f"{ch} MAE [{UNIT_MAP[ch]}]"
        else:
            abs_err = np.abs(err[k])
            true_abs = np.abs(gt[k])
            field = np.asarray(abs_err / (true_abs + 1e-12) * 100.0, dtype=float)
            field[true_abs < MASK_THRESHOLD] = np.nan
            levels = np.linspace(
                0.0,
                np.nanquantile(field, 0.99),
                N_LEVELS,
            )
            title = f"{ch} rel err [%]"

        im = ax.contourf(X, Y, field, levels=levels, cmap="Blues")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(*im.get_clim()))
        analysis.ui.components.apply_axis_labels(ax, 2, Lx, Ly, is_last_row=is_last)

        # Kappa
        ax = axes[r, 3]
        if r == 0:
            im = ax.contourf(
                X,
                Y,
                kappa_field,
                levels=kappa_levels,
                cmap="viridis",
            )
            ax.set_title("kappa [m²]")

        elif r == 1:
            im = ax.contourf(
                X,
                Y,
                kappa_log_field,
                levels=kappa_log_levels,
                cmap="viridis",
            )
            ax.set_title("log10(kappa) [m²]")
        else:
            ax.axis("off")
            continue

        # HIER:
        fig.colorbar(im, ax=ax, fraction=0.04)
        analysis.ui.components.apply_axis_labels(ax, 3, Lx, Ly, is_last_row=is_last)

    fig.suptitle(f"{dataset_name} — {case_label}", fontsize=14)
    fig.tight_layout()
    return fig


# =============================================================================
# TABLE VIEW — REFERENCE INCLUDED IN TABLE
# =============================================================================


def _style_rank_diverging(df: pd.DataFrame, columns: list[str], kind: pd.Series) -> pd.DataFrame:
    """
    Style function to highlight cells based on their rank diverging from the reference case.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be styled.
    columns : list[str]
        The list of columns to apply the styling to.
    kind : pd.Series
        A Series indicating the kind of each row (e.g., "reference" or "worst").

    Returns
    -------
        pd.DataFrame
            A DataFrame with the same shape as `df`, containing the style strings for each cell.

    """
    out = {}

    cmap_low = cm.get_cmap("Blues")
    cmap_high = cm.get_cmap("Purples")

    ref_idx = kind[kind == "reference"].index[0]

    for c in columns:
        col = df[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue

        ref_val = col.loc[ref_idx]

        lower = col[col < ref_val]
        higher = col[col > ref_val]

        # Ranks getrennt
        r_low = lower.rank(method="average", ascending=True)
        r_high = higher.rank(method="average", ascending=True)

        def _cell(
            idx: int,
            *,
            col: pd.Series = col,
            r_low: pd.Series = r_low,
            r_high: pd.Series = r_high,
        ) -> str:
            if idx == ref_idx or not np.isfinite(col.loc[idx]):
                return "background-color: white"

            # ---------- smaller than reference ----------
            if idx in r_low.index:
                t = (r_low.loc[idx] - 1) / max(len(r_low) - 1, 1)
                x = 0.75 - 0.5 * t  # [0.75 → 0.25]
                cr, cg, cb, _ = cmap_low(x)

            # ---------- greater than reference ----------
            elif idx in r_high.index:
                t = (r_high.loc[idx] - 1) / max(len(r_high) - 1, 1)
                x = 0.25 + 0.5 * t  # [0.25 → 0.75]
                cr, cg, cb, _ = cmap_high(x)

            else:
                return "background-color: white"

            return f"background-color: rgba({int(cr * 255)}, {int(cg * 255)}, {int(cb * 255)}, 0.65)"

        out[c] = [_cell(i) for i in df.index]

    return pd.DataFrame(out)


def _short_parameter_name(c: str) -> str:
    """
    Shorten parameter column names for display only.

    Examples
    --------
    par_var -> var
    generator_tensor_parameters_strength -> tensor_strength

    Parameters
    ----------
    c : str
        Original column name.

    Returns
    -------
    str
        Shortened column name.

    """
    if c.startswith("par_"):
        return c[len("par_") :]

    if c.startswith("generator_") and "_parameters_" in c:
        return c.split("_parameters_", 1)[1]

    return c


def plot_outlier_tables_per_channel(*, datasets: dict[str, pd.DataFrame], k: int = 5) -> widgets.VBox:
    """
    Plot tables showing the worst k cases per channel, including the reference case.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.
    k : int, optional
        The number of worst cases to select per channel, by default 5.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the tables for each channel.

    """
    run_btn = widgets.Button(description="Compute tables", button_style="primary", icon="play")
    dataset_sel = widgets.Dropdown(options=list(datasets), description="Select:")
    output = widgets.Output()

    def _parameter_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if _is_parameter_column(c) and c != "par_seed"]

    def _render(_: Any = None) -> None:
        output.clear_output(wait=True)
        name = dataset_sel.value
        df = datasets[name]
        par_cols = _parameter_columns(df)

        if name not in _case_map_cache:
            _case_map_cache[name] = _select_topk_per_channel(datasets={name: df}, k=k)[name]
            _ref_case_cache[name] = _select_reference_case(df)

        with output:
            display(Markdown(f"### `{name}`"))

            for ch in CHANNELS:
                display(Markdown(f"#### Channel `{ch}`"))

                rows = []

                # Worst cases
                for idx in _case_map_cache[name][ch]:
                    row = _as_series(df.loc[idx])
                    _, gt, err, _, _ = _load_npz(row)
                    rels = _rel_l2_per_channel(err, gt)

                    r = {
                        "case_index": int(row.loc["case_index"]),
                        f"rel_l2[{ch}]": rels[ch],
                        "rel_l2_global": _scalar(row.loc["rel_l2"]),
                        "l2_global": _scalar(row.loc["l2"]),
                        "__kind__": "worst",
                    }
                    for p in par_cols:
                        r[p] = _scalar(row.loc[p])
                    rows.append(r)

                # Reference case (APPENDED)
                ref_idx = _ref_case_cache[name]
                row = _as_series(df.loc[ref_idx])
                _, gt, err, _, _ = _load_npz(row)
                rels = _rel_l2_per_channel(err, gt)

                r = {
                    "case_index": f"{int(row.loc['case_index'])} (Ref)",
                    f"rel_l2[{ch}]": rels[ch],
                    "rel_l2_global": _scalar(row.loc["rel_l2"]),
                    "l2_global": _scalar(row.loc["l2"]),
                    "__kind__": "reference",
                }
                for p in par_cols:
                    r[p] = _scalar(row[p])
                rows.append(r)

                df_out = pd.DataFrame(rows)
                kind = df_out.pop("__kind__")
                rename_map = {c: _short_parameter_name(c) for c in df_out.columns}
                df_out = df_out.rename(columns=rename_map)

                display(
                    df_out.style.format({c: _fmt_num for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])})
                    .apply(
                        lambda _, df_out=df_out, kind=kind: _style_rank_diverging(
                            df_out,
                            [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c]) and c != "case_index"],
                            kind,
                        ),
                        axis=None,
                    )
                    .apply(
                        lambda _, df_out=df_out, kind=kind: ["" if kind.loc[i] == "reference" else "background-color: white" for i in df_out.index],
                        axis=0,
                        subset=["case_index"],
                    )
                )

    run_btn.on_click(_render)
    return widgets.VBox([widgets.HBox([dataset_sel, run_btn]), output])


# =============================================================================
# FIELD PLOT — WORST PER-CHANNEL (REFERENCE LAST, LAZY)
# =============================================================================


def _num_cases_outlier_viewer(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    k: int,
    channel: str,
) -> int:
    """
    Determine the number of cases to display in the outlier viewer.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    df : pd.DataFrame
        The DataFrame containing the cases.
    k : int
        The number of worst cases to select per channel.
    channel : str
        The channel to focus on.

    Returns
    -------
        int
            The number of cases to display (worst cases + 1 reference).

    """
    # Lazy cache init
    if dataset_name not in _case_map_cache:
        _case_map_cache[dataset_name] = _select_topk_per_channel(datasets={dataset_name: df}, k=k)[dataset_name]
        _ref_case_cache[dataset_name] = _select_reference_case(df)

    # worst cases + 1 reference
    return len(_case_map_cache[dataset_name][channel]) + 1


def plot_outlier_cases_per_channel(*, datasets: dict[str, pd.DataFrame], k: int = 5) -> widgets.VBox:
    """
    Plot an interactive viewer for the worst k cases per channel, including the reference case.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.
    k : int, optional
        The number of worst cases to select per channel, by default 5.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the interactive case viewer.

    """
    ch_sel = analysis.ui.components.ui_dropdown_channel()
    err_sel = analysis.ui.components.ui_radio_error_mode()

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str, focus_channel: widgets.ValueWidget, error_mode: widgets.ValueWidget) -> Figure:
        """
        Plot function for the interactive case viewer.

        Parameters
        ----------
        idx : int
            The index of the case to plot.
        df : pd.DataFrame
            The DataFrame containing the cases.
        dataset_name : str
            The name of the dataset.
        focus_channel : widgets.ValueWidget
            The widget indicating the focus channel.
        error_mode : widgets.ValueWidget
            The widget indicating the error mode.

        Returns
        -------
            Figure
                The generated matplotlib Figure.

        """
        if dataset_name not in _case_map_cache:
            _case_map_cache[dataset_name] = _select_topk_per_channel(datasets={dataset_name: df}, k=k)[dataset_name]
            _ref_case_cache[dataset_name] = _select_reference_case(df)

        ch = focus_channel.value
        worst = _case_map_cache[dataset_name][ch]
        n = len(worst)

        if idx < n:
            df_idx = worst[idx]
            row = _as_series(df.loc[df_idx])
            case_id = row.get("case_index", df_idx)
            label = f"Case {case_id} | channel={ch}"
        else:
            ref_idx = _ref_case_cache[dataset_name]
            row = _as_series(df.loc[ref_idx])
            case_id = row.get("case_index", ref_idx)
            label = f"Case {case_id} | channel={ch} (reference)"

        return _plot_prediction_overview_case(
            row=row,
            dataset_name=dataset_name,
            case_label=label,
            error_mode=error_mode,
        )

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        enable_dataset_dropdown=True,
        extra_widgets=[ch_sel, err_sel],
        focus_channel=ch_sel,
        error_mode=err_sel,
        n_cases_fn=lambda name, df: _num_cases_outlier_viewer(
            name,
            df,
            k=k,
            channel=str(ch_sel.value),
        ),
    )


# =============================================================================
# TABLE VIEW — EXTREME INPUT PARAMETERS
# =============================================================================


def plot_extreme_input_table(*, datasets: dict[str, pd.DataFrame]) -> widgets.HBox:
    """
    Plot tables summarizing extreme input parameter values for each dataset.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the summary tables.

    """
    tables: list[widgets.Output] = []

    for name, df in datasets.items():
        out = widgets.Output()

        with out:
            par_cols = [c for c in df.columns if _is_parameter_column(c) and c != "par_seed"]
            ref_idx = _select_reference_case(df)
            ref_row = _as_series(df.loc[ref_idx])

            rows = []
            for p in par_cols:
                col = df[p].astype(float)
                rows.append(
                    {
                        "parameter": p,
                        "min": col.min(),
                        "ref": ref_row.loc[p],
                        "max": col.max(),
                        "min/ref": col.min() / ref_row.loc[p] if ref_row.loc[p] != 0 else np.nan,
                        "max/ref": col.max() / ref_row.loc[p] if ref_row.loc[p] != 0 else np.nan,
                    }
                )

            df_out = pd.DataFrame(rows).set_index("parameter")

            display(Markdown(f"### `{name}`"))
            display(df_out.style.format(dict.fromkeys(df_out.columns, _fmt_num)))

        tables.append(out)

    return widgets.HBox(tables)


# =============================================================================
# FIELD PLOT — EXTREME INPUT CASES
# =============================================================================


def _candidate_input_columns(*, datasets: dict[str, pd.DataFrame]) -> list[str]:
    """
    Identify candidate input parameter columns from the datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.

    Returns
    -------
    list[str]
            A list of candidate input parameter column names.

    """
    first_df = next(iter(datasets.values()))
    return [c for c in first_df.columns if _is_parameter_column(c) and c != "par_seed"]


def _select_extreme_inputs(*, datasets: dict[str, pd.DataFrame], column: str) -> dict[str, pd.DataFrame]:
    """
    Select cases with extreme values for a given input parameter column.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.
    column : str
        The input parameter column to analyze.

    Returns
    -------
        dict[str, pd.DataFrame]
            A dictionary mapping dataset names to DataFrames containing the extreme cases.

    """

    def _to_scalar(v: Any) -> float:
        if v is None or pd.isna(v):
            return float("nan")

        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            return float("nan")

        xf = float(arr) if arr.ndim == 0 else float(np.nanmean(arr))

        if not np.isfinite(xf):
            return float("nan")

        return xf

    out = {}
    for name, df in datasets.items():
        v = df[column].apply(_to_scalar)
        out[name] = df.loc[[v.idxmin(), v.idxmax()]].reset_index(drop=True)
    return out


# =============================================================================
# Helper: 4x4 Prediction/GT/Error/Kappa (reusable)
# =============================================================================
def _plot_prediction_overview_case_4x4(
    *,
    row: pd.Series,
    dataset_name: str,
    case_label: str,
    error_mode: widgets.ValueWidget,
) -> Figure:
    """
    Plot a 4x4 overview of prediction, ground truth, error, and kappa for a given case.

    Parameters
    ----------
    row : pd.Series
        The DataFrame row containing the case data.
    dataset_name : str
        The name of the dataset.
    case_label : str
        A label for the case (e.g., "Case 42 | channel=u").
    error_mode : widgets.ValueWidget
        A widget indicating the error mode ("MAE" or "RelErr").

    Returns
    -------
        Figure
            The generated matplotlib Figure.

    """
    cmap_pred_true = "turbo"
    cmap_error = "Blues"
    cmap_kappa = "viridis"
    n_levels = 10
    mask_threshold = 1e-4

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

    for r, label in enumerate(CHANNELS):
        is_last_row = r == nrows - 1

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------
        ax = axes[r, 0]
        k = CHANNEL_INDICES[label]
        field = pred[k]

        im = ax.contourf(
            X,
            Y,
            field,
            levels=analysis.ui.components.compute_levels(field, n_levels),
            cmap=cmap_pred_true,
        )

        if label in {"u", "v", "U"}:
            u = pred[CHANNEL_INDICES["u"]]
            v = pred[CHANNEL_INDICES["v"]]
            analysis.ui.components.overlay_streamlines(ax, X, Y, u, v)

        ax.set_title(f"{label} pred [{UNIT_MAP[label]}]")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(*im.get_clim()))
        analysis.ui.components.apply_axis_labels(ax, 0, Lx, Ly, is_last_row=is_last_row)

        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------
        ax = axes[r, 1]
        field = gt[k]

        im = ax.contourf(
            X,
            Y,
            field,
            levels=analysis.ui.components.compute_levels(field, n_levels),
            cmap=cmap_pred_true,
        )

        if label in {"u", "v", "U"}:
            u = gt[CHANNEL_INDICES["u"]]
            v = gt[CHANNEL_INDICES["v"]]
            analysis.ui.components.overlay_streamlines(ax, X, Y, u, v)

        ax.set_title(f"{label} true [{UNIT_MAP[label]}]")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(*im.get_clim()))
        analysis.ui.components.apply_axis_labels(ax, 1, Lx, Ly, is_last_row=is_last_row)

        # -------------------------------------------------
        # Error
        # -------------------------------------------------
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
            vmax = float(np.nanquantile(valid, clip_q))
            vmax = max(vmax, 1e-12)

            levels_err = np.linspace(0.0, vmax, n_levels)

            err_plot = np.ma.masked_greater(err_field, vmax)
            cmap_obj = plt.get_cmap(cmap_error).copy()
            cmap_obj.set_bad("white")

            im = ax.contourf(X, Y, err_plot, levels=levels_err, cmap=cmap_obj)

            ax.set_title(err_title)
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(0.0, vmax))
            analysis.ui.components.apply_axis_labels(ax, 2, Lx, Ly, is_last_row=is_last_row)

        # -------------------------------------------------
        # Kappa panels
        # -------------------------------------------------
        ax = axes[r, 3]
        if r == 0:
            im = ax.contourf(X, Y, kappa_field, levels=kappa_levels, cmap=cmap_kappa)
            ax.set_title("kappa [m²]")
            fig.colorbar(im, ax=ax, fraction=0.04)
            analysis.ui.components.apply_axis_labels(ax, 3, Lx, Ly, is_last_row=is_last_row)

        elif r == 1:
            im = ax.contourf(X, Y, kappa_log_field, levels=kappa_log_levels, cmap=cmap_kappa)
            ax.set_title("log10(kappa) [m²]")
            fig.colorbar(im, ax=ax, fraction=0.04)
            analysis.ui.components.apply_axis_labels(ax, 3, Lx, Ly, is_last_row=is_last_row)

        else:
            ax.axis("off")

    fig.suptitle(f"{dataset_name} — {case_label}", fontsize=14)
    fig.tight_layout()
    return fig


def plot_extreme_input_cases(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot an interactive viewer for cases with extreme input parameter values.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the interactive case viewer.

    """
    inp_sel = analysis.ui.components.ui_dropdown_input_parameter(parameters=_candidate_input_columns(datasets=datasets))
    err_sel = analysis.ui.components.ui_radio_error_mode()

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str, error_mode: widgets.ValueWidget) -> Figure:
        """
        Plot function for the interactive extreme input case viewer.

        Parameters
        ----------
        idx : int
            The index of the case to plot.
        df : pd.DataFrame
            The DataFrame containing the cases.
        dataset_name : str
            The name of the dataset.
        error_mode : widgets.ValueWidget
            The widget indicating the error mode.

        Returns
        -------
            Figure
                The generated matplotlib Figure.

        """
        column = inp_sel.value
        if column is None:
            msg = "No input parameter selected"
            raise RuntimeError(msg)

        extremes = _select_extreme_inputs(
            datasets={dataset_name: df},
            column=column,
        )[dataset_name]

        row = extremes.iloc[idx]

        kind = "MIN" if idx == 0 else "MAX"
        value = row.loc[column]

        # Case-Nummer sauber bestimmen
        case_id = row.get("case_index", idx)

        case_label = f"Case {case_id} | {inp_sel.value} = {value:.3g} ({kind})"

        return _plot_prediction_overview_case_4x4(
            row=row,
            dataset_name=dataset_name,
            case_label=case_label,
            error_mode=error_mode,
        )

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        enable_dataset_dropdown=True,
        extra_widgets=[inp_sel, err_sel],
        error_mode=err_sel,
        n_cases_fn=lambda *_: 2,
    )
