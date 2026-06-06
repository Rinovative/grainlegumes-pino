"""
===============================================================================
evaluation_plot_physical_consistency.py
===============================================================================
Physical consistency and constraint validation for model evaluation.

Provides:
  - physics-based field constraint evaluation
  - divergence, curl and boundary condition checks
  - consistency metrics relative to domain physics
  - interactive multi-case constraint visualization

Responsibilities:
  - verify model outputs satisfy physical constraints
  - measure deviations from expected physics laws
  - support physics-informed model debugging

Design principles:
  - independent of model architecture (FNO / PINO / UNO)
  - uses canonical field names from domain.permeability mapping
  - constraints are computed on physical grid with proper normalization
  - visualization supports per-case and aggregated constraint analysis

This module does NOT:
  - perform error decomposition (use evaluation_plot_error_decomposition)
  - handle model training or physics incorporation
  - compute global error metrics
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
from matplotlib.lines import Line2D

from src import analysis

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

# =============================================================================
# CONSTANTS
# =============================================================================
EPS = 1e-12
CLIP_Q = 99.5
MU_AIR = 1.8139e-5

# training-like safeties (nur numerisch, nicht fuer "robustness")
EPS_EPS = 1e-6
EPS_K0 = 1e-30
KXY_HAT_CLIP = 0.999
DET_HAT_MIN = 1e-4

EVAL_PAD = 2  # crop for ALL gradient-based metrics (H1 + physics); fixed canonical setting


# =============================================================================
# SMALL NPZ HELPERS (format-normalisierung, ohne "robustness")
# =============================================================================
def _npz_scalar(v: Any) -> Any:
    """
    NPZ scalars can be 0-dim arrays, convert to Python scalars for easier handling.

    Parameters
    ----------
    v : Any
        Value loaded from NPZ, potentially a 0-dim array.

    Returns
    -------
    Any
        If v is a 0-dim array, returns the scalar value. Otherwise, returns v unchanged.

    """
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def _npz_str(z: Any, key: str) -> str:
    """
    Extract a string from an NPZ file, handling potential 0-dim array formats.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.
    key : str
        Key to extract from the NPZ file.

    Returns
    -------
    str
        The string value extracted from the NPZ file.

    Raises
    ------
    KeyError
        If the key is not found in the NPZ file.

    """
    return str(_npz_scalar(z[key]))


def _npz_list_str(z: Any, key: str) -> list[str]:
    """
    Extract a list of strings from an NPZ file, handling potential array formats.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.
    key : str
        Key to extract from the NPZ file.

    Returns
    -------
    list[str]
        The list of strings extracted from the NPZ file.

    Raises
    ------
    KeyError
        If the key is not found in the NPZ file.

    """
    v = z[key]
    if isinstance(v, np.ndarray):
        v = v.tolist()
    return [str(s) for s in v]


def _get_input_field(z: Any, name: str) -> np.ndarray:
    """
    Get an input field by name from the NPZ file, handling potential name variations for compatibility.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.
    name : str
        Name of the input field to retrieve.

    Returns
    -------
    np.ndarray
        The input field array.

    Raises
    ------
    KeyError
        If the input field is not found in the NPZ file.

    """
    fields = _npz_list_str(z, "input_fields")
    x_raw = np.asarray(z["x_raw"], dtype=float)  # (Cin,H,W)

    if name not in fields:
        # minimal compatibility for old artifacts
        if name == "eps" and "phi" in fields:
            name = "phi"
        elif name == "phi" and "eps" in fields:
            name = "eps"

    idx = fields.index(name)  # raises if missing (wanted)
    return x_raw[idx]


def _infer_dx_dy(z: Any) -> tuple[float, float]:
    """
    Infer uniform grid spacings in x and y directions from NPZ input fields.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary-like object.

    Returns
    -------
    tuple[float, float]
        Inferred spacing values as ``(dx, dy)``.

    Raises
    ------
    ValueError
        If either inferred spacing is non-positive.

    """
    x = _get_input_field(z, "x")
    y = _get_input_field(z, "y")

    dx = float(np.mean(np.abs(np.diff(x, axis=1))))
    dy = float(np.mean(np.abs(np.diff(y, axis=0))))

    if dx <= 0.0 or dy <= 0.0:
        msg = f"Invalid dx/dy inferred: dx={dx}, dy={dy}"
        raise ValueError(msg)
    return dx, dy


def _inlet_outlet_masks(z: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Training-consistent inlet/outlet via y-min/y-max + dy threshold.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the inlet and outlet boolean masks, respectively.

    """
    y = _get_input_field(z, "y")
    _, dy = _infer_dx_dy(z)

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    inlet = np.abs(y - y_min) <= 0.5 * dy
    outlet = np.abs(y - y_max) <= 0.5 * dy
    return inlet, outlet


def _crop2(a: np.ndarray, pad: int) -> np.ndarray:
    """
    Crop a 2D array by removing 'pad' pixels from each border.

    Parameters
    ----------
    a : np.ndarray
        Input 2D array to be cropped.
    pad : int
        Number of pixels to remove from each border. If pad <= 0, the original array is returned.

    Returns
    -------
    np.ndarray
        Cropped 2D array.

    """
    if pad <= 0:
        return a
    return a[pad:-pad, pad:-pad]


def _df0_float(df: pd.DataFrame, col: str) -> float:
    """
    Read first-row scalar from df[col] as a strict float (fail-fast).

    Fixes pyright/pylance complaining about pandas Scalar potentially being complex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column to read.
    col : str
        Name of the column to read.

    Returns
    -------
    float
        The scalar value from the first row of the specified column, converted to float.

    Raises
    ------
    KeyError
        If the specified column is not found in the DataFrame.

    """
    v: Any = df[col].iloc[0]

    # unwrap 0-d numpy arrays
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()

    # fail-fast on complex
    if np.iscomplexobj(v):
        msg = f"{col} must be real-valued, got complex: {v!r}"
        raise TypeError(msg)

    out = float(v)  # v is Any here -> typechecker ok
    if not np.isfinite(out) or out <= 0.0:
        msg = f"{col} must be finite and > 0, got: {out!r}"
        raise ValueError(msg)
    return out


# =============================================================================
# GT METRICS (computed from NPZ fields, fail-fast)
# =============================================================================
def _gt_cont_mse_divu(z: Any) -> float:
    """
    Compute GT continuity residual as MSE of velocity divergence over the interior.

    The interior crop uses the fixed canonical padding EVAL_PAD.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary-like object.

    Returns
    -------
    float
        Mean squared divergence of GT velocity.

    """
    pad = EVAL_PAD
    dx, dy = _infer_dx_dy(z)

    gt = np.asarray(z["gt"], dtype=float)  # (C,H,W)
    u = _crop2(gt[1], pad)
    v = _crop2(gt[2], pad)

    _du_dy, du_dx = np.gradient(u, dy, dx, edge_order=1)
    dv_dy, _dv_dx = np.gradient(v, dy, dx, edge_order=1)
    div_u = du_dx + dv_dy

    return float(np.mean(div_u**2))


def _gt_bc_mse(z: Any) -> float:
    """
    GT BC mismatch (training-consistent): bc_mse = MSE(p_in - p_bc_in) + (mean(p_out))^2.

    Training-consistent inlet/outlet via y-min/y-max + dy threshold.

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.

    Returns
    -------
    float
        The mean squared error of the pressure boundary condition mismatch.

    """
    gt = np.asarray(z["gt"], dtype=float)
    p = gt[0]  # (H,W)

    p_bc = np.asarray(z["p_bc"], dtype=float)
    if p_bc.ndim == 3 and p_bc.shape[0] == 1:  # noqa: PLR2004
        p_bc = p_bc[0]

    inlet, outlet = _inlet_outlet_masks(z)

    p_inlet_mse = np.mean((p[inlet] - p_bc[inlet]) ** 2) if np.any(inlet) else 0.0
    p_outlet_mse = (np.mean(p[outlet]) ** 2) if np.any(outlet) else 0.0
    return float(p_inlet_mse + p_outlet_mse)


def _gt_mom_mse(z: Any) -> float:
    """
    GT Darcy-Brinkman momentum residual MSE over interior (pad from variant). Training-consistent K^{-1} (inkl. kxy_hat, K0).

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.

    Returns
    -------
    float
        The mean squared error of the Darcy-Brinkman momentum residual over the interior of the domain.

    """
    pad = EVAL_PAD
    dx, dy = _infer_dx_dy(z)

    gt = np.asarray(z["gt"], dtype=float)
    p = _crop2(gt[0], pad)
    u = _crop2(gt[1], pad)
    v = _crop2(gt[2], pad)

    eps = _crop2(_get_input_field(z, "eps"), pad)
    eps = np.maximum(eps, EPS_EPS)

    kxx_log = _crop2(_get_input_field(z, "kxx"), pad)
    kyy_log = _crop2(_get_input_field(z, "kyy"), pad)

    # kxy_hat optional (falls nicht vorhanden -> 0)
    try:
        kxy_hat = _crop2(_get_input_field(z, "kxy"), pad)
    except ValueError:
        kxy_hat = np.zeros_like(kxx_log)

    kxy_hat = np.clip(kxy_hat, -KXY_HAT_CLIP, KXY_HAT_CLIP)

    Kxx = np.maximum(10.0**kxx_log, EPS_K0)
    Kyy = np.maximum(10.0**kyy_log, EPS_K0)
    K0 = np.sqrt(np.maximum(Kxx * Kyy, EPS_K0))

    # Derivatives
    dp_dy, dp_dx = np.gradient(p, dy, dx, edge_order=1)
    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=1)
    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=1)

    div_u = du_dx + dv_dy

    # Brinkman deviatoric stress
    coef = MU_AIR / eps
    tau_xx = coef * (2.0 * du_dx - (2.0 / 3.0) * div_u)
    tau_yy = coef * (2.0 * dv_dy - (2.0 / 3.0) * div_u)
    tau_xy = coef * (du_dy + dv_dx)

    _dtau_xx_dy, dtau_xx_dx = np.gradient(tau_xx, dy, dx, edge_order=1)
    dtau_xy_dy, dtau_xy_dx = np.gradient(tau_xy, dy, dx, edge_order=1)
    dtau_yy_dy, _dtau_yy_dx = np.gradient(tau_yy, dy, dx, edge_order=1)

    div_tau_x = dtau_xx_dx + dtau_xy_dy
    div_tau_y = dtau_xy_dx + dtau_yy_dy

    # K^{-1} construction (hat-form)
    kxx_hat = Kxx / K0
    kyy_hat = Kyy / K0
    det_hat = kxx_hat * kyy_hat - kxy_hat * kxy_hat
    det_hat_safe = np.maximum(det_hat, DET_HAT_MIN)

    invhat_xx = kyy_hat / det_hat_safe
    invhat_xy = -kxy_hat / det_hat_safe
    invhat_yy = kxx_hat / det_hat_safe

    inv_xx = invhat_xx / K0
    inv_xy = invhat_xy / K0
    inv_yy = invhat_yy / K0

    drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
    drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

    Rx = -dp_dx + div_tau_x - drag_x
    Ry = -dp_dy + div_tau_y - drag_y

    return float(np.mean(Rx**2 + Ry**2))


def _gt_metric_from_npz(npz_path: str | Path, metric: str) -> float:
    """
    Compute a selected GT physical-consistency metric from a single NPZ file.

    Parameters
    ----------
    npz_path : str | Path
        Path to the NPZ artifact.
    metric : str
        Metric key. Supported values are ``"cont_mse"``, ``"bc_mse"``,
        and ``"mom_mse"``.

    Returns
    -------
    float
        Computed GT metric value.

    Raises
    ------
    ValueError
        If ``metric`` is not supported.

    """
    with np.load(Path(npz_path), allow_pickle=True) as z:
        if metric == "cont_mse":
            return _gt_cont_mse_divu(z)
        if metric == "bc_mse":
            return _gt_bc_mse(z)
        if metric == "mom_mse":
            return _gt_mom_mse(z)
        msg = f"Unknown metric: {metric}"
        raise ValueError(msg)


# =============================================================================
# PRESSURE DROP CONSISTENCY (NPZ-based, training-consistent masks)
# =============================================================================
def _dp_rel_err_from_npz(z: Any, *, use_gt: bool) -> float:
    """
    Rel = |dp_pred - dp_bc| / (|dp_bc| + EPS): dp_* computed from inlet/outlet masks (y-based).

    Parameters
    ----------
    z : Any
        NPZ file loaded as a dictionary.
    use_gt : bool
        If True, compute dp_pred from GT fields. If False, compute dp_pred from predicted fields.

    Returns
    -------
    float
        The relative error of the pressure drop consistency.

    """
    p_bc = np.asarray(z["p_bc"], dtype=float)
    if p_bc.ndim == 3 and p_bc.shape[0] == 1:  # noqa: PLR2004
        p_bc = p_bc[0]

    arr = np.asarray(z["gt"] if use_gt else z["pred"], dtype=float)
    p = arr[0]  # (H,W)

    inlet, outlet = _inlet_outlet_masks(z)
    if not (np.any(inlet) and np.any(outlet)):
        return float("nan")

    dp_bc = float(np.mean(p_bc[inlet]) - np.mean(p_bc[outlet]))
    if abs(dp_bc) <= EPS or not np.isfinite(dp_bc):
        return float("nan")

    dp_p = float(np.mean(p[inlet]) - np.mean(p[outlet]))
    if not np.isfinite(dp_p):
        return float("nan")

    return float(abs(dp_p - dp_bc) / (abs(dp_bc) + EPS))


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def _get_scalar_vals(df: pd.DataFrame, col: str, max_cases: int) -> np.ndarray:
    """
    Get scalar values from a DataFrame column, ensuring they are finite and limited to max_cases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the evaluation results.
    col : str
        Name of the column to extract values from.
    max_cases : int
        Maximum number of cases to consider for plotting.

    Returns
    -------
    np.ndarray
        Array of scalar values from the specified column, limited to max_cases and filtered for finiteness.

    """
    # fail-fast on missing col
    s = df.reset_index(drop=True)[col].iloc[: int(max_cases)]
    arr = np.asarray(s, dtype=float)
    return arr[np.isfinite(arr)]


def _plot_cdf(
    ax: Any,
    vals: np.ndarray,
    *,
    label: str,
    color: str | None = None,
    ls: str = "-",
    zorder: int = 1,
) -> Any | None:
    """
    Plot a CDF of the given values on the specified axis.

    Parameters
    ----------
    ax : Any
        The axis on which to plot the CDF.
    vals : np.ndarray
        The values to plot.
    label : str
        The label for the CDF line.
    color : str | None, optional
        The color of the CDF line.
    ls : str, optional
        The linestyle of the CDF line.
    zorder : int, optional
        The z-order of the CDF line.

    Returns
    -------
    Any | None
        The plotted line object or None if no values are finite.

    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    x = np.maximum(np.sort(v), EPS)
    y = np.linspace(0.0, 1.0, x.size)
    (line,) = ax.plot(x, y, lw=2, label=label, color=color, linestyle=ls, zorder=zorder)
    return line


def _plot_metric_cdf_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    col: str,
    xlabel: str,
    title: str,
    gt_label: str = "GT (reference)",
) -> widgets.VBox:
    """
    Plot a CDF viewer for a specified metric column across multiple datasets, including a GT reference curve.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of dataset names to DataFrames.
    col : str
        The column name to plot.
    xlabel : str
        The x-axis label.
    title : str
        The title of the plot.
    gt_label : str, optional
        The label for the GT reference curve.

    Returns
    -------
    widgets.VBox
        The VBox widget containing the CDF viewer.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown(f"## {title}"))
            print("No datasets.")
        return widgets.VBox([out])

    df_ref = datasets[names[0]].reset_index(drop=True)
    gt_cache: dict[str, Any] = {"loaded_until": 0, "vals": []}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Inner plotting function that generates the CDF plot for the specified metric column.

        Parameters
        ----------
        max_cases : int
            The maximum number of cases to include in the plot.
        datasets : dict[str, pd.DataFrame]
            The datasets to plot.

        Returns
        -------
        Figure
            The generated figure object.

        """
        max_cases = int(max_cases)

        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles: list[Any] = []
        labels: list[str] = []

        # model curves
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col=col, max_cases=max_cases)
            line = _plot_cdf(ax, vals, label=name)
            if line is not None:
                handles.append(line)
                labels.append(name)

        # GT curve (computed from df_ref NPZ)
        loaded = int(gt_cache["loaded_until"])
        if max_cases > loaded:
            vals_list: list[float] = list(gt_cache["vals"])
            nmax = min(max_cases, len(df_ref))
            for i in range(loaded, nmax):
                npz_path = str(df_ref.loc[i, "npz_path"])
                vals_list.append(float(_gt_metric_from_npz(npz_path, metric=col)))
            gt_cache["vals"] = vals_list
            gt_cache["loaded_until"] = max_cases

        gt_vals = np.asarray(gt_cache["vals"], dtype=float)
        gt_line = _plot_cdf(ax, gt_vals, label=gt_label, color="black", ls="-", zorder=10)
        if gt_line is not None:
            handles.append(gt_line)
            labels.append(gt_label)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        if handles:
            ax_leg.legend(handles, labels, loc="upper left")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.06, right=0.98)
        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


def _plot_npz_scalar_cdf_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    compute_scalar: Callable[[Any], float],
    gt_compute_scalar: Callable[[Any], float],
    xlabel: str,
    title: str,
    gt_label: str = "GT (reference)",
) -> widgets.VBox:
    """
    Plot a CDF viewer for a scalar metric computed from NPZ files across multiple datasets, including a GT reference curve.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of dataset names to DataFrames.
    compute_scalar : Callable[[Any], float]
        Function to compute the scalar metric from an NPZ file for the model predictions.
    gt_compute_scalar : Callable[[Any], float]
        Function to compute the scalar metric from an NPZ file for the GT reference.
    xlabel : str
        The x-axis label.
    title : str
        The title of the plot.
    gt_label : str, optional
        The label for the GT reference curve.

    Returns
    -------
    widgets.VBox
        The VBox widget containing the CDF viewer.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown(f"## {title}"))
            print("No datasets.")
        return widgets.VBox([out])

    df_ref = datasets[names[0]].reset_index(drop=True)
    cache: dict[str, dict[str, Any]] = {name: {"loaded_until": 0, "vals": []} for name in names}
    gt_cache: dict[str, Any] = {"loaded_until": 0, "vals": []}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        max_cases = int(max_cases)

        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles: list[Any] = []
        labels: list[str] = []

        # model curves
        for name in names:
            df = datasets[name].reset_index(drop=True)
            entry = cache[name]
            loaded = int(entry["loaded_until"])

            if max_cases > loaded:
                vals_list: list[float] = list(entry["vals"])
                nmax = min(max_cases, len(df))
                for i in range(loaded, nmax):
                    npz_path = str(df.loc[i, "npz_path"])
                    with np.load(Path(npz_path), allow_pickle=True) as z:
                        vals_list.append(float(compute_scalar(z)))
                entry["vals"] = vals_list
                entry["loaded_until"] = max_cases

            vals = np.asarray(entry["vals"], dtype=float)
            line = _plot_cdf(ax, vals, label=name)
            if line is not None:
                handles.append(line)
                labels.append(name)

        # GT curve (computed from df_ref NPZ)
        loaded = int(gt_cache["loaded_until"])
        if max_cases > loaded:
            vals_list = list(gt_cache["vals"])
            nmax = min(max_cases, len(df_ref))
            for i in range(loaded, nmax):
                npz_path = str(df_ref.loc[i, "npz_path"])
                with np.load(Path(npz_path), allow_pickle=True) as z:
                    vals_list.append(float(gt_compute_scalar(z)))
            gt_cache["vals"] = vals_list
            gt_cache["loaded_until"] = max_cases

        gt_vals = np.asarray(gt_cache["vals"], dtype=float)
        gt_line = _plot_cdf(ax, gt_vals, label=gt_label, color="black", ls="-", zorder=10)
        if gt_line is not None:
            handles.append(gt_line)
            labels.append(gt_label)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        if handles:
            ax_leg.legend(handles, labels, loc="upper left")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.06, right=0.98)
        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


def _plot_mean_field_map_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    reducer: Callable[[Any], np.ndarray],
    cbar_label: str,
    title: str,
) -> widgets.VBox:
    """
    Build an interactive viewer for mean 2D scalar field maps across datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.
    reducer : Callable[[Any], np.ndarray]
        Function that extracts a 2D scalar field from an NPZ payload.
    cbar_label : str
        Label for the colorbar.
    title : str
        Figure title.

    Returns
    -------
    widgets.VBox
        Viewer widget with case-count controls and rendered maps.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown(f"## {title}"))
            print("No datasets.")
        return widgets.VBox([out])

    cache: dict[str, dict[str, Any]] = {name: {"loaded_until": 0, "sum": None, "count": 0} for name in names}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Render a CDF comparison figure for NPZ-derived scalar metrics.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to include.
        datasets : dict[str, pd.DataFrame]
            Mapping of model names to evaluation DataFrames.

        Returns
        -------
        Figure
            Generated matplotlib figure containing model and GT curves.

        """
        max_cases = int(max_cases)

        fig = plt.figure(figsize=(6 * len(names), 5.8))
        gs = fig.add_gridspec(1, len(names), wspace=0.25)

        for c, name in enumerate(names):
            df = datasets[name].reset_index(drop=True)
            Lx = _df0_float(df, "geometry_Lx")
            Ly = _df0_float(df, "geometry_Ly")

            entry = cache[name]
            loaded = int(entry["loaded_until"])

            if max_cases > loaded:
                nmax = min(max_cases, len(df))
                for i in range(loaded, nmax):
                    npz_path = str(df.loc[i, "npz_path"])
                    with np.load(Path(npz_path), allow_pickle=True) as z:
                        field = np.asarray(reducer(z), dtype=float)
                    if entry["sum"] is None:
                        entry["sum"] = np.zeros_like(field, dtype=float)
                    entry["sum"] += field
                    entry["count"] += 1
                entry["loaded_until"] = max_cases

            ax = fig.add_subplot(gs[0, c])
            ax.set_title(name)

            if entry["count"] == 0:
                ax.axis("off")
                continue

            mean_field = entry["sum"] / float(entry["count"])
            vmax = float(np.nanpercentile(mean_field, CLIP_Q))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = float(np.nanmax(mean_field))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0

            mean_plot = np.clip(mean_field, 0.0, vmax)

            im = ax.imshow(
                mean_plot,
                origin="lower",
                extent=(0.0, Lx, 0.0, Ly),
                aspect="equal",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )

            # axes labels (only left y)
            ax.set_xlabel("x [m]")
            ax.set_xticks([0.0, Lx / 2.0, Lx])
            if c == 0:
                ax.set_ylabel("y [m]")
                ax.set_yticks([0.0, Ly / 2.0, Ly])
            else:
                ax.set_yticks([])

            cb = fig.colorbar(im, ax=ax, fraction=0.025)
            cb.ax.yaxis.set_major_formatter(analysis.ui.components.choose_colorbar_formatter(0.0, vmax))
            cb.set_label(cbar_label)

        fig.suptitle(title, y=0.85)
        fig.subplots_adjust(top=0.95, bottom=0.12, left=0.06, right=0.98, wspace=0.25)
        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def _style_numeric_block_blue(block: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Style numeric values in the specified columns of a DataFrame block with a blue background gradient.

    Parameters
    ----------
    block : pd.DataFrame
        The DataFrame block to style.
    columns : list[str]
        List of column names to apply the styling to.

    Returns
    -------
    pd.DataFrame
        A DataFrame of the same shape as the input block, where the specified columns have a blue background gradient based on their numeric values.

    """
    styled = pd.DataFrame("", index=block.index, columns=block.columns)
    cmap = plt.get_cmap("Blues")

    for col in columns:
        if col not in block:
            continue

        vals_all = pd.to_numeric(block[col], errors="coerce").to_numpy(dtype=float)
        vals = vals_all[np.isfinite(vals_all)]
        if vals.size == 0:
            continue

        qlo, qhi = np.nanquantile(vals, 0.05), np.nanquantile(vals, 0.95)
        vmin, vmax = float(qlo), float(qhi)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax == vmin:
            continue

        for i in block.index:
            v = pd.to_numeric(block.loc[i, col], errors="coerce")
            if not np.isfinite(v):
                continue
            t = float(np.clip((float(v) - vmin) / (vmax - vmin), 0.0, 1.0))
            lo, hi = 0.05, 0.95
            r, g, b, _ = cmap(lo + (hi - lo) * t)
            alpha = 0.55
            styled.loc[i, col] = f"background-color: rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    return styled


def build_physical_consistency_summary_table(
    datasets_eval: dict[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...] = ("phys_mse", "cont_mse", "mom_mse", "bc_mse"),
    stats: tuple[str, ...] = ("median", "mean", "q90", "q95"),
    sort_by: str = "phys_mse_median",
) -> pd.DataFrame:
    """
    Build a summary table of aggregate physical-consistency metrics by model.

    Parameters
    ----------
    datasets_eval : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.
    metrics : tuple[str, ...], optional
        Metric columns to summarize.
    stats : tuple[str, ...], optional
        Aggregate statistics to compute for each metric.
    sort_by : str, optional
        Output column used for ascending sort when present.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by model name.

    """
    stat_fns: dict[str, Callable[[np.ndarray], float]] = {
        "median": lambda a: float(np.nanmedian(a)),
        "mean": lambda a: float(np.nanmean(a)),
        "q90": lambda a: float(np.nanquantile(a, 0.90)),
        "q95": lambda a: float(np.nanquantile(a, 0.95)),
    }

    rows: list[dict[str, float | str]] = []
    for name, df in datasets_eval.items():
        row: dict[str, float | str] = {"model": name}
        for m in metrics:
            arr = np.asarray(df[m], dtype=float)  # fail-fast
            for s in stats:
                row[f"{m}_{s}"] = stat_fns[s](arr)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("model")
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=True)
    return out


def plot_physical_consistency_summary_table(
    *,
    datasets: dict[str, pd.DataFrame],
    title: str = "Physical consistency summary",
    metrics: tuple[str, ...] = ("phys_mse", "cont_mse", "mom_mse", "bc_mse"),
    stats: tuple[str, ...] = ("median", "mean", "q90", "q95"),
    sort_by: str = "phys_mse_median",
) -> widgets.VBox:
    """
    Render a styled table view of physical-consistency summary statistics.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.
    title : str, optional
        Markdown title shown above the table.
    metrics : tuple[str, ...], optional
        Metric columns included in the summary.
    stats : tuple[str, ...], optional
        Aggregate statistics computed per metric.
    sort_by : str, optional
        Column name used to sort the summary when available.

    Returns
    -------
    widgets.VBox
        Widget containing the rendered title and styled summary table.

    """
    summary = build_physical_consistency_summary_table(
        datasets_eval=datasets,
        metrics=metrics,
        stats=stats,
        sort_by=sort_by,
    )

    out = widgets.Output()
    with out:
        display(Markdown(f"## {title}"))

        cols_to_style = [c for c in summary.columns if pd.api.types.is_numeric_dtype(summary[c])]
        style_df = _style_numeric_block_blue(summary, cols_to_style)

        display(summary.style.format("{:.4g}").apply(lambda _: style_df, axis=None))

    return widgets.VBox([out])


# =============================================================================
# PUBLIC PLOTS (thin wrappers)
# =============================================================================
def plot_velocity_divergence(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDFs of continuity residual metric across models with GT reference.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive CDF viewer widget.

    """
    return _plot_metric_cdf_viewer(
        datasets=datasets,
        col="cont_mse",
        xlabel=r"$\mathrm{MSE}(\nabla \cdot \mathbf{u})$",
        title="Mass conservation residual distribution",
    )


def plot_brinkman_residual(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDFs of Darcy-Brinkman momentum residual metric with GT reference.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive CDF viewer widget.

    """
    return _plot_metric_cdf_viewer(
        datasets=datasets,
        col="mom_mse",
        xlabel=r"$\mathrm{MSE}(\mathbf{r}_{\mathrm{mom}})$",
        title="Darcy-Brinkman momentum residual distribution",
    )


def plot_pressure_bc_consistency(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDFs of pressure boundary-condition mismatch metric with GT reference.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive CDF viewer widget.

    """
    return _plot_metric_cdf_viewer(
        datasets=datasets,
        col="bc_mse",
        xlabel=r"$\mathrm{MSE}(p_{\Gamma} - p_{\mathrm{bc}})$",
        title="Pressure boundary consistency",
    )


def plot_pressure_drop_consistency(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDFs of relative pressure-drop mismatch computed from NPZ fields.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive CDF viewer widget.

    """
    return _plot_npz_scalar_cdf_viewer(
        datasets=datasets,
        compute_scalar=lambda z: _dp_rel_err_from_npz(z, use_gt=False),
        gt_compute_scalar=lambda z: _dp_rel_err_from_npz(z, use_gt=True),
        gt_label="GT (reference)",
        xlabel=r"$\frac{|\Delta p_{\mathrm{model}}-\Delta p_{\mathrm{bc}}|}{|\Delta p_{\mathrm{bc}}|+\epsilon}$",
        title="Pressure drop consistency (relative mismatch)",
    )


def plot_mass_conservation_error_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot mean absolute mass-conservation error maps for all datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive map viewer widget.

    """
    return _plot_mean_field_map_viewer(
        datasets=datasets,
        reducer=lambda z: np.abs(np.asarray(z["div_u"], dtype=float)),
        cbar_label=r"mean $|\nabla \cdot \mathbf{u}|$",
        title="Mean absolute mass conservation error map",
    )


def plot_div_eps_u_error_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot mean absolute porosity-weighted continuity residual maps.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive map viewer widget.

    """
    return _plot_mean_field_map_viewer(
        datasets=datasets,
        reducer=lambda z: np.abs(np.asarray(z["div_eps_u"], dtype=float)),
        cbar_label=r"mean $|\nabla \cdot (\varepsilon \mathbf{u})|$",
        title="Mean absolute porosity-weighted continuity residual map",
    )


def plot_brinkman_momentum_residual_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot mean Darcy-Brinkman momentum residual magnitude maps.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive map viewer widget.

    """
    return _plot_mean_field_map_viewer(
        datasets=datasets,
        reducer=lambda z: np.sqrt(np.asarray(z["Rx"], dtype=float) ** 2 + np.asarray(z["Ry"], dtype=float) ** 2),
        cbar_label=r"mean $\left\|\mathbf{r}_{\mathrm{mom}}\right\|$",
        title="Mean Darcy-Brinkman momentum residual map",
    )


def plot_physical_consistency_cdf_grid(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot a 2x2 grid of physical-consistency CDFs with a shared external legend.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of model names to evaluation DataFrames.

    Returns
    -------
    widgets.VBox
        Interactive case-count viewer for the CDF grid.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown("## Physical consistency (CDF grid)"))
            print("No datasets.")
        return widgets.VBox([out])

    df_ref = datasets[names[0]].reset_index(drop=True)

    dp_cache: dict[str, dict[str, Any]] = {name: {"loaded_until": 0, "vals": []} for name in names}
    gt_cache: dict[str, Any] = {"loaded_until": 0, "cont": [], "mom": [], "bc": [], "dp": []}

    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = cycle.by_key().get("color", []) if cycle is not None else []
    if not colors:
        colors = [f"C{i}" for i in range(10)]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(names)}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        max_cases = int(max_cases)

        # update dp per-model caches
        for name in names:
            df = datasets[name].reset_index(drop=True)
            entry = dp_cache[name]
            loaded = int(entry["loaded_until"])
            if max_cases > loaded:
                vals_list = list(entry["vals"])
                nmax = min(max_cases, len(df))
                for i in range(loaded, nmax):
                    npz_path = str(df.loc[i, "npz_path"])
                    with np.load(Path(npz_path), allow_pickle=True) as z:
                        vals_list.append(float(_dp_rel_err_from_npz(z, use_gt=False)))
                entry["vals"] = vals_list
                entry["loaded_until"] = max_cases

        # update GT cache
        loaded = int(gt_cache["loaded_until"])
        if max_cases > loaded:
            cont_list = list(gt_cache["cont"])
            mom_list = list(gt_cache["mom"])
            bc_list = list(gt_cache["bc"])
            dp_list = list(gt_cache["dp"])

            nmax = min(max_cases, len(df_ref))
            for i in range(loaded, nmax):
                npz_path = str(df_ref.loc[i, "npz_path"])
                with np.load(Path(npz_path), allow_pickle=True) as z:
                    cont_list.append(float(_gt_cont_mse_divu(z)))
                    mom_list.append(float(_gt_mom_mse(z)))
                    bc_list.append(float(_gt_bc_mse(z)))
                    dp_list.append(float(_dp_rel_err_from_npz(z, use_gt=True)))

            gt_cache["cont"] = cont_list
            gt_cache["mom"] = mom_list
            gt_cache["bc"] = bc_list
            gt_cache["dp"] = dp_list
            gt_cache["loaded_until"] = max_cases

        # build figure
        fig = plt.figure(figsize=(15.5, 8.5))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.30], wspace=0.25, hspace=0.30)

        ax_mass = fig.add_subplot(gs[0, 0])
        ax_mom = fig.add_subplot(gs[0, 1])
        ax_dp = fig.add_subplot(gs[1, 0])
        ax_bc = fig.add_subplot(gs[1, 1])
        ax_leg = fig.add_subplot(gs[:, 2])
        ax_leg.axis("off")

        def _format_ax(ax: Any, *, xlabel: str, title: str) -> None:
            """
            Apply shared axis formatting for CDF subplots.

            Parameters
            ----------
            ax : Any
                Matplotlib axis to format.
            xlabel : str
                Label for the x-axis.
            title : str
                Subplot title.

            Returns
            -------
            None
                This function updates the axis in place.

            """
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("CDF")
            ax.set_title(title)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # mass
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col="cont_mse", max_cases=max_cases)
            _plot_cdf(ax_mass, vals, label=name, color=color_map[name])
        _plot_cdf(ax_mass, np.asarray(gt_cache["cont"], dtype=float), label="GT (reference)", color="black", zorder=10)
        _format_ax(ax_mass, xlabel=r"$\mathrm{MSE}(\nabla \cdot \mathbf{u})$", title="Mass conservation")

        # mom
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col="mom_mse", max_cases=max_cases)
            _plot_cdf(ax_mom, vals, label=name, color=color_map[name])
        _plot_cdf(ax_mom, np.asarray(gt_cache["mom"], dtype=float), label="GT (reference)", color="black", zorder=10)
        _format_ax(ax_mom, xlabel=r"$\mathrm{MSE}(\mathbf{r}_{\mathrm{mom}})$", title="Brinkman residual")

        # dp
        for name in names:
            vals = np.asarray(dp_cache[name]["vals"], dtype=float)
            _plot_cdf(ax_dp, vals, label=name, color=color_map[name])
        _plot_cdf(ax_dp, np.asarray(gt_cache["dp"], dtype=float), label="GT (reference)", color="black", zorder=10)
        _format_ax(
            ax_dp, xlabel=r"$\frac{|\Delta p_{\mathrm{model}}-\Delta p_{\mathrm{bc}}|}{|\Delta p_{\mathrm{bc}}|+\epsilon}$", title="Pressure drop"
        )

        # bc
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col="bc_mse", max_cases=max_cases)
            _plot_cdf(ax_bc, vals, label=name, color=color_map[name])
        _plot_cdf(ax_bc, np.asarray(gt_cache["bc"], dtype=float), label="GT (reference)", color="black", zorder=10)
        _format_ax(ax_bc, xlabel=r"$\mathrm{MSE}(p_{\Gamma} - p_{\mathrm{bc}})$", title="Pressure BC")

        # legend
        legend_handles = [Line2D([0], [0], color=color_map[name], lw=2, label=name) for name in names]
        legend_handles.append(Line2D([0], [0], color="black", lw=2, label="GT (reference)"))
        legend_labels = [str(h.get_label()) for h in legend_handles]
        ax_leg.legend(legend_handles, legend_labels, loc="upper left", frameon=True)

        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.98)
        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )
