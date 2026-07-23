"""
===============================================================================
evaluation_plot_overview_scoreboard.py
===============================================================================
Create overview tables and model-level comparison plots.

Responsibilities:
  - Summarize global model performance across evaluation DataFrames
  - Plot accuracy/physics tradeoffs for model comparison
  - Load model metadata from current saved-run config.yaml files
  - Produce compact overview tables for notebook panels

Design principles:
  - Overview plots aggregate over cases before comparing models
  - Decision-support plots stay architecture-neutral
  - Architecture metadata comes from model.architecture and model.params
  - Detailed diagnostics remain in specialized plot modules

Boundaries:
  - Spatial error decomposition belongs to evaluation_plot_error_decomposition
  - Physics residual visualization belongs to evaluation_plot_physical_consistency
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from src import common, domain, experiments

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# =============================================================================
# Channels
# =============================================================================
CHANNELS = list(domain.fields.OUTPUT_FIELDS)
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}

# =============================================================================
# Constants (MUST match training & physics plots)
# =============================================================================
MU_AIR = 1.8139e-5
EPS_DET = 1e-4
EPS = 1e-12


# =============================================================================
# Scalar physics metric
# =============================================================================
def _combined_physics_mse(
    df: pd.DataFrame,
    *,
    include_bc: bool = False,
) -> np.ndarray:
    """
    Combine physics MSE score array.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation DataFrame containing physics metrics.
    include_bc : bool, optional
        Whether to include boundary condition metrics, by default False.

    Returns
    -------
    numpy.ndarray
        Combined physics MSE score array.

    """
    bc = pd.to_numeric(df["bc_mse"], errors="coerce").to_numpy(dtype=float) if "bc_mse" in df.columns else None

    score = pd.to_numeric(df["phys_mse"], errors="coerce").to_numpy(dtype=float)

    if include_bc and bc is not None:
        score = score + bc

    return score


def _median_physics_residual(
    df: pd.DataFrame,
    *,
    include_bc: bool = False,
) -> float:
    """
    Median combined physics residual score over all cases.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation DataFrame containing physics metrics.
    include_bc : bool, optional
        Whether to include boundary condition metrics, by default False.

    Returns
    -------
    float
        Median combined physics residual score.

    """
    score = _combined_physics_mse(df, include_bc=include_bc)
    score = score[np.isfinite(score)]

    return float(np.median(score)) if score.size else float("nan")


# =============================================================================
# Model metadata loading
# =============================================================================
def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return a config section mapping or raise a current-schema error."""
    if not isinstance(value, Mapping):
        msg = f"Expected {label} to be a mapping in config.yaml."
        raise TypeError(msg)
    return value


def _find_current_config_from_npz_path(npz_path: Path) -> Path | None:
    """Find the current run config.yaml that owns an artifact path."""
    for parent in [npz_path.parent, *npz_path.parents]:
        cfg_path = common.paths.resolve_run_config_path(parent)
        if cfg_path.is_file():
            return cfg_path
    return None


def _load_current_config_from_df(df: pd.DataFrame) -> dict[str, Any] | None:
    """Load config.yaml for the run represented by an evaluation DataFrame."""
    if df.empty or "npz_path" not in df.columns:
        return None

    cfg_path = _find_current_config_from_npz_path(Path(df.iloc[0]["npz_path"]).expanduser())
    if cfg_path is None:
        return None
    return experiments.config.loader.load_yaml(cfg_path)


def _current_config_sections(
    cfg: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return current-schema model, params, loss, data and run sections."""
    model_cfg = _as_mapping(cfg.get("model"), label='config["model"]')
    params = _as_mapping(model_cfg.get("params"), label='config["model"]["params"]')
    loss_cfg = _as_mapping(cfg.get("loss"), label='config["loss"]')
    data_cfg = _as_mapping(cfg.get("data"), label='config["data"]')
    run_cfg = _as_mapping(cfg.get("run"), label='config["run"]')
    return model_cfg, params, loss_cfg, data_cfg, run_cfg


def _display_architecture(model_cfg: Mapping[str, Any], loss_cfg: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> str | None:
    """Return a display architecture label from current config sections."""
    architecture = model_cfg.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        return None
    if architecture.startswith("PI-"):
        return architecture

    loss_type = loss_cfg.get("type")
    run_name = run_cfg.get("name")
    if loss_type == "pino" or (isinstance(run_name, str) and run_name.startswith("PI-")):
        return f"PI-{architecture}"
    return architecture


def _mode_pair_from_params(params: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return FNO or UNO mode coordinates from current model.params."""
    n_modes = params.get("n_modes")
    if isinstance(n_modes, (list, tuple)) and len(n_modes) == 2:  # noqa: PLR2004
        return float(n_modes[0]), float(n_modes[1])

    if "modes_x" in params and "modes_y" in params:
        return float(params["modes_x"]), float(params["modes_y"])

    return None


def _load_model_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """
    Load model-level architecture and loss parameters from config.yaml.

    One model = one current saved-run config.yaml, resolved via npz_path of
    the first case. Legacy read-only tolerance is limited to returning empty
    metadata when no current config.yaml is present; old config formats are not
    parsed.
    """
    cfg = _load_current_config_from_df(df)
    if cfg is None:
        return {}

    model_cfg, params, loss_cfg, _data_cfg, run_cfg = _current_config_sections(cfg)

    meta: dict[str, Any] = {}

    # --------------------------------------------------
    # Architecture
    # --------------------------------------------------
    meta["architecture"] = _display_architecture(model_cfg, loss_cfg, run_cfg)
    meta["n_layers"] = params.get("n_layers")
    meta["hidden_channels"] = params.get("hidden_channels")

    # Modes
    modes = _mode_pair_from_params(params)
    if modes is not None:
        mx, my = modes
        meta["modes_x"] = mx
        meta["modes_y"] = my
        meta["modes_mean"] = 0.5 * (mx + my)

    # UNO bottleneck
    mode_ratio = params.get("mode_ratio")
    if mode_ratio is not None:
        meta["mode_ratio"] = float(mode_ratio)
        meta["bottleneck_strength"] = 1.0 / meta["mode_ratio"]

    # --------------------------------------------------
    # Physics-informed loss weights
    # --------------------------------------------------
    meta["lambda_phys"] = loss_cfg.get("lambda_phys")
    meta["lambda_p"] = loss_cfg.get("lambda_p")

    return meta


# =============================================================================
# Styling helpers
# =============================================================================


def _fmt_optional(x: Any, fmt: str) -> str:
    """
    Format optional value (float or None) with given format string.

    Parameters
    ----------
    x : Any
        Value to format (float or None).
    fmt : str
        Format string, e.g. "{:.3e}".

    Returns
    -------
    str
        Formatted string or empty string if x is None or NaN.

    """
    if x is None or not np.isfinite(x):
        return ""
    return fmt.format(x)


def _fmt_int_if_close(x: Any) -> str:
    """Format numbers without decimals if they are (almost) integers."""
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:  # noqa: BLE001
        return str(x)

    if not np.isfinite(xf):
        return ""

    if abs(xf - round(xf)) < 1e-9:  # noqa: PLR2004
        return str(round(xf))

    # kompakt, ohne unnötige Nachkommastellen
    return f"{xf:g}"


def _style_numeric_block_blue(block: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Style numeric columns in a DataFrame block with blue colormap.

    Parameters
    ----------
    block : pandas.DataFrame
        DataFrame block to style.
    columns : list[str]
        List of column names to apply styling to.

    Returns
    -------
    pandas.DataFrame
        DataFrame of style strings.

    """
    styled = pd.DataFrame("", index=block.index, columns=block.columns)
    cmap = plt.get_cmap("Blues")

    for col in columns:
        if col not in block:
            continue

        vals = block[col].to_numpy(dtype=float)
        if not np.any(np.isfinite(vals)):
            continue

        qlo, qhi = np.nanquantile(vals, 0.05), np.nanquantile(vals, 0.95)
        vmin, vmax = qlo, qhi

        for i, v in zip(block.index, vals, strict=False):
            if not np.isfinite(v):
                continue
            t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
            t = float(np.clip(t, 0.0, 1.0))
            lo, hi = 0.05, 0.95
            r, g, b, _ = cmap(lo + (hi - lo) * t)
            alpha = 0.55
            styled.loc[i, col] = f"background-color: rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    return styled


# =============================================================================
# Global summary tables
# =============================================================================
def build_global_summary_table(
    datasets_eval: dict[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...] = (
        "rmse_p",
        "rmse_U",
        "l2",
        "rel_l2",
        "h1",
        "rel_h1",
        "phys_mse",
        "mom_mse",
        "cont_mse",
        "bc_mse",
    ),
    stats: tuple[str, ...] = ("median", "mean", "q90"),
) -> pd.DataFrame:
    """
    Build global summary table for multiple evaluation DataFrames.

    Parameters
    ----------
    datasets_eval : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.
    metrics : tuple[str, ...], optional
        Metrics to include, by default ("rel_l2", "l2", "rel_h1", "h1", "mom_mse", "cont_mse", "bc_mse").
    stats : tuple[str, ...], optional
        Statistics to include, by default ("median", "mean", "q90", "q95").
    sort_by : str, optional
        Column to sort by, by default "rel_l2_median".

    Returns
    -------
    pandas.DataFrame
        Summary DataFrame with computed statistics.

    """
    stat_fns = {
        "median": lambda a: float(np.nanmedian(a)),
        "mean": lambda a: float(np.nanmean(a)),
        "q90": lambda a: float(np.nanquantile(a, 0.90)),
    }

    rows: list[dict[str, float | str]] = []
    for name, df in datasets_eval.items():
        row: dict[str, float | str] = {"model": name}

        for m in metrics:
            if m not in df.columns:
                msg = f"Missing column '{m}' in eval df for model '{name}'"
                raise KeyError(msg)

            arr = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
            for s in stats:
                if s not in stat_fns:
                    msg = f"Unknown stat '{s}'"
                    raise KeyError(msg)
                row[f"{m}_{s}"] = stat_fns[s](arr)

        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def plot_overview_global_summary_table(
    *,
    datasets: dict[str, pd.DataFrame],
    title: str = "Global summary",
    metrics: tuple[str, ...] = (
        "rmse_p",
        "rmse_U",
        "l2",
        "rel_l2",
        "h1",
        "rel_h1",
        "phys_mse",
        "mom_mse",
        "cont_mse",
        "bc_mse",
    ),
    stats: tuple[str, ...] = ("median", "mean", "q90"),
) -> widgets.VBox:
    """
    Plot global summary table as a widget.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.
    title : str, optional
        Title for the table, by default "Global summary".
    metrics : tuple[str, ...], optional
        Metrics to include, by default ("rel_l2", "l2", "rel_h1", "h1", "mom_mse", "cont_mse", "bc_mse").
    stats : tuple[str, ...], optional
        Statistics to include, by default ("median", "mean", "q90").
    sort_by : str, optional
        Column to sort by, by default "rel_l2_median".
    number_fmt : str, optional
        Format string for numbers, by default "{:.3e}".

    """
    summary = build_global_summary_table(
        datasets_eval=datasets,
        metrics=metrics,
        stats=stats,
    )

    out = widgets.Output()
    with out:
        display(Markdown(f"## {title}"))

        cols_to_style = [c for c in summary.columns if pd.api.types.is_numeric_dtype(summary[c])]
        style_df = _style_numeric_block_blue(summary, cols_to_style)

        display(summary.style.format("{:.4g}").apply(lambda _: style_df, axis=None))

    return widgets.VBox([out])


# =============================================================================
# Overview plots
# =============================================================================
def plot_overview_scoreboard(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Overview scoreboard plot comparing multiple evaluation groups.

    Each entry in `datasets` represents one comparison item
    (e.g. model, dataset).

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.

    Returns
    -------
    matplotlib.figure.Figure
        The generated scoreboard figure.

    """
    names = list(datasets.keys())

    rmse_p = []
    rmse_U = []
    phys = []

    for df in datasets.values():
        rmse_p.append(float(np.nanmedian(pd.to_numeric(df["rmse_p"], errors="coerce"))))
        rmse_U.append(float(np.nanmedian(pd.to_numeric(df["rmse_U"], errors="coerce"))))
        phys.append(_median_physics_residual(df))

    metrics = {
        "rmse_p": np.asarray(rmse_p),
        "rmse_U": np.asarray(rmse_U),
        "physics": np.asarray(phys),
    }

    norm = {}
    for k, arr in metrics.items():
        ref = np.nanmin(arr)  # best value (lower is better)
        norm[k] = arr / (ref + EPS)

    fig_width = max(10.0, 3.5 * len(names))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(len(names))
    w = 0.25

    ax.bar(x - w, norm["rmse_p"], w, label="RMSE(p)")
    ax.bar(x, norm["rmse_U"], w, label="RMSE(U)")
    ax.bar(x + w, norm["physics"], w, label="Physics residual (MSE)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        names,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Relative score (x best, lower is better)")
    ax.set_title("Global comparison scoreboard\n(relative to best entry)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.30,
        top=0.90,
    )
    return fig


def plot_overview_pareto_error_vs_physics(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Overview Pareto plot: accuracy vs physics consistency.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Pareto figure.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in datasets.items():
        x = float(np.nanmedian(pd.to_numeric(df["rel_l2"], errors="coerce")))

        y = _median_physics_residual(df)

        ax.scatter(x, y, s=80)
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Median relative $L^2$ error")
    ax.set_ylabel("Median combined physics residual (MSE)")
    ax.set_title("Pareto: accuracy vs combined physics consistency")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.16,
        top=0.88,
    )

    return fig


def _build_single_architecture_table(
    *,
    df_arch: pd.DataFrame,
    arch_name: str,
) -> pd.io.formats.style.Styler:
    """
    Build a styled DataFrame table for a single architecture.

    Parameters
    ----------
    df_arch : pandas.DataFrame
        DataFrame containing rows for a single architecture.
    arch_name : str
        Name of the architecture.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame for display.

    """
    df_arch = df_arch.sort_values("__order", ascending=True).reset_index(drop=True)
    df_arch = df_arch.drop(columns=["__order"])

    style_df = pd.DataFrame("", index=df_arch.index, columns=df_arch.columns)

    # apply blue shading to parameters + metrics
    cols_to_style = [c for c in df_arch.columns if pd.api.types.is_numeric_dtype(df_arch[c])]

    style_df.loc[:, :] = _style_numeric_block_blue(df_arch, cols_to_style)

    fmt: dict[Any, Any] = {c: "{:.4g}" for c in df_arch.columns if pd.api.types.is_numeric_dtype(df_arch[c])}

    return df_arch.style.format(fmt).set_caption(f"Architecture: {arch_name}").apply(lambda _: style_df, axis=None)


def plot_overview_architecture_table(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot architecture overview table as a widget.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.

    Returns
    -------
    ipywidgets.VBox
        VBox containing the architecture overview table.

    """
    rows: list[dict[str, Any]] = []

    for name, df in datasets.items():
        meta = _load_model_metadata(df)
        rows.append(
            {
                "__order": len(rows),
                "model": name,
                "architecture": meta.get("architecture"),
                "modes_x": meta.get("modes_x"),
                "modes_y": meta.get("modes_y"),
                "hidden_channels": meta.get("hidden_channels"),
                "n_layers": meta.get("n_layers"),
                "mode_ratio": meta.get("mode_ratio"),
                "lambda_phys": meta.get("lambda_phys"),
                "lambda_p": meta.get("lambda_p"),
                "rmse_p": float(np.nanmedian(pd.to_numeric(df["rmse_p"], errors="coerce"))) if "rmse_p" in df.columns else np.nan,
                "rmse_U": float(np.nanmedian(pd.to_numeric(df["rmse_U"], errors="coerce"))) if "rmse_U" in df.columns else np.nan,
                "l2": float(np.nanmedian(pd.to_numeric(df["l2"], errors="coerce"))),
                "rel_l2": float(np.nanmedian(pd.to_numeric(df["rel_l2"], errors="coerce"))),
                "h1": float(np.nanmedian(pd.to_numeric(df["h1"], errors="coerce"))) if "h1" in df.columns else np.nan,
                "rel_h1": float(np.nanmedian(pd.to_numeric(df["rel_h1"], errors="coerce"))) if "rel_h1" in df.columns else np.nan,
                "phys_mse": float(np.nanmedian(pd.to_numeric(df["phys_mse"], errors="coerce"))) if "phys_mse" in df.columns else np.nan,
                "mom_mse": float(np.nanmedian(pd.to_numeric(df["mom_mse"], errors="coerce"))) if "mom_mse" in df.columns else np.nan,
                "cont_mse": float(np.nanmedian(pd.to_numeric(df["cont_mse"], errors="coerce"))),
                "bc_mse": float(np.nanmedian(pd.to_numeric(df["bc_mse"], errors="coerce"))) if "bc_mse" in df.columns else np.nan,
            }
        )

    df_all = pd.DataFrame(rows)

    arch_order = df_all["architecture"].tolist()
    arch_order = list(dict.fromkeys(arch_order))  # unique, stabil
    df_all["architecture"] = pd.Categorical(df_all["architecture"], categories=arch_order, ordered=True)

    out = widgets.Output()
    with out:
        display(Markdown("## Architecture overview"))

        for arch, df_arch in df_all.groupby("architecture", sort=False):
            display(
                _build_single_architecture_table(
                    df_arch=df_arch,
                    arch_name=str(arch),
                )
            )

    return widgets.VBox([out])
