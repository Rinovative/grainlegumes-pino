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
  - Architecture metadata comes from model.kind and model.params
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
CHANNELS = list(domain.fields.ANALYSIS_FIELDS)
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}

# =============================================================================
# Constants (MUST match training & physics plots)
# =============================================================================
MU_AIR = 1.8139e-5
DETERMINANT_FLOOR = 1e-4
NORMALIZATION_DENOMINATOR_FLOOR = 1e-12


# =============================================================================
# Scalar physics metric
# =============================================================================
def _median_metric(df: pd.DataFrame, column: str) -> float:
    """Return the finite median of one explicit, unit-consistent metric."""
    if column not in df:
        msg = f"Evaluation DataFrame is missing required metric {column!r}."
        raise KeyError(msg)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values < 0.0):
        msg = f"Evaluation metric {column!r} must contain finite non-negative values."
        raise ValueError(msg)
    return float(np.median(values))


# =============================================================================
# Model metadata loading
# =============================================================================
def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return a required config section mapping."""
    if not isinstance(value, Mapping):
        msg = f"Expected {label} to be a mapping in config.yaml."
        raise TypeError(msg)
    return value


def _find_current_config_from_npz_path(npz_path: Path) -> Path | None:
    """Find the run config.yaml that owns an artifact path."""
    for parent in [npz_path.parent, *npz_path.parents]:
        cfg_path = common.paths.resolve_run_config_path(parent)
        if cfg_path.is_file():
            return cfg_path
    return None


def _load_current_config_from_df(df: pd.DataFrame) -> dict[str, Any] | None:
    """Load the required config.yaml for a nonempty artifact DataFrame."""
    if df.empty:
        return None
    if "npz_path" not in df.columns:
        message = "Nonempty artifact data must contain an npz_path column."
        raise KeyError(message)

    npz_path = Path(df.iloc[0]["npz_path"]).expanduser()
    cfg_path = _find_current_config_from_npz_path(npz_path)
    if cfg_path is None:
        message = f"Artifact path is not owned by a run config.yaml: {npz_path}"
        raise FileNotFoundError(message)
    return experiments.config.loader.load_yaml(cfg_path)


def _current_config_sections(
    cfg: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return the model, params, loss, data, and run sections."""
    model_cfg = _as_mapping(cfg.get("model"), label='config["model"]')
    params = _as_mapping(model_cfg.get("params"), label='config["model"]["params"]')
    loss_cfg = _as_mapping(cfg.get("loss"), label='config["loss"]')
    data_cfg = _as_mapping(cfg.get("data"), label='config["data"]')
    run_cfg = _as_mapping(cfg.get("run"), label='config["run"]')
    return model_cfg, params, loss_cfg, data_cfg, run_cfg


def _display_architecture(model_cfg: Mapping[str, Any], loss_cfg: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> str | None:  # noqa: ARG001
    """Return a descriptive label derived from semantic model/loss config."""
    model_kind = model_cfg.get("kind")
    if not isinstance(model_kind, str) or not model_kind:
        return None
    physics_cfg = loss_cfg.get("physics")
    physics_enabled = isinstance(physics_cfg, Mapping) and physics_cfg.get("enabled") is True
    display_kind = model_kind.upper()
    return f"PI-{display_kind}" if physics_enabled else display_kind


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

    One model owns one current saved-run config.yaml, resolved from the first
    case npz_path. Only an empty DataFrame may omit model metadata.
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
    physics_cfg = loss_cfg.get("physics")
    physics_cfg = physics_cfg if isinstance(physics_cfg, Mapping) else {}
    residual_weight = physics_cfg.get("residual_weight")
    boundary_weight = physics_cfg.get("boundary_weight")
    meta["lambda_phys"] = residual_weight.get("target") if isinstance(residual_weight, Mapping) else None
    meta["lambda_p"] = boundary_weight.get("target") if isinstance(boundary_weight, Mapping) else None

    return meta


# =============================================================================
# Styling helpers
# =============================================================================


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
        "rmse_u",
        "rmse_v",
        "rmse_U",
        "rel_l2",
        "rel_h1",
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
        Per-field RMSE, dimensionless relative errors, and separate physics diagnostics to include.
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
        "rmse_u",
        "rmse_v",
        "rmse_U",
        "rel_l2",
        "rel_h1",
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
        Per-field RMSE, dimensionless relative errors, and separate physics diagnostics to include.
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
    """Compare separate dimensionless error and physics residual scores."""
    names = list(datasets)
    metric_specs = (
        ("rel_l2", "Relative L2"),
        ("rel_h1", "Relative H1"),
        ("mom_mse", "Momentum residual MSE"),
        ("cont_mse", "Continuity residual MSE"),
    )
    metrics = {column: np.asarray([_median_metric(frame, column) for frame in datasets.values()]) for column, _label in metric_specs}
    normalized: dict[str, np.ndarray] = {}
    for column, values in metrics.items():
        reference = np.nanmin(values)
        normalized[column] = values / (reference + NORMALIZATION_DENOMINATOR_FLOOR)

    figure_width = max(10.0, 3.5 * len(names))
    figure, axis = plt.subplots(figsize=(figure_width, 5.5))
    positions = np.arange(len(names))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metric_specs))
    for offset, (column, label) in zip(offsets, metric_specs, strict=True):
        axis.bar(positions + offset, normalized[column], width, label=label)

    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=25, ha="right")
    axis.set_ylabel("Ratio to best value per metric (lower is better)")
    axis.set_title("Global comparison scoreboard\n(each metric normalized independently)")
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    axis.legend()
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.90)
    return figure


def plot_overview_pareto_error_vs_physics(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """Plot accuracy against momentum and continuity residuals on separate axes."""
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    physics_metrics = (
        ("mom_mse", "Momentum residual MSE"),
        ("cont_mse", "Continuity residual MSE"),
    )
    for axis, (metric, label) in zip(axes, physics_metrics, strict=True):
        for name, frame in datasets.items():
            relative_l2 = _median_metric(frame, "rel_l2")
            residual = _median_metric(frame, metric)
            axis.scatter(relative_l2, residual, s=80)
            axis.annotate(
                name,
                xy=(relative_l2, residual),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=10,
            )
        axis.set_yscale("log")
        axis.set_xlabel(r"Median relative $L^2$ error")
        axis.set_ylabel(f"Median {label}")
        axis.set_title(f"Accuracy vs {label.lower()}")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88, wspace=0.28)
    return figure


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
                "rmse_p": float(np.nanmedian(pd.to_numeric(df["rmse_p"], errors="coerce"))),
                "rmse_u": float(np.nanmedian(pd.to_numeric(df["rmse_u"], errors="coerce"))),
                "rmse_v": float(np.nanmedian(pd.to_numeric(df["rmse_v"], errors="coerce"))),
                "rmse_U": float(np.nanmedian(pd.to_numeric(df["rmse_U"], errors="coerce"))),
                "rel_l2": float(np.nanmedian(pd.to_numeric(df["rel_l2"], errors="coerce"))),
                "rel_h1": float(np.nanmedian(pd.to_numeric(df["rel_h1"], errors="coerce"))),
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
