"""
===============================================================================
evaluation_plot_run_summary.py
===============================================================================
Summarize authoritative model accuracy and separate physical tradeoffs.

Responsibilities:
  - Build a provenance-backed table of run, dataset, objective, and architecture facts
  - Read ``normalized_macro_rmse`` only from validated aggregate SSE/count evidence
  - Plot accuracy against each explicitly named physical diagnostic
  - Disclose ID/OOD role, selected case count, and exact trainable parameter count

Design principles:
  - Per-case quantiles remain secondary and never reconstruct the primary objective
  - No normalized-to-best, parameter-efficiency, or composite ranking is invented
  - Incompatible physical diagnostics retain distinct axes, names, and units

This module does NOT:
  - Admit incompatible frames or reconstruct aggregate accuracy from case RMSEs
  - Serialize media for W&B or invent composite model rankings
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_PHYSICS_LABELS = {
    "momentum_residual_mse": (r"momentum mean($R_x^2+R_y^2$)", "(Pa/m)^2"),
    "div_velocity_mse": (r"mean($div(u)^2$)", "s^-2"),
    "div_eps_velocity_mse": (r"mean($div(eps u)^2$)", "s^-2"),
    "pressure_boundary_mse": ("pressure boundary diagnostic", "Pa^2"),
}
_ROLE_MARKERS = {"ID": "o", "OOD": "s", "unspecified": "D"}


def _provenance(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Return validated comparison provenance."""
    return dataframe.require_complete_provenance(frame)


def _finite_quantile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    """Return one finite per-case metric quantile."""
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        msg = f"Metric {column!r} must contain finite values."
        raise ValueError(msg)
    return float(np.quantile(values, quantile))


def _model_metadata(frame: pd.DataFrame) -> tuple[str, int, bool, str]:
    """
    Read model family, exact trainable count, PI flag, and continuity annotation.

    Values come only from complete provenance. Missing/non-typed parameter counts,
    physics enablement, or continuity text fail rather than being inferred from
    labels, configuration paths, or architecture names.
    """
    provenance = _provenance(frame)
    model = provenance.get("model")
    if not isinstance(model, Mapping):
        msg = "Artifact provenance model must be a mapping."
        raise TypeError(msg)
    counts = model.get("parameter_counts")
    if not isinstance(counts, Mapping):
        msg = "Artifact provenance model.parameter_counts must be a mapping."
        raise TypeError(msg)
    family = model.get("kind")
    trainable = counts.get("trainable")
    if not isinstance(family, str) or not family or isinstance(trainable, bool) or not isinstance(trainable, int):
        msg = "Run summary requires model kind and exact trainable parameter count."
        raise TypeError(msg)
    physics_enabled = model.get("physics_enabled")
    if type(physics_enabled) is not bool:
        msg = "Run summary requires a boolean model.physics_enabled provenance value."
        raise TypeError(msg)
    physics = provenance.get("physics")
    continuity = physics.get("selected_training_continuity") if isinstance(physics, Mapping) else "not_applicable"
    if not isinstance(continuity, str):
        msg = "Selected training continuity provenance must be a string."
        raise TypeError(msg)
    return family, trainable, physics_enabled, continuity


def build_run_summary_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build one combined run summary from authoritative artifact evidence.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Labelled current artifacts. Runs inside each ID/OOD role must pass the
        shared comparison contract.

    Returns
    -------
    pandas.DataFrame
        One row per labelled artifact role. ``normalized_macro_rmse`` and global
        field RMSE use exact SSE/count evidence; physical field errors and physics
        diagnostics remain separate unit-labelled quantiles.

    Raises
    ------
    ComparisonCompatibilityError, KeyError, TypeError, ValueError
        If provenance, comparison identity, model facts, aggregate evidence, or
        finite per-case metrics are incomplete.

    Notes
    -----
    The table includes run/config/checkpoint identity, exact trainable parameters,
    physics enablement, and selected training continuity without deriving a score.

    """
    dataframe.validate_comparison(datasets)
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        provenance = _provenance(frame)
        run = provenance["run"]
        dataset = provenance["dataset"]
        aggregate = frame.attrs["normalized_macro_rmse"]
        fields = tuple(frame.attrs["output_fields"])
        units = dataframe.field_units(frame)
        family, trainable, physics_enabled, continuity = _model_metadata(frame)
        row: dict[str, Any] = {
            "label": label,
            "dataset_role": dataframe.dataset_role(frame),
            "task_id": frame.attrs["task_id"],
            "run_name": run["name"],
            "architecture_family": family,
            "trainable_parameters": trainable,
            "physics_enabled": physics_enabled,
            "selected_training_continuity": continuity,
            "normalized_macro_rmse": float(aggregate["value"]),
            "relative_l2_median": _finite_quantile(frame, "rel_l2", 0.5),
            "relative_l2_q90": _finite_quantile(frame, "rel_l2", 0.9),
            "relative_h1_median": _finite_quantile(frame, "rel_h1", 0.5),
            "relative_h1_q90": _finite_quantile(frame, "rel_h1", 0.9),
            "dataset_name": dataset["name"],
            "sample_count": len(frame),
            "config_digest": str(run["effective_config_digest"]),
            "checkpoint_digest": str(run["best_checkpoint_sha256"]),
        }
        field_statistics = aggregate["field_statistics"]
        for field in fields:
            row[f"normalized_rmse_{field}"] = float(field_statistics[field]["normalized_rmse"])
            row[f"physical_rmse_{field}_median [{units[field]}]"] = _finite_quantile(frame, f"rmse_{field}", 0.5)
            row[f"physical_rmse_{field}_q90 [{units[field]}]"] = _finite_quantile(frame, f"rmse_{field}", 0.9)
        if "rmse_U" in frame.columns:
            row["physical_rmse_U_median [m/s]"] = _finite_quantile(frame, "rmse_U", 0.5)
            row["physical_rmse_U_q90 [m/s]"] = _finite_quantile(frame, "rmse_U", 0.9)
        for metric in dataframe.STEADY_PHYSICS_METRICS:
            if metric in frame.columns:
                unit = _PHYSICS_LABELS[metric][1]
                row[f"{metric}_median [{unit}]"] = _finite_quantile(frame, metric, 0.5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def plot_accuracy_physics_pareto(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot authoritative accuracy against four separate physics diagnostics.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible steady-flow frames with exact aggregate and residual evidence.

    Returns
    -------
    matplotlib.figure.Figure
        Four linear-axis panels pairing ``normalized_macro_rmse`` with the median
        momentum, dual-continuity, or pressure-boundary diagnostic.

    Raises
    ------
    ComparisonCompatibilityError, KeyError, TypeError, ValueError
        If physics provenance, model metadata, aggregates, or finite non-negative
        metrics are invalid.

    Notes
    -----
    Axes remain separate because diagnostic units/meanings differ. Filled markers
    disclose physics-enabled models; marker shapes disclose ID/OOD role. No
    composite Pareto rank or parameter-efficiency score is calculated.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for axis, metric in zip(axes.flat, dataframe.STEADY_PHYSICS_METRICS, strict=True):
        metric_label, unit = _PHYSICS_LABELS[metric]
        for index, (label, frame) in enumerate(datasets.items()):
            primary = float(frame.attrs["normalized_macro_rmse"]["value"])
            physics_value = _finite_quantile(frame, metric, 0.5)
            if primary < 0.0 or physics_value < 0.0:
                msg = f"Pareto metrics must be non-negative; {label!r} supplied {primary}, {physics_value}."
                raise ValueError(msg)
            family, _count, physics_enabled, continuity = _model_metadata(frame)
            role = dataframe.dataset_role(frame)
            color = colors(index % colors.N)
            axis.scatter(
                primary,
                physics_value,
                marker=_ROLE_MARKERS[role],
                s=80,
                facecolors=color if physics_enabled else "none",
                edgecolors=color,
                linewidths=1.5,
                label=f"{label} ({role})",
            )
            axis.annotate(
                f"{label}\n{family}; continuity={continuity}",
                (primary, physics_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("normalized_macro_rmse [1] (lower is better)")
        axis.set_ylabel(f"median {metric_label} [{unit}] (lower is better)")
        axis.set_title(metric)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside upper center", ncol=min(4, len(handles)))
    figure.suptitle("Accuracy versus physics Pareto small multiples (linear axes)")
    return figure
