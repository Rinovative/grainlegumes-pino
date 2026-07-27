"""
===============================================================================
evaluation_plot_sensitivity_capacity.py
===============================================================================
Separate exact model capacity evidence from exploratory metadata sensitivity.

Responsibilities:
  - Compare authoritative accuracy with exact trainable parameter counts
  - Compute disclosed Spearman associations for scalar source metadata
  - Plot count-disclosed binned metadata/error trends for explicit metrics

Design principles:
  - Capacity uses persisted parameter counts, never a proxy efficiency score
  - Metadata associations are exploratory and do not imply causality
  - Missing/non-finite metadata is rejected or omitted explicitly
  - Error metrics keep their own semantic scales and units

This module does NOT:
  - Admit model or dataset identities for comparison
  - Perform hyperparameter search, causal inference, or model selection
===============================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_DEFAULT_PARAMETER_LIMIT = 8
_TREND_BIN_COUNT = 8
_ROLE_MARKERS = {"ID": "o", "OOD": "s", "unspecified": "D"}


def _model_identity(frame: pd.DataFrame) -> tuple[str, int, Mapping[str, Any]]:
    """
    Read architecture family, exact trainable count, and declared parameters.

    Complete provenance is authoritative. Missing/non-positive counts or malformed
    architecture mappings raise the comparison exception rather than falling back
    to labels, model objects, or proxy capacity measures.
    """
    provenance = dataframe.require_complete_provenance(frame)
    model = provenance.get("model")
    if not isinstance(model, Mapping):
        msg = "Capacity analysis requires model provenance."
        raise dataframe.ComparisonCompatibilityError(msg)
    family = model.get("kind")
    counts = model.get("parameter_counts")
    architecture = model.get("architecture")
    if not isinstance(family, str) or not family or not isinstance(counts, Mapping) or not isinstance(architecture, Mapping):
        msg = "Capacity analysis requires model kind, architecture, and parameter counts."
        raise dataframe.ComparisonCompatibilityError(msg)
    trainable = counts.get("trainable")
    if isinstance(trainable, bool) or not isinstance(trainable, Integral) or int(trainable) <= 0:
        msg = "Capacity analysis requires an exact positive trainable parameter count."
        raise dataframe.ComparisonCompatibilityError(msg)
    return family, int(trainable), architecture


def plot_capacity_accuracy(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot exact model capacity against authoritative aggregate accuracy.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible frames whose provenance includes exact trainable counts and
        declared architecture parameters.

    Returns
    -------
    matplotlib.figure.Figure
        Log-capacity versus ``normalized_macro_rmse`` with architecture facts,
        dataset role, and selected case count disclosed per point.

    Raises
    ------
    ComparisonCompatibilityError, KeyError, TypeError, ValueError
        If comparison identity, model provenance, or aggregate evidence is invalid.

    Notes
    -----
    The plot does not divide accuracy by parameter count or construct an
    efficiency/ranking score.

    """
    dataframe.validate_comparison(datasets)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (label, frame) in enumerate(datasets.items()):
        family, trainable, architecture = _model_identity(frame)
        primary = float(frame.attrs["normalized_macro_rmse"]["value"])
        role = dataframe.dataset_role(frame)
        axis.scatter(
            trainable,
            primary,
            marker=_ROLE_MARKERS[role],
            s=90,
            color=colors(index % colors.N),
            label=f"{label} ({role}), n={len(frame)}",
        )
        declared = ", ".join(f"{key}={value}" for key, value in sorted(architecture.items()))
        axis.annotate(
            f"{label}: {family}\n{declared}",
            (trainable, primary),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("exact trainable parameter count")
    axis.set_ylabel("normalized_macro_rmse [1] (lower is better)")
    axis.set_title("Capacity versus authoritative aggregate accuracy")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    return figure


def _error_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return named per-case normalized/relative response metrics."""
    fields = tuple(frame.attrs["output_fields"])
    return ("rel_l2", "rel_h1", *(f"normalized_rmse_{field}" for field in fields))


def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return one finite numeric column or fail rather than silently dropping rows."""
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        msg = f"Sensitivity column {column!r} must be finite numeric for every case."
        raise ValueError(msg)
    return values


def plot_metadata_error_heatmap(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot exploratory Spearman associations between metadata and case errors.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible frames with flattened finite scalar metadata and named error
        columns.

    Returns
    -------
    matplotlib.figure.Figure
        One annotated correlation matrix per frame on the dimensionless
        ``[-1, 1]`` Spearman scale; undefined constant-column values show ``n/a``.

    Raises
    ------
    ComparisonCompatibilityError, ValueError
        If comparison fails, no eligible metadata exists, or any selected column
        is not finite numeric for every case.

    Notes
    -----
    Associations are descriptive and do not establish causation. Frames without
    eligible metadata are omitted when at least one other frame remains.

    """
    dataframe.validate_comparison(datasets)
    specifications: list[tuple[str, pd.DataFrame, tuple[str, ...]]] = []
    for label, frame in datasets.items():
        columns = dataframe.numeric_metadata_columns(frame)
        if columns:
            specifications.append((label, frame, columns))
    if not specifications:
        msg = "Metadata sensitivity requires at least one finite scalar metadata column."
        raise ValueError(msg)

    width = max(len(columns) for _label, _frame, columns in specifications)
    figure, axes = plt.subplots(
        len(specifications),
        1,
        figsize=(max(9.0, 0.65 * width), 3.6 * len(specifications)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, (label, frame, metadata_columns) in enumerate(specifications):
        errors = _error_columns(frame)
        correlation = np.empty((len(errors), len(metadata_columns)), dtype=float)
        for error_index, error_column in enumerate(errors):
            y_values = _finite_numeric(frame, error_column)
            for metadata_index, metadata_column in enumerate(metadata_columns):
                x_values = _finite_numeric(frame, metadata_column)
                correlation[error_index, metadata_index] = pd.Series(x_values).corr(pd.Series(y_values), method="spearman")
        axis = axes[row_index, 0]
        image = axis.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_xticks(np.arange(len(metadata_columns)), metadata_columns, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(errors)), errors)
        axis.set_title(f"{label} ({dataframe.dataset_role(frame)}), n={len(frame)}")
        for y_index, x_index in np.ndindex(correlation.shape):
            value = correlation[y_index, x_index]
            axis.text(x_index, y_index, "n/a" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
    if image is not None:
        figure.colorbar(image, ax=axes[:, 0], label="Spearman rank correlation [1]", fraction=0.025)
    figure.suptitle("Exploratory metadata-error associations; correlation does not establish causation")
    return figure


def _binned_trend(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce paired values into non-empty quantile bins with exact case counts.

    Centers and responses are within-bin medians. A constant explanatory variable
    yields one bin; duplicated quantile edges are collapsed before assignment.
    """
    unique = np.unique(x_values)
    if unique.size == 1:
        return unique, np.asarray([np.median(y_values)]), np.asarray([len(y_values)])
    bin_count = min(_TREND_BIN_COUNT, unique.size)
    edges = np.unique(np.quantile(x_values, np.linspace(0.0, 1.0, bin_count + 1)))
    assignments = np.clip(np.digitize(x_values, edges[1:-1]), 0, len(edges) - 2)
    centers: list[float] = []
    medians: list[float] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        selected = assignments == index
        if not selected.any():
            continue
        centers.append(float(np.median(x_values[selected])))
        medians.append(float(np.median(y_values[selected])))
        counts.append(int(np.count_nonzero(selected)))
    return np.asarray(centers), np.asarray(medians), np.asarray(counts)


def plot_metadata_error_trends(
    *,
    datasets: Mapping[str, pd.DataFrame],
    metric: str = "rel_l2",
    max_parameters: int = _DEFAULT_PARAMETER_LIMIT,
) -> Figure:
    """
    Plot quantile-binned metadata trends for one explicit error metric.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible frames with flattened finite scalar metadata.
    metric : str, optional
        Named relative or normalized per-case response metric.
    max_parameters : int, optional
        Positive cap on first-seen eligible metadata parameters displayed.

    Returns
    -------
    matplotlib.figure.Figure
        Per-parameter median trends with exact within-bin case counts annotated.

    Raises
    ------
    ComparisonCompatibilityError, ValueError
        If comparison fails, the parameter cap/metric is invalid, metadata is
        unavailable, or selected values are not finite numeric for every case.

    Notes
    -----
    Metadata parameter units are not declared by the artifact schema and are
    labelled accordingly; trends are exploratory rather than causal.

    """
    dataframe.validate_comparison(datasets)
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, int) or max_parameters <= 0:
        msg = "max_parameters must be a positive integer."
        raise ValueError(msg)
    available = list(
        dict.fromkeys(column for frame in datasets.values() for column in dataframe.numeric_metadata_columns(frame) if column in frame.columns)
    )[:max_parameters]
    if not available:
        msg = "Metadata trend analysis requires scalar metadata columns."
        raise ValueError(msg)
    for frame in datasets.values():
        if metric not in _error_columns(frame):
            msg = f"Unsupported metadata trend metric {metric!r}; choose one named normalized/relative metric."
            raise ValueError(msg)

    ncols = min(3, len(available))
    nrows = math.ceil(len(available) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, constrained_layout=True)
    for axis, parameter in zip(axes.flat, available, strict=False):
        plotted = False
        for label, frame in datasets.items():
            if parameter not in frame.columns:
                continue
            x_values = _finite_numeric(frame, parameter)
            y_values = _finite_numeric(frame, metric)
            centers, medians, counts = _binned_trend(x_values, y_values)
            axis.plot(centers, medians, marker="o", label=f"{label} ({dataframe.dataset_role(frame)}), n={len(frame)}")
            for x_value, y_value, count in zip(centers, medians, counts, strict=True):
                axis.annotate(str(count), (x_value, y_value), xytext=(2, 3), textcoords="offset points", fontsize=7)
            plotted = True
        axis.set_title(parameter)
        axis.set_xlabel(f"{parameter} [stored scalar; unit not declared]")
        axis.set_ylabel(f"median {metric} [1]")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=7)
    for axis in axes.flat[len(available) :]:
        axis.axis("off")
    figure.suptitle("Exploratory binned metadata-error trends; annotations are case counts")
    return figure
