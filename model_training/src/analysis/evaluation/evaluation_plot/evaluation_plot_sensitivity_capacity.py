"""
===============================================================================
evaluation_plot_sensitivity_capacity.py
===============================================================================
Separate exact model-capacity evidence from exploratory metadata sensitivity.

Responsibilities:
  - Compare authoritative accuracy with declared architecture values
  - Compare authoritative accuracy with exact trainable parameter counts
  - Compute Spearman associations for supported scientific metadata
  - Plot quantile-binned metadata/error sensitivity in canonical parameter order

Design principles:
  - Capacity uses persisted provenance and never a proxy efficiency score
  - Metadata filtering and labels come from one current-native presentation layer
  - Associations are exploratory and do not imply causality
  - Short, constant, missing, or non-finite evidence is disclosed or rejected

This module does NOT:
  - Infer architecture from run names, paths, or hidden defaults
  - Admit model or dataset identities for comparison
  - Perform hyperparameter search, causal inference, or model selection
===============================================================================
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.artifacts import analysis_artifact_contracts as contracts
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_DEFAULT_PARAMETER_LIMIT = 8
_TREND_BIN_COUNT = 8
_ROLE_MARKERS = {"ID": "o", "OOD": "s", "unspecified": "D"}
_MINIMUM_SENSITIVITY_CASES = 20
_VARIATION_FLOOR = 1e-12
_MINIMUM_QUANTILE_EDGES = 4
_MINIMUM_BIN_CASES = 2
_MINIMUM_SENSITIVITY_BINS = 3


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


def _numeric_architecture_values(
    value: object,
    *,
    prefix: str = "",
    output: dict[str, float] | None = None,
) -> dict[str, float]:
    """Flatten declared numeric architecture values without family-specific names."""
    result = {} if output is None else output
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _numeric_architecture_values(item, prefix=child, output=result)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _numeric_architecture_values(item, prefix=f"{prefix}[{index}]", output=result)
    elif not isinstance(value, bool) and isinstance(value, Real) and prefix:
        numeric = float(value)
        if np.isfinite(numeric):
            result[prefix] = numeric
    return result


def plot_error_vs_architecture_parameters(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot authoritative aggregate error against declared architecture parameters.

    Numeric architecture leaves are flattened in first-seen provenance order.
    No config file, run name, family-specific proxy, or hidden default is read.
    """
    dataframe.validate_comparison(datasets)
    summaries: list[tuple[str, pd.DataFrame, dict[str, float], float]] = []
    parameter_order: list[str] = []
    for label, frame in datasets.items():
        _family, _trainable, architecture = _model_identity(frame)
        values = _numeric_architecture_values(architecture)
        for parameter in values:
            if parameter not in parameter_order:
                parameter_order.append(parameter)
        objective = float(frame.attrs["normalized_group_macro_rmse"]["value"])
        if not np.isfinite(objective) or objective < 0.0:
            msg = "Architecture analysis requires a finite non-negative authoritative aggregate."
            raise ValueError(msg)
        summaries.append((label, frame, values, objective))
    if not parameter_order:
        msg = "Architecture analysis requires at least one declared numeric parameter."
        raise ValueError(msg)

    columns = min(3, len(parameter_order))
    rows = math.ceil(len(parameter_order) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.5 * columns, 4.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    all_positive = all(objective > 0.0 for _label, _frame, _values, objective in summaries)
    for axis, parameter in zip(axes.flat, parameter_order, strict=False):
        for dataset_index, (label, _frame, values, objective) in enumerate(summaries):
            if parameter not in values:
                continue
            axis.scatter(values[parameter], objective, s=70, color=colors(dataset_index % colors.N))
            axis.annotate(
                label,
                (values[parameter], objective),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(parameter)
        axis.set_ylabel("normalized_group_macro_rmse [1]")
        axis.set_yscale("log" if all_positive else "symlog")
        axis.grid(alpha=0.3, which="both")
    for axis in axes.flat[len(parameter_order) :]:
        axis.axis("off")
    figure_title, _count_headings = layout.aggregate_title_context(
        "Authoritative error versus declared architecture parameters",
        layout.effective_case_counts(datasets),
    )
    figure.suptitle(figure_title)
    return figure


def _prefix_objective(frame: pd.DataFrame, max_cases: int) -> tuple[float, int]:
    """Finalize the unchanged task objective on one admitted saved prefix."""
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    selected_count = min(max_cases, len(frame))
    selected = frame.iloc[:selected_count]
    output_groups = [{"id": group_id, "fields": list(fields)} for group_id, fields in frame.attrs["output_groups"]]
    aggregate = contracts.aggregate_normalized_group_macro_rmse(
        selected,
        output_groups=output_groups,
        train_standard_deviations=frame.attrs["train_standard_deviations"],
        normalization_denominator_floor=float(frame.attrs["normalization_denominator_floor"]),
    )
    return float(aggregate["value"]), selected_count


def _architecture_sidebar_text(
    label: str,
    *,
    family: str,
    trainable: int,
    role: str,
    architecture: Mapping[str, Any],
) -> str:
    """Format exact capacity metadata as one compact sidebar block."""
    declared = ", ".join(f"{key}={value}" for key, value in sorted(architecture.items()))
    wrapped = textwrap.fill(declared, width=44, subsequent_indent="  ")
    return f"{label}\n  family={family}, role={role}, trainable={trainable:,}\n  {wrapped}"


def plot_capacity_accuracy(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = 50,
) -> Figure:
    """Plot exact capacity and accuracy with configuration in a right sidebar."""
    dataframe.validate_comparison(datasets)
    figure_title, count_headings = layout.aggregate_title_context(
        "Capacity versus authoritative aggregate accuracy",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    figure = plt.figure(figsize=(14.0, 7.0))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 0.48), wspace=0.18)
    axis = figure.add_subplot(grid[0, 0])
    sidebar_axis = figure.add_subplot(grid[0, 1])
    sidebar_axis.axis("off")
    colors = plt.get_cmap("tab10")
    handles = []
    labels = []
    details = []
    for index, (label, frame) in enumerate(datasets.items()):
        family, trainable, architecture = _model_identity(frame)
        primary, _selected_count = _prefix_objective(frame, max_cases)
        role = dataframe.dataset_role(frame)
        point_label = count_headings[label] or label
        handle = axis.scatter(
            trainable,
            primary,
            marker=_ROLE_MARKERS[role],
            s=90,
            color=colors(index % colors.N),
            label=point_label,
        )
        handles.append(handle)
        labels.append(point_label)
        details.append(
            _architecture_sidebar_text(
                label,
                family=family,
                trainable=trainable,
                role=role,
                architecture=architecture,
            )
        )
    sidebar_axis.legend(handles, labels, title="Model / dataset", loc="upper left", frameon=True)
    details_y = max(0.30, 0.92 - 0.075 * len(labels))
    sidebar_axis.text(
        0.0,
        details_y,
        "Exact declared configuration\n\n" + "\n\n".join(details),
        transform=sidebar_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        wrap=True,
    )
    axis.set_xscale("log")
    axis.set_xlabel("exact trainable parameter count")
    axis.set_ylabel("normalized_group_macro_rmse [1] (lower is better)")
    axis.grid(alpha=0.25, which="both")
    figure.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.98)
    layout.title_over_axes(figure, figure_title, (axis,), y=0.96)
    return figure


def _finite_numeric(frame: pd.DataFrame, column: str, *, max_cases: int) -> np.ndarray:
    """Return one finite numeric saved prefix."""
    values = pd.to_numeric(frame.iloc[:max_cases][column], errors="raise").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        msg = f"Sensitivity column {column!r} must be finite numeric."
        raise ValueError(msg)
    return values


def _selected_fields(
    datasets: Mapping[str, pd.DataFrame],
    channels: tuple[str, ...],
) -> tuple[presentation.DisplayField, ...]:
    """Resolve one non-empty stable channel selection."""
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    by_label = {field.label: field for field in fields}
    magnitude = next((field for field in fields if field.key == "velocity_magnitude"), None)
    if magnitude is not None:
        by_label["U"] = magnitude
    if not channels or any(channel not in by_label for channel in channels):
        msg = f"Select at least one channel from {tuple(by_label)}."  # noqa: S608
        raise ValueError(msg)
    return tuple(by_label[channel] for channel in channels)


def _spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Spearman correlation without invoking SciPy on constant input."""
    x_constant, _x_tolerance = presentation.effectively_constant(x_values)
    y_constant, _y_tolerance = presentation.effectively_constant(y_values)
    if x_constant or y_constant:
        return float("nan")
    return float(pd.Series(x_values).corr(pd.Series(y_values), method="spearman"))


def plot_metadata_error_heatmap(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = 100,
) -> Figure:
    """Restore the all-channel metadata association heatmaps and exclusions."""
    dataframe.validate_comparison(datasets)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    parameters = presentation.metadata_parameters(tuple(datasets.values()), max_cases=max_cases)
    if not parameters:
        msg = "Metadata sensitivity requires supported scientific parameters."
        raise ValueError(msg)
    figure_title, count_headings = layout.aggregate_title_context(
        "Parameter-error association",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    correlations: list[np.ndarray] = []
    for frame in datasets.values():
        matrix = np.empty((len(parameters), len(fields)), dtype=float)
        for parameter_index, parameter in enumerate(parameters):
            x_values = _finite_numeric(frame, parameter, max_cases=max_cases)
            for field_index, field in enumerate(fields):
                y_values = _finite_numeric(frame, field.metric_column, max_cases=max_cases)
                matrix[parameter_index, field_index] = _spearman(x_values, y_values)
        correlations.append(matrix)
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(len(datasets) * (1.25 * len(fields) + 2), 0.25 * len(parameters) + 2),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )
    image = None
    parameter_labels = [presentation.metadata_label(parameter) for parameter in parameters]
    for axis, label, matrix in zip(axes[0], datasets, correlations, strict=True):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_xticks(np.arange(len(fields)), [f"RMSE[{field.matplotlib_label}]" for field in fields])
        axis.set_yticks(np.arange(len(parameters)), parameter_labels)
        axis.set_title(count_headings[label] or label)
        for row, column in np.ndindex(matrix.shape):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="Spearman correlation")
    figure.suptitle(figure_title)
    return figure


def _sensitivity(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return historical P90-P10 of quantile-binned response medians."""
    if x_values.size < _MINIMUM_SENSITIVITY_CASES or np.std(x_values) < _VARIATION_FLOOR or np.std(y_values) < _VARIATION_FLOOR:
        return float("nan")
    edges = np.unique(np.quantile(x_values, np.linspace(0.0, 1.0, 13)))
    if edges.size < _MINIMUM_QUANTILE_EDGES:
        return float("nan")
    assignments = np.clip(np.digitize(x_values, edges[1:-1]), 0, len(edges) - 2)
    medians = [
        float(np.median(y_values[assignments == index]))
        for index in range(len(edges) - 1)
        if np.count_nonzero(assignments == index) >= _MINIMUM_BIN_CASES
    ]
    if len(medians) < _MINIMUM_SENSITIVITY_BINS:
        return float("nan")
    return float(np.percentile(medians, 90) - np.percentile(medians, 10))


def plot_metadata_error_trends(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = 100,
    channels: tuple[str, ...] = ("p", "u", "v", "U"),
) -> Figure:
    """Restore original-order metadata sensitivities with channel checkboxes."""
    dataframe.validate_comparison(datasets)
    fields = _selected_fields(datasets, channels)
    parameters = presentation.metadata_parameters(tuple(datasets.values()), max_cases=max_cases)
    if not parameters:
        msg = "Metadata trend analysis requires supported scientific parameters."
        raise ValueError(msg)
    figure_title, count_headings = layout.aggregate_title_context(
        "Parameter sensitivity",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    figure_height = max(4.8, 0.42 * len(parameters) + 1.8)
    figure = plt.figure(figsize=(8.0 * len(datasets) + 2.5, figure_height))
    grid = figure.add_gridspec(
        1,
        len(datasets) + 1,
        width_ratios=(*([1.0] * len(datasets)), 0.35),
        wspace=0.35,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(datasets))]
    legend_axis = figure.add_subplot(grid[0, -1])
    legend_axis.axis("off")
    y_positions = np.arange(len(parameters))
    handles = []
    for dataset_index, ((label, frame), axis) in enumerate(zip(datasets.items(), axes, strict=True)):
        plotted_values: list[float] = []
        for field in fields:
            values = []
            response = _finite_numeric(frame, field.metric_column, max_cases=max_cases)
            for parameter in parameters:
                explanatory = _finite_numeric(frame, parameter, max_cases=max_cases)
                values.append(_sensitivity(explanatory, response))
            plotted_values.extend(values)
            (line,) = axis.plot(values, y_positions, marker="o", linestyle="-", alpha=0.9, label=field.matplotlib_label)
            if dataset_index == 0:
                handles.append(line)
        axis.set_yticks(y_positions)
        positive = np.asarray(plotted_values, dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if positive.size:
            axis.set_xscale("log")
        else:
            axis.set_xlim(0.0, 1.0)
            axis.text(
                0.5,
                0.5,
                "No resolved sensitivity for these cases",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel("Sensitivity on normalized RMSE (P90 minus P10 of binned medians)")
        axis.set_title(count_headings[label] or label)
        axis.grid(True, which="both", axis="x", alpha=0.3)
    axes[0].set_yticklabels([presentation.metadata_label(parameter) for parameter in parameters])
    axes[0].invert_yaxis()
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    legend_axis.legend(handles, [field.matplotlib_label for field in fields], title="Channel", loc="upper left")
    figure.subplots_adjust(top=0.95, bottom=0.08, left=0.35, right=0.96)
    layout.title_over_axes(figure, figure_title, axes, y=0.98, fontsize=14)
    return figure
