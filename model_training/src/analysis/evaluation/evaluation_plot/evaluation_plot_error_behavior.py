"""
===============================================================================
evaluation_plot_error_behavior.py
===============================================================================
Restore historical predictive-error presentation on current session reducers.

Responsibilities:
  - Compare secondary relative L2 and H1 metrics without redefining the objective
  - Render global/local distributions and GT-versus-prediction field means
  - Plot p, u, v, and TaskSpec velocity-magnitude spatial error reductions
  - Relate field error to target magnitude and left/right boundary distance

Design principles:
  - Learned fields, groups, order, and units come from admitted TaskSpec evidence
  - Complete and bounded-prefix reductions are reused through EvaluationSession
  - Local relative error normalizes each case field by its own reference RMS
  - Percentile clipping changes display scales only, never numerical summaries

This module does NOT:
  - Parse NPZ payloads or reconstruct the authoritative primary aggregate
  - Combine fields with incompatible physical units
  - Own notebook controls, dataset identity, or public panel composition
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 100
_DISTRIBUTION_DEFAULT_CASE_LIMIT = 50
_TARGET_MAGNITUDE_DEFAULT_CASE_LIMIT = 50
_MINIMUM_KDE_SAMPLES = 2
_BAND_LABELS = ("0-5 %", "5-10 %", "10-20 %", "20-40 %", ">40 %")


def _finite(frame: pd.DataFrame, column: str, *, limit: int | None = None) -> np.ndarray:
    """Return finite non-negative current metric values in saved order."""
    selected = frame if limit is None else frame.iloc[:limit]
    values = pd.to_numeric(selected[column], errors="raise").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        msg = f"{column} must contain finite non-negative values."
        raise ValueError(msg)
    return values


def _display_fields(datasets: Mapping[str, pd.DataFrame]) -> tuple[presentation.DisplayField, ...]:
    """Return one shared current display-field order."""
    return presentation.shared_display_fields(tuple(datasets.values()))


def _magnitude_id(field: presentation.DisplayField) -> str:
    """Return the canonical TaskSpec group id for a magnitude display field."""
    if not field.key.endswith("_magnitude"):
        msg = f"Display field {field.key!r} is not a group magnitude."
        raise ValueError(msg)
    return field.key.removesuffix("_magnitude")


def _full_array(summary: sessions.FullEvaluationSummary, field: presentation.DisplayField, attribute: str) -> np.ndarray:
    """Return one full-summary learned or group-magnitude array."""
    if field.is_magnitude:
        return np.asarray(getattr(summary.magnitudes[_magnitude_id(field)], attribute))
    values = cast("np.ndarray", getattr(summary, attribute))
    return np.asarray(values[summary.grid.fields.index(field.component_fields[0])])


def _case_means(
    summary: sessions.FullEvaluationSummary,
    field: presentation.DisplayField,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned reference/prediction spatial means for one display field."""
    if field.is_magnitude:
        magnitude = summary.magnitudes[_magnitude_id(field)]
        return magnitude.case_reference_means, magnitude.case_prediction_means
    index = summary.grid.fields.index(field.component_fields[0])
    return summary.case_reference_means[:, index], summary.case_prediction_means[:, index]


def _prefix_field(
    summary: sessions.PrefixEvaluationSummary,
    field: presentation.DisplayField,
    attribute: str,
) -> Any:
    """Return one learned or magnitude bounded reduction."""
    if field.is_magnitude:
        return getattr(summary.magnitudes[_magnitude_id(field)], attribute)
    mapping = getattr(summary, attribute)
    return mapping[field.component_fields[0]]


def plot_global_error_metrics(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare current secondary relative L2/H1 in the historical 3 x 2 layout."""
    dataframe.validate_comparison(datasets)
    names = tuple(datasets)
    title, count_headings = layout.aggregate_title_context(
        "Global error metrics",
        layout.effective_case_counts(datasets),
    )
    display_names = tuple(count_headings[name] or name for name in names)
    if not names:
        msg = "At least one artifact frame is required."
        raise ValueError(msg)
    metrics = (("Relative L2", "rel_l2"), ("Relative H1", "rel_h1"))
    populations = {column: [_finite(datasets[name], column) for name in names] for _label, column in metrics}
    palette = [plt.get_cmap("tab10")(index % 10) for index in range(len(names))]
    figure = plt.figure(figsize=(21, 10))
    grid = figure.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 0.35), hspace=0.35, wspace=0.25)
    plot_axes: list[Axes] = []
    for column_index, (display_name, column) in enumerate(metrics):
        values_by_dataset = populations[column]
        box_axis = figure.add_subplot(grid[0, column_index])
        density_axis = figure.add_subplot(grid[1, column_index])
        cdf_axis = figure.add_subplot(grid[2, column_index])
        plot_axes.extend((box_axis, density_axis, cdf_axis))
        boxes = box_axis.boxplot(
            values_by_dataset,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, color in zip(boxes["boxes"], palette, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        box_axis.set_xticks([])
        box_axis.set_title(f"{display_name} - Boxplot")
        box_axis.set_ylabel(f"{display_name} (dimensionless)")
        for values, name, color in zip(values_by_dataset, names, palette, strict=True):
            if values.size >= _MINIMUM_KDE_SAMPLES and not np.allclose(values, values[0]):
                coordinates = np.linspace(float(np.min(values)), float(np.max(values)), 400)
                density_axis.plot(coordinates, gaussian_kde(values)(coordinates), color=color, label=name)
            else:
                density_axis.axvline(float(values[0]), color=color, label=name)
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
            cdf_axis.plot(ordered, cumulative, color=color, label=name)
        density_axis.set_title(f"{display_name} - KDE Density")
        density_axis.set_xlabel(f"{display_name} (dimensionless)")
        density_axis.set_ylabel("Density")
        cdf_axis.set_title(f"{display_name} - CDF")
        cdf_axis.set_xlabel(f"{display_name} (dimensionless)")
        cdf_axis.set_ylabel("CDF")
        if all(np.all(values > 0.0) for values in values_by_dataset):
            box_axis.set_yscale("log")
            density_axis.set_xscale("log")
            cdf_axis.set_xscale("log")
        for axis in (box_axis, density_axis, cdf_axis):
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
    legend_axis = figure.add_subplot(grid[:, 2])
    legend_axis.axis("off")
    legend_axis.legend(
        [Line2D([0], [0], color=color, linewidth=8) for color in palette],
        display_names,
        loc="upper left",
    )
    figure.subplots_adjust(top=0.90)
    layout.title_over_axes(figure, title, plot_axes, y=0.97)
    return figure


def _plot_ecdf(axis: Any, values: np.ndarray, *, color: object, label: str | None = None) -> None:
    """Plot one finite empirical cumulative distribution without smoothing."""
    ordered = np.sort(np.asarray(values, dtype=float))
    cumulative = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    axis.step(ordered, cumulative, where="post", color=color, linewidth=2.0, label=label)


def plot_predictive_error_distributions(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DISTRIBUTION_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot transparent casewise and local predictive-error distributions."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}
    title, count_headings = layout.aggregate_title_context(
        "Predictive error distributions (lower is better)",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    colors = {field.key: plt.get_cmap("tab10")(index % 10) for index, field in enumerate(fields)}
    figure, axes = plt.subplots(
        2,
        len(datasets),
        figsize=(6.4 * len(datasets), 7.4),
        squeeze=False,
        sharey="row",
    )
    for column, (dataset_label, frame) in enumerate(datasets.items()):
        count = min(max_cases, len(frame))
        summary = summaries[id(frame)]
        global_axis = axes[0, column]
        local_axis = axes[1, column]
        _plot_ecdf(global_axis, _finite(frame, "rel_l2", limit=count), color="black")
        for field in fields:
            local = _prefix_field(summary, field, "local_relative_error")
            local_axis.plot(
                local.quantiles,
                local.probabilities,
                color=colors[field.key],
                linewidth=2.0,
                label=field.matplotlib_label,
            )
        global_axis.set_title(count_headings[dataset_label] or dataset_label)
        global_axis.set_xlabel("Casewise channel-balanced relative L2 [-]")
        local_axis.set_xlabel("Pointwise |error| / case reference-field RMS [-]")
        for axis in (global_axis, local_axis):
            axis.set_xscale("symlog", linthresh=1e-6)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
        if column == 0:
            global_axis.set_ylabel("Fraction of cases")
            local_axis.set_ylabel("Fraction of case-gridpoint values")
    handles = [Line2D([0], [0], color=colors[field.key], linewidth=2.6) for field in fields]
    figure.legend(
        handles,
        [field.matplotlib_label for field in fields],
        title="Local-error field",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(fields),
        frameon=True,
    )
    figure.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.98, hspace=0.38, wspace=0.22)
    layout.title_over_axes(figure, title, tuple(axes.flat), y=0.985)
    return figure


def plot_mean_field_bias(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DISTRIBUTION_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Restore four-row ground-truth versus prediction case-mean panels."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases) for frame in datasets.values()}
    title, count_headings = layout.aggregate_title_context(
        "GT vs prediction mean",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    figure, axes = plt.subplots(
        len(fields),
        len(datasets),
        figsize=(6 * len(datasets), 9),
        squeeze=False,
    )
    for column, (label, frame) in enumerate(datasets.items()):
        summary = summaries[id(frame)]
        for row, field in enumerate(fields):
            reference, prediction = _case_means(summary, field)
            low = float(min(np.min(reference), np.min(prediction)))
            high = float(max(np.max(reference), np.max(prediction)))
            if np.isclose(low, high):
                high += np.finfo(float).eps
            rmse = float(np.sqrt(np.mean((prediction - reference) ** 2)))
            denominator = float(np.sum((reference - np.mean(reference)) ** 2))
            r2 = 1.0 - float(np.sum((prediction - reference) ** 2)) / denominator if denominator > 0.0 else float("nan")
            axis = axes[row, column]
            axis.plot((low, high), (low, high), "k--", linewidth=1, alpha=0.7)
            axis.scatter(reference, prediction, s=18, alpha=0.45)
            heading = count_headings[label] or label
            prefix = f"{heading}\n" if row == 0 else ""
            axis.set_title(f"{prefix}{field.matplotlib_label}: RMSE={rmse:.3g}, R2={r2:.3g}", fontsize=11)
            if column == 0:
                axis.set_ylabel("Prediction mean")
            if row == len(fields) - 1:
                axis.set_xlabel("GT mean")
            axis.grid(alpha=0.3)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.98, hspace=0.35, wspace=0.25)
    return figure


def _map_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    attribute: str,
    metric_label: str,
    title: str,
    relative_percent: bool = False,
) -> Figure:
    """Render one historical four-row spatial statistic family."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_spatial() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        title,
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    figure, axes = layout.map_subplots(rows=len(fields), columns=len(datasets))
    for column, (label, frame) in enumerate(datasets.items()):
        summary = summaries[id(frame)]
        x_min, x_max, y_min, y_max = summary.grid.extent
        for row, field in enumerate(fields):
            values = _full_array(summary, field, attribute)
            if relative_percent:
                values = values * 100.0
            upper = max(float(np.nanpercentile(values, 99.5)), np.finfo(float).eps)
            displayed = np.ma.masked_greater(values, upper)
            image = axes[row, column].contourf(
                np.linspace(x_min, x_max, values.shape[1]),
                np.linspace(y_min, y_max, values.shape[0]),
                displayed,
                levels=np.linspace(0.0, upper, 11),
                cmap="magma",
            )
            heading = count_headings[label] or label
            prefix = f"{heading}\n" if row == 0 else ""
            axes[row, column].set_title(f"{prefix}{field.matplotlib_label} {metric_label}", fontsize=11)
            axes[row, column].set_aspect("equal")
            layout.add_map_colorbar(figure, image, axes[row, column])
    first_summary = summaries[id(next(iter(datasets.values())))]
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_summary.grid.coordinate_units[0]}]",
        y_label=f"y [{first_summary.grid.coordinate_units[1]}]",
    )
    figure.suptitle(figure_title)
    return figure


def plot_mean_error_maps(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    error_mode: str = "MAE",
) -> Figure:
    """Plot current MAE or field-RMS-relative mean maps in historical layout."""
    if error_mode in {"MAE", "absolute"}:
        return _map_figure(
            datasets=datasets,
            max_cases=max_cases,
            attribute="absolute_error_mean",
            metric_label="MAE",
            title="Mean absolute error maps",
        )
    if error_mode in {"Relative [%]", "local_relative"}:
        return _map_figure(
            datasets=datasets,
            max_cases=max_cases,
            attribute="local_relative_error_mean",
            metric_label="relative error [% of reference RMS]",
            title="Mean relative error maps",
            relative_percent=True,
        )
    msg = "error_mode must be 'MAE' or 'Relative [%]'."
    raise ValueError(msg)


def plot_std_error_maps(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot current signed-error standard deviation in historical layout."""
    return _map_figure(
        datasets=datasets,
        max_cases=max_cases,
        attribute="signed_error_std",
        metric_label="STD error",
        title="Error standard-deviation maps",
    )


def plot_error_vs_target_magnitude(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _TARGET_MAGNITUDE_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Restore four-channel target-magnitude trends with a dedicated legend."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Error vs GT magnitude",
        {label: summaries[id(frame)].case_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=(9.5, 9))
    grid = figure.add_gridspec(len(fields), 2, width_ratios=(1.0, 0.35), hspace=0.35, wspace=0.25)
    axes = [figure.add_subplot(grid[row, 0]) for row in range(len(fields))]
    legend_axis = figure.add_subplot(grid[:, 1])
    legend_axis.axis("off")
    handles: list[Line2D] = []
    for row, field in enumerate(fields):
        axis = axes[row]
        for label, frame in datasets.items():
            trend = _prefix_field(summaries[id(frame)], field, "target_magnitude_error")
            display_label = count_headings[label] or label
            (line,) = axis.plot(trend.centers, trend.medians, marker="o", markersize=4, label=display_label, alpha=0.9)
            if row == 0:
                handles.append(line)
        if field.label != "p":
            all_positive = all(
                np.all(_prefix_field(summaries[id(frame)], field, "target_magnitude_error").centers > 0.0)
                and np.all(_prefix_field(summaries[id(frame)], field, "target_magnitude_error").medians > 0.0)
                for frame in datasets.values()
            )
            if all_positive:
                axis.set_xscale("log")
                axis.set_yscale("log")
        axis.set_title(f"{field.matplotlib_label}: MAE vs |GT|")
        axis.set_ylabel(f"Median MAE [{field.unit}]")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("|GT| (bin center)")
    legend_axis.legend(handles, [str(handle.get_label()) for handle in handles], loc="upper left")
    figure.subplots_adjust(top=0.92, bottom=0.07, left=0.10, right=0.97)
    layout.title_over_axes(figure, figure_title, axes, y=0.98)
    return figure


def plot_boundary_error_decomposition(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    channels: Sequence[str] = ("p", "u", "v", "U"),
) -> Figure:
    """Restore channel-checkbox left/right boundary ratios and external legend."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    by_label = {field.label: field for field in fields}
    magnitude = next((field for field in fields if field.key == "velocity_magnitude"), None)
    if magnitude is not None:
        by_label["U"] = magnitude
    active = tuple(channels)
    if not active or any(channel not in by_label for channel in active):
        msg = f"Select at least one channel from {tuple(by_label)}."  # noqa: S608
        raise ValueError(msg)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Left/right boundary error ratio vs x-distance (interior reference: >40 %)",
        {label: summaries[id(frame)].case_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=(6 * len(datasets) + 2.5, 5))
    grid = figure.add_gridspec(1, len(datasets) + 1, width_ratios=(*([1.0] * len(datasets)), 0.35), wspace=0.25)
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(datasets))]
    legend_axis = figure.add_subplot(grid[0, -1])
    legend_axis.axis("off")
    bar_handles: list[Any] = []
    positions = np.arange(len(active))
    width = 0.8 / len(_BAND_LABELS)
    for axis, (label, frame) in zip(axes, datasets.items(), strict=True):
        summary = summaries[id(frame)]
        means = np.stack([np.asarray(_prefix_field(summary, by_label[channel], "boundary_region_error").means) for channel in active])
        ratios = np.divide(
            means,
            means[:, -1, None],
            out=np.full_like(means, np.nan),
            where=means[:, -1, None] > 0.0,
        )
        for band_index, band_label in enumerate(_BAND_LABELS):
            bars = axis.bar(
                positions + (band_index - (len(_BAND_LABELS) - 1) / 2) * width,
                ratios[:, band_index],
                width,
                label=band_label,
            )
            if len(bar_handles) < len(_BAND_LABELS):
                bar_handles.append(bars[0])
        axis.axhline(1.0, color="black", linestyle="--", alpha=0.4)
        axis.set_xticks(positions, [by_label[channel].matplotlib_label for channel in active])
        axis.set_ylabel("Boundary error ratio (MAE / interior MAE)")
        axis.set_xlabel("Channel")
        axis.set_title(count_headings[label] or label)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    legend_axis.legend(bar_handles, _BAND_LABELS, title="x-distance from left/right boundary", loc="upper left")
    figure.subplots_adjust(top=0.87, bottom=0.07, left=0.001, right=0.98, wspace=0.25)
    layout.title_over_axes(figure, figure_title, axes, y=0.97)
    return figure


def plot_error_maps(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """Retain a compact direct-call four-statistic diagnostic outside panel scope."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame).require_spatial() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Error map diagnostics",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    rows = [(label, frame, field) for label, frame in datasets.items() for field in fields]
    figure, axes = layout.map_subplots(rows=len(rows), columns=4)
    attributes = (
        ("signed_error_mean", "mean signed error", "coolwarm"),
        ("absolute_error_mean", "mean absolute error", "magma"),
        ("signed_error_std", "signed-error standard deviation", "magma"),
        ("absolute_error_q90", "q90 absolute error", "magma"),
    )
    for row, (label, frame, field) in enumerate(rows):
        summary = summaries[id(frame)]
        for column, (attribute, title, cmap) in enumerate(attributes):
            values = _full_array(summary, field, attribute)
            limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
            low = -limit if attribute == "signed_error_mean" else 0.0
            image = axes[row, column].imshow(values, origin="lower", extent=summary.grid.extent, aspect="equal", cmap=cmap, vmin=low, vmax=limit)
            heading = count_headings[label] or label
            axes[row, column].set_title(f"{heading} — {field.matplotlib_label} {title}", fontsize=8)
            layout.add_map_colorbar(figure, image, axes[row, column])
    first_summary = summaries[id(next(iter(datasets.values())))]
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_summary.grid.coordinate_units[0]}]",
        y_label=f"y [{first_summary.grid.coordinate_units[1]}]",
    )
    figure.suptitle(figure_title)
    return figure


def plot_mean_spatial_fields(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """Retain direct current reference/prediction/bias spatial diagnostics."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame).require_spatial() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Mean spatial fields",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    rows = [(label, frame, field) for label, frame in datasets.items() for field in fields]
    figure, axes = layout.map_subplots(rows=len(rows), columns=3)
    for row, (label, frame, field) in enumerate(rows):
        summary = summaries[id(frame)]
        reference = _full_array(summary, field, "reference_mean")
        prediction = _full_array(summary, field, "prediction_mean")
        bias = prediction - reference
        low = float(min(np.min(reference), np.min(prediction)))
        high = float(max(np.max(reference), np.max(prediction)))
        if np.isclose(low, high):
            high += np.finfo(float).eps
        for column, values, title, cmap, vmin, vmax in (
            (0, reference, "mean reference", "viridis", low, high),
            (1, prediction, "mean prediction", "viridis", low, high),
            (
                2,
                bias,
                "mean prediction - reference",
                "coolwarm",
                -max(np.max(np.abs(bias)), np.finfo(float).eps),
                max(np.max(np.abs(bias)), np.finfo(float).eps),
            ),
        ):
            image = axes[row, column].imshow(values, origin="lower", extent=summary.grid.extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            heading = count_headings[label] or label
            axes[row, column].set_title(f"{heading} — {field.matplotlib_label} {title}", fontsize=8)
            layout.add_map_colorbar(figure, image, axes[row, column])
    first_summary = summaries[id(next(iter(datasets.values())))]
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_summary.grid.coordinate_units[0]}]",
        y_label=f"y [{first_summary.grid.coordinate_units[1]}]",
    )
    figure.suptitle(figure_title)
    return figure
