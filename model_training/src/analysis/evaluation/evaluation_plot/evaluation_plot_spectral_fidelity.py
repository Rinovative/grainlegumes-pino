"""
===============================================================================
evaluation_plot_spectral_fidelity.py
===============================================================================
Restore historical output-spectrum presentation on current bounded reducers.

Responsibilities:
  - Compute Hann-windowed radial spectra from physical coordinate spacing
  - Compare reference, prediction, and error spectra for p, u, v, and |u|
  - Plot prediction-to-reference transfer ratios with casewise uncertainty
  - Apply explicit channel selection and optional per-case normalization
  - Mask ratios where reference energy is too small for stable interpretation

Design principles:
  - Spectral evidence is architecture-independent and session-owned
  - Frequencies use inverse coordinate units and field power retains squared units
  - TaskSpec vector magnitude is derived explicitly from declared components
  - Channels remain separate and incompatible physical units never mix

This module does NOT:
  - Parse artifact cases or invent physical coordinate spacing
  - Inspect learned layers, latent activations, or model internals
  - Restore the intentionally omitted learned-layer or latent spectral hooks
  - Own notebook controls or public panel composition
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 50
_REFERENCE_ENERGY_FLOOR = 1e-12
_SPECTRAL_DIMENSIONS = 2


def radial_power_spectrum(
    field: np.ndarray,
    *,
    dx: float,
    dy: float,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the maintained Hann-windowed radial physical-frequency spectrum."""
    return sessions.radial_power_spectrum(field, dx=dx, dy=dy, n_bins=n_bins)


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q10, median, and q90 while preserving wholly masked bins."""
    valid_columns = np.isfinite(values).any(axis=0)
    q10 = np.full(values.shape[1], np.nan)
    median = np.full(values.shape[1], np.nan)
    q90 = np.full(values.shape[1], np.nan)
    if valid_columns.any():
        selected = values[:, valid_columns]
        q10[valid_columns] = np.nanquantile(selected, 0.1, axis=0)
        median[valid_columns] = np.nanquantile(selected, 0.5, axis=0)
        q90[valid_columns] = np.nanquantile(selected, 0.9, axis=0)
    return q10, median, q90


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalize every case spectrum to unit sum, preserving zero spectra."""
    denominator = np.sum(values, axis=1, keepdims=True)
    return np.divide(values, denominator, out=np.array(values, copy=True), where=denominator > 0.0)


def _log_spectral_transfer(
    reference: np.ndarray,
    prediction: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return supported log10 prediction/GT transfer and per-bin support."""
    reference_values = np.asarray(reference, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    frequency_values = np.asarray(frequencies, dtype=float)
    if reference_values.ndim != _SPECTRAL_DIMENSIONS or prediction_values.shape != reference_values.shape:
        msg = "Reference and prediction spectra must have equal (case, frequency) shape."
        raise ValueError(msg)
    if frequency_values.shape != (reference_values.shape[1],):
        msg = "Frequencies must align with the spectral-bin dimension."
        raise ValueError(msg)
    if (
        not np.isfinite(reference_values).all()
        or not np.isfinite(prediction_values).all()
        or not np.isfinite(frequency_values).all()
        or np.any(reference_values < 0.0)
        or np.any(prediction_values < 0.0)
    ):
        msg = "Spectra and frequencies must be finite and non-negative."
        raise ValueError(msg)
    case_floor = _REFERENCE_ENERGY_FLOOR * np.max(reference_values, axis=1, keepdims=True)
    support = (frequency_values[None, :] > 0.0) & (reference_values > case_floor)
    transfer = np.full(reference_values.shape, np.nan, dtype=float)
    stabilized_prediction = np.maximum(prediction_values, case_floor)
    np.log10(
        np.divide(
            stabilized_prediction,
            reference_values,
            out=np.ones_like(reference_values),
            where=support,
        ),
        out=transfer,
        where=support,
    )
    support_count = np.sum(support, axis=0, dtype=int)
    support_fraction = support_count.astype(float) / reference_values.shape[0]
    return transfer, support_fraction, support_count


def _plot_band(
    axis: Axes,
    frequencies: np.ndarray,
    values: np.ndarray,
    *,
    color: object,
    linestyle: str,
    linewidth: float = 2.2,
) -> None:
    """Plot current casewise median and q10-q90 band in historical styling."""
    q10, median, q90 = _quantiles(values)
    valid = (frequencies > 0.0) & np.isfinite(median) & (median > 0.0)
    if not valid.any():
        return
    axis.plot(frequencies[valid], median[valid], color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)
    band = valid & np.isfinite(q10) & np.isfinite(q90) & (q10 > 0.0)
    axis.fill_between(frequencies[band], q10[band], q90[band], color=color, alpha=0.12)


def _summaries(
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
) -> dict[int, sessions.PrefixEvaluationSummary]:
    """Return one validated bounded-prefix summary per current frame."""
    dataframe.validate_comparison(datasets)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    with sessions.scoped_session(datasets) as active_session:
        return {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}


def _spectral(
    summary: sessions.PrefixEvaluationSummary,
    field: presentation.DisplayField,
) -> sessions.SpectralFieldSummary:
    """Return learned-field or TaskSpec magnitude spectral evidence."""
    if field.is_magnitude:
        return summary.magnitudes[field.key.removesuffix("_magnitude")].spectrum
    return summary.spectra[field.component_fields[0]]


def _active_fields(
    datasets: Mapping[str, pd.DataFrame],
    channels: Sequence[str],
) -> tuple[presentation.DisplayField, ...]:
    """Resolve a non-empty ordered historical channel checkbox selection."""
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    by_label = {field.label: field for field in fields}
    magnitude = next((field for field in fields if field.key == "velocity_magnitude"), None)
    if magnitude is not None:
        by_label["U"] = magnitude
    selected = tuple(channels)
    if not selected or any(channel not in by_label for channel in selected):
        visible = tuple(field.label for field in fields)
        msg = f"Select at least one channel from {visible}."  # noqa: S608
        raise ValueError(msg)
    return tuple(by_label[channel] for channel in selected)


def _colors(fields: Sequence[presentation.DisplayField]) -> dict[str, tuple[float, float, float, float]]:
    """Return stable channel colors in selected historical order."""
    cmap = plt.get_cmap("tab10")
    return {field.label: cmap(index % cmap.N) for index, field in enumerate(fields)}


def plot_spectral_demand_prediction_error(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    channels: Sequence[str] = ("p", "u", "v", "U"),
    normalize: bool = True,
) -> Figure:
    """Restore two-row dataset columns for demand/prediction and error spectra."""
    fields = _active_fields(datasets, channels)
    summaries = _summaries(datasets, max_cases)
    colors = _colors(fields)
    figure_title, count_headings = layout.aggregate_title_context(
        "Spectral demand vs prediction and error",
        {label: summaries[id(frame)].case_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=(6.0 * len(datasets) + 2.8, 7.0))
    grid = figure.add_gridspec(
        2,
        len(datasets) + 1,
        width_ratios=(*([1.0] * len(datasets)), 0.35),
        wspace=0.25,
        hspace=0.25,
    )
    top_axes = [figure.add_subplot(grid[0, index]) for index in range(len(datasets))]
    bottom_axes = [figure.add_subplot(grid[1, index]) for index in range(len(datasets))]
    legend_axis = figure.add_subplot(grid[:, -1])
    legend_axis.axis("off")
    for dataset_index, (label, frame) in enumerate(datasets.items()):
        top_axis = top_axes[dataset_index]
        bottom_axis = bottom_axes[dataset_index]
        summary = summaries[id(frame)]
        for field in fields:
            spectrum = _spectral(summary, field)
            reference = _normalize(spectrum.reference) if normalize else spectrum.reference
            prediction = _normalize(spectrum.prediction) if normalize else spectrum.prediction
            error = _normalize(spectrum.error) if normalize else spectrum.error
            _plot_band(top_axis, spectrum.frequencies, reference, color=colors[field.label], linestyle="--", linewidth=2.0)
            _plot_band(top_axis, spectrum.frequencies, prediction, color=colors[field.label], linestyle="-")
            _plot_band(bottom_axis, spectrum.frequencies, error, color=colors[field.label], linestyle="-")
        for axis in (top_axis, bottom_axis):
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
        top_axis.set_title(count_headings[label] or label)
        if dataset_index == 0:
            suffix = " (normalised)" if normalize else ""
            top_axis.set_ylabel(f"Spectral power{suffix}")
            bottom_axis.set_ylabel(f"Error spectral power{suffix}")
        bottom_axis.set_xlabel(f"Spatial frequency k [1/{summary.coordinate_unit}]")
    curve_handles = (
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label="GT demand"),
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-", label="Prediction"),
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-", label="Error"),
    )
    curve_legend = legend_axis.legend(handles=curve_handles, title="Curves", loc="upper left")
    legend_axis.add_artist(curve_legend)
    channel_handles = [Line2D([0], [0], color=colors[field.label], linewidth=2.6, label=field.matplotlib_label) for field in fields]
    legend_axis.legend(
        channel_handles,
        [field.matplotlib_label for field in fields],
        title="Channels",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.80),
    )
    figure.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.98)
    layout.title_over_axes(figure, figure_title, (*top_axes, *bottom_axes), y=0.97)
    return figure


def plot_spectral_transfer_ratio(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    channels: Sequence[str] = ("p", "u", "v", "U"),
) -> Figure:
    """Plot supported log spectral transfer with one compact right sidebar."""
    fields = _active_fields(datasets, channels)
    summaries = _summaries(datasets, max_cases)
    colors = _colors(fields)
    figure_title, count_headings = layout.aggregate_title_context(
        "Log spectral transfer with q10-q90 uncertainty and support",
        {label: summaries[id(frame)].case_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=(12.2, 1.2 + 3.6 * len(datasets)))
    outer_grid = figure.add_gridspec(
        len(datasets),
        2,
        width_ratios=(1.0, 0.28),
        hspace=0.28,
        wspace=0.18,
    )
    transfer_axes = []
    support_axes = []
    for dataset_index, (label, frame) in enumerate(datasets.items()):
        group_grid = outer_grid[dataset_index, 0].subgridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.06)
        transfer_axis = figure.add_subplot(group_grid[0, 0])
        support_axis = figure.add_subplot(group_grid[1, 0], sharex=transfer_axis)
        transfer_axes.append(transfer_axis)
        support_axes.append(support_axis)
        summary = summaries[id(frame)]
        for field in fields:
            spectrum = _spectral(summary, field)
            transfer, support_fraction, _support_count = _log_spectral_transfer(
                spectrum.reference,
                spectrum.prediction,
                spectrum.frequencies,
            )
            q10, median, q90 = _quantiles(transfer)
            valid = (spectrum.frequencies > 0.0) & np.isfinite(median)
            transfer_axis.plot(spectrum.frequencies[valid], median[valid], color=colors[field.label], linewidth=2.2)
            band = valid & np.isfinite(q10) & np.isfinite(q90)
            transfer_axis.fill_between(
                spectrum.frequencies[band],
                q10[band],
                q90[band],
                color=colors[field.label],
                alpha=0.14,
            )
            support_axis.plot(
                spectrum.frequencies[valid],
                support_fraction[valid],
                color=colors[field.label],
                linewidth=1.8,
            )
        transfer_axis.axhline(0.0, linewidth=1.6, linestyle="--", color="black", alpha=0.6)
        transfer_axis.set_xscale("log")
        support_axis.set_xscale("log")
        transfer_axis.grid(True, which="both", linestyle="--", alpha=0.3)
        support_axis.grid(True, which="both", linestyle="--", alpha=0.3)
        support_axis.set_ylim(0.0, 1.05)
        support_axis.set_xlabel(f"Spatial frequency k [1/{summary.coordinate_unit}]")
        transfer_axis.tick_params(labelbottom=False)
        transfer_axis.set_title(count_headings[label] or label, loc="left", fontsize=11)
        transfer_axis.set_ylabel(r"$\log_{10}(S_{pred}/S_{GT})$ [-]")
        support_axis.set_ylabel("Supported fraction")

    legend_axis = figure.add_subplot(outer_grid[:, 1])
    legend_axis.axis("off")
    handles = [Line2D([0], [0], color="black", linewidth=1.6, linestyle="--")]
    labels = ["equal spectral power"]
    handles.extend(Line2D([0], [0], color=colors[field.label], linewidth=2.6) for field in fields)
    labels.extend(field.matplotlib_label for field in fields)
    legend_axis.legend(handles, labels, title="Transfer curves", loc="upper left", frameon=True)
    explanation_y = max(0.05, 0.95 - 0.04 * (len(labels) + 1))
    legend_axis.text(
        0.0,
        explanation_y,
        "Support requires k > 0 and GT power above 10⁻¹² x its case maximum. Zero predictions use that numerical floor.",
        transform=legend_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        wrap=True,
    )
    figure.subplots_adjust(top=0.88, bottom=0.11, left=0.10, right=0.98)
    layout.title_over_axes(figure, figure_title, (*transfer_axes, *support_axes), y=0.97)
    return figure


def plot_spectral_fidelity(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Retain a direct current spectral entry point outside approved panel scope."""
    return plot_spectral_demand_prediction_error(
        datasets=datasets,
        max_cases=max_cases,
        normalize=False,
    )
