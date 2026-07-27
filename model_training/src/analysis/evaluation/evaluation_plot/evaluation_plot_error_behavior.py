"""
===============================================================================
evaluation_plot_error_behavior.py
===============================================================================
Explain predictive error distributions, spatial structure, and boundary behavior.

Responsibilities:
  - Compare normalized/relative case metrics without redefining the objective
  - Render physical mean reference/prediction fields and signed mean bias
  - Separate signed, absolute, standard-deviation, and upper-tail spatial errors
  - Relate field-specific error to target magnitude and boundary distance

Design principles:
  - Learned fields and units come from current artifact TaskSpec provenance
  - Local relative error is ``abs(pred-reference) / reference_RMS`` per case/field
  - Fields with incompatible physical units are never aggregated into one map
  - Every case read follows saved artifact membership and an explicit prefix limit

This module does NOT:
  - Parse NPZ payloads or reconstruct the authoritative primary aggregate
  - Combine learned fields whose physical units are incompatible
===============================================================================
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_LOCAL_DENOMINATOR_FLOOR = 1e-12
_DEFAULT_CASE_LIMIT = 64
_TARGET_BIN_COUNT = 12
_BOUNDARY_BIN_COUNT = 10


def _cdf(axis: Axes, values: np.ndarray, *, label: str) -> None:
    """Plot one empirical CDF without changing metric scale."""
    finite = np.asarray(values, dtype=float)
    if finite.size == 0 or not np.isfinite(finite).all():
        msg = f"CDF values for {label!r} must be finite and non-empty."
        raise ValueError(msg)
    ordered = np.sort(finite)
    probability = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    axis.step(ordered, probability, where="post", label=label)


def _bounded_cases(frame: pd.DataFrame, max_cases: int) -> list[cases.EvaluationCase]:
    """Load a disclosed deterministic artifact prefix."""
    return list(cases.iter_cases(frame, max_cases=min(max_cases, len(frame))))


def plot_predictive_error_distributions(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Plot per-case and local predictive-error empirical distributions.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible artifact frames with shared learned fields/units.
    max_cases : int, optional
        Positive bound on the saved ordered prefix loaded for local grid-point
        error. Stored per-case scalar CDFs always use each complete frame.

    Returns
    -------
    matplotlib.figure.Figure
        CDF panels for normalized field RMSE, physical relative L2/H1, and local
        fieldwise relative absolute error.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If comparison provenance, saved cases, finite metrics, or the positive
        prefix contract is invalid.

    Notes
    -----
    Normalized per-case RMSE is secondary evidence, not the primary aggregate.
    Local error is ``abs(pred-reference)/(reference_RMS + 1e-12)`` for each
    case/field, so incompatible physical fields are never mixed.

    """
    dataframe.validate_comparison(datasets)
    fields = tuple(next(iter(datasets.values())).attrs["output_fields"])
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    for label, frame in datasets.items():
        role = dataframe.dataset_role(frame)
        for field in fields:
            _cdf(
                axes[0],
                pd.to_numeric(frame[f"normalized_rmse_{field}"], errors="raise").to_numpy(dtype=float),
                label=f"{label} ({role}) - {field}; n={len(frame)}",
            )
        _cdf(
            axes[1],
            pd.to_numeric(frame["rel_l2"], errors="raise").to_numpy(dtype=float),
            label=f"{label} ({role}) relative L2; n={len(frame)}",
        )
        _cdf(
            axes[1],
            pd.to_numeric(frame["rel_h1"], errors="raise").to_numpy(dtype=float),
            label=f"{label} ({role}) relative H1; n={len(frame)}",
        )

        local_values: dict[str, list[np.ndarray]] = {field: [] for field in fields}
        loaded = _bounded_cases(frame, max_cases)
        for case in loaded:
            for index, field in enumerate(fields):
                reference_rms = float(np.sqrt(np.mean(case.reference[index] ** 2)))
                local_values[field].append(np.abs(case.error[index]).ravel() / (reference_rms + _LOCAL_DENOMINATOR_FLOOR))
        for field in fields:
            _cdf(
                axes[2],
                np.concatenate(local_values[field]),
                label=f"{label} ({role}) - {field}; n={len(loaded)}",
            )

    axes[0].set_title("Per-case normalized field RMSE distributions")
    axes[0].set_xlabel("normalized_rmse_field [1] (secondary)")
    axes[1].set_title("Per-case physical relative-error distributions")
    axes[1].set_xlabel("relative error [1] (secondary)")
    axes[2].set_title("Local relative absolute error")
    axes[2].set_xlabel("abs(error)/(reference field RMS + 1e-12) [1]")
    for axis in axes:
        axis.set_ylabel("empirical cumulative probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    return figure


def _stack_errors(frame: pd.DataFrame) -> tuple[np.ndarray, cases.EvaluationCase]:
    """
    Stack every learned-field error after proving one shared grid per frame.

    Saved membership order is retained on the leading case axis. Shape or
    coordinate drift raises the comparison exception before spatial reduction.
    """
    loaded = list(cases.iter_cases(frame))
    first = loaded[0]
    for case in loaded[1:]:
        if case.error.shape != first.error.shape or not np.allclose(case.coordinates, first.coordinates):
            msg = "Spatial error maps require identical grids within an artifact dataset."
            raise dataframe.ComparisonCompatibilityError(msg)
    return np.stack([case.error for case in loaded], axis=0), first


def _plot_map(
    axis: Axes,
    values: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    title: str,
    unit: str,
    coordinate_units: tuple[str, str],
    signed: bool,
    absolute_limit: float,
) -> None:
    """
    Render one physical-unit error map with explicit signed scale semantics.

    Signed values use a symmetric ``[-absolute_limit, absolute_limit]`` diverging
    scale; non-negative statistics use ``[0, absolute_limit]``. Coordinates and
    their units are passed through unchanged.
    """
    if signed:
        image = axis.imshow(
            values,
            cmap="coolwarm",
            vmin=-absolute_limit,
            vmax=absolute_limit,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
    else:
        image = axis.imshow(
            values,
            cmap="magma",
            vmin=0.0,
            vmax=absolute_limit,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(f"x [{coordinate_units[0]}]")
    axis.set_ylabel(f"y [{coordinate_units[1]}]")
    axis.figure.colorbar(image, ax=axis, label=unit, fraction=0.046)


def plot_error_maps(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot four spatial error reductions for every dataset and learned field.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable frames whose cases share a grid within each artifact.

    Returns
    -------
    matplotlib.figure.Figure
        Rows for dataset/field pairs and columns for mean signed error, mean
        absolute error, signed-error standard deviation, and q90 absolute error.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If artifact comparison or case/grid validation fails.

    Notes
    -----
    All maps retain the field's physical unit. A row shares one absolute scale;
    the signed mean uses symmetric limits while the other columns are non-negative.

    """
    dataframe.validate_comparison(datasets)
    row_specs = [(label, field, frame) for label, frame in datasets.items() for field in tuple(frame.attrs["output_fields"])]
    figure, axes = plt.subplots(
        len(row_specs),
        4,
        figsize=(18, max(3.2 * len(row_specs), 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    stack_cache: dict[int, tuple[np.ndarray, cases.EvaluationCase]] = {}
    for row_index, (label, field, frame) in enumerate(row_specs):
        cache_key = id(frame)
        if cache_key not in stack_cache:
            stack_cache[cache_key] = _stack_errors(frame)
        error_stack, first = stack_cache[cache_key]
        field_index = first.fields.index(field)
        values = error_stack[:, field_index]
        signed_mean = np.mean(values, axis=0)
        absolute_mean = np.mean(np.abs(values), axis=0)
        signed_std = np.std(values, axis=0)
        absolute_q90 = np.quantile(np.abs(values), 0.9, axis=0)
        limit = float(
            max(
                np.nanmax(np.abs(signed_mean)),
                np.nanmax(absolute_mean),
                np.nanmax(signed_std),
                np.nanmax(absolute_q90),
                np.finfo(float).eps,
            )
        )
        extent = cases.grid_extent(first)
        unit = first.field_units[field]
        prefix = f"{label} ({dataframe.dataset_role(frame)}), {field}, n={len(frame)}"
        _plot_map(
            axes[row_index, 0],
            signed_mean,
            extent=extent,
            title=f"{prefix}\nmean signed error",
            unit=unit,
            coordinate_units=first.coordinate_units,
            signed=True,
            absolute_limit=limit,
        )
        _plot_map(
            axes[row_index, 1],
            absolute_mean,
            extent=extent,
            title=f"{prefix}\nmean absolute error",
            unit=unit,
            coordinate_units=first.coordinate_units,
            signed=False,
            absolute_limit=limit,
        )
        _plot_map(
            axes[row_index, 2],
            signed_std,
            extent=extent,
            title=f"{prefix}\nstandard deviation of signed error",
            unit=unit,
            coordinate_units=first.coordinate_units,
            signed=False,
            absolute_limit=limit,
        )
        _plot_map(
            axes[row_index, 3],
            absolute_q90,
            extent=extent,
            title=f"{prefix}\nq90 absolute error",
            unit=unit,
            coordinate_units=first.coordinate_units,
            signed=False,
            absolute_limit=limit,
        )
    figure.suptitle("Spatial predictive-error maps")
    return figure


def plot_mean_spatial_fields(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot spatial mean reference, prediction, and signed bias for every field.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable schema-4 frames. Cases must share coordinates and learned-
        field order within each artifact.

    Returns
    -------
    matplotlib.figure.Figure
        One row per dataset/field. Mean reference and prediction share a physical
        scale; prediction-minus-reference bias uses a symmetric field-unit scale.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If comparison or saved case/grid contracts fail.

    Notes
    -----
    Means are pointwise across the complete persisted membership; no case prefix
    or cross-field reduction is applied.

    """
    dataframe.validate_comparison(datasets)
    row_specs = [(label, field, frame) for label, frame in datasets.items() for field in tuple(frame.attrs["output_fields"])]
    figure, axes = plt.subplots(
        len(row_specs),
        3,
        figsize=(14, max(3.2 * len(row_specs), 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (label, field, frame) in enumerate(row_specs):
        loaded = list(cases.iter_cases(frame))
        first = loaded[0]
        for case in loaded[1:]:
            if case.prediction.shape != first.prediction.shape or not np.allclose(case.coordinates, first.coordinates):
                msg = "Mean spatial fields require identical grids within an artifact dataset."
                raise dataframe.ComparisonCompatibilityError(msg)
        field_index = first.fields.index(field)
        reference_mean = np.mean(np.stack([case.reference[field_index] for case in loaded]), axis=0)
        prediction_mean = np.mean(np.stack([case.prediction[field_index] for case in loaded]), axis=0)
        signed_bias = prediction_mean - reference_mean
        field_low = float(min(np.min(reference_mean), np.min(prediction_mean)))
        field_high = float(max(np.max(reference_mean), np.max(prediction_mean)))
        if np.isclose(field_low, field_high):
            field_high = field_low + np.finfo(float).eps
        bias_limit = float(max(np.max(np.abs(signed_bias)), np.finfo(float).eps))
        extent = cases.grid_extent(first)
        unit = first.field_units[field]
        prefix = f"{label} ({dataframe.dataset_role(frame)}), {field}, n={len(frame)}"
        for column, values, title in (
            (0, reference_mean, "mean reference"),
            (1, prediction_mean, "mean prediction"),
        ):
            image = axes[row_index, column].imshow(
                values,
                cmap="viridis",
                vmin=field_low,
                vmax=field_high,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            axes[row_index, column].set_title(f"{prefix}\n{title}", fontsize=9)
            axes[row_index, column].set_xlabel(f"x [{first.coordinate_units[0]}]")
            axes[row_index, column].set_ylabel(f"y [{first.coordinate_units[1]}]")
            figure.colorbar(image, ax=axes[row_index, column], label=unit, fraction=0.046)
        _plot_map(
            axes[row_index, 2],
            signed_bias,
            extent=extent,
            title=f"{prefix}\nmean prediction - mean reference",
            unit=unit,
            coordinate_units=first.coordinate_units,
            signed=True,
            absolute_limit=bias_limit,
        )
    figure.suptitle("Spatial mean reference, prediction, and systematic bias")
    return figure


def plot_mean_field_bias(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Compare casewise spatial means of prediction and reference by learned field.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable frames with shared learned fields and physical units.

    Returns
    -------
    matplotlib.figure.Figure
        One scatter panel per field with one point per persisted case and an
        identity line in that field's physical unit.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If artifact comparison or case loading fails.

    Notes
    -----
    Deviation from the identity line exposes signed field-mean bias; fields are
    never combined across incompatible units.

    """
    dataframe.validate_comparison(datasets)
    fields = tuple(next(iter(datasets.values())).attrs["output_fields"])
    figure, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4.5), squeeze=False, constrained_layout=True)
    units = dataframe.field_units(next(iter(datasets.values())))
    for field_index, field in enumerate(fields):
        axis = axes[0, field_index]
        all_values: list[float] = []
        for label, frame in datasets.items():
            reference_mean = []
            prediction_mean = []
            for case in cases.iter_cases(frame):
                reference_mean.append(float(np.mean(case.reference[field_index])))
                prediction_mean.append(float(np.mean(case.prediction[field_index])))
            axis.scatter(reference_mean, prediction_mean, s=22, alpha=0.75, label=f"{label} ({dataframe.dataset_role(frame)}), n={len(frame)}")
            all_values.extend(reference_mean)
            all_values.extend(prediction_mean)
        low, high = min(all_values), max(all_values)
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        axis.set_title(field)
        axis.set_xlabel(f"reference mean [{units[field]}]")
        axis.set_ylabel(f"prediction mean [{units[field]}]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Mean-field bias")
    return figure


def _binned_median(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce paired grid-point values into non-empty equal-width median bins.

    Returned counts are exact selected point counts. A constant explanatory axis
    collapses to one bin rather than manufacturing an artificial range.
    """
    if x_values.size != y_values.size or x_values.size == 0:
        msg = "Binned error trend requires matching non-empty values."
        raise ValueError(msg)
    low, high = float(np.min(x_values)), float(np.max(x_values))
    if math.isclose(low, high):
        return np.asarray([low]), np.asarray([float(np.median(y_values))]), np.asarray([x_values.size])
    edges = np.linspace(low, high, bins + 1)
    assignments = np.clip(np.digitize(x_values, edges) - 1, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for index in range(bins):
        selected = y_values[assignments == index]
        counts[index] = selected.size
        if selected.size:
            medians[index] = float(np.median(selected))
    valid = counts > 0
    return centers[valid], medians[valid], counts[valid]


def plot_error_vs_target_magnitude(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Relate absolute physical error to absolute target magnitude by field.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable artifact frames with shared learned fields/units.
    max_cases : int, optional
        Positive bound on the saved ordered prefix loaded from each frame.

    Returns
    -------
    matplotlib.figure.Figure
        Per-field equal-width target-magnitude bins with median absolute error and
        exact sampled grid-point counts annotated.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If comparison, case loading, prefix, or binned-value contracts fail.

    """
    dataframe.validate_comparison(datasets)
    fields = tuple(next(iter(datasets.values())).attrs["output_fields"])
    units = dataframe.field_units(next(iter(datasets.values())))
    figure, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 4.8), squeeze=False, constrained_layout=True)
    for field_index, field in enumerate(fields):
        axis = axes[0, field_index]
        for label, frame in datasets.items():
            loaded = _bounded_cases(frame, max_cases)
            target = np.concatenate([np.abs(case.reference[field_index]).ravel() for case in loaded])
            error = np.concatenate([np.abs(case.error[field_index]).ravel() for case in loaded])
            centers, medians, counts = _binned_median(target, error, bins=_TARGET_BIN_COUNT)
            axis.plot(centers, medians, marker="o", label=f"{label} ({dataframe.dataset_role(frame)}); n={len(loaded)}")
            for x_value, y_value, count in zip(centers, medians, counts, strict=True):
                axis.annotate(str(int(count)), (x_value, y_value), xytext=(2, 3), textcoords="offset points", fontsize=7)
        axis.set_title(field)
        axis.set_xlabel(f"absolute target [{units[field]}]")
        axis.set_ylabel(f"median absolute error [{units[field]}]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Error versus target magnitude; annotations are sampled grid-point counts")
    return figure


def _boundary_distance(case: cases.EvaluationCase) -> np.ndarray:
    """
    Compute physical distance to the nearest edge of a rectangular domain.

    Distances use coordinate extrema along both axes; the helper does not infer
    curved, masked, or internal boundaries.
    """
    x_values, y_values = case.coordinates
    return np.minimum.reduce(
        (
            x_values - np.nanmin(x_values),
            np.nanmax(x_values) - x_values,
            y_values - np.nanmin(y_values),
            np.nanmax(y_values) - y_values,
        )
    )


def plot_boundary_error_decomposition(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Relate absolute error to physical rectangular-boundary distance by field.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable artifact frames with x/y coordinates in a common unit.
    max_cases : int, optional
        Positive bound on the saved ordered prefix loaded per frame.

    Returns
    -------
    matplotlib.figure.Figure
        Per-field equal-width distance bins with median absolute error and exact
        sampled grid-point counts annotated.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If comparison/case loading fails, coordinate units differ, or a positive
        prefix and valid bins cannot be established.

    Notes
    -----
    Boundary means the closest x/y extent edge; internal or non-rectangular
    boundaries are deliberately outside this diagnostic.

    """
    dataframe.validate_comparison(datasets)
    fields = tuple(next(iter(datasets.values())).attrs["output_fields"])
    units = dataframe.field_units(next(iter(datasets.values())))
    figure, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 4.8), squeeze=False, constrained_layout=True)
    for field_index, field in enumerate(fields):
        axis = axes[0, field_index]
        for label, frame in datasets.items():
            loaded = _bounded_cases(frame, max_cases)
            distance = np.concatenate([_boundary_distance(case).ravel() for case in loaded])
            error = np.concatenate([np.abs(case.error[field_index]).ravel() for case in loaded])
            centers, medians, counts = _binned_median(distance, error, bins=_BOUNDARY_BIN_COUNT)
            if loaded[0].coordinate_units[0] != loaded[0].coordinate_units[1]:
                msg = "Boundary distance requires x and y coordinates with the same physical unit."
                raise dataframe.ComparisonCompatibilityError(msg)
            coordinate_unit = loaded[0].coordinate_units[0]
            axis.plot(centers, medians, marker="o", label=f"{label} ({dataframe.dataset_role(frame)}); n={len(loaded)}")
            for x_value, y_value, count in zip(centers, medians, counts, strict=True):
                axis.annotate(str(int(count)), (x_value, y_value), xytext=(2, 3), textcoords="offset points", fontsize=7)
        axis.set_title(field)
        axis.set_xlabel(f"distance to closest rectangular boundary [{coordinate_unit}]")
        axis.set_ylabel(f"median absolute error [{units[field]}]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Boundary-distance error decomposition; annotations are sampled grid-point counts")
    return figure
