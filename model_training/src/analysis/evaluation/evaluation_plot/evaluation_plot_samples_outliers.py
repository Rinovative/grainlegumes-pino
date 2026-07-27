"""
===============================================================================
evaluation_plot_samples_outliers.py
===============================================================================
Link saved cases, ranked errors, and extreme inputs to task-aware field views.

Responsibilities:
  - Rank explicit predictive/physical case metrics and scalar metadata tails
  - Navigate one saved-membership position consistently across compared models
  - Plot reference, prediction, field-specific absolute error, and task inputs
  - Overlay dimensionless local-error quantiles on physical permeability fields

Design principles:
  - Ranking never accepts retired ``cont_mse`` or ``Rc`` aliases
  - Error overlays normalize one selected field by its own reference RMS
  - Permeability components remain in square metres and are never error aggregates
  - Generic task outputs remain usable when steady-flow-only inputs are unavailable

This module does NOT:
  - Parse unvalidated NPZ payloads or infer case identity from filenames
  - Admit artifact comparisons or redefine predictive and physics metrics
===============================================================================
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_TOP_K = 3


def _metric_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return explicit ranked case metrics, with no ambiguous continuity aliases."""
    fields = tuple(frame.attrs["output_fields"])
    predictive = ("rel_l2", "rel_h1", *(f"normalized_rmse_{field}" for field in fields))
    physics = tuple(metric for metric in dataframe.STEADY_PHYSICS_METRICS if metric in frame.columns)
    return (*predictive, *physics)


def available_case_metrics(frame: pd.DataFrame) -> tuple[str, ...]:
    """
    Return the explicit predictive and physical metrics valid for case ranking.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current artifact frame with TaskSpec outputs and schema-4 columns.

    Returns
    -------
    tuple[str, ...]
        Relative/per-field normalized errors followed by available named physics
        diagnostics; retired ambiguous continuity aliases are never returned.

    """
    return _metric_columns(frame)


def build_outlier_table(
    datasets: Mapping[str, pd.DataFrame],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Build linked descending top-k rows for every supported case metric.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible artifact frames.
    top_k : int, optional
        Positive number of ranks retained independently per frame and metric.

    Returns
    -------
    pandas.DataFrame
        Labels, roles, metric values, ranks, zero-based row positions, and stable
        case/source identities suitable for linked navigation.

    Raises
    ------
    ComparisonCompatibilityError, TypeError, ValueError
        If artifacts are incompatible, ``top_k`` is invalid, or metric columns
        cannot be interpreted as finite numeric values.

    Notes
    -----
    Ranks are frame-local and ties follow NumPy's deterministic argsort order;
    rows are never relabelled as a cross-model global ranking.

    """
    dataframe.validate_comparison(datasets)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        msg = "top_k must be a positive integer."
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        for metric in _metric_columns(frame):
            values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
            for rank, position in enumerate(np.argsort(values)[::-1][:top_k], start=1):
                row = frame.iloc[int(position)]
                rows.append(
                    {
                        "label": label,
                        "dataset_role": dataframe.dataset_role(frame),
                        "artifact_sample_count": len(frame),
                        "metric": metric,
                        "rank": rank,
                        "value": float(values[position]),
                        "row_position": int(position),
                        "case_index": int(row["case_index"]),
                        "source_index": int(row["source_index"]),
                    }
                )
    return pd.DataFrame(rows)


def build_input_extremes_table(
    datasets: Mapping[str, pd.DataFrame],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Build linked low/high ranks for finite scalar source metadata.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible frames with flattened metadata columns.
    top_k : int, optional
        Positive rank count retained independently in each frame and tail.

    Returns
    -------
    pandas.DataFrame
        Parameter values, tail/rank, saved row positions, and stable identities.

    Raises
    ------
    ComparisonCompatibilityError, TypeError, ValueError
        If artifacts are incompatible or ``top_k`` is invalid.

    Notes
    -----
    A metadata column containing any non-finite value is omitted explicitly.
    Low/high ranks are descriptive input extremes, not predictive-error ranks.

    """
    dataframe.validate_comparison(datasets)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        msg = "top_k must be a positive integer."
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        for parameter in dataframe.numeric_metadata_columns(frame):
            values = pd.to_numeric(frame[parameter], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                continue
            orders = {"low": np.argsort(values)[:top_k], "high": np.argsort(values)[::-1][:top_k]}
            for extreme, positions in orders.items():
                for rank, position in enumerate(positions, start=1):
                    row = frame.iloc[int(position)]
                    rows.append(
                        {
                            "label": label,
                            "dataset_role": dataframe.dataset_role(frame),
                            "artifact_sample_count": len(frame),
                            "parameter": parameter,
                            "extreme": extreme,
                            "rank": rank,
                            "value": float(values[position]),
                            "row_position": int(position),
                            "case_index": int(row["case_index"]),
                            "source_index": int(row["source_index"]),
                        }
                    )
    return pd.DataFrame(rows)


def plot_outlier_extreme_tables(*, datasets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Return the two linked ranking tables used by notebook views.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible artifact frames.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Exact ``metric_outliers`` and ``input_extremes`` tables using the module's
        default top-k rank count.

    Raises
    ------
    ComparisonCompatibilityError, TypeError, ValueError
        Propagated from metric and metadata table construction.

    """
    return {
        "metric_outliers": build_outlier_table(datasets),
        "input_extremes": build_input_extremes_table(datasets),
    }


def _image(
    axis: Axes,
    values: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    title: str,
    unit: str,
    coordinate_units: tuple[str, str],
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Render one task field with a disclosed physical unit."""
    image = axis.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel(f"x [{coordinate_units[0]}]")
    axis.set_ylabel(f"y [{coordinate_units[1]}]")
    axis.figure.colorbar(image, ax=axis, label=unit, fraction=0.046)


def _selected_positions(
    datasets: Mapping[str, pd.DataFrame],
    positions: Mapping[str, int] | None,
) -> dict[str, int]:
    """
    Validate an exact zero-based row position for every selected frame.

    Omitted positions default every label to zero. Supplied mappings must match
    dataset labels exactly; booleans, non-integers, and out-of-membership values
    fail before any NPZ case is loaded.
    """
    selected = dict.fromkeys(datasets, 0) if positions is None else dict(positions)
    if set(selected) != set(datasets):
        msg = "Sample positions must identify every dataset label exactly once."
        raise ValueError(msg)
    for label, position in selected.items():
        if isinstance(position, bool) or not isinstance(position, Integral) or not 0 <= int(position) < len(datasets[label]):
            msg = f"Sample row position for {label!r} is outside the artifact membership."
            raise IndexError(msg)
    return {label: int(position) for label, position in selected.items()}


def plot_task_aware_sample(
    *,
    datasets: Mapping[str, pd.DataFrame],
    positions: Mapping[str, int] | None = None,
) -> Figure:
    """
    Plot selected task outputs and optional steady-flow permeability context.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible artifact frames.
    positions : Mapping[str, int] | None, optional
        Exact zero-based saved row position for every label; defaults to zero.

    Returns
    -------
    matplotlib.figure.Figure
        Per-field reference, prediction, q99-clipped absolute error, and available
        physical permeability panels with units and stable case identities.

    Raises
    ------
    ComparisonCompatibilityError, IndexError, FileNotFoundError, TypeError, ValueError
        If comparison, row-position, NPZ case, field, or optional steady context
        contracts fail.

    Notes
    -----
    Reference and prediction share one physical field scale. Error clipping is
    display-only; permeability remains separate physical context in square metres.

    """
    dataframe.validate_comparison(datasets)
    selected = _selected_positions(datasets, positions)
    loaded = {label: cases.load_case(frame, selected[label]) for label, frame in datasets.items()}
    field_rows = sum(len(case.fields) for case in loaded.values())
    steady_rows = sum(case.permeability is not None for case in loaded.values())
    figure, axes = plt.subplots(
        field_rows + steady_rows,
        3,
        figsize=(13, 3.5 * (field_rows + steady_rows)),
        squeeze=False,
        constrained_layout=True,
    )
    row_index = 0
    for label, frame in datasets.items():
        case = loaded[label]
        extent = cases.grid_extent(case)
        prefix = f"{label} ({dataframe.dataset_role(frame)}), artifact n={len(frame)}, case={case.case_index}, source={case.source_index}"
        for field_index, field in enumerate(case.fields):
            reference = case.reference[field_index]
            prediction = case.prediction[field_index]
            absolute_error = np.abs(case.error[field_index])
            low = float(min(np.min(reference), np.min(prediction)))
            high = float(max(np.max(reference), np.max(prediction)))
            if np.isclose(low, high):
                high = low + np.finfo(float).eps
            error_high = float(max(np.quantile(absolute_error, 0.99), np.finfo(float).eps))
            unit = case.field_units[field]
            _image(
                axes[row_index, 0],
                reference,
                extent=extent,
                title=f"{prefix} - {field} reference",
                unit=unit,
                coordinate_units=case.coordinate_units,
                cmap="viridis",
                vmin=low,
                vmax=high,
            )
            _image(
                axes[row_index, 1],
                prediction,
                extent=extent,
                title=f"{prefix} - {field} prediction",
                unit=unit,
                coordinate_units=case.coordinate_units,
                cmap="viridis",
                vmin=low,
                vmax=high,
            )
            _image(
                axes[row_index, 2],
                np.clip(absolute_error, 0.0, error_high),
                extent=extent,
                title=f"{prefix} - {field} absolute error (clipped q99)",
                unit=unit,
                coordinate_units=case.coordinate_units,
                cmap="magma",
                vmin=0.0,
                vmax=error_high,
            )
            row_index += 1
        if case.permeability is not None:
            for column in range(3):
                if column < case.permeability.shape[0]:
                    permeability = case.permeability[column]
                    low = float(np.min(permeability))
                    high = float(max(np.max(permeability), low + np.finfo(float).eps))
                    _image(
                        axes[row_index, column],
                        permeability,
                        extent=extent,
                        title=f"{prefix} - steady context {case.permeability_names[column]}",
                        unit=case.input_field_units[case.permeability_names[column]],
                        coordinate_units=case.coordinate_units,
                        cmap="cividis",
                        vmin=low,
                        vmax=high,
                    )
                else:
                    axes[row_index, column].axis("off")
            row_index += 1
    figure.suptitle("Task-aware sample comparison; steady artifacts include permeability context")
    return figure


def plot_task_aware_sample_at_position(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
) -> Figure:
    """
    Render one shared saved-membership position across comparable datasets.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable artifact frames. A shared row position refers to the same
        saved split member within each ID or OOD comparison role.
    row_position : int, optional
        Zero-based saved-membership position applied to every selected frame.

    Returns
    -------
    matplotlib.figure.Figure
        Multi-model reference, prediction, absolute-error, and optional
        permeability context produced by :func:`plot_task_aware_sample`.

    Raises
    ------
    IndexError
        If the requested position is unavailable in any selected frame.

    """
    positions = dict.fromkeys(datasets, row_position)
    return plot_task_aware_sample(datasets=datasets, positions=positions)


def plot_permeability_error_overlay(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    field: str | None = None,
) -> Figure:
    """
    Overlay dimensionless local output error on physical permeability fields.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable steady-flow artifact frames.
    row_position : int, optional
        Shared zero-based saved-membership position across selected frames.
    field : str | None, optional
        Learned output whose absolute error is divided by that case-field's
        reference RMS. When omitted, use the first TaskSpec output.

    Returns
    -------
    matplotlib.figure.Figure
        Physical ``kxx``, ``kxy``, and ``kyy`` maps with q50/q75/q90 contours of
        ``abs(pred-reference)/(reference RMS + 1e-12)`` where distinct.

    Raises
    ------
    ComparisonCompatibilityError
        If a selected artifact lacks steady-flow permeability context.
    ValueError
        If `field` is not a learned output in every selected frame.

    Notes
    -----
    The overlay is a spatial diagnostic, not a cross-unit aggregate or score.

    """
    dataframe.validate_comparison(datasets)
    selected = _selected_positions(datasets, dict.fromkeys(datasets, row_position))
    figure, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(14, max(3.8 * len(datasets), 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (label, frame) in enumerate(datasets.items()):
        case = cases.load_case(frame, selected[label])
        if case.permeability is None or not case.permeability_names:
            msg = f"Permeability overlays require steady-flow context for {label!r}."
            raise dataframe.ComparisonCompatibilityError(msg)
        selected_field = case.fields[0] if field is None else field
        if selected_field not in case.fields:
            msg = f"Overlay field {selected_field!r} is unavailable for {label!r}: {list(case.fields)}."
            raise ValueError(msg)
        field_index = case.fields.index(selected_field)
        reference_rms = float(np.sqrt(np.mean(case.reference[field_index] ** 2)))
        local_relative = np.abs(case.error[field_index]) / (reference_rms + 1e-12)
        contour_levels = np.unique(np.quantile(local_relative, (0.5, 0.75, 0.9)))
        extent = cases.grid_extent(case)
        for column, (name, permeability) in enumerate(zip(case.permeability_names, case.permeability, strict=True)):
            low = float(np.min(permeability))
            high = float(max(np.max(permeability), low + np.finfo(float).eps))
            _image(
                axes[row_index, column],
                permeability,
                extent=extent,
                title=(
                    f"{label} ({dataframe.dataset_role(frame)}), case={case.case_index}\n{name} with {selected_field} local-relative-error contours"
                ),
                unit=case.input_field_units[name],
                coordinate_units=case.coordinate_units,
                cmap="cividis",
                vmin=low,
                vmax=high,
            )
            if contour_levels.size > 1 or (contour_levels.size == 1 and not np.isclose(contour_levels[0], 0.0)):
                contours = axes[row_index, column].contour(
                    case.coordinates[0],
                    case.coordinates[1],
                    local_relative,
                    levels=contour_levels,
                    colors="white",
                    linewidths=0.8,
                )
                axes[row_index, column].clabel(contours, inline=True, fontsize=7, fmt="%.2g")
    figure.suptitle("Permeability context with dimensionless local-error contours")
    return figure


def plot_linked_input_extreme_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    parameter: str,
    extreme: str = "high",
    rank: int = 1,
) -> Figure:
    """
    Render one ranked low/high metadata-input case per selected dataset.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Comparable artifact frames with flattened finite scalar metadata.
    parameter : str
        Exact numeric metadata column to rank independently in each frame.
    extreme : {"low", "high"}, optional
        Select ascending or descending parameter order.
    rank : int, optional
        One-based rank within the chosen tail.

    Returns
    -------
    matplotlib.figure.Figure
        Linked reference, prediction, error, and optional steady context for the
        selected input extreme in every dataset.

    Raises
    ------
    ValueError
        If the parameter/extreme/rank contract is invalid.
    IndexError
        If `rank` exceeds a selected artifact's membership.

    """
    dataframe.validate_comparison(datasets)
    if extreme not in {"low", "high"}:
        msg = "extreme must be 'low' or 'high'."
        raise ValueError(msg)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        msg = "rank must be a positive integer."
        raise ValueError(msg)
    positions: dict[str, int] = {}
    for label, frame in datasets.items():
        if parameter not in dataframe.numeric_metadata_columns(frame):
            msg = f"Extreme-input parameter {parameter!r} is unavailable for {label!r}."
            raise ValueError(msg)
        values = pd.to_numeric(frame[parameter], errors="raise").to_numpy(dtype=float)
        order = np.argsort(values) if extreme == "low" else np.argsort(values)[::-1]
        if rank > len(order):
            msg = f"Extreme-input rank {rank} exceeds the {len(order)} cases in {label!r}."
            raise IndexError(msg)
        positions[label] = int(order[rank - 1])
    return plot_task_aware_sample(datasets=datasets, positions=positions)


def plot_linked_outlier_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    metric: str = "rel_l2",
    rank: int = 1,
) -> Figure:
    """
    Render the same metric rank independently within every selected dataset.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible frames to rank and display.
    metric : str, optional
        Explicit metric returned by :func:`available_case_metrics`.
    rank : int, optional
        Positive one-based descending rank.

    Returns
    -------
    matplotlib.figure.Figure
        Task-aware field views for the selected outlier in each dataset.

    Raises
    ------
    ValueError
        If the metric or rank domain is invalid.
    IndexError
        If the requested rank exceeds any selected membership.

    """
    dataframe.validate_comparison(datasets)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        msg = "rank must be a positive integer."
        raise ValueError(msg)
    positions: dict[str, int] = {}
    for label, frame in datasets.items():
        if metric not in _metric_columns(frame):
            msg = f"Outlier metric {metric!r} is unavailable for {label!r}."
            raise ValueError(msg)
        order = np.argsort(pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float))[::-1]
        if rank > len(order):
            msg = f"Outlier rank {rank} exceeds the {len(order)} cases in {label!r}."
            raise IndexError(msg)
        positions[label] = int(order[rank - 1])
    return plot_task_aware_sample(datasets=datasets, positions=positions)
