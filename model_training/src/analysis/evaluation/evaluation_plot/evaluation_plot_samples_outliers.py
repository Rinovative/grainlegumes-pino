"""
===============================================================================
evaluation_plot_samples_outliers.py
===============================================================================
Restore case, permeability, outlier, and input-extreme presentation.

Responsibilities:
  - Rank explicit predictive and physical metrics with canonical identity ties
  - Rank supported scientific metadata independently at low and high extremes
  - Render fixed historical sample, permeability, and two-model field layouts
  - Overlay velocity streamlines on regular display-only physical coordinates
  - Build colored outlier and side-by-side metadata-extreme tables

Design principles:
  - Case selection follows exact saved membership and canonical case identity
  - TaskSpec fields and vector groups define p, u, v, and |u| presentation
  - Permeability remains physical context and is never an error aggregate
  - Scaling, clipping, contouring, and regular coordinates affect display only

This module does NOT:
  - Parse unvalidated payloads or infer case identity from filenames
  - Compare models by row position without matching identity and reference fields
  - Redefine predictive, physics, or metadata-ranking semantics
  - Own interactive notebook controls or public panel composition
===============================================================================
"""

from __future__ import annotations

from html import escape
from numbers import Integral
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib import colormaps

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas.io.formats.style import Styler

_DEFAULT_TOP_K = 5
_MINIMUM_COMPARISON_MODELS = 2
_RELATIVE_DENOMINATOR_FLOOR = 1e-12
_CONTOUR_LEVELS = 10
_MINIMUM_TABLE_NUMERIC_ROWS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_REQUIRED_DISPLAY_FIELDS = 4
_STANDARD_NUMBER_LOWER = 1e-4
_STANDARD_NUMBER_UPPER = 1e4


def _metric_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return current case metrics in stable scientific order."""
    fields = presentation.display_fields(frame)
    predictive = ("rel_l2", "rel_h1", *(field.metric_column for field in fields))
    physics = tuple(metric for metric in dataframe.STEADY_PHYSICS_METRICS if metric in frame.columns)
    return tuple(dict.fromkeys(metric for metric in (*predictive, *physics) if metric in frame.columns))


def available_case_metrics(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return explicit current metrics valid for deterministic case ranking."""
    return _metric_columns(frame)


def _rank_positions(frame: pd.DataFrame, values: np.ndarray, *, descending: bool) -> np.ndarray:
    """Return deterministic value order with canonical case/source tie breaking."""
    numeric = np.asarray(values, dtype=float)
    if numeric.shape != (len(frame),) or not np.isfinite(numeric).all():
        msg = "Ranked values must be a finite vector aligned with artifact membership."
        raise ValueError(msg)
    primary = -numeric if descending else numeric
    return np.lexsort(
        (
            frame["source_index"].to_numpy(dtype=np.int64),
            frame["case_index"].to_numpy(dtype=np.int64),
            primary,
        )
    )


def build_outlier_table(
    datasets: Mapping[str, pd.DataFrame],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Build current metric-ranked identities for programmatic linked use."""
    dataframe.validate_comparison(datasets)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        msg = "top_k must be a positive integer."
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        for metric in _metric_columns(frame):
            values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
            for rank, position in enumerate(_rank_positions(frame, values, descending=True)[:top_k], start=1):
                source = frame.iloc[int(position)]
                rows.append(
                    {
                        "label": label,
                        "metric": metric,
                        "rank": rank,
                        "value": float(values[position]),
                        "row_position": int(position),
                        "case_index": int(source["case_index"]),
                        "source_index": int(source["source_index"]),
                    }
                )
    return pd.DataFrame(rows)


def build_input_extremes_table(
    datasets: Mapping[str, pd.DataFrame],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Build current metadata low/high identities for programmatic linked use."""
    dataframe.validate_comparison(datasets)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        msg = "top_k must be a positive integer."
        raise ValueError(msg)
    parameters = presentation.metadata_parameters(tuple(datasets.values()))
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        for parameter in parameters:
            values = pd.to_numeric(frame[parameter], errors="raise").to_numpy(dtype=float)
            for extreme, order in (
                ("low", _rank_positions(frame, values, descending=False)),
                ("high", _rank_positions(frame, values, descending=True)),
            ):
                for rank, position in enumerate(order[:top_k], start=1):
                    source = frame.iloc[int(position)]
                    rows.append(
                        {
                            "label": label,
                            "parameter": parameter,
                            "extreme": extreme,
                            "rank": rank,
                            "value": float(values[position]),
                            "row_position": int(position),
                            "case_index": int(source["case_index"]),
                            "source_index": int(source["source_index"]),
                        }
                    )
    return pd.DataFrame(rows)


def _fmt_number(value: Any) -> str:
    """Format one table value with the compact historical numerical style."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.3g}" if _STANDARD_NUMBER_LOWER <= abs(number) < _STANDARD_NUMBER_UPPER else f"{number:.2e}"


def _rank_style(table: pd.DataFrame, *, reference_row: int) -> pd.DataFrame:
    """Return historical blue/purple cell fills diverging from the reference."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    blue = colormaps["Blues"]
    purple = colormaps["Purples"]
    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        if numeric.notna().sum() < _MINIMUM_TABLE_NUMERIC_ROWS or pd.isna(numeric.iloc[reference_row]):
            continue
        reference = float(numeric.iloc[reference_row])
        lower = numeric < reference
        higher = numeric > reference
        for mask, cmap in ((lower, blue), (higher, purple)):
            selected = numeric[mask]
            if selected.empty:
                continue
            ranks = selected.rank(method="average", pct=True)
            for index, rank in ranks.items():
                red, green, blue_value, _alpha = cmap(0.25 + 0.5 * float(rank))
                styles.loc[index, column] = f"background-color: rgba({int(red * 255)}, {int(green * 255)}, {int(blue_value * 255)}, 0.65)"
    styles.iloc[reference_row] = "background-color: white"
    return styles


def _channel_outlier_styler(frame: pd.DataFrame, field: presentation.DisplayField, *, top_k: int) -> Styler:
    """Build one historical worst-five-plus-reference channel table."""
    values = pd.to_numeric(frame[field.metric_column], errors="raise").to_numpy(dtype=float)
    positions = list(_rank_positions(frame, values, descending=True)[:top_k])
    reference_position = presentation.reference_case_position(frame)
    parameters = presentation.metadata_parameters((frame,))
    rows: list[dict[str, Any]] = []
    for position in (*positions, reference_position):
        source = frame.iloc[int(position)]
        is_reference = int(position) == reference_position
        row: dict[str, Any] = {
            "Case": f"{int(source['case_index'])} (Ref)" if is_reference else int(source["case_index"]),
            f"normalized RMSE [{field.label}]": float(source[field.metric_column]),
            "relative L2": float(source["rel_l2"]),
        }
        row.update({presentation.metadata_label(column): float(source[column]) for column in parameters})
        rows.append(row)
    table = pd.DataFrame(rows)
    reference_row = len(table) - 1
    return table.style.format(_fmt_number).apply(_rank_style, axis=None, reference_row=reference_row)


def plot_outlier_table(
    *,
    datasets: Mapping[str, pd.DataFrame],
    top_k: int = _DEFAULT_TOP_K,
) -> widgets.VBox:
    """Return automatically rendered colored channel outlier tables."""
    dataframe.validate_comparison(datasets)
    if len(datasets) != 1:
        msg = "The historical outlier-table view displays one selected dataset."
        raise ValueError(msg)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        msg = "top_k must be a positive integer."
        raise ValueError(msg)
    label, frame = next(iter(datasets.items()))
    title = layout.case_title("Worst per-channel cases", case_count=len(frame))
    children: list[widgets.Widget] = [widgets.HTML(f"<h2>{title}</h2>"), widgets.HTML(f"<h3>{escape(label)}</h3>")]
    for field in presentation.display_fields(frame):
        styler = _channel_outlier_styler(frame, field, top_k=top_k)
        children.append(widgets.HTML(f"<h4>Channel {escape(field.label)}</h4>{styler.to_html()}"))
    return widgets.VBox(children)


def plot_input_extremes_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Return historical side-by-side min/reference/max metadata tables."""
    dataframe.validate_comparison(datasets)
    parameters = presentation.metadata_parameters(tuple(datasets.values()))
    if not parameters:
        msg = "No supported metadata parameters are available."
        raise ValueError(msg)
    title, count_headings = layout.aggregate_title_context(
        "Extreme input parameters",
        layout.effective_case_counts(datasets),
    )
    tables: list[widgets.Widget] = []
    for label, frame in datasets.items():
        reference = frame.iloc[presentation.reference_case_position(frame, parameters)]
        rows = []
        for parameter in parameters:
            values = pd.to_numeric(frame[parameter], errors="raise").to_numpy(dtype=float)
            reference_value = float(reference[parameter])
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            rows.append(
                {
                    "Parameter": presentation.metadata_label(parameter),
                    "Min": minimum,
                    "Reference": reference_value,
                    "Max": maximum,
                    "Min / ref": minimum / reference_value if reference_value != 0.0 else np.nan,
                    "Max / ref": maximum / reference_value if reference_value != 0.0 else np.nan,
                }
            )
        table = pd.DataFrame(rows).set_index("Parameter")
        heading = count_headings[label] or label
        tables.append(widgets.HTML(f"<h3>{escape(heading)}</h3>{table.style.format(_fmt_number).to_html()}"))
    return widgets.VBox((widgets.HTML(f"<h2>{title}</h2>"), widgets.HBox(tables)))


def plot_outlier_extreme_tables(*, datasets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return programmatic current ranking tables without owning panel composition."""
    return {
        "metric_outliers": build_outlier_table(datasets),
        "input_extremes": build_input_extremes_table(datasets),
    }


def _selected_position(frame: pd.DataFrame, row_position: int) -> int:
    """Validate one zero-based saved-membership position."""
    if isinstance(row_position, bool) or not isinstance(row_position, Integral):
        msg = "row_position must be an integer."
        raise TypeError(msg)
    position = int(row_position)
    if not 0 <= position < len(frame):
        msg = "row_position is outside the artifact membership."
        raise IndexError(msg)
    return position


def _single_dataset(datasets: Mapping[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Return the one dataset selected by a historical local dropdown."""
    dataframe.validate_comparison(datasets)
    if len(datasets) != 1:
        msg = "This historical case view requires one selected dataset."
        raise ValueError(msg)
    return next(iter(datasets.items()))


def _levels(values: np.ndarray, *, count: int = _CONTOUR_LEVELS, nonnegative: bool = False) -> np.ndarray:
    """Return finite stable contour levels without changing field values."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        msg = "A displayed field contains no finite values."
        raise ValueError(msg)
    low = 0.0 if nonnegative else float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high):
        delta = max(abs(low), 1.0) * np.finfo(float).eps * 16.0
        low = 0.0 if nonnegative else low - delta
        high += delta
    return np.linspace(low, high, count)


def _format_axes(axis: Axes, case: cases.EvaluationCase) -> None:
    """Apply exact physical extents and geometry without changing labels."""
    x_values, y_values = case.coordinates
    axis.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
    axis.set_ylim(float(np.min(y_values)), float(np.max(y_values)))
    axis.set_aspect("equal", adjustable="box")


def _clip_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ma.MaskedArray:
    """Saturate finite out-of-range values without creating white masked bands."""
    return np.ma.clip(np.ma.asarray(values), float(levels[0]), float(levels[-1]))


def _contour(
    figure: Figure,
    axis: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    *,
    levels: np.ndarray,
    cmap: str,
    title: str,
    case: cases.EvaluationCase,
) -> Any:
    """Draw one historical filled-contour panel and colorbar."""
    displayed = _clip_to_levels(values, levels)
    image = axis.contourf(x_grid, y_grid, displayed, levels=levels, cmap=cmap)
    axis.set_title(title)
    layout.add_map_colorbar(figure, image, axis)
    _format_axes(axis, case)
    return image


def _velocity_indices(frame: pd.DataFrame, case: cases.EvaluationCase) -> tuple[int, int]:
    """Return the exact first two TaskSpec velocity-component indices."""
    fields = dataframe.output_group_fields(frame, group_id="velocity")
    if len(fields) < _MINIMUM_VELOCITY_COMPONENTS or any(field not in case.fields for field in fields[:_MINIMUM_VELOCITY_COMPONENTS]):
        msg = "Velocity streamlines require two declared components."
        raise dataframe.ComparisonCompatibilityError(msg)
    return case.fields.index(fields[0]), case.fields.index(fields[1])


def _overlay_streamlines(axis: Axes, frame: pd.DataFrame, case: cases.EvaluationCase, values: np.ndarray) -> None:
    """Overlay velocity using authoritative equally spaced display coordinates."""
    first, second = _velocity_indices(frame, case)
    x, y, _x_grid, _y_grid = presentation.display_grid(case)
    axis.streamplot(
        x,
        y,
        values[first],
        values[second],
        color="white",
        density=0.65,
        linewidth=0.55,
        arrowsize=0.55,
    )


def _aggregate_diagonal_permeability(case: cases.EvaluationCase) -> np.ndarray:
    """Return the historical diagonal-only permeability display aggregate."""
    if case.permeability is None:
        msg = "Permeability context is unavailable."
        raise dataframe.ComparisonCompatibilityError(msg)
    by_name = dict(zip(case.permeability_names, case.permeability, strict=True))
    diagonals = [by_name[name] for name in ("kxx", "kyy", "kzz") if name in by_name]
    if not diagonals:
        msg = "No diagonal permeability components are available."
        raise dataframe.ComparisonCompatibilityError(msg)
    return np.mean(np.stack(diagonals), axis=0)


def _normalize_error_mode(error_mode: str) -> str:
    """Normalize the two historical labels and direct-call semantic aliases."""
    aliases = {"MAE": "MAE", "absolute": "MAE", "Relative [%]": "Relative [%]", "local_relative": "Relative [%]"}
    try:
        return aliases[error_mode]
    except KeyError as error:
        msg = "error_mode must be 'MAE' or 'Relative [%]'."
        raise ValueError(msg) from error


def _normalize_scale_mode(scale_mode: str) -> str:
    """Normalize the two historical prediction/reference scale labels."""
    aliases = {
        "Independent": "Independent",
        "independent": "Independent",
        "Shared (GT)": "Shared (GT)",
        "shared": "Shared (GT)",
    }
    try:
        return aliases[scale_mode]
    except KeyError as error:
        msg = "scale_mode must be 'Independent' or 'Shared (GT)'."
        raise ValueError(msg) from error


def _error_values(reference: np.ndarray, prediction: np.ndarray, *, error_mode: str) -> tuple[np.ndarray, str]:
    """Return MAE or current field-RMS-relative percentage evidence."""
    absolute = np.abs(prediction - reference)
    if _normalize_error_mode(error_mode) == "MAE":
        return absolute, "MAE"
    denominator = float(np.sqrt(np.mean(reference**2))) + _RELATIVE_DENOMINATOR_FLOOR
    return absolute / denominator * 100.0, "relative error [% of reference RMS]"


def _resolve_display_field(frame: pd.DataFrame, field: str | None) -> presentation.DisplayField:
    """Resolve one canonical key or concise display label."""
    available = presentation.display_fields(frame)
    if field is None:
        return available[-1]
    for candidate in available:
        aliases = {candidate.key, candidate.label}
        if candidate.key == "velocity_magnitude":
            aliases.add("U")
        if field in aliases:
            return candidate
    msg = f"Unknown display field {field!r}; expected {[item.label for item in available]}."
    raise ValueError(msg)


def _plot_prediction_overview(
    *,
    frame: pd.DataFrame,
    row_position: int,
    error_mode: str,
    scale_mode: str,
    title: str,
) -> Figure:
    """Render the restored fixed 4 x 4 prediction/reference/error/input figure."""
    position = _selected_position(frame, row_position)
    case = cases.load_case(frame, position)
    fields = presentation.display_fields(frame)
    if len(fields) != _REQUIRED_DISPLAY_FIELDS:
        msg = "The restored overview requires four supported display fields."
        raise dataframe.ComparisonCompatibilityError(msg)
    scale = _normalize_scale_mode(scale_mode)
    error = _normalize_error_mode(error_mode)
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    figure, axes = layout.map_subplots(rows=4, columns=4)
    permeability = _aggregate_diagonal_permeability(case)
    positive = permeability[permeability > 0.0]
    if positive.size == 0:
        msg = "Permeability must contain positive values for logarithmic display."
        raise ValueError(msg)
    permeability_log = np.log10(np.maximum(permeability, float(np.min(positive))))

    for row, field in enumerate(fields):
        reference = presentation.case_field(case, field, source="reference")
        prediction = presentation.case_field(case, field, source="prediction")
        if scale == "Shared (GT)":
            reference_levels = _levels(reference)
            prediction_levels = reference_levels
            prediction_plot = prediction
        else:
            reference_levels = _levels(reference)
            prediction_levels = _levels(prediction)
            prediction_plot = np.ma.asarray(prediction)
        _contour(
            figure,
            axes[row, 0],
            x_grid,
            y_grid,
            prediction_plot,
            levels=prediction_levels,
            cmap="turbo",
            title=f"{field.matplotlib_label} pred [{field.unit}]",
            case=case,
        )
        _contour(
            figure,
            axes[row, 1],
            x_grid,
            y_grid,
            reference,
            levels=reference_levels,
            cmap="turbo",
            title=f"{field.matplotlib_label} true [{field.unit}]",
            case=case,
        )
        if field.key in {"u", "v", "velocity_magnitude"}:
            _overlay_streamlines(axes[row, 0], frame, case, case.prediction)
            _overlay_streamlines(axes[row, 1], frame, case, case.reference)
        displayed_error, error_label = _error_values(reference, prediction, error_mode=error)
        upper = max(float(np.quantile(displayed_error, 0.99)), np.finfo(float).eps)
        clipped = np.minimum(displayed_error, upper)
        error_unit = field.unit if error == "MAE" else "%"
        _contour(
            figure,
            axes[row, 2],
            x_grid,
            y_grid,
            clipped,
            levels=_levels(clipped, nonnegative=True),
            cmap="Blues",
            title=f"{field.matplotlib_label} {error_label} [{error_unit}]",
            case=case,
        )
        if row == 0:
            _contour(
                figure,
                axes[row, 3],
                x_grid,
                y_grid,
                permeability,
                levels=_levels(permeability),
                cmap="viridis",
                title="kappa [m²]",
                case=case,
            )
        elif row == 1:
            _contour(
                figure,
                axes[row, 3],
                x_grid,
                y_grid,
                permeability_log,
                levels=_levels(permeability_log),
                cmap="viridis",
                title="log10(kappa / 1 m²)",
                case=case,
            )
        else:
            axes[row, 3].axis("off")
    x_label = f"x [{case.coordinate_units[0]}]"
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=x_label,
        y_label=f"y [{case.coordinate_units[1]}]",
    )
    figure.suptitle(layout.case_title(title, case_number=case.case_index), fontsize=14)
    layout.add_shortened_column_x_decorations(axes, x_label=x_label)
    return figure


def plot_sample_prediction_overview(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    error_mode: str = "MAE",
    scale_mode: str = "Independent",
) -> Figure:
    """Plot the restored 4 x 4 selected-case prediction overview."""
    _label, frame = _single_dataset(datasets)
    return _plot_prediction_overview(
        frame=frame,
        row_position=row_position,
        error_mode=error_mode,
        scale_mode=scale_mode,
        title="Sample GT vs prediction",
    )


def plot_permeability_error_overlay(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    channel: str = "U",
    kappa_scale: str = "log10(kappa)",
    error_mode: str = "MAE",
) -> Figure:
    """Plot the restored 2 x 3 permeability tensor, target, and error view."""
    _label, frame = _single_dataset(datasets)
    position = _selected_position(frame, row_position)
    case = cases.load_case(frame, position)
    if case.permeability is None:
        msg = "Permeability context is unavailable."
        raise dataframe.ComparisonCompatibilityError(msg)
    selected = _resolve_display_field(frame, channel)
    reference = presentation.case_field(case, selected, source="reference")
    prediction = presentation.case_field(case, selected, source="prediction")
    displayed_error, error_label = _error_values(reference, prediction, error_mode=error_mode)
    upper = max(float(np.quantile(displayed_error, 0.99)), np.finfo(float).eps)
    displayed_error = np.minimum(displayed_error, upper)
    contour_levels = np.unique(np.quantile(displayed_error, (0.75, 0.95)))
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    by_name = dict(zip(case.permeability_names, case.permeability, strict=True))
    if not {"kxx", "kxy", "kyy"}.issubset(by_name):
        msg = "The tensor overlay requires kxx, kxy, and kyy."
        raise dataframe.ComparisonCompatibilityError(msg)
    components = (("kxx", by_name["kxx"]), ("kxy", by_name["kxy"]), ("kyx", by_name["kxy"]), ("kyy", by_name["kyy"]))
    if kappa_scale not in {"kappa", "log10(kappa)"}:
        msg = "kappa_scale must be 'kappa' or 'log10(kappa)'."
        raise ValueError(msg)
    figure, axes = layout.map_subplots(rows=2, columns=3)
    for index, (name, values) in enumerate(components):
        row, column = divmod(index, 2)
        shown = values
        title = f"{name} [m²]"
        if kappa_scale == "log10(kappa)" and name in {"kxx", "kyy"}:
            positive = values[values > 0.0]
            if positive.size == 0:
                msg = f"{name} must be positive for logarithmic display."
                raise ValueError(msg)
            shown = np.log10(np.maximum(values, float(np.min(positive))))
            title = f"log10({name} / 1 m²)"
        _contour(
            figure,
            axes[row, column],
            x_grid,
            y_grid,
            shown,
            levels=_levels(shown, count=11),
            cmap="viridis",
            title=title,
            case=case,
        )
        if contour_levels.size:
            axes[row, column].contour(x_grid, y_grid, displayed_error, levels=contour_levels, colors="red", linewidths=1.0)
    _contour(
        figure,
        axes[0, 2],
        x_grid,
        y_grid,
        reference,
        levels=_levels(reference, count=11),
        cmap="turbo",
        title=f"{selected.matplotlib_label} true [{selected.unit}]",
        case=case,
    )
    if contour_levels.size:
        axes[0, 2].contour(x_grid, y_grid, displayed_error, levels=contour_levels, colors="red", linewidths=1.0)
    _contour(
        figure,
        axes[1, 2],
        x_grid,
        y_grid,
        displayed_error,
        levels=_levels(displayed_error, count=11, nonnegative=True),
        cmap="Reds",
        title=f"{selected.matplotlib_label} {error_label}",
        case=case,
    )
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{case.coordinate_units[0]}]",
        y_label=f"y [{case.coordinate_units[1]}]",
    )
    figure.suptitle(layout.case_title("Kappa tensor with error overlay", case_number=case.case_index), fontsize=14)
    return figure


def plot_pressure_velocity_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    model_1: str | None = None,
    model_2: str | None = None,
    scale_mode: str = "Independent",
) -> Figure:
    """Plot restored 3 x 2 pressure/speed truth and two-model predictions."""
    dataframe.validate_comparison(datasets)
    if len(datasets) < _MINIMUM_COMPARISON_MODELS:
        msg = "Pressure/velocity comparison requires at least two models."
        raise ValueError(msg)
    labels = tuple(datasets)
    first_label = labels[0] if model_1 is None else model_1
    second_label = labels[1] if model_2 is None else model_2
    if first_label not in datasets or second_label not in datasets or first_label == second_label:
        msg = "model_1 and model_2 must name two distinct displayed models."
        raise ValueError(msg)
    position = _selected_position(datasets[first_label], row_position)
    first_case = cases.load_case(datasets[first_label], position)
    second_case = cases.load_case(datasets[second_label], position)
    if (
        first_case.case_index != second_case.case_index
        or first_case.source_index != second_case.source_index
        or first_case.fields != second_case.fields
        or not np.array_equal(first_case.reference, second_case.reference)
    ):
        msg = "Model comparison requires exact shared case identity and reference fields."
        raise dataframe.ComparisonCompatibilityError(msg)
    display = presentation.shared_display_fields((datasets[first_label], datasets[second_label]))
    pressure = next(field for field in display if field.label == "p")
    speed = next(field for field in display if field.key == "velocity_magnitude")
    pressure_values = (
        presentation.case_field(first_case, pressure, source="reference"),
        presentation.case_field(first_case, pressure, source="prediction"),
        presentation.case_field(second_case, pressure, source="prediction"),
    )
    speed_values = (
        presentation.case_field(first_case, speed, source="reference"),
        presentation.case_field(first_case, speed, source="prediction"),
        presentation.case_field(second_case, speed, source="prediction"),
    )
    row_labels = ("GT", first_label, second_label)
    row_cases = (first_case, first_case, second_case)
    scale = _normalize_scale_mode(scale_mode)
    _x, _y, x_grid, y_grid = presentation.display_grid(first_case)
    figure, axes = layout.map_subplots(rows=3, columns=2)
    for row, row_label in enumerate(row_labels):
        pressure_levels = _levels(pressure_values[0] if scale == "Shared (GT)" else pressure_values[row])
        speed_levels = _levels(speed_values[0] if scale == "Shared (GT)" else speed_values[row])
        _contour(
            figure,
            axes[row, 0],
            x_grid,
            y_grid,
            pressure_values[row],
            levels=pressure_levels,
            cmap="turbo",
            title=f"{row_label} p [{pressure.unit}]",
            case=first_case,
        )
        _contour(
            figure,
            axes[row, 1],
            x_grid,
            y_grid,
            speed_values[row],
            levels=speed_levels,
            cmap="turbo",
            title=f"{row_label} {speed.matplotlib_label} [{speed.unit}]",
            case=first_case,
        )
        tensor = row_cases[row].reference if row == 0 else row_cases[row].prediction
        _overlay_streamlines(
            axes[row, 1], datasets[first_label] if row < _MINIMUM_COMPARISON_MODELS else datasets[second_label], row_cases[row], tensor
        )
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_case.coordinate_units[0]}]",
        y_label=f"y [{first_case.coordinate_units[1]}]",
    )
    figure.suptitle(layout.case_title("Pressure and velocity comparison", case_number=first_case.case_index), fontsize=14)
    return figure


def plot_task_aware_sample(
    *,
    datasets: Mapping[str, pd.DataFrame],
    positions: Mapping[str, int] | None = None,
) -> Figure:
    """Render current learned fields for direct programmatic callers."""
    dataframe.validate_comparison(datasets)
    selected = dict.fromkeys(datasets, 0) if positions is None else dict(positions)
    if set(selected) != set(datasets):
        msg = "positions must identify every dataset label exactly once."
        raise ValueError(msg)
    row_count = sum(len(frame.attrs["output_fields"]) for frame in datasets.values())
    figure, axes = layout.map_subplots(rows=row_count, columns=3)
    loaded_cases = {label: cases.load_case(frame, _selected_position(frame, selected[label])) for label, frame in datasets.items()}
    case_numbers = {label: case.case_index for label, case in loaded_cases.items()}
    common_case = next(iter(case_numbers.values())) if len(set(case_numbers.values())) == 1 else None
    row = 0
    for label in datasets:
        case = loaded_cases[label]
        heading = label if common_case is not None else layout.case_title(label, case_number=case.case_index)
        extent = cases.grid_extent(case)
        for index, field in enumerate(case.fields):
            reference = case.reference[index]
            prediction = case.prediction[index]
            error = np.abs(prediction - reference)
            shared_low = float(min(np.min(reference), np.min(prediction)))
            shared_high = float(max(np.max(reference), np.max(prediction)))
            if np.isclose(shared_low, shared_high):
                shared_high += np.finfo(float).eps
            for column, values, title, cmap, low, high in (
                (0, reference, "true", "turbo", shared_low, shared_high),
                (1, prediction, "pred", "turbo", shared_low, shared_high),
                (2, error, "MAE", "Blues", 0.0, max(float(np.quantile(error, 0.99)), np.finfo(float).eps)),
            ):
                image = axes[row, column].imshow(values, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=low, vmax=high)
                axes[row, column].set_title(f"{heading} — {field} {title}")
                layout.add_map_colorbar(figure, image, axes[row, column])
            row += 1
    first_case = next(iter(loaded_cases.values()))
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_case.coordinate_units[0]}]",
        y_label=f"y [{first_case.coordinate_units[1]}]",
    )
    if common_case is not None:
        figure.suptitle(layout.case_title("Task-aware sample", case_number=common_case))
    else:
        figure.suptitle("Task-aware samples")
    return figure


def plot_task_aware_sample_at_position(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
) -> Figure:
    """Render one shared saved-membership position across comparable datasets."""
    return plot_task_aware_sample(datasets=datasets, positions=dict.fromkeys(datasets, row_position))


def _outlier_positions(frame: pd.DataFrame, field: presentation.DisplayField, *, top_k: int) -> tuple[int, ...]:
    """Return worst channel positions followed by the parameter-centre reference."""
    values = pd.to_numeric(frame[field.metric_column], errors="raise").to_numpy(dtype=float)
    worst = tuple(int(position) for position in _rank_positions(frame, values, descending=True)[:top_k])
    return (*worst, presentation.reference_case_position(frame))


def plot_linked_outlier_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    channel: str = "U",
    selection_index: int = 0,
    error_mode: str = "MAE",
    scale_mode: str = "Independent",
    top_k: int = _DEFAULT_TOP_K,
) -> Figure:
    """Render one of the worst-five-plus-reference cases for a selected channel."""
    _label, frame = _single_dataset(datasets)
    field = _resolve_display_field(frame, channel)
    positions = _outlier_positions(frame, field, top_k=top_k)
    if isinstance(selection_index, bool) or not isinstance(selection_index, int) or not 0 <= selection_index < len(positions):
        msg = "selection_index is outside the outlier/reference sequence."
        raise IndexError(msg)
    title = "Reference field view" if selection_index == len(positions) - 1 else f"{field.label} outlier field view"
    return _plot_prediction_overview(
        frame=frame,
        row_position=positions[selection_index],
        error_mode=error_mode,
        scale_mode=scale_mode,
        title=title,
    )


def plot_linked_input_extreme_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    parameter: str,
    selection_index: int = 0,
    error_mode: str = "MAE",
    scale_mode: str = "Independent",
) -> Figure:
    """Render the minimum or maximum case for one supported metadata parameter."""
    _label, frame = _single_dataset(datasets)
    parameters = presentation.metadata_parameters((frame,))
    if parameter not in parameters:
        msg = f"Unsupported metadata parameter {parameter!r}."
        raise ValueError(msg)
    if selection_index not in {0, 1}:
        msg = "selection_index must be 0 (minimum) or 1 (maximum)."
        raise IndexError(msg)
    values = pd.to_numeric(frame[parameter], errors="raise").to_numpy(dtype=float)
    order = _rank_positions(frame, values, descending=selection_index == 1)
    title = f"Input extreme: {presentation.metadata_label(parameter)} {'minimum' if selection_index == 0 else 'maximum'}"
    return _plot_prediction_overview(
        frame=frame,
        row_position=int(order[0]),
        error_mode=error_mode,
        scale_mode=scale_mode,
        title=title,
    )
