"""
===============================================================================
evaluation_plot_layout.py
===============================================================================
Define shared current-native layout contracts for evaluation field maps.

Responsibilities:
  - Keep equivalent evaluation maps at one readable notebook-native scale
  - Keep map colorbars proportionate and consistently padded
  - Keep comparable map axes equal-sized with row/column-aware axis labels
  - Size composite line-and-map figures without plot-local magic numbers
  - Center figure titles over data axes when a right-side column is present

Design principles:
  - Layout is presentation-only and never changes scientific values
  - Physical aspect ratios remain owned by the plotting axes
  - One immutable contract serves predictive, physics, sample, and outlier maps

This module does NOT:
  - Load artifacts, select cases, calculate statistics, or clip scientific data
  - Choose colormaps, labels, coordinates, or plot-specific panel composition
  - Own notebook controls, display calls, exports, or lazy rendering
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sized

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.text import Text


@dataclass(frozen=True, slots=True)
class MapLayoutContract:
    """Define one consistent notebook-scale physical-map layout."""

    column_width: float = 5.8
    row_height: float = 3.6
    map_column_spacing: float = 0.18
    map_row_spacing: float = 0.10
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.025
    composite_line_height: float = 3.25
    composite_vertical_gap: float = 0.25
    composite_height_ratios: tuple[float, float] = (1.0, 1.35)
    composite_hspace: float = 0.30
    composite_group_spacing: float = 0.28

    def figure_size(self, *, rows: int, columns: int) -> tuple[float, float]:
        """Return a readable size for an all-map grid."""
        if rows <= 0 or columns <= 0:
            msg = "Map grids require positive row and column counts."
            raise ValueError(msg)
        return self.column_width * columns, self.row_height * rows

    def composite_size(self, *, map_columns: int) -> tuple[float, float]:
        """Return size for one line region above one row of maps."""
        if map_columns <= 0:
            msg = "Composite figures require at least one map column."
            raise ValueError(msg)
        return self.column_width * map_columns, self.composite_line_height + self.composite_vertical_gap + self.row_height


MAP_LAYOUT = MapLayoutContract()


def case_title(
    title: str,
    *,
    case_count: int | None = None,
    case_number: int | None = None,
) -> str:
    """Append exactly one aggregate-count or visible-case suffix."""
    if (case_count is None) == (case_number is None):
        msg = "Supply exactly one of case_count or case_number."
        raise ValueError(msg)
    value = case_count if case_count is not None else case_number
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        msg = "Case counts and visible case numbers must be positive integers."
        raise ValueError(msg)
    numeric = int(value)
    suffix = f"{numeric} {'case' if numeric == 1 else 'cases'}" if case_count is not None else f"Case {numeric}"
    return f"{title} — {suffix}"


def effective_case_counts(
    datasets: Mapping[str, Sized],
    *,
    max_cases: int | None = None,
) -> dict[str, int]:
    """Return actual per-dataset counts used by a complete or bounded view."""
    if not datasets:
        msg = "Case-based titles require at least one labelled dataset."
        raise ValueError(msg)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, Integral) or int(max_cases) <= 0):
        msg = "max_cases must be a positive integer when supplied."
        raise ValueError(msg)
    return {label: len(dataset) if max_cases is None else min(int(max_cases), len(dataset)) for label, dataset in datasets.items()}


def aggregate_title_context(title: str, case_counts: Mapping[str, int]) -> tuple[str, dict[str, str]]:
    """Return an honest figure title and dataset headings for aggregate evidence."""
    if not case_counts:
        msg = "Aggregate title context requires at least one dataset count."
        raise ValueError(msg)
    validated = {label: case_title(label, case_count=count) for label, count in case_counts.items()}
    unique_counts = set(case_counts.values())
    if len(unique_counts) == 1:
        return case_title(title, case_count=next(iter(unique_counts))), dict.fromkeys(case_counts, "")
    return title, validated


def map_subplots(*, rows: int, columns: int, **kwargs: Any) -> tuple[Figure, Any]:
    """Create one equal-column map grid under the shared layout contract."""
    if "figsize" in kwargs or "squeeze" in kwargs or "layout" in kwargs or "gridspec_kw" in kwargs:
        msg = "Shared map grids own figsize, squeeze, layout, and GridSpec spacing."
        raise ValueError(msg)
    return plt.subplots(
        rows,
        columns,
        figsize=MAP_LAYOUT.figure_size(rows=rows, columns=columns),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"wspace": MAP_LAYOUT.map_column_spacing, "hspace": MAP_LAYOUT.map_row_spacing},
        **kwargs,
    )


def add_map_colorbar(
    figure: Figure,
    image: ScalarMappable,
    axis: Axes,
    *,
    label: str | None = None,
    colorbar_axis: Axes | None = None,
) -> Colorbar:
    """Attach one proportionate colorbar under the shared sizing contract."""
    if colorbar_axis is None:
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes(
            "right",
            size=f"{100.0 * MAP_LAYOUT.colorbar_fraction:g}%",
            pad=f"{100.0 * MAP_LAYOUT.colorbar_pad:g}%",
        )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    if label is not None:
        colorbar.set_label(label)
    return colorbar


def _map_axis_grid(axes: Iterable[Iterable[Axes]]) -> tuple[tuple[Axes, ...], ...]:
    """Normalize and validate one rectangular map-axis grid."""
    rows = tuple(tuple(row) for row in axes)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        msg = "Map-axis grids must be non-empty and rectangular."
        raise ValueError(msg)
    return rows


def apply_map_grid_axis_labels(
    axes: Iterable[Iterable[Axes]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    """
    Apply coordinate and tick labels only at visible map-grid edges.

    Parameters
    ----------
    axes : Iterable[Iterable[Axes]]
        Rectangular rows of map axes. Invisible or disabled axes are ignored.
    x_label : str
        Label applied to active maps in the global bottom row.
    y_label : str
        Label applied to active maps in the global leftmost column.

    Raises
    ------
    ValueError
        If the map grid is empty or not rectangular.

    Notes
    -----
    Tick positions are preserved. Repeated tick-label artists are hidden without
    changing map extents, and a standalone active map keeps complete axes.

    """
    rows = _map_axis_grid(axes)
    active = tuple(tuple(axis.get_visible() and axis.axison for axis in row) for row in rows)
    bottom_row = len(rows) - 1
    for row, axis_row in enumerate(rows):
        for column, axis in enumerate(axis_row):
            if not active[row][column]:
                continue
            is_bottom = row == bottom_row
            is_left = column == 0
            axis.set_xlabel(x_label if is_bottom else "")
            axis.set_ylabel(y_label if is_left else "")
            axis.tick_params(axis="x", labelbottom=is_bottom)
            axis.tick_params(axis="y", labelleft=is_left)


def add_shortened_column_x_decorations(
    axes: Iterable[Iterable[Axes]],
    *,
    x_label: str,
) -> tuple[Text, ...]:
    """
    Add layout-excluded x decorations beneath shortened map-grid columns.

    Parameters
    ----------
    axes : Iterable[Iterable[Axes]]
        Rectangular rows of map axes. Invisible or disabled axes are ignored.
    x_label : str
        Visual coordinate label for each shortened column.

    Returns
    -------
    tuple[Text, ...]
        Manual tick-number and axis-label artists attached to shortened columns.

    Notes
    -----
    Tick numbers come from each axis's settled locator and formatter. All
    artists are excluded from layout, so they do not alter GridSpec spacing,
    map geometry, or normal tick-label visibility.

    """
    rows = _map_axis_grid(axes)
    bottom_row = len(rows) - 1
    figure = rows[0][0].figure
    figure.canvas.draw()
    active = tuple(tuple(axis.get_visible() and axis.axison for axis in row) for row in rows)
    visible_bottom_tick_labels = [
        tick_label
        for column, axis in enumerate(rows[bottom_row])
        if active[bottom_row][column]
        for tick_label in axis.get_xticklabels()
        if tick_label.get_visible()
    ]
    tick_style = visible_bottom_tick_labels[0] if visible_bottom_tick_labels else None
    artists: list[Text] = []
    for column in range(len(rows[0])):
        active_rows = [row for row in range(len(rows)) if active[row][column]]
        if not active_rows:
            continue
        last_active_row = max(active_rows)
        if last_active_row == bottom_row:
            continue
        axis = rows[last_active_row][column]
        tick_locations = [float(location) for location in axis.get_xticks()]
        tick_labels = axis.xaxis.get_major_formatter().format_ticks(tick_locations)
        x_limits = sorted(axis.get_xlim())
        tolerance = max(x_limits[1] - x_limits[0], 1.0) * 1e-9
        for location, tick_label in zip(tick_locations, tick_labels, strict=True):
            if not tick_label or not x_limits[0] - tolerance <= location <= x_limits[1] + tolerance:
                continue
            tick_artist = axis.text(
                location,
                -0.035,
                tick_label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                clip_on=False,
                fontproperties=(tick_style.get_fontproperties() if tick_style is not None else axis.xaxis.label.get_fontproperties()),
                color=(tick_style.get_color() if tick_style is not None else axis.xaxis.label.get_color()),
            )
            tick_artist.set_in_layout(False)
            artists.append(tick_artist)
        label_artist = axis.text(
            0.5,
            -0.11,
            x_label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            clip_on=False,
            fontproperties=axis.xaxis.label.get_fontproperties(),
            color=axis.xaxis.label.get_color(),
        )
        label_artist.set_in_layout(False)
        artists.append(label_artist)
    return tuple(artists)


def _center_text_artist(figure: Figure, artist: Text, plot_axes: tuple[Axes, ...]) -> None:
    """Keep one text artist centered over its visible plotting axes."""

    def center_artist(_event: object = None) -> None:
        """Follow final Matplotlib positions without forcing an eager draw."""
        positions = [axis.get_position() for axis in plot_axes if axis.get_visible()]
        if positions:
            left = min(position.x0 for position in positions)
            right = max(position.x1 for position in positions)
            artist.set_x((left + right) / 2.0)

    center_artist()
    figure.canvas.mpl_connect("draw_event", center_artist)


def title_over_axes(
    figure: Figure,
    title: str,
    axes: Iterable[Axes],
    **kwargs: Any,
) -> Text:
    """Keep one figure title centered over its data axes after layout."""
    plot_axes = tuple(axes)
    if not plot_axes:
        msg = "Title centering requires at least one plotting axis."
        raise ValueError(msg)
    title_artist = figure.suptitle(title, **kwargs)
    _center_text_artist(figure, title_artist, plot_axes)
    return title_artist


def text_over_axes(
    figure: Figure,
    text: str,
    axes: Iterable[Axes],
    *,
    y: float,
    **kwargs: Any,
) -> Text:
    """Place figure-level explanatory text above and centered on data axes."""
    plot_axes = tuple(axes)
    if not plot_axes:
        msg = "Explanatory-text centering requires at least one plotting axis."
        raise ValueError(msg)
    text_artist = figure.text(0.5, y, text, **kwargs)
    _center_text_artist(figure, text_artist, plot_axes)
    return text_artist
