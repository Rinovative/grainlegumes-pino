"""
===============================================================================
eda_panel.py
===============================================================================
Compose numbered lazy EDA tabs from validated task-aware DataFrames.

Responsibilities:
  - Wire the shared EDA presentation registry to public statistical/spectral plots
  - Inject labelled datasets without changing plot function signatures
  - Defer figure construction until a user selects a numbered dropdown view
  - Preserve the selected figure as an explicit PDF-export target

Design principles:
  - Construction performs no figure rendering and preserves lazy dropdown behavior
  - Numbering is derived only from the shared immutable presentation registry

This module does NOT:
  - Define display order, labels, or scientific EDA mathematics
  - Own generic widget callbacks or PDF-export behavior
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets

from src.analysis import ui
from src.analysis.presentation import registry as presentation

from . import plots

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd


def _spectral_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    single_plot_function: Callable[..., Any],
    aggregate_plot_function: Callable[..., Any] | None,
    controls: Mapping[str, widgets.ValueWidget] | None = None,
) -> widgets.VBox:
    """Build an automatic spectral viewer over actual manifest case IDs."""
    case_numbers = {name: plots.spectral.available_case_numbers(frame) for name, frame in datasets.items()}
    return ui.viewers.make_dataset_case_scope_viewer(
        datasets=datasets,
        case_numbers_by_dataset=case_numbers,
        single_plot_func=single_plot_function,
        aggregate_plot_func=aggregate_plot_function,
        controls=controls,
        start_cases=100,
        step_size=50,
    )


def _isotropic_viewer(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """Build plot 2-1 with aggregate and exact single-case scopes."""
    return _spectral_viewer(
        datasets=datasets,
        single_plot_function=plots.spectral.plot_isotropic_spectral_case,
        aggregate_plot_function=plots.spectral.plot_isotropic_spectral_summary,
    )


def _directional_viewer(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """Build plot 2-2 with aggregate and exact single-case scopes."""
    return _spectral_viewer(
        datasets=datasets,
        single_plot_function=plots.spectral.plot_directional_spectral_case,
        aggregate_plot_function=plots.spectral.plot_directional_spectral_summary,
    )


def _evolution_viewer(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """Build plot 2-3 with a local automatically updating orientation control."""
    orientation = widgets.Dropdown(
        options=(
            ("Cross-stream spectrum k_x along flow direction y", "cross_stream_along_flow"),
            ("Flow-direction spectrum k_y across cross-stream direction x", "flow_across_cross_stream"),
        ),
        value="cross_stream_along_flow",
        description="Orientation:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )
    return _spectral_viewer(
        datasets=datasets,
        single_plot_function=plots.spectral.plot_vertical_spectral_case,
        aggregate_plot_function=None,
        controls={"orientation": orientation},
    )


def build_eda_panel(
    *,
    datasets: Mapping[str, pd.DataFrame],
    title: str = "Dataset EDA",
) -> widgets.Widget:
    """
    Build the maintained numbered EDA panel without rendering a figure.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Non-empty labelled task-compatible EDA frames.
    title : str, optional
        Human-readable label used by the collapsed open button.

    Returns
    -------
    ipywidgets.Widget
        Lazy tab panel whose numbered tab and dropdown labels follow the shared
        presentation registry and whose current figure can be exported as PDF.

    Raises
    ------
    ValueError
        If no labelled datasets are supplied or a registry callable is missing.

    """
    if not datasets:
        message = "EDA panel construction requires at least one labelled dataset."
        raise ValueError(message)
    toggle = ui.notebook.make_toggle_shortcut(dict(datasets))

    plot_functions: dict[str, Callable[..., Any]] = {
        "metadata_statistics": plots.case_statistics.plot_meta_statistics,
        "parameter_distributions": plots.case_statistics.plot_meta_parameters,
        "field_value_distributions": plots.case_statistics.plot_field_value_distributions,
        "isotropic_spectra": _isotropic_viewer,
        "directional_spectra": _directional_viewer,
        "spectral_evolution": _evolution_viewer,
    }
    export_state = {"fig": None, "plot_name": None, "title": None}
    sections: list[widgets.Widget] = []
    tab_titles: list[str] = []
    for _section, section_label, numbered_plots in presentation.numbered_registry(presentation.EDA_SECTIONS):
        entries = []
        for plot, plot_label in numbered_plots:
            try:
                plot_function = plot_functions[plot.key]
            except KeyError as error:
                message = f"EDA presentation plot {plot.key!r} has no callable."
                raise ValueError(message) from error
            entries.append(toggle(plot_label, plot_function, plot_name=plot_label))
        sections.append(ui.notebook.make_dropdown_section(entries, export_state=export_state, select_first=True))
        tab_titles.append(section_label)
    return ui.notebook.make_lazy_panel_with_tabs(
        sections,
        tab_titles=tab_titles,
        open_btn_text=f"{title} - Open",
        close_btn_text="Close",
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )
