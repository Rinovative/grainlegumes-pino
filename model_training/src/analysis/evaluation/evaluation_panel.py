"""
===============================================================================
evaluation_panel.py
===============================================================================
Compose lazy, selectable scientific evaluation sections for artifact notebooks.

Responsibilities:
  - Group public plot functions by the scientific question they answer
  - Derive model, dataset, field, case, metric, parameter, tail, and rank controls
  - Defer NPZ reads and figure construction until an explicit render action
  - Preserve the current figure as an explicit PDF-export target

Design principles:
  - Input frames must already satisfy artifact comparison compatibility
  - Construction performs no NPZ reads or figure rendering
  - Every semantic control is derived from options shared by selected frames

This module does NOT:
  - Parse artifacts, decide scientific compatibility, or implement plot mathematics
  - Render the curated non-interactive bundle or upload media to W&B
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets

from src import analysis
from src.analysis.presentation import registry as presentation

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd

EVALUATION_SECTION_KEYS = tuple(section.key for section in presentation.EVALUATION_SECTIONS)


def _shared_options(options_by_frame: list[tuple[str, ...]]) -> tuple[str, ...]:
    """Return first-frame order restricted to options available in every frame."""
    if not options_by_frame:
        return ()
    shared = set(options_by_frame[0])
    for options in options_by_frame[1:]:
        shared.intersection_update(options)
    return tuple(option for option in options_by_frame[0] if option in shared)


def _dropdown(options: tuple[str, ...], *, description: str) -> widgets.Dropdown:
    """Build one non-empty semantic dropdown with a readable notebook label."""
    if not options:
        msg = f"No shared options are available for {description.rstrip(':').lower()}."
        raise ValueError(msg)
    return widgets.Dropdown(
        options=options,
        value=options[0],
        description=description,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )


def _integer_control(*, value: int, minimum: int, maximum: int, description: str) -> widgets.IntSlider:
    """Build one bounded integer control for saved-membership navigation."""
    return widgets.IntSlider(
        value=value,
        min=minimum,
        max=maximum,
        step=1,
        description=description,
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="360px"),
    )


def _build_sections(
    toggle: Callable[..., tuple[str, Callable[[], Any], str]],
    datasets: Mapping[str, pd.DataFrame],
) -> dict[str, tuple[list[tuple[str, Callable[[], Any], str]], str]]:
    """
    Wire numbered presentation entries to plot callables and semantic controls.

    Parameters
    ----------
    toggle : callable
        Dataset-injecting dropdown-entry factory from ``analysis.ui.notebook``.
    datasets : Mapping[str, pandas.DataFrame]
        Validated labelled artifact frames used to derive shared fields, metrics,
        metadata parameters, and safe navigation bounds.

    Returns
    -------
    dict[str, tuple[list[tuple[str, callable, str]], str]]
        Stable section key to numbered lazy dropdown entries and numbered tab title.

    Notes
    -----
    Registry order and names come only from :mod:`analysis.presentation`. This
    function owns callable wiring and widgets; it does not load NPZ cases or
    construct figures. The optional extreme-input view is omitted when artifacts
    expose no numeric metadata, and the remaining labels stay consecutive.

    """
    plots = analysis.evaluation.plots
    viewer = analysis.ui.viewers.make_controlled_viewer
    frames = list(datasets.values())
    max_cases = min(len(frame) for frame in frames)
    output_fields = _shared_options([tuple(frame.attrs["output_fields"]) for frame in frames])
    case_metrics = _shared_options([plots.samples_outliers.available_case_metrics(frame) for frame in frames])
    trend_metrics = _shared_options(
        [
            tuple(
                metric
                for metric in ("rel_l2", "rel_h1", *(f"normalized_rmse_{field}" for field in frame.attrs["output_fields"]))
                if metric in frame.columns
            )
            for frame in frames
        ]
    )
    metadata_parameters = _shared_options([analysis.evaluation.dataframe.numeric_metadata_columns(frame) for frame in frames])

    def controlled(
        title: str,
        plot_func: Callable[..., Any],
        *,
        controls: Mapping[str, widgets.ValueWidget] | None = None,
    ) -> tuple[str, Callable[[], Any], str]:
        """Create one lazily rendered, model-selectable dropdown entry."""
        return toggle(
            title,
            viewer,
            plot_func=plot_func,
            controls=controls,
            plot_name=title,
        )

    def prefix_control() -> widgets.IntSlider:
        """Return a fresh bounded saved-prefix control for one aggregate view."""
        return _integer_control(
            value=min(64, max_cases),
            minimum=1,
            maximum=max_cases,
            description="Cases (saved prefix):",
        )

    entry_builders: dict[str, Callable[[str], tuple[str, Callable[[], Any], str]]] = {
        "authoritative_run_summary": lambda title: controlled(title, plots.run_summary.build_run_summary_table),
        "accuracy_physics_pareto": lambda title: controlled(title, plots.run_summary.plot_accuracy_physics_pareto),
        "predictive_error_distributions": lambda title: controlled(
            title,
            plots.error_behavior.plot_predictive_error_distributions,
            controls={"max_cases": prefix_control()},
        ),
        "mean_reference_prediction": lambda title: controlled(title, plots.error_behavior.plot_mean_spatial_fields),
        "spatial_error_statistics": lambda title: controlled(title, plots.error_behavior.plot_error_maps),
        "case_mean_bias": lambda title: controlled(title, plots.error_behavior.plot_mean_field_bias),
        "target_magnitude_error": lambda title: controlled(
            title,
            plots.error_behavior.plot_error_vs_target_magnitude,
            controls={"max_cases": prefix_control()},
        ),
        "boundary_distance_error": lambda title: controlled(
            title,
            plots.error_behavior.plot_boundary_error_decomposition,
            controls={"max_cases": prefix_control()},
        ),
        "residual_distributions": lambda title: controlled(title, plots.physical_consistency.plot_residual_distributions),
        "residual_maps": lambda title: controlled(title, plots.physical_consistency.plot_spatial_residuals),
        "pressure_boundary_drop": lambda title: controlled(title, plots.physical_consistency.plot_pressure_boundary_summary),
        "output_spectra_transfer": lambda title: controlled(
            title,
            plots.spectral_fidelity.plot_spectral_fidelity,
            controls={"max_cases": prefix_control()},
        ),
        "parameter_count_accuracy": lambda title: controlled(title, plots.sensitivity_capacity.plot_capacity_accuracy),
        "metadata_error_associations": lambda title: controlled(title, plots.sensitivity_capacity.plot_metadata_error_heatmap),
        "metadata_error_trends": lambda title: controlled(
            title,
            plots.sensitivity_capacity.plot_metadata_error_trends,
            controls={"metric": _dropdown(trend_metrics, description="Error metric:")},
        ),
        "shared_case_comparison": lambda title: controlled(
            title,
            plots.samples_outliers.plot_task_aware_sample_at_position,
            controls={
                "row_position": _integer_control(
                    value=0,
                    minimum=0,
                    maximum=max_cases - 1,
                    description="Saved row position:",
                )
            },
        ),
        "permeability_error_overlay": lambda title: controlled(
            title,
            plots.samples_outliers.plot_permeability_error_overlay,
            controls={
                "row_position": _integer_control(
                    value=0,
                    minimum=0,
                    maximum=max_cases - 1,
                    description="Saved row position:",
                ),
                "field": _dropdown(output_fields, description="Output field:"),
            },
        ),
        "outlier_extreme_tables": lambda title: controlled(title, plots.samples_outliers.plot_outlier_extreme_tables),
        "worst_case_fields": lambda title: controlled(
            title,
            plots.samples_outliers.plot_linked_outlier_cases,
            controls={
                "metric": _dropdown(case_metrics, description="Ranking metric:"),
                "rank": _integer_control(value=1, minimum=1, maximum=max_cases, description="Worst-case rank:"),
            },
        ),
        "extreme_input_fields": lambda title: controlled(
            title,
            plots.samples_outliers.plot_linked_input_extreme_cases,
            controls={
                "parameter": _dropdown(metadata_parameters, description="Input parameter:"),
                "extreme": _dropdown(("high", "low"), description="Tail:"),
                "rank": _integer_control(value=1, minimum=1, maximum=max_cases, description="Extreme rank:"),
            },
        ),
    }

    sections: dict[str, tuple[list[tuple[str, Callable[[], Any], str]], str]] = {}
    for section_index, section in enumerate(presentation.EVALUATION_SECTIONS, start=1):
        active_plots = tuple(plot for plot in section.plots if plot.key != "extreme_input_fields" or metadata_parameters)
        entries = []
        for plot_index, plot in enumerate(active_plots, start=1):
            try:
                builder = entry_builders[plot.key]
            except KeyError as error:
                message = f"Evaluation presentation plot {plot.key!r} has no callable."
                raise ValueError(message) from error
            label = presentation.plot_display_label(section_index, plot_index, plot.name)
            entries.append(builder(label))
        sections[section.key] = (
            entries,
            presentation.section_display_label(section_index, section.name),
        )
    return sections


def build_evaluation_panel(
    *,
    datasets_eval: Mapping[str, pd.DataFrame],
    title: str,
    sections: list[str] | str = "all",
) -> widgets.Widget:
    """
    Build a lazy artifact-evaluation panel with selectable scientific views.

    Parameters
    ----------
    datasets_eval : Mapping[str, pandas.DataFrame]
        Labelled artifact frames. Compatibility is checked before any widget is
        constructed; model/dataset checkboxes permit focused comparisons later.
    title : str
        Human-readable panel label used by the collapsed open button.
    sections : list[str] | {"all"}, optional
        Ordered section keys to expose, or every reviewed section.

    Returns
    -------
    ipywidgets.Widget
        Collapsible tab panel. Plot selection and the first render are both lazy;
        controlled Matplotlib figures remain exportable as PDF.

    Raises
    ------
    ComparisonCompatibilityError
        If artifact identities, memberships, fields, or formulas are incompatible.
    ValueError
        If an unknown section key is requested.
    TypeError
        If `sections` is neither ``"all"`` nor a non-empty string list.

    """
    analysis.evaluation.dataframe.validate_comparison(datasets_eval)
    datasets = dict(datasets_eval)
    toggle = analysis.ui.notebook.make_toggle_shortcut(datasets)
    registry = _build_sections(toggle, datasets)
    if sections == "all":
        section_keys = list(EVALUATION_SECTION_KEYS)
    elif isinstance(sections, list) and sections and all(isinstance(key, str) for key in sections):
        unknown = sorted(set(sections).difference(registry))
        if unknown:
            msg = f"Unknown evaluation sections: {unknown}."
            raise ValueError(msg)
        section_keys = sections
    else:
        msg = "sections must be 'all' or a non-empty list of section keys."
        raise TypeError(msg)

    export_state = {"fig": None, "plot_name": None, "title": None}
    ui_sections = [analysis.ui.notebook.make_dropdown_section(registry[key][0], export_state=export_state) for key in section_keys]
    tab_titles = [registry[key][1] for key in section_keys]
    return analysis.ui.notebook.make_lazy_panel_with_tabs(
        ui_sections,
        tab_titles=tab_titles,
        open_btn_text=f"{title} - Open Evaluation",
        close_btn_text="Close",
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )
