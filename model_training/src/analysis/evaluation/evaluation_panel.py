"""
===============================================================================
evaluation_panel.py
===============================================================================
Compose historical evaluation panels over current admitted artifacts.

Responsibilities:
  - Keep the approved single-model and comparison view order explicit
  - Bind every view to its historically appropriate local controls and defaults
  - Accept caller-provided context labels and session-owned canonical frames
  - Keep comparison roles separate without an outer dataset selector
  - Preserve lazy first rendering and panel-local figure export

Design principles:
  - Strict artifact loading and EvaluationSession binding precede construction
  - Dataset, model, channel, case, and parameter controls remain view-specific
  - Presentation labels never alter canonical identity or numerical cache keys
  - Panel construction opens no case payloads and computes no plot mathematics

This module does NOT:
  - Discover, generate, repair, rebuild, or admit artifact payloads
  - Recreate a historical dataframe or generic interaction registry
  - Infer dataset identity from display names, paths, or row positions
  - Cache figures, case arrays, or numerical reductions
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets

from src import analysis

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from src.analysis.evaluation.evaluation_session import EvaluationSession

_MINIMUM_COMPARISON_RUNS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_OUTLIER_COUNT = 5

SINGLE_MODEL_EVALUATION_SECTION_KEYS = (
    "overview",
    "global_error",
    "architecture",
    "error_decomposition",
    "physical_consistency",
    "spectral_analysis",
    "parameter_sensitivity",
    "sample_viewer",
    "outlier_analysis",
)
COMPARISON_EVALUATION_SECTION_KEYS = SINGLE_MODEL_EVALUATION_SECTION_KEYS


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Define one notebook-provided dataset context and its ordered model frames."""

    key: str
    label: str
    datasets: Mapping[str, pd.DataFrame]


def _dropdown(
    options: Sequence[str] | Sequence[tuple[str, str]],
    *,
    description: str,
    value: str | None = None,
) -> widgets.Dropdown:
    """Build one compact historical scientific dropdown."""
    resolved = tuple(options)
    if not resolved:
        msg = f"No options are available for {description.rstrip(':').lower()}."
        raise ValueError(msg)
    first = resolved[0][1] if isinstance(resolved[0], tuple) else resolved[0]
    return widgets.Dropdown(
        options=resolved,
        value=first if value is None else value,
        description=description,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )


def _physics_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether every frame supplies compatible maintained physics evidence."""
    try:
        analysis.evaluation.dataframe.validate_comparison(datasets, require_physics=True)
    except analysis.evaluation.dataframe.ComparisonCompatibilityError:
        return False
    return True


def _permeability_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether every frame declares the historical permeability tensor inputs."""
    required = {"kxx", "kxy", "kyy"}
    return all(required.issubset(frame.attrs.get("input_fields", ())) for frame in datasets.values())


def _pressure_velocity_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether the synchronized two-model pressure/velocity view is valid."""
    if len(datasets) < _MINIMUM_COMPARISON_RUNS:
        return False
    roles = {analysis.evaluation.dataframe.dataset_role(frame) for frame in datasets.values()}
    if len(roles) != 1:
        return False
    try:
        for frame in datasets.values():
            analysis.evaluation.dataframe.single_output_group_field(frame, group_id="pressure")
            velocity = analysis.evaluation.dataframe.output_group_fields(frame, group_id="velocity")
            if len(velocity) < _MINIMUM_VELOCITY_COMPONENTS:
                return False
    except analysis.evaluation.dataframe.ComparisonCompatibilityError:
        return False
    return True


def _make_outlier_table_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    top_k: int = _OUTLIER_COUNT,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """Render the historical automatic outlier tables with one local dropdown."""
    if not datasets:
        msg = "Outlier tables require at least one labelled dataset."
        raise ValueError(msg)
    selector = analysis.ui.components.ui_dropdown_dataset(list(datasets)) if len(datasets) > 1 else None
    if selector is not None:
        selector.layout.width = "229px"
    output = analysis.ui.components.ui_output_plot()

    def render(_: object = None) -> None:
        """Rebuild the tables immediately after a local dataset change."""
        if selector is None:
            selected_name = next(iter(datasets))
        else:
            selected_name = selector.value
            if not isinstance(selected_name, str):
                msg = "Dataset dropdown values must be strings."
                raise TypeError(msg)
        analysis.ui.viewers.render_figure(
            out=output,
            plot_func=plot_func,
            kwargs={"datasets": {selected_name: datasets[selected_name]}, "top_k": top_k},
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    if selector is not None:
        selector.observe(render, names="value")
    render()
    children: list[widgets.Widget] = [] if selector is None else [selector]
    children.append(output)
    return widgets.VBox(children, layout=widgets.Layout(align_items="flex-start"))


def _build_sections(
    datasets: Mapping[str, pd.DataFrame],
    *,
    comparison: bool,
) -> dict[str, tuple[list[tuple[str, Callable[..., Any], str]], str]]:
    """Bind the approved views directly to their historical per-view controls."""
    plots = analysis.evaluation.plots
    viewers = analysis.ui.viewers
    controls = analysis.ui.components
    dataset_map = dict(datasets)
    analysis.evaluation.dataframe.validate_comparison(dataset_map)
    frames = tuple(dataset_map.values())
    display_fields = analysis.evaluation.presentation.shared_display_fields(frames)
    channels = tuple(field.label for field in display_fields)
    parameters = analysis.evaluation.presentation.metadata_parameters(frames)
    parameter_options = tuple((analysis.evaluation.presentation.metadata_label(parameter), parameter) for parameter in parameters)
    toggle = analysis.ui.notebook.make_toggle_shortcut(dataset_map)

    overview_entries = [
        toggle(
            "Overview: Summary table",
            plots.run_summary.plot_run_summary_table,
            plot_name="Overview: Summary table",
        ),
    ]
    if comparison:
        overview_entries.extend(
            (
                toggle(
                    "Overview: Global comparison summary",
                    plots.run_summary.plot_relative_comparison_scoreboard,
                    plot_name="Overview: Global comparison summary",
                ),
                toggle(
                    "Overview: Pareto (Error vs Physics)",
                    plots.run_summary.plot_accuracy_physics_pareto,
                    plot_name="Overview: Pareto (Error vs Physics)",
                ),
            )
        )
    global_error_entries = [
        toggle(
            "1-1. Global error metrics",
            plots.error_behavior.plot_global_error_metrics,
            plot_name="1-1. Global error metrics",
        ),
        toggle(
            "1-2. Global error distribution",
            viewers.make_casecount_viewer,
            plot_name="1-2. Global error distribution",
            plot_func=plots.error_behavior.plot_predictive_error_distributions,
            start_cases=50,
            step_size=50,
        ),
        toggle(
            "1-3. GT vs Prediction (mean)",
            viewers.make_casecount_viewer,
            plot_name="1-3. GT vs Prediction (mean)",
            plot_func=plots.error_behavior.plot_mean_field_bias,
            start_cases=50,
            step_size=50,
        ),
        toggle(
            "1-4. Mean error maps",
            viewers.make_casecount_viewer,
            plot_name="1-4. Mean error maps",
            plot_func=plots.error_behavior.plot_mean_error_maps,
            start_cases=100,
            step_size=50,
            controls={"error_mode": controls.ui_radio_error_mode()},
        ),
        toggle(
            "1-5. Std error maps",
            viewers.make_casecount_viewer,
            plot_name="1-5. Std error maps",
            plot_func=plots.error_behavior.plot_std_error_maps,
            start_cases=100,
            step_size=50,
        ),
    ]

    architecture_entries = [
        toggle(
            "2-1. Model-family configuration",
            plots.run_summary.plot_architecture_table,
            plot_name="2-1. Model-family configuration",
        ),
        toggle(
            "2-2. Exact trainable parameters vs performance",
            viewers.make_casecount_viewer,
            plot_name="2-2. Exact trainable parameters vs performance",
            plot_func=plots.sensitivity_capacity.plot_capacity_accuracy,
            start_cases=50,
            step_size=50,
        ),
    ]

    error_decomposition_entries = [
        toggle(
            "3-1. Error vs |GT| magnitude",
            viewers.make_casecount_viewer,
            plot_name="3-1. Error vs |GT| magnitude",
            plot_func=plots.error_behavior.plot_error_vs_target_magnitude,
            start_cases=50,
            step_size=50,
            allow_dataset_selection=True,
            selector_title="datasets",
        ),
        toggle(
            "3-2. Boundary vs interior error",
            viewers.make_casecount_viewer,
            plot_name="3-2. Boundary vs interior error",
            plot_func=plots.error_behavior.plot_boundary_error_decomposition,
            start_cases=100,
            step_size=50,
            controls={"channels": controls.ui_checkbox_channels(channels=channels)},
        ),
    ]

    physical_entries: list[tuple[str, Callable[..., Any], str]] = []
    if _physics_available(dataset_map):
        physical_entries = [
            toggle(
                "4-1. Physical consistency summary table",
                plots.physical_consistency.plot_physical_consistency_summary_table,
                plot_name="4-1. Physical consistency summary table",
            ),
            toggle(
                "4-2. Physical consistency CDF grid (2x2)",
                viewers.make_casecount_viewer,
                plot_name="4-2. Physical consistency CDF grid (2x2)",
                plot_func=plots.physical_consistency.plot_physical_consistency_cdf_grid,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-3. Velocity continuity residual (∇·u)",
                viewers.make_casecount_viewer,
                plot_name="4-3. Velocity continuity residual (∇·u)",
                plot_func=plots.physical_consistency.plot_velocity_divergence,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-4. Porosity-weighted continuity residual (∇·(εu))",
                viewers.make_casecount_viewer,
                plot_name="4-4. Porosity-weighted continuity residual (∇·(εu))",
                plot_func=plots.physical_consistency.plot_div_eps_u_consistency,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-5. Darcy–Brinkman operator residual",  # noqa: RUF001
                viewers.make_casecount_viewer,
                plot_name="4-5. Darcy–Brinkman operator residual",  # noqa: RUF001
                plot_func=plots.physical_consistency.plot_brinkman_residual,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-6. Pressure and boundary-condition consistency",
                viewers.make_casecount_viewer,
                plot_name="4-6. Pressure and boundary-condition consistency",
                plot_func=plots.physical_consistency.plot_pressure_consistency,
                start_cases=100,
                step_size=50,
            ),
        ]

    spectral_entries = [
        toggle(
            "5-1. Demand vs prediction + error",
            viewers.make_casecount_viewer,
            plot_name="5-1. Demand vs prediction + error",
            plot_func=plots.spectral_fidelity.plot_spectral_demand_prediction_error,
            start_cases=50,
            step_size=50,
            controls={
                "channels": controls.ui_checkbox_channels(channels=channels),
                "normalize": controls.ui_checkbox_normalise(default=True),
            },
        ),
        toggle(
            "5-2. Log spectral transfer with support",
            viewers.make_casecount_viewer,
            plot_name="5-2. Log spectral transfer with support",
            plot_func=plots.spectral_fidelity.plot_spectral_transfer_ratio,
            start_cases=50,
            step_size=50,
            controls={"channels": controls.ui_checkbox_channels(channels=channels)},
        ),
    ]

    sensitivity_entries: list[tuple[str, Callable[..., Any], str]] = []
    if parameters:
        sensitivity_entries = [
            toggle(
                "6-1. Parameter-error correlation (heatmap)",
                viewers.make_casecount_viewer,
                plot_name="6-1. Parameter-error correlation (heatmap)",
                plot_func=plots.sensitivity_capacity.plot_metadata_error_heatmap,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "6-2. Error vs input parameter (binned trend)",
                viewers.make_casecount_viewer,
                plot_name="6-2. Error vs input parameter (binned trend)",
                plot_func=plots.sensitivity_capacity.plot_metadata_error_trends,
                start_cases=100,
                step_size=50,
                controls={"channels": controls.ui_checkbox_channels(channels=channels)},
            ),
        ]

    sample_entries = [
        toggle(
            "7-1. Sample GT vs Prediction",
            viewers.make_indexed_viewer,
            plot_name="7-1. Sample GT vs Prediction",
            plot_func=plots.samples_outliers.plot_sample_prediction_overview,
            controls={
                "scale_mode": controls.ui_radio_pred_scale_mode(),
                "error_mode": controls.ui_radio_error_mode(),
            },
            dataset_selection="dropdown",
        ),
    ]
    if _permeability_available(dataset_map):
        sample_entries.append(
            toggle(
                "7-2. Kappa tensor with error overlay",
                viewers.make_indexed_viewer,
                plot_name="7-2. Kappa tensor with error overlay",
                plot_func=plots.samples_outliers.plot_permeability_error_overlay,
                controls={
                    "kappa_scale": controls.ui_radio_kappa_scale(),
                    "channel": controls.ui_dropdown_channel(channels=channels),
                    "error_mode": controls.ui_radio_error_mode(),
                },
                dataset_selection="dropdown",
            )
        )
    if comparison and _pressure_velocity_available(dataset_map):
        labels = tuple(dataset_map)
        sample_entries.append(
            toggle(
                "7-3. Pressure & velocity field comparison",
                viewers.make_indexed_viewer,
                plot_name="7-3. Pressure & velocity field comparison",
                plot_func=plots.samples_outliers.plot_pressure_velocity_comparison,
                controls={
                    "model_1": _dropdown(labels, description="Model 1:", value=labels[0]),
                    "model_2": _dropdown(labels, description="Model 2:", value=labels[1]),
                    "scale_mode": controls.ui_radio_pred_scale_mode(),
                },
                dataset_selection="all",
            )
        )

    outlier_entries = [
        toggle(
            "8-1. Worst per-channel cases (tables)",
            _make_outlier_table_viewer,
            plot_name="8-1. Worst per-channel cases (tables)",
            plot_func=plots.samples_outliers.plot_outlier_table,
            top_k=_OUTLIER_COUNT,
        ),
        toggle(
            "8-2. Worst per-channel cases (field plots)",
            viewers.make_indexed_viewer,
            plot_name="8-2. Worst per-channel cases (field plots)",
            plot_func=plots.samples_outliers.plot_linked_outlier_cases,
            controls={
                "channel": controls.ui_dropdown_channel(channels=channels),
                "error_mode": controls.ui_radio_error_mode(),
            },
            dataset_selection="dropdown",
            max_positions=_OUTLIER_COUNT + 1,
            index_to_kwargs=lambda index: {"selection_index": index},
        ),
    ]
    if parameters:
        outlier_entries.extend(
            (
                toggle(
                    "8-3. Extreme input parameters (table view)",
                    plots.samples_outliers.plot_input_extremes_table,
                    plot_name="8-3. Extreme input parameters (table view)",
                ),
                toggle(
                    "8-4. Extreme input parameter cases (field plots)",
                    viewers.make_indexed_viewer,
                    plot_name="8-4. Extreme input parameter cases (field plots)",
                    plot_func=plots.samples_outliers.plot_linked_input_extreme_cases,
                    controls={
                        "parameter": _dropdown(parameter_options, description="Parameter:"),
                        "error_mode": controls.ui_radio_error_mode(),
                    },
                    dataset_selection="dropdown",
                    max_positions=2,
                    index_to_kwargs=lambda index: {"selection_index": index},
                ),
            )
        )

    registry = {
        "overview": (overview_entries, "Overview"),
        "global_error": (global_error_entries, "Global Error Analysis"),
        "architecture": (architecture_entries, "Architecture Sensitivity"),
        "error_decomposition": (error_decomposition_entries, "Error Decomposition"),
        "physical_consistency": (physical_entries, "Physical Consistency"),
        "spectral_analysis": (spectral_entries, "Spectral & Representation Analysis"),
        "parameter_sensitivity": (sensitivity_entries, "Error Sensitivity"),
        "sample_viewer": (sample_entries, "Sample Viewer"),
        "outlier_analysis": (outlier_entries, "Outlier & Extreme Case Analysis"),
    }
    return {key: value for key, value in registry.items() if value[0]}


def _selected_section_keys(
    sections: Sequence[str] | str,
    *,
    registry: Mapping[str, object],
) -> list[str]:
    """Validate explicit section selection against the fixed approved order."""
    approved = SINGLE_MODEL_EVALUATION_SECTION_KEYS
    if sections == "all":
        return [key for key in approved if key in registry]
    if isinstance(sections, str) or not isinstance(sections, Sequence) or not sections:
        msg = "sections must be 'all' or a non-empty sequence of section keys."
        raise TypeError(msg)
    selected = tuple(sections)
    if not all(isinstance(key, str) for key in selected):
        msg = "sections must contain only string keys."
        raise TypeError(msg)
    if len(selected) != len(set(selected)):
        msg = "sections must not contain duplicate keys."
        raise ValueError(msg)
    unknown = sorted(set(selected).difference(approved))
    if unknown:
        msg = f"Unknown evaluation sections: {unknown}."
        raise ValueError(msg)
    unavailable = [key for key in selected if key not in registry]
    if unavailable:
        msg = f"Evaluation sections are unavailable for these artifacts: {unavailable}."
        raise ValueError(msg)
    return list(selected)


def _build_panel(
    *,
    datasets: Mapping[str, pd.DataFrame],
    comparison: bool,
    sections: Sequence[str] | str,
) -> widgets.Widget:
    """Build one lazy panel without an outer dataset selector."""
    dataset_map = dict(datasets)
    registry = _build_sections(dataset_map, comparison=comparison)
    section_keys = _selected_section_keys(sections, registry=registry)
    if not section_keys:
        msg = "The selected artifacts provide no approved evaluation sections."
        raise ValueError(msg)
    export_state = {"fig": None, "plot_name": None, "title": None}
    ui_sections = [
        analysis.ui.notebook.make_dropdown_section(
            registry[key][0],
            export_state=export_state,
            select_first=True,
        )
        for key in section_keys
    ]
    open_text = "Model comparison - Open Evaluation" if comparison else "Single model - Open Evaluation"
    return analysis.ui.notebook.make_lazy_panel_with_tabs(
        ui_sections,
        tab_titles=[registry[key][1] for key in section_keys],
        open_btn_text=open_text,
        close_btn_text="Close",
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )


def _normalize_contexts(
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    *,
    comparison: bool,
) -> tuple[EvaluationContext, ...]:
    """Validate ordered contexts and exact live-session frame ownership."""
    if session.closed:
        msg = "Cannot build an evaluation panel from a closed session."
        raise analysis.evaluation.session.EvaluationSessionClosedError(msg)
    if isinstance(contexts, (str, bytes)) or not isinstance(contexts, Sequence) or not contexts:
        msg = "contexts must be a non-empty ordered sequence of EvaluationContext values."
        raise TypeError(msg)
    canonical_frames = tuple(session.canonical_frames.values())
    normalized = []
    for position, context in enumerate(contexts):
        if not isinstance(context, EvaluationContext):
            msg = f"contexts[{position}] must be an EvaluationContext."
            raise TypeError(msg)
        key = context.key.strip() if isinstance(context.key, str) else ""
        label = context.label.strip() if isinstance(context.label, str) else ""
        if not key or not label:
            msg = "Every evaluation context requires non-blank key and label text."
            raise ValueError(msg)
        datasets = dict(context.datasets)
        if comparison and len(datasets) < _MINIMUM_COMPARISON_RUNS:
            msg = f"Evaluation context {key!r} must contain at least two models."
            raise ValueError(msg)
        if not comparison and len(datasets) != 1:
            msg = f"Evaluation context {key!r} must contain exactly one model."
            raise ValueError(msg)
        if any(not isinstance(name, str) or not name.strip() for name in datasets):
            msg = f"Evaluation context {key!r} contains a blank or non-string model label."
            raise TypeError(msg)
        for model_label, frame in datasets.items():
            if not any(frame is canonical for canonical in canonical_frames):
                msg = f"Evaluation context {key!r} frame {model_label!r} is not an exact canonical frame owned by the supplied session."
                raise ValueError(msg)
        analysis.evaluation.dataframe.validate_comparison(datasets)
        normalized.append(EvaluationContext(key=key, label=label, datasets=datasets))
    keys = tuple(context.key for context in normalized)
    labels = tuple(context.label for context in normalized)
    if len(keys) != len(set(keys)):
        msg = "Evaluation context keys must be unique."
        raise ValueError(msg)
    if len(labels) != len(set(labels)):
        msg = "Evaluation context labels must be unique."
        raise ValueError(msg)
    return tuple(normalized)


def build_single_model_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
) -> widgets.Widget:
    """Build one historical single-model panel containing every supplied dataset."""
    normalized = _normalize_contexts(session, contexts, comparison=False)
    model_labels = tuple(next(iter(context.datasets)) for context in normalized)
    if len(set(model_labels)) != 1:
        msg = "Single-model contexts must all refer to the same presentation model label."
        raise ValueError(msg)
    datasets = {context.label: next(iter(context.datasets.values())) for context in normalized}
    return _build_panel(datasets=datasets, comparison=False, sections=sections)


def build_comparison_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
) -> widgets.Widget:
    """Build separate historical role-local comparison panels without a selector."""
    normalized = _normalize_contexts(session, contexts, comparison=True)
    panels = [_build_panel(datasets=context.datasets, comparison=True, sections=sections) for context in normalized]
    if len(panels) == 1:
        return panels[0]
    children: list[widgets.Widget] = []
    for context, panel in zip(normalized, panels, strict=True):
        children.extend((widgets.HTML(f"<h3>{escape(context.label)}</h3>"), panel))
    return widgets.VBox(children)
