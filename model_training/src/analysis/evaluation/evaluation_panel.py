"""
===============================================================================
evaluation_panel.py
===============================================================================
Build interactive evaluation panels with tabbed sections and export.

Responsibilities:
  - Register plot functions into named evaluation sections
  - Assemble dropdown sections into tabbed panels
  - Pass export state through notebook UI helpers

Design principles:
  - Panels compose existing plot functions
  - Section registration is declarative
  - Widget state stays local to the panel builder

Boundaries:
  - Plot rendering belongs to analysis.evaluation.plots
  - Widget primitives belong to analysis.ui
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import analysis

if TYPE_CHECKING:
    from collections.abc import Callable

    import ipywidgets as widgets


# =====================================================================
# Section registry
# =====================================================================
def _build_sections(toggle: Callable[[str, Callable[..., object]], widgets.Widget]) -> dict[str, tuple[list[widgets.Widget], str]]:
    """
    Build the registry of available sections.

    Parameters
    ----------
    toggle : function
        Shortcut function to build toggles for plots.

    Returns
    -------
    dict
        Mapping {section_key: (list_of_toggle_widgets, tab_title)}.

    """
    return {
        # --------------------------------------------------------------
        "overview": (
            [
                toggle(
                    "Overview: Summary table",
                    analysis.evaluation.plots.overview.plot_overview_global_summary_table,
                ),
                toggle(
                    "Overview: Global comparison summary",
                    analysis.evaluation.plots.overview.plot_overview_scoreboard,
                ),
                toggle(
                    "Overview: Pareto (Error vs Physics)",
                    analysis.evaluation.plots.overview.plot_overview_pareto_error_vs_physics,
                ),
                toggle(
                    "Overview: Architecture & hyperparameter table",
                    analysis.evaluation.plots.overview.plot_overview_architecture_table,
                ),
            ],
            "Overview",
        ),
        # --------------------------------------------------------------
        "global_error": (
            [
                toggle("1-1. Global error metrics", analysis.evaluation.plots.global_error.plot_global_error_metrics),
                toggle("1-2. Global error distribution", analysis.evaluation.plots.global_error.plot_error_distribution),
                toggle("1-3. GT vs Prediction (mean)", analysis.evaluation.plots.global_error.plot_global_gt_vs_pred),
                toggle("1-4. Mean error maps", analysis.evaluation.plots.global_error.plot_mean_error_maps),
                toggle("1-5. Std error maps", analysis.evaluation.plots.global_error.plot_std_error_maps),
            ],
            "Global Error Analysis",
        ),
        # --------------------------------------------------------------
        "architecture": (
            [
                toggle(
                    "2-1. Error vs architecture parameters",
                    analysis.evaluation.plots.architecture.plot_error_vs_architecture_parameters,
                ),
                toggle(
                    "2-2. Capacity vs performance", analysis.evaluation.plots.architecture.plot_capacity_vs_performance
                ),
                toggle("2-3. Parameter efficiency", analysis.evaluation.plots.architecture.plot_parameter_efficiency),
            ],
            "Architecture Sensitivity",
        ),
        # --------------------------------------------------------------
        "error_decomposition": (
            [
                toggle("3-1. Error vs |GT| magnitude", analysis.evaluation.plots.error_decomposition.plot_error_vs_gt_magnitude),
                toggle(
                    "3-2. Boundary vs interior error", analysis.evaluation.plots.error_decomposition.plot_error_vs_boundary_distance
                ),
            ],
            "Error Decomposition",
        ),
        # --------------------------------------------------------------
        "physical_consistency": (
            [
                toggle(
                    "4-1. Physical consistency summary table",
                    analysis.evaluation.plots.physical_consistency.plot_physical_consistency_summary_table,
                ),
                toggle(
                    "4-2. Physical consistency CDF grid (2x2)",
                    analysis.evaluation.plots.physical_consistency.plot_physical_consistency_cdf_grid,
                ),
                toggle(
                    "4-3. Velocity divergence (∇·u)",
                    analysis.evaluation.plots.physical_consistency.plot_velocity_divergence,
                ),
                toggle(
                    "4-4. Mass conservation error map",
                    analysis.evaluation.plots.physical_consistency.plot_mass_conservation_error_map,
                ),
                toggle(
                    "4-5. Darcy-Brinkman operator residual",
                    analysis.evaluation.plots.physical_consistency.plot_brinkman_residual,
                ),
                toggle(
                    "4-6. Darcy-Brinkman momentum residual map",
                    analysis.evaluation.plots.physical_consistency.plot_brinkman_momentum_residual_map,
                ),
                toggle(
                    "4-7. Pressure drop consistency (Δp)",
                    analysis.evaluation.plots.physical_consistency.plot_pressure_drop_consistency,
                ),
                toggle(
                    "4-8. Pressure boundary consistency (p_bc)",
                    analysis.evaluation.plots.physical_consistency.plot_pressure_bc_consistency,
                ),
                toggle(
                    "4-9. Porosity-weighted continuity residual map (∇·(εu))",
                    analysis.evaluation.plots.physical_consistency.plot_div_eps_u_error_map,
                ),
            ],
            "Physical Consistency",
        ),
        # --------------------------------------------------------------
        "spectral": (
            [
                toggle(
                    "5-1. Demand vs prediction + error",
                    analysis.evaluation.plots.spectral.plot_spectral_demand_prediction_error,
                ),
                toggle(
                    "5-2. Spectral transfer ratio (Pred/GT)",
                    analysis.evaluation.plots.spectral.plot_spectral_transfer_ratio,
                ),
                toggle(
                    "5-3. Learned layer x frequency heatmap",
                    analysis.evaluation.plots.spectral.plot_learned_layer_frequency_heatmap,
                ),
            ],
            "Spectral & Representation Analysis",
        ),
        "error_sensitivity": (
            [
                toggle(
                    "6-1. Parameter-error correlation (heatmap)",
                    analysis.evaluation.plots.parameter_sensitivity.plot_parameter_error_heatmap,
                ),
                toggle(
                    "6-2. Error vs input parameter (binned trend)",
                    analysis.evaluation.plots.parameter_sensitivity.plot_error_vs_parameter_trend,
                ),
            ],
            "Error Sensitivity",
        ),
        # --------------------------------------------------------------
        "sample_viewer": (
            [
                toggle("7-1. Sample GT vs Prediction", analysis.evaluation.plots.sample_viewer.plot_sample_prediction_overview),
                toggle(
                    "7-2. Kappa tensor with error overlay",
                    analysis.evaluation.plots.sample_viewer.plot_sample_kappa_tensor_with_overlay,
                ),
                toggle(
                    "7-3. Pressure & velocity field comparison",
                    analysis.evaluation.plots.sample_viewer.plot_pu_two_model_comparison,
                ),
            ],
            "Sample Viewer",
        ),
        # --------------------------------------------------------------
        "outliers": (
            [
                toggle(
                    "8-1. Worst per-channel cases (tables)",
                    analysis.evaluation.plots.outliers.plot_outlier_tables_per_channel,
                ),
                toggle(
                    "8-2. Worst per-channel cases (field plots)",
                    analysis.evaluation.plots.outliers.plot_outlier_cases_per_channel,
                ),
                toggle(
                    "8-3. Extreme input parameters (table view)", analysis.evaluation.plots.outliers.plot_extreme_input_table
                ),
                toggle(
                    "8-4. Extreme input parameter cases (field plots)",
                    analysis.evaluation.plots.outliers.plot_extreme_input_cases,
                ),
            ],
            "Outlier & Extreme Case Analysis",
        ),
    }


# =====================================================================
# Public API
# =====================================================================
def build_evaluation_panel(
    *,
    datasets_eval: dict,
    title: str,
    sections: list[str] | str = "all",
) -> widgets.Widget:
    """
    Build an evaluation panel from an explicit list of sections.

    Parameters
    ----------
    datasets_eval : dict
        Mapping {label: eval_dataframe}
    title : str
        Title shown on open button
    sections : list[str] or "all"
        Which sections to include

    """
    toggle = analysis.ui.notebook.make_toggle_shortcut(dfs=datasets_eval)
    registry = _build_sections(toggle)

    section_keys = list(registry.keys()) if sections == "all" else sections

    export_state = {"fig": None, "plot_name": None, "title": None}

    ui_sections = []
    tab_titles = []

    for key in section_keys:
        plots, tab_title = registry[key]
        ui_sections.append(analysis.ui.notebook.make_dropdown_section(plots, export_state=export_state))
        tab_titles.append(tab_title)

    return analysis.ui.notebook.make_lazy_panel_with_tabs(
        ui_sections,
        tab_titles=tab_titles,
        open_btn_text=f"{title} - Open Evaluation",
        close_btn_text="Close",
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )
