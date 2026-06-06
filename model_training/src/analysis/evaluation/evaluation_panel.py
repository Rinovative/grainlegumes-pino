"""
===============================================================================
evaluation_panel.py
===============================================================================
Builder for interactive evaluation panels with tabbed sections and export.

Provides:
  - build_evaluation_panel: construct tabbed panel from plot sections
  - support for collapsible panels with open/close buttons
  - PDF export integration via analysis_ui_notebook

Responsibilities:
  - organize multiple evaluation plot sections into a single interactive interface
  - manage panel visibility and user interaction
  - coordinate with analysis_ui_notebook for figure export

This module does NOT:
  - contain plot functions (those belong in evaluation_plot_*.py)
  - handle figure creation or rendering (delegates to plot functions)
  - manage dataset selection or case navigation (uses pre-made viewers)
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import analysis
from src.analysis import evaluation

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
                    evaluation.evaluation_plot.evaluation_plot_overview_scoreboard.plot_overview_global_summary_table,
                ),
                toggle(
                    "Overview: Global comparison summary",
                    evaluation.evaluation_plot.evaluation_plot_overview_scoreboard.plot_overview_scoreboard,
                ),
                toggle(
                    "Overview: Pareto (Error vs Physics)",
                    evaluation.evaluation_plot.evaluation_plot_overview_scoreboard.plot_overview_pareto_error_vs_physics,
                ),
                toggle(
                    "Overview: Architecture & hyperparameter table",
                    evaluation.evaluation_plot.evaluation_plot_overview_scoreboard.plot_overview_architecture_table,
                ),
            ],
            "Overview",
        ),
        # --------------------------------------------------------------
        "global_error": (
            [
                toggle("1-1. Global error metrics", evaluation.evaluation_plot.evaluation_plot_global_error_analysis.plot_global_error_metrics),
                toggle("1-2. Global error distribution", evaluation.evaluation_plot.evaluation_plot_global_error_analysis.plot_error_distribution),
                toggle("1-3. GT vs Prediction (mean)", evaluation.evaluation_plot.evaluation_plot_global_error_analysis.plot_global_gt_vs_pred),
                toggle("1-4. Mean error maps", evaluation.evaluation_plot.evaluation_plot_global_error_analysis.plot_mean_error_maps),
                toggle("1-5. Std error maps", evaluation.evaluation_plot.evaluation_plot_global_error_analysis.plot_std_error_maps),
            ],
            "Global Error Analysis",
        ),
        # --------------------------------------------------------------
        "architecture": (
            [
                toggle(
                    "2-1. Error vs architecture parameters",
                    evaluation.evaluation_plot.evaluation_plot_architecture_sensitivity.plot_error_vs_architecture_parameters,
                ),
                toggle(
                    "2-2. Capacity vs performance", evaluation.evaluation_plot.evaluation_plot_architecture_sensitivity.plot_capacity_vs_performance
                ),
                toggle("2-3. Parameter efficiency", evaluation.evaluation_plot.evaluation_plot_architecture_sensitivity.plot_parameter_efficiency),
            ],
            "Architecture Sensitivity",
        ),
        # --------------------------------------------------------------
        "error_decomposition": (
            [
                toggle("3-1. Error vs |GT| magnitude", evaluation.evaluation_plot.evaluation_plot_error_decomposition.plot_error_vs_gt_magnitude),
                toggle(
                    "3-2. Boundary vs interior error", evaluation.evaluation_plot.evaluation_plot_error_decomposition.plot_error_vs_boundary_distance
                ),
            ],
            "Error Decomposition",
        ),
        # --------------------------------------------------------------
        "physical_consistency": (
            [
                toggle(
                    "4-1. Physical consistency summary table",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_physical_consistency_summary_table,
                ),
                toggle(
                    "4-2. Physical consistency CDF grid (2x2)",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_physical_consistency_cdf_grid,
                ),
                toggle(
                    "4-3. Velocity divergence (∇·u)",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_velocity_divergence,
                ),
                toggle(
                    "4-4. Mass conservation error map",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_mass_conservation_error_map,
                ),
                toggle(
                    "4-5. Darcy-Brinkman operator residual",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_brinkman_residual,
                ),
                toggle(
                    "4-6. Darcy-Brinkman momentum residual map",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_brinkman_momentum_residual_map,
                ),
                toggle(
                    "4-7. Pressure drop consistency (Δp)",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_pressure_drop_consistency,
                ),
                toggle(
                    "4-8. Pressure boundary consistency (p_bc)",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_pressure_bc_consistency,
                ),
                toggle(
                    "4-9. Porosity-weighted continuity residual map (∇·(εu))",
                    evaluation.evaluation_plot.evaluation_plot_physical_consistency.plot_div_eps_u_error_map,
                ),
            ],
            "Physical Consistency",
        ),
        # --------------------------------------------------------------
        "spectral": (
            [
                toggle(
                    "5-1. Demand vs prediction + error",
                    evaluation.evaluation_plot.evaluation_plot_spectral_analysis.plot_spectral_demand_prediction_error,
                ),
                toggle(
                    "5-2. Spectral transfer ratio (Pred/GT)",
                    evaluation.evaluation_plot.evaluation_plot_spectral_analysis.plot_spectral_transfer_ratio,
                ),
                toggle(
                    "5-3. Learned layer x frequency heatmap",
                    evaluation.evaluation_plot.evaluation_plot_spectral_analysis.plot_learned_layer_frequency_heatmap,
                ),
            ],
            "Spectral & Representation Analysis",
        ),
        "error_sensitivity": (
            [
                toggle(
                    "6-1. Parameter-error correlation (heatmap)",
                    evaluation.evaluation_plot.evaluation_plot_parameter_sensitivity.plot_parameter_error_heatmap,
                ),
                toggle(
                    "6-2. Error vs input parameter (binned trend)",
                    evaluation.evaluation_plot.evaluation_plot_parameter_sensitivity.plot_error_vs_parameter_trend,
                ),
            ],
            "Error Sensitivity",
        ),
        # --------------------------------------------------------------
        "sample_viewer": (
            [
                toggle("7-1. Sample GT vs Prediction", evaluation.evaluation_plot.evaluation_plot_sample_viewer.plot_sample_prediction_overview),
                toggle(
                    "7-2. Kappa tensor with error overlay",
                    evaluation.evaluation_plot.evaluation_plot_sample_viewer.plot_sample_kappa_tensor_with_overlay,
                ),
                toggle(
                    "7-3. Pressure & velocity field comparison",
                    evaluation.evaluation_plot.evaluation_plot_sample_viewer.plot_pu_two_model_comparison,
                ),
            ],
            "Sample Viewer",
        ),
        # --------------------------------------------------------------
        "outliers": (
            [
                toggle(
                    "8-1. Worst per-channel cases (tables)",
                    evaluation.evaluation_plot.evaluation_plot_outlier_analysis.plot_outlier_tables_per_channel,
                ),
                toggle(
                    "8-2. Worst per-channel cases (field plots)",
                    evaluation.evaluation_plot.evaluation_plot_outlier_analysis.plot_outlier_cases_per_channel,
                ),
                toggle(
                    "8-3. Extreme input parameters (table view)", evaluation.evaluation_plot.evaluation_plot_outlier_analysis.plot_extreme_input_table
                ),
                toggle(
                    "8-4. Extreme input parameter cases (field plots)",
                    evaluation.evaluation_plot.evaluation_plot_outlier_analysis.plot_extreme_input_cases,
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
