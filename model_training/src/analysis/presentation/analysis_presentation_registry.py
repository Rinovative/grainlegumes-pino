"""
===============================================================================
analysis_presentation_registry.py
===============================================================================
Declare user-facing EDA and evaluation section and plot presentation order.

Responsibilities:
  - Keep section keys, tab names, plot keys, and display names reviewable together
  - Derive consecutive hierarchical labels without changing scientific APIs
  - Validate registry keys and names before notebook panels are constructed

Design principles:
  - Numbering is presentation metadata derived from tuple order
  - Plot functions remain independent from tab position and display numbering
  - Reordering one view changes no file name, function name, or scientific identity

This module does NOT:
  - Wire plot callables or construct semantic notebook controls
  - Implement plot mathematics or admit artifact scientific contracts
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclass(frozen=True)
class PlotPresentation:
    """
    Define one stable plot key and its unnumbered user-facing name.

    Parameters
    ----------
    key : str
        Stable callable-registry identifier; numbering never participates.
    name : str
        Human-readable name before hierarchical numbering is derived.

    Notes
    -----
    Instances are immutable presentation metadata, not scientific plot identity.

    """

    key: str
    name: str


@dataclass(frozen=True)
class SectionPresentation:
    """
    Define one ordered notebook section and its ordered plot choices.

    Parameters
    ----------
    key : str
        Stable section identifier used by panel selection APIs.
    name : str
        Unnumbered user-facing tab name.
    plots : tuple[PlotPresentation, ...]
        Non-empty plot specifications in display order.

    Notes
    -----
    Instances are frozen; tuple order is the sole source of display numbering.

    """

    key: str
    name: str
    plots: tuple[PlotPresentation, ...]


EDA_SECTIONS = (
    SectionPresentation(
        key="metadata_fields",
        name="Metadata & field statistics",
        plots=(
            PlotPresentation("metadata_statistics", "Metadata statistics"),
            PlotPresentation("parameter_distributions", "Parameter distributions"),
            PlotPresentation("field_value_distributions", "Field value distributions"),
        ),
    ),
    SectionPresentation(
        key="spectral_analysis",
        name="Spectral analysis",
        plots=(
            PlotPresentation("isotropic_spectra", "Isotropic spectra and cumulative energy"),
            PlotPresentation("directional_spectra", "Flow and cross-stream directional spectra"),
            PlotPresentation("spectral_evolution", "Cross-stream spectral evolution along flow direction"),
        ),
    ),
)

EVALUATION_SECTIONS = (
    SectionPresentation(
        key="run_summary",
        name="Run summary",
        plots=(
            PlotPresentation("authoritative_run_summary", "Authoritative run summary"),
            PlotPresentation("accuracy_physics_pareto", "Accuracy versus physics Pareto"),
        ),
    ),
    SectionPresentation(
        key="global_error_analysis",
        name="Global error analysis",
        plots=(
            PlotPresentation("predictive_error_distributions", "Predictive error distributions"),
            PlotPresentation("mean_reference_prediction", "Mean reference and prediction fields"),
            PlotPresentation("spatial_error_statistics", "Spatial error statistics"),
        ),
    ),
    SectionPresentation(
        key="error_decomposition",
        name="Error decomposition",
        plots=(
            PlotPresentation("case_mean_bias", "Case-mean prediction bias"),
            PlotPresentation("target_magnitude_error", "Error versus target magnitude"),
            PlotPresentation("boundary_distance_error", "Boundary-distance error"),
        ),
    ),
    SectionPresentation(
        key="physical_consistency",
        name="Physical consistency",
        plots=(
            PlotPresentation("residual_distributions", "Residual distributions"),
            PlotPresentation("residual_maps", "Momentum and dual-continuity maps"),
            PlotPresentation("pressure_boundary_drop", "Pressure boundary and drop diagnostics"),
        ),
    ),
    SectionPresentation(
        key="spectral_fidelity",
        name="Spectral fidelity",
        plots=(PlotPresentation("output_spectra_transfer", "Reference, prediction, error, and transfer spectra"),),
    ),
    SectionPresentation(
        key="architecture_capacity",
        name="Architecture & capacity",
        plots=(PlotPresentation("parameter_count_accuracy", "Exact parameter count versus accuracy"),),
    ),
    SectionPresentation(
        key="input_sensitivity",
        name="Input sensitivity",
        plots=(
            PlotPresentation("metadata_error_associations", "Metadata-error associations"),
            PlotPresentation("metadata_error_trends", "Binned metadata trends with counts"),
        ),
    ),
    SectionPresentation(
        key="sample_viewer",
        name="Sample viewer",
        plots=(
            PlotPresentation("shared_case_comparison", "Shared-case model comparison"),
            PlotPresentation("permeability_error_overlay", "Permeability and local-error overlay"),
        ),
    ),
    SectionPresentation(
        key="outliers_extremes",
        name="Outliers & extreme cases",
        plots=(
            PlotPresentation("outlier_extreme_tables", "Outlier and extreme-case tables"),
            PlotPresentation("worst_case_fields", "Worst-case field viewer"),
            PlotPresentation("extreme_input_fields", "Extreme-input field viewer"),
        ),
    ),
)


def section_display_label(section_index: int, name: str) -> str:
    """
    Format one section label from a positive one-based display position.

    Parameters
    ----------
    section_index : int
        Positive one-based section position.
    name : str
        Non-empty unnumbered section name; surrounding whitespace is removed.

    Returns
    -------
    str
        Label in ``N. Name`` form.

    Raises
    ------
    ValueError
        If the position is non-positive/non-integral or the name is blank.

    """
    if isinstance(section_index, bool) or not isinstance(section_index, int) or section_index <= 0:
        message = f"section_index must be a positive integer, got {section_index!r}."
        raise ValueError(message)
    if not isinstance(name, str) or not name.strip():
        message = "Section display names must be non-empty strings."
        raise ValueError(message)
    return f"{section_index}. {name.strip()}"


def plot_display_label(section_index: int, plot_index: int, name: str) -> str:
    """
    Format one plot label from positive one-based section and plot positions.

    Parameters
    ----------
    section_index, plot_index : int
        Positive one-based hierarchy positions.
    name : str
        Non-empty unnumbered plot name; surrounding whitespace is removed.

    Returns
    -------
    str
        Label in ``N-M. Name`` form.

    Raises
    ------
    ValueError
        If either position is non-positive/non-integral or the name is blank.

    """
    section_display_label(section_index, "section")
    if isinstance(plot_index, bool) or not isinstance(plot_index, int) or plot_index <= 0:
        message = f"plot_index must be a positive integer, got {plot_index!r}."
        raise ValueError(message)
    if not isinstance(name, str) or not name.strip():
        message = "Plot display names must be non-empty strings."
        raise ValueError(message)
    return f"{section_index}-{plot_index}. {name.strip()}"


def validate_registry(sections: Sequence[SectionPresentation]) -> None:
    """
    Validate stable keys, user-facing names, and non-empty ordered sections.

    Parameters
    ----------
    sections : Sequence[SectionPresentation]
        Declarative section registry in public display order.

    Raises
    ------
    TypeError
        If a registry entry has the wrong immutable presentation type.
    ValueError
        If the registry is empty or contains empty/duplicate keys or names.

    """
    if not sections:
        message = "A presentation registry must contain at least one section."
        raise ValueError(message)
    section_keys: set[str] = set()
    plot_keys: set[str] = set()
    for section in sections:
        if not isinstance(section, SectionPresentation):
            message = "Presentation registries must contain SectionPresentation entries."
            raise TypeError(message)
        if not section.key or not section.name.strip() or not section.plots:
            message = "Every presentation section requires a key, name, and at least one plot."
            raise ValueError(message)
        if section.key in section_keys:
            message = f"Duplicate presentation section key: {section.key!r}."
            raise ValueError(message)
        section_keys.add(section.key)
        for plot in section.plots:
            if not isinstance(plot, PlotPresentation):
                message = "Presentation sections must contain PlotPresentation entries."
                raise TypeError(message)
            if not plot.key or not plot.name.strip():
                message = "Every presentation plot requires a non-empty key and name."
                raise ValueError(message)
            if plot.key in plot_keys:
                message = f"Duplicate presentation plot key: {plot.key!r}."
                raise ValueError(message)
            plot_keys.add(plot.key)


def numbered_registry(
    sections: Sequence[SectionPresentation],
) -> Iterator[tuple[SectionPresentation, str, tuple[tuple[PlotPresentation, str], ...]]]:
    """
    Yield registry entries with labels derived only from public display order.

    Parameters
    ----------
    sections : Sequence[SectionPresentation]
        Declarative section registry in the order shown to notebook users.

    Yields
    ------
    tuple
        Section specification, numbered tab label, and ordered plot
        specifications paired with numbered dropdown labels.

    """
    validate_registry(sections)
    for section_index, section in enumerate(sections, start=1):
        plots = tuple((plot, plot_display_label(section_index, plot_index, plot.name)) for plot_index, plot in enumerate(section.plots, start=1))
        yield section, section_display_label(section_index, section.name), plots


validate_registry(EDA_SECTIONS)
validate_registry(EVALUATION_SECTIONS)
