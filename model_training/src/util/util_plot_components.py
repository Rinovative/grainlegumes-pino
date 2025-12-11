"""
Reusable UI components for interactive scientific viewers.

This module contains small, stateless widget constructors that are
used by higher-level navigator functions in util_plot.py. Each component
creates exactly one UI element and contains no dataset logic or callbacks.
"""

from __future__ import annotations

import ipywidgets as widgets

# ============================================================================
# Case index selector
# ============================================================================


def ui_case_index(n_cases: int, start_idx: int = 0) -> widgets.BoundedIntText:
    """
    Create a bounded integer selector for choosing a simulation case.

    Parameters
    ----------
    n_cases : int
        Number of available cases.
    start_idx : int, optional
        Zero-based initial index. Displayed value is start_idx + 1.

    Returns
    -------
    widgets.BoundedIntText
        Integer selector with range [1, n_cases].

    """
    return widgets.BoundedIntText(
        value=start_idx + 1,
        min=1,
        max=n_cases,
        description="Case:",
        layout=widgets.Layout(width="140px"),
    )


# ============================================================================
# Previous / Next buttons
# ============================================================================


def ui_prev_next() -> tuple[widgets.Button, widgets.Button]:
    """
    Create previous and next step buttons.

    Returns
    -------
    tuple
        (prev_button, next_button)

    """
    prev_btn = widgets.Button(description="←", layout={"width": "40px"})
    next_btn = widgets.Button(description="→", layout={"width": "40px"})
    return prev_btn, next_btn


# ============================================================================
# Dataset dropdown selector
# ============================================================================


def ui_dataset_dropdown(names: list[str]) -> widgets.Dropdown:
    """
    Create a dropdown widget for selecting among multiple datasets.

    Parameters
    ----------
    names : list of str
        Names of available datasets.

    Returns
    -------
    widgets.Dropdown
        Dropdown widget with the given names.

    """
    return widgets.Dropdown(
        options=names,
        value=names[0],
        description="Dataset:",
        layout=widgets.Layout(width="auto"),
    )


# ============================================================================
# Error mode selector (MAE / Relative %)
# ============================================================================


def ui_error_mode_selector() -> widgets.RadioButtons:
    """
    Create a simple radio-button selector for switching error-map mode.

    Options:
        - MAE
        - Relative [%]

    Returns
    -------
    widgets.RadioButtons
        Stateless radio selector widget.

    """
    return widgets.RadioButtons(
        options=["MAE", "Relative [%]"],
        value="MAE",
        layout={"width": "90px", "margin": "0 0 0 12px"},
    )


# ============================================================================
# Output area for figure rendering
# ============================================================================


def ui_output() -> widgets.Output:
    """
    Create an output container for rendering matplotlib figures.

    Returns
    -------
    widgets.Output
        Output widget used to display plot content.

    """
    return widgets.Output()
