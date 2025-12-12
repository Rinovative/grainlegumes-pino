"""
Reusable UI components for interactive scientific viewers.

This module contains small, stateless widget constructors that are
used by higher-level navigator functions in util_plot.py.
"""

from __future__ import annotations

import ipywidgets as widgets

# =============================================================================
# GENERIC BUILDING BLOCKS (internal use only)
# =============================================================================


def _build_dropdown(
    *,
    options: list[str],
    value: str,
    description: str,
    width: str,
) -> widgets.Dropdown:
    """
    Create internal generic dropdown builder.

    Parameters
    ----------
    options : list[str]
        Dropdown options.
    value : str
        Default selected value.
    description : str
        Dropdown label.
    width : str
        CSS width of the dropdown.

    Returns
    -------
    widgets.Dropdown
        Configured dropdown widget.

    """
    return widgets.Dropdown(
        options=options,
        value=value,
        description=description,
        layout=widgets.Layout(width=width),
    )


def _build_radio(
    *,
    options: list[str],
    value: str,
    width: str,
    margin: str | None = None,
) -> widgets.RadioButtons:
    """
    Create internal generic radio-button builder.

    Parameters
    ----------
    options : list[str]
        Radio button options.
    value : str
        Default selected value.
    width : str
        CSS width of the radio button group.
    margin : str | None, optional
        CSS margin around the radio button group, by default None.

    Returns
    -------
    widgets.RadioButtons
        Configured radio button widget.

    """
    layout: dict[str, str] = {"width": width}
    if margin is not None:
        layout["margin"] = margin

    return widgets.RadioButtons(
        options=options,
        value=value,
        layout=layout,
    )


def _build_int_step_control(
    *,
    value: int,
    minimum: int,
    maximum: int,
    step: int,
    description: str,
    width: str,
    prev_label: str,
    next_label: str,
) -> tuple[widgets.IntText | widgets.IntSlider, widgets.Button, widgets.Button]:
    """
    Create internal generic integer step control builder.

    Parameters
    ----------
    value : int
        Initial value.
    minimum : int
        Minimum value.
    maximum : int
        Maximum value.
    step : int
        Step size.
    description : str
        Control label.
    width : str
        CSS width of the control.
    prev_label : str
        Label for the "previous" button.
    next_label : str
        Label for the "next" button.

    Returns
    -------
    tuple[widgets.IntText | widgets.IntSlider, widgets.Button, widgets.Button]
        Control widget, previous button, next button.

    """
    if step == 1:
        # discrete index → text input
        control: widgets.IntText | widgets.IntSlider = widgets.BoundedIntText(
            value=value,
            min=minimum,
            max=maximum,
            description=description,
            layout=widgets.Layout(width=width),
        )
    else:
        # aggregation / count → slider
        control = widgets.IntSlider(
            value=value,
            min=minimum,
            max=maximum,
            step=step,
            description=description,
            continuous_update=False,
            readout=True,
        )

    prev_btn = widgets.Button(description=prev_label)
    next_btn = widgets.Button(description=next_label)

    return control, prev_btn, next_btn


# =============================================================================
# SEMANTIC NAVIGATION COMPONENTS
# =============================================================================


def ui_step_case_index(
    *,
    n_cases: int,
    start_idx: int = 0,
) -> tuple[widgets.BoundedIntText, widgets.Button, widgets.Button]:
    """
    Step control for selecting individual case index.

    0-based index internally, but 1-based display to user.

    Parameters
    ----------
    n_cases : int
        Total number of cases.
    start_idx : int, optional
        Initial case index (0-based, default: 0).

    Returns
    -------
    tuple[widgets.BoundedIntText, widgets.Button, widgets.Button]
        Control widget, previous button, next button.

    """
    control, prev_btn, next_btn = _build_int_step_control(
        value=start_idx + 1,
        minimum=1,
        maximum=n_cases,
        step=1,
        description="Case:",
        width="140px",
        prev_label="←",
        next_label="→",
    )

    prev_btn.layout = widgets.Layout(width="40px")
    next_btn.layout = widgets.Layout(width="40px")

    return control, prev_btn, next_btn  # type: ignore[return-value]


def ui_step_case_count(
    *,
    start_cases: int,
    min_cases: int,
    max_cases: int,
    step_size: int,
) -> tuple[widgets.IntSlider, widgets.Button, widgets.Button]:
    """
    Step control for selecting number of cases to display.

    Parameters
    ----------
    start_cases : int
        Initial number of cases.
    min_cases : int
        Minimum number of cases.
    max_cases : int
        Maximum number of cases.
    step_size : int
        Step size for increasing/decreasing case count.

    Returns
    -------
    tuple[widgets.IntSlider, widgets.Button, widgets.Button]
        Control widget, previous button, next button.

    """
    control, prev_btn, next_btn = _build_int_step_control(
        value=start_cases,
        minimum=min_cases,
        maximum=max_cases,
        step=step_size,
        description="Cases:",
        width="auto",
        prev_label="⟨",
        next_label="⟩",
    )
    return control, prev_btn, next_btn  # type: ignore[return-value]


# =============================================================================
# SEMANTIC DROPDOWN SELECTORS
# =============================================================================


def ui_dropdown_dataset(names: list[str]) -> widgets.Dropdown:
    """
    Dropdown selector for dataset names.

    Parameters
    ----------
    names : list[str]
        Available dataset names.

    Returns
    -------
    widgets.Dropdown
        Configured dataset dropdown.

    """
    return _build_dropdown(
        options=names,
        value=names[0],
        description="Dataset:",
        width="auto",
    )


def ui_dropdown_channel(
    *,
    channels: list[str] | None = None,
    default: str = "U",
) -> widgets.Dropdown:
    """
    Dropdown selector for data channels.

    Parameters
    ----------
    channels : list[str] | None, optional
        Available channels. If None, defaults to ["p", "u", "v", "U"].
    default : str, optional
        Default selected channel, by default "U".

    Returns
    -------
    widgets.Dropdown
        Configured channel dropdown.

    """
    channels = channels or ["p", "u", "v", "U"]

    return _build_dropdown(
        options=channels,
        value=default,
        description="Channel:",
        width="auto",
    )


# =============================================================================
# SEMANTIC RADIO SELECTORS
# =============================================================================


def ui_radio_error_mode() -> widgets.RadioButtons:
    """
    Radio button selector for error mode (MAE vs. Relative).

    Returns
    -------
    widgets.RadioButtons
        Configured error mode radio buttons.

    """
    return _build_radio(
        options=["MAE", "Relative [%]"],
        value="MAE",
        width="90px",
        margin="0 0 0 12px",
    )


def ui_radio_kappa_scale() -> widgets.RadioButtons:
    """
    Radio button selector for permeability scaling.

    Options
    -------
    - "kappa"       : physical permeability [m²]
    - "log10(kappa)": logarithmic permeability

    Returns
    -------
    widgets.RadioButtons
        Configured kappa scaling radio buttons.

    """
    return _build_radio(
        options=["kappa", "log10(kappa)"],
        value="log10(kappa)",
        width="100px",
        margin="0 0 0 12px",
    )


# =============================================================================
# OUTPUT CONTAINER
# =============================================================================


def ui_output_plot() -> widgets.Output:
    """
    Output container for plots.

    Returns
    -------
    widgets.Output
        Configured output widget.

    """
    return widgets.Output()
