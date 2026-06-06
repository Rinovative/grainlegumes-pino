"""
===============================================================================
analysis_ui_components.py
===============================================================================
Reusable UI components and plotting helpers for interactive scientific viewers.

Provides:
  - CheckboxGroup protocol for widget grouping
  - dropdown and radio button constructors
  - checkbox group construction
  - step control widgets (case index, case count sliders)
  - output plot containers
  - colorbar formatter selection and axis label application
  - streamline overlay utilities

Design principles:
  - all constructors are stateless and return ipywidgets.Widget instances
  - no semantic logic or navigation state
  - used exclusively by analysis_ui_viewers for viewer construction
  - colorbar formatters are automatically selected based on data range
  - axis labels are applied consistently across plot types

This module does NOT:
  - contain navigation logic or re-render callbacks (use analysis_ui_viewers)
  - handle notebook execution or display lifecycle
  - contain plot functions or data processing
===============================================================================
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import ipywidgets as widgets
import matplotlib.ticker as mticker
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# =============================================================================
# TYPE CONTRACTS
# =============================================================================


class CheckboxGroup(Protocol):
    """
    Protocol for a group of checkboxes contained in a VBox.

    Attributes
    ----------
    boxes : dict[str, widgets.Checkbox]
        Mapping from checkbox label to checkbox widget.

    """

    boxes: dict[str, widgets.Checkbox]


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
    description: str | None = None,
    description_width: str = "initial",
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
    description : str | None, optional
        Optional label shown left of the radio group.
    description_width : str, optional
        CSS-like width for the description area, default "initial".

    Returns
    -------
    widgets.RadioButtons
        Configured radio button widget.

    """
    layout_kwargs: dict[str, str] = {"width": width}
    if margin is not None:
        layout_kwargs["margin"] = margin

    return widgets.RadioButtons(
        options=options,
        value=value,
        description=description or "",
        layout=widgets.Layout(**layout_kwargs),
        style={"description_width": description_width},
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


def _build_checkbox_group(
    *,
    options: list[str],
    defaults: list[str],
    description: str | None = None,
    n_cols: int = 2,
) -> widgets.VBox:
    """
    Create internal generic checkbox group builder.

    Parameters
    ----------
    options : list[str]
        Available checkbox options.
    defaults : list[str]
        Options enabled by default.
    description : str | None, optional
        Optional group label shown above the checkboxes.
    n_cols : int, optional
        Number of columns for checkbox layout (default: 2).

    Returns
    -------
    widgets.VBox
        VBox containing the checkbox group.

    Notes
    -----
    The returned VBox exposes a public `boxes` attribute
    mapping option -> Checkbox widget.

    """
    boxes = {
        opt: widgets.Checkbox(
            value=opt in defaults,
            description=opt,
            indent=False,
            layout=widgets.Layout(
                margin="0px",
                width="auto",
            ),
            style={"description_width": "auto"},
        )
        for opt in options
    }

    # -------------------------------------------------
    # Layout: grid (e.g. p u / v U)
    # -------------------------------------------------
    grid = widgets.GridBox(
        children=list(boxes.values()),
        layout=widgets.Layout(
            grid_template_columns=" ".join(["auto"] * n_cols),
            grid_gap="0px 15px",
        ),
    )

    children: list[widgets.Widget] = []
    if description is not None:
        children.append(widgets.Label(description))

    children.append(grid)

    box = widgets.VBox(
        children,
        layout=widgets.Layout(
            margin="0px 0px 0px 25px",
        ),
    )
    box.boxes = boxes
    return box


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
        description="Select:",
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


def ui_dropdown_input_parameter(
    *,
    parameters: list[str],
    default: str | None = None,
) -> widgets.Dropdown:
    """
    Dropdown selector for input parameters (par_*).

    Parameters
    ----------
    parameters : list[str]
        Available input parameters.
    default : str | None, optional
        Default selected parameter. If None, first entry is used.

    Returns
    -------
    widgets.Dropdown
        Configured input-parameter dropdown.

    """
    if not parameters:
        msg = "No input parameters available for dropdown."
        raise ValueError(msg)

    return _build_dropdown(
        options=parameters,
        value=default or parameters[0],
        description="Parameter:",
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
        description="Error mode:",
        description_width="initial",
    )


def ui_radio_pred_scale_mode() -> widgets.RadioButtons:
    """
    Radio button selector for prediction/GT scaling.

    Options
    -------
    - "Independent" : pred and GT get their own scales (current behaviour)
    - "Shared (GT)" : pred uses GT scale; pred outliers outside GT range are masked

    Returns
    -------
    widgets.RadioButtons
        Configured scale-mode radio buttons.

    """
    return _build_radio(
        options=["Independent", "Shared (GT)"],
        value="Independent",
        width="auto",
        margin="0 0 0 12px",
        description="Pred/GT scale:",
        description_width="initial",
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
        description="Kappa scale:",
        description_width="initial",
    )


# =============================================================================
# SEMANTIC CHECKBOX SELECTORS
# =============================================================================


def ui_checkbox_channels(
    *,
    channels: list[str] | None = None,
    default_on: list[str] | None = None,
) -> widgets.VBox:
    """
    Checkbox selector for output channels.

    Parameters
    ----------
    channels : list[str] | None, optional
        Available channels. Defaults to ["p", "u", "v", "U"].
    default_on : list[str] | None, optional
        Channels enabled by default. Defaults to all channels.

    Returns
    -------
    widgets.VBox
        Checkbox group for channel selection.

    Notes
    -----
    The returned widget exposes an internal `_boxes` attribute
    mapping channel name -> Checkbox. This is used by plot
    functions to determine which channels are active.

    """
    channels = channels or ["p", "u", "v", "U"]
    default_on = default_on or channels

    return _build_checkbox_group(
        options=channels,
        defaults=default_on,
    )


def ui_checkbox_datasets(
    *,
    dataset_names: list[str],
    default_on: list[str] | None = None,
) -> widgets.VBox:
    """
    Checkbox selector for datasets.

    Parameters
    ----------
    dataset_names : list[str]
        Available dataset names.
    default_on : list[str] | None, optional
        Datasets enabled by default. Defaults to all datasets.

    Returns
    -------
    widgets.VBox
        Checkbox group for dataset selection.

    Notes
    -----
    The returned widget exposes a public `boxes` attribute
    mapping dataset name -> Checkbox widget.

    """
    default_on = default_on or dataset_names

    return _build_checkbox_group(
        options=dataset_names,
        defaults=default_on,
    )


def ui_checkbox_log_scale(
    *,
    description: str = "log10 for scale parameters",
    default: bool = False,
) -> widgets.Checkbox:
    """
    Checkbox selector for enabling log10 scaling.

    Parameters
    ----------
    description : str, optional
        Checkbox label.
    default : bool, optional
        Default checkbox state.

    Returns
    -------
    widgets.Checkbox
        Configured log-scale checkbox.

    """
    return widgets.Checkbox(
        value=default,
        description=description,
    )


def ui_checkbox_normalise(
    *,
    description: str = "Normalise",
    default: bool = True,
    width: str = "160px",
) -> widgets.Checkbox:
    """
    Checkbox selector for normalisation toggles.

    Parameters
    ----------
    description : str, optional
        Checkbox label.
    default : bool, optional
        Default checkbox state.
    width : str, optional
        CSS width.

    Returns
    -------
    widgets.Checkbox
        Configured checkbox.

    """
    return widgets.Checkbox(
        value=default,
        description=description,
        indent=False,
        layout=widgets.Layout(width=width),
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


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

# COLORBAR FORMATTERS


def _max_decimals_from_ticks(ticks: np.ndarray, *, cap: int = 6) -> int:
    """
    Determine the maximum number of decimals needed across ticks AFTER stripping trailing zeros per tick.

    Example:
        ticks contain 0.18 -> max_decimals = 2 -> show 0.20 (not 0.2)

    Parameters
    ----------
    ticks : np.ndarray
        Tick values.
    cap : int, optional
        Maximum number of decimals to consider (default: 6).

    Returns
    -------
    int
        Maximum number of decimals needed.

    """
    t = np.asarray(ticks, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return 0

    max_dec = 0
    for x in t:
        s = f"{x:.{cap}f}".rstrip("0").rstrip(".")
        if "." in s:
            max_dec = max(max_dec, len(s.split(".")[1]))
    return max_dec


def choose_colorbar_formatter(
    vmin: float,
    vmax: float,
    *,
    ticks: np.ndarray | None = None,
) -> mticker.Formatter:
    """
    Show consistent tick labels with trailing zeros kept.

    - If scientific notation is used (very small/large magnitude), use "%.2e".
    - Otherwise, choose a fixed number of decimals.
      If `ticks` is provided, decimals are derived from the smallest tick step so that
      labels like 0.20 are shown when 0.18 exists (same decimals across all ticks).

    Parameters
    ----------
    vmin : float
        Minimum colorbar value.
    vmax : float
        Maximum colorbar value.
    ticks : np.ndarray | None
        Optional tick values to derive a stable decimal count.

    Returns
    -------
    matplotlib.ticker.Formatter
        Formatter instance.

    """
    sig = 3
    sci_low, sci_high = -3, 3  # scientific if exponent <= -3 or >= 3

    vr = max(abs(vmin), abs(vmax))

    # Handle all-zero (or invalid) ranges
    if vr == 0 or not math.isfinite(vr):
        return mticker.FormatStrFormatter("%.2f")

    exp = math.floor(math.log10(vr))

    # Scientific notation
    if exp <= sci_low or exp >= sci_high:
        return mticker.FormatStrFormatter(f"%.{sig - 1}e")

    # Fixed decimals
    decimals_default = max(0, (sig - 1) - exp)
    decimals = decimals_default

    if ticks is not None:
        t = np.asarray(ticks, dtype=float)
        t = t[np.isfinite(t)]
        t = np.unique(np.sort(t))

        if t.size >= 2:  # noqa: PLR2004
            diffs = np.diff(t)
            diffs = diffs[diffs > 0]

            if diffs.size > 0:
                step = float(np.min(diffs))
                # decimals so that step is representable (keep trailing zeros)
                # e.g. step=0.05 -> 2 decimals, step=0.1 -> 1 decimal
                decimals_from_step = int(max(0, math.ceil(-math.log10(step) - 1e-12)))
                decimals = max(decimals_default, decimals_from_step)

    def _fmt(x: float, _pos: int | None = None) -> str:
        s = f"{x:.{decimals}f}"

        # Avoid "-0.00"
        if float(s) == 0.0:
            s = s.lstrip("-")

        return s

    return mticker.FuncFormatter(_fmt)


# CONTOUR LEVELS


_MIN_LEVEL_COUNT = 2


def compute_levels(arr: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute contour levels for given data array.

    Parameters
    ----------
    arr : np.ndarray
        Input data array.
    n : int, optional
        Desired number of levels (default: 10).

    Returns
    -------
    np.ndarray
        Computed contour levels.

    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Use 1st and 99th percentiles to avoid outliers
    q_lo, q_hi = np.quantile(arr, [0.01, 0.99])
    vmin, vmax = float(q_lo), float(q_hi)

    if vmin == vmax:
        eps = 1e-12
        return np.linspace(vmin - eps, vmax + eps, n)

    # Use MaxNLocator to get "nice" levels
    locator = mticker.MaxNLocator(nbins=n)

    levels = np.asarray(locator.tick_values(vmin, vmax), dtype=np.float64)
    levels = np.unique(levels).astype(np.float64, copy=False)

    if len(levels) < _MIN_LEVEL_COUNT or not np.all(np.diff(levels) > 0):
        levels = np.linspace(vmin, vmax, n, dtype=np.float64)

    return levels


# AXIS LABEL HELPERS


def apply_axis_labels(
    ax: Axes,
    col: int,
    Lx: float,
    Ly: float,
    *,
    is_last_row: bool,
) -> None:
    """
    Apply consistent axis labels and ticks to subplot axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to modify.
    col : int
        Column index of the subplot.
    Lx : float
        Length of the domain in x-direction.
    Ly : float
        Length of the domain in y-direction.
    is_last_row : bool
        Whether the subplot is in the last row.

    Returns
    -------
    None
        Modifies the Axes in place.

    """
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    yticks = [0.0, 0.25, 0.5, 0.75]
    ax.set_yticks(yticks)

    if col == 0:
        ax.set_ylabel("y [m]")
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if is_last_row:
        ax.set_xlabel("x [m]")
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


# FLOW OVERLAYS


def overlay_streamlines(ax: Axes, X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """
    Overlay streamlines on the given Axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to modify.
    X : np.ndarray
        X-coordinates meshgrid.
    Y : np.ndarray
        Y-coordinates meshgrid.
    u : np.ndarray
        Velocity component in x-direction.
    v : np.ndarray
        Velocity component in y-direction.

    Returns
    -------
    None
        Modifies the Axes in place.

    """
    ax.streamplot(
        X,
        Y,
        u,
        v,
        color=(0, 0, 0, 0.6),
        density=1.0,
        linewidth=0.6,
        arrowsize=0.6,
        minlength=0.1,
        integration_direction="both",
    )
