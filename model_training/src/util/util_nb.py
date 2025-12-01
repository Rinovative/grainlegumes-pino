"""
Jupyter notebook utility functions for interactive plotting and widgets.

This module provides:
- Interactive dropdown sections for plot selection
- Collapsible panels with tabbed content
- A helper for generating toggle entries for plot lists
- Utility helpers for robust display of arbitrary Jupyter outputs

The functions in this module are designed to work reliably with large
Matplotlib figures and debug-heavy plot functions by avoiding the
`with output:` context manager, which can swallow stdout and block
expensive rendering operations.
"""

from collections.abc import Callable, Sequence
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from matplotlib.figure import Figure

# ============================================================================
# Helper utilities
# ============================================================================


def _sanitize_name(name: str) -> str:
    """
    Convert a plot name into a filename-safe format.

    Args:
        name (str): Original plot name or title.

    Returns:
        str: Normalised filename-safe string.

    """
    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace("–", "-").replace("—", "-")  # noqa: RUF001
    return name.replace("/", "_")


def _show_anything(value: Any) -> None:
    """
    Display arbitrary content in a Jupyter notebook without blocking.

    Supported:
        - Matplotlib figures
        - Objects with a `.show()` method (Plotly etc.)
        - Strings
        - Generic displayable objects

    Args:
        value: Object to display.

    """
    if isinstance(value, Figure):
        display(value)
        plt.close(value)
        return

    if hasattr(value, "show") and callable(value.show):
        value.show()
        return

    if isinstance(value, str):
        print(value)
        return

    if value is not None:
        display(value)


# ============================================================================
# Dropdown section (non-blocking, debug-safe)
# ============================================================================


def make_dropdown_section(plots: list, description: str = "Plot:") -> widgets.VBox:
    """
    Create a dropdown menu for selecting and displaying plots.

    Each element in `plots` must be:
        (title: str, function: Callable, plot_name: str)

    The dropdown executes the associated plot function and displays the
    resulting figure or widget inside an Output() widget. Debug output is
    not swallowed because the plot is *not* executed inside a `with output:` block.

    Args:
        plots (list):
            List of (title, function, plot_name) tuples.
        description (str, optional):
            Label text next to the dropdown. Default: "Plot:".

    Returns:
        widgets.VBox:
            Combined dropdown + output area suitable for use in tabs.

    """
    dropdown = widgets.Dropdown(
        options=[(title, i) for i, (title, _, _) in enumerate(plots)],
        description=description,
        style={"description_width": "initial"},
    )

    output = widgets.Output()
    last_idx = {"idx": None}

    def on_change(change: dict) -> None:
        idx = change["new"]

        if last_idx["idx"] == idx:
            return

        plot_func = plots[idx][1]
        output.clear_output(wait=False)

        result = plot_func()
        with output:
            _show_anything(result)

        last_idx["idx"] = idx

    dropdown.observe(on_change, names="value")
    on_change({"new": 0})  # initial plot

    return widgets.VBox([dropdown, output])


# ============================================================================
# Toggle helper
# ============================================================================


def make_toggle_shortcut(
    df: pd.DataFrame,
    dataset_name: str = "",
    df_alt: pd.DataFrame | None = None,
    dataset_name_alt: str | None = None,
) -> Callable:
    """
    Create a helper to build (title, callable, plot_name) tuples for dropdowns.

    The returned `toggle()` automatically injects `df`, `df_alt`,
    `dataset_name`, and `dataset_name_alt` into any plot function
    that declares parameters with those names.

    Args:
        df (pd.DataFrame):
            Primary DataFrame.
        dataset_name (str):
            Name of the primary dataset.
        df_alt (pd.DataFrame or None):
            Optional secondary DataFrame.
        dataset_name_alt (str or None):
            Optional secondary dataset name.

    Returns:
        Callable:
            toggle(title, func, plot_name=None, **kwargs) → (title, callable, name)

    """
    counter = {"i": 0}

    def toggle(
        title: str,
        func: Callable[..., Any],
        plot_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, Callable[[], Any], str]:
        name = plot_name or f"plot_{counter['i']:03d}"
        name = _sanitize_name(name)
        counter["i"] += 1

        argnames = func.__code__.co_varnames

        if "df" in argnames:
            kwargs.setdefault("df", df)
        if "df_alt" in argnames:
            kwargs.setdefault("df_alt", df_alt)
        if "dataset_name" in argnames:
            kwargs.setdefault("dataset_name", dataset_name)
        if "dataset_name_alt" in argnames:
            kwargs.setdefault("dataset_name_alt", dataset_name_alt)

        return (title, lambda: func(**kwargs), name)

    return toggle


# ============================================================================
# Collapsible tabbed panel
# ============================================================================


def make_lazy_panel_with_tabs(
    sections: Sequence[widgets.Widget],
    tab_titles: Sequence[str] | None = None,
    open_btn_text: str = "Open section",
    close_btn_text: str = "Close",
) -> widgets.Output:
    """
    Create a collapsible UI panel containing tabs.

    The panel is initially closed and replaced by an "Open" button.
    Clicking "Open" reveals the tabs; clicking "Close" collapses the panel.

    Args:
        sections (Sequence[widgets.Widget]):
            Widgets to be placed into separate tabs.
        tab_titles (Sequence[str], optional):
            Titles for each tab. Defaults to numeric labels.
        open_btn_text (str, optional):
            Label for the open button.
        close_btn_text (str, optional):
            Label for the close button.

    Returns:
        widgets.Output:
            A widget handle for display inside Jupyter.

    """
    main_out = widgets.Output()

    open_btn = widgets.Button(description=open_btn_text, button_style="primary")
    close_btn = widgets.Button(description=close_btn_text, button_style="danger")

    tabs = widgets.Tab(children=list(sections))
    if tab_titles:
        for i, name in enumerate(tab_titles):
            tabs.set_title(i, name)
    else:
        for i in range(len(sections)):
            tabs.set_title(i, f"Tab {i + 1}")

    panel = widgets.VBox([close_btn, tabs])

    def show_panel(_: None = None) -> None:
        with main_out:
            main_out.clear_output()
            display(panel)

    def show_open(_: None = None) -> None:
        with main_out:
            main_out.clear_output()
            display(open_btn)

    open_btn.on_click(show_panel)
    close_btn.on_click(show_open)

    show_open()
    return main_out
