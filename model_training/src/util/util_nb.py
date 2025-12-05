"""
Jupyter notebook utility functions for interactive plotting and widgets.

This module provides:
- Functions for creating interactive plot dropdowns
- Functions for creating collapsible widget panels with tabs
- Helper functions for displaying and managing plots
"""

from collections.abc import Callable, Sequence
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.figure import Figure


def _sanitize_name(name: str) -> str:
    """
    Normalize a plot name into a filename-safe format.

    Converts to lowercase, replaces spaces and various dashes,
    and removes invalid path characters.

    Args:
        name (str): Original plot name or title.

    Returns:
        str: Sanitized name suitable for filenames.

    """
    return name.lower().replace(" ", "_").replace("–", "-").replace("—", "-").replace("/", "_")  # noqa: RUF001


def _show_anything(result: Any) -> None:
    """
    Display an arbitrary result object in a Jupyter notebook.

    Supports:
    - Matplotlib figures
    - Plotly or other objects with .show() method
    - Strings
    - Generic displayable objects (e.g. DataFrames, widgets)

    Args:
        result: Object to display.

    Returns:
        None

    """
    if isinstance(result, Figure):
        display(result)
        plt.close(result)
    elif hasattr(result, "show") and callable(result.show):
        result.show()
    elif isinstance(result, str):
        print(result)
    elif result is not None:
        display(result)


def make_dropdown_section(plots: list) -> Any:
    """
    Create an interactive dropdown section for selecting and displaying plots.

    Each entry in 'plots' must be a tuple (title, function, plot_name).

    Args:
        plots (list): List of tuples (title, function, plot_name).

    Returns:
        widgets.VBox: Interactive section with dropdown and output area.

    """
    dropdown = widgets.Dropdown(
        options=[(title, i) for i, (title, _, _) in enumerate(plots)],
        style={"description_width": "initial"},
        layout=widgets.Layout(width="227px"),
    )
    output = widgets.Output()
    last_idx = {"idx": None}

    def on_plot_change(change: dict) -> None:
        idx = change["new"]
        if last_idx["idx"] == idx:
            return

        plot_func = plots[idx][1]

        with output:
            output.clear_output(wait=True)
            plt.close("all")

            result = plot_func()
            if isinstance(result, tuple):
                result = result[0]

            if isinstance(result, Figure):
                display(result)
                plt.close(result)
            else:
                _show_anything(result)

        last_idx["idx"] = idx

    dropdown.observe(on_plot_change, names="value")
    on_plot_change({"type": "change", "name": "value", "new": 0})

    return widgets.VBox([dropdown, output])


def make_toggle_shortcut(
    df: pd.DataFrame,
    dataset_name: str = "",
    df_alt: pd.DataFrame | None = None,
    dataset_name_alt: str | None = None,
) -> Callable:
    """
    Create a toggle shortcut function for injecting common parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        Primary dataset for plotting functions.
    dataset_name : str, optional
        Name of the primary dataset for titles. Default is "".
    df_alt : pandas.DataFrame, optional
        Alternative dataset for plotting functions. Default is None.
    dataset_name_alt : str, optional
        Name of the alternative dataset for titles. Default is None.

    Returns
    -------
    Callable
        A function that generates toggle entries for interactive plots.

    """
    counter = {"i": 0}

    def toggle(title: str, func: Callable[..., Any], plot_name: str | None = None, **kwargs: Any) -> tuple[str, Callable[[], Any], str]:
        # auto plot name
        if plot_name is None:
            plot_name = f"plot_{counter['i']:03d}"
            counter["i"] += 1
        else:
            plot_name = _sanitize_name(plot_name)

        varnames = func.__code__.co_varnames

        # inject primary df
        if "df" in varnames:
            kwargs.setdefault("df", df)

        # inject primary dataset name
        if dataset_name is not None and "dataset_name" in varnames:
            kwargs.setdefault("dataset_name", dataset_name)

        # inject alternative df
        if df_alt is not None and "df_alt" in varnames:
            kwargs.setdefault("df_alt", df_alt)

        # inject alternative dataset name
        if dataset_name_alt is not None and "dataset_name_alt" in varnames:
            kwargs.setdefault("dataset_name_alt", dataset_name_alt)

        return (title, lambda: func(**kwargs), plot_name)

    return toggle


def make_lazy_panel_with_tabs(
    sections: Sequence[widgets.Widget],
    tab_titles: Sequence[str] | None = None,
    open_btn_text: str = "Open section",
    close_btn_text: str = "Close",
) -> widgets.Output:
    """
    Create a collapsible widget panel containing multiple tabs.

    The panel can be opened and closed using buttons, and each tab may contain
    arbitrary widgets (e.g., dropdown sections, plots, layouts).

    Args:
        sections (list): List of widget objects to be used as tabs.
        tab_titles (list, optional): Titles for the tabs. Defaults to numbered tabs.
        open_btn_text (str, optional): Label for the open button. Defaults to "Open section".
        close_btn_text (str, optional): Label for the close button. Defaults to "Close".

    Returns:
        widgets.Output: A widget container suitable for direct notebook display.

    """
    main_out = widgets.Output()
    open_btn = widgets.Button(description=open_btn_text, button_style="primary", layout=widgets.Layout(width="auto"))
    close_btn = widgets.Button(description=close_btn_text, button_style="danger", layout=widgets.Layout(width="145px"))

    tabs = widgets.Tab(children=sections)
    if tab_titles is not None:
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
    else:
        for i in range(len(sections)):
            tabs.set_title(i, f"Tab {i + 1}")

    panel = widgets.VBox([close_btn, tabs])

    def show_panel(_: None = None) -> None:
        with main_out:
            clear_output()
            display(panel)

    def show_open(_: None = None) -> None:
        with main_out:
            clear_output()
            display(open_btn)

    open_btn.on_click(show_panel)
    close_btn.on_click(show_open)
    show_open()
    return main_out
