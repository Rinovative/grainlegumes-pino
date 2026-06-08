"""
===============================================================================
analysis_ui_notebook.py
===============================================================================
Build notebook dropdown sections, lazy panels and figure exports.

Responsibilities:
  - Assemble plot functions into dropdown sections
  - Build collapsible tabbed notebook panels
  - Manage Matplotlib figure export state

Design principles:
  - Panels render lazily to keep notebooks responsive
  - Export state is passed explicitly between callbacks
  - Display helpers handle figures, widgets and rich objects uniformly

Boundaries:
  - Primitive widgets belong to analysis_ui_components
  - Case-level renderers belong to analysis_ui_viewers
===============================================================================
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
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

    Parameters
    ----------
    name: str
        The original name to sanitize.

    Returns
    -------
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

    Parameters
    ----------
    result: Any
        The object to display.

    Returns
    -------
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


def make_dropdown_section(plots: list, *, export_state: dict | None = None) -> Any:
    """
    Create an interactive dropdown section for multiple plots.

    Parameters
    ----------
    plots: list
        A list of tuples in the form (title, plot_function, plot_name).
        - title: str - Display name for the dropdown option.
        - plot_function: Callable - A function that generates the plot when called.
        - plot_name: str - A sanitized name used for export context.
    export_state: dict, optional
        A dictionary to store export information (e.g., current figure, plot name).
        This allows the export button in the panel to access the current plot for PDF export.

    Returns
    -------
        widgets.VBox: A widget container suitable for direct notebook display.

    """
    dropdown = widgets.Dropdown(
        options=[(title, i) for i, (title, _, _) in enumerate(plots)],
        style={"description_width": "initial"},
        layout=widgets.Layout(width="230px"),
    )
    output = widgets.Output()
    last_idx = {"idx": None}

    def on_plot_change(change: dict) -> None:
        """Render the selected plot and update export state."""
        idx = change["new"]
        if last_idx["idx"] == idx:
            return

        title, plot_func, plot_name = plots[idx]

        with output:
            output.clear_output(wait=True)
            plt.close("all")

            # Tell analysis_ui_viewers where to store figures rendered inside viewers/callbacks
            if export_state is not None:
                from . import analysis_ui_viewers as _viewers  # local import to avoid circular import  # noqa: PLC0415

                _viewers.set_export_context(export_state, plot_name=plot_name, title=title)

            result = plot_func()
            if isinstance(result, tuple):
                result = result[0]

            # update export target for direct Figure returns
            if export_state is not None:
                export_state["title"] = title
                export_state["plot_name"] = plot_name

                # WICHTIG: NICHT auf None ueberschreiben, wenn ein Viewer-Widget zurueckkommt.
                # In dem Fall setzt analysis_ui_viewers._render_figure die Figure bereits in export_state.
                if isinstance(result, Figure):
                    export_state["fig"] = result

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
    dfs: dict[str, pd.DataFrame] | list[pd.DataFrame],
) -> Callable:
    """
    Create a toggle shortcut that injects one or multiple datasets into interactive viewer functions.

    Viewer functions that support multi-dataset viewing must include a
    parameter named 'datasets'. In this case, toggle() will inject a
    dictionary:

        datasets = { dataset_name: DataFrame, ... }

    If dfs is provided as a list, dataset names are autogenerated as:
        "df0", "df1", ...

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame] or list[pd.DataFrame]
        One or several datasets to be passed to viewer functions.

    Returns
    -------
    Callable
        A function that generates toggle entries for dropdown sections.

    """
    counter = {"i": 0}

    # normalize dfs to dict[str, DataFrame]
    dataset_map = dfs if isinstance(dfs, dict) else {f"df{i}": df for i, df in enumerate(dfs)}

    def toggle(title: str, func: Callable[..., Any], plot_name: str | None = None, **kwargs: Any) -> tuple[str, Callable[[], Any], str]:
        """Return one dropdown entry with dataset injection applied."""
        # auto-generate plot name
        if plot_name is None:
            plot_name = f"plot_{counter['i']:03d}"
            counter["i"] += 1
        else:
            plot_name = _sanitize_name(plot_name)

        # inject datasets if viewer supports it
        fn_args = func.__code__.co_varnames
        if "datasets" in fn_args:
            kwargs.setdefault("datasets", dataset_map)

        # wrap function call for dropdown section
        return (title, lambda: func(**kwargs), plot_name)

    return toggle


def make_lazy_panel_with_tabs(
    sections: Sequence[widgets.Widget],
    tab_titles: Sequence[str] | None = None,
    open_btn_text: str = "Open section",
    close_btn_text: str = "Close",
    *,
    export_state: dict | None = None,
    export_dir: str = "exports",
    export_btn_text: str = "Export PDF",
) -> widgets.Output:
    """
    Create a collapsible widget panel containing multiple tabs.

    The panel can be opened and closed using buttons, and each tab may contain
    arbitrary widgets (e.g., dropdown sections, plots, layouts).

    Parameters
    ----------
    sections: Sequence[widgets.Widget]
        A sequence of widget containers to be placed in individual tabs.
    tab_titles: Sequence[str], optional
        Titles for each tab. If not provided, tabs will be titled "Tab 1", "Tab 2", etc.
    open_btn_text: str, optional
        Text for the button that opens the panel. Default is "Open section".
    close_btn_text: str, optional
        Text for the button that closes the panel. Default is "Close".
    export_state: dict, optional
        A dictionary to store export information (e.g., current figure, plot name).
        This allows the export button in the panel to access the current plot for PDF export.
    export_dir: str, optional
        Directory where exported PDF files will be saved. Default is "exports".
    export_btn_text: str, optional
        Text for the button that triggers PDF export. Default is "Export PDF".

    Returns
    -------
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

    status_out = widgets.Output()

    export_btn = widgets.Button(
        description=export_btn_text,
        button_style="success",
        layout=widgets.Layout(width="145px"),
    )

    def do_export(_: None = None) -> None:
        """Export the current Matplotlib figure to PDF."""
        with status_out:
            status_out.clear_output(wait=True)

            if export_state is None:
                print("[Export] export_state ist None.")
                return

            fig = export_state.get("fig", None)
            if fig is None:
                print("[Export] Kein Matplotlib-Figure Objekt verfuegbar (zuerst einen Plot anzeigen).")
                return

            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = export_state.get("plot_name") or "plot"
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{stem}_{ts}.pdf"

            fig.savefig(out_path, bbox_inches="tight")
            print(f"[Export] Gespeichert: {out_path}")

    export_btn.on_click(do_export)

    header = widgets.HBox([close_btn, export_btn])
    panel = widgets.VBox([header, status_out, tabs])

    def show_panel(_: None = None) -> None:
        """Display the expanded panel."""
        with main_out:
            clear_output()
            display(panel)

    def show_open(_: None = None) -> None:
        """Display the collapsed open button."""
        with main_out:
            clear_output()
            display(open_btn)

    open_btn.on_click(show_panel)
    close_btn.on_click(show_open)
    show_open()
    return main_out
