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

This module does NOT:
  - Define primitive widgets or domain-specific scientific controls
  - Load artifact cases or implement case-level plot mathematics
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
    Convert a display label to the module's minimal export filename stem.

    Text is lowercased; spaces become underscores; Unicode dashes become ASCII
    hyphens; and forward slashes become underscores. No broader path-policy
    validation is performed here.
    """
    return name.lower().replace(" ", "_").replace("–", "-").replace("—", "-").replace("/", "_")  # noqa: RUF001


def _show_anything(result: Any) -> None:
    """
    Display one supported result in the active notebook output context.

    Matplotlib figures are displayed and closed, objects with ``show`` invoke it,
    strings are printed, and other non-``None`` values use IPython display.
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
    Build one lazy dropdown whose entries render notebook views on selection.

    Parameters
    ----------
    plots : list
        Ordered ``(title, zero-argument callable, export_name)`` entries. Index
        ``-1`` is reserved for the initial non-rendering prompt.
    export_state : dict | None, optional
        Shared mutable state receiving the current title/name and direct or
        viewer-rendered Matplotlib figure for later PDF export.

    Returns
    -------
    ipywidgets.VBox
        Dropdown and output area. A plot callable runs only when its entry is
        selected for the first time after a different selection.

    Notes
    -----
    Selecting the prompt clears output. Before rendering, the prior export figure
    is cleared; non-figure widgets may populate it later through viewer callbacks.

    """
    dropdown = widgets.Dropdown(
        options=[("Choose a view…", -1), *((title, i) for i, (title, _, _) in enumerate(plots))],
        value=-1,
        description="View:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="360px"),
    )
    output = widgets.Output()
    last_idx: dict[str, int | None] = {"idx": None}

    def on_plot_change(change: dict) -> None:
        """
        Render a newly selected entry and synchronize shared export state.

        Repeated selection is ignored, the prompt clears output, and direct
        figures replace the export target while viewer widgets may update it
        later through the shared viewer context.
        """
        idx = change["new"]
        if last_idx["idx"] == idx:
            return
        if idx == -1:
            with output:
                output.clear_output(wait=True)
            last_idx["idx"] = idx
            return

        title, plot_func, plot_name = plots[idx]

        with output:
            output.clear_output(wait=True)
            plt.close("all")

            # Tell analysis_ui_viewers where to store figures rendered inside viewers/callbacks
            if export_state is not None:
                from . import analysis_ui_viewers as _viewers  # local import to avoid circular import  # noqa: PLC0415

                export_state["fig"] = None
                _viewers.set_export_context(export_state, plot_name=plot_name, title=title)

            result = plot_func()
            if isinstance(result, tuple):
                result = result[0]

            # update export target for direct Figure returns
            if export_state is not None:
                export_state["title"] = title
                export_state["plot_name"] = plot_name

                # Viewer callbacks may already have stored their rendered Figure
                # through the shared export context; preserve it for non-Figure widgets.
                if isinstance(result, Figure):
                    export_state["fig"] = result

            if isinstance(result, Figure):
                display(result)
                plt.close(result)
            else:
                _show_anything(result)

        last_idx["idx"] = idx

    dropdown.observe(on_plot_change, names="value")

    return widgets.VBox([dropdown, output])


def make_toggle_shortcut(
    dfs: dict[str, pd.DataFrame] | list[pd.DataFrame],
) -> Callable:
    """
    Create a dropdown-entry factory that injects labelled datasets into viewers.

    Parameters
    ----------
    dfs : dict[str, pandas.DataFrame] | list[pandas.DataFrame]
        Dataset mapping, or an ordered list assigned stable ``df0``, ``df1``, ...
        labels within this factory.

    Returns
    -------
    Callable
        Factory returning ``(title, zero-argument callable, export_name)``. A
        ``datasets`` keyword is injected only when the target callable declares
        that parameter; caller-supplied keyword arguments otherwise pass through.

    Notes
    -----
    Missing export names are generated monotonically. Supplied names use the
    module's minimal filename-stem sanitizer.

    """
    counter = {"i": 0}

    # normalize dfs to dict[str, DataFrame]
    dataset_map = dfs if isinstance(dfs, dict) else {f"df{i}": df for i, df in enumerate(dfs)}

    def toggle(title: str, func: Callable[..., Any], plot_name: str | None = None, **kwargs: Any) -> tuple[str, Callable[[], Any], str]:
        """
        Bind one target callable and its export identity into a lazy entry.

        The normalized dataset mapping is injected only for callables declaring a
        ``datasets`` local/parameter name; invocation remains deferred.
        """
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
    Build a collapsible tab panel with optional current-figure PDF export.

    Parameters
    ----------
    sections : Sequence[ipywidgets.Widget]
        Already constructed tab contents; scientific views inside them may remain
        lazy according to their own dropdown/viewer contract.
    tab_titles : Sequence[str] | None, optional
        Tab labels, or generated ``Tab N`` names when omitted.
    open_btn_text, close_btn_text : str, optional
        Labels for panel visibility controls.
    export_state : dict | None, optional
        Shared mutable mapping expected to contain ``fig`` and optional
        ``plot_name`` for the current Matplotlib export target.
    export_dir : str, optional
        Directory created on export; an empty string resolves to the process
        working directory.
    export_btn_text : str, optional
        PDF-export button label.

    Returns
    -------
    ipywidgets.Output
        Output initially displaying only the open button.

    Notes
    -----
    Opening/closing replaces notebook output but preserves tab/widget state.
    Export is user-triggered, creates the directory, and writes a UTC-timestamped
    PDF from the current figure; missing state is reported without writing.

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
        """
        Write the current export figure to a UTC-timestamped PDF on button click.

        Directory creation and file publication occur only when shared state holds
        a figure; otherwise the callback reports status without filesystem writes.
        """
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
