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
from inspect import signature
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

    Text is lowercased. Spaces become underscores. Unicode dashes become ASCII
    hyphens, and forward slashes become underscores. No broader path-policy
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


def _invoke_dropdown_entry(
    plot_func: Callable[..., Any],
    *,
    export_state: dict[str, Any] | None,
    export_plot_name: str,
    export_title: str,
) -> Any:
    """Invoke one entry with explicit export context when it declares support."""
    try:
        parameters = signature(plot_func).parameters
    except (TypeError, ValueError):
        return plot_func()
    context = {
        "export_state": export_state,
        "export_plot_name": export_plot_name,
        "export_title": export_title,
    }
    return plot_func(**{name: value for name, value in context.items() if name in parameters})


def make_dropdown_section(
    plots: list,
    *,
    export_state: dict | None = None,
    select_first: bool = False,
) -> Any:
    """
    Build one lazy dropdown whose entries render notebook views on selection.

    Parameters
    ----------
    plots : list
        Ordered ``(title, lazy callable, export_name)`` entries.
    export_state : dict | None, optional
        Shared mutable state receiving the current title/name and direct or
        viewer-rendered Matplotlib figure for later PDF export.
    select_first : bool, optional
        Omit the prompt and mark the first entry for activation when its tab opens.

    Returns
    -------
    ipywidgets.VBox
        Dropdown and output area. A plot callable runs only when its entry is
        selected for the first time after a different selection.

    Notes
    -----
    Selecting the prompt clears output. A first-entry section remains lazy until
    its tab opens. Before rendering, the prior export figure is cleared. Non-figure
    widgets may populate it later through viewer callbacks.

    """
    plot_options = [(title, i) for i, (title, _, _) in enumerate(plots)]
    dropdown = widgets.Dropdown(
        options=plot_options if select_first else [("Choose a view…", -1), *plot_options],
        value=None if select_first else -1,
        description="" if select_first else "View:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="230px" if select_first else "360px"),
    )
    output = widgets.Output()
    last_idx: dict[str, int | None] = {"idx": None}

    def on_plot_change(change: dict) -> None:
        """
        Render a newly selected entry and synchronize panel-local export state.

        Repeated selection is ignored, the prompt clears output, and direct
        figures replace the export target while viewer widgets may update it
        later through the explicitly supplied viewer state.
        """
        idx = change["new"]
        if last_idx["idx"] == idx:
            return
        if idx == -1:
            if export_state is not None:
                export_state["fig"] = None
                export_state["plot_name"] = None
                export_state["title"] = None
            with output:
                output.clear_output(wait=True)
            last_idx["idx"] = idx
            return

        title, plot_func, plot_name = plots[idx]

        with output:
            output.clear_output(wait=True)

            if export_state is not None:
                previous = export_state.get("fig")
                if isinstance(previous, Figure):
                    plt.close(previous)
                export_state["fig"] = None
                export_state["plot_name"] = plot_name
                export_state["title"] = title

            result = _invoke_dropdown_entry(
                plot_func,
                export_state=export_state,
                export_plot_name=plot_name,
                export_title=title,
            )
            if isinstance(result, tuple):
                result = result[0]

            # update export target for direct Figure returns
            if export_state is not None:
                export_state["title"] = title
                export_state["plot_name"] = plot_name

                # Viewer callbacks may already have stored their rendered Figure
                # in this panel's explicit state. Preserve it for non-Figure widgets.
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
        Factory returning ``(title, lazy callable, export_name)``. The callable
        accepts panel-local export context from :func:`make_dropdown_section`. A
        ``datasets`` keyword is injected only when the target declares it. Other
        caller-supplied keyword arguments pass through unchanged.

    Notes
    -----
    Missing export names are generated monotonically. Supplied names use the
    module's minimal filename-stem sanitizer.

    """
    counter = {"i": 0}

    # normalize dfs to dict[str, DataFrame]
    dataset_map = dfs if isinstance(dfs, dict) else {f"df{i}": df for i, df in enumerate(dfs)}

    def toggle(title: str, func: Callable[..., Any], plot_name: str | None = None, **kwargs: Any) -> tuple[str, Callable[..., Any], str]:
        """Bind one target callable and panel-local export identity lazily."""
        if plot_name is None:
            resolved_plot_name = f"plot_{counter['i']:03d}"
            counter["i"] += 1
        else:
            resolved_plot_name = _sanitize_name(plot_name)

        parameters = signature(func).parameters
        bound_kwargs = dict(kwargs)
        if "datasets" in parameters:
            bound_kwargs.setdefault("datasets", dataset_map)

        def invoke(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> Any:
            """Invoke the target with context only when its API declares it."""
            call_kwargs = dict(bound_kwargs)
            context = {
                "export_state": export_state,
                "export_plot_name": export_plot_name,
                "export_title": export_title,
            }
            call_kwargs.update({name: value for name, value in context.items() if name in parameters})
            return func(**call_kwargs)

        return title, invoke, resolved_plot_name

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
        Already constructed tab contents. Scientific views inside them may remain
        lazy according to their own dropdown/viewer contract.
    tab_titles : Sequence[str] | None, optional
        Tab labels, or generated ``Tab N`` names when omitted.
    open_btn_text, close_btn_text : str, optional
        Labels for panel visibility controls.
    export_state : dict | None, optional
        Shared mutable mapping expected to contain ``fig`` and optional
        ``plot_name`` for the current Matplotlib export target.
    export_dir : str, optional
        Directory created on export. An empty string resolves to the process
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
    PDF from the current figure. Missing state is reported without writing.

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
        a figure. Otherwise the callback reports status without filesystem writes.
        """
        with status_out:
            status_out.clear_output(wait=True)

            if export_state is None:
                print("[Export] Export is unavailable for this panel.")
                return

            fig = export_state.get("fig", None)
            if fig is None:
                print("[Export] No figure is selected. Render a plot first.")
                return

            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = export_state.get("plot_name") or "plot"
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{stem}_{ts}.pdf"

            fig.savefig(out_path, bbox_inches="tight")
            print(f"[Export] Saved: {out_path}")

    export_btn.on_click(do_export)

    header = widgets.HBox([close_btn, export_btn])
    panel = widgets.VBox([header, status_out, tabs])

    initialized_tabs: set[int] = set()
    tab_export_states = [{"fig": None, "plot_name": None, "title": None} for _ in sections]

    def _snapshot_tab(index: int | None) -> None:
        """Preserve the current view's export target before it is hidden."""
        if export_state is None or index is None or not 0 <= index < len(tab_export_states):
            return
        tab_export_states[index] = {
            "fig": export_state.get("fig"),
            "plot_name": export_state.get("plot_name"),
            "title": export_state.get("title"),
        }

    def activate_selected_view(_: object = None) -> None:
        """Render a tab's first view once, or restore its preserved export target."""
        selected_index = tabs.selected_index
        if selected_index is None:
            return
        section_children = getattr(sections[selected_index], "children", ())
        if not section_children or not isinstance(section_children[0], widgets.Dropdown):
            return
        dropdown = section_children[0]
        if selected_index not in initialized_tabs:
            if dropdown.index is None and dropdown.options:
                dropdown.index = 0
            initialized_tabs.add(selected_index)
            _snapshot_tab(selected_index)
            return
        if export_state is not None:
            export_state.update(tab_export_states[selected_index])

    def show_panel(_: None = None) -> None:
        """Display the expanded panel with the active tab already rendered."""
        _snapshot_tab(tabs.selected_index)
        activate_selected_view()
        with main_out:
            clear_output()
            display(panel)

    def show_open(_: None = None) -> None:
        """Display the collapsed open button without discarding tab state."""
        _snapshot_tab(tabs.selected_index)
        with main_out:
            clear_output()
            display(open_btn)

    def on_tab_change(change: dict[str, object]) -> None:
        """Snapshot the hidden tab and immediately initialize or restore the new tab."""
        old_index = change.get("old")
        _snapshot_tab(old_index if isinstance(old_index, int) else None)
        activate_selected_view()

    tabs.observe(on_tab_change, names="selected_index")
    open_btn.on_click(show_panel)
    close_btn.on_click(show_open)
    show_open()
    return main_out
