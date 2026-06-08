"""
===============================================================================
analysis_ui_viewers.py
===============================================================================
Build interactive viewers for case-level and aggregate analysis plots.

Responsibilities:
  - Render case-by-case Matplotlib figures inside widgets
  - Manage dataset and case navigation controls
  - Store current figures for notebook export

Design principles:
  - Viewer callbacks receive explicit datasets and render functions
  - Widget construction delegates to analysis_ui_components
  - Export context is updated only around rendered figures

Boundaries:
  - Notebook section composition belongs to analysis_ui_notebook
  - Domain-specific plot logic belongs to analysis plot modules
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.figure import Figure

from . import analysis_ui_components as components

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


# =============================================================================
# INTERNAL HELPERS (viewer-agnostic, no semantics)
# =============================================================================


def _render_figure(
    *,
    out: widgets.Output,
    plot_func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Render a figure inside an output widget.

    Parameters
    ----------
    out : widgets.Output
        Output widget to render into.
    plot_func : callable
        Plotting function that returns a Figure or other displayable object.
    args : tuple, optional
        Positional arguments for the plot function (default: ()).
    kwargs : dict, optional
        Keyword arguments for the plot function (default: None).


    """
    kwargs = kwargs or {}

    with out:
        out.clear_output(wait=True)

        result = plot_func(*args, **kwargs)

        # Accept (fig, ...) as well
        fig: Figure | None = None
        if isinstance(result, Figure):
            fig = result
        elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], Figure):
            fig = result[0]

        if fig is not None:
            # Update export target if available
            export_state = _EXPORT_CTX.get("export_state")
            if isinstance(export_state, dict):
                export_state["fig"] = fig

                pn = _EXPORT_CTX.get("plot_name")
                tt = _EXPORT_CTX.get("title")

                if isinstance(pn, str) and pn:
                    export_state["plot_name"] = pn
                if isinstance(tt, str) and tt:
                    export_state["title"] = tt

            display(fig)
            plt.close(fig)
            return

        # Non-figure results (rare): still display them
        if result is not None:
            display(result)


def _attach_widget_rerender(
    widgets_list: list[widgets.Widget],
    render_func: Callable[[], None],
) -> None:
    """
    Attach a re-render callback to multiple widgets.

    Any change of a widget's value triggers the provided render function.

    Supports:
        - Widgets with a `value` trait
        - VBox checkbox groups exposing a `.boxes` dict
    """
    for w in widgets_list:
        # ---------------------------------------------
        # Case 1: standard ValueWidget (Dropdown, Radio)
        # ---------------------------------------------
        if hasattr(w, "observe") and hasattr(w, "value"):
            w.observe(lambda _: render_func(), names="value")
            continue

        # ---------------------------------------------
        # Case 2: checkbox group (VBox with .boxes)
        # ---------------------------------------------
        if hasattr(w, "boxes"):
            for cb in w.boxes.values():  # type: ignore[attr-defined]
                cb.observe(lambda _: render_func(), names="value")


# =============================================================================
# EXPORT CONTEXT (set by analysis_ui_notebook before running a plot)
# =============================================================================
_EXPORT_CTX: dict[str, Any] = {}


def set_export_context(export_state: dict | None, *, plot_name: str | None = None, title: str | None = None) -> None:
    """
    Set the export context for the next plot rendering.

    Parameters
    ----------
    export_state : dict | None
        Export state dictionary to populate (or None to disable).
    plot_name : str | None, optional
        Plot name for export (default: None).
    title : str | None, optional
        Plot title for export (default: None).

    """
    _EXPORT_CTX.clear()
    _EXPORT_CTX.update(
        {
            "export_state": export_state,
            "plot_name": plot_name,
            "title": title,
        }
    )


# =============================================================================
# 1) CASE VIEWER (single-case visualisations)
# =============================================================================


def make_interactive_case_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    start_idx: int = 0,
    enable_dataset_dropdown: bool = True,
    extra_widgets: list[widgets.Widget] | None = None,
    n_cases_fn: Callable[[str, pd.DataFrame], int] | None = None,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Interactive viewer for case-indexed plots.

    Parameters
    ----------
    plot_func : callable
        Function of the form:
            plot_func(case_idx=N, df=df, dataset_name=name, **kwargs)
        Must return a matplotlib Figure.
    datasets : dict[str, DataFrame]
        Mapping: dataset_name -> dataset DataFrame.
    start_idx : int, optional
        Initial zero-based case index (default: 0).
    enable_dataset_dropdown : bool, optional
        Whether to show dataset dropdown (default: True).
    extra_widgets : list[widgets.Widget] | None, optional
        Additional widgets to include in the header.
        These widgets trigger re-rendering on value change.
    n_cases_fn : callable | None, optional
        Function of the form:
            n_cases_fn(dataset_name, df) -> int
        to determine the number of cases in a dataset.
        If None, defaults to len(df) (default: None).
    **plot_kwargs : Any
        Forwarded into the plot function.

    Returns
    -------
    widgets.VBox
        Complete interactive viewer.

    """
    dataset_names = list(datasets.keys())
    active_dataset = dataset_names[0]

    # ------------------------------------------------------------------
    # Dataset selector
    # ------------------------------------------------------------------
    dataset_dropdown = components.ui_dropdown_dataset(dataset_names) if enable_dataset_dropdown and len(dataset_names) > 1 else None

    # ------------------------------------------------------------------
    # Case index step control
    # ------------------------------------------------------------------
    df_active = datasets[active_dataset]

    n_cases_active = n_cases_fn(active_dataset, df_active) if n_cases_fn is not None else len(df_active)

    case_index, prev_btn, next_btn = components.ui_step_case_index(
        n_cases=n_cases_active,
        start_idx=start_idx,
    )

    # ------------------------------------------------------------------
    # Output container
    # ------------------------------------------------------------------
    out = components.ui_output_plot()
    extra_widgets = extra_widgets or []

    # ------------------------------------------------------------------
    # Render logic
    # ------------------------------------------------------------------
    def _render() -> None:
        if dataset_dropdown is not None:
            name: str = dataset_dropdown.value  # pyright: ignore[reportAssignmentType]
        else:
            name = active_dataset

        df = datasets[name]

        n_cases = n_cases_fn(name, df) if n_cases_fn is not None else len(df)

        case_idx = case_index.value - 1
        case_idx = max(0, min(n_cases - 1, case_idx))

        _render_figure(
            out=out,
            plot_func=plot_func,
            args=(case_idx,),
            kwargs={
                "df": df,
                "dataset_name": name,
                **plot_kwargs,
            },
        )

    def _step(delta: int) -> None:
        case_index.value = max(
            1,
            min(case_index.max, case_index.value + delta),
        )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    case_index.observe(lambda _: _render(), names="value")

    if dataset_dropdown is not None:

        def _on_dataset_change(change: dict) -> None:
            df_new = datasets[change["new"]]

            n_cases_new = n_cases_fn(change["new"], df_new) if n_cases_fn is not None else len(df_new)

            case_index.max = n_cases_new
            case_index.value = min(case_index.value, n_cases_new)
            _render()

        dataset_dropdown.observe(_on_dataset_change, names="value")

    _attach_widget_rerender(extra_widgets, _render)

    # Initial render
    _render()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    header_items: list[widgets.Widget] = [
        case_index,
        prev_btn,
        next_btn,
        *extra_widgets,
    ]

    if dataset_dropdown is not None:
        header_items.append(dataset_dropdown)

    header = widgets.HBox(header_items)

    return widgets.VBox([header, out])


# =============================================================================
# 2) CASECOUNT VIEWER (multi-case aggregations)
# =============================================================================


def make_casecount_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    start_cases: int = 100,
    step_size: int = 50,
    extra_widgets: list[widgets.Widget] | None = None,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Viewer for plots that aggregate over a variable number of cases.

    Parameters
    ----------
    plot_func : callable
        Function of the form:
            plot_func(datasets=datasets, max_cases=N, **kwargs)
        Must return a matplotlib Figure.
    datasets : dict[str, DataFrame]
        Mapping: dataset_name -> dataset DataFrame.
    start_cases : int, optional
        Initial number of cases to include (default: 50).
    step_size : int, optional
        Step size for increasing/decreasing case count (default: 50).
    extra_widgets : list[widgets.Widget] | None, optional
        Additional widgets to include in the header.
        These widgets trigger re-rendering on value change.
    **plot_kwargs : Any
        Forwarded into the plot function.

    Returns
    -------
    widgets.VBox
        Complete interactive viewer.

    """
    max_cases_global = min(len(df) for df in datasets.values())

    case_count, prev_btn, next_btn = components.ui_step_case_count(
        start_cases=min(start_cases, max_cases_global),
        min_cases=0,
        max_cases=max_cases_global,
        step_size=step_size,
    )

    out = components.ui_output_plot()
    extra_widgets = extra_widgets or []

    # ------------------------------------------------------------------
    # Render logic
    # ------------------------------------------------------------------
    def _render() -> None:
        _render_figure(
            out=out,
            plot_func=plot_func,
            kwargs={
                "datasets": datasets,
                "max_cases": int(case_count.value),
                **plot_kwargs,
            },
        )

    def _step(delta: int) -> None:
        new_val = case_count.value + delta * step_size
        case_count.value = max(1, min(max_cases_global, new_val))

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    case_count.observe(lambda _: _render(), names="value")

    _attach_widget_rerender(extra_widgets, _render)

    # Initial render
    _render()

    header = widgets.HBox(
        [
            case_count,
            prev_btn,
            next_btn,
            *extra_widgets,
        ],
        layout=widgets.Layout(
            align_items="center",
        ),
    )

    return widgets.VBox([header, out])
