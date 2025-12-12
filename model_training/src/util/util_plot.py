"""
Unified interactive plot navigators.

This module provides exactly two viewer types:

1) make_interactive_case_viewer(...)
    - Shows ONE case at a time
    - Dataset selection via dropdown
    - Case navigation via step control (index + arrows)
    - Arbitrary extra widgets supported
    - Used for all case-dependent visualisations

2) make_casecount_viewer(...)
    - Aggregates statistics over N cases
    - Case-count navigation via step control (slider + arrows)
    - Arbitrary extra widgets supported
    - Used for global error metrics, GT-vs-Pred cached plots, etc.

Design principles
-----------------
- Viewers contain logic and semantics
- All UI elements are constructed via util_plot_components
- No widget construction is duplicated here
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from src.util.util_plot_components import (
    ui_dropdown_dataset,
    ui_output_plot,
    ui_step_case_count,
    ui_step_case_index,
)

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
    Render a matplotlib figure into a widgets.Output container.

    Centralised rendering logic:
        - clear output
        - call plot function
        - display figure
        - close figure

    Parameters
    ----------
    out : widgets.Output
        Output widget to render into.
    plot_func : callable
        Plotting function returning a matplotlib Figure.
    args : tuple, optional
        Positional arguments forwarded to plot_func.
    kwargs : dict, optional
        Keyword arguments forwarded to plot_func.

    """
    kwargs = kwargs or {}

    with out:
        out.clear_output(wait=True)
        fig = plot_func(*args, **kwargs)
        display(fig)
        plt.close(fig)


def _attach_widget_rerender(
    widgets_list: list[widgets.Widget],
    render_func: Callable[[], None],
) -> None:
    """
    Attach a re-render callback to multiple widgets.

    Any change of a widget's value triggers the provided render function.

    Parameters
    ----------
    widgets_list : list of widgets.Widget
        Widgets whose value changes should trigger re-rendering.
    render_func : callable
        Zero-argument render function.

    """
    for w in widgets_list:
        w.observe(lambda _: render_func(), names="value")


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
    dataset_dropdown = ui_dropdown_dataset(dataset_names) if enable_dataset_dropdown and len(dataset_names) > 1 else None

    # ------------------------------------------------------------------
    # Case index step control
    # ------------------------------------------------------------------
    df_active = datasets[active_dataset]
    case_index, prev_btn, next_btn = ui_step_case_index(
        n_cases=len(df_active),
        start_idx=start_idx,
    )

    # ------------------------------------------------------------------
    # Output container
    # ------------------------------------------------------------------
    out = ui_output_plot()
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

        case_idx = case_index.value - 1
        case_idx = max(0, min(len(df) - 1, case_idx))

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
            case_index.max = len(df_new)
            case_index.value = min(case_index.value, len(df_new))
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
    start_cases: int = 50,
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

    case_count, prev_btn, next_btn = ui_step_case_count(
        start_cases=min(start_cases, max_cases_global),
        min_cases=0,
        max_cases=max_cases_global,
        step_size=step_size,
    )

    out = ui_output_plot()
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
        ]
    )

    return widgets.VBox([header, out])
