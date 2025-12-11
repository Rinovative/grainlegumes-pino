"""
Unified interactive plot navigators.

This file provides exactly two viewer types:

1) make_interactive_case_viewer(...)
    - Shows ONE case at a time
    - Optional dataset dropdown
    - Optional prev/next navigation
    - Optional extra widgets (e.g. error_mode)
    - Used for all case-dependent visualisations

2) make_casecount_viewer(...)
    - Aggregates statistics over N cases
    - Controls N through a slider
    - Used for global error metrics, GT-vs-Pred cached plots, etc.

A third viewer remains for global (non-case) selection:
    make_interactive_global_plot_dropdown(...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from src.util.util_plot_components import (
    ui_case_index,
    ui_dataset_dropdown,
    ui_output,
    ui_prev_next,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


def make_interactive_case_viewer(  # noqa: C901, PLR0912, PLR0915
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    start_idx: int = 0,
    enable_case_index: bool = True,
    enable_prev_next: bool = True,
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
    enable_case_index : bool, optional
        Whether to show case index selector (default: True).
    enable_prev_next : bool, optional
        Whether to show prev/next buttons (default: True).
    enable_dataset_dropdown : bool, optional
        Whether to show dataset dropdown (default: True).
    extra_widgets : list[widgets.Widget] | None, optional
        Additional widgets to include in the header (default: None).
        These widgets will trigger re-rendering when their value changes.
    **plot_kwargs : any
        Forwarded into the plot function.

    Returns
    -------
    widgets.VBox

    """
    names = list(datasets.keys())

    if enable_dataset_dropdown and len(names) > 1:
        dropdown = ui_dataset_dropdown(names)
        active_name = dropdown.value or names[0]
    else:
        dropdown = None
        active_name = names[0]

    if enable_case_index:
        df_active = datasets[active_name]
        idx = ui_case_index(len(df_active), start_idx)
    else:
        idx = None

    if enable_prev_next and enable_case_index:
        prev_btn, next_btn = ui_prev_next()
    else:
        prev_btn, next_btn = None, None

    out = ui_output()
    extra_widgets = extra_widgets or []

    def _render() -> None:
        name = active_name if dropdown is None else (dropdown.value or names[0])
        df = datasets[name]

        if idx is not None:
            case_idx = idx.value - 1
            case_idx = max(0, min(len(df) - 1, case_idx))
        else:
            case_idx = 0

        with out:
            out.clear_output(wait=True)
            fig = plot_func(case_idx, df=df, dataset_name=name, **plot_kwargs)
            display(fig)
            plt.close(fig)

    def _step(delta: int) -> None:
        if idx is None:
            return
        name = active_name if dropdown is None else (dropdown.value or names[0])
        df = datasets[name]
        idx.value = max(1, min(len(df), idx.value + delta))

    if prev_btn is not None:
        prev_btn.on_click(lambda _: _step(-1))
    if next_btn is not None:
        next_btn.on_click(lambda _: _step(1))

    if idx is not None:
        idx.observe(lambda _: _render(), names="value")

    if dropdown is not None:

        def _on_dataset_change(change: dict) -> None:
            df_new = datasets[change["new"]]
            if idx is not None:
                idx.max = len(df_new)
                idx.value = min(idx.value, len(df_new))
            _render()

        dropdown.observe(_on_dataset_change, names="value")

    for widget in extra_widgets:
        widget.observe(lambda _: _render(), names="value")

    _render()

    header_items: list[widgets.Widget] = []

    if idx is not None:
        header_items.append(idx)
    if prev_btn is not None:
        header_items.append(prev_btn)
    if next_btn is not None:
        header_items.append(next_btn)

    header_items.extend(extra_widgets)

    if dropdown is not None:
        header_items.append(dropdown)

    header = widgets.HBox(header_items)

    return widgets.VBox([header, out])


# ============================================================================
# 2) CASECOUNT VIEWER (unchanged — this is a separate mode)
# ============================================================================


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
            plot_func(datasets={name: df}, max_cases=N, **kwargs)
        Must return a matplotlib Figure.
    datasets : dict[str, DataFrame]
        Mapping: dataset_name -> dataset DataFrame.
    start_cases : int, optional
        Initial number of cases to include (default: 50).
    step_size : int, optional
        Step size for the case count slider (default: 50).
    extra_widgets : list[widgets.Widget] | None, optional
        Additional widgets to include in the header (default: None).
        These widgets will trigger re-rendering when their value changes.
    **plot_kwargs : any
        Forwarded into the plot function.

    Returns
    -------
    widgets.VBox

    """
    max_cases_global = min(len(df) for df in datasets.values())

    case_slider = widgets.IntSlider(
        value=min(start_cases, max_cases_global),
        min=1,
        max=max_cases_global,
        step=step_size,
        description="Cases:",
        continuous_update=False,
        readout=True,
    )

    prev_btn = widgets.Button(description="⟨")
    next_btn = widgets.Button(description="⟩")

    out = widgets.Output()
    extra_widgets = extra_widgets or []

    def _render(_change: None = None) -> None:
        with out:
            out.clear_output(wait=True)
            fig = plot_func(datasets=datasets, max_cases=int(case_slider.value), **plot_kwargs)
            display(fig)
            plt.close(fig)

    def _step(delta: int) -> None:
        new_val = case_slider.value + delta * step_size
        new_val = max(1, min(max_cases_global, new_val))
        case_slider.value = new_val

    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    case_slider.observe(_render, names="value")

    for w in extra_widgets:
        w.observe(_render, names="value")

    _render()

    header = [case_slider, prev_btn, next_btn]
    header.extend(extra_widgets)

    return widgets.VBox([widgets.HBox(header), out])


# ============================================================================
# 3) GLOBAL (NO-CASE) DROPDOWN VIEWER
# ============================================================================


def make_interactive_global_plot_dropdown(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Viewer for global plots that do not depend on case indexing.

    Parameters
    ----------
    plot_func : callable
        Function of the form:
            plot_func(dfs={name: df}, dataset_name=name, **kwargs)
        Must return a matplotlib Figure.
    datasets : dict[str, DataFrame]
        Mapping: dataset_name -> dataset DataFrame.
    **plot_kwargs : any
        Forwarded into the plot function.

    Returns
    -------
    widgets.VBox

    """
    names = list(datasets.keys())

    if len(names) > 1:
        dropdown = ui_dataset_dropdown(names)
        active_name = dropdown.value or names[0]
    else:
        dropdown = None
        active_name = names[0]

    out = ui_output()

    def _render() -> None:
        name = active_name if dropdown is None else (dropdown.value or names[0])
        df = datasets[name]
        with out:
            out.clear_output(wait=True)
            fig = plot_func(dfs={name: df}, dataset_name=name, **plot_kwargs)
            display(fig)
            plt.close(fig)

    if dropdown is not None:
        dropdown.observe(lambda _: _render(), names="value")

    _render()

    if dropdown is None:
        return widgets.VBox([out])

    return widgets.VBox([widgets.HBox([dropdown]), out])
