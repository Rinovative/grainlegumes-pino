"""
Generic interactive navigation utilities for notebooks.

This module provides:
- A universal case navigator (Prev/Next)
- A generic viewer that forwards any extra arguments to a plot function
- No assumptions about data structure, fields, dimensions or layout

All specialised modules (EDA, spectral, evaluation, histograms, etc.)
define their own plot_func(case_idx, **kwargs) that returns a Figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


# ======================================================================
# Universal navigator
# ======================================================================


def make_case_navigator(
    n_cases: int,
    plot_func: Callable[..., Figure],
    *,
    start_idx: int = 0,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Create a universal navigator for stepping through cases.

    Parameters
    ----------
    n_cases : int
        Total number of cases available.
    plot_func : callable
        Must have signature: plot_func(case_idx, **kwargs) -> Figure
    start_idx : int
        Initial case index to display.
    **plot_kwargs :
        Arbitrary keyword arguments forwarded to plot_func.

    Returns
    -------
    widgets.VBox
        Fully interactive notebook widget.

    """
    idx = widgets.BoundedIntText(
        value=start_idx,
        min=0,
        max=n_cases - 1,
        description="Case:",
        layout=widgets.Layout(width="140px"),
    )
    prev_btn = widgets.Button(description="←", layout={"width": "38px"})
    next_btn = widgets.Button(description="→", layout={"width": "38px"})
    out = widgets.Output()

    def _render(case_idx: int) -> None:
        with out:
            out.clear_output(wait=True)
            fig = plot_func(case_idx, **plot_kwargs)
            display(fig)
            plt.close(fig)

    def _step(delta: int) -> None:
        idx.value = max(0, min(n_cases - 1, idx.value + delta))

    idx.observe(lambda c: _render(c["new"]), names="value")
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))

    _render(start_idx)

    return widgets.VBox([widgets.HBox([idx, prev_btn, next_btn]), out])


# ======================================================================
# Simple wrapper for convenience
# ======================================================================


def make_interactive_plot(
    n_cases: int,
    plot_func: Callable[..., Figure],
    *,
    start_idx: int = 0,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Build a complete interactive viewer from any plot function.

    Parameters
    ----------
    n_cases : int
        Number of cases to iterate over.
    plot_func : callable
        plot_func(case_idx, **kwargs) -> Figure
    start_idx : int
        Initial index shown.
    **plot_kwargs :
        Extra keyword arguments passed into plot_func.

    Returns
    -------
    widgets.VBox
        Universal plot navigator.

    """
    return make_case_navigator(
        n_cases=n_cases,
        plot_func=plot_func,
        start_idx=start_idx,
        **plot_kwargs,
    )
