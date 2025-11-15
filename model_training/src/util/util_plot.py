"""
Generic plotting utilities and interactive case navigation for notebooks.

This module centralises:
- Field extraction from DataFrames
- Generic figure layout with multiple subplots
- Case navigator widget (← →)
- High-level interactive viewer for any plot function

All EDA, evaluation and sensitivity modules rely on this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ======================================================================
# Internal: extract fields for a single case
# ======================================================================


def _prepare_fields(
    df: pd.DataFrame,
    case_idx: int,
    fields_to_plot: Sequence[str],
) -> tuple[dict[str, np.ndarray], float, float, str]:
    """
    Extract selected fields and compute grid spacing for one simulation case.

    Args:
        df (pd.DataFrame): Dataset containing simulation cases.
        case_idx (int): Case index (0-based).
        fields_to_plot (list[str]): Keys to extract from the DataFrame.

    Returns:
        tuple:
            dict[str, np.ndarray]: Mapping field_name -> 2D array.
            float: dx grid spacing in x-direction.
            float: dy grid spacing in y-direction.
            str: Case title for figures.

    """
    row = df.iloc[case_idx]

    x = np.asarray(row["x"])
    y = np.asarray(row["y"])
    dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
    dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))

    fields = {key: np.asarray(row[key], dtype=float) for key in fields_to_plot}

    title = f"Case_{case_idx + 1:04d}"
    return fields, dx, dy, title


# ======================================================================
# Internal: generic multi-field figure
# ======================================================================


def _plot_base(
    fields: dict[str, np.ndarray],
    dx: float,
    dy: float,
    dataset_name: str,
    title: str,
    field_plotter: Callable[[Axes, str, np.ndarray, float, float, int], None],
    title_suffix: str,
    figsize: tuple[int, int],
) -> Figure:
    """
    Build a figure with one subplot per field using the supplied plot callback.

    Args:
        fields (dict): Mapping field_name -> 2D array.
        dx (float): Grid spacing x.
        dy (float): Grid spacing y.
        dataset_name (str): Name of dataset batch.
        title (str): Base case title.
        field_plotter (callable): Called for each subplot.
        title_suffix (str): Extra text appended to the figure title.
        figsize (tuple): Figure size in inches.

    Returns:
        Figure: Fully built Matplotlib figure.

    """
    ncols = len(fields)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    axes = axes[0]

    for j, (label, field) in enumerate(fields.items()):
        field_plotter(axes[j], label, field, dx, dy, j)

    fig.suptitle(f"Batch: {dataset_name} - {title} - {title_suffix} ", fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


# ======================================================================
# Internal: df + index → figure
# ======================================================================


def _make_figure_generic(
    df: pd.DataFrame,
    dataset_name: str,
    case_idx: int,
    field_plotter: Callable[[Axes, str, np.ndarray, float, float, int], None],
    fields_to_plot: Sequence[str],
    title_suffix: str,
    figsize: tuple[int, int],
) -> Figure:
    """
    Create a figure for a specific case index using the generic plot template.

    Args:
        df (pd.DataFrame): Simulation dataset.
        dataset_name (str): Dataset name.
        case_idx (int): Case index to plot.
        field_plotter (callable): Per-field plot function.
        fields_to_plot (list[str]): Keys to extract for plotting.
        title_suffix (str): Extra text for figure titles.
        figsize (tuple): Figure size.

    Returns:
        Figure: Matplotlib figure.

    """
    fields, dx, dy, title = _prepare_fields(df, case_idx, fields_to_plot)
    return _plot_base(
        fields=fields,
        dx=dx,
        dy=dy,
        dataset_name=dataset_name,
        title=title,
        field_plotter=field_plotter,
        title_suffix=title_suffix,
        figsize=figsize,
    )


# ======================================================================
# Public API: case navigator widget
# ======================================================================


def make_case_navigator(
    n_cases: int,
    plot_func: Callable[[int], Figure],
) -> widgets.VBox:
    """
    Create an interactive navigator for stepping through simulation cases.

    Args:
        n_cases (int): Number of cases available.
        plot_func (callable): Maps case_idx → Matplotlib Figure.

    Returns:
        widgets.VBox: Interactive panel for Jupyter notebooks.

    """
    idx = widgets.BoundedIntText(value=1, min=1, max=n_cases, description="Case:")
    prev_btn = widgets.Button(description="←", layout={"width": "38px"})
    next_btn = widgets.Button(description="→", layout={"width": "38px"})
    out = widgets.Output()

    def _render(case_idx: int) -> None:
        with out:
            clear_output(wait=True)
            fig = plot_func(case_idx)
            display(fig)
            plt.close(fig)

    def _step(delta: int) -> None:
        idx.value = max(1, min(n_cases, idx.value + delta))

    # Connect events
    idx.observe(lambda c: _render(c["new"] - 1), names="value")
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))

    # Initial render
    _render(0)

    return widgets.VBox([widgets.HBox([idx, prev_btn, next_btn]), out])


# ======================================================================
# Public API: universal interactive viewer
# ======================================================================


def make_interactive_plot(
    df: pd.DataFrame,
    dataset_name: str,
    field_plotter: Callable[[Axes, str, np.ndarray, float, float, int], None],
    fields_to_plot: Sequence[str],
    title_suffix: str = "",
    figsize: tuple[int, int] = (12, 6),
) -> widgets.VBox:
    """
    Build a complete interactive viewer for a dataset.

    Args:
        df (pd.DataFrame): Dataset containing simulation cases.
        dataset_name (str): Name of dataset batch.
        field_plotter (callable): Function applied to each subplot.
        fields_to_plot (list[str]): Keys to extract from df.
        title_suffix (str, optional): Extra text for titles. Defaults to "".
        figsize (tuple[int, int], optional): Figure size. Defaults to (12, 6).

    Returns:
        widgets.VBox: Navigator widget.

    """

    def _make_fig(case_idx: int) -> Figure:
        return _make_figure_generic(
            df=df,
            dataset_name=dataset_name,
            case_idx=case_idx,
            field_plotter=field_plotter,
            fields_to_plot=fields_to_plot,
            title_suffix=title_suffix,
            figsize=figsize,
        )

    return make_case_navigator(len(df), _make_fig)
