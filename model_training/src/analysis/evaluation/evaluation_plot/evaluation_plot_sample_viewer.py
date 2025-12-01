"""
Stable evaluation viewer WITH a colorbar at every subplot.

Designed for VS Code Notebook stability:
- No constrained_layout
- No plt.show()
- AGG backend
- One Output widget
- Per-subplot colorbars (fast + safe)
"""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

# ==============================================================================
# NPZ LOADING
# ==============================================================================


def _load_npz(row, debug: bool = False):
    """
    Load pred/gt/err arrays from NPZ.

    Parameters
    ----------
    row : pandas.Series with 'npz_path'
    debug : bool

    Returns
    -------
    pred, gt, err : np.ndarray

    """
    npz_path = Path(row["npz_path"])
    if debug:
        print(f"[DEBUG] Load NPZ: {npz_path}")

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    return data["pred"][0], data["gt"][0], data["err"][0]


# ==============================================================================
# FIGURE DRAWING (WITH COLORBARS)
# ==============================================================================


def _draw_case(df, idx: int, dataset_name: str, debug: bool = False):
    """
    Draw a 4×3 grid: pred / gt / abs(err) for channels p,u,v,U.

    Each subplot gets its own colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    if debug:
        print(f"[DEBUG] Drawing case {idx}")

    pred, gt, err = _load_npz(df.iloc[idx], debug=debug)

    # No constrained_layout → stable
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(13, 10),
        constrained_layout=False,
    )

    fig.suptitle(f"{dataset_name} – Case {idx}", fontsize=12, y=0.98)

    channels = ["p", "u", "v", "U"]
    cmap_pred = "viridis"
    cmap_gt = "viridis"
    cmap_err = "coolwarm"

    for i, label in enumerate(channels):
        # Pred
        ax = axes[i, 0]
        im = ax.imshow(pred[i], cmap=cmap_pred)
        ax.set_title(f"{label} pred", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.01)

        # GT
        ax = axes[i, 1]
        im = ax.imshow(gt[i], cmap=cmap_gt)
        ax.set_title(f"{label} true", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.01)

        # Error
        ax = axes[i, 2]
        im = ax.imshow(np.abs(err[i]), cmap=cmap_err)
        ax.set_title(f"{label} abs(err)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.01)

    return fig


# ==============================================================================
# PUBLIC WIDGET VIEWER
# ==============================================================================


def plot_sample_prediction_overview(
    df,
    dataset_name: str = "",
    debug: bool = False,
) -> widgets.VBox:
    """
    Robust evaluation viewer (Prev/Next) with colorbars at every subplot.

    Parameters
    ----------
    df : pandas.DataFrame with 'npz_path'
    dataset_name : str
    debug : bool

    Returns
    -------
    VBox : fully interactive viewer

    """
    n_cases = len(df)

    # Widgets
    idx = widgets.BoundedIntText(
        value=0,
        min=0,
        max=n_cases - 1,
        description="Case:",
        layout=widgets.Layout(width="140px"),
    )
    prev_btn = widgets.Button(description="← Prev", layout=widgets.Layout(width="70px"))
    next_btn = widgets.Button(description="Next →", layout=widgets.Layout(width="70px"))
    out = widgets.Output()

    # Rendering function
    def _render(case_idx: int):
        with out:
            clear_output(wait=True)
            fig = _draw_case(df, case_idx, dataset_name, debug=debug)
            display(fig)
            plt.close(fig)

    # Buttons
    def _step(delta: int):
        idx.value = max(0, min(n_cases - 1, idx.value + delta))

    idx.observe(lambda e: _render(e["new"]), names="value")
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))

    # First figure
    _render(0)

    return widgets.VBox(
        [
            widgets.HBox([idx, prev_btn, next_btn]),
            out,
        ]
    )
