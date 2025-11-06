from __future__ import annotations

from typing import Callable, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ============================================================
# --- Core spectral utilities ---
# ============================================================


def _hann2d(ny: int, nx: int) -> np.ndarray:
    """
    Creates a 2D Hann window for spectral smoothing.

    Args:
        ny (int): Number of rows (y-dimension).
        nx (int): Number of columns (x-dimension).

    Returns:
        np.ndarray: 2D Hann weighting matrix of shape (ny, nx).
    """
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return np.outer(wy, wx)


def _fft2_psd(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D FFT and corresponding Power Spectral Density (PSD).

    Args:
        field (np.ndarray): 2D field data.
        dx (float): Spatial step in x-direction.
        dy (float): Spatial step in y-direction.

    Returns:
        tuple:
            np.ndarray: Centered 2D power spectral density.
            np.ndarray: Frequency grid kx.
            np.ndarray: Frequency grid ky.
    """
    a = np.asarray(field, dtype=float)
    a -= np.mean(a)
    a *= _hann2d(*a.shape)
    F = np.fft.fft2(a)
    PSD = np.abs(F) ** 2 / (a.shape[0] * a.shape[1])
    kx = np.fft.fftfreq(a.shape[1], d=dx)
    ky = np.fft.fftfreq(a.shape[0], d=dy)
    kx, ky = np.meshgrid(kx, ky)
    return np.fft.fftshift(PSD), np.fft.fftshift(kx), np.fft.fftshift(ky)


def _radial_spectrum(PSD: np.ndarray, kx: np.ndarray, ky: np.ndarray, n_bins: int = 200):
    """
    Computes the isotropic radial spectrum E(k) from a 2D PSD.

    Args:
        PSD (np.ndarray): Power spectral density.
        kx (np.ndarray): Wave-number grid in x-direction.
        ky (np.ndarray): Wave-number grid in y-direction.
        n_bins (int, optional): Number of bins for radial averaging. Defaults to 200.

    Returns:
        tuple:
            np.ndarray: Radial wavenumber centers.
            np.ndarray: Corresponding energy density values E(k).
    """
    kr = np.sqrt(kx**2 + ky**2).ravel()
    ps = PSD.ravel()
    mask = np.isfinite(kr) & np.isfinite(ps)
    kr, ps = kr[mask], ps[mask]
    edges = np.linspace(kr.min(), kr.max(), n_bins + 1)
    idx = np.digitize(kr, edges) - 1
    sums = np.bincount(idx, weights=ps)
    counts = np.bincount(idx)
    n = min(len(sums), len(edges) - 1)
    E = sums[:n] / np.maximum(counts[:n], 1)
    k_centers = 0.5 * (edges[:-1] + edges[1:])
    return k_centers[:n], E


# ============================================================
# --- Shared helpers for EDA plots ---
# ============================================================


def _prepare_fields(df, case_idx: int) -> tuple[dict[str, np.ndarray], float, float, str]:
    """
    Extracts κxx, p, U fields and grid spacing for a given case.

    Args:
        df (pd.DataFrame): DataFrame containing simulation cases.
        case_idx (int): Case index (0-based).

    Returns:
        tuple:
            dict[str, np.ndarray]: Dictionary with field arrays ('κₓₓ', 'p', 'U').
            float: dx (spatial step in x-direction).
            float: dy (spatial step in y-direction).
            str: Case title string.
    """
    row = df.iloc[case_idx]
    x, y = np.asarray(row["x"]), np.asarray(row["y"])
    dx = np.nanmedian(np.abs(np.diff(x, axis=1)))
    dy = np.nanmedian(np.abs(np.diff(y, axis=0)))
    fields = {
        "κₓₓ": np.asarray(row["kappaxx"], dtype=float),
        "p": np.asarray(row["p"], dtype=float),
        "U": np.asarray(row.get("U", np.sqrt(row["u"] ** 2 + row["v"] ** 2)), dtype=float),
    }
    return fields, float(dx), float(dy), f"Case_{case_idx + 1:04d}"


def make_case_navigator(n_cases: int, plot_func: Callable[[int], Figure]) -> widgets.VBox:
    """
    Creates a simple navigation widget (← / →) to cycle through simulation cases.

    Args:
        n_cases (int): Number of available cases.
        plot_func (callable): Function accepting an integer case index (0-based)
            and returning a Matplotlib figure.

    Returns:
        widgets.VBox: Interactive navigation panel with output area.
    """
    idx = widgets.BoundedIntText(value=1, min=1, max=n_cases, description="Case:", layout={"width": "140px"})
    prev = widgets.Button(description="←", layout={"width": "36px"})
    nxt = widgets.Button(description="→", layout={"width": "36px"})
    out = widgets.Output(layout={"border": "1px solid #ddd", "padding": "5px"})

    def render(i):
        with out:
            clear_output(wait=True)
            plt.ioff()
            display(plot_func(i - 1))
            plt.close()

    def step(d):
        idx.value = max(1, min(n_cases, idx.value + d))

    idx.observe(lambda c: render(c["new"]), names="value")
    prev.on_click(lambda _: step(-1))
    nxt.on_click(lambda _: step(1))
    render(1)
    return widgets.VBox([widgets.HBox([idx, prev, nxt]), out])


def _plot_base(
    df,
    dataset_name: str,
    case_idx: int,
    field_plotter: Callable[[Axes, str, np.ndarray, float, float, int], None],
    title_suffix: str,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Generic plotting template for spectral visualizations.

    Handles:
      - field extraction and FFT preparation
      - figure creation and layout
      - case title and dataset annotation

    Args:
        df (pd.DataFrame): Dataset containing simulation cases.
        dataset_name (str): Name of the current batch.
        case_idx (int): Case index (0-based).
        field_plotter (callable): Function defining how each field is visualized
            (called per subplot with signature `(ax, label, field, dx, dy, j)`).
        title_suffix (str): Text appended to figure title.
        figsize (tuple, optional): Figure size in inches. Defaults to (12, 6).

    Returns:
        matplotlib.figure.Figure: Generated Matplotlib figure.
    """
    fields, dx, dy, title = _prepare_fields(df, case_idx)
    ncols = len(fields)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for j, (label, field) in enumerate(fields.items()):
        field_plotter(axes[j], label, field, dx, dy, j)

    fig.suptitle(f"{title} – {dataset_name}{title_suffix}", fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


# ============================================================
# --- Global Field Spectra Overview (κxx, p, U)
# ============================================================


def plot_field_spectra_overview(df, dataset_name: str) -> widgets.VBox:
    """
    Interactive global spectral overview for fields κxx, p, and U.

    Shows:
      - 2D power spectra for each field (individual color scales)
      - Normalized radial spectra E(k)

    Args:
        df (pd.DataFrame): Dataset with 'x', 'y', 'kappaxx', 'p', and 'U' fields.
        dataset_name (str): Batch name used in figure titles.

    Returns:
        widgets.VBox: Interactive case navigation widget with spectral plots.
    """
    df = df.reset_index(drop=True)
    cmap = "inferno"

    def field_plotter(ax, label, field, dx, dy, j):
        PSD, kx, ky = _fft2_psd(field, dx, dy)
        vmin, vmax = np.nanpercentile(np.log10(PSD + 1e-20), [2, 98])
        im = ax.pcolormesh(
            kx,
            ky,
            np.log10(PSD + 1e-20),
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{label} 2D spectrum", fontsize=10)
        ax.set_xlabel("kₓ")
        ax.set_ylabel("kᵧ")
        fig = ax.get_figure()
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("log₁₀ PSD", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    return make_case_navigator(
        len(df),
        lambda i: _plot_base(df, dataset_name, i, field_plotter, " (2D overview)", figsize=(12, 8)),
    )


# ============================================================
# --- Vertical Spectral Evolution (Linienplots)
# ============================================================


def plot_vertical_spectral_lines(df, dataset_name: str) -> widgets.VBox:
    """
    Interactive vertical spectral evolution view (E(k, y)) for κxx, p, and U.

    For each field, plots two radial spectra:
      - Lower region (around y ≈ 0.05 m)
      - Upper region (around y ≈ 0.70 m)

    Args:
        df (pd.DataFrame): Dataset with 'x', 'y', 'kappaxx', 'p', and 'U' fields.
        dataset_name (str): Batch name used in figure titles.

    Returns:
        widgets.VBox: Interactive case navigation widget with vertical spectra.
    """
    df = df.reset_index(drop=True)

    def field_plotter(ax, label, field, dx, dy, j):
        y = np.asarray(df.iloc[0]["y"])[:, 0]
        y_low, y_high, win = 0.05, 0.70, 0.01
        mask_low = (y >= y_low - win) & (y <= y_low + win)
        mask_high = (y >= y_high - win) & (y <= y_high + win)
        seg_low = field[mask_low].mean(axis=0, keepdims=True)
        seg_high = field[mask_high].mean(axis=0, keepdims=True)
        y_low_mean = np.mean(y[mask_low])
        y_high_mean = np.mean(y[mask_high])

        PSD_low, kx, ky = _fft2_psd(seg_low, dx, dy)
        PSD_high, _, _ = _fft2_psd(seg_high, dx, dy)
        k, E_low = _radial_spectrum(PSD_low, kx, ky)
        _, E_high = _radial_spectrum(PSD_high, kx, ky)

        ax.loglog(
            k,
            np.maximum(E_low, 1e-300),
            lw=1.5,
            color="C0",
            label=f"y={y_low_mean:.2f} m",
        )
        ax.loglog(
            k,
            np.maximum(E_high, 1e-300),
            lw=1.5,
            color="C1",
            label=f"y={y_high_mean:.2f} m",
        )
        ax.set_title(label)
        ax.set_xlabel("Wavenumber k")
        ax.grid(True, which="both", ls=":")
        ax.legend(fontsize=8)

    return make_case_navigator(
        len(df),
        lambda i: _plot_base(df, dataset_name, i, field_plotter, " (vertical lines)", figsize=(12, 4)),
    )
