"""
Exploratory spectral analysis tools for porous-media simulation data.

This module provides utilities to compute spectral quantities
(two-dimensional PSDs and radial energy spectra) and to build
interactive spectral visualisations for simulation fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure


# ======================================================================
# Internal spectral utilities
# ======================================================================


def _hann2d(ny: int, nx: int) -> np.ndarray:
    """
    Create a two-dimensional Hann window.

    A separable Hann weighting is applied in both spatial directions
    to reduce edge effects before computing the FFT.

    Args:
        ny (int): Number of points in the vertical direction.
        nx (int): Number of points in the horizontal direction.

    Returns:
        np.ndarray: Hann window of shape (ny, nx).

    """
    return np.outer(np.hanning(ny), np.hanning(nx))


def _fft2_psd(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the two-dimensional FFT and power spectral density.

    The input field is mean-subtracted and Hann-windowed. The result
    is returned in centred form (fftshifted), together with the
    corresponding wavenumber grids.

    Args:
        field (np.ndarray): Two-dimensional scalar field.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.

    Returns:
        tuple:
            np.ndarray: Centred power spectral density.
            np.ndarray: Centred wavenumber grid kx.
            np.ndarray: Centred wavenumber grid ky.

    """
    a = np.asarray(field, float)
    a -= np.mean(a)
    a *= _hann2d(*a.shape)

    F = np.fft.fft2(a)
    PSD = np.abs(F) ** 2 / a.size

    kx = np.fft.fftfreq(a.shape[1], d=dx)
    ky = np.fft.fftfreq(a.shape[0], d=dy)
    kx, ky = np.meshgrid(kx, ky)

    return np.fft.fftshift(PSD), np.fft.fftshift(kx), np.fft.fftshift(ky)


def _radial_spectrum(
    PSD: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the isotropic radial energy spectrum E(k).

    The two-dimensional PSD is binned by radial wavenumber distance.
    The result is the mean spectral energy in each radial band.

    Args:
        PSD (np.ndarray): Power spectral density.
        kx (np.ndarray): Wavenumber grid in the x direction.
        ky (np.ndarray): Wavenumber grid in the y direction.
        n_bins (int): Number of radial bins.

    Returns:
        tuple:
            np.ndarray: Radial wavenumber centres.
            np.ndarray: Energy density E(k).

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

    k_centres = 0.5 * (edges[:-1] + edges[1:])
    return k_centres[:n], E


# ======================================================================
# Field lists
# ======================================================================

INPUT_KEYS = ["kappaxx"]
OUTPUT_KEYS = ["p", "U"]
ALL_KEYS = INPUT_KEYS + OUTPUT_KEYS


# ======================================================================
# Interactive 2D spectral overview
# ======================================================================


def plot_spectral_overview(df: pd.DataFrame, dataset_name: str) -> widgets.VBox:
    """
    Build an interactive viewer for 2D spectral maps (PSD).

    Each case is visualised by transforming the fields listed in
    `ALL_KEYS` into log-scaled 2D PSD maps using FFT. A separate subplot
    is shown for each field, including an individual colourbar.

    Navigation across cases is handled by the generic util_plot navigator.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation dataset. Must contain:
        - the fields defined in ALL_KEYS
        - spatial coordinates "x" and "y" for computing dx, dy
    dataset_name : str
        Name used in the subplot title (typically the batch name).

    Returns
    -------
    widgets.VBox
        Fully interactive spectral viewer with next/previous navigation.

    """
    df = df.reset_index(drop=True)

    def plot_case(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Plot a multi-field 2D PSD overview for a single simulation case."""
        row = df.iloc[idx]

        fields = {key: np.asarray(row[key], float) for key in ALL_KEYS}

        x = np.asarray(row["x"])
        y = np.asarray(row["y"])
        dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
        dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))

        ncols = len(fields)
        fig, axes = plt.subplots(1, ncols, figsize=(12, 4))
        axes = axes.ravel()

        for ax, (label, field) in zip(axes, fields.items(), strict=True):
            PSD, kx, ky = _fft2_psd(field, dx, dy)
            logPSD = np.log10(PSD + 1e-20)

            vmin, vmax = np.nanpercentile(logPSD, [2, 98])

            im = ax.pcolormesh(
                kx,
                ky,
                logPSD,
                cmap="inferno",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{label} spectrum")
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

        fig.suptitle(f"{dataset_name} - Case {idx}", fontsize=12, y=0.98)
        fig.tight_layout()
        return fig

    return util.util_plot.make_interactive_plot(
        n_cases=len(df),
        plot_func=plot_case,
        df=df,
        dataset_name=dataset_name,
    )


# ======================================================================
# Interactive vertical spectral evolution
# ======================================================================


def plot_spectral_vertical(df: pd.DataFrame, dataset_name: str) -> widgets.VBox:
    """
    Build an interactive viewer for vertical spectral evolution.

    For each field in `ALL_KEYS`, two radial spectra are computed:
    one from a thin slice near the bottom of the domain and one
    from a slice near mid-height. This highlights changes in spatial
    structure across the domain height.

    Navigation across cases is handled by the generic util_plot navigator.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation dataset. Must contain:
        - the fields defined in ALL_KEYS
        - spatial coordinates "x" and "y" for computing dx, dy
    dataset_name : str
        Name used in the subplot title (typically the batch name).

    Returns
    -------
    widgets.VBox
        Fully interactive viewer with per-field vertical line spectra.

    """
    df = df.reset_index(drop=True)

    def plot_case(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Plot vertical (bottom/mid) radial spectra for a single case."""
        row = df.iloc[idx]
        fields = {key: np.asarray(row[key], float) for key in ALL_KEYS}

        x = np.asarray(row["x"])
        y = np.asarray(row["y"])
        dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
        dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))

        ncols = len(fields)
        fig, axes = plt.subplots(1, ncols, figsize=(12, 4))
        axes = axes.ravel()

        y_coords = np.linspace(0, 1, fields[ALL_KEYS[0]].shape[0])
        y_low, y_high = 0.05, 0.70
        win = 0.01

        mask_low = (y_coords >= y_low - win) & (y_coords <= y_low + win)
        mask_high = (y_coords >= y_high - win) & (y_coords <= y_high + win)

        for ax, (label, field) in zip(axes, fields.items(), strict=True):
            seg_low = np.atleast_2d(field[mask_low].mean(axis=0))
            seg_high = np.atleast_2d(field[mask_high].mean(axis=0))

            PSD_low, kx_low, ky_low = _fft2_psd(seg_low, dx, dy)
            PSD_high, kx_high, ky_high = _fft2_psd(seg_high, dx, dy)

            k_low, E_low = _radial_spectrum(PSD_low, kx_low, ky_low)
            k_high, E_high = _radial_spectrum(PSD_high, kx_high, ky_high)

            ax.loglog(k_low, E_low, lw=1.4, label=f"y={y_low:.2f}")
            ax.loglog(k_high, E_high, lw=1.4, label=f"y={y_high:.2f}")
            ax.set_title(label)
            ax.set_xlabel("k")
            ax.grid(True, which="both", ls=":")
            ax.legend(fontsize=8)

        fig.suptitle(f"{dataset_name} - Case {idx}", fontsize=12, y=0.98)
        fig.tight_layout()
        return fig

    return util.util_plot.make_interactive_plot(
        n_cases=len(df),
        plot_func=plot_case,
        df=df,
        dataset_name=dataset_name,
    )
