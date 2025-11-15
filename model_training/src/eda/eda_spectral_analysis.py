"""
Exploratory spectral analysis tools for porous-media simulation data.

This module provides utilities to compute spectral quantities
(two-dimensional PSDs and radial energy spectra) and to build
interactive spectral visualisations for simulation fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes


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
# 1) Interactive 2D spectral overview
# ======================================================================


def plot_spectral_overview(df: pd.DataFrame, dataset_name: str) -> widgets.VBox:
    """
    Create an interactive viewer for two-dimensional spectral maps.

    For each case the fields kappaxx, p and U are transformed into
    PSD maps that are displayed side by side. Navigation across all
    simulation cases is handled by util.util_plots.

    Args:
        df (pd.DataFrame): Simulation dataset.
        dataset_name (str): Name used in figure titles.

    Returns:
        widgets.VBox: Interactive spectral viewer.

    """
    df = df.reset_index(drop=True)

    def _plot_single_2d_spectrum(
        ax: Axes,
        label: str,
        field: np.ndarray,
        dx: float,
        dy: float,
        _: int,
    ) -> None:
        """
        Plot a single two-dimensional PSD map.

        The PSD is displayed as a log-scaled colour plot with axes
        corresponding to wavenumbers in x and y direction.

        Args:
            ax (Axes): Axes object to draw into.
            label (str): Name of the field.
            field (np.ndarray): Scalar field values.
            dx (float): Grid spacing in the x direction.
            dy (float): Grid spacing in the y direction.
            j (int): Subplot index.

        Returns:
            None

        """
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

        ax.set_aspect("equal")
        ax.set_title(f"{label} spectrum")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")

        fig = ax.figure
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("log10 PSD")

    return util.util_plot.make_interactive_plot(
        df=df,
        dataset_name=dataset_name,
        field_plotter=_plot_single_2d_spectrum,
        fields_to_plot=ALL_KEYS,
        title_suffix="2D spectral overview",
        figsize=(12, 4),
    )


# ======================================================================
# 2) Interactive vertical spectral evolution
# ======================================================================


def plot_spectral_vertical(df: pd.DataFrame, dataset_name: str) -> widgets.VBox:
    """
    Create an interactive viewer for vertical spectral evolution.

    For each field the spectral energy is shown as a radial spectrum
    at two vertical slices in the domain (near the bottom and the mid
    height). This visualises vertical changes in spatial structure.

    Args:
        df (pd.DataFrame): Simulation dataset.
        dataset_name (str): Batch name used in figure titles.

    Returns:
        widgets.VBox: Interactive spectral line viewer.

    """
    df = df.reset_index(drop=True)

    def _plot_single_vertical_spectrum(
        ax: Axes,
        label: str,
        field: np.ndarray,
        dx: float,
        dy: float,
        _: int,
    ) -> None:
        """
        Plot vertical line spectra for a single field.

        Two radial spectra are computed from narrow vertical slices
        at y≈0.05 and y≈0.70 to highlight changes in spectral
        structure across the domain height.

        Args:
            ax (Axes): Axes to draw into.
            label (str): Field name.
            field (np.ndarray): Two-dimensional scalar field.
            dx (float): Grid spacing in the x direction.
            dy (float): Grid spacing in the y direction.
            j (int): Subplot index.

        Returns:
            None

        """
        ny = field.shape[0]
        y = np.linspace(0, 1, ny)

        y_low, y_high = 0.05, 0.70
        win = 0.01

        mask_low = (y >= y_low - win) & (y <= y_low + win)
        mask_high = (y >= y_high - win) & (y <= y_high + win)

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

    return util.util_plot.make_interactive_plot(
        df=df,
        dataset_name=dataset_name,
        field_plotter=_plot_single_vertical_spectrum,
        fields_to_plot=ALL_KEYS,
        title_suffix="vertical spectral lines",
        figsize=(12, 4),
    )
