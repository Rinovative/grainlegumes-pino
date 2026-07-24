"""
===============================================================================
eda_plot_spectral_analysis.py
===============================================================================
Plot frequency-domain summaries for exploratory dataset analysis.

Responsibilities:
  - Compute and plot 2D power spectra
  - Compute radial energy spectra
  - Build interactive spectral dataset viewers

Design principles:
  - Spectral summaries use Hann-windowed FFTs
  - Wavenumber plots expose dominant spatial scales
  - Dataset selection is shared with EDA case-statistics helpers

Boundaries:
  - Model spectral diagnostics belong to evaluation_plot_spectral_analysis
  - Physics residual diagnostics belong to learning.losses and analysis.evaluation
===============================================================================
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src import analysis

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure

    CheckboxGroup = analysis.ui.components.CheckboxGroup


# ======================================================================
# Internal DataFrame utilities
# ======================================================================
def _infer_field_keys(df: pd.DataFrame) -> list[str]:
    """
    Infer 2D field keys from a DataFrame by inspecting the first row.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing simulation case data.

    Returns
    -------
    list[str]
        List of column names corresponding to 2D numeric fields.

    """
    keys: list[str] = []

    sample = df.iloc[0]
    for col in df.columns:
        if col in {"x", "y", "meta"}:
            continue

        arr = np.asarray(sample[col])
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):  # noqa: PLR2004
            keys.append(col)

    return keys


def _infer_ncols(n_items: int, *, max_cols: int = 4) -> int:
    """
    Infer a suitable number of subplot columns for a given number of items.

    Parameters
    ----------
    n_items : int
        Total number of items to plot.
    max_cols : int
        Maximum number of columns allowed.

    Returns
    -------
    int
        Number of columns to use for subplots.

    """
    return min(max_cols, max(1, int(np.ceil(np.sqrt(n_items)))))


# ======================================================================
# Internal spectral utilities
# ======================================================================


def _hann2d(ny: int, nx: int) -> np.ndarray:
    """
    Create a two-dimensional Hann window.

    A separable Hann weighting is applied in both spatial directions
    to reduce edge effects before computing the FFT.

    Parameters
    ----------
    ny : int
        Number of grid points in the y direction.
    nx : int
        Number of grid points in the x direction.

    Returns
    -------
        np.ndarray: Hann window of shape (ny, nx).

    """
    return np.outer(np.hanning(ny), np.hanning(nx))


def _fft2_psd(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the two-dimensional FFT and power spectral density.

    The input field is mean-subtracted and Hann-windowed. The result
    is returned in centred form (fftshifted), together with the
    corresponding wavenumber grids.

    Parameters
    ----------
    field : np.ndarray
        2D array representing the spatial field to analyze.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.

    Returns
    -------
        tuple:
            np.ndarray: Centred power spectral density.
            np.ndarray: Centred wavenumber grid kx.
            np.ndarray: Centred wavenumber grid ky.

    """
    a = field.astype(float, copy=True)
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

    Parameters
    ----------
    PSD : np.ndarray
        2D power spectral density.
    kx, ky : np.ndarray
        Wavenumber grids corresponding to the PSD.
    n_bins : int
        Number of radial bins to use.

    Returns
    -------
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


def _directional_spectrum(
    PSD: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    axis: str,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D directional energy spectrum Ex(kx) or Ey(ky).

    Parameters
    ----------
    PSD : np.ndarray
        2D power spectral density.
    kx, ky : np.ndarray
        Wavenumber grids.
    axis : {"x", "y"}
        Direction for spectral reduction.
    n_bins : int
        Number of bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (k_centres, E_dir)

    """
    if axis == "x":
        k = np.abs(kx).ravel()
    elif axis == "y":
        k = np.abs(ky).ravel()
    else:
        msg = "axis must be 'x' or 'y'"
        raise ValueError(msg)

    ps = PSD.ravel()

    mask = np.isfinite(k) & np.isfinite(ps)
    k, ps = k[mask], ps[mask]

    edges = np.linspace(0.0, k.max(), n_bins + 1)
    idx = np.digitize(k, edges) - 1

    sums = np.bincount(idx, weights=ps)
    counts = np.bincount(idx)

    n = min(len(sums), len(edges) - 1)
    E = sums[:n] / np.maximum(counts[:n], 1)

    k_centres = 0.5 * (edges[:-1] + edges[1:])
    return k_centres[:n], E


# ======================================================================
# Interactive 2D spectral overview
# ======================================================================


def plot_spectral_overview(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Build an interactive viewer for 2D spectral maps (PSD).

    Each case is visualised by transforming the fields listed in
    `ALL_KEYS` into log-scaled 2D PSD maps using FFT. A separate subplot
    is shown for each field, including an individual colourbar.

    Navigation across cases is handled by the generic analysis.ui.viewers navigator.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping: dataset_name -> DataFrame
        Each DataFrame must contain:
        - the fields defined in ALL_KEYS
        - spatial coordinates "x" and "y" for computing dx, dy

    Returns
    -------
    widgets.VBox
        Fully interactive spectral viewer with next/previous navigation.


    """
    # Navigator übernimmt Auswahl → kein df oder dataset_name direkt nötig

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Plot a multi-field 2D PSD overview for a single simulation case."""
        row = df.iloc[idx]

        field_keys = _infer_field_keys(df)
        fields = {key: np.asarray(row.loc[key], float) for key in field_keys}

        x = np.asarray(row.loc["x"])
        y = np.asarray(row.loc["y"])
        dx = float(np.nanmedian(np.diff(np.unique(x))))
        dy = float(np.nanmedian(np.diff(np.unique(y))))

        n_items = len(fields)
        ncols = _infer_ncols(n_items, max_cols=4)
        nrows = int(np.ceil(n_items / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, (label, field) in zip(axes[: len(fields)], fields.items(), strict=True):
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
            ax.set_xlabel(r"Wavenumber $k_x$ [$\frac{1}{m}$]")
            ax.set_ylabel(r"Wavenumber $k_y$ [$\frac{1}{m}$]")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

        for ax in axes[len(fields) :]:
            ax.axis("off")

        fig.suptitle(f"2D spectral maps: {dataset_name} - Case {idx + 1}", fontsize=12, y=0.98)
        fig.subplots_adjust(top=0.97, wspace=0.5, hspace=0.01)
        return fig

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
    )


# ======================================================================
# Interactive vertical spectral evolution
# ======================================================================


def plot_spectral_vertical(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Build an interactive viewer for vertical spectral evolution.

    For each field in `ALL_KEYS`, two radial spectra are computed:
    one from a thin slice near the bottom of the domain and one
    from a slice near mid-height. This highlights changes in spatial
    structure across the domain height.

    Navigation across cases is handled by the generic analysis.ui.viewers navigator.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping: dataset_name -> DataFrame
        Each DataFrame must contain:
        - the fields defined in ALL_KEYS
        - spatial coordinates "x" and "y" for computing dx, dy

    Returns
    -------
    widgets.VBox
        Fully interactive viewer with per-field vertical line spectra.

    """

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Plot vertical (bottom/mid) radial spectra for a single case."""
        row = df.iloc[idx]
        field_keys = _infer_field_keys(df)
        fields = {key: np.asarray(row.loc[key], float) for key in field_keys}

        x_arr = np.asarray(row.loc["x"])
        y_arr = np.asarray(row.loc["y"])
        dx = float(np.nanmedian(np.diff(np.unique(x_arr))))
        dy = float(np.nanmedian(np.diff(np.unique(y_arr))))

        n_items = len(fields)
        ncols = _infer_ncols(n_items, max_cols=4)
        nrows = int(np.ceil(n_items / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        field_keys = list(fields.keys())
        y_coords = np.linspace(0, 1, fields[field_keys[0]].shape[0])
        y_low, y_high = 0.05, 0.70
        win = 0.01

        mask_low = (y_coords >= y_low - win) & (y_coords <= y_low + win)
        mask_high = (y_coords >= y_high - win) & (y_coords <= y_high + win)

        for ax, (label, field) in zip(axes[: len(fields)], fields.items(), strict=True):
            seg_low = np.atleast_2d(field[mask_low].mean(axis=0))
            seg_high = np.atleast_2d(field[mask_high].mean(axis=0))

            PSD_low, kx_low, ky_low = _fft2_psd(seg_low, dx, dy)
            PSD_high, kx_high, ky_high = _fft2_psd(seg_high, dx, dy)

            k_low, E_low = _radial_spectrum(PSD_low, kx_low, ky_low)
            k_high, E_high = _radial_spectrum(PSD_high, kx_high, ky_high)

            mask_l = (k_low > 0) & (E_low > 0)
            mask_h = (k_high > 0) & (E_high > 0)

            if np.count_nonzero(mask_l) > 1:
                ax.loglog(k_low[mask_l], E_low[mask_l], lw=1.4, label=f"y={y_low:.2f}")

            if np.count_nonzero(mask_h) > 1:
                ax.loglog(k_high[mask_h], E_high[mask_h], lw=1.4, label=f"y={y_high:.2f}")
            ax.set_title(label)
            ax.set_xlabel(r"Wavenumber $k$ [$\frac{1}{m}$]")
            ax.set_ylabel(r"Spectral energy $E(k)$")
            ax.grid(True, which="both", ls=":")
            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

        for ax in axes[len(fields) :]:
            ax.axis("off")

        fig.suptitle(f"Vertical spectral profiles: {dataset_name} - Case {idx + 1}", fontsize=12, y=0.98)
        fig.subplots_adjust(top=0.94, wspace=0.40, hspace=0.3)
        return fig

    return analysis.ui.viewers.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=True,
    )


# ============================================================================
# CUMULATIVE SPECTRAL ENERGY (CASECOUNT VIEWER)
# ============================================================================


def plot_spectral_cumulative(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot cumulative radial spectral energy distributions.

    For each field, the cumulative spectral energy E_cum(k) is computed
    per case and aggregated across the first N cases using the median.
    This plot is intended to guide the selection of n_modes by identifying
    the effective spectral bandwidth of the data.

    The x-axis is intentionally limited to k <= 50.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets to plot.

    Returns
    -------
    widgets.VBox
        Interactive widget for viewing cumulative spectral energy distributions.

    """
    names = list(datasets.keys())

    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        dataset_selector: CheckboxGroup,
    ) -> Figure:
        active = analysis.eda.plots.case_statistics._selected_datasets(dataset_selector)  # noqa: SLF001 -- sibling plot helper

        sample_df = datasets[active[0]]
        field_keys = _infer_field_keys(sample_df)

        n_items = len(field_keys)
        ncols = _infer_ncols(n_items, max_cols=4)
        nrows = math.ceil(n_items / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols + 2.0, 3.6 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        cmap = plt.get_cmap("tab10")
        dataset_colors = {name: cmap(i % 10) for i, name in enumerate(active)}

        for ax, field in zip(axes[:n_items], field_keys, strict=True):
            for name in active:
                df = datasets[name].iloc[:max_cases]

                E_cum_all: list[np.ndarray] = []
                k_ref: np.ndarray | None = None

                for _, row in df.iterrows():
                    arr = np.asarray(row.loc[field], float)

                    x = np.asarray(row.loc["x"])
                    y = np.asarray(row.loc["y"])
                    dx = float(np.nanmedian(np.diff(np.unique(x))))
                    dy = float(np.nanmedian(np.diff(np.unique(y))))

                    PSD, kx, ky = _fft2_psd(arr, dx, dy)
                    k, E = _radial_spectrum(PSD, kx, ky)

                    mask = (k > 0) & (k <= 50) & np.isfinite(E)  # noqa: PLR2004
                    if np.count_nonzero(mask) < 2:  # noqa: PLR2004
                        continue

                    k = k[mask]
                    E = E[mask]

                    E_cum = np.cumsum(E)
                    E_cum /= E_cum[-1]

                    if k_ref is None:
                        k_ref = k
                    else:
                        E_cum = np.interp(k_ref, k, E_cum)

                    E_cum_all.append(E_cum)

                if not E_cum_all or k_ref is None:
                    continue

                E_cum_array = np.vstack(E_cum_all)

                med = np.nanmedian(E_cum_array, axis=0)
                q10 = np.nanpercentile(E_cum_array, 10, axis=0)
                q90 = np.nanpercentile(E_cum_array, 90, axis=0)

                color = dataset_colors[name]
                ax.plot(k_ref, med, lw=2.0, color=color)
                ax.fill_between(k_ref, q10, q90, color=color, alpha=0.25)

            for lvl in (0.90, 0.95, 0.99):
                ax.axhline(lvl, ls="--", lw=1.0, alpha=0.5)

            ax.set_xlim(0.0, 50.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(field)
            ax.set_xlabel(r"Wavenumber $k$ [$\frac{1}{m}$]")
            ax.set_ylabel(r"Cumulative energy $E_{\mathrm{cum}}$")
            ax.grid(True, which="both", ls=":")

        for ax in axes[n_items:]:
            ax.axis("off")

        legend_handles = [Line2D([], [], lw=4, color=dataset_colors[name], alpha=0.8) for name in active]

        fig.legend(
            legend_handles,
            active,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.98, 0.97),
        )

        fig.suptitle(
            f"Cumulative spectral energy (first {max_cases} cases)",
            fontsize=12,
        )
        fig.subplots_adjust(top=0.95, right=0.85, wspace=0.35, hspace=0.35)

        return fig

    ds = analysis.ui.components.ui_checkbox_datasets(dataset_names=names)

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )


# ============================================================================
# ANISOTROPIC CUMULATIVE SPECTRAL ENERGY (Ex, Ey)
# ============================================================================


def plot_spectral_cumulative_directional(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot cumulative directional spectral energy Ex(kx) and Ey(ky).

    For each field, cumulative energy is computed separately in x-
    and y-direction and aggregated across the first N cases using
    the median. This plot directly indicates anisotropic bandwidth
    requirements for n_modes_x vs n_modes_y.

    The x-axis is limited to k <= 50.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary of datasets to plot.

    Returns
    -------
    widgets.VBox
        Interactive widget for viewing cumulative directional spectral energy.

    """
    names = list(datasets.keys())

    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        dataset_selector: CheckboxGroup,
    ) -> Figure:
        active = analysis.eda.plots.case_statistics._selected_datasets(dataset_selector)  # noqa: SLF001 -- sibling plot helper

        sample_df = datasets[active[0]]
        field_keys = _infer_field_keys(sample_df)

        nrows = len(field_keys)
        ncols = 2

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(9.0 + 2.0, 3.4 * nrows),
            squeeze=False,
        )

        cmap = plt.get_cmap("tab10")
        dataset_colors = {name: cmap(i % 10) for i, name in enumerate(active)}

        for i, field in enumerate(field_keys):
            ax_x = axes[i, 0]
            ax_y = axes[i, 1]

            for name in active:
                df = datasets[name].iloc[:max_cases]

                Ex_all: list[np.ndarray] = []
                Ey_all: list[np.ndarray] = []
                kx_ref: np.ndarray | None = None
                ky_ref: np.ndarray | None = None

                for _, row in df.iterrows():
                    arr = np.asarray(row.loc[field], float)

                    x = np.asarray(row.loc["x"])
                    y = np.asarray(row.loc["y"])
                    dx = float(np.nanmedian(np.diff(np.unique(x))))
                    dy = float(np.nanmedian(np.diff(np.unique(y))))

                    PSD, kx, ky = _fft2_psd(arr, dx, dy)

                    kx_v, Ex = _directional_spectrum(PSD, kx, ky, axis="x")
                    ky_v, Ey = _directional_spectrum(PSD, kx, ky, axis="y")

                    mask_x = (kx_v > 0) & (kx_v <= 50) & np.isfinite(Ex)  # noqa: PLR2004
                    mask_y = (ky_v > 0) & (ky_v <= 50) & np.isfinite(Ey)  # noqa: PLR2004

                    if np.count_nonzero(mask_x) > 2:  # noqa: PLR2004
                        Ex = np.cumsum(Ex[mask_x])
                        Ex /= Ex[-1]
                        kx_v = kx_v[mask_x]
                        Ex = Ex if kx_ref is None else np.interp(kx_ref, kx_v, Ex)
                        kx_ref = kx_v if kx_ref is None else kx_ref
                        Ex_all.append(Ex)

                    if np.count_nonzero(mask_y) > 2:  # noqa: PLR2004
                        Ey = np.cumsum(Ey[mask_y])
                        Ey /= Ey[-1]
                        ky_v = ky_v[mask_y]
                        Ey = Ey if ky_ref is None else np.interp(ky_ref, ky_v, Ey)
                        ky_ref = ky_v if ky_ref is None else ky_ref
                        Ey_all.append(Ey)

                color = dataset_colors[name]

                if Ex_all and kx_ref is not None:
                    Ex_arr = np.vstack(Ex_all)
                    ax_x.plot(kx_ref, np.nanmedian(Ex_arr, axis=0), lw=2.0, color=color)
                    ax_x.fill_between(
                        kx_ref,
                        np.nanpercentile(Ex_arr, 10, axis=0),
                        np.nanpercentile(Ex_arr, 90, axis=0),
                        color=color,
                        alpha=0.25,
                    )

                if Ey_all and ky_ref is not None:
                    Ey_arr = np.vstack(Ey_all)
                    ax_y.plot(ky_ref, np.nanmedian(Ey_arr, axis=0), lw=2.0, color=color)
                    ax_y.fill_between(
                        ky_ref,
                        np.nanpercentile(Ey_arr, 10, axis=0),
                        np.nanpercentile(Ey_arr, 90, axis=0),
                        color=color,
                        alpha=0.25,
                    )

            for ax, title in zip((ax_x, ax_y), ("Ex(kx)", "Ey(ky)"), strict=True):
                for lvl in (0.90, 0.95, 0.99):
                    ax.axhline(lvl, ls="--", lw=1.0, alpha=0.5)

                ax.set_xlim(0.0, 50.0)
                ax.set_ylim(0.0, 1.02)
                ax.set_title(f"{field} - {title}")
                ax.set_xlabel(r"Wavenumber $k$ [$\frac{1}{m}$]")
                ax.set_ylabel("Cumulative energy")
                ax.grid(True, which="both", ls=":")

        legend_handles = [Line2D([], [], lw=4, color=dataset_colors[name], alpha=0.8) for name in active]

        fig.legend(
            legend_handles,
            active,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.98, 0.97),
        )

        fig.suptitle(
            f"Directional cumulative spectral energy (first {max_cases} cases)",
            fontsize=12,
        )
        fig.subplots_adjust(top=0.95, right=0.85, wspace=0.30, hspace=0.35)

        return fig

    ds = analysis.ui.components.ui_checkbox_datasets(dataset_names=names)

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )
