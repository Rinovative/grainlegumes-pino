"""
===============================================================================
evaluation_plot_spectral_fidelity.py
===============================================================================
Compare prediction and reference spectra in task output space.

Responsibilities:
  - Compute Hann-windowed radial spectra from physical coordinate spacing
  - Compare reference, prediction, and error energy for each learned field
  - Show transfer ratios with casewise uncertainty over bounded saved prefixes
  - Mask ratios where reference energy is too small for stable interpretation

Design principles:
  - The transform is architecture-independent and never hooks latent activations
  - Frequencies use inverse coordinate units and field powers retain squared units
  - Learned fields are analyzed separately; incompatible output units never mix

This module does NOT:
  - Parse artifact cases or silently invent physical coordinate spacing
  - Answer dataset-only spectral questions or inspect latent model activations
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 64
_MIN_SPECTRAL_SIZE = 2
_REFERENCE_ENERGY_FLOOR = 1e-12


def radial_power_spectrum(
    field: np.ndarray,
    *,
    dx: float,
    dy: float,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a Hann-windowed radial mean-power spectrum in physical frequency.

    Parameters
    ----------
    field : numpy.ndarray
        Finite two-dimensional physical output field.
    dx, dy : float
        Positive physical x/y grid spacing.
    n_bins : int | None, optional
        Number of equal-width radial-frequency bins; defaults to half the shorter
        grid dimension with a minimum of two.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Bin-center spatial frequencies and mean power; empty bins are retained as
        zeros to preserve alignment.

    Raises
    ------
    ValueError
        If field rank/shape/finiteness, spacing, or bin count is invalid.

    Notes
    -----
    The transform removes the field mean, applies a separable Hann window, and
    uses ``abs(fft2(...))**2 / field.size``. Frequencies have inverse-coordinate
    units and power retains squared field units.

    """
    values = np.asarray(field, dtype=float)
    if values.ndim != _MIN_SPECTRAL_SIZE or min(values.shape) < _MIN_SPECTRAL_SIZE or not np.isfinite(values).all():
        msg = "radial_power_spectrum requires one finite 2D field with both dimensions >= 2."
        raise ValueError(msg)
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        msg = "Spectral grid spacing must be finite and positive."
        raise ValueError(msg)
    bins = max(_MIN_SPECTRAL_SIZE, min(values.shape) // _MIN_SPECTRAL_SIZE) if n_bins is None else n_bins
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < _MIN_SPECTRAL_SIZE:
        msg = "n_bins must be an integer >= 2."
        raise ValueError(msg)

    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    transformed = np.fft.fft2((values - np.mean(values)) * window)
    power = np.abs(transformed) ** 2 / values.size
    kx = np.fft.fftfreq(values.shape[1], d=dx)
    ky = np.fft.fftfreq(values.shape[0], d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    radial = np.hypot(kx_grid, ky_grid).ravel()
    energy = power.ravel()
    edges = np.linspace(0.0, float(np.max(radial)), bins + 1)
    assignments = np.clip(np.digitize(radial, edges, right=False) - 1, 0, bins - 1)
    sums = np.bincount(assignments, weights=energy, minlength=bins)
    counts = np.bincount(assignments, minlength=bins)
    means = sums / np.maximum(counts, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q10, median, and q90 without accepting a wholly masked column."""
    valid_columns = np.isfinite(values).any(axis=0)
    q10 = np.full(values.shape[1], np.nan)
    median = np.full(values.shape[1], np.nan)
    q90 = np.full(values.shape[1], np.nan)
    if valid_columns.any():
        selected = values[:, valid_columns]
        q10[valid_columns] = np.nanquantile(selected, 0.1, axis=0)
        median[valid_columns] = np.nanquantile(selected, 0.5, axis=0)
        q90[valid_columns] = np.nanquantile(selected, 0.9, axis=0)
    return q10, median, q90


def _case_spectra(
    frame: pd.DataFrame,
    *,
    field_index: int,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Stack aligned reference, prediction, and error spectra for one output field.

    The bounded saved prefix is loaded through the shared case reader. Every case
    within a frame must retain identical shape, coordinate spacing, and derived
    frequency bins; no interpolation masks grid drift.
    """
    loaded = list(cases.iter_cases(frame, max_cases=min(max_cases, len(frame))))
    first = loaded[0]
    dx, dy = cases.grid_spacing(first)
    spectra: dict[str, list[np.ndarray]] = {"reference": [], "prediction": [], "error": []}
    k_reference: np.ndarray | None = None
    for case in loaded:
        case_dx, case_dy = cases.grid_spacing(case)
        if case.shape != first.shape or not np.isclose(case_dx, dx) or not np.isclose(case_dy, dy):
            msg = "Spectral aggregation requires a shared grid and spacing within each artifact."
            raise dataframe.ComparisonCompatibilityError(msg)
        for name, array in (
            ("reference", case.reference[field_index]),
            ("prediction", case.prediction[field_index]),
            ("error", case.error[field_index]),
        ):
            k_values, energy = radial_power_spectrum(array, dx=dx, dy=dy)
            if k_reference is None:
                k_reference = k_values
            elif not np.allclose(k_values, k_reference):
                msg = "Spectral bins changed within one artifact."
                raise dataframe.ComparisonCompatibilityError(msg)
            spectra[name].append(energy)
    if k_reference is None:
        msg = "Spectral aggregation requires at least one case."
        raise ValueError(msg)
    coordinate_unit = first.coordinate_units[0]
    return (
        k_reference,
        np.stack(spectra["reference"]),
        np.stack(spectra["prediction"]),
        np.stack(spectra["error"]),
        coordinate_unit,
    )


def _plot_band(
    axis: Axes,
    k_values: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: object,
    linestyle: str = "-",
) -> None:
    """Plot a finite positive median spectrum with q10-q90 case bands."""
    q10, median, q90 = _quantiles(values)
    valid = (k_values > 0.0) & np.isfinite(median) & (median > 0.0)
    if not valid.any():
        return
    axis.plot(k_values[valid], median[valid], color=color, linestyle=linestyle, label=label)
    band = valid & np.isfinite(q10) & np.isfinite(q90) & (q10 > 0.0)
    axis.fill_between(k_values[band], q10[band], q90[band], color=color, alpha=0.14)


def plot_spectral_fidelity(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Plot output-space spectra and masked prediction/reference transfer.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible frames with shared task outputs and physical units.
    max_cases : int, optional
        Positive bound on the saved ordered prefix aggregated per frame.

    Returns
    -------
    matplotlib.figure.Figure
        Per-field reference, prediction, and error power plus transfer-ratio
        panels with q10--q90 case bands and disclosed case counts.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If comparison identity, case grids, finite arrays, spacing, spectral bins,
        or the positive prefix contract fails.

    Notes
    -----
    Transfer is masked where reference power is at most ``1e-12`` of that case's
    maximum. Learned fields remain separate so power with incompatible units is
    never aggregated.

    """
    dataframe.validate_comparison(datasets)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    first_frame = next(iter(datasets.values()))
    fields = tuple(first_frame.attrs["output_fields"])
    units = dataframe.field_units(first_frame)
    figure, axes = plt.subplots(
        len(fields),
        2,
        figsize=(13, max(4.2 * len(fields), 4.5)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    for field_index, field in enumerate(fields):
        spectrum_axis, transfer_axis = axes[field_index]
        coordinate_units: set[str] = set()
        for dataset_index, (label, frame) in enumerate(datasets.items()):
            k_values, reference, prediction, error, coordinate_unit = _case_spectra(
                frame,
                field_index=field_index,
                max_cases=max_cases,
            )
            coordinate_units.add(coordinate_unit)
            color = colors(dataset_index % colors.N)
            disclosed = f"{label} ({dataframe.dataset_role(frame)}), n={min(max_cases, len(frame))}"
            _plot_band(spectrum_axis, k_values, reference, label=f"{disclosed} reference", color=color)
            _plot_band(
                spectrum_axis,
                k_values,
                prediction,
                label=f"{disclosed} prediction",
                color=color,
                linestyle="--",
            )
            _plot_band(
                spectrum_axis,
                k_values,
                error,
                label=f"{disclosed} error",
                color=color,
                linestyle=":",
            )

            reference_floor = _REFERENCE_ENERGY_FLOOR * np.max(reference, axis=1, keepdims=True)
            transfer = np.full_like(reference, np.nan)
            safe = reference > reference_floor
            np.divide(prediction, reference, out=transfer, where=safe)
            q10, median, q90 = _quantiles(transfer)
            valid = (k_values > 0.0) & np.isfinite(median) & (median > 0.0)
            transfer_axis.plot(k_values[valid], median[valid], color=color, label=disclosed)
            band = valid & np.isfinite(q10) & np.isfinite(q90) & (q10 > 0.0)
            transfer_axis.fill_between(k_values[band], q10[band], q90[band], color=color, alpha=0.18)

        coordinate_label = next(iter(coordinate_units)) if len(coordinate_units) == 1 else "coordinate-unit"
        spectrum_axis.set_xscale("log")
        spectrum_axis.set_yscale("log")
        spectrum_axis.set_title(f"{field}: radial spectral fidelity")
        spectrum_axis.set_xlabel(f"spatial frequency [1/{coordinate_label}]")
        spectrum_axis.set_ylabel(f"windowed radial mean power [{units[field]}^2]")
        spectrum_axis.grid(alpha=0.25, which="both")
        spectrum_axis.legend(fontsize=6)

        transfer_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        transfer_axis.set_xscale("log")
        transfer_axis.set_yscale("log")
        transfer_axis.set_title(f"{field}: prediction/reference transfer")
        transfer_axis.set_xlabel(f"spatial frequency [1/{coordinate_label}]")
        transfer_axis.set_ylabel("power ratio [1]")
        transfer_axis.grid(alpha=0.25, which="both")
        transfer_axis.legend(fontsize=7)
    figure.suptitle("Output-space spectral fidelity; bands=q10-q90 across cases; transfer masks reference power <= 1e-12 of each case maximum")
    return figure
