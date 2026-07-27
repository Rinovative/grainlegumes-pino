"""
===============================================================================
eda_plot_spectral_analysis.py
===============================================================================
Compare bounded dataset spectra without loading model activations or artifacts.

Responsibilities:
  - Compute Hann-windowed isotropic and x/y directional power spectra
  - Show cumulative bandwidth with casewise uncertainty over ordered prefixes
  - Resolve horizontal spectral composition as a function of physical height
  - Enforce shared task fields, units, and internally identical Cartesian grids

Design principles:
  - Frequencies use coordinate-derived units of inverse metres
  - Height-resolved power is normalized within each row before case aggregation
  - Dataset comparisons never combine physical fields with incompatible units

This module does NOT:
  - Materialize merged datasets or infer undeclared task fields
  - Compare model predictions with references or inspect model activations
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 64
_MIN_GRID_SIZE = 2


def _field_names(frame: pd.DataFrame) -> tuple[str, ...]:
    """
    Resolve declared numeric 2D non-coordinate fields in TaskSpec order.

    ``field_names`` and ``field_roles`` attrs are authoritative. Missing metadata,
    an empty frame contract, or the absence of any eligible array fails instead
    of inferring fields from arbitrary columns.
    """
    raw_declared = frame.attrs.get("field_names")
    if not isinstance(raw_declared, (list, tuple)) or not raw_declared or any(not isinstance(name, str) or not name for name in raw_declared):
        msg = "EDA spectral analysis requires task-aware field_names metadata."
        raise ValueError(msg)
    declared = tuple(raw_declared)
    roles = frame.attrs.get("field_roles")
    if not isinstance(roles, dict) or any(name not in roles for name in declared):
        msg = "EDA spectral analysis requires TaskSpec field_roles metadata."
        raise ValueError(msg)
    sample = frame.iloc[0]
    fields = tuple(
        name for name in declared if roles[name] != "coordinate" and name in frame.columns and np.asarray(sample[name]).ndim == _MIN_GRID_SIZE
    )
    if not fields:
        msg = "EDA spectral analysis found no declared numeric 2D fields."
        raise ValueError(msg)
    return fields


def _validate_datasets(datasets: dict[str, pd.DataFrame], *, max_cases: int) -> tuple[str, ...]:
    """
    Admit comparable EDA frames and a positive ordered-prefix bound.

    Every label/frame must be non-empty and expose identical task identity,
    declared spectral fields, and complete physical-unit mappings. Dataset
    fingerprint equality is not required because this view compares datasets.
    """
    if not datasets:
        msg = "At least one EDA dataset is required."
        raise ValueError(msg)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    reference_fields: tuple[str, ...] | None = None
    reference_units: object = None
    reference_task: object = None
    for label, frame in datasets.items():
        if not label or frame.empty:
            msg = "EDA datasets require non-empty labels and frames."
            raise ValueError(msg)
        fields = _field_names(frame)
        units = frame.attrs.get("field_units")
        task = frame.attrs.get("task_id")
        if not isinstance(units, dict) or any(field not in units for field in fields) or not isinstance(task, str):
            msg = "EDA spectra require task_id and field_units metadata."
            raise ValueError(msg)
        if reference_fields is None:
            reference_fields, reference_units, reference_task = fields, units, task
        elif fields != reference_fields or units != reference_units or task != reference_task:
            msg = "EDA spectral comparisons require identical task fields and physical units."
            raise ValueError(msg)
    if reference_fields is None:
        msg = "EDA dataset validation did not establish field metadata."
        raise RuntimeError(msg)
    return reference_fields


def _spacing(row: pd.Series) -> tuple[float, float, str]:
    """
    Derive positive median Cartesian x/y spacing from one EDA case.

    Explicit finite task inputs ``x`` and ``y`` must each contain at least two
    increasing unique coordinates. The maintained EDA coordinate contract uses
    metres, returned as the disclosed unit string.
    """
    if "x" not in row or "y" not in row:
        msg = "EDA spectral analysis requires explicit x and y task inputs."
        raise ValueError(msg)
    x_values = np.unique(np.asarray(row["x"], dtype=float))
    y_values = np.unique(np.asarray(row["y"], dtype=float))
    if x_values.size < _MIN_GRID_SIZE or y_values.size < _MIN_GRID_SIZE:
        msg = "EDA coordinate fields must contain at least two unique values per axis."
        raise ValueError(msg)
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    if dx <= 0.0 or dy <= 0.0 or not np.isfinite((dx, dy)).all():
        msg = "EDA coordinate spacing must be finite and positive."
        raise ValueError(msg)
    return dx, dy, "m"


def _power_grid(field: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean-centered Hann-windowed FFT power on physical frequency grids.

    The input must be one finite 2D field with both axes at least length two.
    Power is ``abs(fft2((field-mean)*window))**2 / field.size``; frequency grids
    use the caller's positive physical ``dx`` and ``dy``.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != _MIN_GRID_SIZE or min(values.shape) < _MIN_GRID_SIZE or not np.isfinite(values).all():
        msg = "EDA spectra require finite 2D fields."
        raise ValueError(msg)
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    transformed = np.fft.fft2((values - np.mean(values)) * window)
    power = np.abs(transformed) ** 2 / values.size
    kx = np.fft.fftfreq(values.shape[1], d=dx)
    ky = np.fft.fftfreq(values.shape[0], d=dy)
    return power, *np.meshgrid(kx, ky)


def _binned_mean(coordinate: np.ndarray, power: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce frequency-grid power into fixed equal-width coordinate bins.

    Bin centers cover zero through the observed maximum. Empty bins are retained
    with zero mean power so spectra from identical grids stay aligned.
    """
    values = np.asarray(coordinate, dtype=float).ravel()
    energy = np.asarray(power, dtype=float).ravel()
    edges = np.linspace(0.0, float(np.max(values)), bins + 1)
    assignments = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    sums = np.bincount(assignments, weights=energy, minlength=bins)
    counts = np.bincount(assignments, minlength=bins)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def _spectra(field: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return isotropic radial and absolute x/y directional mean-power spectra.

    All three reductions share ``max(2, min(shape)//2)`` equal-width bins and
    preserve physical inverse-coordinate frequencies.
    """
    power, kx, ky = _power_grid(field, dx=dx, dy=dy)
    bins = max(_MIN_GRID_SIZE, min(field.shape) // _MIN_GRID_SIZE)
    radial_k, radial = _binned_mean(np.hypot(kx, ky), power, bins=bins)
    x_k, x_energy = _binned_mean(np.abs(kx), power, bins=bins)
    y_k, y_energy = _binned_mean(np.abs(ky), power, bins=bins)
    return radial_k, radial, x_k, x_energy, y_k, y_energy


def _case_spectra(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Stack aligned isotropic and directional spectra for one declared field.

    The first bounded saved prefix is used without reranking. Frequency grids
    must remain identical within the frame; the result carries exact selected
    count and coordinate unit for disclosure.
    """
    selected = frame.iloc[: min(max_cases, len(frame))]
    collections: list[list[np.ndarray]] = [[], [], []]
    coordinates: list[np.ndarray | None] = [None, None, None]
    coordinate_unit = "m"
    for _index, row in selected.iterrows():
        dx, dy, coordinate_unit = _spacing(row)
        radial_k, radial, x_k, x_energy, y_k, y_energy = _spectra(np.asarray(row[field], dtype=float), dx=dx, dy=dy)
        for axis_index, (k_values, energy) in enumerate(((radial_k, radial), (x_k, x_energy), (y_k, y_energy))):
            reference = coordinates[axis_index]
            if reference is None:
                coordinates[axis_index] = k_values
            elif not np.allclose(reference, k_values):
                msg = "EDA spectral aggregation requires identical grids within each dataset."
                raise ValueError(msg)
            collections[axis_index].append(energy)
    radial_coordinate, x_coordinate, y_coordinate = coordinates
    if radial_coordinate is None or x_coordinate is None or y_coordinate is None:
        msg = "EDA spectral aggregation did not establish frequency grids."
        raise RuntimeError(msg)
    return (
        radial_coordinate,
        np.stack(collections[0]),
        x_coordinate,
        np.stack(collections[1]),
        y_coordinate,
        np.stack(collections[2]),
        len(selected),
        coordinate_unit,
    )


def _vertical_spectral_map(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Return median horizontal spectral fractions resolved by physical height.

    Each horizontal row is mean-centered and Hann-windowed before the real FFT.
    Power is normalized within that row, so the resulting map describes scale
    composition rather than mixing the physical units of different task fields.
    """
    selected = frame.iloc[: min(max_cases, len(frame))]
    spectra: list[np.ndarray] = []
    reference_frequency: np.ndarray | None = None
    reference_height: np.ndarray | None = None
    coordinate_unit = "m"
    for _index, row in selected.iterrows():
        values = np.asarray(row[field], dtype=float)
        x_grid = np.asarray(row["x"], dtype=float)
        y_grid = np.asarray(row["y"], dtype=float)
        if values.ndim != _MIN_GRID_SIZE or x_grid.shape != values.shape or y_grid.shape != values.shape:
            msg = "Vertical spectral evolution requires field, x, and y arrays on the same 2D grid."
            raise ValueError(msg)
        if not np.isfinite(values).all() or not np.isfinite(x_grid).all() or not np.isfinite(y_grid).all():
            msg = "Vertical spectral evolution requires finite field and coordinate arrays."
            raise ValueError(msg)
        x_values = np.median(x_grid, axis=0)
        y_values = np.median(y_grid, axis=1)
        if x_values.size < _MIN_GRID_SIZE or y_values.size < _MIN_GRID_SIZE:
            msg = "Vertical spectral evolution requires at least two grid points per axis."
            raise ValueError(msg)
        dx_values = np.diff(x_values)
        dy_values = np.diff(y_values)
        if np.any(dx_values <= 0.0) or np.any(dy_values <= 0.0):
            msg = "Vertical spectral evolution requires increasing rectilinear coordinates."
            raise ValueError(msg)
        dx = float(np.median(dx_values))
        frequency = np.fft.rfftfreq(values.shape[1], d=dx)[1:]
        centered = values - np.mean(values, axis=1, keepdims=True)
        transformed = np.fft.rfft(centered * np.hanning(values.shape[1]), axis=1)[:, 1:]
        power = np.abs(transformed) ** 2 / values.shape[1]
        totals = np.sum(power, axis=1, keepdims=True)
        fractions = np.divide(power, totals, out=np.zeros_like(power), where=totals > 0.0)
        if reference_frequency is None:
            reference_frequency = frequency
            reference_height = y_values
        elif not np.allclose(reference_frequency, frequency) or reference_height is None or not np.allclose(reference_height, y_values):
            msg = "Vertical spectral aggregation requires identical grids within each dataset."
            raise ValueError(msg)
        spectra.append(fractions)
    if reference_frequency is None or reference_height is None or not spectra:
        msg = "Vertical spectral aggregation did not establish a frequency-height grid."
        raise RuntimeError(msg)
    return reference_frequency, reference_height, np.median(np.stack(spectra), axis=0), len(selected), coordinate_unit


def _band(axis: Axes, k_values: np.ndarray, energy: np.ndarray, *, label: str, color: object) -> None:
    """Plot positive median power with q10-q90 case bands."""
    q10, median, q90 = np.quantile(energy, (0.1, 0.5, 0.9), axis=0)
    valid = (k_values > 0.0) & (median > 0.0)
    axis.plot(k_values[valid], median[valid], color=color, label=label)
    band = valid & (q10 > 0.0)
    axis.fill_between(k_values[band], q10[band], q90[band], color=color, alpha=0.18)


def _cumulative(energy: np.ndarray) -> np.ndarray:
    """Return casewise cumulative energy after omitting the DC bin."""
    positive = np.maximum(energy[:, 1:], 0.0)
    cumulative = np.cumsum(positive, axis=1)
    totals = cumulative[:, -1:]
    return np.divide(cumulative, totals, out=np.zeros_like(cumulative), where=totals > 0.0)


def plot_isotropic_spectral_summary(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Compare isotropic field spectra and cumulative bandwidth across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames with physical coordinates, fields, and units.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.

    Returns
    -------
    matplotlib.figure.Figure
        Per-field radial median power with q10--q90 case bands and cumulative
        non-DC energy fractions, each disclosing selected case counts.

    Raises
    ------
    ValueError, RuntimeError
        If task/field/unit contracts, finite Cartesian grids, or aligned spectral
        bins cannot establish a comparison.

    Notes
    -----
    Power retains squared field units; cumulative energy is dimensionless. No
    interpolation is used when within-frame frequency grids differ.

    """
    fields = _validate_datasets(datasets, max_cases=max_cases)
    figure, axes = plt.subplots(len(fields), 2, figsize=(13, 4 * len(fields)), squeeze=False, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    units = next(iter(datasets.values())).attrs["field_units"]
    for field_index, field in enumerate(fields):
        for dataset_index, (label, frame) in enumerate(datasets.items()):
            radial_k, radial, _x_k, _x_energy, _y_k, _y_energy, count, coordinate_unit = _case_spectra(frame, field, max_cases=max_cases)
            color = colors(dataset_index % colors.N)
            disclosed = f"{label}, n={count}"
            _band(axes[field_index, 0], radial_k, radial, label=disclosed, color=color)
            cumulative = _cumulative(radial)
            q10, median, q90 = np.quantile(cumulative, (0.1, 0.5, 0.9), axis=0)
            axes[field_index, 1].plot(radial_k[1:], median, color=color, label=disclosed)
            axes[field_index, 1].fill_between(radial_k[1:], q10, q90, color=color, alpha=0.18)
        axes[field_index, 0].set_xscale("log")
        axes[field_index, 0].set_yscale("log")
        axes[field_index, 0].set_title(f"{field}: isotropic radial power")
        axes[field_index, 0].set_xlabel(f"spatial frequency [1/{coordinate_unit}]")
        axes[field_index, 0].set_ylabel(f"windowed radial mean power [{units[field]}^2]")
        axes[field_index, 1].set_xscale("log")
        axes[field_index, 1].set_ylim(0.0, 1.02)
        axes[field_index, 1].set_title(f"{field}: cumulative isotropic energy")
        axes[field_index, 1].set_xlabel(f"spatial frequency [1/{coordinate_unit}]")
        axes[field_index, 1].set_ylabel("cumulative energy fraction [1]")
        for axis in axes[field_index]:
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=7)
    figure.suptitle(f"Dataset EDA: isotropic spectra and cumulative energy; first <= {max_cases} ordered cases")
    return figure


def _directional_axis(
    axis: Axes,
    *,
    datasets: dict[str, pd.DataFrame],
    field: str,
    direction: str,
    max_cases: int,
) -> None:
    """
    Plot directional median power and cumulative energy on explicit twin axes.

    The primary axis is log-log physical mean power with q10--q90 bands; the
    secondary linear axis shows median cumulative non-DC energy. Dataset labels
    disclose exact prefix counts.
    """
    colors = plt.get_cmap("tab10")
    cumulative_axis = axis.twinx()
    coordinate_unit = "m"
    for dataset_index, (label, frame) in enumerate(datasets.items()):
        _radial_k, _radial, x_k, x_energy, y_k, y_energy, count, coordinate_unit = _case_spectra(frame, field, max_cases=max_cases)
        k_values, energy = (x_k, x_energy) if direction == "x" else (y_k, y_energy)
        color = colors(dataset_index % colors.N)
        disclosed = f"{label}, n={count}"
        _band(axis, k_values, energy, label=f"{disclosed} power", color=color)
        cumulative = _cumulative(energy)
        median = np.quantile(cumulative, 0.5, axis=0)
        cumulative_axis.plot(k_values[1:], median, color=color, linestyle="--", label=f"{disclosed} cumulative")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(f"{direction}-direction spatial frequency [1/{coordinate_unit}]")
    axis.set_ylabel("directional mean power")
    cumulative_axis.set_ylim(0.0, 1.02)
    cumulative_axis.set_ylabel("cumulative energy fraction [1]")
    axis.grid(alpha=0.25, which="both")
    handles, labels = axis.get_legend_handles_labels()
    cumulative_handles, cumulative_labels = cumulative_axis.get_legend_handles_labels()
    axis.legend((*handles, *cumulative_handles), (*labels, *cumulative_labels), fontsize=6)


def plot_directional_spectral_summary(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Compare x- and y-direction spectral bandwidth across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames on internally identical Cartesian grids.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.

    Returns
    -------
    matplotlib.figure.Figure
        Separate x/y panels per non-coordinate field, with median directional
        power, q10--q90 bands, cumulative energy, and disclosed case counts.

    Raises
    ------
    ValueError, RuntimeError
        If task/field/unit metadata, finite grids, or aligned frequency bins fail.

    Notes
    -----
    Directional spectra remain separate so anisotropic bandwidth is visible; no
    scalar cross-direction score is calculated.

    """
    fields = _validate_datasets(datasets, max_cases=max_cases)
    figure, axes = plt.subplots(len(fields), 2, figsize=(13, 4 * len(fields)), squeeze=False, constrained_layout=True)
    for field_index, field in enumerate(fields):
        for axis_index, direction in enumerate(("x", "y")):
            _directional_axis(
                axes[field_index, axis_index],
                datasets=datasets,
                field=field,
                direction=direction,
                max_cases=max_cases,
            )
            axes[field_index, axis_index].set_title(f"{field}: {direction}-direction power and cumulative energy")
    figure.suptitle(f"Dataset EDA: directional spectral bandwidth; first <= {max_cases} ordered cases")
    return figure


def plot_vertical_spectral_evolution(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Plot horizontal spectral composition as a function of physical height.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames with identical declared fields/units and an
        internally shared increasing Cartesian grid.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.

    Returns
    -------
    matplotlib.figure.Figure
        One frequency-height map per dataset and non-coordinate field. Values are
        casewise median log10 row-normalized power fractions.

    Raises
    ------
    ValueError, RuntimeError
        If dataset metadata, fields, coordinates, finite values, or within-frame
        grids cannot establish the declared spectral comparison.

    Notes
    -----
    Each horizontal row is mean-centered and Hann-windowed before the real FFT.
    Omitting the DC bin and normalizing within a row makes the map dimensionless;
    it describes scale composition rather than absolute field power.

    """
    fields = _validate_datasets(datasets, max_cases=max_cases)
    figure, axes = plt.subplots(
        len(fields),
        len(datasets),
        figsize=(5.5 * len(datasets), 4.0 * len(fields)),
        squeeze=False,
        constrained_layout=True,
    )
    for field_index, field in enumerate(fields):
        for dataset_index, (label, frame) in enumerate(datasets.items()):
            frequency, height, fractions, count, coordinate_unit = _vertical_spectral_map(
                frame,
                field,
                max_cases=max_cases,
            )
            log_fraction = np.log10(np.maximum(fractions, np.finfo(float).tiny))
            axis = axes[field_index, dataset_index]
            image = axis.pcolormesh(frequency, height, log_fraction, shading="auto", cmap="magma")
            axis.set_xscale("log")
            axis.set_title(f"{label}: {field}, n={count}")
            axis.set_xlabel(f"horizontal spatial frequency [1/{coordinate_unit}]")
            axis.set_ylabel(f"height [{coordinate_unit}]")
            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("log10 row-normalized power fraction [1]")
    figure.suptitle(f"Dataset EDA: horizontal spectral evolution with height; first <= {max_cases} ordered cases")
    return figure
