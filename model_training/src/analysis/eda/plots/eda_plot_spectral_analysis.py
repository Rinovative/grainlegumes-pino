"""
===============================================================================
eda_plot_spectral_analysis.py
===============================================================================
Compare bounded dataset spectra without loading model activations or artifacts.

Responsibilities:
  - Compute Hann-windowed isotropic and x/y directional power spectra
  - Show cumulative bandwidth with casewise uncertainty over ordered prefixes
  - Resolve both complementary position-dependent spectral orientations
  - Enforce shared task contracts, stored representations, and Cartesian grids

Design principles:
  - Frequencies use coordinate-derived units of inverse metres
  - Flow-position-resolved power is normalized within each row before case aggregation
  - Dataset comparisons never combine incompatible stored field representations

This module does NOT:
  - Load model-training datasets or infer undeclared task fields
  - Compare model predictions with references or inspect model activations
===============================================================================
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.analysis.presentation.analysis_field_labels import field_label

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 100
_LEGEND_FONT_SIZE = 10
_MIN_GRID_SIZE = 2
_ROW_LABEL_X = -0.34
SpectralEvolutionOrientation = Literal["cross_stream_along_flow", "flow_across_cross_stream"]
_POWER_FRACTION_FLOOR = 1e-12
_CROSS_STREAM_ALONG_FLOW: Final[SpectralEvolutionOrientation] = "cross_stream_along_flow"
_FLOW_ACROSS_CROSS_STREAM: Final[SpectralEvolutionOrientation] = "flow_across_cross_stream"
_CASE_ID_PATTERN = re.compile(r"case_([0-9]{4,})")
_DIMENSIONLESS_REPRESENTATION_LABELS = {
    "dimensionless_log10_ratio_to_1_m2": "log10(k / 1 m²)",
    "dimensionless_cross_component_ratio_to_geometric_mean": "kij / sqrt(kii kjj)",
}
_PHYSICAL_REPRESENTATION_LABELS = {
    "identity": "physical values",
    "identity_before_train_normalization": "physical values",
    "derived_speed_magnitude": "derived magnitude |u|",
}
_PHYSICAL_VALUE_REPRESENTATIONS = frozenset(
    {
        "identity",
        "identity_before_train_normalization",
        "derived_speed_magnitude",
    }
)


def available_case_numbers(frame: pd.DataFrame) -> tuple[int, ...]:
    """Return authoritative dataset-local case numbers in DataFrame order."""
    if frame.empty:
        msg = "EDA case navigation requires a non-empty frame."
        raise ValueError(msg)
    numbers: list[int] = []
    for sample_id in frame.index:
        match = _CASE_ID_PATTERN.fullmatch(sample_id) if isinstance(sample_id, str) else None
        if match is None:
            msg = f"EDA case navigation requires canonical case IDs, got {sample_id!r}."
            raise ValueError(msg)
        numbers.append(int(match.group(1)))
    if len(numbers) != len(set(numbers)):
        msg = "EDA case navigation requires unique numeric case IDs."
        raise ValueError(msg)
    return tuple(numbers)


def _case_row(frame: pd.DataFrame, case_number: int) -> pd.Series:
    """Resolve one exact dataset-local case number without positional clamping."""
    if isinstance(case_number, bool) or not isinstance(case_number, int):
        msg = "case_number must be an integer."
        raise TypeError(msg)
    numbers = available_case_numbers(frame)
    for sample_id, number in zip(frame.index, numbers, strict=True):
        if number == case_number:
            row = frame.loc[sample_id]
            if not hasattr(row, "index"):
                msg = f"Case ID {sample_id!r} did not resolve to one row."
                raise RuntimeError(msg)
            return row
    msg = f"Requested case {case_number} is unavailable in this dataset. Choose a shared case number."
    raise ValueError(msg)


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


def _select_datasets(
    datasets: dict[str, pd.DataFrame],
    dataset_names: Sequence[str] | None,
) -> dict[str, pd.DataFrame]:
    """Select labelled frames in authoritative input order."""
    if not datasets:
        msg = "At least one EDA dataset is required."
        raise ValueError(msg)
    available = tuple(datasets)
    requested = available if dataset_names is None else tuple(dataset_names)
    if not requested:
        msg = "Select at least one dataset."
        raise ValueError(msg)
    if len(requested) != len(set(requested)):
        msg = "EDA dataset selection cannot contain duplicates."
        raise ValueError(msg)
    unknown = tuple(name for name in requested if name not in datasets)
    if unknown:
        msg = f"Unknown EDA dataset selection: {unknown!r}."
        raise ValueError(msg)
    requested_set = set(requested)
    return {name: datasets[name] for name in available if name in requested_set}


def _case_count_text(datasets: dict[str, pd.DataFrame], *, max_cases: int) -> str:
    """Report only actual displayed case counts; selection mechanics stay internal."""
    counts = {label: min(max_cases, len(frame)) for label, frame in datasets.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) == 1:
        return f"{next(iter(unique_counts))} cases"
    return ", ".join(f"{label}: {count} cases" for label, count in counts.items())


def _validate_datasets(datasets: dict[str, pd.DataFrame], *, max_cases: int) -> tuple[str, ...]:
    """
    Admit comparable EDA frames and a positive ordered-prefix bound.

    Every label/frame must be non-empty and expose an identical TaskSpec
    digest, declared spectral fields, physical-unit mappings, and stored-value
    representations. Dataset fingerprint equality is not required because this
    view compares datasets.
    """
    if not datasets:
        msg = "At least one EDA dataset is required."
        raise ValueError(msg)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    reference_fields: tuple[str, ...] | None = None
    reference_units: object = None
    reference_representations: object = None
    reference_task: object = None
    reference_contract: object = None
    for label, frame in datasets.items():
        if not label or frame.empty:
            msg = "EDA datasets require non-empty labels and frames."
            raise ValueError(msg)
        fields = _field_names(frame)
        units = frame.attrs.get("field_units")
        representations = frame.attrs.get("field_representations")
        task = frame.attrs.get("task_id")
        contract = frame.attrs.get("task_contract_digest")
        if (
            not isinstance(units, dict)
            or any(field not in units for field in fields)
            or not isinstance(representations, dict)
            or any(field not in representations for field in fields)
            or not isinstance(task, str)
            or not isinstance(contract, str)
        ):
            msg = "EDA spectra require task, contract, physical-unit, and stored-representation metadata."
            raise ValueError(msg)
        if reference_fields is None:
            reference_fields = fields
            reference_units = units
            reference_representations = representations
            reference_task = task
            reference_contract = contract
        elif (
            fields != reference_fields
            or units != reference_units
            or representations != reference_representations
            or task != reference_task
            or contract != reference_contract
        ):
            msg = "EDA spectral comparisons require one TaskSpec contract, field order, physical units, and stored representations."
            raise ValueError(msg)
    if reference_fields is None:
        msg = "EDA dataset validation did not establish field metadata."
        raise RuntimeError(msg)
    return reference_fields


def _stored_representation_label(frame: pd.DataFrame, field: str) -> str:
    """Return one explicit human-readable stored-value representation."""
    representations = frame.attrs.get("field_representations")
    raw_representation = representations.get(field) if isinstance(representations, dict) else None
    if not isinstance(raw_representation, str):
        msg = f"EDA field {field!r} has no stored-representation metadata."
        raise TypeError(msg)
    representation = raw_representation
    labels = {**_DIMENSIONLESS_REPRESENTATION_LABELS, **_PHYSICAL_REPRESENTATION_LABELS}
    try:
        return labels[representation]
    except KeyError as error:
        msg = f"EDA field {field!r} has unsupported stored representation {representation!r}."
        raise ValueError(msg) from error


def _field_display_label(field: str) -> str:
    """Return the canonical Matplotlib field label without changing internal keys."""
    scientific_labels = {
        "kxx": r"$\kappa_{xx}$",
        "kxy": r"$\kappa_{xy}$",
        "kyy": r"$\kappa_{yy}$",
        "eps": r"$\varepsilon$",
        "p_bc": r"$p_{\mathrm{bc}}$",
    }
    return scientific_labels.get(field, field_label(field, mathtext=True))


def _field_row_label(frame: pd.DataFrame, field: str, *, include_representation: bool = False) -> str:
    """Return one concise field-row label, optionally disclosing stored representation."""
    label = _field_display_label(field)
    if not include_representation:
        return label
    representation = _stored_representation_label(frame, field)
    if representation in {"physical values", "derived magnitude |u|"}:
        return label
    return f"{label}\n{representation}"


def _label_matrix_row(axis: Axes, label: str, *, field: str) -> None:
    """Place one field label clear of unit-bearing axes, emphasizing only magnitude."""
    annotation = axis.annotate(
        label,
        xy=(_ROW_LABEL_X, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        multialignment="center",
        fontsize=11,
        fontweight="bold" if field == "U" else "normal",
        annotation_clip=False,
    )
    annotation.set_gid("matrix-row-label")


def _matrix_with_legend_sidebar(nrows: int) -> tuple[Figure, np.ndarray, Axes]:
    """Build the same two-column matrix plus right legend rail used by plots 1-x."""
    figure = plt.figure(figsize=(13, 4 * nrows), constrained_layout=True)
    grid = figure.add_gridspec(nrows, 3, width_ratios=(1.0, 1.0, 0.35))
    axes = np.asarray([[figure.add_subplot(grid[row, column]) for column in range(2)] for row in range(nrows)], dtype=object)
    legend_axis = figure.add_subplot(grid[:, -1])
    legend_axis.axis("off")
    return figure, axes, legend_axis


def _add_sidebar_legend(axis: Axes, handles: Sequence[Artist], labels: Sequence[str]) -> None:
    """Add one shared legend at the top of a dedicated axis-free right rail."""
    axis.legend(
        handles,
        labels,
        loc="upper left",
        title="Dataset",
        fontsize=_LEGEND_FONT_SIZE,
    )


def _spectral_power_ylabel(frame: pd.DataFrame, field: str) -> str:
    """Label spectral power from stored values, not pre-transform physical units."""
    representations = frame.attrs["field_representations"]
    representation = representations[field]
    if representation in _DIMENSIONLESS_REPRESENTATION_LABELS:
        return f"Mean spectral power [-]\nStored: {_stored_representation_label(frame, field)}"
    if representation not in _PHYSICAL_VALUE_REPRESENTATIONS:
        msg = f"EDA field {field!r} has unsupported stored representation {representation!r}."
        raise ValueError(msg)
    units = frame.attrs["field_units"]
    return f"Mean spectral power [({units[field]})²]"


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
    Power is ``abs(fft2((field-mean)*window))**2 / field.size``. Frequency grids
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


def _row_spectra(
    row: pd.Series,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute all maintained spectra for one exact EDA case row."""
    dx, dy, coordinate_unit = _spacing(row)
    return (*_spectra(np.asarray(row[field], dtype=float), dx=dx, dy=dy), coordinate_unit)


def _case_spectra(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Stack aligned isotropic and directional spectra for one declared field.

    The first bounded saved prefix is used without reranking. Frequency grids
    must remain identical within the frame. The result carries exact selected
    count and coordinate unit for disclosure.
    """
    selected = frame.iloc[: min(max_cases, len(frame))]
    collections: list[list[np.ndarray]] = [[], [], []]
    coordinates: list[np.ndarray | None] = [None, None, None]
    coordinate_unit = "m"
    for _index, row in selected.iterrows():
        radial_k, radial, x_k, x_energy, y_k, y_energy, coordinate_unit = _row_spectra(row, field)
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


def _spectral_evolution_case_map(
    row: pd.Series,
    field: str,
    *,
    orientation: SpectralEvolutionOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return one correctly oriented position-resolved spectral-fraction map."""
    values = np.asarray(row[field], dtype=float)
    x_grid = np.asarray(row["x"], dtype=float)
    y_grid = np.asarray(row["y"], dtype=float)
    if values.ndim != _MIN_GRID_SIZE or x_grid.shape != values.shape or y_grid.shape != values.shape:
        msg = "Spectral evolution requires field, x, and y arrays on the same 2D grid."
        raise ValueError(msg)
    if not np.isfinite(values).all() or not np.isfinite(x_grid).all() or not np.isfinite(y_grid).all():
        msg = "Spectral evolution requires finite field and coordinate arrays."
        raise ValueError(msg)
    x_values = np.median(x_grid, axis=0)
    y_values = np.median(y_grid, axis=1)
    if x_values.size < _MIN_GRID_SIZE or y_values.size < _MIN_GRID_SIZE:
        msg = "Spectral evolution requires at least two grid points per axis."
        raise ValueError(msg)
    dx_values = np.diff(x_values)
    dy_values = np.diff(y_values)
    if np.any(dx_values <= 0.0) or np.any(dy_values <= 0.0):
        msg = "Spectral evolution requires increasing rectilinear coordinates."
        raise ValueError(msg)

    if orientation == _CROSS_STREAM_ALONG_FLOW:
        transform_axis = 1
        spacing = float(np.median(dx_values))
        position = y_values
        window = np.hanning(values.shape[1])[np.newaxis, :]
        centered = values - np.mean(values, axis=1, keepdims=True)
        transformed = np.fft.rfft(centered * window, axis=transform_axis)[:, 1:]
        power = np.abs(transformed) ** 2 / values.shape[1]
    elif orientation == _FLOW_ACROSS_CROSS_STREAM:
        transform_axis = 0
        spacing = float(np.median(dy_values))
        position = x_values
        window = np.hanning(values.shape[0])[:, np.newaxis]
        centered = values - np.mean(values, axis=0, keepdims=True)
        transformed = np.fft.rfft(centered * window, axis=transform_axis)[1:, :]
        power = (np.abs(transformed) ** 2 / values.shape[0]).T
    else:
        msg = f"Unsupported spectral-evolution orientation: {orientation!r}."
        raise ValueError(msg)

    frequency = np.fft.rfftfreq(values.shape[transform_axis], d=spacing)[1:]
    totals = np.sum(power, axis=1, keepdims=True)
    fractions = np.divide(power, totals, out=np.zeros_like(power), where=totals > 0.0)
    return frequency, position, fractions, "m"


def _vertical_case_map(
    row: pd.Series,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Retain the original cross-stream-along-flow reducer for internal callers."""
    return _spectral_evolution_case_map(row, field, orientation=_CROSS_STREAM_ALONG_FLOW)


def _spectral_evolution_map(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
    orientation: SpectralEvolutionOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Return median spectral fractions on one frequency/retained-position grid."""
    selected = frame.iloc[: min(max_cases, len(frame))]
    spectra: list[np.ndarray] = []
    reference_frequency: np.ndarray | None = None
    reference_position: np.ndarray | None = None
    coordinate_unit = "m"
    for _index, row in selected.iterrows():
        frequency, position, fractions, coordinate_unit = _spectral_evolution_case_map(
            row,
            field,
            orientation=orientation,
        )
        if reference_frequency is None:
            reference_frequency = frequency
            reference_position = position
        elif not np.allclose(reference_frequency, frequency) or reference_position is None or not np.allclose(reference_position, position):
            msg = "Spectral-evolution aggregation requires identical grids within each dataset."
            raise ValueError(msg)
        spectra.append(fractions)
    if reference_frequency is None or reference_position is None or not spectra:
        msg = "Spectral-evolution aggregation did not establish a frequency-position grid."
        raise RuntimeError(msg)
    return reference_frequency, reference_position, np.median(np.stack(spectra), axis=0), len(selected), coordinate_unit


def _vertical_spectral_map(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Retain the original aggregate reducer for internal callers."""
    return _spectral_evolution_map(
        frame,
        field,
        max_cases=max_cases,
        orientation=_CROSS_STREAM_ALONG_FLOW,
    )


def _orientation_labels(orientation: SpectralEvolutionOrientation, coordinate_unit: str) -> tuple[str, str, str]:
    """Return figure title and explicit frequency/position axis labels."""
    if orientation == _CROSS_STREAM_ALONG_FLOW:
        return (
            "Cross-stream spectral evolution along flow direction y",
            f"Cross-stream spatial frequency k_x [1/{coordinate_unit}]",
            f"Flow-direction position y [{coordinate_unit}]",
        )
    if orientation == _FLOW_ACROSS_CROSS_STREAM:
        return (
            "Flow-direction spectral evolution across cross-stream direction x",
            f"Flow-direction spatial frequency k_y [1/{coordinate_unit}]",
            f"Cross-stream position x [{coordinate_unit}]",
        )
    msg = f"Unsupported spectral-evolution orientation: {orientation!r}."
    raise ValueError(msg)


def _log_power_fraction(fractions: np.ndarray) -> np.ndarray:
    """Apply the documented fixed floor used by both evolution orientations."""
    return np.log10(np.maximum(fractions, _POWER_FRACTION_FLOOR))


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


def _set_log_frequency_axis(axis: Axes) -> None:
    """Use readable major log ticks without dense minor-frequency labels."""
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
    axis.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(float(value) for value in np.arange(2, 10) * 0.1), numticks=12))
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())


def plot_isotropic_spectral_summary(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    dataset_names: Sequence[str] | None = None,
) -> Figure:
    """
    Compare isotropic field spectra and cumulative bandwidth across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames with physical coordinates, fields, and units.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.
    dataset_names : collections.abc.Sequence[str] | None, optional
        Explicit dataset labels to compare. Omission preserves all input labels.

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
    Power retains squared stored-value units. Transformed dimensionless
    representations remain dimensionless. Cumulative energy is dimensionless,
    and no interpolation is used when within-frame frequency grids differ.

    """
    selected_datasets = _select_datasets(datasets, dataset_names)
    fields = _validate_datasets(selected_datasets, max_cases=max_cases)
    figure, axes, legend_axis = _matrix_with_legend_sidebar(len(fields))
    colors = plt.get_cmap("tab10")
    reference_frame = next(iter(selected_datasets.values()))
    for field_index, field in enumerate(fields):
        for dataset_index, (label, frame) in enumerate(selected_datasets.items()):
            radial_k, radial, _x_k, _x_energy, _y_k, _y_energy, _count, coordinate_unit = _case_spectra(frame, field, max_cases=max_cases)
            color = colors(dataset_index % colors.N)
            _band(axes[field_index, 0], radial_k, radial, label=label, color=color)
            cumulative = _cumulative(radial)
            q10, median, q90 = np.quantile(cumulative, (0.1, 0.5, 0.9), axis=0)
            axes[field_index, 1].plot(radial_k[1:], median, color=color, label=label)
            axes[field_index, 1].fill_between(radial_k[1:], q10, q90, color=color, alpha=0.18)
        _set_log_frequency_axis(axes[field_index, 0])
        axes[field_index, 0].set_yscale("log")
        if field_index == 0:
            axes[field_index, 0].set_title("Isotropic power")
            axes[field_index, 1].set_title("Cumulative energy")
        _label_matrix_row(axes[field_index, 0], _field_row_label(reference_frame, field), field=field)
        axes[field_index, 0].set_xlabel(f"Spatial frequency k [1/{coordinate_unit}]")
        axes[field_index, 0].set_ylabel(_spectral_power_ylabel(reference_frame, field))
        _set_log_frequency_axis(axes[field_index, 1])
        axes[field_index, 1].set_ylim(0.0, 1.02)
        axes[field_index, 1].set_xlabel(f"Spatial frequency k [1/{coordinate_unit}]")
        axes[field_index, 1].set_ylabel("Cumulative energy [-]")
        for axis in axes[field_index]:
            axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_sidebar_legend(legend_axis, handles, labels)
    case_text = _case_count_text(selected_datasets, max_cases=max_cases)
    figure.suptitle(f"Isotropic spectra — {case_text}")
    return figure


def plot_isotropic_spectral_case(
    *,
    datasets: dict[str, pd.DataFrame],
    case_number: int,
    dataset_names: Sequence[str] | None = None,
) -> Figure:
    """Compare one exact dataset-local case's isotropic spectra."""
    selected_datasets = _select_datasets(datasets, dataset_names)
    fields = _validate_datasets(selected_datasets, max_cases=1)
    figure, axes, legend_axis = _matrix_with_legend_sidebar(len(fields))
    colors = plt.get_cmap("tab10")
    reference_frame = next(iter(selected_datasets.values()))
    for field_index, field in enumerate(fields):
        coordinate_unit = "m"
        for dataset_index, (label, frame) in enumerate(selected_datasets.items()):
            row = _case_row(frame, case_number)
            radial_k, radial, _x_k, _x_energy, _y_k, _y_energy, coordinate_unit = _row_spectra(row, field)
            color = colors(dataset_index % colors.N)
            valid = (radial_k > 0.0) & (radial > 0.0)
            axes[field_index, 0].plot(radial_k[valid], radial[valid], color=color, label=label)
            cumulative = _cumulative(radial[np.newaxis, :])[0]
            axes[field_index, 1].plot(radial_k[1:], cumulative, color=color, label=label)
        _set_log_frequency_axis(axes[field_index, 0])
        axes[field_index, 0].set_yscale("log")
        if field_index == 0:
            axes[field_index, 0].set_title("Isotropic power")
            axes[field_index, 1].set_title("Cumulative energy")
        _label_matrix_row(axes[field_index, 0], _field_row_label(reference_frame, field), field=field)
        axes[field_index, 0].set_xlabel(f"Spatial frequency k [1/{coordinate_unit}]")
        axes[field_index, 0].set_ylabel(_spectral_power_ylabel(reference_frame, field))
        _set_log_frequency_axis(axes[field_index, 1])
        axes[field_index, 1].set_ylim(0.0, 1.02)
        axes[field_index, 1].set_xlabel(f"Spatial frequency k [1/{coordinate_unit}]")
        axes[field_index, 1].set_ylabel("Cumulative energy [-]")
        for axis in axes[field_index]:
            axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_sidebar_legend(legend_axis, handles, labels)
    figure.suptitle(f"Isotropic spectra — Case {case_number}")
    return figure


def _directional_axis(
    axis: Axes,
    *,
    datasets: dict[str, pd.DataFrame],
    field: str,
    direction: str,
    max_cases: int | None = None,
    case_number: int | None = None,
) -> tuple[list[Artist], list[str]]:
    """Plot one physical directional spectrum and return its shared legend entries."""
    if (max_cases is None) == (case_number is None):
        msg = "Directional spectra require exactly one aggregate bound or case number."
        raise ValueError(msg)
    colors = plt.get_cmap("tab10")
    cumulative_axis = axis.twinx()
    coordinate_unit = "m"
    has_positive_power = False
    for dataset_index, (label, frame) in enumerate(datasets.items()):
        if case_number is None:
            if max_cases is None:
                msg = "Aggregate directional scope lost its case bound."
                raise RuntimeError(msg)
            _radial_k, _radial, x_k, x_energy, y_k, y_energy, _count, coordinate_unit = _case_spectra(
                frame,
                field,
                max_cases=max_cases,
            )
        else:
            row = _case_row(frame, case_number)
            _radial_k, _radial, x_k, x_values, y_k, y_values, coordinate_unit = _row_spectra(row, field)
            x_energy = x_values[np.newaxis, :]
            y_energy = y_values[np.newaxis, :]
        if direction == "y":
            k_values, energy = y_k, y_energy
        elif direction == "x":
            k_values, energy = x_k, x_energy
        else:
            message = f"Unsupported spectral direction: {direction!r}."
            raise ValueError(message)
        color = colors(dataset_index % colors.N)
        if case_number is None:
            median_power = np.quantile(energy, 0.5, axis=0)
            has_positive_power = has_positive_power or bool(np.any((k_values > 0.0) & (median_power > 0.0)))
            _band(axis, k_values, energy, label=f"{label} power", color=color)
        else:
            power = energy[0]
            valid = (k_values > 0.0) & (power > 0.0)
            has_positive_power = has_positive_power or bool(np.any(valid))
            axis.plot(k_values[valid], power[valid], color=color, label=f"{label} power")
        cumulative = _cumulative(energy)
        curve = np.quantile(cumulative, 0.5, axis=0) if case_number is None else cumulative[0]
        cumulative_axis.plot(k_values[1:], curve, color=color, linestyle="--", label=f"{label} cumulative")
    _set_log_frequency_axis(axis)
    axis.set_yscale("log")
    if direction == "y":
        axis.set_xlabel(f"Flow-direction spatial frequency k_y [1/{coordinate_unit}]")
    else:
        axis.set_xlabel(f"Cross-stream spatial frequency k_x [1/{coordinate_unit}]")
    axis.set_ylabel(_spectral_power_ylabel(next(iter(datasets.values())), field))
    cumulative_axis.set_ylim(0.0, 1.02)
    cumulative_axis.set_ylabel("Cumulative energy [-]")
    if not has_positive_power:
        axis.text(0.5, 0.5, "No positive non-DC spectral power", transform=axis.transAxes, ha="center", va="center")
    axis.grid(alpha=0.25, which="both")
    handles, labels = axis.get_legend_handles_labels()
    cumulative_handles, cumulative_labels = cumulative_axis.get_legend_handles_labels()
    return [*handles, *cumulative_handles], [*labels, *cumulative_labels]


def plot_directional_spectral_summary(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    dataset_names: Sequence[str] | None = None,
) -> Figure:
    """
    Compare flow-direction y and cross-stream x spectra across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames on internally identical Cartesian grids.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.
    dataset_names : collections.abc.Sequence[str] | None, optional
        Explicit dataset labels to compare. Omission preserves all input labels.

    Returns
    -------
    matplotlib.figure.Figure
        Separate flow-y/cross-stream-x panels per field, with median directional
        power, q10--q90 bands, cumulative energy, and disclosed case counts.

    Raises
    ------
    ValueError, RuntimeError
        If task/field/unit metadata, finite grids, or aligned frequency bins fail.

    Notes
    -----
    Directional spectra remain separate so anisotropic bandwidth is visible. No
    scalar cross-direction score is calculated.

    """
    selected_datasets = _select_datasets(datasets, dataset_names)
    fields = _validate_datasets(selected_datasets, max_cases=max_cases)
    figure, axes, legend_axis = _matrix_with_legend_sidebar(len(fields))
    legend_handles: list[Artist] = []
    legend_labels: list[str] = []
    for field_index, field in enumerate(fields):
        for axis_index, direction in enumerate(("y", "x")):
            handles, labels = _directional_axis(
                axes[field_index, axis_index],
                datasets=selected_datasets,
                field=field,
                direction=direction,
                max_cases=max_cases,
            )
            if not legend_handles:
                legend_handles, legend_labels = handles, labels
        if field_index == 0:
            axes[field_index, 0].set_title("Flow direction y (k_y)")
            axes[field_index, 1].set_title("Cross-stream direction x (k_x)")
        _label_matrix_row(axes[field_index, 0], _field_row_label(next(iter(selected_datasets.values())), field), field=field)
    _add_sidebar_legend(legend_axis, legend_handles, legend_labels)
    case_text = _case_count_text(selected_datasets, max_cases=max_cases)
    figure.suptitle(f"Directional spectra — {case_text}")
    return figure


def plot_directional_spectral_case(
    *,
    datasets: dict[str, pd.DataFrame],
    case_number: int,
    dataset_names: Sequence[str] | None = None,
) -> Figure:
    """Compare one exact case along flow-y and cross-stream-x directions."""
    selected_datasets = _select_datasets(datasets, dataset_names)
    fields = _validate_datasets(selected_datasets, max_cases=1)
    for frame in selected_datasets.values():
        _case_row(frame, case_number)
    figure, axes, legend_axis = _matrix_with_legend_sidebar(len(fields))
    legend_handles: list[Artist] = []
    legend_labels: list[str] = []
    for field_index, field in enumerate(fields):
        for axis_index, direction in enumerate(("y", "x")):
            handles, labels = _directional_axis(
                axes[field_index, axis_index],
                datasets=selected_datasets,
                field=field,
                direction=direction,
                case_number=case_number,
            )
            if not legend_handles:
                legend_handles, legend_labels = handles, labels
        if field_index == 0:
            axes[field_index, 0].set_title("Flow direction y (k_y)")
            axes[field_index, 1].set_title("Cross-stream direction x (k_x)")
        _label_matrix_row(axes[field_index, 0], _field_row_label(next(iter(selected_datasets.values())), field), field=field)
    _add_sidebar_legend(legend_axis, legend_handles, legend_labels)
    figure.suptitle(f"Directional spectra — Case {case_number}")
    return figure


def plot_vertical_spectral_evolution(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    dataset_names: Sequence[str] | None = None,
    orientation: SpectralEvolutionOrientation = _CROSS_STREAM_ALONG_FLOW,
) -> Figure:
    """
    Plot either position-resolved physical spectral orientation.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames with identical declared fields/units and an
        internally shared increasing Cartesian grid.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.
    dataset_names : collections.abc.Sequence[str] | None, optional
        Explicit dataset labels to compare. Omission preserves all input labels.
    orientation : {"cross_stream_along_flow", "flow_across_cross_stream"}, optional
        Spatial transform and retained-position orientation. The default preserves
        cross-stream k_x spectra resolved along flow-direction position y.

    Returns
    -------
    matplotlib.figure.Figure
        One frequency/orthogonal-position map per dataset and field. Values are
        casewise median log10 row-normalized power fractions.

    Raises
    ------
    ValueError, RuntimeError
        If dataset metadata, fields, coordinates, finite values, or within-frame
        grids cannot establish the declared spectral comparison.

    Notes
    -----
    The selected transform axis is mean-centered and Hann-windowed before the
    real FFT. Omitting the DC bin and normalizing at each retained position makes
    the map dimensionless. Both orientations use a fixed 1e-12 numerical floor
    and log10 color range from -12 to 0, including honest zero-energy annotation.

    """
    selected_datasets = _select_datasets(datasets, dataset_names)
    fields = _validate_datasets(selected_datasets, max_cases=max_cases)
    case_text = _case_count_text(selected_datasets, max_cases=max_cases)
    figure, axes = plt.subplots(
        len(fields),
        len(selected_datasets),
        figsize=(5.5 * len(selected_datasets), 4.0 * len(fields)),
        squeeze=False,
        constrained_layout=True,
    )
    for field_index, field in enumerate(fields):
        for dataset_index, (label, frame) in enumerate(selected_datasets.items()):
            frequency, position, fractions, _count, coordinate_unit = _spectral_evolution_map(
                frame,
                field,
                max_cases=max_cases,
                orientation=orientation,
            )
            log_fraction = _log_power_fraction(fractions)
            axis = axes[field_index, dataset_index]
            image = axis.pcolormesh(frequency, position, log_fraction, shading="auto", cmap="magma", vmin=np.log10(_POWER_FRACTION_FLOOR), vmax=0.0)
            _set_log_frequency_axis(axis)
            if field_index == 0:
                axis.set_title(label)
            if dataset_index == 0:
                _label_matrix_row(axis, _field_row_label(frame, field, include_representation=True), field=field)
            title, xlabel, ylabel = _orientation_labels(orientation, coordinate_unit)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if not np.any(fractions > 0.0):
                axis.text(0.5, 0.5, "No positive non-DC spectral power", transform=axis.transAxes, ha="center", va="center", color="white")
            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("log10 row-normalized power fraction [-]")
    figure.suptitle(f"{title} — {case_text}")
    return figure


def plot_vertical_spectral_case(
    *,
    datasets: dict[str, pd.DataFrame],
    case_number: int,
    dataset_names: Sequence[str] | None = None,
    orientation: SpectralEvolutionOrientation = _CROSS_STREAM_ALONG_FLOW,
) -> Figure:
    """Compare either position-resolved spectral orientation for one case."""
    selected_datasets = _select_datasets(datasets, dataset_names)
    fields = _validate_datasets(selected_datasets, max_cases=1)
    figure, axes = plt.subplots(
        len(fields),
        len(selected_datasets),
        figsize=(5.5 * len(selected_datasets), 4.0 * len(fields)),
        squeeze=False,
        constrained_layout=True,
    )
    for field_index, field in enumerate(fields):
        for dataset_index, (label, frame) in enumerate(selected_datasets.items()):
            row = _case_row(frame, case_number)
            frequency, position, fractions, coordinate_unit = _spectral_evolution_case_map(row, field, orientation=orientation)
            log_fraction = _log_power_fraction(fractions)
            axis = axes[field_index, dataset_index]
            image = axis.pcolormesh(frequency, position, log_fraction, shading="auto", cmap="magma", vmin=np.log10(_POWER_FRACTION_FLOOR), vmax=0.0)
            _set_log_frequency_axis(axis)
            if field_index == 0:
                axis.set_title(label)
            if dataset_index == 0:
                _label_matrix_row(axis, _field_row_label(frame, field, include_representation=True), field=field)
            title, xlabel, ylabel = _orientation_labels(orientation, coordinate_unit)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if not np.any(fractions > 0.0):
                axis.text(0.5, 0.5, "No positive non-DC spectral power", transform=axis.transAxes, ha="center", va="center", color="white")
            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("log10 row-normalized power fraction [-]")
    figure.suptitle(f"{title} — Case {case_number}")
    return figure
