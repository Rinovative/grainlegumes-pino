"""
===============================================================================
evaluation_case.py
===============================================================================
Load one current-schema artifact case for every evaluation plot and viewer.

Responsibilities:
  - Parse explicit NPZ paths already validated by the artifact service
  - Enforce TaskSpec field order, units, case identity, and finite arrays
  - Expose coordinates, learned outputs, inputs, and steady residuals uniformly
  - Reject every undeclared array and duplicate parsing convention

Design principles:
  - One reader serves static figures, widgets, outliers, and W&B renderers
  - Learned output fields remain separate from optional derived artifact fields
  - Full-grid residual arrays remain distinct from cropped scalar diagnostics

This module does NOT:
  - Admit raw Parquet tables or validate cross-artifact provenance compatibility
  - Aggregate cases or answer plot-specific scientific questions
===============================================================================
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

_COMMON_ARRAY_NAMES = frozenset(
    {
        "case_index",
        "source_index",
        "split_local_index",
        "pred",
        "gt",
        "err",
        "artifact_fields",
        "artifact_units",
        "input_fields",
        "output_fields",
        "output_units",
        "x_raw",
        "y_raw",
        "meta",
    }
)
_STEADY_ARRAY_NAMES = frozenset(
    {
        "kappa_encoded",
        "kappa",
        "kappa_names",
        "p_bc",
        "coordinates",
        "Rx",
        "Ry",
        "div_u",
        "div_eps_u",
    }
)
_STEADY_RESIDUAL_NAMES = ("Rx", "Ry", "div_u", "div_eps_u")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """
    Store one validated task-aware artifact case.

    Parameters
    ----------
    case_index, source_index, split_local_index : int
        Exact identities cross-checked between Parquet and NPZ payloads.
    fields, units : tuple[str, ...]
        Learned TaskSpec output fields and physical units in declared order.
    prediction, reference, error : numpy.ndarray
        Finite learned-output arrays with shape ``(field, y, x)``; ``error`` is
        validated as prediction minus reference.
    coordinates : numpy.ndarray
        Coordinate grids with shape ``(2, y, x)``. Pixel coordinates are used
        only when the artifact has no explicit coordinate inputs.
    coordinate_units : tuple[str, str]
        Units for x/y coordinates, or ``index`` for a disclosed fallback grid.
    input_fields, input_units : tuple[str, ...]
        Exact stored input-field order and provenance-declared units.
    inputs : numpy.ndarray
        Finite raw input tensor with shape ``(input_field, y, x)``.
    metadata : Mapping[str, Any]
        Source metadata without authoritative identity duplication.
    residuals : Mapping[str, numpy.ndarray]
        Full-grid steady residual arrays, empty for generic tasks.
    permeability : numpy.ndarray | None
        Physical permeability components in square metres when available.
    permeability_names : tuple[str, ...]
        Names aligned with the permeability channel axis.
    pressure_boundary : numpy.ndarray | None
        Optional pressure boundary field with shape ``(1, y, x)``.

    Notes
    -----
    The dataclass is frozen and slotted, but contained arrays and mappings remain
    caller-visible objects and are not made deeply immutable.

    """

    case_index: int
    source_index: int
    split_local_index: int
    fields: tuple[str, ...]
    units: tuple[str, ...]
    prediction: np.ndarray
    reference: np.ndarray
    error: np.ndarray
    coordinates: np.ndarray
    coordinate_units: tuple[str, str]
    input_fields: tuple[str, ...]
    input_units: tuple[str, ...]
    inputs: np.ndarray
    metadata: Mapping[str, Any]
    residuals: Mapping[str, np.ndarray]
    permeability: np.ndarray | None
    permeability_names: tuple[str, ...]
    pressure_boundary: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shared spatial grid shape."""
        return int(self.prediction.shape[-2]), int(self.prediction.shape[-1])

    @property
    def field_units(self) -> dict[str, str]:
        """Return learned field units in declared order."""
        return dict(zip(self.fields, self.units, strict=True))

    @property
    def input_field_units(self) -> dict[str, str]:
        """Return input field units in declared order."""
        return dict(zip(self.input_fields, self.input_units, strict=True))


def _string_vector(
    value: np.ndarray,
    *,
    label: str,
    require_unique: bool = True,
) -> tuple[str, ...]:
    """
    Decode one rank-one non-empty NPZ string vector.

    Every element must stringify to non-empty text; field-name vectors additionally
    reject duplicates while unit vectors may repeat physical units.
    """
    if value.ndim != 1:
        msg = f"{label} must be a rank-one string vector."
        raise ValueError(msg)
    result = tuple(str(item) for item in value.tolist())
    if not result or any(not item for item in result):
        msg = f"{label} must contain non-empty strings."
        raise ValueError(msg)
    if require_unique and len(result) != len(set(result)):
        msg = f"{label} must contain unique strings."
        raise ValueError(msg)
    return result


def _scalar_integer(value: np.ndarray, *, label: str) -> int:
    """Return one exact non-boolean integer scalar."""
    raw = value.item() if value.ndim == 0 else None
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        msg = f"{label} must be an integer scalar."
        raise TypeError(msg)
    return int(raw)


def _finite_array(
    value: np.ndarray,
    *,
    label: str,
    rank: int,
) -> np.ndarray:
    """
    Normalize one NPZ array to real finite float values at an exact rank.

    Non-numeric, complex, rank-mismatched, NaN, or infinite payloads fail before
    shape relationships are checked by the calling case parser.
    """
    array = np.asarray(value)
    if array.ndim != rank or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        msg = f"{label} must be a real numeric rank-{rank} array."
        raise TypeError(msg)
    result = np.asarray(array, dtype=float)
    if not np.isfinite(result).all():
        msg = f"{label} contains non-finite values."
        raise ValueError(msg)
    return result


def _provenance(frame: pd.DataFrame) -> Mapping[str, Any] | None:
    """Return optional validated provenance carried by the DataFrame."""
    value = frame.attrs.get("artifact_provenance")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = "Evaluation DataFrame artifact_provenance attr must be a mapping."
        raise TypeError(msg)
    return value


def _coordinate_units(frame: pd.DataFrame) -> tuple[str, str]:
    """Return x/y units from artifact provenance when available."""
    provenance = _provenance(frame)
    evaluator = provenance.get("evaluator") if provenance is not None else None
    raw_units = evaluator.get("input_units") if isinstance(evaluator, Mapping) else None
    if isinstance(raw_units, Mapping):
        return str(raw_units.get("x", "index")), str(raw_units.get("y", "index"))
    return "index", "index"


def _coordinates(
    payload: Mapping[str, np.ndarray],
    *,
    input_fields: tuple[str, ...],
    inputs: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    """
    Resolve validated physical coordinates or a disclosed pixel-index fallback.

    An explicit ``coordinates`` payload wins. Otherwise declared ``x``/``y``
    input channels may supply the grid; only when neither exact shape is
    available is a unit-index mesh synthesized and flagged as non-physical.
    """
    if "coordinates" in payload:
        coordinates = _finite_array(payload["coordinates"], label="coordinates", rank=3)
        if coordinates.shape != (2, *shape):
            msg = f"coordinates shape {coordinates.shape} does not match (2, {shape[0]}, {shape[1]})."
            raise ValueError(msg)
        return coordinates, True
    if "x" in input_fields and "y" in input_fields:
        coordinates = inputs[[input_fields.index("x"), input_fields.index("y")]]
        if coordinates.shape == (2, *shape):
            return coordinates, True
    y_index, x_index = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        indexing="ij",
    )
    return np.stack((x_index, y_index), axis=0), False


def _metadata(value: np.ndarray) -> Mapping[str, Any]:
    """
    Parse one scalar JSON metadata object without duplicated authoritative identity.

    Non-scalar/non-text payloads, invalid JSON, non-object results, and duplicated
    case/source/split identity all fail before constructing :class:`EvaluationCase`.
    """
    if value.ndim != 0:
        msg = "Artifact case meta must be a scalar JSON string."
        raise ValueError(msg)
    raw = value.item()
    if not isinstance(raw, str):
        msg = "Artifact case meta must be a JSON string."
        raise TypeError(msg)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        msg = "Artifact case meta is invalid JSON."
        raise ValueError(msg) from error
    if not isinstance(parsed, dict):
        msg = "Artifact case meta must decode to an object."
        raise TypeError(msg)
    if {"case_index", "source_index", "split_local_index"}.intersection(parsed):
        msg = "Artifact case meta duplicates authoritative identity."
        raise ValueError(msg)
    return parsed


def load_case(frame: pd.DataFrame, row_position: int) -> EvaluationCase:  # noqa: C901, PLR0912, PLR0915
    """
    Load one NPZ case by deterministic DataFrame row position.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluation frame built through :mod:`evaluation_dataframe` with complete
        field, unit, identity, and provenance attrs.
    row_position : int
        Zero-based position in exact saved artifact membership order.

    Returns
    -------
    EvaluationCase
        Validated learned fields, coordinates, inputs, metadata, and optional
        steady-flow diagnostics.

    Raises
    ------
    IndexError
        If ``row_position`` lies outside the frame membership.
    FileNotFoundError
        If the explicitly stored NPZ path is absent.
    KeyError, TypeError, ValueError
        If required arrays, identities, fields, units, shapes, finite values, or
        numerical relationships violate the current case contract.

    Notes
    -----
    The function never searches for a replacement NPZ or infers identity from a
    filename. Parquet and NPZ scalar identities must agree exactly.

    """
    if isinstance(row_position, bool) or not isinstance(row_position, Integral):
        msg = "row_position must be an integer."
        raise TypeError(msg)
    position = int(row_position)
    if not 0 <= position < len(frame):
        msg = f"row_position {position} is outside [0, {len(frame)})."
        raise IndexError(msg)
    row = frame.iloc[position]
    raw_path = row.loc["npz_path"]
    if not isinstance(raw_path, str) or not raw_path:
        msg = "Evaluation row npz_path must be a non-empty string."
        raise TypeError(msg)
    path = Path(raw_path)
    if not path.is_file():
        msg = f"Artifact case NPZ does not exist: {path}"
        raise FileNotFoundError(msg)

    task_id = frame.attrs.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        msg = "Evaluation DataFrame must carry one validated task_id attribute."
        raise ValueError(msg)
    expected_names = set(_COMMON_ARRAY_NAMES)
    if task_id == "steady_flow":
        expected_names.update(_STEADY_ARRAY_NAMES)
    with np.load(path, allow_pickle=False) as stored:
        names = set(stored.files)
        missing = sorted(expected_names.difference(names))
        unexpected = sorted(names.difference(expected_names))
        if missing or unexpected:
            msg = f"Artifact case schema mismatch: missing={missing}, unexpected={unexpected}."
            raise ValueError(msg)
        payload = {name: np.asarray(stored[name]) for name in stored.files}

    identities = {
        "case_index": _scalar_integer(payload["case_index"], label="case_index"),
        "source_index": _scalar_integer(payload["source_index"], label="source_index"),
        "split_local_index": _scalar_integer(payload["split_local_index"], label="split_local_index"),
    }
    for name, value in identities.items():
        if int(row.loc[name]) != value:
            msg = f"Artifact NPZ {name} does not match its Parquet row."
            raise ValueError(msg)

    output_fields = _string_vector(payload["output_fields"], label="output_fields")
    output_units = _string_vector(payload["output_units"], label="output_units", require_unique=False)
    frame_fields = tuple(frame.attrs.get("output_fields", output_fields))
    frame_units = tuple(frame.attrs.get("output_units", output_units))
    if output_fields != frame_fields or output_units != frame_units:
        msg = "Artifact NPZ output fields/units contradict the evaluation DataFrame."
        raise ValueError(msg)

    artifact_fields = _string_vector(payload["artifact_fields"], label="artifact_fields")
    artifact_units = _string_vector(payload["artifact_units"], label="artifact_units", require_unique=False)
    if artifact_fields[: len(output_fields)] != output_fields or artifact_units[: len(output_units)] != output_units:
        msg = "Artifact derived field order does not begin with the learned TaskSpec outputs."
        raise ValueError(msg)

    prediction_all = _finite_array(payload["pred"], label="pred", rank=3)
    reference_all = _finite_array(payload["gt"], label="gt", rank=3)
    error_all = _finite_array(payload["err"], label="err", rank=3)
    if prediction_all.shape != reference_all.shape or prediction_all.shape != error_all.shape:
        msg = "Artifact pred, gt, and err arrays must have identical shapes."
        raise ValueError(msg)
    if prediction_all.shape[0] != len(artifact_fields):
        msg = "Artifact field names do not match pred/gt/err channel count."
        raise ValueError(msg)
    if not np.allclose(error_all, prediction_all - reference_all, rtol=1e-6, atol=1e-9):
        msg = "Artifact err array does not equal pred minus gt."
        raise ValueError(msg)

    learned_count = len(output_fields)
    prediction = prediction_all[:learned_count]
    reference = reference_all[:learned_count]
    error = error_all[:learned_count]
    inputs = _finite_array(payload["x_raw"], label="x_raw", rank=3)
    input_fields = _string_vector(payload["input_fields"], label="input_fields")
    if inputs.shape[0] != len(input_fields):
        msg = "Artifact input_fields do not match x_raw channel count."
        raise ValueError(msg)
    frame_input_fields = tuple(frame.attrs.get("input_fields", ()))
    frame_input_units = tuple(frame.attrs.get("input_units", ()))
    if input_fields != frame_input_fields or len(frame_input_units) != len(input_fields):
        msg = "Artifact NPZ input fields contradict complete evaluator provenance."
        raise ValueError(msg)
    coordinates, physical_coordinates = _coordinates(
        payload,
        input_fields=input_fields,
        inputs=inputs,
        shape=(prediction.shape[-2], prediction.shape[-1]),
    )
    coordinate_units = _coordinate_units(frame) if physical_coordinates else ("index", "index")

    residual_presence = tuple(name in payload for name in _STEADY_RESIDUAL_NAMES)
    if any(residual_presence) and not all(residual_presence):
        msg = "Steady residual arrays must contain Rx, Ry, div_u, and div_eps_u together."
        raise ValueError(msg)
    residuals = {name: _finite_array(payload[name], label=name, rank=2) for name in _STEADY_RESIDUAL_NAMES if name in payload}
    if any(array.shape != prediction.shape[-2:] for array in residuals.values()):
        msg = "Steady residual arrays must use the full learned-output grid."
        raise ValueError(msg)

    permeability = None
    permeability_names: tuple[str, ...] = ()
    if "kappa" in payload or "kappa_names" in payload:
        if "kappa" not in payload or "kappa_names" not in payload:
            msg = "Steady permeability requires kappa and kappa_names together."
            raise ValueError(msg)
        permeability = _finite_array(payload["kappa"], label="kappa", rank=3)
        permeability_names = _string_vector(payload["kappa_names"], label="kappa_names")
        if permeability.shape[0] != len(permeability_names):
            msg = "kappa_names do not match the permeability channel count."
            raise ValueError(msg)

    pressure_boundary = None
    if "p_bc" in payload:
        pressure_boundary = _finite_array(payload["p_bc"], label="p_bc", rank=3)
        if pressure_boundary.shape[0] != 1 or pressure_boundary.shape[-2:] != prediction.shape[-2:]:
            msg = "p_bc must have shape (1, y, x) on the learned-output grid."
            raise ValueError(msg)

    return EvaluationCase(
        case_index=identities["case_index"],
        source_index=identities["source_index"],
        split_local_index=identities["split_local_index"],
        fields=output_fields,
        units=output_units,
        prediction=prediction,
        reference=reference,
        error=error,
        coordinates=coordinates,
        coordinate_units=coordinate_units,
        input_fields=input_fields,
        input_units=frame_input_units,
        inputs=inputs,
        metadata=_metadata(payload["meta"]),
        residuals=residuals,
        permeability=permeability,
        permeability_names=permeability_names,
        pressure_boundary=pressure_boundary,
    )


def iter_cases(
    frame: pd.DataFrame,
    *,
    max_cases: int | None = None,
) -> Iterator[EvaluationCase]:
    """
    Yield validated cases in exact saved artifact membership order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current evaluation frame containing admitted NPZ paths and case identity.
    max_cases : int | None, optional
        Positive prefix bound. ``None`` yields every saved row.

    Yields
    ------
    EvaluationCase
        One schema-validated physical case loaded through :func:`load_case`.

    Raises
    ------
    ValueError
        If ``max_cases`` is non-positive, boolean, or non-integral.
    IndexError, FileNotFoundError, KeyError, TypeError
        Propagated from case loading when persisted membership cannot be read.

    Notes
    -----
    Prefixing never sorts or reranks the frame, so displayed order remains
    identical to persisted artifact membership.

    """
    if max_cases is None:
        count = len(frame)
    else:
        if isinstance(max_cases, bool) or not isinstance(max_cases, Integral) or int(max_cases) <= 0:
            msg = "max_cases must be a positive integer or None."
            raise ValueError(msg)
        count = min(len(frame), int(max_cases))
    for position in range(count):
        yield load_case(frame, position)


def grid_extent(case: EvaluationCase) -> tuple[float, float, float, float]:
    """
    Return the physical image extent in x-min, x-max, y-min, y-max order.

    The values come from the case coordinate fields and therefore retain their
    declared coordinate units; no pixel-center or half-cell padding is added.
    """
    x_values, y_values = case.coordinates
    return (
        float(np.nanmin(x_values)),
        float(np.nanmax(x_values)),
        float(np.nanmin(y_values)),
        float(np.nanmax(y_values)),
    )


def grid_spacing(case: EvaluationCase) -> tuple[float, float]:
    """
    Return positive median physical x/y grid spacing for spectral frequencies.

    Differences are taken along the declared Cartesian x and y axes. Degenerate
    or non-finite coordinates fail instead of silently falling back to pixels;
    the only unit-spacing fallback applies to an axis with no differences.
    """
    x_values, y_values = case.coordinates
    dx_values = np.abs(np.diff(x_values, axis=1)).ravel()
    dy_values = np.abs(np.diff(y_values, axis=0)).ravel()
    dx = float(np.nanmedian(dx_values)) if dx_values.size else 1.0
    dy = float(np.nanmedian(dy_values)) if dy_values.size else 1.0
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        msg = "Artifact coordinates do not define positive finite grid spacing."
        raise ValueError(msg)
    return dx, dy
