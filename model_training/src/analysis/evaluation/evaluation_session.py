"""
===============================================================================
evaluation_session.py
===============================================================================
Reuse validated evaluation cases and plot-independent numerical reductions.

Responsibilities:
  - Bind caches to canonical run, role, membership, and output-manifest identity
  - Reuse immutable selected cases under explicit entry and byte limits
  - Compute all-field full-membership and bounded-prefix reductions once
  - Expose operation and retained-byte counters for reproducible benchmarks
  - Release every cache and temporary projection at an explicit session boundary

Design principles:
  - Display labels and figure presentation never participate in numerical keys
  - ID, OOD, run, checkpoint, normalizer, and payload identities remain separate
  - Exact spatial quantiles use bounded session-owned scratch, not case retention
  - Local-error curves use one versioned bounded empirical-quantile definition
  - Direct plot calls receive a scoped ephemeral session without global state

This module does NOT:
  - Load run directories, generate artifacts, run inference, or repair payloads
  - Cache Matplotlib figures, widgets, display labels, or export state
  - Keep an unbounded process-global case or numerical cache
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, cast

import numpy as np

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pandas as pd

    from src.analysis.evaluation.evaluation_artifact_loader import LoadedRunArtifacts

SESSION_ATTR = "_evaluation_session"
SESSION_KEY_ATTR = "_evaluation_artifact_key"
LOCAL_EMPIRICAL_QUANTILE_DEFINITION = "local_relative_error_empirical_quantiles_v1"
LOCAL_EMPIRICAL_QUANTILE_VERSION = 1
_MINIMUM_MAGNITUDE_COMPONENTS = 2
LOCAL_EMPIRICAL_QUANTILE_GRID_SIZE = 1025
LOCAL_EMPIRICAL_QUANTILE_PROBABILITIES = np.linspace(
    0.0,
    1.0,
    LOCAL_EMPIRICAL_QUANTILE_GRID_SIZE,
    dtype=np.float64,
)
LOCAL_EMPIRICAL_QUANTILE_PROBABILITIES.setflags(write=False)

__all__ = [
    "LOCAL_EMPIRICAL_QUANTILE_DEFINITION",
    "LOCAL_EMPIRICAL_QUANTILE_GRID_SIZE",
    "LOCAL_EMPIRICAL_QUANTILE_PROBABILITIES",
    "LOCAL_EMPIRICAL_QUANTILE_VERSION",
    "ArtifactKey",
    "BinnedMeanSummary",
    "BinnedMedianSummary",
    "EmpiricalQuantileSummary",
    "EvaluationArtifactChangedError",
    "EvaluationSession",
    "EvaluationSessionClosedError",
    "FullEvaluationSummary",
    "FullMagnitudeSummary",
    "GridDescriptor",
    "PrefixEvaluationSummary",
    "PrefixMagnitudeSummary",
    "SpectralFieldSummary",
    "bound_session",
    "radial_power_spectrum",
    "scoped_session",
]

_LOCAL_DENOMINATOR_FLOOR = 1e-12
_TARGET_BIN_COUNT = 12
_BOUNDARY_BIN_COUNT = 10
_BOUNDARY_REGION_EDGES = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0)
_REFERENCE_ENERGY_FLOOR = 1e-12
_MIN_SPECTRAL_SIZE = 2


class _SessionLimits(TypedDict, total=False):
    """Type supported keyword-only resource limits for one session."""

    max_case_entries: int
    max_case_bytes: int
    max_aggregate_bytes: int
    max_spill_bytes: int
    working_bytes: int


class EvaluationSessionClosedError(RuntimeError):
    """
    Signal use after an evaluation session has released its resources.

    Notes
    -----
    Closed sessions never reopen or reconstruct their caches. Callers must create
    a new load-only session for another explicit evaluation lifetime.

    """


class EvaluationArtifactChangedError(RuntimeError):
    """
    Signal that bound artifact identity or payload witnesses changed.

    Notes
    -----
    The affected artifact is invalidated before this error is raised. Cached data
    is never reused after a frame token or payload witness changes.

    """


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """
    Store the exact cache identity for one run-owned artifact role.

    Parameters
    ----------
    root : str
        Resolved artifact root used only for role-local payload access.
    run_name, task_id : str
        Canonical persisted run and task identifiers.
    config_digest, checkpoint_digest, normalizer_digest : str
        Exact completed-run scientific identity digests.
    split_role : str
        Explicit ``eval`` or ``ood`` persisted role.
    dataset_fingerprint, membership_digest : str
        Dataset and effective ordered-membership identities.
    manifest_digest : str
        Canonical digest of the persisted artifact output manifest.
    frame_digest : str
        Digest of live identity attrs and ordered row membership.

    """

    root: str
    run_name: str
    task_id: str
    config_digest: str
    checkpoint_digest: str
    normalizer_digest: str
    split_role: str
    dataset_fingerprint: str
    membership_digest: str
    manifest_digest: str
    frame_digest: str

    @property
    def token(self) -> str:
        """
        Return a stable compact token for live frame identity checks.

        Returns
        -------
        str
            Canonical SHA-256 digest of every artifact-key field.

        """
        return _digest(self.__dict__ if hasattr(self, "__dict__") else {field.name: getattr(self, field.name) for field in fields(self)})


@dataclass(frozen=True, slots=True)
class GridDescriptor:
    """
    Store immutable grid and learned-field semantics retained by summaries.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Physical coordinate grids with shape ``(2, y, x)``.
    coordinate_units : tuple[str, str]
        Declared x and y coordinate units.
    fields, units : tuple[str, ...]
        Learned output fields and aligned physical units.

    """

    coordinates: np.ndarray
    coordinate_units: tuple[str, str]
    fields: tuple[str, ...]
    units: tuple[str, ...]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """
        Return the rectangular physical extent of the retained grid.

        Returns
        -------
        tuple[float, float, float, float]
            Minimum and maximum x followed by minimum and maximum y.

        """
        x_values, y_values = self.coordinates
        return (
            float(np.min(x_values)),
            float(np.max(x_values)),
            float(np.min(y_values)),
            float(np.max(y_values)),
        )


@dataclass(frozen=True, slots=True)
class FullMagnitudeSummary:
    """Store complete spatial reductions for one TaskSpec vector-group magnitude."""

    component_fields: tuple[str, ...]
    unit: str
    reference_mean: np.ndarray
    prediction_mean: np.ndarray
    signed_error_mean: np.ndarray
    absolute_error_mean: np.ndarray
    local_relative_error_mean: np.ndarray
    signed_error_std: np.ndarray
    absolute_error_q90: np.ndarray
    case_reference_means: np.ndarray
    case_prediction_means: np.ndarray


@dataclass(frozen=True, slots=True)
class FullEvaluationSummary:
    """
    Store complete-membership reductions shared by evaluation plots.

    Parameters
    ----------
    sample_count : int
        Exact persisted membership count.
    grid : GridDescriptor
        Shared immutable grid and field semantics.
    reference_mean, prediction_mean : numpy.ndarray | None
        Pointwise complete-membership learned-field means.
    signed_error_mean, absolute_error_mean : numpy.ndarray | None
        Pointwise signed and absolute error means.
    signed_error_std, absolute_error_q90 : numpy.ndarray | None
        Pointwise signed-error standard deviation and exact q90 absolute error.
    case_reference_means, case_prediction_means : numpy.ndarray
        Per-case spatial means with shape ``(case, field)``.
    momentum_mean, div_velocity_mean, div_eps_velocity_mean : numpy.ndarray | None
        Full-grid mean steady residual magnitudes.
    pressure_declared, pressure_predicted, pressure_absolute_error : numpy.ndarray | None
        Per-case pressure-drop evidence in pascals.
    spatial_error, residual_error, pressure_error : str | None, optional
        Deferred compatibility failures for question-specific admission.

    """

    sample_count: int
    grid: GridDescriptor
    reference_mean: np.ndarray | None
    prediction_mean: np.ndarray | None
    signed_error_mean: np.ndarray | None
    absolute_error_mean: np.ndarray | None
    local_relative_error_mean: np.ndarray | None
    signed_error_std: np.ndarray | None
    absolute_error_q90: np.ndarray | None
    case_reference_means: np.ndarray
    case_prediction_means: np.ndarray
    momentum_mean: np.ndarray | None
    div_velocity_mean: np.ndarray | None
    div_eps_velocity_mean: np.ndarray | None
    pressure_declared: np.ndarray | None
    pressure_predicted: np.ndarray | None
    pressure_absolute_error: np.ndarray | None
    magnitudes: Mapping[str, FullMagnitudeSummary]
    spatial_error: str | None = None
    residual_error: str | None = None
    pressure_error: str | None = None

    def require_spatial(self) -> FullEvaluationSummary:
        """
        Require and return the complete spatial reduction family.

        Returns
        -------
        FullEvaluationSummary
            This immutable summary when spatial reductions are available.

        Raises
        ------
        ComparisonCompatibilityError
            If case grids were incompatible during the shared scan.

        """
        if self.spatial_error is not None:
            raise dataframe.ComparisonCompatibilityError(self.spatial_error)
        return self

    def require_residuals(self) -> FullEvaluationSummary:
        """
        Require and return the complete residual reduction family.

        Returns
        -------
        FullEvaluationSummary
            This immutable summary when residual reductions are available.

        Raises
        ------
        ComparisonCompatibilityError
            If required residual arrays or shared-grid semantics were absent.

        """
        if self.residual_error is not None:
            raise dataframe.ComparisonCompatibilityError(self.residual_error)
        return self

    def require_pressure(self) -> FullEvaluationSummary:
        """
        Require and return the complete pressure-drop reduction family.

        Returns
        -------
        FullEvaluationSummary
            This immutable summary when pressure-drop evidence is available.

        Raises
        ------
        ComparisonCompatibilityError
            If pressure output or boundary-condition evidence was unavailable.

        """
        if self.pressure_error is not None:
            raise dataframe.ComparisonCompatibilityError(self.pressure_error)
        return self


@dataclass(frozen=True, slots=True)
class EmpiricalQuantileSummary:
    """
    Store bounded deterministic empirical quantiles and exact disclosure evidence.

    Parameters
    ----------
    definition : str
        Stable named numerical definition.
    version : int
        Definition schema version.
    probabilities, quantiles : numpy.ndarray
        Fixed probability grid and aligned linear empirical quantiles.
    source_point_count : int
        Exact number of grid points represented.
    exact_minimum, exact_maximum : float
        Exact empirical endpoints retained independently for disclosure.

    """

    definition: str
    version: int
    probabilities: np.ndarray
    quantiles: np.ndarray
    source_point_count: int
    exact_minimum: float
    exact_maximum: float


@dataclass(frozen=True, slots=True)
class BinnedMedianSummary:
    """
    Store exact non-empty bin centers, medians, and point counts.

    Parameters
    ----------
    centers, medians : numpy.ndarray
        Aligned explanatory-axis centers and response medians.
    counts : numpy.ndarray
        Exact selected grid-point count for every retained bin.

    """

    centers: np.ndarray
    medians: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True, slots=True)
class BinnedMeanSummary:
    """
    Store fixed bin centers, arithmetic means, and selected point counts.

    Parameters
    ----------
    centers, means : numpy.ndarray
        Aligned explanatory-axis centers and response arithmetic means.
    counts : numpy.ndarray
        Exact selected grid-point count for every fixed bin.

    """

    centers: np.ndarray
    means: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True, slots=True)
class SpectralFieldSummary:
    """
    Store per-case aligned spectra for plot-time uncertainty bands.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Shared physical radial-frequency bin centers.
    reference, prediction, error : numpy.ndarray
        Per-case radial mean-power arrays with shape ``(case, frequency)``.

    """

    frequencies: np.ndarray
    reference: np.ndarray
    prediction: np.ndarray
    error: np.ndarray


@dataclass(frozen=True, slots=True)
class PrefixMagnitudeSummary:
    """Store bounded reductions for one TaskSpec vector-group magnitude."""

    component_fields: tuple[str, ...]
    unit: str
    local_relative_error: EmpiricalQuantileSummary
    target_magnitude_error: BinnedMedianSummary
    boundary_distance_error: BinnedMedianSummary
    boundary_region_error: BinnedMeanSummary
    spectrum: SpectralFieldSummary


@dataclass(frozen=True, slots=True)
class PrefixEvaluationSummary:
    """
    Store all bounded-prefix numerical evidence shared by prefix plots.

    Parameters
    ----------
    case_count : int
        Exact saved-prefix case count.
    fields, units : tuple[str, ...]
        Learned output fields and aligned physical units.
    coordinate_unit : str
        Shared physical coordinate unit used by distance and frequency labels.
    local_relative_error : collections.abc.Mapping
        Field-keyed bounded empirical-quantile summaries.
    target_magnitude_error, boundary_distance_error : collections.abc.Mapping
        Field-keyed exact binned-median summaries.
    boundary_region_error : collections.abc.Mapping
        Field-keyed left/right normalized-distance band means and point counts.
    spectra : collections.abc.Mapping
        Field-keyed per-case radial spectra.

    """

    case_count: int
    fields: tuple[str, ...]
    units: tuple[str, ...]
    coordinate_unit: str
    local_relative_error: Mapping[str, EmpiricalQuantileSummary]
    target_magnitude_error: Mapping[str, BinnedMedianSummary]
    boundary_distance_error: Mapping[str, BinnedMedianSummary]
    boundary_region_error: Mapping[str, BinnedMeanSummary]
    spectra: Mapping[str, SpectralFieldSummary]
    magnitudes: Mapping[str, PrefixMagnitudeSummary]


@dataclass(slots=True)
class _Binding:
    """
    Store one live frame binding and its immutable identity witnesses.

    Parameters
    ----------
    frame : pandas.DataFrame
        Exact live frame owned by the binding.
    key : ArtifactKey
        Canonical run-role and payload cache identity.
    frame_token : str
        Digest of identity-bearing frame attrs and ordered row membership.
    witnesses : tuple[tuple[str, int, int], ...]
        Path, byte-size, and nanosecond-mtime payload witnesses.
    invalid : bool, optional
        Whether a detected change permanently rejected this binding.

    """

    frame: pd.DataFrame
    key: ArtifactKey
    frame_token: str
    witnesses: tuple[tuple[str, int, int], ...]
    invalid: bool = False


@dataclass(slots=True)
class _CacheEntry:
    """
    Store one retained cache value and its accounted array bytes.

    Parameters
    ----------
    value : Any
        Immutable case or numerical summary.
    size : int
        Unique retained ndarray bytes charged to its cache.

    """

    value: Any
    size: int


def _jsonable(value: Any) -> Any:
    """
    Normalize nested scientific metadata for deterministic JSON hashing.

    Parameters
    ----------
    value : Any
        Nested metadata, path, NumPy value, or scalar to normalize.

    Returns
    -------
    Any
        JSON-compatible content with mapping keys in deterministic order.

    Notes
    -----
    Unsupported leaf objects are represented explicitly so hashing remains total
    without granting them control over JSON serialization.

    """
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest(value: Any) -> str:
    """
    Return a canonical SHA-256 digest for JSON-normalized input.

    Parameters
    ----------
    value : Any
        Scientific identity content accepted by :func:`_jsonable`.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    """
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frame_identity_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Return identity-bearing attrs and exact ordered row membership.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluation artifact frame with canonical membership columns.

    Returns
    -------
    dict[str, Any]
        Hash-ready identity attrs plus ordered case and payload membership.

    """
    membership = [
        (int(case_index), int(source_index), int(split_local_index), str(npz_path))
        for case_index, source_index, split_local_index, npz_path in zip(
            frame["case_index"].tolist(),
            frame["source_index"].tolist(),
            frame["split_local_index"].tolist(),
            frame["npz_path"].tolist(),
            strict=True,
        )
    ]
    return {
        "task_id": frame.attrs.get("task_id"),
        "output_fields": frame.attrs.get("output_fields"),
        "output_units": frame.attrs.get("output_units"),
        "input_fields": frame.attrs.get("input_fields"),
        "input_units": frame.attrs.get("input_units"),
        "artifact_root": frame.attrs.get("artifact_root"),
        "artifact_provenance": frame.attrs.get("artifact_provenance"),
        "membership": membership,
    }


def _artifact_key(frame: pd.DataFrame) -> ArtifactKey:
    """
    Build a run, role, membership, and output-manifest-bound key.

    Parameters
    ----------
    frame : pandas.DataFrame
        Provenance-complete evaluation artifact frame.

    Returns
    -------
    ArtifactKey
        Exact scientific and payload identity used by session caches.

    Raises
    ------
    TypeError, ComparisonCompatibilityError
        If required completed-run provenance mappings are absent or malformed.

    """
    provenance = dataframe.require_complete_provenance(frame)
    run = provenance.get("run")
    dataset = provenance.get("dataset")
    selection = provenance.get("selection")
    if not isinstance(run, Mapping) or not isinstance(dataset, Mapping) or not isinstance(selection, Mapping):
        msg = "Evaluation session requires run, dataset, and selection provenance mappings."
        raise TypeError(msg)
    root = str(Path(str(frame.attrs.get("artifact_root", ""))).resolve())
    frame_digest = _digest(_frame_identity_payload(frame))
    outputs = provenance.get("outputs", {})
    manifest_digest = _digest(outputs)
    return ArtifactKey(
        root=root,
        run_name=str(run.get("name", "")),
        task_id=str(run["task"]),
        config_digest=str(run.get("effective_config_digest", "")),
        checkpoint_digest=str(run.get("best_checkpoint_sha256", "")),
        normalizer_digest=str(run.get("normalizer_sha256", "")),
        split_role=str(provenance.get("split_role", "")),
        dataset_fingerprint=str(dataset.get("fingerprint", "")),
        membership_digest=str(selection["effective_ordered_source_indices_sha256"]),
        manifest_digest=manifest_digest,
        frame_digest=frame_digest,
    )


def _payload_paths(frame: pd.DataFrame) -> tuple[Path, ...]:
    """
    Return every live file whose stat witness protects cached case data.

    Parameters
    ----------
    frame : pandas.DataFrame
        Bound evaluation artifact frame.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted resolved NPZ, provenance, parquet, and manifest-declared paths.

    """
    paths = {Path(str(value)).resolve() for value in frame["npz_path"].tolist()}
    root_value = frame.attrs.get("artifact_root")
    provenance = frame.attrs.get("artifact_provenance")
    if root_value is not None:
        root = Path(str(root_value)).resolve()
        provenance_path = root / "artifact_provenance.json"
        if provenance_path.is_file():
            paths.add(provenance_path)
        if isinstance(provenance, Mapping):
            outputs = provenance.get("outputs")
            if isinstance(outputs, Mapping):
                parquet = outputs.get("parquet")
                npz_entries = outputs.get("npz")
                entries = []
                if isinstance(parquet, Mapping):
                    entries.append(parquet)
                if isinstance(npz_entries, list):
                    entries.extend(item for item in npz_entries if isinstance(item, Mapping))
                for entry in entries:
                    relative = entry.get("path")
                    if isinstance(relative, str) and relative:
                        paths.add((root / relative).resolve())
    return tuple(sorted(paths))


def _witnesses(frame: pd.DataFrame) -> tuple[tuple[str, int, int], ...]:
    """
    Snapshot stat witnesses for every identity-bearing payload file.

    Parameters
    ----------
    frame : pandas.DataFrame
        Bound evaluation artifact frame.

    Returns
    -------
    tuple[tuple[str, int, int], ...]
        Resolved path, byte size, and nanosecond modification time per file.

    Raises
    ------
    OSError
        If a declared payload cannot be inspected.

    """
    result = []
    for path in _payload_paths(frame):
        stat = path.stat()
        result.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(result)


def _array_root(array: np.ndarray) -> np.ndarray:
    """
    Return the deepest NumPy array base retained by one view.

    Parameters
    ----------
    array : numpy.ndarray
        Array whose owned or shared storage must be identified.

    Returns
    -------
    numpy.ndarray
        Deepest ndarray in the base chain for unique byte accounting.

    """
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def _case_nbytes(case: cases.EvaluationCase) -> int:
    """
    Count unique NumPy array storage retained by one validated case.

    Parameters
    ----------
    case : EvaluationCase
        Fully validated case whose arrays may share backing storage.

    Returns
    -------
    int
        Sum of unique root-array bytes retained by the case.

    """
    arrays = [case.prediction, case.reference, case.error, case.coordinates, case.inputs]
    arrays.extend(case.residuals.values())
    if case.permeability is not None:
        arrays.append(case.permeability)
    if case.pressure_boundary is not None:
        arrays.append(case.pressure_boundary)
    roots = {_array_root(np.asarray(array)).__array_interface__["data"][0]: _array_root(np.asarray(array)) for array in arrays}
    return int(sum(array.nbytes for array in roots.values()))


def _readonly(array: np.ndarray) -> np.ndarray:
    """
    Return an immutable NumPy view without changing numeric values.

    Parameters
    ----------
    array : numpy.ndarray
        Array to expose through cached immutable state.

    Returns
    -------
    numpy.ndarray
        NumPy representation with its write flag disabled.

    Notes
    -----
    The helper does not copy. Callers that need ownership must copy first.

    """
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _freeze_metadata(value: Any) -> Any:
    """
    Recursively freeze JSON-like selected-case metadata.

    Parameters
    ----------
    value : Any
        Nested mapping, sequence, NumPy array, or scalar metadata value.

    Returns
    -------
    Any
        Mapping proxies, tuples, read-only arrays, or unchanged scalar leaves.

    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return value


def _freeze_case(case: cases.EvaluationCase) -> cases.EvaluationCase:
    """
    Make every caller-visible array and mapping in a cached case immutable.

    Parameters
    ----------
    case : EvaluationCase
        Fully validated case created by the uncached parser.

    Returns
    -------
    EvaluationCase
        Case with read-only arrays and mapping-proxy metadata containers.

    """
    arrays = (
        case.prediction,
        case.reference,
        case.error,
        case.coordinates,
        case.inputs,
    )
    for array in arrays:
        _readonly(array)
    for array in case.residuals.values():
        _readonly(array)
    if case.permeability is not None:
        _readonly(case.permeability)
    if case.pressure_boundary is not None:
        _readonly(case.pressure_boundary)
    return replace(
        case,
        metadata=_freeze_metadata(case.metadata),
        residuals=MappingProxyType(dict(case.residuals)),
    )


def _value_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """
    Count unique retained NumPy bytes in a nested immutable summary.

    Parameters
    ----------
    value : Any
        Dataclass, mapping, sequence, array, or scalar summary value.
    seen : set[int] | None, optional
        Shared identity set used while recursively avoiding double counting.

    Returns
    -------
    int
        Unique root-array bytes reachable from ``value``.

    """
    visited = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        root = _array_root(value)
        root_id = id(root)
        if root_id in visited:
            return 0
        visited.add(root_id)
        return int(root.nbytes)
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if is_dataclass(value):
        return sum(_value_nbytes(getattr(value, field.name), visited) for field in fields(value))
    if isinstance(value, Mapping):
        return sum(_value_nbytes(item, visited) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_value_nbytes(item, visited) for item in value)
    return 0


def radial_power_spectrum(
    field: np.ndarray,
    *,
    dx: float,
    dy: float,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the maintained Hann-windowed radial mean-power spectrum.

    Parameters
    ----------
    field : numpy.ndarray
        Finite two-dimensional physical output field.
    dx, dy : float
        Positive physical x and y grid spacing.
    n_bins : int | None, optional
        Equal-width radial-frequency bin count. The default uses half the shorter
        grid dimension with a minimum of two.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Bin-center spatial frequencies and aligned radial mean power.

    Raises
    ------
    ValueError
        If field rank, shape, finiteness, spacing, or bin count is invalid.

    Notes
    -----
    The transform removes the field mean, applies a separable Hann window, and
    preserves squared physical field units.

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
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def _boundary_distance(case: cases.EvaluationCase) -> np.ndarray:
    """
    Compute physical distance to the closest rectangular extent edge.

    Parameters
    ----------
    case : EvaluationCase
        Validated case carrying physical x and y coordinate grids.

    Returns
    -------
    numpy.ndarray
        Non-negative distance map aligned with the case grid.

    """
    x_values, y_values = case.coordinates
    return np.minimum.reduce(
        (
            x_values - np.min(x_values),
            np.max(x_values) - x_values,
            y_values - np.min(y_values),
            np.max(y_values) - y_values,
        )
    )


def _horizontal_boundary_fraction(case: cases.EvaluationCase) -> np.ndarray:
    """Return normalized distance from the closest left/right x-extent edge."""
    x_values = case.coordinates[0]
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    half_width = 0.5 * (x_max - x_min)
    if not np.isfinite(half_width) or half_width <= 0.0:
        msg = "Horizontal boundary diagnostics require a positive physical x extent."
        raise ValueError(msg)
    return np.minimum(x_values - x_min, x_max - x_values) / half_width


def _boundary_region_mean(
    fractions: np.ndarray,
    absolute_error: np.ndarray,
) -> BinnedMeanSummary:
    """Reduce fixed historical left/right distance bands without another case scan."""
    fraction_values = np.asarray(fractions, dtype=float)
    error_values = np.asarray(absolute_error, dtype=float)
    if fraction_values.shape != error_values.shape or not np.isfinite(fraction_values).all() or not np.isfinite(error_values).all():
        msg = "Boundary-region fractions and errors must be aligned finite arrays."
        raise ValueError(msg)
    edges = np.asarray(_BOUNDARY_REGION_EDGES, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=np.int64)
    for index, (lower, upper) in enumerate(pairwise(edges)):
        selected = (
            (fraction_values >= lower) & (fraction_values <= upper)
            if index == len(centers) - 1
            else (fraction_values >= lower) & (fraction_values < upper)
        )
        counts[index] = int(np.count_nonzero(selected))
        if counts[index]:
            means[index] = float(np.mean(error_values[selected]))
    return BinnedMeanSummary(
        centers=_readonly(centers),
        means=_readonly(means),
        counts=_readonly(counts),
    )


def _pressure_group_field(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """Resolve the task-owned scalar pressure group without field literals."""
    try:
        return dataframe.single_output_group_field(frame, group_id="pressure"), None
    except dataframe.ComparisonCompatibilityError as error:
        return None, str(error)


def _pressure_drop(
    case: cases.EvaluationCase,
    *,
    pressure_field: str,
) -> tuple[float, float, float]:
    """
    Compute declared and predicted pressure drop with absolute mismatch.

    Parameters
    ----------
    case : EvaluationCase
        Validated case with pressure output and pressure-boundary evidence.
    pressure_field : str
        Sole field declared by the task-owned scalar pressure group.

    Returns
    -------
    tuple[float, float, float]
        Declared drop, predicted drop, and absolute mismatch in pascals.

    Raises
    ------
    ComparisonCompatibilityError
        If pressure output or boundary evidence is absent.
    ValueError
        If inlet and outlet masks cannot be resolved from physical coordinates.

    """
    if pressure_field not in case.fields or case.pressure_boundary is None:
        msg = "Pressure-drop analysis requires the task-owned scalar pressure group and boundary evidence."
        raise dataframe.ComparisonCompatibilityError(msg)
    pressure = case.prediction[case.fields.index(pressure_field)]
    boundary = case.pressure_boundary[0]
    y_values = case.coordinates[1]
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    spacing = cases.grid_spacing(case)[1]
    inlet = np.isclose(y_values, y_min, rtol=0.0, atol=0.51 * spacing)
    outlet = np.isclose(y_values, y_max, rtol=0.0, atol=0.51 * spacing)
    if not inlet.any() or not outlet.any():
        msg = "Pressure-drop analysis could not resolve inlet/outlet coordinate masks."
        raise ValueError(msg)
    declared = float(np.mean(boundary[inlet]) - np.mean(boundary[outlet]))
    predicted = float(np.mean(pressure[inlet]) - np.mean(pressure[outlet]))
    return declared, predicted, abs(predicted - declared)


def _binned_median(x_values: np.ndarray, y_values: np.ndarray, *, bins: int) -> BinnedMedianSummary:
    """
    Return exact non-empty equal-width median bins.

    Parameters
    ----------
    x_values, y_values : numpy.ndarray
        Aligned explanatory and response values of any shape.
    bins : int
        Requested number of equal-width bins.

    Returns
    -------
    BinnedMedianSummary
        Centers, exact medians, and counts for non-empty bins only.

    Raises
    ------
    ValueError
        If flattened inputs are empty or have different sizes.

    """
    x_array = np.asarray(x_values, dtype=float).ravel()
    y_array = np.asarray(y_values, dtype=float).ravel()
    if x_array.size != y_array.size or x_array.size == 0:
        msg = "Binned error trend requires matching non-empty values."
        raise ValueError(msg)
    low, high = float(np.min(x_array)), float(np.max(x_array))
    if np.isclose(low, high):
        return BinnedMedianSummary(
            centers=_readonly(np.asarray([low])),
            medians=_readonly(np.asarray([float(np.median(y_array))])),
            counts=_readonly(np.asarray([x_array.size], dtype=np.int64)),
        )
    edges = np.linspace(low, high, bins + 1)
    assignments = np.clip(np.digitize(x_array, edges) - 1, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=np.int64)
    for index in range(bins):
        selected = y_array[assignments == index]
        counts[index] = selected.size
        if selected.size:
            medians[index] = float(np.median(selected))
    valid = counts > 0
    return BinnedMedianSummary(
        centers=_readonly(centers[valid]),
        medians=_readonly(medians[valid]),
        counts=_readonly(counts[valid]),
    )


def _empirical_quantiles(values: np.ndarray) -> EmpiricalQuantileSummary:
    """
    Build the versioned bounded local-error quantile representation.

    Parameters
    ----------
    values : numpy.ndarray
        Finite non-empty local relative-error population.

    Returns
    -------
    EmpiricalQuantileSummary
        Fixed-grid linear quantiles plus exact population size and endpoints.

    Raises
    ------
    ValueError
        If the supplied population is empty or contains non-finite values.

    """
    numeric = np.asarray(values, dtype=float).ravel()
    if numeric.size == 0 or not np.isfinite(numeric).all():
        msg = "Local relative-error values must be finite and non-empty."
        raise ValueError(msg)
    quantiles = np.quantile(
        numeric,
        LOCAL_EMPIRICAL_QUANTILE_PROBABILITIES,
        method="linear",
    )
    quantiles[0] = float(np.min(numeric))
    quantiles[-1] = float(np.max(numeric))
    return EmpiricalQuantileSummary(
        definition=LOCAL_EMPIRICAL_QUANTILE_DEFINITION,
        version=LOCAL_EMPIRICAL_QUANTILE_VERSION,
        probabilities=LOCAL_EMPIRICAL_QUANTILE_PROBABILITIES,
        quantiles=_readonly(quantiles),
        source_point_count=int(numeric.size),
        exact_minimum=float(quantiles[0]),
        exact_maximum=float(quantiles[-1]),
    )


def _magnitude_group_specs(
    frame: pd.DataFrame,
    case: cases.EvaluationCase,
) -> tuple[tuple[str, tuple[str, ...], tuple[int, ...], str], ...]:
    """Return canonical same-unit multi-field TaskSpec groups and case indices."""
    units = dict(zip(case.fields, case.units, strict=True))
    result: list[tuple[str, tuple[str, ...], tuple[int, ...], str]] = []
    for group_id_value, fields_value in frame.attrs.get("output_groups", ()):
        group_id = str(group_id_value)
        component_fields = tuple(str(field) for field in fields_value)
        if len(component_fields) < _MINIMUM_MAGNITUDE_COMPONENTS:
            continue
        if any(field not in case.fields for field in component_fields):
            msg = f"Output group {group_id!r} contains a field absent from the admitted case."
            raise dataframe.ComparisonCompatibilityError(msg)
        group_units = {units[field] for field in component_fields}
        if len(group_units) != 1:
            msg = f"Output group {group_id!r} cannot form a magnitude across unlike physical units."
            raise dataframe.ComparisonCompatibilityError(msg)
        indices = tuple(case.fields.index(field) for field in component_fields)
        result.append((group_id, component_fields, indices, next(iter(group_units))))
    return tuple(result)


class EvaluationSession:
    """
    Own bounded case and numerical caches for one explicit evaluation lifetime.

    Notes
    -----
    A session is intentionally stateful and must be closed. It never caches
    figures, presentation labels, or values from another run-role identity.

    """

    def __init__(
        self,
        datasets: Mapping[str, pd.DataFrame] | None = None,
        *,
        max_case_entries: int = 8,
        max_case_bytes: int = 256 * 1024**2,
        max_aggregate_bytes: int = 512 * 1024**2,
        max_spill_bytes: int = 4 * 1024**3,
        working_bytes: int = 96 * 1024**2,
    ) -> None:
        """
        Initialize one empty bounded cache lifetime and optionally bind frames.

        Parameters
        ----------
        datasets : Mapping[str, pandas.DataFrame] | None, optional
            Provenance-complete frames to bind. Mapping labels are presentation
            only and do not enter artifact keys.
        max_case_entries : int, optional
            Maximum selected cases retained by the LRU.
        max_case_bytes : int, optional
            Maximum unique array bytes retained by selected cases.
        max_aggregate_bytes : int, optional
            Maximum immutable numerical-summary bytes retained in memory.
        max_spill_bytes : int, optional
            Maximum temporary projection bytes owned by this session.
        working_bytes : int, optional
            Target memory bound for blockwise exact spatial reductions.

        Raises
        ------
        ValueError
            If any resource limit is not a positive integer.
        TypeError, ComparisonCompatibilityError
            If an optionally supplied frame lacks complete artifact identity.

        """
        limits = (max_case_entries, max_case_bytes, max_aggregate_bytes, max_spill_bytes, working_bytes)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0 for value in limits):
            msg = "Evaluation session limits must be positive integers."
            raise ValueError(msg)
        self.max_case_entries = int(max_case_entries)
        self.max_case_bytes = int(max_case_bytes)
        self.max_aggregate_bytes = int(max_aggregate_bytes)
        self.max_spill_bytes = int(max_spill_bytes)
        self.working_bytes = int(working_bytes)
        self._lock = threading.RLock()
        self._closed = False
        self._bindings: dict[int, _Binding] = {}
        self._canonical_frames: dict[tuple[str, str], pd.DataFrame] = {}
        self._case_cache: OrderedDict[tuple[ArtifactKey, int], _CacheEntry] = OrderedDict()
        self._aggregate_cache: OrderedDict[tuple[Any, ...], _CacheEntry] = OrderedDict()
        self._case_bytes = 0
        self._aggregate_bytes = 0
        self._spill_bytes = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="evaluation-session-")
        self._statistics: dict[str, int] = {
            "npz_opens": 0,
            "case_validations": 0,
            "case_cache_hits": 0,
            "case_cache_misses": 0,
            "case_cache_evictions": 0,
            "case_cache_oversize": 0,
            "aggregate_hits": 0,
            "aggregate_misses": 0,
            "aggregate_evictions": 0,
            "full_scans": 0,
            "full_case_visits": 0,
            "prefix_scans": 0,
            "prefix_case_visits": 0,
            "artifact_guard_checks": 0,
            "artifact_invalidations": 0,
            "case_bytes_peak": 0,
            "aggregate_bytes_peak": 0,
            "spill_bytes_peak": 0,
        }
        if datasets is not None:
            self.bind(datasets)

    @classmethod
    def from_loaded_runs(
        cls,
        loaded_runs: Iterable[LoadedRunArtifacts],
        **limits: Unpack[_SessionLimits],
    ) -> Self:
        """
        Build one session from strict load-only completed-run results.

        Parameters
        ----------
        loaded_runs : Iterable[LoadedRunArtifacts]
            Non-empty strict-loader results. Each canonical run contributes its
            validated ``eval`` and ``ood`` frames exactly once.
        **limits : int
            Optional resource limits accepted by :class:`EvaluationSession`.

        Returns
        -------
        EvaluationSession
            Live session with frames bound under ``(run_name, split_role)`` keys.

        Raises
        ------
        TypeError
            If an item does not expose the strict loaded-run contract.
        ValueError
            If input is empty, run names repeat, or artifact roles contradict
            their required ``eval`` and ``ood`` bindings.

        Notes
        -----
        The method uses structural access to the strict loader result. The loader
        module never imports this session module, so no runtime cycle is created.
        Presentation labels remain a separate notebook concern.

        """
        runs = tuple(loaded_runs)
        if not runs:
            msg = "from_loaded_runs requires at least one completed-run result."
            raise ValueError(msg)
        session = cls(**limits)
        try:
            cls._bind_loaded_runs(session, runs)
        except Exception:
            session.close()
            raise
        else:
            return session

    @staticmethod
    def _bind_loaded_runs(
        session: EvaluationSession,
        runs: tuple[LoadedRunArtifacts, ...],
    ) -> None:
        """
        Validate and bind strict-loader results to one live session.

        Parameters
        ----------
        session : EvaluationSession
            Empty live destination session.
        runs : tuple[LoadedRunArtifacts, ...]
            Non-empty strict-loader results already normalized by the caller.

        Raises
        ------
        TypeError
            If an item does not expose the strict loaded-run contract.
        ValueError
            If run names repeat or role-local artifacts contradict canonical
            run and role identity.

        """
        seen_names: set[str] = set()
        for loaded in runs:
            run_name = getattr(loaded, "run_name", None)
            id_artifact = getattr(loaded, "id_artifact", None)
            ood_artifact = getattr(loaded, "ood_artifact", None)
            if not isinstance(run_name, str) or not run_name or id_artifact is None or ood_artifact is None:
                msg = "loaded_runs items must expose run_name, id_artifact, and ood_artifact."
                raise TypeError(msg)
            if run_name in seen_names:
                msg = f"Duplicate canonical loaded run name: {run_name!r}."
                raise ValueError(msg)
            seen_names.add(run_name)
            for expected_role, artifact in (("eval", id_artifact), ("ood", ood_artifact)):
                split_role = getattr(artifact, "split_role", None)
                frame = getattr(artifact, "frame", None)
                if split_role != expected_role or frame is None:
                    msg = f"Loaded run {run_name!r} must bind one {expected_role!r} artifact frame."
                    raise ValueError(msg)
                key = session.bind_frame(frame)
                if key.run_name != run_name or key.split_role != expected_role:
                    canonical_key = (run_name, expected_role)
                    msg = f"Loaded artifact provenance contradicts canonical run-role key {canonical_key!r}."
                    raise ValueError(msg)

    @property
    def canonical_frames(self) -> Mapping[tuple[str, str], pd.DataFrame]:
        """
        Return a read-only map from canonical run-role keys to bound frames.

        Returns
        -------
        collections.abc.Mapping
            Exact ``(run_name, split_role)`` bindings. Display labels are absent.

        Raises
        ------
        EvaluationSessionClosedError
            If the explicit session lifetime already ended.

        """
        self._require_open()
        return MappingProxyType(dict(self._canonical_frames))

    def __deepcopy__(self, _memo: dict[int, object]) -> EvaluationSession:
        """
        Keep the live session accessor stable when pandas copies frame attrs.

        Parameters
        ----------
        _memo : dict[int, object]
            Standard deepcopy memo, intentionally unused.

        Returns
        -------
        EvaluationSession
            This same stateful session instance.

        """
        return self

    def __enter__(self) -> Self:
        """
        Return this live session for context-managed use.

        Returns
        -------
        EvaluationSession
            This session while its explicit lifetime remains open.

        Raises
        ------
        EvaluationSessionClosedError
            If the session was already released.

        """
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """
        Release all session-owned resources when leaving a context.

        Parameters
        ----------
        _exc_type, _exc, _traceback : object
            Context exception details, intentionally ignored during cleanup.

        """
        self.close()

    @property
    def closed(self) -> bool:
        """
        Return whether all session-owned resources were released.

        Returns
        -------
        bool
            ``True`` after the first explicit or context-managed close.

        """
        return self._closed

    @property
    def counters(self) -> Mapping[str, int]:
        """
        Return a read-only operation and retained-byte snapshot.

        Returns
        -------
        collections.abc.Mapping
            Counter names mapped to exact cumulative operations, current bytes,
            peak bytes, and bound-artifact count.

        Notes
        -----
        The returned mapping is detached from mutable session statistics.

        """
        snapshot = dict(self._statistics)
        snapshot.update(
            {
                "case_bytes_current": self._case_bytes,
                "aggregate_bytes_current": self._aggregate_bytes,
                "spill_bytes_current": self._spill_bytes,
                "bound_artifacts": len(self._bindings),
            }
        )
        return MappingProxyType(snapshot)

    def _require_open(self) -> None:
        """
        Reject an operation that targets a released session.

        Raises
        ------
        EvaluationSessionClosedError
            If the explicit cache lifetime has ended.

        """
        if self._closed:
            msg = "EvaluationSession is closed."
            raise EvaluationSessionClosedError(msg)

    def bind(self, datasets: Mapping[str, pd.DataFrame]) -> None:
        """
        Bind validated frames while excluding presentation labels from identity.

        Parameters
        ----------
        datasets : Mapping[str, pandas.DataFrame]
            Non-empty provenance-complete frames. Mapping keys are ignored for
            numerical identity and remain available to the caller for display.

        Raises
        ------
        EvaluationSessionClosedError
            If the session lifetime already ended.
        ValueError, TypeError, ComparisonCompatibilityError
            If input is empty, already belongs to another session, or lacks exact
            artifact identity.

        """
        self._require_open()
        if not datasets:
            msg = "EvaluationSession.bind requires at least one frame."
            raise ValueError(msg)
        for frame in datasets.values():
            self.bind_frame(frame)

    def bind_frame(self, frame: pd.DataFrame) -> ArtifactKey:
        """
        Bind one provenance-complete frame and return its exact artifact key.

        Parameters
        ----------
        frame : pandas.DataFrame
            Strictly admitted evaluation artifact frame.

        Returns
        -------
        ArtifactKey
            Canonical run-role and payload identity used by every cache.

        Raises
        ------
        EvaluationSessionClosedError
            If the session is closed.
        ValueError
            If another live session owns the frame or the canonical run-role key
            is already bound to a different frame.
        TypeError, ComparisonCompatibilityError
            If the frame lacks complete completed-run provenance.
        OSError
            If an identity-bearing payload file cannot be inspected.

        """
        with self._lock:
            self._require_open()
            existing = frame.attrs.get(SESSION_ATTR)
            if existing is not None and existing is not self and not getattr(existing, "closed", False):
                msg = "Evaluation frame is already bound to another live session."
                raise ValueError(msg)
            key = _artifact_key(frame)
            token = _digest(_frame_identity_payload(frame))
            binding = _Binding(frame=frame, key=key, frame_token=token, witnesses=_witnesses(frame))
            canonical_key = (key.run_name, key.split_role)
            existing_frame = self._canonical_frames.get(canonical_key)
            if existing_frame is not None and existing_frame is not frame:
                msg = f"Canonical evaluation run-role key is already bound: {canonical_key!r}."
                raise ValueError(msg)
            self._bindings[id(frame)] = binding
            self._canonical_frames[canonical_key] = frame
            frame.attrs[SESSION_ATTR] = self
            frame.attrs[SESSION_KEY_ATTR] = key.token
            return key

    def _binding(self, frame: pd.DataFrame) -> _Binding:
        """
        Return one unchanged live binding after identity and payload guards.

        Parameters
        ----------
        frame : pandas.DataFrame
            Candidate frame expected to belong to this session.

        Returns
        -------
        _Binding
            Original binding after live attrs and file witnesses still match.

        Raises
        ------
        EvaluationSessionClosedError
            If the session lifetime ended.
        ValueError
            If the frame is not owned by this session.
        EvaluationArtifactChangedError
            If identity attrs or any protected payload witness changed.

        """
        self._require_open()
        binding = self._bindings.get(id(frame))
        if binding is None or binding.frame is not frame or frame.attrs.get(SESSION_ATTR) is not self:
            msg = "Evaluation frame is not bound to this session."
            raise ValueError(msg)
        if binding.invalid:
            msg = "Bound evaluation artifact changed. Create a new load-only session."
            raise EvaluationArtifactChangedError(msg)
        current_token = _digest(_frame_identity_payload(frame))
        if current_token != binding.frame_token or frame.attrs.get(SESSION_KEY_ATTR) != binding.key.token:
            self._invalidate(binding)
            msg = "Bound frame identity changed. Reload the artifact into a new session."
            raise EvaluationArtifactChangedError(msg)
        self._statistics["artifact_guard_checks"] += 1
        try:
            current_witnesses = _witnesses(frame)
        except OSError as error:
            self._invalidate(binding)
            msg = "A bound artifact payload is missing or unreadable."
            raise EvaluationArtifactChangedError(msg) from error
        if current_witnesses != binding.witnesses:
            self._invalidate(binding)
            msg = "A bound artifact payload changed. Reload and revalidate before reuse."
            raise EvaluationArtifactChangedError(msg)
        return binding

    def _invalidate(self, binding: _Binding) -> None:
        """
        Evict all values for one changed artifact and reject further reuse.

        Parameters
        ----------
        binding : _Binding
            Artifact binding whose identity guard failed.

        Notes
        -----
        Invalidation is idempotent and updates both cache byte accounts.

        """
        if binding.invalid:
            return
        binding.invalid = True
        self._statistics["artifact_invalidations"] += 1
        for cache_key in [key for key in self._case_cache if key[0] == binding.key]:
            entry = self._case_cache.pop(cache_key)
            self._case_bytes -= entry.size
        for cache_key in [key for key in self._aggregate_cache if binding.key in key]:
            entry = self._aggregate_cache.pop(cache_key)
            self._aggregate_bytes -= entry.size

    def _read_case(self, frame: pd.DataFrame, position: int) -> cases.EvaluationCase:
        """
        Open, fully validate, and freeze one uncached NPZ case.

        Parameters
        ----------
        frame : pandas.DataFrame
            Bound artifact frame defining exact saved membership.
        position : int
            Zero-based row position in that membership.

        Returns
        -------
        EvaluationCase
            Fully validated case with immutable caller-visible state.

        Notes
        -----
        The operation counters distinguish the physical NPZ open and validation.

        """
        self._statistics["npz_opens"] += 1
        loaded = cases._load_case_uncached(frame, position)  # noqa: SLF001
        self._statistics["case_validations"] += 1
        return _freeze_case(loaded)

    def load_case(self, frame: pd.DataFrame, row_position: int) -> cases.EvaluationCase:
        """
        Return one immutable selected case through the byte-budget LRU.

        Parameters
        ----------
        frame : pandas.DataFrame
            Frame bound to this live session.
        row_position : int
            Zero-based position in exact saved membership order.

        Returns
        -------
        EvaluationCase
            Fully validated case whose caller-visible arrays are read-only.

        Raises
        ------
        EvaluationSessionClosedError
            If the session lifetime ended.
        EvaluationArtifactChangedError
            If bound frame identity or a protected payload witness changed.
        IndexError, FileNotFoundError, KeyError, TypeError, ValueError
            If row selection or persisted NPZ case validation fails.

        Notes
        -----
        A case larger than the configured byte budget is returned uncached and
        disclosed through ``case_cache_oversize``.

        """
        with self._lock:
            binding = self._binding(frame)
            if isinstance(row_position, bool) or not isinstance(row_position, Integral):
                msg = "row_position must be an integer."
                raise TypeError(msg)
            position = int(row_position)
            cache_key = (binding.key, position)
            entry = self._case_cache.get(cache_key)
            if entry is not None:
                self._case_cache.move_to_end(cache_key)
                self._statistics["case_cache_hits"] += 1
                return entry.value
            self._statistics["case_cache_misses"] += 1
            loaded = self._read_case(frame, position)
            size = _case_nbytes(loaded)
            if size > self.max_case_bytes:
                self._statistics["case_cache_oversize"] += 1
                return loaded
            while self._case_cache and (len(self._case_cache) >= self.max_case_entries or self._case_bytes + size > self.max_case_bytes):
                _old_key, old_entry = self._case_cache.popitem(last=False)
                self._case_bytes -= old_entry.size
                self._statistics["case_cache_evictions"] += 1
            self._case_cache[cache_key] = _CacheEntry(loaded, size)
            self._case_bytes += size
            self._statistics["case_bytes_peak"] = max(self._statistics["case_bytes_peak"], self._case_bytes)
            return loaded

    def _aggregate(self, cache_key: tuple[Any, ...], builder: Any) -> Any:
        """
        Return or build one immutable byte-accounted numerical summary.

        Parameters
        ----------
        cache_key : tuple[Any, ...]
            Artifact-bound numerical identity independent of presentation labels.
        builder : callable
            Zero-argument constructor invoked only on a cache miss.

        Returns
        -------
        Any
            Cached or newly built immutable numerical summary.

        Raises
        ------
        MemoryError
            If one summary alone exceeds the aggregate byte budget.

        """
        entry = self._aggregate_cache.get(cache_key)
        if entry is not None:
            self._aggregate_cache.move_to_end(cache_key)
            self._statistics["aggregate_hits"] += 1
            return entry.value
        self._statistics["aggregate_misses"] += 1
        value = builder()
        size = _value_nbytes(value)
        if size > self.max_aggregate_bytes:
            msg = f"Evaluation numerical summary requires {size} bytes, above max_aggregate_bytes={self.max_aggregate_bytes}."
            raise MemoryError(msg)
        while self._aggregate_cache and self._aggregate_bytes + size > self.max_aggregate_bytes:
            _old_key, old_entry = self._aggregate_cache.popitem(last=False)
            self._aggregate_bytes -= old_entry.size
            self._statistics["aggregate_evictions"] += 1
        self._aggregate_cache[cache_key] = _CacheEntry(value, size)
        self._aggregate_bytes += size
        self._statistics["aggregate_bytes_peak"] = max(self._statistics["aggregate_bytes_peak"], self._aggregate_bytes)
        return value

    def _reserve_spill(self, size: int, *, name: str) -> Path:
        """
        Reserve bounded session scratch and return its private file path.

        Parameters
        ----------
        size : int
            Projection bytes to charge before file creation.
        name : str
            Private filename within the session temporary directory.

        Returns
        -------
        pathlib.Path
            Reserved path for a session-owned memory-mapped projection.

        Raises
        ------
        MemoryError
            If the reservation exceeds the remaining scratch budget.

        """
        if self._spill_bytes + size > self.max_spill_bytes:
            msg = f"Evaluation scratch projection requires {size} bytes, above remaining max_spill_bytes budget."
            raise MemoryError(msg)
        self._spill_bytes += size
        self._statistics["spill_bytes_peak"] = max(self._statistics["spill_bytes_peak"], self._spill_bytes)
        return Path(self._temporary.name) / name

    def _release_spill(self, size: int, path: Path) -> None:
        """
        Release accounted scratch bytes and remove the private file.

        Parameters
        ----------
        size : int
            Previously reserved byte charge.
        path : pathlib.Path
            Session-private projection path to unlink when present.

        """
        self._spill_bytes -= size
        with suppress(OSError):
            path.unlink(missing_ok=True)

    def full_summary(
        self,
        frame: pd.DataFrame,
        max_cases: int | None = None,
    ) -> FullEvaluationSummary:
        """Return complete or explicitly bounded all-field spatial reductions."""
        with self._lock:
            binding = self._binding(frame)
            if max_cases is None:
                count = len(frame)
            else:
                if isinstance(max_cases, bool) or not isinstance(max_cases, Integral) or int(max_cases) <= 0:
                    msg = "max_cases must be a positive integer when supplied."
                    raise ValueError(msg)
                count = min(int(max_cases), len(frame))
            return self._aggregate(("full", binding.key, count), lambda: self._build_full(frame, count))

    def _build_full(self, frame: pd.DataFrame, count: int) -> FullEvaluationSummary:  # noqa: PLR0915
        """Build exact spatial, residual, pressure, and vector-magnitude reductions."""
        self._statistics["full_scans"] += 1
        first = self._read_case(frame, 0)
        self._statistics["full_case_visits"] += 1
        field_count = len(first.fields)
        magnitude_specs = _magnitude_group_specs(frame, first)
        magnitude_count = len(magnitude_specs)
        summary_field_count = field_count + magnitude_count
        height, width = first.shape
        error_bytes = count * summary_field_count * height * width * np.dtype(np.float64).itemsize
        error_path = self._reserve_spill(error_bytes, name=f"full-error-{self._statistics['full_scans']}.bin")
        error_store = np.memmap(
            error_path,
            dtype=np.float64,
            mode="w+",
            shape=(count, summary_field_count, height, width),
        )
        reference_sum = np.zeros_like(first.reference, dtype=np.float64)
        prediction_sum = np.zeros_like(first.prediction, dtype=np.float64)
        magnitude_reference_sum = np.zeros((magnitude_count, height, width), dtype=np.float64)
        magnitude_prediction_sum = np.zeros((magnitude_count, height, width), dtype=np.float64)
        local_relative_error_sum = np.zeros((summary_field_count, height, width), dtype=np.float64)
        case_reference_means = np.empty((count, field_count), dtype=np.float64)
        case_prediction_means = np.empty((count, field_count), dtype=np.float64)
        case_magnitude_reference_means = np.empty((count, magnitude_count), dtype=np.float64)
        case_magnitude_prediction_means = np.empty((count, magnitude_count), dtype=np.float64)
        required_residuals = {"Rx", "Ry", "div_u", "div_eps_u"}
        residual_available = set(first.residuals) == required_residuals
        residual_error = None if residual_available else "Steady-flow residual maps require Rx, Ry, div_u, and div_eps_u."
        momentum_sum = np.zeros((height, width), dtype=np.float64) if residual_available else None
        div_velocity_sum = np.zeros((height, width), dtype=np.float64) if residual_available else None
        div_eps_velocity_sum = np.zeros((height, width), dtype=np.float64) if residual_available else None
        pressure_declared = np.empty(count, dtype=np.float64)
        pressure_predicted = np.empty(count, dtype=np.float64)
        pressure_error_values = np.empty(count, dtype=np.float64)
        pressure_field, pressure_error = _pressure_group_field(frame)
        spatial_error: str | None = None

        def consume(case: cases.EvaluationCase, position: int) -> None:
            """Accumulate one immutable admitted case into every shared reduction."""
            nonlocal residual_error, pressure_error, spatial_error
            case_reference_means[position] = np.mean(case.reference, axis=(1, 2))
            case_prediction_means[position] = np.mean(case.prediction, axis=(1, 2))
            magnitude_reference = np.empty((magnitude_count, height, width), dtype=np.float64)
            magnitude_prediction = np.empty((magnitude_count, height, width), dtype=np.float64)
            for magnitude_index, (_group_id, _fields, indices, _unit) in enumerate(magnitude_specs):
                magnitude_reference[magnitude_index] = np.sqrt(np.sum(case.reference[np.asarray(indices)] ** 2, axis=0))
                magnitude_prediction[magnitude_index] = np.sqrt(np.sum(case.prediction[np.asarray(indices)] ** 2, axis=0))
            if magnitude_count:
                case_magnitude_reference_means[position] = np.mean(magnitude_reference, axis=(1, 2))
                case_magnitude_prediction_means[position] = np.mean(magnitude_prediction, axis=(1, 2))
            compatible = case.prediction.shape == first.prediction.shape and np.allclose(case.coordinates, first.coordinates)
            if not compatible:
                spatial_error = "Spatial evaluation summaries require identical grids within an artifact dataset."
            if spatial_error is None:
                reference_sum[:] += case.reference
                prediction_sum[:] += case.prediction
                magnitude_reference_sum[:] += magnitude_reference
                magnitude_prediction_sum[:] += magnitude_prediction
                error_store[position, :field_count] = case.error
                error_store[position, field_count:] = magnitude_prediction - magnitude_reference
                field_rms = np.sqrt(np.mean(case.reference**2, axis=(1, 2)))
                local_relative_error_sum[:field_count] += np.abs(case.error) / (field_rms[:, None, None] + _LOCAL_DENOMINATOR_FLOOR)
                if magnitude_count:
                    magnitude_rms = np.sqrt(np.mean(magnitude_reference**2, axis=(1, 2)))
                    local_relative_error_sum[field_count:] += np.abs(magnitude_prediction - magnitude_reference) / (
                        magnitude_rms[:, None, None] + _LOCAL_DENOMINATOR_FLOOR
                    )
            if residual_error is None:
                if set(case.residuals) != required_residuals or not compatible:
                    residual_error = "Spatial residual aggregation requires identical full grids and array semantics."
                else:
                    cast("np.ndarray", momentum_sum)[:] += np.sqrt(case.residuals["Rx"] ** 2 + case.residuals["Ry"] ** 2)
                    cast("np.ndarray", div_velocity_sum)[:] += np.abs(case.residuals["div_u"])
                    cast("np.ndarray", div_eps_velocity_sum)[:] += np.abs(case.residuals["div_eps_u"])
            if pressure_error is None:
                try:
                    declared, predicted, absolute_error = _pressure_drop(case, pressure_field=cast("str", pressure_field))
                except (ValueError, dataframe.ComparisonCompatibilityError) as error:
                    pressure_error = str(error)
                else:
                    pressure_declared[position] = declared
                    pressure_predicted[position] = predicted
                    pressure_error_values[position] = absolute_error

        try:
            consume(first, 0)
            for position in range(1, count):
                case = self._read_case(frame, position)
                self._statistics["full_case_visits"] += 1
                consume(case, position)
            if spatial_error is None:
                reference_mean: np.ndarray | None = reference_sum / count
                prediction_mean: np.ndarray | None = prediction_sum / count
                all_signed_mean = np.empty((summary_field_count, height, width), dtype=np.float64)
                all_absolute_mean = np.empty_like(all_signed_mean)
                all_signed_std = np.empty_like(all_signed_mean)
                all_absolute_q90 = np.empty_like(all_signed_mean)
                flattened = error_store.reshape(count, summary_field_count, height * width)
                bytes_per_point = max(1, count * summary_field_count * np.dtype(np.float64).itemsize * 4)
                block_points = max(1, self.working_bytes // bytes_per_point)
                for block_start in range(0, height * width, block_points):
                    block_stop = min(height * width, block_start + block_points)
                    block = np.asarray(flattened[:, :, block_start:block_stop])
                    all_signed_mean.reshape(summary_field_count, -1)[:, block_start:block_stop] = np.mean(block, axis=0)
                    all_signed_std.reshape(summary_field_count, -1)[:, block_start:block_stop] = np.std(block, axis=0)
                    absolute = np.abs(block)
                    all_absolute_mean.reshape(summary_field_count, -1)[:, block_start:block_stop] = np.mean(absolute, axis=0)
                    all_absolute_q90.reshape(summary_field_count, -1)[:, block_start:block_stop] = np.quantile(absolute, 0.9, axis=0)
                signed_mean: np.ndarray | None = all_signed_mean[:field_count]
                absolute_mean: np.ndarray | None = all_absolute_mean[:field_count]
                local_relative_mean: np.ndarray | None = local_relative_error_sum[:field_count] / count
                signed_std: np.ndarray | None = all_signed_std[:field_count]
                absolute_q90: np.ndarray | None = all_absolute_q90[:field_count]
                magnitude_reference_mean = magnitude_reference_sum / count
                magnitude_prediction_mean = magnitude_prediction_sum / count
                magnitude_summaries = {
                    group_id: FullMagnitudeSummary(
                        component_fields=component_fields,
                        unit=unit,
                        reference_mean=_readonly(magnitude_reference_mean[index]),
                        prediction_mean=_readonly(magnitude_prediction_mean[index]),
                        signed_error_mean=_readonly(all_signed_mean[field_count + index]),
                        absolute_error_mean=_readonly(all_absolute_mean[field_count + index]),
                        local_relative_error_mean=_readonly(local_relative_error_sum[field_count + index] / count),
                        signed_error_std=_readonly(all_signed_std[field_count + index]),
                        absolute_error_q90=_readonly(all_absolute_q90[field_count + index]),
                        case_reference_means=_readonly(case_magnitude_reference_means[:, index]),
                        case_prediction_means=_readonly(case_magnitude_prediction_means[:, index]),
                    )
                    for index, (group_id, component_fields, _indices, unit) in enumerate(magnitude_specs)
                }
            else:
                reference_mean = prediction_mean = signed_mean = absolute_mean = local_relative_mean = signed_std = absolute_q90 = None
                magnitude_summaries = {}
        finally:
            error_store.flush()
            self._release_spill(error_bytes, error_path)

        if residual_error is None:
            momentum_mean: np.ndarray | None = cast("np.ndarray", momentum_sum) / count
            div_velocity_mean: np.ndarray | None = cast("np.ndarray", div_velocity_sum) / count
            div_eps_velocity_mean: np.ndarray | None = cast("np.ndarray", div_eps_velocity_sum) / count
        else:
            momentum_mean = div_velocity_mean = div_eps_velocity_mean = None
        if pressure_error is not None:
            declared_values = predicted_values = absolute_values = None
        else:
            declared_values = pressure_declared
            predicted_values = pressure_predicted
            absolute_values = pressure_error_values
        grid = GridDescriptor(
            coordinates=_readonly(np.array(first.coordinates, copy=True)),
            coordinate_units=first.coordinate_units,
            fields=first.fields,
            units=first.units,
        )
        return FullEvaluationSummary(
            sample_count=count,
            grid=grid,
            reference_mean=None if reference_mean is None else _readonly(reference_mean),
            prediction_mean=None if prediction_mean is None else _readonly(prediction_mean),
            signed_error_mean=None if signed_mean is None else _readonly(signed_mean),
            absolute_error_mean=None if absolute_mean is None else _readonly(absolute_mean),
            local_relative_error_mean=None if local_relative_mean is None else _readonly(local_relative_mean),
            signed_error_std=None if signed_std is None else _readonly(signed_std),
            absolute_error_q90=None if absolute_q90 is None else _readonly(absolute_q90),
            case_reference_means=_readonly(case_reference_means),
            case_prediction_means=_readonly(case_prediction_means),
            momentum_mean=None if momentum_mean is None else _readonly(momentum_mean),
            div_velocity_mean=None if div_velocity_mean is None else _readonly(div_velocity_mean),
            div_eps_velocity_mean=None if div_eps_velocity_mean is None else _readonly(div_eps_velocity_mean),
            pressure_declared=None if declared_values is None else _readonly(declared_values),
            pressure_predicted=None if predicted_values is None else _readonly(predicted_values),
            pressure_absolute_error=None if absolute_values is None else _readonly(absolute_values),
            magnitudes=MappingProxyType(magnitude_summaries),
            spatial_error=spatial_error,
            residual_error=residual_error,
            pressure_error=pressure_error,
        )

    def prefix_summary(self, frame: pd.DataFrame, max_cases: int) -> PrefixEvaluationSummary:
        """
        Return all all-field reductions for one deterministic saved prefix.

        Parameters
        ----------
        frame : pandas.DataFrame
            Frame bound to this live session.
        max_cases : int
            Positive saved-membership prefix bound.

        Returns
        -------
        PrefixEvaluationSummary
            Immutable local-error, binned-trend, and spectral evidence.

        Raises
        ------
        EvaluationSessionClosedError
            If the session lifetime ended.
        EvaluationArtifactChangedError
            If bound frame identity or a protected payload witness changed.
        ValueError
            If ``max_cases`` is not positive or numerical evidence is invalid.
        MemoryError
            If scratch or retained-summary byte limits are insufficient.
        IndexError, FileNotFoundError, KeyError, TypeError
            If persisted case membership or payload validation fails.

        """
        with self._lock:
            binding = self._binding(frame)
            if isinstance(max_cases, bool) or not isinstance(max_cases, Integral) or int(max_cases) <= 0:
                msg = "max_cases must be a positive integer."
                raise ValueError(msg)
            count = min(int(max_cases), len(frame))
            return self._aggregate(("prefix", binding.key, count), lambda: self._build_prefix(frame, count))

    def _build_prefix(self, frame: pd.DataFrame, count: int) -> PrefixEvaluationSummary:
        """
        Build all bounded-prefix reductions in one case pass across every field.

        Parameters
        ----------
        frame : pandas.DataFrame
            Bound artifact frame in exact saved membership order.
        count : int
            Effective positive prefix length, already capped by frame length.

        Returns
        -------
        PrefixEvaluationSummary
            Immutable local-error, trend, and spectral evidence for all fields.

        Notes
        -----
        The projection is bounded by ``max_spill_bytes`` and deleted immediately
        after reduction. Only fixed-size summaries remain in the aggregate LRU.

        """
        self._statistics["prefix_scans"] += 1
        first = self._read_case(frame, 0)
        self._statistics["prefix_case_visits"] += 1
        field_count = len(first.fields)
        magnitude_specs = _magnitude_group_specs(frame, first)
        height, width = first.shape
        point_count = height * width
        projection_bytes = count * (2 * field_count + 2) * point_count * np.dtype(np.float64).itemsize
        projection_path = self._reserve_spill(projection_bytes, name=f"prefix-{self._statistics['prefix_scans']}.bin")
        projection = np.memmap(
            projection_path,
            dtype=np.float64,
            mode="w+",
            shape=(count, 2 * field_count + 2, height, width),
        )
        spacings = np.empty((count, 2), dtype=np.float64)
        coordinate_units: list[tuple[str, str]] = []

        def consume(case: cases.EvaluationCase, position: int) -> None:
            """
            Write one validated case into the bounded numerical projection.

            Parameters
            ----------
            case : EvaluationCase
                Immutable validated prefix case.
            position : int
                Destination row in the session-owned projection.

            Raises
            ------
            ComparisonCompatibilityError
                If prefix cases do not share one learned-field grid shape.

            """
            if case.reference.shape != first.reference.shape:
                msg = "Bounded-prefix evaluation summaries require one shared grid shape within an artifact."
                raise dataframe.ComparisonCompatibilityError(msg)
            projection[position, :field_count] = case.reference
            projection[position, field_count : 2 * field_count] = case.error
            projection[position, -2] = _boundary_distance(case)
            projection[position, -1] = _horizontal_boundary_fraction(case)
            spacings[position] = cases.grid_spacing(case)
            coordinate_units.append(case.coordinate_units)

        try:
            consume(first, 0)
            for position in range(1, count):
                case = self._read_case(frame, position)
                self._statistics["prefix_case_visits"] += 1
                consume(case, position)
            projection.flush()
            references = projection[:, :field_count]
            errors = projection[:, field_count : 2 * field_count]
            distances = projection[:, -2]
            boundary_fractions = projection[:, -1]
            local: dict[str, EmpiricalQuantileSummary] = {}
            target: dict[str, BinnedMedianSummary] = {}
            boundary: dict[str, BinnedMedianSummary] = {}
            boundary_regions: dict[str, BinnedMeanSummary] = {}
            spectra: dict[str, SpectralFieldSummary] = {}
            dx, dy = spacings[0]
            if not np.allclose(spacings, spacings[0]):
                msg = "Spectral aggregation requires a shared grid and spacing within each artifact."
                raise dataframe.ComparisonCompatibilityError(msg)

            def reduce_field(
                reference: np.ndarray,
                error: np.ndarray,
            ) -> tuple[
                EmpiricalQuantileSummary,
                BinnedMedianSummary,
                BinnedMedianSummary,
                BinnedMeanSummary,
                SpectralFieldSummary,
            ]:
                """Reduce one learned field or canonical vector magnitude."""
                reference_rms = np.sqrt(np.mean(reference**2, axis=(1, 2)))
                local_values = np.abs(error) / (reference_rms[:, None, None] + _LOCAL_DENOMINATOR_FLOOR)
                local_summary = _empirical_quantiles(local_values)
                target_summary = _binned_median(np.abs(reference), np.abs(error), bins=_TARGET_BIN_COUNT)
                boundary_summary = _binned_median(distances, np.abs(error), bins=_BOUNDARY_BIN_COUNT)
                boundary_region_summary = _boundary_region_mean(boundary_fractions, np.abs(error))
                spectral_values: dict[str, list[np.ndarray]] = {"reference": [], "prediction": [], "error": []}
                frequency_reference: np.ndarray | None = None
                for position in range(count):
                    for name, values in (
                        ("reference", reference[position]),
                        ("prediction", reference[position] + error[position]),
                        ("error", error[position]),
                    ):
                        frequencies, energy = radial_power_spectrum(values, dx=float(dx), dy=float(dy))
                        if frequency_reference is None:
                            frequency_reference = frequencies
                        elif not np.allclose(frequencies, frequency_reference):
                            msg = "Spectral bins changed within one artifact."
                            raise dataframe.ComparisonCompatibilityError(msg)
                        spectral_values[name].append(energy)
                spectrum = SpectralFieldSummary(
                    frequencies=_readonly(cast("np.ndarray", frequency_reference)),
                    reference=_readonly(np.stack(spectral_values["reference"])),
                    prediction=_readonly(np.stack(spectral_values["prediction"])),
                    error=_readonly(np.stack(spectral_values["error"])),
                )
                return local_summary, target_summary, boundary_summary, boundary_region_summary, spectrum

            for field_index, field in enumerate(first.fields):
                reductions = reduce_field(
                    np.asarray(references[:, field_index]),
                    np.asarray(errors[:, field_index]),
                )
                local[field], target[field], boundary[field], boundary_regions[field], spectra[field] = reductions

            magnitude_summaries: dict[str, PrefixMagnitudeSummary] = {}
            for group_id, component_fields, indices, unit in magnitude_specs:
                component_indices = np.asarray(indices)
                magnitude_reference = np.sqrt(np.sum(np.asarray(references[:, component_indices]) ** 2, axis=1))
                component_prediction = np.asarray(references[:, component_indices]) + np.asarray(errors[:, component_indices])
                magnitude_prediction = np.sqrt(np.sum(component_prediction**2, axis=1))
                reductions = reduce_field(magnitude_reference, magnitude_prediction - magnitude_reference)
                magnitude_summaries[group_id] = PrefixMagnitudeSummary(
                    component_fields=component_fields,
                    unit=unit,
                    local_relative_error=reductions[0],
                    target_magnitude_error=reductions[1],
                    boundary_distance_error=reductions[2],
                    boundary_region_error=reductions[3],
                    spectrum=reductions[4],
                )
            if coordinate_units[0][0] != coordinate_units[0][1]:
                msg = "Boundary distance requires x and y coordinates with the same physical unit."
                raise dataframe.ComparisonCompatibilityError(msg)
            return PrefixEvaluationSummary(
                case_count=count,
                fields=first.fields,
                units=first.units,
                coordinate_unit=coordinate_units[0][0],
                local_relative_error=MappingProxyType(local),
                target_magnitude_error=MappingProxyType(target),
                boundary_distance_error=MappingProxyType(boundary),
                boundary_region_error=MappingProxyType(boundary_regions),
                spectra=MappingProxyType(spectra),
                magnitudes=MappingProxyType(magnitude_summaries),
            )
        finally:
            projection.flush()
            self._release_spill(projection_bytes, projection_path)

    def close(self) -> None:
        """
        Release all caches, detach live frames, and delete session scratch.

        Notes
        -----
        Closing is idempotent. A closed session rejects every later access and
        never reconstructs detached state.

        """
        with self._lock:
            if self._closed:
                return
            for binding in self._bindings.values():
                frame = binding.frame
                if frame.attrs.get(SESSION_ATTR) is self:
                    frame.attrs.pop(SESSION_ATTR, None)
                    frame.attrs.pop(SESSION_KEY_ATTR, None)
            self._case_cache.clear()
            self._aggregate_cache.clear()
            self._case_bytes = 0
            self._aggregate_bytes = 0
            self._spill_bytes = 0
            self._bindings.clear()
            self._canonical_frames.clear()
            self._temporary.cleanup()
            self._closed = True


def bound_session(frame: pd.DataFrame) -> EvaluationSession | None:
    """
    Return a live frame-owned session or ``None`` for direct callers.

    Parameters
    ----------
    frame : pandas.DataFrame
        Candidate evaluation frame.

    Returns
    -------
    EvaluationSession | None
        Live attached session or ``None`` when no session was bound.

    Raises
    ------
    TypeError
        If the internal frame accessor is malformed.
    EvaluationSessionClosedError
        If the frame still references a released session.

    """
    value = frame.attrs.get(SESSION_ATTR)
    if value is None:
        return None
    if not isinstance(value, EvaluationSession):
        msg = "Evaluation frame contains an invalid session accessor."
        raise TypeError(msg)
    if value.closed:
        msg = "Evaluation frame is bound to a closed session."
        raise EvaluationSessionClosedError(msg)
    return value


@contextmanager
def scoped_session(datasets: Mapping[str, pd.DataFrame]) -> Iterator[EvaluationSession]:
    """
    Reuse one bound session or create an ephemeral direct-call session.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Frames consumed by one public plot call.

    Yields
    ------
    EvaluationSession
        Existing shared session or scoped fallback released after the call.

    Raises
    ------
    ValueError
        If input is empty or compared frames mix session ownership.
    EvaluationSessionClosedError
        If a frame references a released session.
    TypeError, ComparisonCompatibilityError, OSError
        If ephemeral-session binding cannot establish exact artifact identity.

    Notes
    -----
    The fallback preserves direct plot behavior without creating global state.

    """
    sessions = {bound_session(frame) for frame in datasets.values()}
    sessions.discard(None)
    if len(sessions) > 1:
        msg = "Compared evaluation frames belong to different live sessions."
        raise ValueError(msg)
    if sessions:
        session = cast("EvaluationSession", next(iter(sessions)))
        if any(bound_session(frame) is not session for frame in datasets.values()):
            msg = "Every compared frame must belong to the same evaluation session."
            raise ValueError(msg)
        yield session
        return
    with EvaluationSession(datasets) as ephemeral:
        yield ephemeral
