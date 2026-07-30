"""
===============================================================================
analysis_artifacts.py
===============================================================================
Create persistent evaluation artifacts from trained neural-operator runs.

Responsibilities:
  - Run deterministic inference over explicit evaluation loaders
  - Write Parquet scalar metrics and NPZ field artifacts
  - Compute field, physics, boundary and metadata diagnostics
  - Keep artifact columns stable for downstream analysis modules

Design principles:
  - Artifacts are reproducible, split-aware, and ordered by saved membership
  - Physical units and the caller-provided TaskSpec field order are explicit
  - Provenance publishes last and binds every completed payload digest
  - Heavy inference work stays out of plotting modules

This module does NOT:
  - Reconstruct models or choose the saved split membership to evaluate
  - Reuse, rebuild, lock, or upload an existing artifact cache target
  - Render interactive or curated scientific figures

Notes:
  Artifact contents:
    - Parquet stores case-level scalar metrics, artifact paths and JSON-safe metadata
    - NPZ stores predictions, targets, errors, kappa fields, raw tensors and residual fields
    - residual metrics use the same canonical evaluation crop for model comparability
  Schema:
    Artifacts
    ---------
    Parquet (global, per case):
        - case_index        : stable one-based source case id
        - source_index      : zero-based original final-dataset index
        - split_local_index : zero-based position in the saved split
        - npz_path          : path to the corresponding NPZ artifact

        # Relative field errors (dimensionless, channel-balanced)
        - rel_l2            : channel-balanced mean relative L2 on [p, u, v] over the full domain
        - rel_h1            : channel-balanced mean relative H1 on [p, u, v] over the interior (cropped by EVAL_PAD)

        # Book-friendly metrics (physical units, intuitive)
        - rmse_p            : RMSE of pressure p over the full domain
        - rmse_u            : RMSE of x-velocity u over the full domain
        - rmse_v            : RMSE of y-velocity v over the full domain
        - rmse_U            : RMSE of speed magnitude U=sqrt(u^2+v^2) over the full domain

        # Physics residual metrics (interior-cropped by EVAL_PAD)
        - momentum_residual_mse : mean(Rx**2 + Ry**2), unit (Pa/m)^2
        - div_velocity_mse      : mean(div(u)**2), unit 1/s^2
        - div_eps_velocity_mse  : mean(div(eps*u)**2), unit 1/s^2

        # Boundary metrics (full-grid inlet/outlet masks, unit Pa^2)
        - pressure_inlet_mse            : mean_inlet((p-p_bc)**2)
        - pressure_outlet_mean_square   : mean_outlet(p)**2
        - pressure_boundary_mse         : sum of the preceding two terms

        # Normalized primary-objective evidence (one triplet per TaskSpec output)
        - normalized_sse_<field>   : float64 per-case normalized squared-error sum
        - normalized_count_<field> : exact selected normalized element count
        - normalized_rmse_<field>  : sqrt(per-case SSE/count), never averaged for the objective

        # Diagnostics
        - kappa_names       : list of available permeability tensor components
        - meta              : JSON-safe metadata dictionary (stored as JSON string)

    NPZ (local, full fields per case):
        - case_index   : stable one-based source case id
        - source_index : zero-based original final-dataset index
        - split_local_index : zero-based position in the saved split
        - pred         : (C_artifact, H, W) prediction aligned with artifact_fields
        - gt           : (C_artifact, H, W) ground truth aligned with artifact_fields
        - err          : (C_artifact, H, W) prediction error (pred - gt)
        - artifact_fields : list[str] names aligned with pred/gt/err
        - artifact_units  : list[str] physical units aligned with artifact_fields

        - kappa_encoded: (C_kappa, H, W) task-stored permeability representations
        - kappa        : (C_kappa, H, W) physical permeability components
        - kappa_names  : list[str], same order as kappa channels
        - p_bc         : (1, H, W) pressure boundary condition
        - coordinates  : (2, H, W) physical x/y coordinate fields

        # Declared inputs and targets retained for downstream analysis
        - x_raw        : (C_in, H, W) raw input tensor (physical units)
        - y_raw        : (C_out, H, W) raw target tensor (physical units)
        - input_fields : list[str] canonical input channel names
        - output_fields: list[str] canonical learned-output names aligned with y_raw
        - output_units : list[str] physical units aligned with output_fields

        # Physics diagnostic fields (full fields, not cropped)
        - Rx           : (H, W) x-momentum residual field
        - Ry           : (H, W) y-momentum residual field
        - div_u        : (H, W) divergence field div(u)
        - div_eps_u    : (H, W) divergence field div(eps u)

        - meta         : JSON string with full metadata

    Provenance publishes last and contains the scientific identity, exact output
    digests, and aggregate normalized_macro_rmse obtained by summing each field's
    per-case SSE/count before finalizing field RMSEs and their arithmetic mean.
===============================================================================

"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import common, domain

from . import analysis_timing as timing

# ============================================================================
# Global constants
# ============================================================================

INTERNAL_KAPPA_NAMES = set(domain.permeability.INTERNAL_KAPPA_2D_ORDER) | set(domain.permeability.INTERNAL_KAPPA_3D_ORDER)
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 1
RESIDUAL_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"
NORMALIZED_OBJECTIVE_TOLERANCE = {"rtol": 1e-12, "atol": 1e-12}

# ----------------------------------------------------------------------------
# Canonical evaluation settings (for model-to-model comparability)
# ----------------------------------------------------------------------------
EVAL_PAD = 2  # crop for ALL gradient-based metrics (H1 + physics)
ARTIFACT_DERIVATIVE_KIND = "spectral"
ARTIFACT_DERIVATIVE_EXTENSION = "reflect"
_MIN_NORMALIZED_TENSOR_RANK = 3
_MIN_BATCH_TENSOR_RANK = 2


def normalized_statistic_columns(field_name: str) -> tuple[str, str, str]:
    """
    Return task-derived per-case normalized SSE, count, and RMSE columns.

    Parameters
    ----------
    field_name : str
        Exact non-empty TaskSpec output field name.

    Returns
    -------
    tuple[str, str, str]
        Ordered sufficient-statistic and convenience-RMSE column names.

    Raises
    ------
    ValueError
        If the field name is empty or not text.

    """
    if not isinstance(field_name, str) or not field_name:
        msg = "Artifact output field names must be non-empty strings."
        raise ValueError(msg)
    return (
        f"normalized_sse_{field_name}",
        f"normalized_count_{field_name}",
        f"normalized_rmse_{field_name}",
    )


def normalized_case_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    output_fields: Iterable[str],
) -> list[dict[str, float | int]]:
    """
    Return exact float64 normalized SSE/count/RMSE evidence per sample and field.

    The subtraction and square are performed after conversion to float64, which
    matches the authoritative runtime ``MacroRMSEMetric`` accumulation policy.
    No per-case RMSE is used when the dataset-level macro objective is finalized.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Matching normalized tensors in batch, channel, and spatial order.
    output_fields : Iterable[str]
        Unique TaskSpec output names aligned with the channel axis.

    Returns
    -------
    list[dict[str, float | int]]
        One row per case containing each field's exact normalized squared-error
        sum, element count, and derived per-case RMSE.

    Raises
    ------
    ValueError
        If tensor shapes/ranks or channel-to-field alignment disagree.
    FloatingPointError
        If any normalized sufficient statistic is non-finite.

    """
    fields = tuple(output_fields)
    if prediction.shape != target.shape or prediction.ndim < _MIN_NORMALIZED_TENSOR_RANK:
        msg = (
            "Normalized prediction and target must have matching "
            f"batch/channel/spatial shapes, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
        raise ValueError(msg)
    if prediction.shape[1] != len(fields) or len(fields) != len(set(fields)):
        msg = "Normalized artifact statistics require unique output fields matching the tensor channels."
        raise ValueError(msg)
    error_squared = (prediction.double() - target.double()).square()
    rows: list[dict[str, float | int]] = []
    for sample_index in range(prediction.shape[0]):
        row: dict[str, float | int] = {}
        for field_index, field_name in enumerate(fields):
            sse_column, count_column, rmse_column = normalized_statistic_columns(field_name)
            field_error = error_squared[sample_index, field_index]
            squared_error_sum = float(field_error.sum().detach().cpu().item())
            element_count = int(field_error.numel())
            field_rmse = math.sqrt(squared_error_sum / element_count)
            if not math.isfinite(squared_error_sum) or not math.isfinite(field_rmse):
                msg = f"Normalized artifact statistics are non-finite for field {field_name!r}, sample {sample_index}."
                raise FloatingPointError(msg)
            row[sse_column] = squared_error_sum
            row[count_column] = element_count
            row[rmse_column] = field_rmse
        rows.append(row)
    return rows


def aggregate_normalized_macro_rmse(
    frame: pd.DataFrame,
    *,
    output_fields: Iterable[str],
) -> dict[str, Any]:
    """
    Finalize global field RMSEs and their unweighted arithmetic macro mean.

    Per-field SSE and element counts are summed in exact membership order.
    Per-case or per-batch RMSE values never enter the aggregate.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema per-case table containing normalized evidence columns.
    output_fields : Iterable[str]
        Unique TaskSpec output names in declared field order.

    Returns
    -------
    dict[str, Any]
        Objective semantics, global per-field SSE/count/RMSE values, their
        unweighted arithmetic RMSE mean, and the declared agreement tolerance.

    Raises
    ------
    KeyError
        If any required evidence column is absent.
    TypeError, ValueError
        If fields, evidence values, counts, or per-case RMSE consistency fail.
    RuntimeError, FloatingPointError
        If no elements can be finalized or the aggregate is non-finite.

    """
    fields = tuple(output_fields)
    if not fields or len(fields) != len(set(fields)):
        msg = "Artifact macro RMSE requires unique non-empty output fields."
        raise ValueError(msg)
    if not frame.columns.is_unique:
        msg = "Artifact macro RMSE cannot consume duplicate DataFrame columns."
        raise ValueError(msg)

    field_summary: dict[str, dict[str, float | int]] = {}
    field_rmse_values: list[float] = []
    for field_name in fields:
        sse_column, count_column, rmse_column = normalized_statistic_columns(field_name)
        missing = [name for name in (sse_column, count_column, rmse_column) if name not in frame.columns]
        if missing:
            msg = f"Artifact macro RMSE is missing normalized evidence columns: {missing}."
            raise KeyError(msg)
        squared_error_sum = 0.0
        element_count = 0
        for row_index, (raw_sse, raw_count, raw_rmse) in enumerate(
            zip(frame[sse_column].tolist(), frame[count_column].tolist(), frame[rmse_column].tolist(), strict=True)
        ):
            if isinstance(raw_sse, bool) or not isinstance(raw_sse, (int, float, np.integer, np.floating)):
                msg = f"{sse_column} row {row_index} must be a real scalar."
                raise TypeError(msg)
            if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)) or int(raw_count) <= 0:
                msg = f"{count_column} row {row_index} must be a positive integer."
                raise TypeError(msg)
            sse_value = float(raw_sse)
            rmse_value = float(raw_rmse)
            if not math.isfinite(sse_value) or sse_value < 0.0 or not math.isfinite(rmse_value) or rmse_value < 0.0:
                msg = f"Normalized evidence for field {field_name!r}, row {row_index} must be finite and non-negative."
                raise ValueError(msg)
            expected_case_rmse = math.sqrt(sse_value / int(raw_count))
            if not math.isclose(
                rmse_value,
                expected_case_rmse,
                rel_tol=NORMALIZED_OBJECTIVE_TOLERANCE["rtol"],
                abs_tol=NORMALIZED_OBJECTIVE_TOLERANCE["atol"],
            ):
                msg = f"{rmse_column} row {row_index} does not match its SSE/count evidence."
                raise ValueError(msg)
            squared_error_sum += sse_value
            element_count += int(raw_count)
        if element_count <= 0:
            msg = f"Artifact macro RMSE cannot finalize field {field_name!r} without elements."
            raise RuntimeError(msg)
        field_rmse = math.sqrt(squared_error_sum / element_count)
        field_summary[field_name] = {
            "normalized_squared_error_sum": squared_error_sum,
            "normalized_element_count": element_count,
            "normalized_rmse": field_rmse,
        }
        field_rmse_values.append(field_rmse)

    value = float(np.mean(np.asarray(field_rmse_values, dtype=np.float64)))
    if not math.isfinite(value):
        msg = "Artifact normalized_macro_rmse finalized to a non-finite value."
        raise FloatingPointError(msg)
    return {
        "objective_id": "normalized_macro_rmse",
        "reduction": "field_macro_element_mean",
        "space": "normalized",
        "fields": list(fields),
        "direction": "minimize",
        "value": value,
        "field_statistics": field_summary,
        "agreement_tolerance": dict(NORMALIZED_OBJECTIVE_TOLERANCE),
    }


def _provenance_with_aggregate(
    provenance: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    output_fields: Iterable[str],
) -> dict[str, Any]:
    """
    Add the authoritative aggregate to caller-owned generation provenance.

    Caller-supplied aggregate results are rejected so only recomputed Parquet
    SSE/count evidence can determine the published primary objective.
    """
    payload = dict(provenance)
    if "aggregate" in payload:
        msg = "Caller-provided artifact provenance cannot define aggregate results."
        raise ValueError(msg)
    payload["aggregate"] = aggregate_normalized_macro_rmse(frame, output_fields=output_fields)
    return payload


# =============================================================================
# JSON / type normalisation utilities
# =============================================================================


def meta_to_jsonable(obj: Any) -> Any:
    """
    Recursively convert supported tensor and NumPy values to JSON-native values.

    Parameters
    ----------
    obj : Any
        Tensor, NumPy value, mapping, list/tuple, or already JSON-native leaf.

    Returns
    -------
    Any
        Tensors and arrays become scalars/lists, NumPy scalars become Python
        scalars, and nested dictionaries/sequences are converted recursively.

    Notes
    -----
    Unrecognized leaf objects pass through unchanged; callers that require JSON
    serialization must still supply JSON-compatible custom leaves.

    """
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return float(arr) if arr.ndim == 0 else arr.tolist()

    if isinstance(obj, np.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


def _value_has_batch_axis(value: Any, batch_size: int) -> bool:
    """Return whether a collated metadata value exposes the leading batch axis."""
    if isinstance(value, torch.Tensor | np.ndarray):
        return value.ndim > 0 and value.shape[0] == batch_size
    if isinstance(value, Mapping):
        return bool(value) and all(_value_has_batch_axis(item, batch_size) for item in value.values())
    return False


def _select_collated_case(value: Any, *, case_offset: int, batch_size: int) -> Any:
    """
    Slice one case from nested collated metadata while preserving batch rank.

    Tensor/array values with the declared leading batch size retain a size-one
    axis. Mappings and nested sequences recurse; non-batched values pass through
    unchanged so dataset metadata structure is preserved beside ``x`` and ``y``.
    """
    if isinstance(value, torch.Tensor | np.ndarray):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[case_offset : case_offset + 1]
        return value
    if isinstance(value, Mapping):
        return {key: _select_collated_case(item, case_offset=case_offset, batch_size=batch_size) for key, item in value.items()}
    if isinstance(value, list | tuple):
        if value and all(_value_has_batch_axis(item, batch_size) for item in value):
            items = [_select_collated_case(item, case_offset=case_offset, batch_size=batch_size) for item in value]
            return tuple(items) if isinstance(value, tuple) else items
        if len(value) == batch_size:
            return value[case_offset : case_offset + 1]
        items = [_select_collated_case(item, case_offset=case_offset, batch_size=batch_size) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _iter_artifact_cases(
    loader: Iterable[Mapping[str, Any]],
    *,
    max_cases: int | None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """
    Yield deterministic size-one cases from arbitrary positive loader batches.

    Nested collated metadata is sliced with the same leading-axis semantics as
    ``x`` and ``y``. The returned position is contiguous in loader order, while
    saved ``source_index`` and ``split_local_index`` values remain payload data
    for the caller to validate.

    Parameters
    ----------
    loader : Iterable[Mapping[str, Any]]
        Ordered batches containing compatible tensor ``x`` and ``y`` entries.
    max_cases : int | None
        Optional exclusive bound on yielded cases.

    Yields
    ------
    tuple[int, dict[str, Any]]
        Contiguous artifact position and one-sample batch mapping.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        If a batch is not a mapping, lacks compatible tensors, or is empty.

    """
    case_position = 0
    for raw_batch in loader:
        if not isinstance(raw_batch, Mapping):
            msg = f"Artifact loader batches must be mappings, got {type(raw_batch).__name__}."
            raise TypeError(msg)
        inputs = raw_batch.get("x")
        targets = raw_batch.get("y")
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            msg = "Artifact loader batches must contain tensor keys 'x' and 'y'."
            raise TypeError(msg)
        if inputs.ndim < _MIN_BATCH_TENSOR_RANK or targets.ndim < _MIN_BATCH_TENSOR_RANK or inputs.shape[0] != targets.shape[0]:
            msg = f"Artifact input/target batches must share a non-empty leading axis, got {tuple(inputs.shape)} and {tuple(targets.shape)}."
            raise ValueError(msg)
        batch_size = int(inputs.shape[0])
        if batch_size <= 0:
            msg = "Artifact loader produced an empty batch."
            raise RuntimeError(msg)
        for case_offset in range(batch_size):
            if max_cases is not None and case_position >= max_cases:
                return
            case = {
                "x": inputs[case_offset : case_offset + 1],
                "y": targets[case_offset : case_offset + 1],
                "source_index": _select_collated_case(
                    raw_batch.get("source_index"),
                    case_offset=case_offset,
                    batch_size=batch_size,
                ),
                "split_local_index": _select_collated_case(
                    raw_batch.get("split_local_index"),
                    case_offset=case_offset,
                    batch_size=batch_size,
                ),
                "meta": _select_collated_case(
                    raw_batch.get("meta", {}),
                    case_offset=case_offset,
                    batch_size=batch_size,
                ),
            }
            yield case_position, case
            case_position += 1


def ordered_indices_sha256(indices: Iterable[int]) -> str:
    """
    Return the canonical digest for an ordered integer membership.

    Parameters
    ----------
    indices : Iterable[int]
        Source indices in saved membership order; order and duplicates, if any,
        participate in the canonical compact-JSON byte representation.

    Returns
    -------
    str
        Lowercase SHA-256 digest of that exact ordered representation.

    """
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_provenance_path(save_root: str | Path) -> Path:
    """Return the versioned provenance sidecar path for an artifact root."""
    return Path(save_root) / ARTIFACT_PROVENANCE_FILENAME


def artifact_output_manifest(save_root: str | Path) -> dict[str, Any]:
    """
    Build the exact digest manifest for one complete artifact payload.

    Parameters
    ----------
    save_root : str | pathlib.Path
        Artifact target containing exactly one Parquet table and a non-empty
        ``npz`` directory.

    Returns
    -------
    dict[str, Any]
        Artifact-relative paths and SHA-256 digests in deterministic name order.

    Raises
    ------
    RuntimeError
        If the target lacks exactly one Parquet file or any NPZ case payload.

    """
    root = Path(save_root)
    parquet_files = sorted(root.glob("*.parquet"))
    npz_files = sorted((root / "npz").glob("*.npz"))
    if len(parquet_files) != 1:
        msg = f"Artifact payload must contain exactly one Parquet file, found {len(parquet_files)} in {root}."
        raise RuntimeError(msg)
    if not npz_files:
        msg = f"Artifact payload contains no NPZ files: {root / 'npz'}"
        raise RuntimeError(msg)

    def entry(payload_path: Path) -> dict[str, Any]:
        """Bind one artifact-relative payload path to its complete-file digest."""
        return {
            "path": payload_path.relative_to(root).as_posix(),
            "sha256": common.serialization.file_sha256(payload_path),
        }

    return {
        "parquet": entry(parquet_files[0]),
        "npz": [entry(payload_path) for payload_path in npz_files],
    }


def write_artifact_provenance(save_root: str | Path, provenance: Mapping[str, Any]) -> Path:
    """
    Atomically publish provenance and payload digests as the completion marker.

    Parameters
    ----------
    save_root : str | pathlib.Path
        Fully written staged artifact target.
    provenance : Mapping[str, Any]
        Scientific identity to enrich with exact output digests.

    Returns
    -------
    pathlib.Path
        Newly published ``artifact_provenance.json`` path.

    Raises
    ------
    FileExistsError
        If a completion marker already exists; provenance is never overwritten.
    RuntimeError
        If the artifact payload is incomplete.

    Notes
    -----
    Publishing this sidecar last is the cache-completion transaction boundary.

    """
    provenance_path = artifact_provenance_path(save_root)
    if provenance_path.exists():
        msg = f"Refusing to overwrite existing artifact provenance: {provenance_path}"
        raise FileExistsError(msg)

    payload = meta_to_jsonable(dict(provenance))
    if not isinstance(payload, dict):
        msg = "Artifact provenance must normalise to a JSON object."
        raise TypeError(msg)
    if "outputs" in payload:
        msg = "Caller-provided artifact provenance cannot define output digests."
        raise ValueError(msg)
    payload["outputs"] = artifact_output_manifest(save_root)

    return common.serialization.atomic_write_json(provenance_path, payload)


def _require_batch_scalar_int(batch: Mapping[str, Any], key: str) -> int:
    """
    Extract one exact integer identity from a size-one artifact case mapping.

    Torch, NumPy, singleton-sequence, and native integer encodings are accepted;
    booleans, floating values, missing keys, and multi-value containers fail so
    persisted membership identity is never coerced ambiguously.
    """
    if key not in batch:
        msg = f"Artifact batches must provide top-level {key!r} identity."
        raise KeyError(msg)

    value = batch[key]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            msg = f"Artifact batch {key!r} must contain exactly one integer value; got {value!r}."
            raise TypeError(msg)
        return int(value.detach().cpu().item())

    if isinstance(value, np.ndarray):
        if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
            msg = f"Artifact batch {key!r} must contain exactly one integer value; got {value!r}."
            raise TypeError(msg)
        return int(value.reshape(-1)[0])

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            msg = f"Artifact batch {key!r} must contain exactly one integer value; got {value!r}."
            raise TypeError(msg)
        value = value[0]

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"Artifact batch {key!r} must be an integer; got {type(value).__name__}."
        raise TypeError(msg)
    return int(value)


def _require_finite_artifact_tensor(value: torch.Tensor, *, label: str) -> None:
    """Reject non-floating or non-finite tensors before artifact publication."""
    if not value.is_floating_point() or value.is_complex():
        msg = f"{label} must be a real floating-point tensor."
        raise TypeError(msg)
    if not torch.isfinite(value).all():
        msg = f"{label} contains non-finite values."
        raise FloatingPointError(msg)


def _artifact_effective_case_count(provenance: Mapping[str, Any]) -> int:
    """
    Read the positive generated-row contract from provenance selection evidence.

    Missing mappings, booleans, zero, and negative counts fail before any output
    directory is populated.
    """
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping):
        msg = "Artifact provenance must contain a selection mapping."
        raise TypeError(msg)

    count = selection.get("effective_case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        msg = "Artifact provenance selection.effective_case_count must be a positive integer."
        raise ValueError(msg)
    return count


def _validate_generated_source_indices(provenance: Mapping[str, Any], source_indices: list[int]) -> None:
    """
    Prove generated loader order matches the provenance membership digest.

    The complete ordered source-index sequence participates in the digest, so
    reordering, omission, substitution, or duplication fails before provenance
    can publish the target as complete.
    """
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping):
        msg = "Artifact provenance must contain a selection mapping."
        raise TypeError(msg)
    expected_digest = selection.get("effective_ordered_source_indices_sha256")
    if not isinstance(expected_digest, str) or not expected_digest:
        msg = "Artifact provenance must contain an effective ordered source-index digest."
        raise TypeError(msg)
    actual_digest = ordered_indices_sha256(source_indices)
    if actual_digest != expected_digest:
        msg = f"Generated ordered source_index values do not match artifact provenance: expected digest {expected_digest}, got {actual_digest}."
        raise RuntimeError(msg)


def _ensure_artifact_targets_absent(save_root: Path, dataset_name: str) -> None:
    """
    Enforce create-only semantics for one staged artifact target.

    Existing Parquet, provenance, NPZ, or interrupted temporary files are all
    reported together and rejected; generation never overwrites partial output.
    """
    npz_dir = save_root / "npz"
    candidates = [
        save_root / f"{dataset_name}.parquet",
        artifact_provenance_path(save_root),
        *sorted(npz_dir.glob("*.npz")),
        *sorted(save_root.glob(".*.tmp")),
        *sorted(npz_dir.glob(".*.tmp")),
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        msg = f"Refusing to overwrite existing or interrupted artifacts:\n{formatted}"
        raise FileExistsError(msg)


# =============================================================================
# Kappa field utilities (fields only, no scalar statistics)
# =============================================================================


def detect_kappa_channels_from_inputs(include_inputs: list[str]) -> list[str]:
    """
    Detect permeability-related input channels based on their names.

    Parameters
    ----------
    include_inputs : list[str]
        List of canonical input channel names.

    Returns
    -------
    list[str]
        Names of all channels that represent permeability components.

    """
    return [name for name in include_inputs if name in INTERNAL_KAPPA_NAMES]


def extract_kappa(
    x_tensor: torch.Tensor,
    *,
    input_fields: list[str],
    kappa_names: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract stored and physical permeability fields from a BCHW input tensor.

    Parameters
    ----------
    x_tensor : torch.Tensor
        Input tensor with shape ``(batch, input_field, y, x)``.
    input_fields : list[str]
        Canonical input-channel names aligned with the tensor channel axis.
    kappa_names : list[str]
        Ordered permeability components to extract. An empty list returns
        zero-channel tensors with the original batch and spatial dimensions.

    Returns
    -------
    dict[str, torch.Tensor]
        ``kappa_encoded`` retains stored task representations; ``kappa`` contains
        physical square-metre components in the same channel order.

    Raises
    ------
    KeyError
        If requested permeability names are absent or physical reconstruction
        lacks required ``kxx``/``kyy`` diagonal components.

    Notes
    -----
    Diagonals are reconstructed as ``10**stored``. ``kxy`` is its stored
    dimensionless ratio times ``sqrt(kxx * kyy)``.

    """
    if not kappa_names:
        return {
            "kappa_encoded": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
            "kappa": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
        }

    index_map = {name: i for i, name in enumerate(input_fields)}
    kappa_indices = [index_map[name] for name in kappa_names]

    # Task-encoded permeability representations as stored in the dataset.
    kappa_encoded = x_tensor[:, kappa_indices, :, :]

    # Physical permeability reconstruction in square metres.
    kappa_phys = torch.zeros_like(kappa_encoded)
    name_to_pos = {name: i for i, name in enumerate(kappa_names)}

    # kxx, kyy (always log10-physical)
    kxx = torch.pow(10.0, kappa_encoded[:, name_to_pos["kxx"]])
    kyy = torch.pow(10.0, kappa_encoded[:, name_to_pos["kyy"]])
    kappa_phys[:, name_to_pos["kxx"]] = kxx
    kappa_phys[:, name_to_pos["kyy"]] = kyy

    # kxy is a dimensionless ratio to sqrt(kxx * kyy).
    if "kxy" in name_to_pos:
        kxy_ratio = kappa_encoded[:, name_to_pos["kxy"]]
        kappa_phys[:, name_to_pos["kxy"]] = kxy_ratio * torch.sqrt(kxx * kyy)

    return {
        "kappa_encoded": kappa_encoded,
        "kappa": kappa_phys,
    }


# =============================================================================
# Run-directory utilities
# =============================================================================


def infer_current_run_dir(save_root: Path) -> Path:
    """
    Find the nearest current-contract run ancestor of an artifact path.

    Parameters
    ----------
    save_root : pathlib.Path
        Artifact or staging path from which to walk toward the filesystem root.

    Returns
    -------
    pathlib.Path
        Nearest current run ancestor, or the original path when none is found.

    Notes
    -----
    The result is used only for diagnostic metadata and logging. Model, loader,
    processor, and device ownership remain explicit caller inputs.

    """
    candidate = Path(save_root)
    while candidate.parent != candidate:
        if common.paths.is_current_run_dir(candidate):
            return candidate
        candidate = candidate.parent
    return Path(save_root)


# =============================================================================
# Main artifact generator
# =============================================================================


def _timing_case_ids(
    *,
    timing_cases: list[dict[str, Any]] | None,
    timing_case_ids: Sequence[str] | None,
    expected_case_count: int,
) -> tuple[str, ...] | None:
    """Validate the optional direct-forward collector and authoritative IDs."""
    if timing_cases is None and timing_case_ids is None:
        return None
    if timing_cases is None or timing_case_ids is None:
        msg = "Artifact timing requires both a collector and authoritative case IDs."
        raise ValueError(msg)
    if timing_cases:
        msg = "Artifact timing collector must be empty at generation start."
        raise ValueError(msg)
    case_ids = tuple(timing_case_ids)
    if len(case_ids) != expected_case_count or len(case_ids) != len(set(case_ids)):
        msg = "Artifact timing case IDs must exactly match unique effective membership."
        raise ValueError(msg)
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        msg = "Artifact timing case IDs must be non-empty strings."
        raise TypeError(msg)
    return case_ids


def _generate_steady_flow_artifacts(  # noqa: PLR0915
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    max_cases: int | None = None,
    publication_root: str | Path | None = None,
    timing_cases: list[dict[str, Any]] | None = None,
    timing_case_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate steady-Brinkman evaluation artifacts from saved-membership batches.

    Each case is inferred in normalized space, inverse-transformed to physical
    fields, and evaluated for channel-balanced relative errors, normalized
    sufficient statistics, dual continuity, momentum, and pressure-boundary
    diagnostics. One NPZ is written per case and one Parquet row per membership.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated steady-flow task owning exact channel names, units, and physics.
    model : Any
        Reconstructed best-checkpoint neural operator.
    loader : Iterable[Mapping[str, Any]]
        Deterministic saved-split batches with explicit source/local identity.
    processor : Any
        Restored training input/output normalizers.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Empty exact target or staging directory.
    dataset_name : str
        Logical dataset name used for the Parquet filename.
    provenance : Mapping[str, Any]
        Versioned request evidence, including selected training continuity and
        exact effective membership digest.
    max_cases : int | None, optional
        Positive case bound counted across arbitrary positive batch sizes.
    publication_root : str | pathlib.Path | None, optional
        Final target recorded in Parquet NPZ paths while writing a sibling stage.
    timing_cases : list[dict[str, Any]] | None, optional
        Empty collector populated with direct per-case forward durations.
    timing_case_ids : Sequence[str] | None, optional
        Authoritative case IDs aligned with effective saved membership.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Per-case table and atomically written Parquet payload path.

    Raises
    ------
    KeyError, TypeError, ValueError, RuntimeError, FloatingPointError
        If task/provenance semantics, batch identity, shapes, finite values,
        physics fields, metrics, or effective membership violate the contract.
    FileExistsError
        If any complete or interrupted target payload already exists.

    Notes
    -----
    Full-grid residual arrays coexist with scalar gradient diagnostics cropped by
    ``EVAL_PAD``. Provenance, including exact output digests and the aggregate
    normalized macro RMSE, publishes only after every payload succeeds.

    """
    save_root = Path(save_root)
    published_root = save_root if publication_root is None else Path(publication_root)
    expected_case_count = _artifact_effective_case_count(provenance)
    authoritative_timing_ids = _timing_case_ids(
        timing_cases=timing_cases,
        timing_case_ids=timing_case_ids,
        expected_case_count=expected_case_count,
    )
    _ensure_artifact_targets_absent(save_root, dataset_name)
    model.eval()

    # Infer run_dir from save_root for logging/metadata only.
    run_dir = infer_current_run_dir(save_root)
    run_name = run_dir.name
    physics_provenance = provenance.get("physics")
    if not isinstance(physics_provenance, Mapping):
        msg = "Steady-flow artifact provenance must contain a physics mapping."
        raise TypeError(msg)
    raw_training_continuity = physics_provenance.get("selected_training_continuity")
    if not isinstance(raw_training_continuity, str):
        msg = "Steady-flow physics provenance selected_training_continuity must be a string."
        raise TypeError(msg)
    selected_training_continuity = domain.physics.brinkman.validate_continuity_kind(raw_training_continuity)
    physics_variant = f"dual-continuity-{ARTIFACT_DERIVATIVE_KIND}-{ARTIFACT_DERIVATIVE_EXTENSION}"

    print(
        "[ARTIFACTS]",
        f"save_root={save_root}",
        f"run_dir={run_dir}",
        f"run_name={run_name}",
        f"variant={physics_variant}",
        sep="\n  - ",
    )

    # --------------------------------------------------
    # Build domain-owned residual diagnostics
    # --------------------------------------------------
    physics_evaluator = domain.physics.brinkman.resolve_physics_evaluator(task.physics.kind)
    derivative_operator = domain.physics.derivatives.build_derivative_operator(
        ARTIFACT_DERIVATIVE_KIND,
        extension=ARTIFACT_DERIVATIVE_EXTENSION,
    )
    print(
        f"[ARTIFACTS] Using domain physics diagnostics kind={task.physics.kind} "
        f"derivatives={ARTIFACT_DERIVATIVE_KIND}/{ARTIFACT_DERIVATIVE_EXTENSION} pad={EVAL_PAD}"
    )

    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    generated_source_indices: list[int] = []

    # Detect available kappa channels from field contracts
    kappa_names = detect_kappa_channels_from_inputs(list(task.input_names))

    for idx, batch in _iter_artifact_cases(loader, max_cases=max_cases):
        split_local_index = _require_batch_scalar_int(batch, "split_local_index")
        source_index = _require_batch_scalar_int(batch, "source_index")
        if split_local_index != idx:
            msg = f"Artifact loader order does not match saved split-local identity: iteration={idx}, split_local_index={split_local_index}."
            raise RuntimeError(msg)
        if source_index < 0:
            msg = f"Artifact source_index must be non-negative, got {source_index}."
            raise ValueError(msg)
        case_id = source_index + 1
        generated_source_indices.append(source_index)

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        _require_finite_artifact_tensor(x, label=f"Artifact case {case_id} inputs")
        _require_finite_artifact_tensor(y, label=f"Artifact case {case_id} targets")

        # Preserve generator metadata in a JSON-safe form.
        source_meta = meta_to_jsonable(batch.get("meta", {}))
        meta_clean = dict(source_meta) if isinstance(source_meta, dict) else {"source_meta": source_meta}
        reserved_meta_keys = {"case_index", "source_index", "split_local_index"}.intersection(meta_clean)
        if reserved_meta_keys:
            msg = f"Source metadata contains reserved artifact identity keys and cannot be preserved unambiguously: {sorted(reserved_meta_keys)}."
            raise KeyError(msg)
        # Pressure boundary condition (stored for diagnostics)
        p_bc_idx = task.input_names.index("p_bc")
        p_bc = x[:, p_bc_idx : p_bc_idx + 1].detach().cpu()

        # Permeability fields (no scalar stats here)
        kappa_info = extract_kappa(
            x,
            input_fields=list(task.input_names),
            kappa_names=kappa_names,
        )

        # --------------------------------------------------
        # Forward pass (model operates in normalized space)
        # --------------------------------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda" and start_time is not None:
            start_time.record(torch.cuda.current_stream())

        with torch.inference_mode():
            x_norm = processor.in_normalizer.transform(x)
            y_hat_norm, forward_s = timing.measure_forward(
                model=model,
                normalized_inputs=x_norm,
                device=device,
            )
            y_norm = processor.out_normalizer.transform(y)
            y_hat = processor.out_normalizer.inverse_transform(y_hat_norm)

        if device.type == "cuda" and end_time is not None:
            end_time.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            inference_time_ms = start_time.elapsed_time(end_time) if start_time is not None else None
        else:
            inference_time_ms = None

        _require_finite_artifact_tensor(x_norm, label=f"Artifact case {case_id} normalized inputs")
        _require_finite_artifact_tensor(y_hat_norm, label=f"Artifact case {case_id} normalized prediction")
        _require_finite_artifact_tensor(y_norm, label=f"Artifact case {case_id} normalized target")
        _require_finite_artifact_tensor(y_hat, label=f"Artifact case {case_id} physical prediction")
        if y_hat.shape != y.shape:
            msg = f"Artifact prediction/target shape mismatch: {tuple(y_hat.shape)} != {tuple(y.shape)}."
            raise RuntimeError(msg)

        # --------------------------------------------------
        # Physics diagnostics (exact training-consistent implementation)
        # --------------------------------------------------
        with torch.no_grad():
            diag = physics_evaluator(
                x,
                y_hat,
                input_fields=task.input_names,
                output_fields=task.output_names,
                derivatives=derivative_operator,
                continuity=selected_training_continuity,
                boundary=task.physics.boundary,
                interior_crop=EVAL_PAD,
            ).as_dict()
        for diagnostic_name, diagnostic_value in diag.items():
            _require_finite_artifact_tensor(
                diagnostic_value,
                label=f"Artifact case {case_id} physics diagnostic {diagnostic_name}",
            )

        momentum_residual_mse = float(diag["momentum_residual_mse"].detach().cpu().item())
        div_velocity_mse = float(diag["div_velocity_mse"].detach().cpu().item())
        div_eps_velocity_mse = float(diag["div_eps_velocity_mse"].detach().cpu().item())
        pressure_boundary_mse = float(diag["pressure_boundary_mse"].detach().cpu().item())
        pressure_inlet_mse = float(diag["pressure_inlet_mse"].detach().cpu().item())
        pressure_outlet_mean_square = float(diag["pressure_outlet_mean_square"].detach().cpu().item())
        Rx_np = diag["Rx"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Ry_np = diag["Ry"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divu_np = diag["div_u"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divepsu_np = diag["div_eps_u"].detach().cpu().squeeze(0).squeeze(0).numpy()

        # --------------------------------------------------
        # Outputs (de-normalised, physical units)
        # --------------------------------------------------
        p, u, v = y_hat[:, 0:1], y_hat[:, 1:2], y_hat[:, 2:3]
        p_gt, u_gt, v_gt = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        U = torch.sqrt(u**2 + v**2)
        U_gt = torch.sqrt(u_gt**2 + v_gt**2)

        # Book-friendly metrics (physical units)
        rmse_p = torch.sqrt(torch.mean((p - p_gt) ** 2)).item()
        rmse_u = torch.sqrt(torch.mean((u - u_gt) ** 2)).item()
        rmse_v = torch.sqrt(torch.mean((v - v_gt) ** 2)).item()
        rmse_U = torch.sqrt(torch.mean((U - U_gt) ** 2)).item()

        # Full tensors for NPZ export / plotting (includes U)
        y_hat_ext = torch.cat([p, u, v, U], dim=1)
        y_ext = torch.cat([p_gt, u_gt, v_gt, U_gt], dim=1)
        err_ext = y_hat_ext - y_ext

        # Main metric tensors (ONLY p,u,v)
        y_hat_main = torch.cat([p, u, v], dim=1)
        y_main = torch.cat([p_gt, u_gt, v_gt], dim=1)
        err_main = y_hat_main - y_main

        # Grid spacing from coordinate fields (physical)
        idx_x = task.input_names.index("x")
        idx_y = task.input_names.index("y")
        dx = float((x[0, idx_x, 0, 1] - x[0, idx_x, 0, 0]).abs().detach().cpu().item())
        dy = float((x[0, idx_y, 1, 0] - x[0, idx_y, 0, 0]).abs().detach().cpu().item())

        metric_denominator_floor = 1e-12

        # ------------------------------------------------------------------
        # Dimensionless relative L2/H1, normalized independently per field
        # ------------------------------------------------------------------
        rel_l2_per_channel: list[float] = []
        rel_h1_per_channel: list[float] = []

        for c in range(y_main.shape[1]):  # c in {p,u,v}
            e_c = err_main[:, c : c + 1]
            r_c = y_main[:, c : c + 1]

            # Relative L2 per channel (global norm ratio)
            l2_e_c = torch.linalg.norm(e_c)
            l2_r_c = torch.linalg.norm(r_c)
            rel_l2_c = (l2_e_c / (l2_r_c + metric_denominator_floor)).item()
            rel_l2_per_channel.append(float(rel_l2_c))

            # Relative H1 per channel (interior, with gradients)
            de_dy_c, de_dx_c = torch.gradient(e_c, spacing=(dy, dx), dim=(2, 3))
            dr_dy_c, dr_dx_c = torch.gradient(r_c, spacing=(dy, dx), dim=(2, 3))

            if EVAL_PAD > 0:
                e_i = e_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                de_dx_i = de_dx_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                de_dy_i = de_dy_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]

                r_i = r_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                dr_dx_i = dr_dx_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                dr_dy_i = dr_dy_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
            else:
                e_i, de_dx_i, de_dy_i = e_c, de_dx_c, de_dy_c
                r_i, dr_dx_i, dr_dy_i = r_c, dr_dx_c, dr_dy_c

            h1_e_c = torch.sqrt((e_i.pow(2) + de_dx_i.pow(2) + de_dy_i.pow(2)).mean())
            h1_r_c = torch.sqrt((r_i.pow(2) + dr_dx_i.pow(2) + dr_dy_i.pow(2)).mean())

            rel_h1_c = (h1_e_c / (h1_r_c + metric_denominator_floor)).item()
            rel_h1_per_channel.append(float(rel_h1_c))

        rel_l2 = float(np.mean(rel_l2_per_channel))
        rel_h1 = float(np.mean(rel_h1_per_channel))
        normalized_statistics = normalized_case_statistics(
            y_hat_norm,
            y_norm,
            output_fields=task.output_names,
        )[0]
        scalar_metrics = (
            rmse_p,
            rmse_u,
            rmse_v,
            rmse_U,
            rel_l2,
            rel_h1,
            momentum_residual_mse,
            div_velocity_mse,
            div_eps_velocity_mse,
            pressure_boundary_mse,
            pressure_inlet_mse,
            pressure_outlet_mean_square,
            *normalized_statistics.values(),
        )
        if not np.isfinite(np.asarray(scalar_metrics, dtype=float)).all():
            msg = f"Artifact case {case_id} produced non-finite scalar metrics."
            raise FloatingPointError(msg)

        # --------------------------------------------------
        # Write NPZ artifact
        # --------------------------------------------------
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        if npz_path.exists():
            msg = f"Refusing to overwrite an existing NPZ artifact: {npz_path}"
            raise FileExistsError(msg)
        x_raw = x.squeeze(0).detach().cpu().numpy()  # (C_in,H,W)
        y_raw = y.squeeze(0).detach().cpu().numpy()  # (C_out,H,W)

        artifact_fields = (*task.output_names, "U")
        artifact_units = (*(field.unit for field in task.outputs), "m/s")
        npz_payload = {
            "case_index": np.int64(case_id),
            "source_index": np.int64(source_index),
            "split_local_index": np.int64(split_local_index),
            "pred": y_hat_ext.squeeze(0).cpu().numpy(),
            "gt": y_ext.squeeze(0).cpu().numpy(),
            "err": err_ext.squeeze(0).cpu().numpy(),
            "artifact_fields": np.asarray(artifact_fields),
            "artifact_units": np.asarray(artifact_units),
            "kappa_encoded": kappa_info["kappa_encoded"].squeeze(0).cpu().numpy(),
            "kappa": kappa_info["kappa"].squeeze(0).cpu().numpy(),
            "kappa_names": np.asarray(kappa_names),
            "p_bc": p_bc.squeeze(0).numpy(),
            "coordinates": x_raw[[task.input_names.index("x"), task.input_names.index("y")]],
            "meta": json.dumps(meta_clean),
            "x_raw": x_raw,
            "y_raw": y_raw,
            "input_fields": np.asarray(task.input_names),
            "output_fields": np.asarray(task.output_names),
            "output_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "Rx": Rx_np,
            "Ry": Ry_np,
            "div_u": divu_np,
            "div_eps_u": divepsu_np,
        }

        def write_npz(temp_path: Path, content: dict[str, Any] = npz_payload) -> None:
            """Serialize one steady-flow case into the atomic writer's temporary path."""
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, **content)

        common.serialization.atomic_path_write(npz_path, write_npz)

        # --------------------------------------------------
        # Parquet row (scalar metrics + metadata only)
        # --------------------------------------------------
        rows.append(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "task_id": task.id,
                "output_fields": list(task.output_names),
                "output_units": [field.unit for field in task.outputs],
                "inference_time_ms": inference_time_ms,
                "case_index": case_id,
                "source_index": source_index,
                "split_local_index": split_local_index,
                "npz_path": str(published_root / "npz" / npz_path.name),
                "rel_l2": rel_l2,
                "rel_h1": rel_h1,
                "rmse_p": rmse_p,
                "rmse_u": rmse_u,
                "rmse_v": rmse_v,
                "rmse_U": rmse_U,
                "kappa_names": kappa_names,
                "momentum_residual_mse": momentum_residual_mse,
                "div_velocity_mse": div_velocity_mse,
                "div_eps_velocity_mse": div_eps_velocity_mse,
                "pressure_boundary_mse": pressure_boundary_mse,
                "pressure_inlet_mse": pressure_inlet_mse,
                "pressure_outlet_mean_square": pressure_outlet_mean_square,
                **normalized_statistics,
                "meta": json.dumps(meta_clean),
            }
        )
        if authoritative_timing_ids is not None and timing_cases is not None:
            timing_cases.append(
                {
                    "case_id": authoritative_timing_ids[idx],
                    "source_index": source_index,
                    "neural_operator_forward_s": forward_s,
                }
            )

    df = pd.DataFrame(rows)
    if not df.columns.is_unique:
        msg = "Steady-flow artifact table contains duplicate columns."
        raise RuntimeError(msg)
    _validate_generated_source_indices(provenance, generated_source_indices)
    if len(df) != expected_case_count:
        msg = f"Artifact generation produced {len(df)} cases, expected {expected_case_count} from provenance."
        raise RuntimeError(msg)

    parquet_path = save_root / f"{dataset_name}.parquet"
    common.serialization.atomic_path_write(
        parquet_path,
        lambda temp_path: df.to_parquet(temp_path, index=False),
    )
    complete_provenance = _provenance_with_aggregate(
        provenance,
        frame=df,
        output_fields=task.output_names,
    )
    write_artifact_provenance(save_root, complete_provenance)

    return df, parquet_path


def _generate_generic_artifacts(
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    max_cases: int | None = None,
    publication_root: str | Path | None = None,
    timing_cases: list[dict[str, Any]] | None = None,
    timing_case_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate artifacts without assuming concrete output fields or physics.

    Each saved-split case is inverse-transformed into physical units, checked for
    finite task-shaped tensors, and written as one compressed NPZ plus one
    Parquet row. Normalized sufficient statistics preserve exact objective
    aggregation. Provenance publishes last as the cache completion marker.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated task owning ordered input/output fields and units.
    model : Any
        Reconstructed best-checkpoint model.
    loader : Iterable[Mapping[str, Any]]
        Deterministic saved-membership batches with explicit identity fields.
    processor : Any
        Restored training normalizer processor.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Empty exact artifact target or sibling staging directory.
    dataset_name : str
        Logical name used for the single Parquet payload.
    provenance : Mapping[str, Any]
        Request identity and expected membership evidence.
    max_cases : int | None, optional
        Positive case bound counted across batch boundaries.
    publication_root : str | pathlib.Path | None, optional
        Final target persisted in NPZ path columns during staged generation.
    timing_cases : list[dict[str, Any]] | None, optional
        Empty collector populated with direct per-case forward durations.
    timing_case_ids : Sequence[str] | None, optional
        Authoritative case IDs aligned with effective saved membership.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Generated case table and its atomically written Parquet payload path.

    Raises
    ------
    KeyError, TypeError, ValueError, RuntimeError, FloatingPointError
        If loader identity/order, task fields, tensor shapes, metadata, finite
        values, case counts, or selected membership violate the contract.
    FileExistsError
        If complete, partial, or temporary output already occupies the target.

    Notes
    -----
    Relative H1 uses grid-index gradients because a generic task declares no
    coordinate convention. ``publication_root`` records final public NPZ paths
    without weakening validation of the staging tree itself.

    """
    root = Path(save_root)
    published_root = root if publication_root is None else Path(publication_root)
    expected_case_count = _artifact_effective_case_count(provenance)
    authoritative_timing_ids = _timing_case_ids(
        timing_cases=timing_cases,
        timing_case_ids=timing_case_ids,
        expected_case_count=expected_case_count,
    )
    _ensure_artifact_targets_absent(root, dataset_name)
    model.eval()
    npz_dir = root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    generated_source_indices: list[int] = []

    for iteration, batch in _iter_artifact_cases(loader, max_cases=max_cases):
        split_local_index = _require_batch_scalar_int(batch, "split_local_index")
        source_index = _require_batch_scalar_int(batch, "source_index")
        if split_local_index != iteration:
            msg = f"Artifact loader order does not match saved split-local identity: iteration={iteration}, split_local_index={split_local_index}."
            raise RuntimeError(msg)
        if source_index < 0:
            msg = f"Artifact source_index must be non-negative, got {source_index}."
            raise ValueError(msg)
        case_index = source_index + 1
        generated_source_indices.append(source_index)

        source_meta = meta_to_jsonable(batch.get("meta", {}))
        metadata = dict(source_meta) if isinstance(source_meta, dict) else {"source_meta": source_meta}
        reserved = {"case_index", "source_index", "split_local_index"}.intersection(metadata)
        if reserved:
            msg = f"Source metadata contains reserved artifact identity keys: {sorted(reserved)}."
            raise KeyError(msg)

        inputs = batch["x"].to(device)
        targets = batch["y"].to(device)
        _require_finite_artifact_tensor(inputs, label=f"Artifact case {case_index} inputs")
        _require_finite_artifact_tensor(targets, label=f"Artifact case {case_index} targets")
        with torch.inference_mode():
            normalized_inputs = processor.in_normalizer.transform(inputs)
            normalized_prediction, forward_s = timing.measure_forward(
                model=model,
                normalized_inputs=normalized_inputs,
                device=device,
            )
            normalized_target = processor.out_normalizer.transform(targets)
            prediction = processor.out_normalizer.inverse_transform(normalized_prediction)
        _require_finite_artifact_tensor(
            normalized_inputs,
            label=f"Artifact case {case_index} normalized inputs",
        )
        _require_finite_artifact_tensor(
            normalized_prediction,
            label=f"Artifact case {case_index} normalized prediction",
        )
        _require_finite_artifact_tensor(
            normalized_target,
            label=f"Artifact case {case_index} normalized target",
        )
        _require_finite_artifact_tensor(
            prediction,
            label=f"Artifact case {case_index} physical prediction",
        )
        if prediction.shape != targets.shape:
            msg = f"Prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(targets.shape)}."
            raise RuntimeError(msg)
        if prediction.shape[1] != len(task.output_names):
            msg = f"Artifact output channel count {prediction.shape[1]} does not match task fields {list(task.output_names)}."
            raise RuntimeError(msg)

        prediction_cpu = prediction.squeeze(0).detach().cpu()
        target_cpu = targets.squeeze(0).detach().cpu()
        error_cpu = prediction_cpu - target_cpu
        npz_path = npz_dir / f"case_{case_index:04d}.npz"
        payload = {
            "case_index": np.int64(case_index),
            "source_index": np.int64(source_index),
            "split_local_index": np.int64(split_local_index),
            "pred": prediction_cpu.numpy(),
            "gt": target_cpu.numpy(),
            "err": error_cpu.numpy(),
            "artifact_fields": np.asarray(task.output_names),
            "artifact_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "x_raw": inputs.squeeze(0).detach().cpu().numpy(),
            "y_raw": target_cpu.numpy(),
            "input_fields": np.asarray(task.input_names),
            "output_fields": np.asarray(task.output_names),
            "output_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "meta": json.dumps(metadata),
        }

        def write_npz(temp_path: Path, content: dict[str, Any] = payload) -> None:
            """Serialize one generic task case into the atomic writer's temporary path."""
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, **content)

        common.serialization.atomic_path_write(npz_path, write_npz)
        normalized_statistics = normalized_case_statistics(
            normalized_prediction,
            normalized_target,
            output_fields=task.output_names,
        )[0]
        relative_l2_values: list[float] = []
        relative_h1_values: list[float] = []
        denominator_floor = 1e-12
        for channel in range(len(task.output_names)):
            field_error = error_cpu[channel]
            field_target = target_cpu[channel]
            relative_l2_values.append(float(torch.linalg.vector_norm(field_error) / (torch.linalg.vector_norm(field_target) + denominator_floor)))
            error_y, error_x = torch.gradient(field_error, dim=(-2, -1))
            target_y, target_x = torch.gradient(field_target, dim=(-2, -1))
            error_h1 = torch.sqrt((field_error.square() + error_x.square() + error_y.square()).mean())
            target_h1 = torch.sqrt((field_target.square() + target_x.square() + target_y.square()).mean())
            relative_h1_values.append(float(error_h1 / (target_h1 + denominator_floor)))
        row: dict[str, Any] = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "task_id": task.id,
            "output_fields": list(task.output_names),
            "output_units": [field.unit for field in task.outputs],
            "inference_time_ms": None,
            "case_index": case_index,
            "source_index": source_index,
            "split_local_index": split_local_index,
            "npz_path": str(published_root / "npz" / npz_path.name),
            "rel_l2": float(np.mean(relative_l2_values)),
            "rel_h1": float(np.mean(relative_h1_values)),
            "meta": json.dumps(metadata),
            **normalized_statistics,
        }
        for channel, field in enumerate(task.outputs):
            row[f"rmse_{field.name}"] = float(torch.sqrt(torch.mean(error_cpu[channel].square())).item())
        rows.append(row)
        if authoritative_timing_ids is not None and timing_cases is not None:
            timing_cases.append(
                {
                    "case_id": authoritative_timing_ids[iteration],
                    "source_index": source_index,
                    "neural_operator_forward_s": forward_s,
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.columns.is_unique:
        msg = "Generic artifact table contains duplicate columns."
        raise RuntimeError(msg)
    _validate_generated_source_indices(provenance, generated_source_indices)
    if len(frame) != expected_case_count:
        msg = f"Artifact generation produced {len(frame)} cases, expected {expected_case_count}."
        raise RuntimeError(msg)
    parquet_path = root / f"{dataset_name}.parquet"
    common.serialization.atomic_path_write(
        parquet_path,
        lambda temp_path: frame.to_parquet(temp_path, index=False),
    )
    complete_provenance = _provenance_with_aggregate(
        provenance,
        frame=frame,
        output_fields=task.output_names,
    )
    write_artifact_provenance(root, complete_provenance)
    return frame, parquet_path


def generate_artifacts(
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    max_cases: int | None = None,
    publication_root: str | Path | None = None,
    timing_cases: list[dict[str, Any]] | None = None,
    timing_case_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate artifacts through the TaskSpec-driven storage contract.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated task owning ordered input/output fields, units, and physics.
    model : Any
        Reconstructed best-checkpoint model.
    loader : Iterable[Mapping[str, Any]]
        Deterministic saved-split batches with any positive batch size.
    processor : Any
        Restored training normalizer processor.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Exact empty artifact target or staging directory.
    dataset_name : str
        Logical dataset name used for the Parquet filename.
    provenance : Mapping[str, Any]
        Exact cache request identity; generated aggregate and output digests are
        appended only after all case payloads succeed.
    max_cases : int | None, optional
        Positive effective saved-split case limit across batch boundaries.
    publication_root : str | pathlib.Path | None, optional
        Final target recorded in Parquet NPZ paths while payloads are staged.
    timing_cases : list[dict[str, Any]] | None, optional
        Empty collector populated with direct per-case forward durations.
    timing_case_ids : Sequence[str] | None, optional
        Authoritative case IDs aligned with effective saved membership.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Generated table and atomically written Parquet payload path.

    Raises
    ------
    KeyError, FileExistsError, TypeError, ValueError, RuntimeError, FloatingPointError
        If target occupancy, task/request identity, batch membership, tensor
        values, scientific diagnostics, or publication evidence is invalid.

    Notes
    -----
    The maintained steady-flow task uses its dual-continuity diagnostic adapter;
    TaskSpecs without that physics selector use generic named field/unit storage. This function
    creates payloads only: cache locking and target replacement belong to the
    artifact service. Provenance is always the final completion marker.

    """
    dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    if task.id == domain.tasks.steady_flow.STEADY_FLOW.id:
        return _generate_steady_flow_artifacts(
            task=task,
            model=model,
            loader=loader,
            processor=processor,
            device=device,
            save_root=save_root,
            dataset_name=dataset_name,
            provenance=provenance,
            max_cases=max_cases,
            publication_root=publication_root,
            timing_cases=timing_cases,
            timing_case_ids=timing_case_ids,
        )
    return _generate_generic_artifacts(
        task=task,
        model=model,
        loader=loader,
        processor=processor,
        device=device,
        save_root=save_root,
        dataset_name=dataset_name,
        provenance=provenance,
        max_cases=max_cases,
        publication_root=publication_root,
        timing_cases=timing_cases,
        timing_case_ids=timing_case_ids,
    )
