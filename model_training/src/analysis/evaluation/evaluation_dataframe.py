"""
===============================================================================
evaluation_dataframe.py
===============================================================================
Build and compare evaluation DataFrames from the current artifact contract.

Responsibilities:
  - Admit only exact current artifact tables and provenance metadata
  - Preserve exact normalized_group_macro_rmse sufficient-statistic aggregates
  - Flatten source metadata without colliding with authoritative columns
  - Validate task, objective, units, formulas, role, and membership before plots

Design principles:
  - DataFrames carry validated provenance in attrs
  - Plot modules consume this module instead of recreating scientific identities
  - Per-case relative metrics remain secondary to the runtime primary objective
  - Architecture, physics enablement, and training continuity are dimensions

This module does NOT:
  - Generate, repair, or publish artifact payloads
  - Parse per-case NPZ arrays or render scientific visualizations
  - Infer compatibility from run names, paths, or architecture similarity
===============================================================================
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.artifacts import contracts

COMPLETED_RUN_CONFIG_ATTR = "completed_run_config"
PRIMARY_OBJECTIVE_ID = "normalized_group_macro_rmse"
PRIMARY_OBJECTIVE_DEFINITION = {
    "id": PRIMARY_OBJECTIVE_ID,
    "kind": "group_macro_rmse",
    "space": "physical",
    "reduction": "group_macro_element_mean",
    "direction": "minimize",
}
STEADY_PHYSICS_METRICS = (
    "momentum_residual_mse",
    "div_velocity_mse",
    "div_eps_velocity_mse",
    "pressure_boundary_mse",
)
PRESSURE_BOUNDARY_METRICS = (
    "pressure_inlet_mse",
    "pressure_outlet_mean_square",
    "pressure_boundary_mse",
)
_BASE_ARTIFACT_COLUMNS = frozenset(
    {
        "artifact_schema_version",
        "task_id",
        "output_fields",
        "output_units",
        "case_index",
        "source_index",
        "split_local_index",
        "npz_path",
        "meta",
        "inference_time_ms",
    }
)
_MAX_FLATTENED_SEQUENCE_LENGTH = 4
_OUTPUT_GROUP_ENTRY_LENGTH = 2
_STEADY_PHYSICS_COLUMNS = frozenset(
    {
        *STEADY_PHYSICS_METRICS,
        *PRESSURE_BOUNDARY_METRICS,
    }
)


class ComparisonCompatibilityError(ValueError):
    """
    Signal that admitted artifacts cannot answer one shared scientific question.

    Raised by comparison and plot-admission boundaries for incompatible task,
    objective, schema, field/unit, dataset-membership, or physics-formula
    identities. It is distinct from malformed raw value errors, which retain
    ``TypeError`` or ``ValueError`` during individual artifact admission.
    """


def output_group_fields(frame: pd.DataFrame, *, group_id: str) -> tuple[str, ...]:
    """
    Return one validated task-owned output group's declared fields.

    Parameters
    ----------
    frame : pandas.DataFrame
        Admitted evaluation frame carrying canonical output-group attributes.
    group_id : str
        Exact semantic group identifier declared by the authoritative TaskSpec.

    Returns
    -------
    tuple[str, ...]
        Non-empty group fields in task-declared order.

    Raises
    ------
    TypeError
        If the requested identifier is not a non-empty string.
    ComparisonCompatibilityError
        If canonical group evidence is absent, malformed, or ambiguous.

    """
    if not isinstance(group_id, str) or not group_id:
        msg = "Output group identifier must be a non-empty string."
        raise TypeError(msg)
    raw_groups = frame.attrs.get("output_groups")
    if not isinstance(raw_groups, tuple):
        msg = "Evaluation frame does not carry canonical task-owned output groups."
        raise ComparisonCompatibilityError(msg)
    matches: list[tuple[str, ...]] = []
    for item in raw_groups:
        if not isinstance(item, tuple) or len(item) != _OUTPUT_GROUP_ENTRY_LENGTH:
            msg = "Evaluation frame output-group evidence is malformed."
            raise ComparisonCompatibilityError(msg)
        candidate_id, candidate_fields = item
        if (
            not isinstance(candidate_id, str)
            or not isinstance(candidate_fields, tuple)
            or not candidate_fields
            or any(not isinstance(field, str) or not field for field in candidate_fields)
        ):
            msg = "Evaluation frame output-group evidence is malformed."
            raise ComparisonCompatibilityError(msg)
        if candidate_id == group_id:
            matches.append(candidate_fields)
    if len(matches) != 1:
        msg = f"Evaluation frame must declare exactly one output group {group_id!r}."
        raise ComparisonCompatibilityError(msg)
    return matches[0]


def single_output_group_field(frame: pd.DataFrame, *, group_id: str) -> str:
    """Return the sole field of one task-owned scalar output group."""
    fields = output_group_fields(frame, group_id=group_id)
    if len(fields) != 1:
        msg = f"Evaluation output group {group_id!r} must contain exactly one scalar field."
        raise ComparisonCompatibilityError(msg)
    return fields[0]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one mapping or fail at its semantic label."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _string_sequence(
    value: Any,
    *,
    label: str,
    require_unique: bool = False,
) -> tuple[str, ...]:
    """
    Normalize one list/tuple/NumPy sequence to non-empty strings.

    Scalars, blank elements, and malformed containers fail. Callers may request
    duplicate rejection for semantic field-name contracts while units may repeat.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value or any(not isinstance(item, str) or not item for item in value):
        msg = f"{label} must be a non-empty sequence of strings."
        raise TypeError(msg)
    result = tuple(value)
    if require_unique and len(result) != len(set(result)):
        msg = f"{label} contains duplicate names."
        raise ValueError(msg)
    return result


def _validate_metric_values(frame: pd.DataFrame, columns: set[str]) -> None:
    """Require current-schema scalar metrics to be finite and non-negative."""
    for column in sorted(columns):
        values = frame[column].tolist()
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            msg = f"Evaluation artifact metric {column!r} must contain real scalar values."
            raise TypeError(msg)
        numeric = np.asarray(values, dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0.0).any():
            msg = f"Evaluation artifact metric {column!r} must contain finite non-negative values."
            raise ValueError(msg)


def _validate_membership(frame: pd.DataFrame) -> None:
    """
    Prove source, split-local, and case identity columns encode saved membership.

    Values must be exact integers. Split-local positions are contiguous from zero,
    case IDs equal source index plus one, and source membership has no duplicates.
    """
    values = {name: frame[name].tolist() for name in ("source_index", "split_local_index", "case_index")}
    flattened = tuple(value for column in values.values() for value in column)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in flattened):
        msg = "Artifact identity columns must contain integers."
        raise TypeError(msg)
    source = tuple(int(value) for value in values["source_index"])
    local = tuple(int(value) for value in values["split_local_index"])
    cases = tuple(int(value) for value in values["case_index"])
    if local != tuple(range(len(frame))):
        msg = "Artifact split_local_index values must preserve contiguous membership order."
        raise ValueError(msg)
    if cases != tuple(value + 1 for value in source):
        msg = "Artifact case_index values must equal source_index + 1."
        raise ValueError(msg)
    if len(set(source)) != len(source):
        msg = "Artifact source_index membership contains duplicates."
        raise ValueError(msg)


def _declared_group_metric_columns(provenance: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return persisted per-case group metric ids from evaluator provenance."""
    if provenance is None:
        return ()
    evaluator = _mapping(provenance.get("evaluator"), label="provenance.evaluator")
    metrics = evaluator.get("metrics")
    if not isinstance(metrics, list):
        msg = "Artifact evaluator metrics must be a list."
        raise TypeError(msg)
    columns: list[str] = []
    for metric in metrics:
        if not isinstance(metric, Mapping) or metric.get("kind") not in {"group_rmse", "vector_rmse"}:
            continue
        metric_id = metric.get("id")
        if not isinstance(metric_id, str) or not metric_id:
            msg = "Artifact evaluator group metric ids must be non-empty strings."
            raise TypeError(msg)
        columns.append(metric_id)
    if len(columns) != len(set(columns)):
        msg = "Artifact evaluator contains duplicate group metric ids."
        raise ValueError(msg)
    return tuple(columns)


def _validate_artifact_table(
    frame: pd.DataFrame,
    *,
    provenance: Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """
    Admit one non-empty Parquet frame against the closed current contract.

    The validator fixes task/output semantics from the first row, requires them
    on every row, rejects every unexpected column, verifies finite non-negative
    metrics, and proves exact saved membership order.
    """
    if not frame.columns.is_unique:
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].tolist()))
        msg = f"Evaluation DataFrame contains duplicate columns: {duplicates}."
        raise ValueError(msg)
    if frame.empty:
        msg = "Evaluation artifact table must contain at least one case."
        raise ValueError(msg)
    missing_base = sorted(_BASE_ARTIFACT_COLUMNS.difference(frame.columns))
    if missing_base:
        msg = f"Evaluation artifact table is missing required schema columns: {missing_base}."
        raise ValueError(msg)

    schema_values = frame["artifact_schema_version"].tolist()
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in schema_values):
        msg = "artifact_schema_version must contain integer values."
        raise TypeError(msg)
    if {int(value) for value in schema_values} != {contracts.ARTIFACT_SCHEMA_VERSION}:
        msg = f"Evaluation artifact table requires schema version {contracts.ARTIFACT_SCHEMA_VERSION}."
        raise ValueError(msg)

    task_values = frame["task_id"].tolist()
    if any(not isinstance(value, str) or not value for value in task_values) or len(set(task_values)) != 1:
        msg = "Evaluation artifact table must contain one non-empty task_id."
        raise ValueError(msg)
    task_id = str(task_values[0])
    output_fields = _string_sequence(
        frame.iloc[0].loc["output_fields"],
        label="output_fields",
        require_unique=True,
    )
    output_units = _string_sequence(frame.iloc[0].loc["output_units"], label="output_units")
    if len(output_fields) != len(output_units):
        msg = "Evaluation artifact output_fields and output_units lengths differ."
        raise ValueError(msg)
    for row_index in range(len(frame)):
        row_fields = _string_sequence(
            frame.iloc[row_index].loc["output_fields"],
            label=f"row {row_index} output_fields",
            require_unique=True,
        )
        row_units = _string_sequence(
            frame.iloc[row_index].loc["output_units"],
            label=f"row {row_index} output_units",
        )
        if row_fields != output_fields or row_units != output_units:
            msg = "Evaluation artifact output fields or units change across case rows."
            raise ValueError(msg)

    predictive = {"rel_l2", "rel_h1"}
    physical = {column for field in output_fields for column in contracts.physical_statistic_columns(field)}
    normalized = {column for field in output_fields for column in contracts.normalized_statistic_columns(field)}
    group_metrics = set(_declared_group_metric_columns(provenance))
    expected = set(_BASE_ARTIFACT_COLUMNS | predictive | physical | normalized | group_metrics)
    metrics = set(predictive | physical | normalized | group_metrics)
    if task_id == "steady_flow":
        steady = set(_STEADY_PHYSICS_COLUMNS | {"physical_rmse_speed_magnitude", "kappa_names"})
        expected.update(steady)
        metrics.update(_STEADY_PHYSICS_COLUMNS | {"physical_rmse_speed_magnitude"})
    missing = sorted(expected.difference(frame.columns))
    unexpected = sorted(set(frame.columns).difference(expected))
    if missing or unexpected:
        msg = f"Evaluation artifact table schema mismatch: missing={missing}, unexpected={unexpected}."
        raise ValueError(msg)
    _validate_metric_values(frame, metrics)
    _validate_membership(frame)
    return task_id, output_fields, output_units


def _parse_meta(value: Any) -> dict[str, Any]:
    """
    Normalize mapping or JSON-text metadata to one object.

    Invalid JSON, non-object results, and unsupported input types fail rather than
    being stringified into unstable sensitivity columns.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        msg = f"Artifact meta must be a JSON object or mapping, got {type(value).__name__}."
        raise TypeError(msg)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        msg = "Artifact meta must contain valid JSON."
        raise ValueError(msg) from error
    if not isinstance(parsed, dict):
        msg = f"Artifact meta JSON must decode to an object, got {type(parsed).__name__}."
        raise TypeError(msg)
    return parsed


def _to_scalar(value: Any) -> Any:
    """Convert scalar NumPy arrays to native values."""
    if isinstance(value, np.ndarray) and (value.ndim == 0 or value.size == 1):
        return value.item()
    return value


def flatten_meta_scalars(
    obj: Any,
    *,
    prefix: str = "",
    out: dict[str, float | int | bool | str] | None = None,
) -> dict[str, float | int | bool | str]:
    """
    Flatten stable scalar source metadata into underscore-delimited columns.

    Parameters
    ----------
    obj : Any
        Nested mappings, short sequences, NumPy scalars, or scalar leaves.
    prefix : str, optional
        Existing column-name prefix used for recursive calls.
    out : dict[str, float | int | bool | str] | None, optional
        Destination mapping. When supplied, it is mutated and returned.

    Returns
    -------
    dict[str, float | int | bool | str]
        Scalar leaves keyed by flattened paths.

    Notes
    -----
    Singleton sequences unwrap. Sequences of length two through four receive
    numeric suffixes. Longer sequences, non-scalar arrays, unsupported leaves,
    and unprefixed scalar roots are deliberately omitted.

    """
    result = {} if out is None else out
    value = _to_scalar(obj)
    if isinstance(value, Mapping):
        for key, item in value.items():
            new_prefix = f"{prefix}_{key}" if prefix else str(key)
            flatten_meta_scalars(item, prefix=new_prefix, out=result)
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            flatten_meta_scalars(value[0], prefix=prefix, out=result)
        elif len(value) <= _MAX_FLATTENED_SEQUENCE_LENGTH:
            for index, item in enumerate(value):
                flatten_meta_scalars(item, prefix=f"{prefix}_{index}", out=result)
    elif isinstance(value, (int, float, bool, str)) and prefix:
        result[prefix] = value
    return result


def _aggregate_contract(
    provenance: Mapping[str, Any],
    *,
    output_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """Return task groups, raw training scales, and transform floor from provenance."""
    evaluator = _mapping(provenance.get("evaluator"), label="provenance.evaluator")
    groups = contracts.output_group_payload(evaluator.get("output_groups", ()))
    grouped_fields = tuple(field for group in groups for field in group["fields"])
    if grouped_fields != output_fields:
        msg = "Artifact output groups do not partition TaskSpec output fields in order."
        raise ComparisonCompatibilityError(msg)
    normalizer = _mapping(provenance.get("normalizer"), label="provenance.normalizer")
    raw_scales = _mapping(
        normalizer.get("output_standard_deviations"),
        label="provenance.normalizer.output_standard_deviations",
    )
    if set(raw_scales) != set(output_fields):
        msg = "Artifact train standard deviations do not map exactly the output fields."
        raise ComparisonCompatibilityError(msg)
    scales: dict[str, float] = {}
    for field in output_fields:
        value = raw_scales[field]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) <= 0.0:
            msg = f"Artifact train standard deviation for {field!r} must be finite and strictly positive."
            raise ValueError(msg)
        scales[field] = float(value)
    denominator_floor = normalizer.get("denominator_floor")
    if (
        isinstance(denominator_floor, bool)
        or not isinstance(denominator_floor, Real)
        or not np.isfinite(float(denominator_floor))
        or float(denominator_floor) < 0.0
    ):
        msg = "Artifact normalizer denominator floor must be finite and non-negative."
        raise ValueError(msg)
    return groups, scales, float(denominator_floor)


def _validated_provenance(
    provenance: Mapping[str, Any],
    *,
    task_id: str,
    output_fields: tuple[str, ...],
    output_units: tuple[str, ...],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Admit the provenance fields consumed by the public analysis surface.

    Current schema versions, task/field/unit contracts, group objective,
    aggregate evidence, split role, dataset fingerprint, and positive model
    counts must agree with the already validated Parquet table. Scientific
    contradictions raise :class:`ComparisonCompatibilityError`.
    """
    payload = dict(provenance)
    for field, expected in (
        ("provenance_schema_version", contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION),
        ("artifact_schema_version", contracts.ARTIFACT_SCHEMA_VERSION),
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) != expected:
            msg = f"Evaluation provenance requires integer {field}={expected}."
            raise ValueError(msg)

    run = _mapping(payload.get("run"), label="provenance.run")
    evaluator = _mapping(payload.get("evaluator"), label="provenance.evaluator")
    if run.get("task") != task_id:
        msg = "Artifact provenance task identity contradicts the Parquet table."
        raise ComparisonCompatibilityError(msg)
    input_fields = _string_sequence(
        evaluator.get("input_fields"),
        label="provenance.evaluator.input_fields",
        require_unique=True,
    )
    raw_input_units = _mapping(evaluator.get("input_units"), label="provenance.evaluator.input_units")
    input_units = tuple(raw_input_units.get(field) for field in input_fields)
    if any(not isinstance(unit, str) or not unit for unit in input_units):
        msg = "Artifact provenance input units must cover every declared input field."
        raise ComparisonCompatibilityError(msg)
    if _string_sequence(evaluator.get("output_fields"), label="provenance.evaluator.output_fields", require_unique=True) != output_fields:
        msg = "Artifact provenance output fields contradict the Parquet table."
        raise ComparisonCompatibilityError(msg)
    raw_units = _mapping(evaluator.get("output_units"), label="provenance.evaluator.output_units")
    if tuple(raw_units.get(field) for field in output_fields) != output_units:
        msg = "Artifact provenance output units contradict the Parquet table."
        raise ComparisonCompatibilityError(msg)

    objective = _mapping(evaluator.get("objective"), label="provenance.evaluator.objective")
    actual_objective = {key: objective.get(key) for key in PRIMARY_OBJECTIVE_DEFINITION}
    if actual_objective != PRIMARY_OBJECTIVE_DEFINITION:
        msg = f"Artifact primary objective is incompatible: {actual_objective!r}."
        raise ComparisonCompatibilityError(msg)
    objective_fields = objective.get("fields")
    resolved_fields = output_fields if objective_fields == "all" else _string_sequence(objective_fields, label="provenance objective fields")
    if resolved_fields != output_fields:
        msg = "Artifact primary objective fields do not match TaskSpec output semantics."
        raise ComparisonCompatibilityError(msg)
    output_groups, _train_scales, _denominator_floor = _aggregate_contract(
        payload,
        output_fields=output_fields,
    )
    if output_groups != aggregate.get("groups"):
        msg = "Artifact aggregate output groups contradict evaluator provenance."
        raise ComparisonCompatibilityError(msg)

    stored_aggregate = _mapping(payload.get("aggregate"), label="provenance.aggregate")
    if dict(stored_aggregate) != dict(aggregate):
        msg = "Artifact provenance aggregate does not match exact Parquet sufficient statistics."
        raise ComparisonCompatibilityError(msg)

    split_role = payload.get("split_role")
    if split_role not in {"eval", "ood"}:
        msg = f"Artifact split_role must be explicit eval or ood, got {split_role!r}."
        raise ComparisonCompatibilityError(msg)
    selection = _mapping(payload.get("selection"), label="provenance.selection")
    effective_count = selection.get("effective_case_count")
    if isinstance(effective_count, bool) or not isinstance(effective_count, Integral) or int(effective_count) <= 0:
        msg = "Artifact selection effective_case_count must be positive."
        raise TypeError(msg)
    dataset = _mapping(payload.get("dataset"), label="provenance.dataset")
    if not isinstance(dataset.get("fingerprint"), str) or not dataset.get("fingerprint"):
        msg = "Artifact provenance dataset fingerprint must be non-empty."
        raise TypeError(msg)
    if not isinstance(dataset.get("data_contract_digest"), str) or not dataset.get("data_contract_digest"):
        msg = "Artifact provenance dataset data_contract_digest must be non-empty."
        raise TypeError(msg)
    raw_physics = payload.get("physics")
    if raw_physics is not None:
        physics = _mapping(raw_physics, label="provenance.physics")
        residual_schema_version = physics.get("residual_schema_version")
        if (
            isinstance(residual_schema_version, bool)
            or not isinstance(residual_schema_version, Integral)
            or int(residual_schema_version) != contracts.RESIDUAL_SCHEMA_VERSION
        ):
            msg = f"Artifact physics requires integer residual_schema_version={contracts.RESIDUAL_SCHEMA_VERSION}."
            raise ValueError(msg)

    model = _mapping(payload.get("model"), label="provenance.model")
    counts = _mapping(model.get("parameter_counts"), label="provenance.model.parameter_counts")
    for name in ("total", "trainable"):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            msg = f"Exact model parameter count {name!r} must be a positive integer."
            raise TypeError(msg)
    if int(counts["trainable"]) > int(counts["total"]):
        msg = "Trainable parameter count cannot exceed total parameter count."
        raise ValueError(msg)
    return payload


def _apply_contract_attrs(
    frame: pd.DataFrame,
    *,
    task_id: str,
    output_fields: tuple[str, ...],
    output_units: tuple[str, ...],
    aggregate: Mapping[str, Any] | None,
    provenance: Mapping[str, Any] | None,
    artifact_root: str | None,
) -> None:
    """
    Attach the authoritative analysis contract to a DataFrame in place.

    Table-only frames receive schema/task/field attrs and an explicit
    ``provenance_complete=False`` marker. Complete provenance additionally binds
    input fields/units, split role, artifact root, and residual schema after
    validation. No row values are changed here.
    """
    frame.attrs["artifact_schema_version"] = contracts.ARTIFACT_SCHEMA_VERSION
    frame.attrs["task_id"] = task_id
    frame.attrs["output_fields"] = output_fields
    frame.attrs["output_units"] = output_units
    if aggregate is not None:
        frame.attrs[PRIMARY_OBJECTIVE_ID] = dict(aggregate)
    if artifact_root is not None:
        frame.attrs["artifact_root"] = artifact_root
    if provenance is None:
        frame.attrs["provenance_complete"] = False
        return

    if aggregate is None:
        msg = "Complete artifact provenance requires a finalized group objective."
        raise RuntimeError(msg)
    validated = _validated_provenance(
        provenance,
        task_id=task_id,
        output_fields=output_fields,
        output_units=output_units,
        aggregate=aggregate,
    )
    frame.attrs["provenance_complete"] = True
    frame.attrs["artifact_provenance"] = validated
    frame.attrs["provenance_schema_version"] = validated["provenance_schema_version"]
    evaluator = _mapping(validated["evaluator"], label="provenance.evaluator")
    input_fields = _string_sequence(evaluator["input_fields"], label="provenance.evaluator.input_fields", require_unique=True)
    input_units = _mapping(evaluator["input_units"], label="provenance.evaluator.input_units")
    frame.attrs["input_fields"] = input_fields
    frame.attrs["input_units"] = tuple(str(input_units[field]) for field in input_fields)
    output_groups, train_standard_deviations, denominator_floor = _aggregate_contract(
        validated,
        output_fields=output_fields,
    )
    frame.attrs["output_groups"] = tuple((str(group["id"]), tuple(group["fields"])) for group in output_groups)
    frame.attrs["train_standard_deviations"] = dict(train_standard_deviations)
    frame.attrs["normalization_denominator_floor"] = denominator_floor
    frame.attrs["dataset_role"] = validated["split_role"]
    frame.attrs["residual_schema_version"] = (
        validated.get("physics", {}).get("residual_schema_version") if isinstance(validated.get("physics"), Mapping) else None
    )


def build_eval_df(frame_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and expand one raw artifact table through the authoritative path.

    Parameters
    ----------
    frame_raw : pandas.DataFrame
        Non-empty current-schema Parquet table. Optional ``artifact_provenance``
        and ``artifact_root`` attrs are retained only after validation.

    Returns
    -------
    pandas.DataFrame
        A copy without the JSON ``meta`` column, augmented by stable flattened
        scalar metadata and authoritative schema/task/field attrs. Complete
        provenance additionally supplies the finalized objective.

    Raises
    ------
    KeyError, TypeError, ValueError, ComparisonCompatibilityError
        If the closed table schema, membership, metric evidence, metadata names,
        provenance, or aggregate contract is invalid.

    Notes
    -----
    Metadata may not collide with authoritative table columns. The input frame is
    not mutated. Provenance-free inputs remain explicitly table-only.

    """
    provenance = frame_raw.attrs.get("artifact_provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        msg = "DataFrame artifact_provenance attr must be a mapping."
        raise TypeError(msg)
    task_id, output_fields, output_units = _validate_artifact_table(
        frame_raw,
        provenance=provenance,
    )
    aggregate: dict[str, Any] | None = None
    if provenance is not None:
        groups, train_standard_deviations, denominator_floor = _aggregate_contract(
            provenance,
            output_fields=output_fields,
        )
        aggregate = contracts.aggregate_normalized_group_macro_rmse(
            frame_raw,
            output_groups=groups,
            train_standard_deviations=train_standard_deviations,
            normalization_denominator_floor=denominator_floor,
        )
    raw_root = frame_raw.attrs.get("artifact_root")
    artifact_root = str(raw_root) if raw_root is not None else None

    frame = frame_raw.copy()
    meta_features = frame["meta"].apply(lambda value: flatten_meta_scalars(_parse_meta(value)))
    metadata_frame = pd.DataFrame(meta_features.tolist(), index=frame.index)
    collisions = sorted(set(frame.columns).intersection(metadata_frame.columns))
    if collisions:
        msg = f"Artifact metadata collides with authoritative table columns: {collisions}."
        raise ValueError(msg)
    frame = pd.concat([frame.drop(columns=["meta"]), metadata_frame], axis=1)
    if not frame.columns.is_unique:
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].tolist()))
        msg = f"Evaluation DataFrame contains duplicate columns: {duplicates}."
        raise ValueError(msg)
    _apply_contract_attrs(
        frame,
        task_id=task_id,
        output_fields=output_fields,
        output_units=output_units,
        aggregate=aggregate,
        provenance=provenance,
        artifact_root=artifact_root,
    )
    return frame


def load_and_build_eval_df(parquet_path: str | Path) -> pd.DataFrame:
    """
    Load and validate one explicit raw current-schema Parquet table.

    Parameters
    ----------
    parquet_path : str | pathlib.Path
        Exact table path. No sibling provenance file is inferred or scanned.

    Returns
    -------
    pandas.DataFrame
        Table-only evaluation frame with ``provenance_complete=False``.

    Raises
    ------
    OSError, KeyError, TypeError, ValueError
        If Parquet loading or the current table contract fails.

    Notes
    -----
    Use :func:`load_evaluation_artifact` for comparison-ready provenance.

    """
    return build_eval_df(pd.read_parquet(Path(parquet_path)))


def load_evaluation_artifact(artifact_root: str | Path) -> pd.DataFrame:
    """
    Load one explicit artifact root and its declared Parquet payload.

    Parameters
    ----------
    artifact_root : str | pathlib.Path
        Exact artifact directory containing current provenance and one manifest-
        declared Parquet table.

    Returns
    -------
    pandas.DataFrame
        Evaluation frame with validated complete provenance and resolved artifact
        root attrs.

    Raises
    ------
    OSError, TypeError, ValueError, ComparisonCompatibilityError
        If provenance is unreadable, the manifest does not identify one contained
        Parquet file, or table/provenance contracts disagree.

    Notes
    -----
    The function never scans for alternate tables or provenance sidecars.

    """
    root = Path(artifact_root).resolve()
    provenance_path = contracts.artifact_provenance_path(root)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Artifact provenance is unreadable: {provenance_path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(provenance, Mapping):
        msg = "Artifact provenance must contain one JSON object."
        raise TypeError(msg)
    outputs = _mapping(provenance.get("outputs"), label="provenance.outputs")
    parquet = _mapping(outputs.get("parquet"), label="provenance.outputs.parquet")
    parquet_name = parquet.get("path")
    if not isinstance(parquet_name, str) or not parquet_name or Path(parquet_name).suffix != ".parquet":
        msg = "Artifact output manifest must declare one Parquet path."
        raise ValueError(msg)
    parquet_path = (root / parquet_name).resolve()
    if not parquet_path.is_relative_to(root) or not parquet_path.is_file():
        msg = "Declared artifact Parquet payload is missing or escapes its root."
        raise ValueError(msg)
    try:
        computed_outputs = contracts.artifact_output_manifest(root)
    except (OSError, RuntimeError) as error:
        msg = f"Artifact output manifest cannot be recomputed for {root}: {error}"
        raise ValueError(msg) from error
    if dict(outputs) != computed_outputs:
        msg = "Artifact output manifest does not match the current payload files."
        raise ValueError(msg)
    frame_raw = pd.read_parquet(parquet_path)
    frame_raw.attrs["artifact_root"] = str(root)
    frame_raw.attrs["artifact_provenance"] = dict(provenance)
    frame = build_eval_df(frame_raw)
    resolved_paths = [
        contracts.resolve_case_payload_path(
            root,
            stored_path,
            expected_filename=f"case_{int(case_index):04d}.npz",
        )
        for case_index, stored_path in zip(
            frame["case_index"].tolist(),
            frame["npz_path"].tolist(),
            strict=True,
        )
    ]
    frame.loc[:, "npz_path"] = [str(path) for path in resolved_paths]
    return frame


def require_complete_provenance(frame: pd.DataFrame) -> Mapping[str, Any]:
    """
    Return complete current provenance already admitted with an evaluation frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame produced by :func:`build_eval_df` with provenance attrs or by
        :func:`load_evaluation_artifact`.

    Returns
    -------
    Mapping[str, Any]
        Current provenance used for comparison and scientific labels.

    Raises
    ------
    ComparisonCompatibilityError
        If provenance validation was bypassed or frame attrs were lost.

    """
    if frame.attrs.get("provenance_complete") is not True:
        msg = "This analysis requires a DataFrame loaded with complete artifact provenance."
        raise ComparisonCompatibilityError(msg)
    provenance = frame.attrs.get("artifact_provenance")
    if not isinstance(provenance, Mapping):
        msg = "Evaluation DataFrame lost its artifact provenance mapping."
        raise ComparisonCompatibilityError(msg)
    return provenance


def field_units(frame: pd.DataFrame) -> dict[str, str]:
    """
    Return TaskSpec output units in exact declared field order.

    The mapping is reconstructed only from validated frame attributes. Missing
    or misaligned field/unit sequences fail before plots can label mixed units.
    """
    fields = tuple(frame.attrs.get("output_fields", ()))
    units = tuple(frame.attrs.get("output_units", ()))
    if not fields or len(fields) != len(units):
        msg = "Evaluation DataFrame has no valid output field/unit contract."
        raise ValueError(msg)
    return dict(zip(fields, units, strict=True))


def dataset_role(frame: pd.DataFrame) -> str:
    """
    Return ``ID`` or ``OOD`` from the explicit saved split role.

    Unknown roles are disclosed as ``unspecified`` rather than inferred from a
    dataset or run name.
    """
    role = frame.attrs.get("dataset_role")
    if role == "eval":
        return "ID"
    if role == "ood":
        return "OOD"
    return "unspecified"


def _comparison_identity(provenance: Mapping[str, Any], *, physics: bool) -> dict[str, Any]:
    """
    Build the exact role-local identity used for scientific comparison admission.

    Task/objective/schema/field/unit and dataset/effective-membership evidence
    always participate. When ``physics`` is requested, residual equations,
    derivative/crop settings, boundary semantics, and array definitions join the
    identity. Architecture and training continuity deliberately do not.
    """
    evaluator = _mapping(provenance.get("evaluator"), label="provenance.evaluator")
    identity: dict[str, Any] = {
        "task": provenance.get("run", {}).get("task"),
        "task_contract_digest": provenance.get("run", {}).get("task_contract_digest"),
        "artifact_schema_version": provenance.get("artifact_schema_version"),
        "provenance_schema_version": provenance.get("provenance_schema_version"),
        "objective": evaluator.get("objective"),
        "input_fields": evaluator.get("input_fields"),
        "input_units": evaluator.get("input_units"),
        "output_fields": evaluator.get("output_fields"),
        "output_units": evaluator.get("output_units"),
        "predictive_metrics": evaluator.get("predictive_metrics"),
        "split_role": provenance.get("split_role"),
        "dataset_fingerprint": provenance.get("dataset", {}).get("fingerprint"),
        "dataset_data_contract_digest": provenance.get("dataset", {}).get("data_contract_digest"),
        "saved_membership_digest": provenance.get("dataset", {}).get("saved_membership_digest"),
        "effective_membership_digest": provenance.get("selection", {}).get("effective_ordered_source_indices_sha256"),
        "effective_case_count": provenance.get("selection", {}).get("effective_case_count"),
    }
    if physics:
        raw_physics = _mapping(provenance.get("physics"), label="provenance.physics")
        identity["physics"] = {
            "residual_schema_version": raw_physics.get("residual_schema_version"),
            "equation_kind": raw_physics.get("equation_kind"),
            "boundary_condition_kind": raw_physics.get("boundary_condition_kind"),
            "derivatives": raw_physics.get("derivatives"),
            "interior_crop": raw_physics.get("interior_crop"),
            "scalar_definitions": raw_physics.get("scalar_definitions"),
            "array_definitions": raw_physics.get("array_definitions"),
            "residual_evaluation_region": raw_physics.get("residual_evaluation_region"),
        }
    return identity


def validate_comparison(
    datasets: Mapping[str, pd.DataFrame],
    *,
    require_physics: bool = False,
) -> None:
    """
    Reject incompatible artifact groups before plotting or aggregation.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Non-empty labelled frames with complete admitted provenance.
    require_physics : bool, optional
        Also require the maintained steady-flow residual schema and exact physics
        equation, derivative, crop, boundary, and array definitions.

    Raises
    ------
    TypeError, ValueError
        If labels or the dataset mapping are malformed.
    ComparisonCompatibilityError
        If frames lack complete provenance or disagree on required identities.

    Notes
    -----
    ID and OOD roles may coexist. Within each role, dataset fingerprint and saved
    effective membership must match. Across roles, task/objective/schema/field/
    unit semantics still agree. Architecture, hyperparameters, physics enablement,
    and selected training continuity remain comparison dimensions, not blockers.

    """
    if not datasets:
        msg = "At least one evaluation dataset is required."
        raise ValueError(msg)
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for label, frame in datasets.items():
        if not isinstance(label, str) or not label:
            msg = "Evaluation dataset labels must be non-empty strings."
            raise TypeError(msg)
        provenance = require_complete_provenance(frame)
        if require_physics:
            if frame.attrs.get("task_id") != "steady_flow":
                msg = f"Physics plots are unavailable for task {frame.attrs.get('task_id')!r}."
                raise ComparisonCompatibilityError(msg)
            residual_schema_version = frame.attrs.get("residual_schema_version")
            if (
                isinstance(residual_schema_version, bool)
                or not isinstance(residual_schema_version, Integral)
                or int(residual_schema_version) != contracts.RESIDUAL_SCHEMA_VERSION
            ):
                msg = f"Physics plots require integer residual schema {contracts.RESIDUAL_SCHEMA_VERSION}."
                raise ComparisonCompatibilityError(msg)
        role = str(provenance.get("split_role"))
        grouped.setdefault(role, []).append((label, _comparison_identity(provenance, physics=require_physics)))

    reference_global: dict[str, Any] | None = None
    for role, entries in grouped.items():
        first_label, first = entries[0]
        for label, identity in entries[1:]:
            if identity != first:
                msg = (
                    f"Incompatible {role!r} artifacts: {first_label!r} and {label!r} "
                    "differ in task/objective/schema/formulas/units/dataset/membership."
                )
                raise ComparisonCompatibilityError(msg)
        common = {
            key: first[key]
            for key in (
                "task",
                "task_contract_digest",
                "artifact_schema_version",
                "provenance_schema_version",
                "objective",
                "input_fields",
                "input_units",
                "output_fields",
                "output_units",
                "predictive_metrics",
            )
        }
        if reference_global is None:
            reference_global = common
        elif common != reference_global:
            msg = "ID and OOD artifact groups use incompatible task/objective/schema/field/unit contracts."
            raise ComparisonCompatibilityError(msg)


def numeric_metadata_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """
    Return source-metadata columns eligible for sensitivity controls.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluation frame after metadata flattening and contract attrs attachment.

    Returns
    -------
    tuple[str, ...]
        First-seen DataFrame order of non-authoritative columns with at least one
        value coercible to a finite numeric candidate.

    Notes
    -----
    Identity, predictive, physical, boundary, and normalized sufficient-statistic
    columns are excluded. Callers that require complete finite rows must validate
    the returned columns more strictly for their own analysis.

    """
    reserved = set(_BASE_ARTIFACT_COLUMNS)
    reserved.discard("meta")
    fields = tuple(frame.attrs.get("output_fields", ()))
    reserved.update(
        {
            "rel_l2",
            "rel_h1",
            "physical_rmse_speed_magnitude",
            *STEADY_PHYSICS_METRICS,
            *PRESSURE_BOUNDARY_METRICS,
        }
    )
    provenance = frame.attrs.get("artifact_provenance")
    if isinstance(provenance, Mapping):
        reserved.update(_declared_group_metric_columns(provenance))
    for field in fields:
        reserved.update(
            {
                *contracts.physical_statistic_columns(field),
                *contracts.normalized_statistic_columns(field),
            }
        )
    columns: list[str] = []
    for column in frame.columns:
        if column in reserved:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            columns.append(str(column))
    return tuple(columns)
