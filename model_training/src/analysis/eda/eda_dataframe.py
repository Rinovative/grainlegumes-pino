"""
===============================================================================
eda_dataframe.py
===============================================================================
Materialize bounded task-aware EDA frames from generated COMSOL batches.

Responsibilities:
  - Load a validated manifest-ordered generated-batch prefix
  - Preserve task field physical units, stored representations, roles, and identities
  - Derive speed magnitude when the task declares compatible velocity outputs

Design principles:
  - EDA reads the generation domain without resolving training datasets
  - Bounded prefixes preserve manifest order and disclose available case counts
  - Task contracts remain authoritative for field names and scientific metadata

This module does NOT:
  - Build final training datasets or fit preprocessing state
  - Read model outputs, checkpoints, or experiment run directories
  - Implement plots or notebook controls. EDA plot and UI modules own presentation
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec


def generate_eda_dataframe(
    batch_name: str,
    *,
    task: TaskSpec,
    generated_data_root: str | Path | None = None,
    show_progress: bool = True,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Materialize one validated manifest-ordered generated-batch prefix.

    The shared generated-source reader enforces the same manifest hashes, unit
    headers, Cartesian grid, finite values, porosity, permeability, transforms,
    and task field ordering as the final dataset builder. No model-training path
    is resolved or opened.
    """
    from data_generation import load_generated_batch  # noqa: PLC0415

    loaded: dict[str, Any] = load_generated_batch(
        batch_name,
        task_id=task.id,
        generated_data_root=generated_data_root,
        show_progress=show_progress,
        max_cases=max_cases,
    )
    loaded_task = loaded["task"]
    if loaded_task.contract_digest != task.contract_digest:
        msg = f"Generated reader task contract does not match requested task {task.id!r}."
        raise ValueError(msg)
    sample_ids = loaded["sample_ids"]
    rows = loaded["rows"]
    available = loaded["available_case_count"]
    frame = pd.DataFrame(rows, index=pd.Index(sample_ids, name="sample_id"))
    declared_fields = (*task.inputs, *task.outputs)
    field_names = tuple(field.name for field in declared_fields)
    field_units = {field.name: field.unit for field in declared_fields}
    field_representations = {field.name: field.representation for field in declared_fields}
    field_roles: dict[str, str] = {field.name: field.role for field in declared_fields}
    if {"u", "v"}.issubset(task.output_names) and field_units["u"] == field_units["v"]:
        speed = [np.hypot(np.asarray(u, dtype=float), np.asarray(v, dtype=float)) for u, v in zip(frame["u"], frame["v"], strict=True)]
        frame["U"] = pd.Series(speed, index=frame.index, dtype=object)
        field_names = (*field_names, "U")
        field_units["U"] = field_units["u"]
        field_representations["U"] = "derived_speed_magnitude"
        field_roles["U"] = "derived_speed"
    frame.attrs["task_id"] = task.id
    frame.attrs["task_contract_digest"] = task.contract_digest
    frame.attrs["field_names"] = field_names
    frame.attrs["field_units"] = field_units
    frame.attrs["field_representations"] = field_representations
    frame.attrs["field_roles"] = field_roles
    frame.attrs["generated_batch_identity"] = loaded["generated_batch_identity"]
    frame.attrs["source_manifest_sha256"] = loaded["manifest_sha256"]
    frame.attrs["loaded_case_count"] = len(sample_ids)
    frame.attrs["available_case_count"] = available
    frame.attrs["spatial_shape"] = None if frame.empty else tuple(frame.iloc[0][task.input_names[0]].shape)
    root = loaded["generated_data_root"]
    loading_scope = f"the first {len(sample_ids)} of {available}" if len(sample_ids) < available else f"all {available}"
    logs = [
        f"[INFO] Loading {loading_scope} samples.",
        f"[INFO] Generated batch: {batch_name!r} for task {task.id!r} from {root}",
        f"[INFO] Final DataFrame contains {len(frame)} samples.",
        f"[INFO] Columns: {', '.join(frame.columns)}",
    ]
    if not frame.empty:
        shapes = {column: getattr(frame[column].iloc[0], "shape", None) for column in frame.columns}
        logs.append(f"[INFO] Example shapes: {shapes}")
    return frame, logs
