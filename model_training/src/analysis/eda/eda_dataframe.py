"""
Materialize bounded task-aware EDA frames from generated COMSOL batches.

EDA owns a generation-domain view: it validates and reads parameter metadata,
raw fields, the terminal manifest, and processed reference solutions without
resolving final training datasets or model outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec


def generate_eda_dataframe(
    batch_name: str,
    *,
    task: TaskSpec,
    generated_data_root: str | Path | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Materialize one validated manifest-ordered generated-batch prefix.

    The shared generated-source reader enforces the same manifest hashes, unit
    headers, Cartesian grid, finite values, porosity, permeability, transforms,
    and task field ordering as the final dataset builder. No model-training path
    is resolved or opened.
    """
    from data_generation import build_training_dataset as generated  # noqa: PLC0415

    loaded: dict[str, Any] = generated.load_generated_batch_for_eda(
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
    frame.attrs["task_id"] = task.id
    frame.attrs["task_contract_digest"] = task.contract_digest
    frame.attrs["field_names"] = (*task.input_names, *task.output_names)
    frame.attrs["field_units"] = {field.name: field.unit for field in (*task.inputs, *task.outputs)}
    frame.attrs["field_roles"] = {field.name: field.role for field in (*task.inputs, *task.outputs)}
    frame.attrs["generated_batch_identity"] = loaded["generated_batch_identity"]
    frame.attrs["source_manifest_sha256"] = loaded["manifest_sha256"]
    frame.attrs["loaded_case_count"] = len(sample_ids)
    frame.attrs["available_case_count"] = available
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
