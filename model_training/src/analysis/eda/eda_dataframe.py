"""
===============================================================================
eda_dataframe.py
===============================================================================
Materialize bounded task-aware EDA frames from immutable merged datasets.

Responsibilities:
  - Resolve logical dataset identifiers only below ``DATASET_ROOT``
  - Reuse strict dataset schema, content-identity, and fingerprint validation
  - Preserve TaskSpec field order, units, roles, sample IDs, and source metadata
  - Load an explicit ordered prefix without mutating the stored dataset

Design principles:
  - Input is a current merged dataset paired with its authoritative TaskSpec
  - Output arrays retain stored physical representations and declared field order
  - Prefix selection preserves dataset identity and never mutates stored samples

This module does NOT:
  - Reimplement dataset schema, fingerprint, or content-identity validation
  - Compute statistical or spectral visualizations
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from torch import Tensor

from src import common, datasets

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec


def _validated_case_limit(max_cases: int | None, *, available: int) -> int:
    """Return the requested positive case count capped by dataset size."""
    if max_cases is None:
        return available
    if isinstance(max_cases, bool) or not isinstance(max_cases, int):
        msg = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise TypeError(msg)
    if max_cases <= 0:
        msg = f"max_cases must be positive, got {max_cases}."
        raise ValueError(msg)
    return min(max_cases, available)


def generate_eda_dataframe(
    dataset_name: str,
    *,
    task: TaskSpec,
    dataset_root: str | Path | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Materialize one ordered prefix of a strict task dataset for EDA.

    Parameters
    ----------
    dataset_name : str
        Logical dataset identifier resolved below ``DATASET_ROOT``.
    task : TaskSpec
        Authoritative task defining ordered input/output fields, units, and roles.
    dataset_root : str | pathlib.Path | None, optional
        Explicit independent root. When omitted, resolve ``DATASET_ROOT`` through
        :mod:`common.paths`.
    show_progress : bool, optional
        Display a local progress bar while materializing samples.
    max_cases : int | None, optional
        Positive maximum number of samples from stored identity order.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Physical/stored task-field arrays plus isolated source metadata, and
        human-readable loading messages. Frame attrs retain task contract,
        field units/roles, dataset identity, and loaded/available case counts.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError, RuntimeError
        If path resolution, dataset schema/identity, requested count, or sample
        tensor contracts are invalid.

    Notes
    -----
    Loading is read-only and never changes dataset order or stored samples.

    """
    dataset_path = common.paths.resolve_dataset_path(
        dataset_name,
        dataset_root=dataset_root,
    )
    dataset = datasets.simulation.create_task_dataset(dataset_path, task=task)
    available = len(dataset)
    case_count = _validated_case_limit(max_cases, available=available)

    logs = [
        (f"[INFO] Loading the first {case_count} of {available} samples." if case_count < available else f"[INFO] Loading all {available} samples."),
        f"[INFO] Dataset: {dataset_name!r} for task {task.id!r} from {dataset_path}",
    ]

    selected = tuple(enumerate(dataset.identity.sample_ids[:case_count]))
    iterator: Iterable[tuple[int, str]] = selected
    if show_progress:
        from tqdm.auto import tqdm  # noqa: PLC0415

        iterator = tqdm(selected, desc="Loading cases", unit="case")

    rows: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    for position, sample_id in iterator:
        sample = dataset[position]
        inputs = sample["x"]
        outputs = sample["y"]
        if not isinstance(inputs, Tensor) or not isinstance(outputs, Tensor):
            msg = f"Strict dataset sample {sample_id!r} did not expose tensor x/y values."
            raise TypeError(msg)

        row: dict[str, Any] = {name: inputs[channel].detach().cpu().numpy() for channel, name in enumerate(task.input_names)}
        row.update({name: outputs[channel].detach().cpu().numpy() for channel, name in enumerate(task.output_names)})
        row["meta"] = sample["meta"]
        rows.append(row)
        sample_ids.append(sample_id)

    frame = pd.DataFrame(
        rows,
        index=pd.Index(sample_ids, name="sample_id"),
    )
    frame.attrs["task_id"] = task.id
    frame.attrs["task_contract_digest"] = task.contract_digest
    frame.attrs["field_names"] = (*task.input_names, *task.output_names)
    frame.attrs["field_units"] = {field.name: field.unit for field in (*task.inputs, *task.outputs)}
    frame.attrs["field_roles"] = {field.name: field.role for field in (*task.inputs, *task.outputs)}
    frame.attrs["dataset_identity"] = dataset.identity.as_dict()
    frame.attrs["loaded_case_count"] = case_count
    frame.attrs["available_case_count"] = available
    shapes = {column: getattr(frame[column].iloc[0], "shape", None) for column in frame.columns}
    logs.extend(
        (
            f"[INFO] Final DataFrame contains {len(frame)} samples.",
            f"[INFO] Columns: {', '.join(frame.columns)}",
            f"[INFO] Example shapes: {shapes}",
        )
    )
    return frame, logs
