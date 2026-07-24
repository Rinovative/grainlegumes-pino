"""
Build task-aware exploratory-analysis DataFrames from strict datasets.

The loader resolves immutable merged datasets under ``DATASET_ROOT`` and
delegates all schema, field, shape, identity, and fingerprint validation to the
shared simulation dataset API.
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
    Load one strict task dataset into an exploratory-analysis DataFrame.

    Parameters
    ----------
    dataset_name : str
        Logical dataset identifier under ``DATASET_ROOT``.
    task : TaskSpec
        Authoritative task defining ordered input and output fields.
    dataset_root : str | Path | None, optional
        Explicit dataset root. When omitted, resolve ``DATASET_ROOT`` through
        :mod:`common.paths`.
    show_progress : bool, optional
        Display a progress bar while materializing samples.
    max_cases : int | None, optional
        Positive maximum number of ordered samples to load.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Task-field arrays and isolated source metadata, plus loading messages.

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
    shapes = {column: getattr(frame[column].iloc[0], "shape", None) for column in frame.columns}
    logs.extend(
        (
            f"[INFO] Final DataFrame contains {len(frame)} samples.",
            f"[INFO] Columns: {', '.join(frame.columns)}",
            f"[INFO] Example shapes: {shapes}",
        )
    )
    return frame, logs
