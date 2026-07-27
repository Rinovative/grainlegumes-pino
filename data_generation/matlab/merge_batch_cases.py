"""
===============================================================================
merge_batch_cases.py
===============================================================================
Merge strict task cases into one content-bound training dataset.

Responsibilities:
  - Read prepared cases from the case-data root in deterministic filename order
  - Validate each task schema, fingerprint, identifier, dtype, and spatial shape
  - Stack task-ordered tensors and atomically publish merged dataset identity

Design principles:
  - TaskSpec alone owns learned fields and channel order
  - Ordered sample membership and source identities participate in the fingerprint
  - Existing merged targets are authoritative and are never replaced implicitly

This module does NOT:
  - Build cases from COMSOL exports or choose a training split
  - Filter fields, repair invalid cases, or provide compatibility for stale schemas
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from src import common, datasets, domain
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path


def merge_batch_cases(
    batch_name: str,
    verbose: bool = False,
    *,
    task_id: str = "steady_flow",
    data_root: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> dict[str, Any]:
    """
    Merge one directory of strict cases using a task-owned field contract.

    Parameters
    ----------
    batch_name : str
        Logical dataset identifier and batch directory name.
    verbose : bool, optional
        Show merge progress and tensor metadata.
    task_id : str, optional
        Exact registered task identifier.
    data_root : Path | str | None, optional
        Explicit case-preparation root containing ``raw/<batch>/cases``.
    dataset_root : Path | str | None, optional
        Explicit merged-dataset root used by training and EDA.

    Returns
    -------
    dict[str, Any]
        Dataset path, verified fingerprint, sample count, task identity, and
        stacked ``(N, C, ...)`` input/output shapes.

    Raises
    ------
    FileExistsError
        If the merged output already exists.
    RuntimeError
        If the case directory contains no canonical case files.
    TypeError
        If a loaded payload violates an exact persisted type contract.
    ValueError
        If any case violates schema, task fields, content identity, filename
        identity, dtype, or shape consistency.

    Notes
    -----
    The merged payload is published through the shared atomic serializer. Case
    files remain owned by the case-data root; this function neither mutates nor
    removes them and never overwrites an existing merged target.

    """
    task = domain.tasks.registry.get_task(task_id)
    case_batch_dir = common.paths.resolve_case_dataset_dir(batch_name, data_root=data_root)
    cases_dir = case_batch_dir / "cases"
    batch_dir = common.paths.resolve_dataset_dir(batch_name, dataset_root=dataset_root)
    destination = common.paths.resolve_dataset_path(batch_name, dataset_root=dataset_root)
    if destination.exists():
        msg = f"Refusing to overwrite existing merged dataset: {destination}"
        raise FileExistsError(msg)

    case_files = sorted(cases_dir.glob("case_*.pt"))
    if not case_files:
        msg = f"No strict case files found in {cases_dir}"
        raise RuntimeError(msg)

    validated_cases: list[datasets.identity.ValidatedCase] = []
    for case_path in tqdm(case_files, desc=f"Merging {batch_name}", unit="file", disable=not verbose):
        payload = torch.load(case_path, map_location="cpu", weights_only=False)
        validated = datasets.identity.validate_case_payload(
            payload,
            task=task,
            verify_content=True,
        )
        if validated.case_id != case_path.stem:
            msg = f"Case filename/sample identity mismatch: {case_path.stem!r} != {validated.case_id!r}."
            raise ValueError(msg)
        validated_cases.append(validated)

    sample_ids = [case.case_id for case in validated_cases]
    if len(sample_ids) != len(set(sample_ids)):
        msg = f"Duplicate case identifiers are not allowed: {sample_ids}."
        raise ValueError(msg)
    input_shapes = {tuple(case.inputs.shape) for case in validated_cases}
    output_shapes = {tuple(case.outputs.shape) for case in validated_cases}
    input_dtypes = {str(case.inputs.dtype) for case in validated_cases}
    output_dtypes = {str(case.outputs.dtype) for case in validated_cases}
    if len(input_shapes) != 1 or len(output_shapes) != 1:
        msg = f"Cases have inconsistent tensor shapes: inputs={sorted(input_shapes)}, outputs={sorted(output_shapes)}."
        raise ValueError(msg)
    if len(input_dtypes) != 1 or len(output_dtypes) != 1:
        msg = f"Cases have inconsistent tensor dtypes: inputs={sorted(input_dtypes)}, outputs={sorted(output_dtypes)}."
        raise ValueError(msg)

    inputs = torch.stack([case.inputs for case in validated_cases], dim=0)
    outputs = torch.stack([case.outputs for case in validated_cases], dim=0)
    merged = datasets.identity.build_merged_dataset_payload(
        task=task,
        dataset_id=batch_name,
        sample_ids=sample_ids,
        source_identities=[case.source_identity for case in validated_cases],
        source_metadata=[case.source_metadata for case in validated_cases],
        case_fingerprints=[case.fingerprint for case in validated_cases],
        inputs=inputs,
        outputs=outputs,
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    common.serialization.atomic_torch_save(merged, destination)

    if verbose:
        print(f"Input fields: {list(task.input_names)}")
        print(f"Output fields: {list(task.output_names)}")
        print(f"Inputs shape: {tuple(inputs.shape)}")
        print(f"Outputs shape: {tuple(outputs.shape)}")
        print(f"Dataset fingerprint: {merged['dataset_fingerprint']}")

    return {
        "batch_name": batch_name,
        "task": task.id,
        "n_cases": len(validated_cases),
        "inputs_shape": tuple(inputs.shape),
        "outputs_shape": tuple(outputs.shape),
        "dataset_path": destination,
        "dataset_fingerprint": merged["dataset_fingerprint"],
    }
