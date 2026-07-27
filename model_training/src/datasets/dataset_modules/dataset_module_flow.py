"""
===============================================================================
dataset_module_flow.py
===============================================================================
Convert validated task datasets into neural-operator tensors.

Responsibilities:
  - Validate current merged-dataset and single-case payloads
  - Preserve the exact task-owned input and output channel order
  - Populate sample dictionaries with model-ready tensors

Design principles:
  - No field filtering, aliasing, reordering, or schema coercion is allowed
  - Case and merged inputs use the same strict content validators
  - Tensor construction depends on TaskSpec order, not concrete field names

This module does NOT:
  - Load files, split membership, fit normalizers, or mutate persistent storage
  - Define case/merged schemas, fingerprints, or task field semantics
  - Translate historical payloads or accept partially verified content
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.datasets import dataset_identity as identity

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


class FlowModule:
    """
    Bind one strict task payload to batched model-ready tensors.

    Parameters
    ----------
    data : dict[str, Any]
        Current merged-dataset or single-case payload. The raw mapping is retained
        by reference; validated tensors are not normalized.
    task : TaskSpec
        Authoritative task contract for schema, channel order, and identity.

    Attributes
    ----------
    raw_data : dict[str, Any]
        Original payload mapping retained by reference.
    task : TaskSpec
        Authoritative immutable task contract used for validation.
    mode : str
        ``"merged"`` or ``"single"`` according to ``schema_kind``.
    inputs, outputs
        Batched task-order tensors. A case payload gains a leading batch axis of
        length one; merged tensors retain their existing batch axis.
    dataset_identity : DatasetIdentity | None
        Verified merged identity, or ``None`` for a single case.
    fields : dict[str, list[str]]
        Isolated input/output name lists in TaskSpec order.

    Raises
    ------
    TypeError
        If payload containers, metadata, or tensors violate the current schema.
    ValueError
        If schema fields, shapes, identities, or recomputed content fingerprints
        disagree with the TaskSpec, or ``schema_kind`` is unsupported.

    Notes
    -----
    Construction always enables strict tensor-byte verification. It performs no
    file I/O, split selection, normalization, persistent mutation, or historical
    schema translation.

    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        task: TaskSpec,
    ) -> None:
        """Strictly verify content and materialize batched tensor bindings."""
        self.raw_data = data
        self.task = task
        schema_kind = data.get("schema_kind")
        self.dataset_identity: identity.DatasetIdentity | None
        if schema_kind == identity.MERGED_DATASET_SCHEMA_KIND:
            self.mode = "merged"
            self.dataset_identity = identity.validate_merged_dataset_payload(
                data,
                task=task,
                verify_content=True,
            )
            self.inputs = data["inputs"]
            self.outputs = data["outputs"]
        elif schema_kind == identity.CASE_SCHEMA_KIND:
            self.mode = "single"
            validated = identity.validate_case_payload(
                data,
                task=task,
                verify_content=True,
            )
            self.dataset_identity = None
            self.inputs = validated.inputs.unsqueeze(0)
            self.outputs = validated.outputs.unsqueeze(0)
        else:
            msg = (
                "Unsupported dataset schema. Expected schema_kind "
                f"{identity.MERGED_DATASET_SCHEMA_KIND!r} or {identity.CASE_SCHEMA_KIND!r}; "
                f"got {schema_kind!r}."
            )
            raise ValueError(msg)

        self.fields = {
            "inputs": list(task.input_names),
            "outputs": list(task.output_names),
        }

    def apply(self, idx: int, sample: dict[str, Any]) -> None:
        """
        Insert one exact task input/output tensor pair into a sample.

        Parameters
        ----------
        idx : int
            Standard leading-axis tensor index. For single-case payloads, ``0``
            and Python's equivalent negative index ``-1`` address the sole row.
        sample : dict[str, Any]
            Mutable sample populated under ``sample["x"]["input"]`` and
            ``sample["y"]["output"]``; existing nested mappings are reused.

        Raises
        ------
        IndexError
            If ``idx`` is outside the batched tensor bounds.
        TypeError
            If an existing ``sample["x"]`` or ``sample["y"]`` value does not
            support keyed assignment.

        Notes
        -----
        Tensor rows are assigned as views; the source tensors and payload are not
        copied or mutated.

        """
        x = sample.setdefault("x", {})
        y = sample.setdefault("y", {})
        x["input"] = self.inputs[idx]
        y["output"] = self.outputs[idx]
