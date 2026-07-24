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
  - Case and merged inputs use the same task contract and validator
  - Tensor construction is independent of any concrete task field names

Boundaries:
  - Dataset identity algorithms belong to datasets.identity
  - Splitting and normalization belong to datasets.base
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.datasets import dataset_identity as identity

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


class FlowModule:
    """Expose strict task tensors from a merged or single-case payload."""

    def __init__(
        self,
        data: dict[str, Any],
        *,
        task: TaskSpec,
    ) -> None:
        """
        Validate and materialize a current dataset payload.

        Parameters
        ----------
        data : dict[str, Any]
            Current merged-dataset or single-case payload.
        task : TaskSpec
            Authoritative task contract used for exact schema validation.

        Raises
        ------
        ValueError
            If schema, fields, shapes, sample identity, or stored fingerprint
            are invalid, or strict content verification finds a mismatch.

        """
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
            Source sample index. Single-case payloads accept only zero.
        sample : dict[str, Any]
            Mutable sample populated under ``x.input`` and ``y.output``.

        """
        x = sample.setdefault("x", {})
        y = sample.setdefault("y", {})
        x["input"] = self.inputs[idx]
        y["output"] = self.outputs[idx]
