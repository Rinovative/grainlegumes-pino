"""Bind one validated final training dataset to model-ready tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.datasets import dataset_identity as identity

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


class FlowModule:
    """Strictly verify one current training dataset and expose task tensors."""

    def __init__(self, data: dict[str, Any], *, task: TaskSpec) -> None:
        """Validate and retain one final payload under its authoritative task."""
        self.raw_data = data
        self.task = task
        schema_kind = data.get("schema_kind")
        if schema_kind != identity.TRAINING_DATASET_SCHEMA_KIND:
            msg = f"Unsupported dataset schema. Expected schema_kind {identity.TRAINING_DATASET_SCHEMA_KIND!r}; got {schema_kind!r}."
            raise ValueError(msg)
        self.dataset_identity = identity.validate_training_dataset_payload(
            data,
            task=task,
            verify_content=True,
        )
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]
        self.fields = {"inputs": list(task.input_names), "outputs": list(task.output_names)}

    def apply(self, idx: int, sample: dict[str, Any]) -> None:
        """Insert one task-order input/output tensor pair into ``sample``."""
        x = sample.setdefault("x", {})
        y = sample.setdefault("y", {})
        x["input"] = self.inputs[idx]
        y["output"] = self.outputs[idx]
