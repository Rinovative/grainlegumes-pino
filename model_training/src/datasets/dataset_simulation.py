"""
===============================================================================
dataset_simulation.py
===============================================================================
Load the single current task-aware final training-dataset format.

Responsibilities:
  - Load one final dataset payload from an explicit path
  - Verify it through the shared flow module and identity contract
  - Expose task-ordered input/output tensors with isolated source metadata

Design principles:
  - Dataset admission is strict before any sample is returned
  - Task field order is authoritative for model-facing tensors
  - Source metadata is copied so callers cannot mutate persisted payload state

This module does NOT:
  - Build or publish final datasets; ``data_generation`` owns construction
  - Fit normalizers or split memberships; dataset-base services own those steps
  - Resolve logical dataset names or storage roots; ``common.paths`` owns paths
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from .dataset_modules import flow

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


class PhysicsDataset(Dataset[dict[str, Any]]):
    """Expose strictly verified final-dataset tensors in task channel order."""

    def __init__(self, data_path: str | Path, *, task: TaskSpec) -> None:
        """Load and validate one exact final dataset file."""
        path = Path(data_path)
        if not path.is_file():
            msg = f"Training dataset file does not exist: {path}"
            raise FileNotFoundError(msg)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            msg = f"Training dataset must contain a dictionary payload: {path}"
            raise TypeError(msg)
        self.path = path
        self.task = task
        self.input_fields = list(task.input_names)
        self.output_fields = list(task.output_names)
        self.data = payload
        self.flow_module = flow.FlowModule(payload, task=task)
        self.identity = self.flow_module.dataset_identity

    def __len__(self) -> int:
        """Return the verified ordered sample count."""
        return self.identity.sample_count

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one task-order input/output pair and isolated case metadata."""
        sample: dict[str, Any] = {}
        self.flow_module.apply(idx, sample)
        raw_meta = self.data["source_metadata"][idx]
        if not isinstance(raw_meta, Mapping):
            msg = f"Training source_metadata[{idx}] must be a mapping: {self.path}"
            raise TypeError(msg)
        return {
            "x": sample["x"]["input"],
            "y": sample["y"]["output"],
            "meta": deepcopy(dict(raw_meta)),
        }


def create_task_dataset(data_path: str | Path, *, task: TaskSpec) -> PhysicsDataset:
    """Construct the shared final-dataset implementation for training/inference."""
    return PhysicsDataset(data_path, task=task)
