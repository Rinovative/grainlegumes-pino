"""
===============================================================================
dataset_simulation.py
===============================================================================
Load validated task-aware simulation datasets.

Responsibilities:
  - Load current merged datasets or strict directories of case payloads
  - Validate schema, fields, shapes, ordered identity, and fingerprints
  - Return identical model-ready x/y tensors to training and inference

Design principles:
  - Callers supply one resolved TaskSpec; fields are never selected ad hoc
  - Merged and case-directory modes share the same identity validators
  - Unsupported payloads and noncanonical aliases fail before sample exposure

Boundaries:
  - Schema and fingerprint algorithms belong to datasets.identity
  - Split construction and normalization belong to datasets.base
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from . import dataset_identity as identity
from .dataset_modules import flow

if TYPE_CHECKING:
    from torch import Tensor

    from src.domain.tasks.domain_task_spec import TaskSpec


class PhysicsDataset(Dataset[dict[str, Any]]):
    """Expose strict task tensors from a merged file or case directory."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        task: TaskSpec,
    ) -> None:
        """
        Initialize and validate a current task dataset.

        Parameters
        ----------
        data_path : str | Path
            Merged ``.pt`` file or directory of strict case files.
        task : TaskSpec
            Authoritative task contract used by every producer and consumer.

        """
        path = Path(data_path)
        self.path = path
        self.task = task
        self.input_fields = list(task.input_names)
        self.output_fields = list(task.output_names)
        self.data: dict[str, Any] | None = None
        self.case_files: list[Path] = []
        self.flow_module: flow.FlowModule | None = None

        if path.is_dir():
            self.mode = "cases"
            self.case_files = sorted(path.glob("case_*.pt"))
            if not self.case_files:
                msg = f"No case_XXXX.pt files found in directory: {path}"
                raise RuntimeError(msg)

            validated_cases: list[identity.CaseIdentity] = []
            for case_path in self.case_files:
                payload = torch.load(case_path, map_location="cpu", weights_only=False)
                validated = identity.validate_case_identity(
                    payload,
                    task=task,
                    verify_content=True,
                )
                if validated.case_id != case_path.stem:
                    msg = f"Case filename/sample identity mismatch: {case_path.stem!r} != {validated.case_id!r}."
                    raise ValueError(msg)
                validated_cases.append(validated)
            dataset_id = path.parent.name if path.name == "cases" else path.name
            self.identity = identity.case_collection_identity(
                task=task,
                dataset_id=dataset_id,
                cases=validated_cases,
            )
            return

        if not path.is_file():
            msg = f"Dataset path does not exist: {path}"
            raise FileNotFoundError(msg)
        self.mode = "merged"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            msg = f"Merged dataset must contain a dictionary payload: {path}"
            raise TypeError(msg)
        self.data = payload
        self.flow_module = flow.FlowModule(payload, task=task)
        if self.flow_module.dataset_identity is None:
            msg = "Merged FlowModule did not expose dataset identity."
            raise RuntimeError(msg)
        self.identity = self.flow_module.dataset_identity

    def __len__(self) -> int:
        """
        Return the number of validated ordered samples.

        Returns
        -------
        int
            Dataset sample count.

        """
        return self.identity.sample_count

    def _load_case(self, idx: int) -> dict[str, Any]:
        """
        Load one already-admitted case and revalidate it before use.

        Parameters
        ----------
        idx : int
            Ordered case index.

        Returns
        -------
        dict[str, Any]
            Model-ready ``x``/``y`` tensors and source metadata.

        """
        case_path = self.case_files[idx]
        payload = torch.load(case_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            msg = f"Case file must contain a dictionary payload: {case_path}"
            raise TypeError(msg)
        module = flow.FlowModule(payload, task=self.task)
        sample: dict[str, Any] = {}
        module.apply(0, sample)
        x_tensor: Tensor = sample["x"]["input"]
        y_tensor: Tensor = sample["y"]["output"]
        raw_meta = payload["source_metadata"]
        if not isinstance(raw_meta, Mapping):
            msg = f"Case source_metadata must be a mapping: {case_path}"
            raise TypeError(msg)
        return {"x": x_tensor, "y": y_tensor, "meta": deepcopy(dict(raw_meta))}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return one validated model-ready sample.

        Parameters
        ----------
        idx : int
            Ordered sample index.

        Returns
        -------
        dict[str, Any]
            ``x`` and ``y`` tensors plus isolated per-sample source metadata.

        """
        if self.mode == "cases":
            return self._load_case(idx)
        module = self.flow_module
        data = self.data
        if module is None or data is None:
            msg = "FlowModule or merged payload is not initialized in merged mode."
            raise RuntimeError(msg)
        sample: dict[str, Any] = {}
        module.apply(idx, sample)
        raw_meta = data["source_metadata"][idx]
        if not isinstance(raw_meta, Mapping):
            msg = f"Merged source_metadata[{idx}] must be a mapping: {self.path}"
            raise TypeError(msg)
        return {
            "x": sample["x"]["input"],
            "y": sample["y"]["output"],
            "meta": deepcopy(dict(raw_meta)),
        }


def create_task_dataset(
    data_path: str | Path,
    *,
    task: TaskSpec,
) -> PhysicsDataset:
    """
    Construct the shared training/inference dataset implementation.

    Parameters
    ----------
    data_path : str | Path
        Merged dataset file or strict case directory.
    task : TaskSpec
        Authoritative task contract.

    Returns
    -------
    PhysicsDataset
        Fully validated task-aware dataset.

    """
    return PhysicsDataset(data_path, task=task)
