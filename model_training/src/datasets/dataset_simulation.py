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
  - Every admitted payload receives strict tensor-byte identity verification
  - Returned metadata is isolated while tensor channel order remains task-owned

This module does NOT:
  - Create, merge, repair, overwrite, split, or normalize stored datasets
  - Define schemas, fingerprints, task fields, or DataLoader worker behavior
  - Accept historical payloads, field aliases, or unverified lazy membership
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
    """
    Expose verified task tensors from a merged file or strict case directory.

    Parameters
    ----------
    data_path : str | pathlib.Path
        Merged ``.pt`` payload or directory containing ordered ``case_*.pt``
        payloads. A directory named ``cases`` inherits its dataset ID from the
        parent; other directories use their own name.
    task : TaskSpec
        Authoritative schema, field order, layout, and identity contract.

    Attributes
    ----------
    mode : str
        ``"merged"`` or ``"cases"``.
    identity : DatasetIdentity
        Verified portable ordered identity for the complete source.
    input_fields, output_fields : list[str]
        Task-order channel names exposed by every sample.
    case_files : list[pathlib.Path]
        Lexically sorted case paths in directory mode; empty in merged mode.

    Raises
    ------
    FileNotFoundError
        If ``data_path`` is neither an existing file nor directory.
    OSError, RuntimeError
        If a payload cannot be loaded, a case directory is empty, or merged
        state cannot be initialized consistently.
    TypeError, ValueError
        If serialized containers, current schema, ordered fields, identities,
        tensor geometry, metadata, or strict content fingerprints are invalid.

    Notes
    -----
    Construction strictly verifies all tensor bytes before exposing membership.
    Directory samples are loaded and reverified again on access; merged tensors
    remain resident. Returned ``x``/``y`` channels retain TaskSpec physical/stored
    representation, and metadata is deep-copied for consumer isolation.

    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        task: TaskSpec,
    ) -> None:
        """Load and strictly verify the complete source before sample exposure."""
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
            Task-order ``x``/``y`` tensors and a deep copy of source metadata.

        Raises
        ------
        IndexError
            If ``idx`` is outside the ordered case-file list.
        OSError, RuntimeError
            If the admitted file can no longer be loaded.
        TypeError, ValueError
            If the case changed after construction or now violates strict schema,
            metadata, identity, or content-fingerprint validation.

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
            Channel-first ``x`` and ``y`` tensors in TaskSpec order plus isolated
            per-sample source metadata.

        Raises
        ------
        IndexError
            If ``idx`` is outside the ordered dataset bounds.
        OSError, TypeError, ValueError, RuntimeError
            If a directory case cannot be reloaded/revalidated, merged state is
            inconsistent, or source metadata is not a mapping.

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
        Fully validated task-aware dataset shared by training and inference.

    Raises
    ------
    FileNotFoundError, OSError, RuntimeError, TypeError, ValueError
        Propagated from ``PhysicsDataset`` when path loading or strict dataset
        validation fails.

    """
    return PhysicsDataset(data_path, task=task)
