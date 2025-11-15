"""
Dataset definition for simulation-based PINO training.

This module implements the PermeabilityFlowDataset, which combines
the BaseDataset with a physics-specific FlowModule. It provides
input/output tensors (x, y) formatted for neural operator training
and supports both merged training datasets and directories of
individual case files for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.dataset.dataset_base import BaseDataset
from src.dataset.dataset_module.dataset_module_flow import FlowModule

if TYPE_CHECKING:
    from torch import Tensor


class PermeabilityFlowDataset(BaseDataset):
    """
    Dataset for steady-state flow simulations with permeability fields.

    Supports:
        - merged training dataset (single `.pt` file)
        - evaluation from individual `case_XXXX.pt` files

    The FlowModule constructs model-ready tensors for PINO/FNO models.
    """

    def __init__(
        self,
        data_path: str,
        include_inputs: list[str] | None = None,
        include_outputs: list[str] | None = None,
    ) -> None:
        """
        Initialise dataset from either a merged `.pt` file or a case directory.

        Args:
            data_path:
                Path to merged dataset or directory containing `case_XXXX.pt`.
            include_inputs:
                Field names to include in x. None = all.
            include_outputs:
                Field names to include in y. None = all.

        """
        path = Path(data_path)

        self.mode: str
        self.case_files: list[Path] = []
        self.flow_module: FlowModule | None = None
        self.include_inputs = include_inputs
        self.include_outputs = include_outputs

        # -----------------------------------------------------------
        # Evaluation mode: directory of case files
        # -----------------------------------------------------------
        if path.is_dir():
            self.mode = "cases"
            files = sorted(path.glob("case_*.pt"))

            if not files:
                msg = f"No case_XXXX.pt files found in directory: {path}"
                raise RuntimeError(msg)

            self.case_files = list(files)
            self.data = None  # type: ignore[assignment]
            return

        # -----------------------------------------------------------
        # Training mode: merged dataset
        # -----------------------------------------------------------
        self.mode = "merged"
        super().__init__(data_path)  # loads into self.data: dict[str, Tensor]

        self.flow_module = FlowModule(
            self.data,  # type: ignore[arg-type]
            include_inputs=include_inputs,
            include_outputs=include_outputs,
        )

    # ---------------------------------------------------------------

    def __len__(self) -> int:
        """
        Get Number of samples in dataset.

        Returns:
            int: sample count

        """
        if self.mode == "merged":
            return self.data["inputs"].shape[0]  # type: ignore[index]

        return len(self.case_files)

    # ---------------------------------------------------------------

    def _load_case(self, idx: int) -> dict[str, Any]:
        """
        Load and process one `case_XXXX.pt` file in evaluation mode.

        Returns:
            dict containing "x", "y", and optional "meta"

        """
        case_path = self.case_files[idx]
        case_dict: dict[str, Any] = torch.load(case_path)

        module = FlowModule(
            case_dict,
            include_inputs=self.include_inputs,
            include_outputs=self.include_outputs,
        )

        sample: dict[str, Any] = {}
        module.apply(0, sample)

        x_tensor: Tensor = sample["x"]["input"]
        y_tensor: Tensor = sample["y"]["output"]
        meta: dict[str, Any] = case_dict.get("meta", {})

        return {"x": x_tensor, "y": y_tensor, "meta": meta}

    # ---------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """
        Retrieve one dataset item.

        Training mode:
            returns {"x": Tensor, "y": Tensor}

        Evaluation mode:
            returns {"x": Tensor, "y": Tensor, "meta": dict}
        """
        if self.mode == "merged":
            fm = self.flow_module
            if fm is None:
                msg = "FlowModule is not initialised in merged mode."
                raise RuntimeError(msg)

            sample: dict[str, Any] = {}
            fm.apply(idx, sample)

            x_tensor: Tensor = sample["x"]["input"]
            y_tensor: Tensor = sample["y"]["output"]

            return {"x": x_tensor, "y": y_tensor}

        return self._load_case(idx)
