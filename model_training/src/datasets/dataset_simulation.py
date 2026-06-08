"""
===============================================================================
dataset_simulation.py
===============================================================================
Load simulation-based steady-flow samples for training and evaluation.

Responsibilities:
  - Load merged training datasets or directories of case files
  - Delegate flow-field tensor construction to dataset modules
  - Return x/y tensors and optional case metadata

Design principles:
  - Merged and case-directory layouts share one sample interface
  - Case ordering is deterministic through sorted case paths
  - Channel ordering is delegated to domain-aware flow modules

Boundaries:
  - Split creation and DataLoader construction belong to datasets.base
  - Model, loss and training orchestration belong to learning

Notes:
  Data layouts:
    1) Merged training dataset (single `.pt` file)
       -------------------------------------------------
       Produced by `merge_batch_cases.py`, with structure:
              {
                "inputs":  Tensor [N, C_in, H, W],
                "outputs": Tensor [N, C_out, H, W],
                "fields": {
                     "inputs":  list[str],  # channel names in order
                     "outputs": list[str],
                },
              }
         In this mode, the dataset behaves like a standard
        operator-learning dataset over N samples.

    2) Evaluation dataset from individual case files
       -------------------------------------------------
       A directory containing `case_XXXX.pt` files produced by
       `build_batch_dataset.py`, each with structure:
              {
                "input_fields":  dict[str, 2D-array],
                "output_fields": dict[str, 2D-array],
                "meta":          dict,
              }
         These cases are converted on-the-fly into tensors with a
         synthetic batch dimension of size 1, then reduced back to
         [C_in, H, W] / [C_out, H, W] via the FlowModule.

    In both modes, the FlowModule constructs model-ready tensors for
    PINO/FNO models, and __getitem__ returns:
        {"x": Tensor, "y": Tensor}          # merged mode
        {"x": Tensor, "y": Tensor, "meta": dict}  # cases mode

===============================================================================

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from . import dataset_base as base
from .dataset_modules import flow

if TYPE_CHECKING:
    from torch import Tensor


class PhysicsDataset(base.BaseDataset):
    """
    Dataset for steady-state flow simulations with permeability fields.

    Supports two data layouts:

    1) Merged training dataset (single `.pt` file)
       -------------------------------------------------
       Produced by `merge_batch_cases.py`, with structure:

           {
               "inputs":  Tensor [N, C_in, H, W],
               "outputs": Tensor [N, C_out, H, W],
               "fields": {
                   "inputs":  list[str],  # channel names in order
                   "outputs": list[str],
               },
           }

       In this mode, the dataset behaves like a standard
       operator-learning dataset over N samples.

    2) Evaluation dataset from individual case files
       -------------------------------------------------
       A directory containing `case_XXXX.pt` files produced by
       `build_batch_dataset.py`, each with structure:

           {
               "input_fields":  dict[str, 2D-array],
               "output_fields": dict[str, 2D-array],
               "meta":          dict,
           }

       These cases are converted on-the-fly into tensors with a
       synthetic batch dimension of size 1, then reduced back to
       [C_in, H, W] / [C_out, H, W] via the FlowModule.

    In both modes, the FlowModule constructs model-ready tensors for
    PINO/FNO models, and __getitem__ returns:

        {"x": Tensor, "y": Tensor}          # merged mode
        {"x": Tensor, "y": Tensor, "meta": dict}  # cases mode
    """

    def __init__(
        self,
        data_path: str,
        include_inputs: list[str] | None = None,
        include_outputs: list[str] | None = None,
    ) -> None:
        """
        Initialize dataset from either a merged `.pt` file or a case directory.

        Parameters
        ----------
        data_path : str
            Path to a merged dataset file (`<batch_name>.pt`) or to a
            directory containing `case_XXXX.pt` files.
        include_inputs : list[str] | None, optional
            Optional list of input channel names to include in x.
            If None, all available input channels are used (default: None).
        include_outputs : list[str] | None, optional
            Optional list of output channel names to include in y.
            If None, all available output channels are used (default: None).

        """
        path = Path(data_path)

        self.mode: str
        self.case_files: list[Path] = []
        self.flow_module: flow.FlowModule | None = None
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
            # BaseDataset expects `self.data`, but we do not use it in cases mode.
            self.data = None  # type: ignore[assignment]
            return

        # -----------------------------------------------------------
        # Training mode: merged dataset
        # -----------------------------------------------------------
        self.mode = "merged"

        # BaseDataset.__init__ loads the serialized dict into self.data
        super().__init__(data_path)  # self.data: dict[str, Tensor]

        # FlowModule handles channel ordering and selection for merged data
        self.flow_module = flow.FlowModule(
            self.data,  # type: ignore[arg-type]
            include_inputs=include_inputs,
            include_outputs=include_outputs,
        )

    # ---------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples:
                - For merged mode: N (first dimension of "inputs")
                - For cases mode:  number of `case_XXXX.pt` files

        """
        if self.mode == "merged":
            return self.data["inputs"].shape[0]  # type: ignore[index]

        return len(self.case_files)

    # ---------------------------------------------------------------

    def _load_case(self, idx: int) -> dict[str, Any]:
        """
        Load and process a single `case_XXXX.pt` file in evaluation mode.

        The raw case format is converted via FlowModule to

            x: Tensor [C_in, H, W]
            y: Tensor [C_out, H, W]

        A shallow copy of the metadata is returned under "meta".

        Parameters
        ----------
        idx : int
            Case index in the sorted list of `case_XXXX.pt` files.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
                - "x":    Tensor [C_in, H, W]
                - "y":    Tensor [C_out, H, W]
                - "meta": dict with case metadata

        """
        case_path = self.case_files[idx]
        case_dict: dict[str, Any] = torch.load(case_path)

        module = flow.FlowModule(
            case_dict,
            include_inputs=self.include_inputs,
            include_outputs=self.include_outputs,
        )

        sample: dict[str, Any] = {}
        # Single-case format has an artificial batch dimension of size 1
        module.apply(0, sample)

        x_tensor: Tensor = sample["x"]["input"]
        y_tensor: Tensor = sample["y"]["output"]
        meta: dict[str, Any] = case_dict.get("meta", {})

        return {"x": x_tensor, "y": y_tensor, "meta": meta}

    # ---------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Retrieve one dataset item.

        Training (merged) mode:
            Returns a dictionary
                {
                    "x": Tensor [C_in, H, W],
                    "y": Tensor [C_out, H, W],
                }

        Evaluation (cases) mode:
            Returns a dictionary
                {
                    "x":    Tensor [C_in, H, W],
                    "y":    Tensor [C_out, H, W],
                    "meta": dict,
                }

        Parameters
        ----------
        idx : int
            Sample index (0-based).

        Returns
        -------
        dict[str, Any]
            Sample dictionary with model-ready tensors and optional metadata.

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
