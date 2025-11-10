"""Dataset definition for simulation-based PINO training.

This module implements the PermeabilityFlowDataset, which combines
the BaseDataset with a physics-specific FlowModule. It provides
input/output tensors (x, y) formatted for neural operator training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.dataset.dataset_base import BaseDataset
from src.dataset.dataset_module.dataset_module_flow import FlowModule

if TYPE_CHECKING:
    import torch


class PermeabilityFlowDataset(BaseDataset):
    """Dataset for steady-state flow simulations with permeability fields.

    This dataset wraps the flattened dataset structure from `merge_batch_cases.py`
    and provides ready-to-use input/output tensors for PINO/FNO training.

    Expected structure in `.pt` file:
        {
            "inputs":  torch.Tensor [N, C_in, H, W],
            "outputs": torch.Tensor [N, C_out, H, W],
            "fields": {
                "inputs": [...],
                "outputs": [...]
            }
        }
    """

    def __init__(
        self,
        data_path: str,
        include_inputs: list[str] | None = None,
        include_outputs: list[str] | None = None,
    ) -> None:
        """Initialize the dataset and attach the FlowModule.

        Args:
            data_path:
                Path to the merged `.pt` dataset.
            include_inputs:
                Optional list of input field names to include (subset of `fields["inputs"]`).
                Default = None → all input channels used.
            include_outputs:
                Optional list of output field names to include (subset of `fields["outputs"]`).
                Default = None → all output channels used.

        """
        super().__init__(data_path)
        self.flow_module = FlowModule(
            self.data,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one preprocessed sample for index `idx`.

        Returns:
            dict[str, torch.Tensor]:
                {
                    "x": tensor [C_in, H, W],
                    "y": tensor [C_out, H, W],
                }

        """
        sample: dict[str, Any] = {}
        self.flow_module.apply(idx, sample)

        x_tensor = sample["x"]["input"]
        y_tensor = sample["y"]["output"]

        return {"x": x_tensor, "y": y_tensor}
