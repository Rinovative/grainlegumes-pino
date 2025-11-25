"""
FlowModule for PINO/FNO datasets (merged + single-case).

This class provides a unified interface for loading flow-related
datasets used in operator learning. It supports two dataset formats:

1) Merged datasets
   ----------------
   Expected structure:
       data = {
           "inputs":  Tensor [N, C_in, H, W],
           "outputs": Tensor [N, C_out, H, W],
           "fields": {
               "inputs":  [...],   # ordered list of input channel names
               "outputs": [...],   # ordered list of output channel names
           }
       }

   Input channels (in canonical order):
       x, y,
       kappaxx, kappayx, kappazx,
       kappaxy, kappayy, kappazy,
       kappaxz, kappayz, kappazz
   These represent the spatial coordinates and the 3x3 permeability tensor
   stored in Voigt-like flattened form.

   Output channels:
       p, u, v, U
   Pressure, velocity components (u, v) and velocity magnitude U.

   All channels are stacked along the channel dimension.

2) Single-case datasets
   ----------------------
   Expected structure:
       data = {
           "input_fields":  dict[field_name → 2D array],
           "output_fields": dict[field_name → 2D array],
           "meta": {...}
       }

   Each case is stored individually and converted to tensors with an
   artificial batch dimension of size 1. Channel names are taken directly
   from the dictionary keys.

Both dataset formats produce a unified interface via:
    module.apply(i, sample)

which inserts:
    sample["x"]["input"]  → Tensor [C_in_sel, H, W]
    sample["y"]["output"] → Tensor [C_out_sel, H, W]
"""

from __future__ import annotations

from typing import Any

import torch


class FlowModule:
    """Unified loader for flow datasets with optional channel selection."""

    def __init__(
        self,
        data: dict[str, Any],
        include_inputs: list[str] | None = None,
        include_outputs: list[str] | None = None,
    ) -> None:
        """
        Initialize the module for merged or single-case dataset formats.

        Parameters
        ----------
        data : dict
            Dataset dictionary. Two formats are supported:

            Merged dataset format:
                Required keys:
                    - "inputs":  Tensor [N, C_in, H, W]
                                 Channels (in order):
                                     x, y,
                                     kappaxx, kappayx, kappazx,
                                     kappaxy, kappayy, kappazy,
                                     kappaxz, kappayz, kappazz
                    - "outputs": Tensor [N, C_out, H, W]
                                 Channels:
                                     p, u, v, U
                    - "fields": {
                          "inputs":  list of input channel names,
                          "outputs": list of output channel names,
                      }

            Single-case format:
                Required keys:
                    - "input_fields":  dict[field_name → 2D array]
                    - "output_fields": dict[field_name → 2D array]
                    - "meta": metadata dictionary
                These are converted into tensors with a batch dimension (size 1).

        include_inputs : list[str] | None
            Optional list of input channel names to select.
            If None, all input channels are used.

        include_outputs : list[str] | None
            Optional list of output channel names to select.
            If None, all output channels are used.

        """
        self.raw_data = data

        # -------------------------------------------------------------
        # Detect dataset mode
        # -------------------------------------------------------------
        if all(k in data for k in ("inputs", "outputs", "fields")):
            self.mode = "merged"

            self.inputs = data["inputs"]  # [N, C_in, H, W]
            self.outputs = data["outputs"]  # [N, C_out, H, W]
            self.fields = data["fields"]  # {"inputs": [...], "outputs": [...]}

        elif all(k in data for k in ("input_fields", "output_fields", "meta")):
            self.mode = "single"

            input_dict = data["input_fields"]
            output_dict = data["output_fields"]

            # Deterministic key order
            input_names = list(input_dict.keys())
            output_names = list(output_dict.keys())

            # Convert arrays → tensors
            input_stack = torch.stack([torch.tensor(input_dict[name], dtype=torch.float32) for name in input_names], dim=0)
            output_stack = torch.stack([torch.tensor(output_dict[name], dtype=torch.float32) for name in output_names], dim=0)

            # Artificial batch dimension
            self.inputs = input_stack.unsqueeze(0)
            self.outputs = output_stack.unsqueeze(0)

            self.fields = {
                "inputs": input_names,
                "outputs": output_names,
            }

        else:
            msg = (
                "Unsupported dataset format. Expected either:\n"
                "  merged dataset:      keys ['inputs','outputs','fields']\n"
                "  single-case dataset: keys ['input_fields','output_fields','meta']\n"
                f"Got keys: {list(data.keys())}"
            )
            raise KeyError(msg)

        # -------------------------------------------------------------
        # Channel selection
        # -------------------------------------------------------------
        all_in = self.fields["inputs"]
        all_out = self.fields["outputs"]

        if include_inputs is None:
            self.input_idx = list(range(len(all_in)))
        else:
            self.input_idx = [all_in.index(name) for name in include_inputs]

        if include_outputs is None:
            self.output_idx = list(range(len(all_out)))
        else:
            self.output_idx = [all_out.index(name) for name in include_outputs]

    # -------------------------------------------------------------
    def apply(self, idx: int, sample: dict[str, Any]) -> None:
        """
        Insert a selected (x, y) pair into a dataset sample.

        Parameters
        ----------
        idx : int
            Case index. Must be 0 for single-case datasets.

        sample : dict
            Sample dictionary to be populated. On return contains:
                sample["x"]["input"]  : Tensor [C_in_sel, H, W]
                sample["y"]["output"] : Tensor [C_out_sel, H, W]

        """
        x = sample.setdefault("x", {})
        y = sample.setdefault("y", {})

        x["input"] = self.inputs[idx, self.input_idx]
        y["output"] = self.outputs[idx, self.output_idx]
