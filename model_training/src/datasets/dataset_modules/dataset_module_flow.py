"""
===============================================================================
dataset_module_flow.py
===============================================================================
Convert flow dataset dictionaries into neural-operator tensors.

Responsibilities:
  - Support merged dataset dictionaries and single-case dictionaries
  - Select input and output fields in canonical domain order
  - Populate sample dictionaries with model-ready x/y tensors

Design principles:
  - Channel selection is explicit and deterministic
  - Domain field sets define canonical tensor order
  - Data conversion stays separate from DataLoader construction

Boundaries:
  - Dataset splitting and normalization belong to datasets.base
  - Model construction belongs to learning.models

Notes:
  Expected input formats:
    1) Merged dataset format:
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
         In this mode, the module behaves like a standard
        operator-learning dataset over N samples.

    2) Single-case format:
       -------------------------------------------------
       A dictionary with structure:
              {
                "input_fields":  dict[str, 2D-array],
                "output_fields": dict[str, 2D-array],
                "meta":          dict,
              }
         These are converted on-the-fly into tensors with a synthetic batch dimension of size 1, then reduced back to
         [C_in, H, W] / [C_out, H, W] via the FlowModule.
         The module returns a single sample with index 0 in this mode.

    In both modes, the FlowModule constructs model-ready tensors for
    PINO/FNO models, and __getitem__ returns:
        {"x": Tensor, "y": Tensor}          # merged mode
        {"x": Tensor, "y": Tensor, "meta": dict}  # single-case mode

===============================================================================

"""

from __future__ import annotations

from typing import Any

import torch

from src import domain


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
                                Input channels (in canonical internal order):
                                    x, y,
                                    kxx, kyy (, kzz),
                                    kxy (, kxz, kyz),
                                    eps, p_bc
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

            available_inputs = list(input_dict.keys())

            # --- Dimension robust bestimmen ---
            dim = 3 if {"kxx", "kyy", "kzz"}.issubset(available_inputs) else 2

            # --- Kanonische Reihenfolge aus Schema ---
            input_names = domain.field_sets.default_training_inputs(dim)
            output_names = domain.field_sets.default_training_outputs(dim)

            # --- Safety: nur vorhandene Felder ---
            input_names = [k for k in input_names if k in input_dict]
            output_names = [k for k in output_names if k in output_dict]

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
