"""FlowModule for merged PINO training datasets (with dynamic field selection)."""

from __future__ import annotations

from typing import Any


class FlowModule:
    """Handle merged PINO datasets with optional input/output channel selection."""

    def __init__(
        self,
        data: dict[str, Any],
        include_inputs: list[str] | None = None,
        include_outputs: list[str] | None = None,
    ) -> None:
        """Initialize FlowModule for merged datasets.

        Args:
            data:
                Dictionary with keys `'inputs'`, `'outputs'`, and `'fields'` created by `merge_batch_cases.py`.
                - `'inputs'`: tensor `[N, C_in, H, W]` containing input fields:
                    `x`, `y`,
                    `kappaxx`, `kappayx`, `kappazx`,
                    `kappaxy`, `kappayy`, `kappazy`,
                    `kappaxz`, `kappayz`, `kappazz`.
                - `'outputs'`: tensor `[N, C_out, H, W]` containing target fields:
                    `p`, `u`, `v`, `U`.
                - `'fields'`: metadata dictionary with lists of channel names for both
                  (`fields["inputs"]`, `fields["outputs"]`).


            include_inputs:
                Optional list of input field names to include (subset of `fields["inputs"]`).
                Default is `None` → all inputs used.

            include_outputs:
                Optional list of output field names to include (subset of `fields["outputs"]`).
                Default is `None` → all outputs used.

        """
        self.data = data

        self.inputs = data["inputs"]  # [N, C_in, H, W]
        self.outputs = data["outputs"]  # [N, C_out, H, W]
        self.fields = data["fields"]  # {'inputs': [...], 'outputs': [...]}

        # --- Field selection ---
        all_input_fields = self.fields["inputs"]
        all_output_fields = self.fields["outputs"]

        if include_inputs is None:
            self.input_idx = list(range(len(all_input_fields)))
        else:
            self.input_idx = [all_input_fields.index(name) for name in include_inputs if name in all_input_fields]

        if include_outputs is None:
            self.output_idx = list(range(len(all_output_fields)))
        else:
            self.output_idx = [all_output_fields.index(name) for name in include_outputs if name in all_output_fields]

    def apply(self, idx: int, sample: dict[str, Any]) -> None:
        """Insert one (x, y) pair into a dataset sample dict."""
        x = sample.setdefault("x", {})
        y = sample.setdefault("y", {})

        x["input"] = self.inputs[idx, self.input_idx]  # [len(input_idx), H, W]
        y["output"] = self.outputs[idx, self.output_idx]  # [len(output_idx), H, W]
