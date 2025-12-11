"""
===============================================================================

 merge_batch_cases.
===============================================================================
Author:  Rino M. Albertin
Date:    2025-10-28
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Merges all individual case_XXXX.pt files of a simulation batch into a single
training dataset file for PINO/FNO.

Each case file contains:
    - input_fields:  x, y, kappa* (already transformed in build_batch_dataset)
    - output_fields: p, u, v, U
    - meta:          simulation metadata

This script:
    1. Loads all case_XXXX.pt files.
    2. Selects the desired input and output channels.
    3. Stacks them into tensors of shape:
           inputs:  (N, C_in,  ny, nx)
           outputs: (N, C_out, ny, nx)
    4. Saves a single <batch_name>.pt dataset for model_training.
    5. Copies meta.pt into the dataset folder if available.

No additional transformations (such as log10) are applied here.
===============================================================================
"""  # noqa: INP001

import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def merge_batch_cases(  # noqa: C901, PLR0915
    batch_name: str,
    keep_input_fields: list[str] | None = None,
    keep_output_fields: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Merge all individual case_XXXX.pt files of a batch into one dataset for PINO/FNO training.

    Parameters
    ----------
    batch_name : str
        Name of the simulation batch.
    keep_input_fields : list[str] or None
        Input fields to keep. If None, defaults to all permeability components
        plus x and y coordinates.
    keep_output_fields : list[str] or None
        Output fields to keep. If None, defaults to velocity and pressure fields.
    verbose : bool
        If True, prints structure information for the first case.

    Returns
    -------
    dict
        Summary information including dataset path and shapes.

    """
    if keep_input_fields is None:
        keep_input_fields = [
            "x",
            "y",
            "kappaxx",
            "kappayx",
            "kappazx",
            "kappaxy",
            "kappayy",
            "kappazy",
            "kappaxz",
            "kappayz",
            "kappazz",
        ]
    if keep_output_fields is None:
        keep_output_fields = ["p", "u", "v", "U"]

    log = []

    matlab_dir = Path(__file__).resolve().parent
    base_root = matlab_dir.parents[1]

    src_batch = base_root / "data" / "raw" / batch_name
    cases_dir = src_batch / "cases"
    src_meta = src_batch / "meta.pt"

    dst_batch_dir = base_root / "model_training" / "data" / "raw" / batch_name
    dst_batch_dir.mkdir(parents=True, exist_ok=True)

    dst_data_path = dst_batch_dir / f"{batch_name}.pt"
    dst_meta_path = dst_batch_dir / "meta.pt"

    log.append(f"Merging batch: {batch_name}")
    log.append(f"Source cases: {cases_dir}")
    log.append(f"Destination: {dst_batch_dir}")

    case_files = sorted(cases_dir.glob("case_*.pt"))
    if not case_files:
        msg = f"No .pt case files found in {cases_dir}"
        raise RuntimeError(msg)

    # -------------------- preview first case structure --------------------
    first_case_path = case_files[0]
    first_case = torch.load(first_case_path, map_location="cpu", weights_only=False)

    input_fields_first = {k: v for k, v in first_case["input_fields"].items() if k in keep_input_fields}
    output_fields_first = {k: v for k, v in first_case["output_fields"].items() if k in keep_output_fields}

    if verbose:
        print("\nExample structure for first case:")
        print("--------------------------------------------------")
        print("input_fields:")
        for k, v in input_fields_first.items():
            arr = np.array(v)
            print(f"  {k:10s}  shape={arr.shape}, dtype={arr.dtype}")
        print("output_fields:")
        for k, v in output_fields_first.items():
            arr = np.array(v)
            print(f"  {k:10s}  shape={arr.shape}, dtype={arr.dtype}")
        print("--------------------------------------------------\n")

    # -------------------- main merging loop --------------------
    inputs_list = []
    outputs_list = []

    pbar = tqdm(
        total=len(case_files),
        desc=f"Merging {batch_name}",
        unit="file",
        disable=not verbose,
    )

    for case_path in case_files:
        data_case = torch.load(case_path, map_location="cpu", weights_only=False)

        input_fields = {k: v for k, v in data_case["input_fields"].items() if k in keep_input_fields}
        output_fields = {k: v for k, v in data_case["output_fields"].items() if k in keep_output_fields}

        if input_fields:
            input_stack = np.stack([input_fields[k] for k in input_fields], axis=0)
            inputs_list.append(torch.tensor(input_stack, dtype=torch.float32))

        if output_fields:
            output_stack = np.stack([output_fields[k] for k in output_fields], axis=0)
            outputs_list.append(torch.tensor(output_stack, dtype=torch.float32))

        pbar.update(1)

    pbar.close()

    if not inputs_list or not outputs_list:
        log.append(f"[WARNING] No valid tensors created from {cases_dir}")
        return {
            "batch_name": batch_name,
            "n_cases": 0,
            "inputs_shape": (),
            "outputs_shape": (),
            "dst_dir": dst_batch_dir,
            "meta_copied": False,
            "log": log,
        }

    inputs_tensor = torch.stack(inputs_list, dim=0)
    outputs_tensor = torch.stack(outputs_list, dim=0)

    log.append(f"Inputs shape: {tuple(inputs_tensor.shape)}")
    log.append(f"Outputs shape: {tuple(outputs_tensor.shape)}")
    log.append(f"Cases merged: {len(inputs_list)}")

    # Save merged dataset
    final_input_fields = [f for f in keep_input_fields if f in input_fields_first]
    final_output_fields = [f for f in keep_output_fields if f in output_fields_first]

    batch_dataset = {
        "inputs": inputs_tensor,
        "outputs": outputs_tensor,
        "fields": {
            "inputs": final_input_fields,
            "outputs": final_output_fields,
        },
    }

    torch.save(batch_dataset, dst_data_path)
    log.append(f"Saved dataset: {dst_data_path}")

    meta_copied = False
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta_path)
        meta_copied = True
        log.append(f"Copied meta.pt to {dst_meta_path}")
    else:
        log.append(f"No meta.pt found at {src_meta}")

    return {
        "batch_name": batch_name,
        "n_cases": len(inputs_list),
        "inputs_shape": tuple(inputs_tensor.shape),
        "outputs_shape": tuple(outputs_tensor.shape),
        "dst_dir": dst_batch_dir,
        "meta_copied": meta_copied,
        "log": log,
    }


if __name__ == "__main__":
    result = merge_batch_cases("lhs_var80_plog100_seed3001", verbose=True)
    for line in result["log"]:
        print(line)
