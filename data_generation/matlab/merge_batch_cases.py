"""
============================================================
 merge_batch_cases.py
============================================================
Author:  Rino M. Albertin
Date:    2025-10-28
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Merges all individual case_XXXX.pt files of a simulation batch
into one consolidated dataset for PINO training and copies the
corresponding meta.pt file to the model_training directory.

DIRECTORY STRUCTURE
-------------------
Input:
    data/raw/<batch_name>/cases/case_XXXX.pt
    data/raw/<batch_name>/meta.pt

Output:
    model_training/data/raw/<batch_name>/
        ├── <batch_name>.pt
        └── meta.pt

USAGE
-----
As module:
    from merge_batch_cases import merge_batch_cases
    result = merge_batch_cases("samples_uniform_var10_N1000", verbose=True)

As script:
    python merge_batch_cases.py

verbose=True  → shows tqdm progress bar and structure preview  
verbose=False → silent execution, logs returned only
============================================================
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil


def merge_batch_cases(batch_name: str,
                      keep_input_fields=None,
                      keep_output_fields=None,
                      verbose: bool = False) -> dict:
    """
    Merges all case_XXXX.pt files of a batch into one dataset
    for PINO training and copies meta.pt.

    Parameters
    ----------
    batch_name : str
        Name of the simulation batch.
    keep_input_fields : list[str], optional
        List of input field names to include.
    keep_output_fields : list[str], optional
        List of output field names to include.
    verbose : bool, optional
        If True, shows structure preview and tqdm progress bar.

    Returns
    -------
    dict
        {
            "batch_name": str,
            "n_cases": int,
            "inputs_shape": tuple,
            "outputs_shape": tuple,
            "dst_dir": Path,
            "meta_copied": bool,
            "log": list[str]
        }
    """
    if keep_input_fields is None:
        keep_input_fields = ["kappaxx"]
    if keep_output_fields is None:
        keep_output_fields = ["p"]

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
        raise RuntimeError(f"No .pt case files found in {cases_dir}")

    # -------------------- preview first case structure --------------------
    first_case_path = case_files[0]
    first_case = torch.load(first_case_path, map_location="cpu", weights_only=False)
    input_fields_first = {k: v for k, v in first_case["input_fields"].items()
                          if k in keep_input_fields}
    output_fields_first = {k: v for k, v in first_case["output_fields"].items()
                           if k in keep_output_fields}

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

    pbar = tqdm(total=len(case_files),
                desc=f"Merging {batch_name}",
                unit="file",
                disable=not verbose)

    for case_path in case_files:
        data_case = torch.load(case_path, map_location="cpu", weights_only=False)
        input_fields = {k: v for k, v in data_case["input_fields"].items()
                        if k in keep_input_fields}
        output_fields = {k: v for k, v in data_case["output_fields"].items()
                         if k in keep_output_fields}

        if not all(k in input_fields for k in keep_input_fields) or \
           not all(k in output_fields for k in keep_output_fields):
            log.append(f"Skipped {case_path.name}: missing fields")
            pbar.update(1)
            continue

        input_stack = np.stack([input_fields[k] for k in keep_input_fields], axis=0)
        output_stack = np.stack([output_fields[k] for k in keep_output_fields], axis=0)

        inputs_list.append(torch.tensor(input_stack, dtype=torch.float32))
        outputs_list.append(torch.tensor(output_stack, dtype=torch.float32))

        pbar.update(1)

    pbar.close()

    if not inputs_list:
        raise RuntimeError(f"No valid cases merged in {cases_dir}")

    inputs_tensor = torch.stack(inputs_list, dim=0)
    outputs_tensor = torch.stack(outputs_list, dim=0)

    log.append(f"Inputs shape: {tuple(inputs_tensor.shape)}")
    log.append(f"Outputs shape: {tuple(outputs_tensor.shape)}")
    log.append(f"Cases merged: {len(inputs_list)}")

    # Save merged dataset
    batch_dataset = {
        "inputs": inputs_tensor,
        "outputs": outputs_tensor,
        "fields": {
            "inputs": keep_input_fields,
            "outputs": keep_output_fields,
        },
    }
    torch.save(batch_dataset, dst_data_path)
    log.append(f"Saved dataset: {dst_data_path}")

    # Copy meta file
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
    result = merge_batch_cases("samples_uniform_var10_N1000", keep_input_fields=["x", "y", "kappaxx"], keep_output_fields=["u", "v", "p"], verbose=True)
    for line in result["log"]:
        print(line)