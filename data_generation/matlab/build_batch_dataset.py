"""============================================================

 build_batch_dataset.py.
============================================================
Author:  Rino M. Albertin
Date:    2025-10-28
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Converts a full COMSOL simulation batch into structured PyTorch `.pt` files.
Each case combines numerical results (`.csv`) with metadata (`.json`)
and is saved as an individual `.pt` file containing:
    - input_fields  (x, y, permeability tensors)
    - output_fields (pressure and velocity)
    - meta          (batch generation parameters)
A batch-level `meta.pt` file is also generated.

DIRECTORY STRUCTURE
-------------------
Input:
    data_generation/data/processed/<batch_name>/case_XXXX_sol.csv
    data_generation/data/raw/<batch_name>/case_XXXX.json
    data_generation/data/meta/<batch_name>.json / <batch_name>.csv
Output:
    data/raw/<batch_name>/
        ├── cases/        (individual .pt files)
        └── meta.pt       (batch metadata)
============================================================
"""  # noqa: INP001

import json
from pathlib import Path
from typing import Any

import numpy as np  # --- added
import pandas as pd
import torch
from tqdm import tqdm


def build_batch_dataset(batch_name: str, verbose: bool = False) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Build structured .pt case files and a batch-level meta.pt file.

    Parameters
    ----------
    batch_name : str
        Name of the simulation batch.
    verbose : bool, optional
        If True, displays a structure preview and tqdm progress bar.

    Returns
    -------
    dict
        Summary information including output paths and log messages.

    """
    log = []

    # ----------------------------- paths -----------------------------
    proj_root = Path(__file__).resolve().parents[2]
    gen_data_dir = proj_root / "data_generation" / "data"

    proc_dir = gen_data_dir / "processed" / batch_name
    raw_dir = gen_data_dir / "raw" / batch_name
    meta_dir = gen_data_dir / "meta"

    out_root = proj_root / "data" / "raw"
    out_batch = out_root / batch_name
    cases_dir = out_batch / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    log.append(f"Processing batch: {batch_name}")

    # ----------------------- matching json/csv -----------------------
    json_files = sorted(raw_dir.glob("case_*.json"))
    csv_files = sorted(proc_dir.glob("case_*_sol.csv"))

    json_names = {f.stem for f in json_files}
    csv_names = {f.stem.replace("_sol", "") for f in csv_files}
    common = sorted(json_names.intersection(csv_names))

    if not common:
        msg = f"No matching case_XXXX.json / case_XXXX_sol.csv found in {batch_name}"
        raise RuntimeError(msg)

    log.append(f"Found {len(common)} matching case pairs.")

    # -------------------------- helper --------------------------
    def load_case(csv_path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict[Any, Any]]:
        with meta_path.open() as f:
            meta_full = json.load(f)
        meta = {k: v for k, v in meta_full.items() if k != "parameters"}
        if "parameters" in meta_full:
            meta["parameters"] = {k: v for k, v in meta_full["parameters"].items() if k != "rng_state"}

        # Read header
        with csv_path.open() as f:
            lines = f.readlines()
        header_line = [line for line in lines if line.strip().startswith("%")][-1]
        raw_header = [h.strip() for h in header_line[1:].split(",")]

        if raw_header.count("y") > 1:
            second_y_index = [i for i, h in enumerate(raw_header) if h == "y"][1]
            raw_header[second_y_index] = "y_on"

        # Let pandas handle duplicates automatically
        df = pd.read_csv(
            csv_path,
            comment="%",
            skip_blank_lines=True,
            names=raw_header,
            index_col=False,
        )

        return df, meta

    # -------------------- load first case for preview --------------------
    first_case = common[0]
    csv_path = proc_dir / f"{first_case}_sol.csv"
    meta_path = raw_dir / f"{first_case}.json"
    df, meta = load_case(csv_path, meta_path)
    nx, ny = meta["geometry"]["nx"], meta["geometry"]["ny"]

    dropped_input_fields, dropped_output_fields = [], []

    for c in [c for c in df.columns if c.startswith("br.kappa")]:
        arr = df[c].to_numpy().reshape(ny, nx)
        if not np.any(arr):
            dropped_input_fields.append(c.replace("br.", ""))

    for key in ["u", "v", "p", "br.U"]:
        if key in df.columns:
            arr = df[key].to_numpy().reshape(ny, nx)
            if not np.any(arr):
                dropped_output_fields.append(key.replace("br.", ""))

    if verbose and (dropped_input_fields or dropped_output_fields):
        print("\n[INFO] Constant-zero fields detected and excluded:")
        if dropped_input_fields:
            print(f"  Inputs:  {', '.join(dropped_input_fields)}")
        if dropped_output_fields:
            print(f"  Outputs: {', '.join(dropped_output_fields)}")
        print("--------------------------------------------------")

    input_fields = {}
    if "x" in df.columns and "y" in df.columns:
        input_fields["x"] = df["x"].to_numpy().reshape(ny, nx)
        input_fields["y"] = df["y"].to_numpy().reshape(ny, nx)
    for c in [c for c in df.columns if c.startswith("br.kappa")]:
        clean = c.replace("br.", "")
        if clean in dropped_input_fields:  # --- added
            continue  # --- added
        input_fields[clean] = df[c].to_numpy().reshape(ny, nx)

    output_fields = {}
    for key in ["u", "v", "p", "br.U"]:
        clean = key.replace("br.", "")
        if key in df.columns and clean not in dropped_output_fields:  # --- added
            arr = df[key].to_numpy().reshape(ny, nx)
            output_fields[clean] = arr

    if verbose:
        print("\nExample structure for first case:")
        print("--------------------------------------------------")
        print("input_fields:")
        for k, v in input_fields.items():
            print(f"  {k:10s}  shape={v.shape}, dtype={v.dtype}")
        print("output_fields:")
        for k, v in output_fields.items():
            print(f"  {k:10s}  shape={v.shape}, dtype={v.dtype}")
        print("meta keys:", list(meta.keys()))
        print("--------------------------------------------------\n")

    # -------------------------- main loop --------------------------
    pbar = tqdm(total=len(common), desc=f"Building {batch_name}", unit="file", disable=not verbose)

    for name in common:
        csv_path = proc_dir / f"{name}_sol.csv"
        meta_path = raw_dir / f"{name}.json"
        df, meta = load_case(csv_path, meta_path)
        nx, ny = meta["geometry"]["nx"], meta["geometry"]["ny"]

        input_fields = {}
        if "x" in df.columns and "y" in df.columns:
            input_fields["x"] = df["x"].to_numpy().reshape(ny, nx)
            input_fields["y"] = df["y"].to_numpy().reshape(ny, nx)
        for c in [c for c in df.columns if c.startswith("br.kappa")]:
            clean = c.replace("br.", "")
            if clean in dropped_input_fields:  # --- added
                continue  # --- added
            input_fields[clean] = df[c].to_numpy().reshape(ny, nx)

        output_fields = {}
        for key in ["u", "v", "p", "br.U"]:
            clean = key.replace("br.", "")
            if key not in df.columns or clean in dropped_output_fields:  # --- added
                continue  # --- added
            arr = df[key].to_numpy().reshape(ny, nx)
            output_fields[clean] = arr

        data_case = {
            "input_fields": input_fields,
            "output_fields": output_fields,
            "meta": meta,
        }
        torch.save(data_case, cases_dir / f"{name}.pt")
        pbar.update(1)

    pbar.close()

    # ---------------------------- meta ----------------------------
    meta_json_path = meta_dir / f"{batch_name}.json"
    meta_csv_path = meta_dir / f"{batch_name}.csv"
    meta_saved = False

    if meta_json_path.exists() and meta_csv_path.exists():
        with meta_json_path.open() as f:
            meta_json = json.load(f)
        meta_csv = pd.read_csv(meta_csv_path)
        meta_struct = {"json": meta_json, "csv": meta_csv.to_dict(orient="list")}
        torch.save(meta_struct, out_batch / "meta.pt")
        meta_saved = True
        log.append("Meta file saved.")
    else:
        log.append("No meta files found.")

    return {
        "batch_name": batch_name,
        "n_cases": len(common),
        "cases_dir": cases_dir,
        "out_batch": out_batch,
        "meta_saved": meta_saved,
        "log": log,
    }


if __name__ == "__main__":
    result = build_batch_dataset("samples_uniform_var20_N1000", verbose=True)
    for line in result["log"]:
        print(line)
