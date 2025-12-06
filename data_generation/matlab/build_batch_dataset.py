"""
===============================================================================

 build_batch_dataset.
===============================================================================
Author:  Rino M. Albertin
Date:    2025-10-28
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Reads raw COMSOL simulation outputs and converts them into structured PyTorch
`.pt` case files for PINO/FNO training and evaluation.

Each case file contains:
    - input_fields:   x, y, kappa* tensors (kappa already log10-transformed)
    - output_fields:  p, u, v, U
    - meta:           simulation metadata

Directory structure created:

    data/raw/<batch_name>/
        ├── cases/       (case_XXXX.pt)
        └── meta.pt      (optional batch metadata)
===============================================================================
"""  # noqa: INP001

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def build_batch_dataset(batch_name: str, verbose: bool = False) -> dict:  # noqa: C901, PLR0912, PLR0915
    """
    Convert one COMSOL batch into structured `.pt` case files.

    Parameters
    ----------
    batch_name : str
        Name of the simulation batch.
    verbose : bool
        If True, print additional information.

    Returns
    -------
    dict
        Summary information for logging.

    """
    log = []

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

    json_files = sorted(raw_dir.glob("case_*.json"))
    csv_files = sorted(proc_dir.glob("case_*_sol.csv"))

    json_names = {f.stem for f in json_files}
    csv_names = {f.stem.replace("_sol", "") for f in csv_files}
    common = sorted(json_names.intersection(csv_names))

    if not common:
        msg = f"No valid matching cases found for {batch_name}"
        raise RuntimeError(msg)

    log.append(f"Found {len(common)} matching cases.")

    def load_case(csv_path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        with meta_path.open() as f:
            meta_full = json.load(f)

        meta = {k: v for k, v in meta_full.items() if k != "parameters"}
        if "parameters" in meta_full:
            meta["parameters"] = {k: v for k, v in meta_full["parameters"].items() if k != "rng_state"}

        with csv_path.open() as f:
            lines = f.readlines()

        header_line = [line for line in lines if line.strip().startswith("%")][-1]
        raw_header = [h.strip() for h in header_line[1:].split(",")]

        if raw_header.count("y") > 1:
            idx = [i for i, h in enumerate(raw_header) if h == "y"][1]
            raw_header[idx] = "y_on"

        df = pd.read_csv(
            csv_path,
            comment="%",
            skip_blank_lines=True,
            names=raw_header,
            index_col=False,
        )

        return df, meta

    first = common[0]
    df_first, meta_first = load_case(proc_dir / f"{first}_sol.csv", raw_dir / f"{first}.json")
    nx, ny = meta_first["geometry"]["nx"], meta_first["geometry"]["ny"]

    dropped_input = []
    dropped_output = []

    for c in [c for c in df_first.columns if c.startswith("br.kappa")]:
        arr = df_first[c].to_numpy().reshape(ny, nx)
        if not np.any(arr):
            dropped_input.append(c.replace("br.", ""))

    for key in ["p", "u", "v", "br.U"]:
        if key in df_first.columns:
            arr = df_first[key].to_numpy().reshape(ny, nx)
            if not np.any(arr):
                dropped_output.append(key.replace("br.", ""))

    input_fields_preview = {}
    output_fields_preview = {}

    if "x" in df_first.columns and "y" in df_first.columns:
        input_fields_preview["x"] = df_first["x"].to_numpy().reshape(ny, nx)
        input_fields_preview["y"] = df_first["y"].to_numpy().reshape(ny, nx)

    eps = 1e-12
    for col in [c for c in df_first.columns if c.startswith("br.kappa")]:
        clean = col.replace("br.", "")
        if clean in dropped_input:
            continue
        raw = df_first[col].to_numpy().reshape(ny, nx)
        input_fields_preview[clean] = np.log10(raw + eps)

    for key in ["p", "u", "v", "br.U"]:
        clean = key.replace("br.", "")
        if key in df_first.columns and clean not in dropped_output:
            output_fields_preview[clean] = df_first[key].to_numpy().reshape(ny, nx)

    if verbose:
        print("\nExample structure for first case:")
        print("--------------------------------------------------")
        print("input_fields:")
        for k, v in input_fields_preview.items():
            print(f"  {k:10s}  shape={v.shape}, dtype={v.dtype}")
        print("output_fields:")
        for k, v in output_fields_preview.items():
            print(f"  {k:10s}  shape={v.shape}, dtype={v.dtype}")
        print("meta keys:", list(meta_first.keys()))
        print("--------------------------------------------------\n")

    pbar = tqdm(
        total=len(common),
        desc=f"Building {batch_name}",
        unit="file",
        disable=not verbose,
    )

    for name in common:
        csv_path = proc_dir / f"{name}_sol.csv"
        meta_path = raw_dir / f"{name}.json"

        df, meta = load_case(csv_path, meta_path)
        nx, ny = meta["geometry"]["nx"], meta["geometry"]["ny"]

        input_fields: dict[str, np.ndarray] = {}
        output_fields: dict[str, np.ndarray] = {}

        if "x" in df.columns and "y" in df.columns:
            input_fields["x"] = df["x"].to_numpy().reshape(ny, nx)
            input_fields["y"] = df["y"].to_numpy().reshape(ny, nx)

        eps = 1e-12
        for col in [c for c in df.columns if c.startswith("br.kappa")]:
            clean = col.replace("br.", "")
            if clean in dropped_input:
                continue
            raw = df[col].to_numpy().reshape(ny, nx)
            input_fields[clean] = np.log10(raw + eps)

        for key in ["p", "u", "v", "br.U"]:
            clean = key.replace("br.", "")
            if key in df.columns and clean not in dropped_output:
                output_fields[clean] = df[key].to_numpy().reshape(ny, nx)

        data_case = {
            "input_fields": input_fields,
            "output_fields": output_fields,
            "meta": meta,
        }

        torch.save(data_case, cases_dir / f"{name}.pt")
        pbar.update(1)

    pbar.close()

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
        log.append("Saved meta.pt")
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
    result = build_batch_dataset("lhs_var20_plog100_seed9", verbose=True)
    for line in result["log"]:
        print(line)
