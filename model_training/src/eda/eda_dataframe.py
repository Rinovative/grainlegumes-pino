"""
Utility functions for data loading and processing.

This module provides functions to:
- Load and convert PyTorch tensors to NumPy arrays
- Process COMSOL simulation case files
- Generate pandas DataFrames from simulation data
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray


def _to_numpy(value: Any) -> NDArray[np.float64] | None:
    """
    Convert arbitrary values (PyTorch tensors, lists, tuples) into NumPy arrays.

    Args:
        value: Object to convert (tensor, list, tuple, array, etc.).

    Returns:
        np.ndarray | None: Converted array, or None if conversion fails.

    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return cast("NDArray[np.float64]", arr)

    if isinstance(value, (list, tuple)):
        return np.array(value)

    try:
        return np.array(value)
    except (TypeError, ValueError):
        return None


def _iter_cases_sorted(cases_dir: Path) -> list[tuple[int, Path]]:
    """
    Collect all `case_XXXX.pt` files in numeric order.

    Args:
        cases_dir (Path): Path to the directory containing case files.

    Returns:
        list[tuple[int, Path]]: List of (case_index, file_path) pairs sorted by index.

    """
    candidates = sorted(cases_dir.glob("case_*.pt"))
    indexed: list[tuple[int, Path]] = []
    for p in candidates:
        suffix = p.stem.split("_", 1)[1]
        case_idx = int(suffix)
        indexed.append((case_idx, p))
    indexed.sort(key=lambda t: t[0])
    return indexed


def _extract_fields(sample: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Extract relevant data fields from a loaded `.pt` dictionary.

    Converts tensors to NumPy arrays and flattens the nested structure
    into a single row dictionary for DataFrame assembly.

    Args:
        sample (dict): Loaded PyTorch dictionary containing 'input_fields',
            'output_fields', and optional 'meta'.

    Returns:
        tuple:
            dict: Flat mapping of extracted fields.
            list[str]: Missing or inconsistent field names.

    """
    row: dict[str, Any] = {}
    missing: list[str] = []

    input_fields = sample.get("input_fields", {}) or {}
    for key in ("x", "y"):
        val = _to_numpy(input_fields.get(key))
        if val is None:
            missing.append(f"input_fields.{key}")
        row[key] = val

    for k, v in input_fields.items():
        if k.startswith("kappa"):
            val = _to_numpy(v)
            if val is None:
                missing.append(f"input_fields.{k}")
            row[k] = val

    output_fields = sample.get("output_fields", {}) or {}
    for key in ("p", "u", "v", "U"):
        val = _to_numpy(output_fields.get(key))
        if val is None:
            missing.append(f"output_fields.{key}")
        row[key] = val

    meta = sample.get("meta", {}) or {}
    if not isinstance(meta, dict):
        missing.append("meta (non-dict)")
        meta = {}
    row["meta"] = meta

    return row, missing


def generate_eda_dataframe(  # noqa: C901, PLR0912, PLR0915
    dataset_name: str,
    base_dir: str = "../../data/raw",
    show_progress: bool = False,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all `.pt` cases of a COMSOL batch and assemble them into a DataFrame.

    Args:
        dataset_name (str): Name of the COMSOL batch folder (e.g. "batch_var0.5").
        base_dir (str, optional): Base directory containing the batch folders.
            Defaults to "../../data/raw".
        show_progress (bool, optional): Whether to show a tqdm progress bar.
            Defaults to False.
        max_cases (int, optional): If given, limits how many cases are loaded
            (starting from the first).

    Returns:
        tuple:
            pd.DataFrame: Indexed by case number with columns
                ['x', 'y', all 'kappa*', 'p', 'u', 'v', 'U', 'meta'].
            list[str]: Log messages describing loading steps and warnings.

    Raises:
        FileNotFoundError: If required directories or `.pt` files are missing.
        RuntimeError: If a case file cannot be loaded or has unexpected structure.
        TypeError: If a `.pt` file does not contain a dict.

    """
    logs: list[str] = []
    base = Path(base_dir)
    batch_dir = base / dataset_name
    cases_dir = batch_dir / "cases"

    if not batch_dir.exists():
        msg = f"Batch directory not found: {batch_dir}"
        raise FileNotFoundError(msg)
    if not cases_dir.exists():
        msg = f"'cases' subdirectory not found: {cases_dir}"
        raise FileNotFoundError(msg)

    indexed_cases = _iter_cases_sorted(cases_dir)
    if not indexed_cases:
        msg = f"No .pt files found in {cases_dir}."
        raise FileNotFoundError(msg)

    if max_cases is not None and max_cases < len(indexed_cases):
        indexed_cases = indexed_cases[:max_cases]
        logs.append(f"[INFO] Loading only the first {max_cases} of {len(indexed_cases)} cases.")
    else:
        logs.append(f"[INFO] Loading all {len(indexed_cases)} cases.")

    logs.append(f"[INFO] Batch: '{dataset_name}' from {cases_dir}")

    # --- meta.pt loading ---
    try:
        meta_batch = torch.load(batch_dir / "meta.pt", map_location="cpu")
        if isinstance(meta_batch, dict):
            short_keys = ", ".join(list(meta_batch.keys())[:5])
            logs.append(f"[INFO] meta.pt loaded (keys: {short_keys} ...)")
        else:
            logs.append("[WARN] meta.pt present but not a dict.")
    except FileNotFoundError:
        logs.append("[INFO] meta.pt not found (optional).")
    except (OSError, RuntimeError, ValueError) as e:
        logs.append(f"[WARN] Could not load meta.pt: {e!s}")

    iterator: Iterable[tuple[int, Path]] = indexed_cases
    if show_progress:
        from tqdm.auto import tqdm  # noqa: PLC0415

        iterator = tqdm(indexed_cases, desc="Loading cases", unit="case")

    rows: list[dict[str, Any]] = []
    indices: list[int] = []
    all_kappa_keys: set[str] = set()

    for case_idx, path in iterator:
        try:
            sample = torch.load(path, map_location="cpu")
        except (OSError, RuntimeError, ValueError) as e:
            msg = f"Failed to load '{path}': {e}"
            raise RuntimeError(msg) from e

        if not isinstance(sample, dict):
            msg = f"Unexpected structure in '{path}': expected dict."
            raise TypeError(msg)

        row, missing = _extract_fields(sample)
        if missing:
            miss_list = ", ".join(missing)
            logs.append(f"[WARN] Case {case_idx:04d}: missing or inconsistent fields: {miss_list}")

        for k in row:
            if k.startswith("kappa"):
                all_kappa_keys.add(k)

        rows.append(row)
        indices.append(case_idx)

    kappa_cols = sorted(all_kappa_keys)
    preferred_order = ["x", "y", *kappa_cols, "p", "u", "v", "U", "meta"]

    df = pd.DataFrame(rows, index=indices)
    for col in preferred_order:
        if col not in df.columns:
            df[col] = None
    df = df.loc[:, preferred_order]
    df = df.sort_index()

    logs.append(f"[INFO] Final DataFrame contains {len(df)} cases.")
    logs.append(f"[INFO] kappa columns: {', '.join(kappa_cols) if kappa_cols else '-'}")

    # --- shape preview ---
    shape_keys = ["x", "y", *kappa_cols, "p", "u", "v", "U"]
    shapes_preview = {key: (None if df[key].isna().all() else getattr(df[key].dropna().iloc[0], "shape", None)) for key in shape_keys}
    logs.append(f"[INFO] Example shapes: {shapes_preview}")

    return df, logs
