"""
===============================================================================
eda_dataframe.py
===============================================================================
Load simulation case data and assemble EDA DataFrames.

Responsibilities:
  - Parse case files and optional batch metadata
  - Convert tensor and NumPy fields to consistent arrays
  - Extract input/output fields and metadata into tabular summaries
  - Resolve default data roots through common.paths

Design principles:
  - DataFrame construction is data-driven and field-name agnostic
  - Input files are read in deterministic order
  - Diagnostics describe loaded shapes and skipped fields

Boundaries:
  - Model evaluation artifacts belong to analysis.artifacts
  - Statistical evaluation plots belong to analysis.evaluation
Notes:
    Path resolution:
    - base_dir: If None, dynamically resolved via common.paths.get_data_root()
    - Ensures reproducibility across different working directories
  ===============================================================================
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch

from src import common

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray


def _to_numpy(value: Any) -> NDArray[np.float64] | None:
    """
    Convert arbitrary values (PyTorch tensors, lists, tuples) into NumPy arrays.

    Parameters
    ----------
    value : Any
        Input value to convert.

    Returns
    -------
    NDArray[np.float64] or None
        Converted NumPy array, or None if conversion is not possible.

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

    Parameters
    ----------
    cases_dir : Path
        Directory containing case files.

    Returns
    -------
    list[tuple[int, Path]]
        List of (case_index, file_path) tuples sorted by case_index.

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
    Extract all input_fields and output_fields from a case sample.

    Parameters
    ----------
    sample : dict
        Loaded case sample from a .pt file.

    Returns
    -------
    tuple:
        dict: Extracted fields as NumPy arrays.
        list: List of missing or inconsistent field names.

    """
    row: dict[str, Any] = {}
    missing: list[str] = []

    # ---------------------------
    # input_fields (ALL)
    # ---------------------------
    input_fields = sample.get("input_fields", {}) or {}
    if not isinstance(input_fields, dict):
        missing.append("input_fields (non-dict)")
        input_fields = {}

    for name, value in input_fields.items():
        arr = _to_numpy(value)
        if arr is None:
            missing.append(f"input_fields.{name}")
        row[name] = arr

    # ---------------------------
    # output_fields (ALL)
    # ---------------------------
    output_fields = sample.get("output_fields", {}) or {}
    if not isinstance(output_fields, dict):
        missing.append("output_fields (non-dict)")
        output_fields = {}

    for name, value in output_fields.items():
        arr = _to_numpy(value)
        if arr is None:
            missing.append(f"output_fields.{name}")
        row[name] = arr

    # ---------------------------
    # meta (untouched)
    # ---------------------------
    meta = sample.get("meta", {}) or {}
    if not isinstance(meta, dict):
        missing.append("meta (non-dict)")
        meta = {}
    row["meta"] = meta

    return row, missing


def generate_eda_dataframe(
    dataset_name: str,
    base_dir: str | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all `.pt` cases of a COMSOL batch and assemble them into a DataFrame.

    All input_fields and output_fields are collected automatically.
    No field names are hard-coded. Fully future-proof.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset batch.
    base_dir : str | None, optional
        Base directory containing dataset batches. If None, uses common.paths
        to resolve the data root. Default is None.
    show_progress : bool
        If True, displays a progress bar during loading.
    max_cases : int or None
        Maximum number of cases to load. If None, loads all cases.

    Returns
    -------
    tuple
        pd.DataFrame
            DataFrame containing all extracted fields and meta.
        list[str]
            Log messages generated during loading.

    """
    logs: list[str] = []

    if base_dir is None:
        base_dir = str(common.paths.get_data_root() / "raw")
    batch_dir = Path(base_dir) / dataset_name
    cases_dir = batch_dir / "cases"

    if not batch_dir.exists():
        msg = f"Batch directory not found: {batch_dir}"
        raise FileNotFoundError(msg)
    if not cases_dir.exists():
        msg = f"'cases' subdirectory not found: {cases_dir}"
        raise FileNotFoundError(msg)

    indexed_cases = _iter_cases_sorted(cases_dir)
    if not indexed_cases:
        msg = f"No .pt files found in {cases_dir}"
        raise FileNotFoundError(msg)

    if max_cases is not None and max_cases < len(indexed_cases):
        indexed_cases = indexed_cases[:max_cases]
        logs.append(f"[INFO] Loading only the first {max_cases} of {len(indexed_cases)} cases.")
    else:
        logs.append(f"[INFO] Loading all {len(indexed_cases)} cases.")

    logs.append(f"[INFO] Batch: '{dataset_name}' from {cases_dir}")

    # --- meta.pt (optional, informational only) -------------------------
    try:
        meta_batch = torch.load(batch_dir / "meta.pt", map_location="cpu")
        if isinstance(meta_batch, dict):
            keys_preview = ", ".join(list(meta_batch.keys())[:5])
            logs.append(f"[INFO] meta.pt loaded (keys: {keys_preview} ...)")
        else:
            logs.append("[WARN] meta.pt present but not a dict.")
    except FileNotFoundError:
        logs.append("[INFO] meta.pt not found (optional).")
    except (OSError, RuntimeError, ValueError) as e:
        logs.append(f"[WARN] Could not load meta.pt: {e!s}")

    iterator: Iterable[tuple[int, Path]] = indexed_cases
    if show_progress:
        from tqdm.auto import tqdm  # noqa: PLC0415

        iterator = tqdm(iterator, desc="Loading cases", unit="case")

    rows: list[dict[str, Any]] = []
    indices: list[int] = []

    # ------------------------------------------------------------------
    # Main loading loop (FULLY DATA-DRIVEN)
    # ------------------------------------------------------------------
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
            logs.append(f"[WARN] Case {case_idx:04d}: missing or inconsistent fields: {', '.join(missing)}")

        rows.append(row)
        indices.append(case_idx)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, index=indices).sort_index()

    logs.append(f"[INFO] Final DataFrame contains {len(df)} cases.")
    logs.append(f"[INFO] Columns: {', '.join(df.columns)}")

    # --- shape preview (best-effort) -----------------------------------
    shapes_preview: dict[str, Any] = {}
    for col in df.columns:
        series = df[col].dropna()
        if not series.empty:
            shapes_preview[col] = getattr(series.iloc[0], "shape", None)
        else:
            shapes_preview[col] = None

    logs.append(f"[INFO] Example shapes: {shapes_preview}")

    return df, logs
