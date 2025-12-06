"""
Create persistent evaluation artifacts for PINO and FNO models.

This module runs deterministic inference on a dataset and stores reusable
artifacts for downstream analysis and visualisation. For each dataset,
a Parquet summary file is created, and for each case an individual NPZ
file is written containing prediction fields, ground truth, errors and
metadata.

Artifacts
---------
- Parquet summary file with global per-case metrics
- One NPZ file per case containing:
  * pred_plog : model prediction in physical space
  * gt_plog   : ground truth in physical space
  * err_plog  : prediction error in physical space
  * meta      : JSON-safe metadata
  * l2_plog, rel_l2_plog

Notes
-----
Neural network outputs are expected to contain the physical fields
[p, u, v, U].

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


# =============================================================================
# JSON-SAFE CLEANER
# =============================================================================

NUMPY_INT_TYPES = (
    np.int_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)

NUMPY_FLOAT_TYPES = (
    np.float16,
    np.float32,
    np.float64,
)


def meta_to_jsonable(obj: Any) -> Any:  # noqa: PLR0911
    """
    Recursively convert numpy/tensor values into JSON-safe Python types.

    Args:
        obj (Any): Arbitrary metadata value.

    Returns:
        Any: Object converted to JSON-serializable form.

    """
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return float(arr) if arr.ndim == 0 else arr.tolist()

    if isinstance(obj, np.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()

    if isinstance(obj, NUMPY_INT_TYPES):  # pyright: ignore[reportArgumentType]
        return int(obj)

    if isinstance(obj, NUMPY_FLOAT_TYPES):  # pyright: ignore[reportArgumentType]
        return float(obj)

    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


# =============================================================================
# ARTIFACT GENERATOR
# =============================================================================


def generate_artifacts(
    *,
    model: Any,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, Path]:
    """
    Run inference on all cases and generate persistent evaluation artifacts.

    The following operations are performed for each case:
        1. Load input fields and metadata.
        2. Apply training-consistent normalisation.
        3. Run forward inference.
        4. Denormalise prediction into physical space.
        5. Compute error fields and L2 metrics.
        6. Store a NPZ file containing prediction, ground truth, error and metadata.

    A Parquet summary file is generated containing per-case metrics and
    NPZ file paths.

    Args:
        model (Any): Trained PINO or FNO model in evaluation mode.
        loader (DataLoader): Deterministic evaluation DataLoader.
        processor (Any): NeuralOp normalisation processor.
        device (torch.device): Device used for inference.
        save_root (str | Path): Root directory for storing artifacts.
        dataset_name (str): Name used for the Parquet summary file.

    Returns:
        tuple:
            df (pd.DataFrame): Summary dataframe of all cases.
            parquet_path (Path): Output path to the Parquet summary file.

    """
    model.eval()

    save_root = Path(save_root)
    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    # Normaliser tensors already on correct device
    in_mean = processor.in_normalizer.mean.to(device)
    in_std = processor.in_normalizer.std.to(device)
    out_mean = processor.out_normalizer.mean.to(device)
    out_std = processor.out_normalizer.std.to(device)

    rows: list[dict[str, Any]] = []

    for idx, batch in enumerate(loader):
        case_id = idx + 1

        # Input / ground truth
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        # Metadata
        meta_raw = batch.get("meta", {})
        meta_clean = meta_to_jsonable(meta_raw)

        # Normalisation (Z-score)
        with torch.no_grad():
            x_norm = (x - in_mean) / (in_std + 1e-12)
            y_hat_norm = model(x_norm)
            y_hat = y_hat_norm * (out_std + 1e-12) + out_mean

        # Error metrics
        err = y_hat - y
        l2 = torch.linalg.norm(err).item()
        rel_l2 = l2 / (torch.linalg.norm(y).item() + 1e-12)

        # Store NPZ
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        np.savez_compressed(
            npz_path,
            pred_plog=y_hat.cpu().numpy(),
            gt_plog=y.cpu().numpy(),
            err_plog=err.cpu().numpy(),
            meta=json.dumps(meta_clean),
        )

        rows.append(
            {
                "case_index": case_id,
                "npz_path": str(npz_path),
                "l2_plog": l2,
                "rel_l2_plog": rel_l2,
                "meta": meta_clean,
            }
        )

    # Parquet summary
    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
