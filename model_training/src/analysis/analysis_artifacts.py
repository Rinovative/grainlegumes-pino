"""
Create persistent evaluation artifacts for PINO/FNO models.

Generated artifacts:
    - One Parquet file per dataset (global statistics)
    - One NPZ file per case (prediction, ground truth, error, meta)

Notes
-----
NN outputs are fields [p, u, v, U].

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


def meta_to_jsonable(obj: Any) -> Any:  # noqa: C901, PLR0911
    """Recursively convert tensors, numpy values and arrays into JSON-safe Python objects."""
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1 and arr.size == 1:
            return float(arr[0])
        return arr.tolist()

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return float(obj)
        if obj.ndim == 1 and obj.size == 1:
            return float(obj[0])
        return obj.tolist()

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
    Run inference on all cases and store reusable artifacts.

    The physical prediction is obtained as:
        y_hat = processor.out_normalizer.inverse_transform(y_hat_norm)

    `postprocess` from DefaultDataProcessor is NOT used here because it
    merges dicts and causes type confusion when metadata contains strings.

    Stored per-case fields:
        - pred_plog  : denormalised prediction (physical space)
        - gt_plog    : ground truth
        - err_plog   : absolute error (physical space)
        - meta       : JSON-safe case metadata
        - l2_plog / rel_l2_plog
    """
    model.eval()
    save_root = Path(save_root)
    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, batch in enumerate(loader):
        case_id = idx + 1

        # --- get fields ---
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        meta_raw = batch.get("meta", {})
        meta_clean = meta_to_jsonable(meta_raw)

        # --- forward ---
        with torch.no_grad():
            x_norm = processor.in_normalizer.transform(x)
            y_hat_norm = model(x_norm)

            # denormalisation without touching meta
            y_hat = processor.out_normalizer.inverse_transform(y_hat_norm)

        # --- errors ---
        err = y_hat - y
        l2 = torch.linalg.norm(err).item()
        rel_l2 = l2 / (torch.linalg.norm(y).item() + 1e-12)

        # --- write NPZ ---
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

    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
