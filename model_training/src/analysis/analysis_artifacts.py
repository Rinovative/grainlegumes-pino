"""
Create persistent evaluation artifacts for PINO/FNO models.

Generated artifacts:
    - One Parquet file per dataset (global statistics)
    - One NPZ file per case (prediction, ground truth, error, meta)

Note:
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


# ----------------------------------------------------------------------------- #
# JSON-SAFE CLEANER                                                            #
# ----------------------------------------------------------------------------- #

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
    """Recursively convert tensors, numpy types and arrays into JSON-serialisable Python types."""
    # torch Tensor
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1 and arr.size == 1:
            return float(arr[0])
        return arr.tolist()

    # numpy array
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return float(obj)
        if obj.ndim == 1 and obj.size == 1:
            return float(obj[0])
        return obj.tolist()

    # numpy scalar ints/floats
    if isinstance(obj, NUMPY_INT_TYPES):  # pyright: ignore[reportArgumentType]
        return int(obj)
    if isinstance(obj, NUMPY_FLOAT_TYPES):  # pyright: ignore[reportArgumentType]
        return float(obj)

    # dict
    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


# ----------------------------------------------------------------------------- #
# ARTIFACT GENERATOR                                                           #
# ----------------------------------------------------------------------------- #


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
    Run inference over all cases and create reusable artifacts.

    Stored values are ONLY:
        • pred_plog
        • gt_plog
        • err_plog
        • meta
        • l2_plog / rel_l2_plog
    """
    model.eval()

    save_root = Path(save_root)
    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, batch in enumerate(loader):
        case_id = idx + 1

        # Load fields
        x_tensor = batch["x"].to(device)
        y_tensor = batch["y"].to(device)
        meta_raw = batch.get("meta", {})
        meta_clean = meta_to_jsonable(meta_raw)

        # NeuralOp preprocess
        proc_in = processor.preprocess({"x": x_tensor, "y": y_tensor})
        x_norm = proc_in["x"]

        with torch.no_grad():
            y_hat_norm = model(x_norm)
            y_hat, _ = processor.postprocess(y_hat_norm, {"y": y_tensor})

        # Errors in NN space (plog)
        err_plog = y_hat - y_tensor
        l2_plog = torch.linalg.norm(err_plog).item()
        rel_l2_plog = l2_plog / (torch.linalg.norm(y_tensor).item() + 1e-12)

        # Save NPZ
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        np.savez_compressed(
            npz_path,
            pred_plog=y_hat.detach().cpu().numpy(),
            gt_plog=y_tensor.detach().cpu().numpy(),
            err_plog=err_plog.detach().cpu().numpy(),
            meta=json.dumps(meta_clean),
        )

        # Add row to Parquet
        rows.append(
            {
                "case_index": case_id,
                "npz_path": str(npz_path),
                "l2_plog": l2_plog,
                "rel_l2_plog": rel_l2_plog,
                "meta": meta_clean,
            }
        )

    # Write Parquet
    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
