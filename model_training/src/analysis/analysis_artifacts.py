"""
Create persistent evaluation artifacts for PINO/FNO models.

Generated artifacts:
    - One Parquet file per dataset (global statistics)
    - One NPZ file per case (prediction, ground truth, error, meta)
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


def meta_to_jsonable(obj: Any) -> Any:  # noqa: PLR0911
    """Recursively convert tensors, numpy types and arrays into JSON-serialisable Python types."""
    # torch tensors
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy scalar types
    if isinstance(obj, NUMPY_INT_TYPES):  # pyright: ignore[reportArgumentType]
        return int(obj)

    if isinstance(obj, NUMPY_FLOAT_TYPES):  # pyright: ignore[reportArgumentType]
        return float(obj)

    # dict recurse
    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    # list/tuple recurse
    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    # already JSON safe
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

    The dataset returns:
        batch["x"] → Tensor [B, C_in,  H, W]
        batch["y"] → Tensor [B, C_out, H, W]

    NeuralOp expects [B, C, H, W].
    Therefore no additional unsqueeze is needed.
    """
    model.eval()

    save_root = Path(save_root)
    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for idx, batch in enumerate(loader):
        case_id = idx + 1

        # Already batched (B=1)
        x_tensor = batch["x"].to(device)
        y_tensor = batch["y"].to(device)

        # Original metadata (may contain numpy/torch types)
        meta_raw = batch.get("meta", {})

        # Clean to JSON-safe
        meta_clean = meta_to_jsonable(meta_raw)

        # ------------------------------------------------------------------ #
        # Normalisation (NeuralOp API)
        # ------------------------------------------------------------------ #
        proc_in = processor.preprocess({"x": x_tensor, "y": y_tensor})
        x_norm = proc_in["x"]

        with torch.no_grad():
            y_hat_norm = model(x_norm)
            y_hat, _ = processor.postprocess(y_hat_norm, {"y": y_tensor})

        # ------------------------------------------------------------------ #
        # Errors
        # ------------------------------------------------------------------ #
        err = y_hat - y_tensor
        l2 = torch.linalg.norm(err).item()
        rel_l2 = l2 / (torch.linalg.norm(y_tensor).item() + 1e-12)

        # ------------------------------------------------------------------ #
        # Save NPZ
        # ------------------------------------------------------------------ #
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        np.savez_compressed(
            npz_path,
            pred=y_hat.detach().cpu().numpy(),
            gt=y_tensor.detach().cpu().numpy(),
            err=err.detach().cpu().numpy(),
            meta=json.dumps(meta_clean),
        )

        # For Parquet
        rows.append(
            {
                "case_index": case_id,
                "l2": l2,
                "rel_l2": rel_l2,
                "npz_path": str(npz_path),
                "meta": meta_clean,
            }
        )

    # ---------------------------------------------------------------------- #
    # Build parquet summary
    # ---------------------------------------------------------------------- #
    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
