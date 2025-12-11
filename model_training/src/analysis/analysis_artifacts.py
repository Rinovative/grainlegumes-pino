"""
Create persistent evaluation artifacts for PINO and FNO models.

This module runs deterministic inference on simulation datasets and stores
reusable artifacts for all downstream evaluation and visualisation modules.

High-level design
-----------------
Each dataset is evaluated once and produces:

1) A *lightweight* Parquet summary table (one row per case) with:
    - global error metrics (L2, relative L2)
    - basic permeability statistics (per channel)
    - the list of available kappa tensor components
    - the file path to the dense NPZ artifact
    - JSON-safe metadata (geometry, sampling parameters, etc.)

2) One *dense* NPZ file per case containing the full fields:
    - pred        : model prediction in physical space
    - gt          : ground truth in physical space
    - err         : prediction error field (pred - gt)
    - kappa_log   : log10-permeability tensor components
    - kappa       : physical permeability tensor components
    - kappa_names : list of component names (e.g. ["kappaxx", "kappayy"])
    - meta        : JSON-safe metadata dictionary

This separation keeps the Parquet table fast to load and aggregate
(global comparisons, filters, correlations) while the NPZ files hold
all spatial information needed for local, high-resolution analysis.

Artifacts
---------
Parquet (global, per-case):
    - case_index        : integer case id
    - npz_path          : path to the corresponding NPZ artifact
    - l2                : global L2 error over the full domain
    - rel_l2            : global relative L2 error
    - kappa_mean        : per-channel mean(kappa) over the domain
    - kappa_std         : per-channel std(kappa)
    - kappa_min         : per-channel min(kappa)
    - kappa_max         : per-channel max(kappa)
    - kappa_names       : list of available tensor components
    - meta              : JSON-serialisable metadata dict
                          (can include geometry and sampling parameters)

NPZ (local, full fields per case):
    - pred        : (C_out, H, W)
    - gt          : (C_out, H, W)
    - err         : (C_out, H, W)
    - kappa_log   : (C_kappa, H, W)
    - kappa       : (C_kappa, H, W)
    - kappa_names : list[str], same order as kappa channels
    - meta        : JSON-safe dict with arbitrary metadata

Operational Notes
-----------------
Input tensors are assumed to follow the canonical layout

    [x, y,
     kappaxx, kappayx, kappazx,
     kappaxy, kappayy, kappazy,
     kappaxz, kappayz, kappazz]

where:
    - channels 0-1 are spatial coordinates (x, y)
    - channels 2.. are log10-permeability components.

Datasets may contain *partial* permeability information, for example:
    - only kappaxx and kappayy
    - symmetric 2D tensor components (xx, xy, yx, yy)
    - the full 3x3 tensor (9 components)

This module automatically detects how many kappa components exist and
assigns their canonical names accordingly.

Output tensors (pred / gt / err) are expected to follow

    [p, u, v, U]

with:
    - p : pressure
    - u : x-velocity
    - v : y-velocity
    - U : velocity magnitude

Downstream usage
----------------
The resulting artifacts are consumed by the evaluation plots, e.g.:

    • Global error metrics (1-1):
        - uses df["l2"], df["rel_l2"] for violin/KDE/CDF plots.

    • Global/local error distribution viewer (1-2):
        - uses df["l2"] for global summaries and
          df["npz_path"] to load 'gt' and 'err' and build
          pixel-wise local error distributions on the fly.

    • Global GT vs Pred means (1-3):
        - uses df["npz_path"] to load 'gt' and 'pred', then
          computes per-case mean values per channel.

    • Mean error maps (1-4):
        - uses df["npz_path"] to load 'err' and aggregate
          global mean absolute error maps per dataset/channel.

    • 4x4 interactive sample viewer:
        - uses df["npz_path"] for pred/gt/err/kappa and
          metadata entries like geom_Lx / geom_Ly (if stored in meta)
          to construct consistent physical coordinate axes.

Local information (error maps, relative error, kappa aggregation) is
therefore *not* pre-aggregated in the Parquet file, but derived directly
from the NPZ artifacts for maximal flexibility and reproducibility.

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


# ============================================================================
# TYPE NORMALISATION
# ============================================================================

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
    Convert tensors, numpy values and nested structures into JSON-safe types.

    This function recursively transforms:
        - torch.Tensor → float or list
        - numpy.ndarray → float or list
        - numpy scalar types → Python scalars
        - dict/list/tuple → recursively processed structures.

    Parameters
    ----------
    obj : Any
        Arbitrary Python object.

    Returns
    -------
    Any
        JSON-serialisable representation.

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


# ============================================================================
# KAPPA EXTRACTION UTILITIES
# ============================================================================

CANONICAL_KAPPA = [
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


def detect_kappa_channels(x_tensor: torch.Tensor) -> list[str]:
    """
    Determine which kappa tensor components exist in the input tensor.

    Assumes:
        x_tensor shape = [1, C_in, H, W]
        channels 0-1 = spatial coordinates (x, y)
        channels 2.. = kappa components.

    Parameters
    ----------
    x_tensor : Tensor
        Input tensor from the dataset.

    Returns
    -------
    list[str]
        Canonical kappa component names matching the available channels.

    """
    num_kappa = x_tensor.shape[1] - 2
    return CANONICAL_KAPPA[:num_kappa]


def extract_kappa(x_tensor: torch.Tensor, kappa_names: list[str]) -> dict[str, Any]:
    """
    Extract log-kappa channels, convert them to physical values, and compute basic per-channel statistics.

    Parameters
    ----------
    x_tensor : Tensor
        Input tensor with shape [1, C_in, H, W].
    kappa_names : list[str]
        Names associated with the available kappa channels.

    Returns
    -------
    dict
        {
            "kappa_log": Tensor,
            "kappa": Tensor,
            "stats": {
                "mean": list[float],
                "std":  list[float],
                "min":  list[float],
                "max":  list[float],
            }
        }

    """
    kappa_log = x_tensor[:, 2 : 2 + len(kappa_names), :, :]
    kappa = torch.pow(10.0, kappa_log)

    stats = {
        "mean": kappa.mean(dim=(2, 3)).squeeze(0).cpu().tolist(),
        "std": kappa.std(dim=(2, 3)).squeeze(0).cpu().tolist(),
        "min": kappa.amin(dim=(2, 3)).squeeze(0).cpu().tolist(),
        "max": kappa.amax(dim=(2, 3)).squeeze(0).cpu().tolist(),
    }

    return {
        "kappa_log": kappa_log,
        "kappa": kappa,
        "stats": stats,
    }


# ============================================================================
# MAIN ARTIFACT GENERATOR
# ============================================================================


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

    For each case, the following operations are executed:
        1. Load model inputs (x), outputs (y) and metadata.
        2. Detect which permeability components are present.
        3. Extract log-kappa fields and convert them to physical kappa.
        4. Apply the training-consistent normalisation pipeline.
        5. Perform forward inference.
        6. Denormalise predictions to physical output space.
        7. Compute error fields and L2 / relative L2 global metrics.
        8. Write one NPZ file containing predictions, ground truth,
           kappa tensors and metadata.

    After processing all cases, a Parquet summary file is written that
    includes:
        - global error metrics (L2, rel L2)
        - kappa statistics (mean, std, min, max)
        - list of available tensor components
        - path to NPZ artifacts
        - cleaned metadata.

    Parameters
    ----------
    model : Any
        Trained neural operator model (PINO or FNO).
    loader : DataLoader
        Deterministic evaluation loader.
    processor : Any
        NeuralOp normalisation processor used during training.
    device : torch.device
        Device used for inference.
    save_root : str or Path
        Directory into which all artifacts are written.
    dataset_name : str
        Base name for the Parquet summary file.

    Returns
    -------
    df : pandas.DataFrame
        Per-case summary table with metrics and statistics.
    parquet_path : pathlib.Path
        Output path of the generated Parquet file.

    """
    model.eval()

    save_root = Path(save_root)
    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    # Normalisation tensors
    in_mean = processor.in_normalizer.mean.to(device)
    in_std = processor.in_normalizer.std.to(device)
    out_mean = processor.out_normalizer.mean.to(device)
    out_std = processor.out_normalizer.std.to(device)

    rows: list[dict[str, Any]] = []

    # Detect kappa channels once from the first batch
    first_batch = next(iter(loader))
    kappa_names = detect_kappa_channels(first_batch["x"])

    # ---------------------------------------
    # Main evaluation loop
    # ---------------------------------------
    for idx, batch in enumerate(loader):
        case_id = idx + 1

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        meta_clean = meta_to_jsonable(batch.get("meta", {}))

        # Extract permeabilities
        kappa_info = extract_kappa(x, kappa_names)
        kappa_log = kappa_info["kappa_log"]
        kappa = kappa_info["kappa"]
        stats = kappa_info["stats"]

        # Forward pass
        with torch.no_grad():
            x_norm = (x - in_mean) / (in_std + 1e-12)
            y_hat_norm = model(x_norm)
            y_hat = y_hat_norm * (out_std + 1e-12) + out_mean

        err = y_hat - y
        l2 = torch.linalg.norm(err).item()
        rel_l2 = l2 / (torch.linalg.norm(y).item() + 1e-12)

        # NPZ path
        npz_path = npz_dir / f"case_{case_id:04d}.npz"

        # Save per-case artifact
        np.savez_compressed(
            npz_path,
            pred=y_hat.cpu().numpy(),
            gt=y.cpu().numpy(),
            err=err.cpu().numpy(),
            kappa_log=kappa_log.cpu().numpy(),
            kappa=kappa.cpu().numpy(),
            kappa_names=np.array(kappa_names, dtype=object),
            meta=json.dumps(meta_clean),
        )

        rows.append(
            {
                "case_index": case_id,
                "npz_path": str(npz_path),
                "l2": l2,
                "rel_l2": rel_l2,
                "kappa_mean": stats["mean"],
                "kappa_std": stats["std"],
                "kappa_min": stats["min"],
                "kappa_max": stats["max"],
                "kappa_names": kappa_names,
                "meta": meta_clean,
            }
        )

    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
