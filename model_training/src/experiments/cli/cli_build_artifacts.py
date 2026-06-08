"""
===============================================================================
cli_build_artifacts.py
===============================================================================
Generate analysis artifacts for trained model runs.

Responsibilities:
  - Discover trained model directories
  - Load saved inference contexts for each model/dataset pair
  - Generate or reuse Parquet and NPZ analysis artifacts
  - Release GPU memory between model runs

Design principles:
  - CLI orchestration stays thin
  - Existing artifacts are reused when available
  - Inference and artifact generation are delegated to reusable modules

Boundaries:
  - Inference context loading belongs to learning.inference.context
  - Artifact writing belongs to analysis.artifacts

Notes:
  - Models are discovered by presence of checkpoints in PROCESSED_ROOT
  - Artifacts are cached in model-specific analysis subdirectories
  - GPU memory is aggressively cleaned after each model to prevent fragmentation
===============================================================================

"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

from src import analysis, learning

if TYPE_CHECKING:
    from collections.abc import Iterable

# ======================================================================
# Global config
# ======================================================================
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "model_training" / "data" / "processed"

ID_DATASET = "lhs_var80_seed3001"
OOD_DATASETS = ["lhs_var120_seed4001"]

# Case limit for testing / debugging (set to None for full runs)
MAX_CASES: int | None = None
# ======================================================================
# Utilities
# ======================================================================


def cleanup_gpu() -> None:
    """
    Aggressively clean GPU memory after inference.

    Performs garbage collection, clears CUDA cache, and collects IPC handles
    to free GPU memory for the next batch. Safe to call when CUDA is unavailable.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def iter_model_dirs(root: Path) -> Iterable[str]:
    """
    Iterate over all model directories that contain trained checkpoints.

    A directory is considered a valid model if it contains either
    `best_model_state_dict.pt` or `model_state_dict.pt`.

    Parameters
    ----------
    root : Path
        Root directory containing model run subdirectories.

    Yields
    ------
    str
        Name (basename) of each valid model directory, in sorted order.

    """
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue

        if (p / "best_model_state_dict.pt").exists() or (p / "model_state_dict.pt").exists():
            yield p.name


def run_or_load_artifacts(
    *,
    model_name: str,
    dataset_name: str,
    max_cases: int | None,
) -> pd.DataFrame:
    """
    Load or generate artifacts for one (model, dataset) pair.

    Checks for existing artifacts (Parquet or NPZ) and loads them if available.
    If artifacts don't exist, runs deterministic inference via
    learning.inference.context.load_inference_context() and
    generates Parquet+NPZ artifacts via analysis.artifacts.generate_artifacts().

    Parameters
    ----------
    model_name : str
        Model directory name (assumed to exist in PROCESSED_ROOT).
    dataset_name : str
        Dataset name for evaluation (used to locate raw data and save artifacts).
    max_cases : int or None
        Maximum cases to process. If None, processes all cases in the dataset.

    Returns
    -------
    pd.DataFrame
        Artifact summary DataFrame (empty if artifacts were not generated or found).

    Notes
    -----
    - Checkpoint priority: best_model_state_dict.pt > model_state_dict.pt
    - Artifacts cached in model_name/analysis/{id,ood}/dataset_name/
    - Exceptions during inference are logged and return empty DataFrame (no re-raise)

    """
    run_dir = PROCESSED_ROOT / model_name
    best_ckpt = run_dir / "best_model_state_dict.pt"
    last_ckpt = run_dir / "model_state_dict.pt"

    if best_ckpt.exists():
        checkpoint_path = best_ckpt
    elif last_ckpt.exists():
        checkpoint_path = last_ckpt
    else:
        msg = f"No checkpoint found in {run_dir}"
        raise FileNotFoundError(msg)
    dataset_path = RAW_ROOT / dataset_name / "cases"

    save_root = run_dir / "analysis" / "id" if dataset_name == ID_DATASET else run_dir / "analysis" / "ood" / dataset_name
    npz_dir = save_root / "npz"
    parquet_path = save_root / f"{dataset_name}.parquet"

    print(f"[RUN] {model_name} | {dataset_name}")
    print(f"      checkpoint={checkpoint_path}")
    print(f"      save_root={save_root}")

    # --------------------------------------------------
    # HARD SKIP: artifacts already exist
    # --------------------------------------------------
    if parquet_path.exists():
        print(f"[LOAD] {model_name} | {dataset_name} (parquet)")
        return pd.read_parquet(parquet_path)

    if npz_dir.exists() and any(npz_dir.glob("*.npz")):
        print(f"[SKIP] {model_name} | {dataset_name} (npz exists, no parquet)")
        return pd.DataFrame()

    try:
        model, loader, processor, device = learning.inference.context.load_inference_context(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            batch_size=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] {model_name} | {dataset_name}")
        print(f"       Reason: {type(e).__name__}: {e}")
        return pd.DataFrame()

    df, _ = analysis.artifacts.generate_artifacts(
        model=model,
        loader=loader,
        processor=processor,
        device=device,
        save_root=save_root,
        dataset_name=dataset_name,
        max_cases=max_cases,
    )

    # explizit loeschen
    del model, loader, processor
    cleanup_gpu()

    return df


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """
    Generate all analysis artifacts for all trained models and datasets.

    Orchestrates evaluation across all models found in PROCESSED_ROOT and
    all datasets (ID_DATASET and OOD_DATASETS). For each (model, dataset) pair,
    calls run_or_load_artifacts() to generate or retrieve cached artifacts.

    Configuration (hardcoded at module level):
        - ID_DATASET: In-distribution dataset name
        - OOD_DATASETS: Out-of-distribution dataset names
        - MAX_CASES: Case limit per dataset (None = all)

    Notes
    -----
    - Skips models if ID evaluation fails (no OOD evaluation for that model)
    - Cleans GPU memory after each model to prevent fragmentation
    - Prints detailed [INFO], [LOAD], [SKIP], [RUN] messages for debugging

    """
    model_names = list(iter_model_dirs(PROCESSED_ROOT))
    print(f"[INFO] Found {len(model_names)} models")

    for model_name in model_names:
        print(f"\n=== {model_name} ===")

        # -----------------
        # ID
        # -----------------
        df_raw_id = run_or_load_artifacts(
            model_name=model_name,
            dataset_name=ID_DATASET,
            max_cases=MAX_CASES,
        )
        if df_raw_id.empty:
            print(f"[SKIP] {model_name} | ID evaluation skipped")
            continue
        _ = analysis.evaluation.dataframe.build_eval_df(df_raw_id)

        # -----------------
        # OOD
        # -----------------
        for ood in OOD_DATASETS:
            df_raw_ood = run_or_load_artifacts(
                model_name=model_name,
                dataset_name=ood,
                max_cases=MAX_CASES,
            )

            if df_raw_ood.empty:
                print(f"[SKIP] {model_name} | OOD {ood} skipped")
                continue

            _ = analysis.evaluation.dataframe.build_eval_df(df_raw_ood)

        cleanup_gpu()

    print("\n[DONE] All artifacts generated.")


if __name__ == "__main__":
    main()
