"""
Inference utilities for PINO/FNO model evaluation.

This module provides a clean, reproducible inference pipeline.
It reconstructs the full evaluation context without relying on
`neuralop.training.load_training_state`. Instead, everything needed
for inference is rebuilt manually and deterministically.

Pipeline steps:

1. Load `config.json` from the run directory
2. Rebuild model architecture from the configuration
3. Load the `state_dict` (e.g., best_model_state_dict.pt)
4. Load the dataset (`PermeabilityFlowDataset`)
5. Recompute input/output normalisation using **all samples**
   (mirrors your training behaviour using full_train)
6. Optionally subsample the dataset for OOD evaluation
7. Build a clean evaluation DataLoader

The function `load_inference_context()` returns:

    model, loader, processor, device

Ready to run inference, evaluation, sensitivity analysis, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.models import FNO
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset.dataset_simulation import PermeabilityFlowDataset

if TYPE_CHECKING:
    from torch import Tensor


# -----------------------------------------------------------------------------#
# Config loading                                                               #
# -----------------------------------------------------------------------------#
def _load_config(config_path: Path) -> dict[str, Any]:
    """
    Load the training configuration stored in `config.json`.

    Args:
        config_path: Full path to config.json.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the JSON file is missing.

    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------#
# Model reconstruction                                                         #
# -----------------------------------------------------------------------------#
def _build_model_from_config(model_cfg: dict[str, Any]) -> nn.Module:
    """
    Rebuild the model architecture exactly as it was during training.

    The config contains:
        - "architecture": e.g. "FNO"
        - "model_params": dict with ALL model hyperparameters
          (as saved by the training pipeline)

    This function also fixes JSON serialization issues:
        neuralop stores skip-connection types internally as tuples,
        which become lists in config.json. We convert them back to strings.

    Args:
        model_cfg: The `"model"` section of config.json.

    Returns:
        The instantiated model, fully matching the training configuration.

    Raises:
        NotImplementedError: On unknown model architectures.

    """
    arch = model_cfg["architecture"]
    params = dict(model_cfg["model_params"])  # safe copy

    if arch != "FNO":
        msg = f"Unknown architecture: {arch}"
        raise NotImplementedError(msg)

    # --- Fix JSON → tuple → list serialization for skip types ----------------
    for key in ["channel_mlp_skip", "fno_skip"]:
        val = params.get(key)
        if isinstance(val, list) and len(val) == 1:
            params[key] = val[0]  # convert ["soft-gating"] → "soft-gating"

    # --- Instantiate model with full parameter set ---------------------------
    return FNO(**params)


# -----------------------------------------------------------------------------#
# Normalisation reconstruction                                                 #
# -----------------------------------------------------------------------------#
def _build_normalizer_from_dataset(
    dataset: PermeabilityFlowDataset,
) -> DefaultDataProcessor:
    """
    Recompute input/output normalisation using all dataset samples.

    Mirrors the training behaviour where global mean/std are computed
    over the full in-distribution dataset.

    Args:
        dataset: Instance of PermeabilityFlowDataset.

    Returns:
        DefaultDataProcessor with fitted input/output normalizers.

    """
    xs: list[Tensor] = []
    ys: list[Tensor] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        xs.append(sample["x"])
        ys.append(sample["y"])

    x_tensor = torch.stack(xs, dim=0)
    y_tensor = torch.stack(ys, dim=0)

    in_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
    out_norm = UnitGaussianNormalizer(dim=[0, 2, 3])

    in_norm.fit(x_tensor)
    out_norm.fit(y_tensor)

    return DefaultDataProcessor(
        in_normalizer=in_norm,
        out_normalizer=out_norm,
    )


# -----------------------------------------------------------------------------#
# Evaluation DataLoader                                                        #
# -----------------------------------------------------------------------------#
def _build_eval_loader(
    dataset: Dataset[Any],
    batch_size: int,
) -> DataLoader:
    """
    Build a pure evaluation DataLoader.

    - No shuffling
    - No multiprocessing
    - Deterministic

    Args:
        dataset: The dataset instance.
        batch_size: Batch size for inference.

    Returns:
        Configured evaluation DataLoader.

    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# -----------------------------------------------------------------------------#
# Public API                                                                   #
# -----------------------------------------------------------------------------#
def load_inference_context(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 1,
    ood_fraction: float = 1.0,
    prefer_cuda: bool = True,
) -> tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]:
    """
    Build the complete inference context for FNO/PINO evaluation.

    Typical usage:
        model, loader, processor, device = load_inference_context(
            dataset_path=dataset_pt_path,
            checkpoint_path=checkpoint_path,
            batch_size=1,
            ood_fraction=1.0,
        )

    Args:
        dataset_path: Path to the `.pt` dataset (ID or OOD dataset).
        checkpoint_path: Path to the saved `state_dict` (e.g. best_model_state_dict.pt).
                         A `config.json` must exist in the same directory.
        batch_size: Evaluation batch size. Default = 1.
        ood_fraction: Fraction of samples to keep (for OOD testing).
                      Default = 1.0 → use all samples.
        prefer_cuda: If True, move model & processor to GPU when available.

    Returns:
        model: Loaded and device-ready FNO model
        loader: Evaluation DataLoader
        processor: Reconstructed normalisation
        device: torch.device used

    """
    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)

    # 1) Load config
    cfg = _load_config(checkpoint_path.parent / "config.json")
    model_cfg = cfg["model"]

    # 2) Load dataset
    full_dataset = PermeabilityFlowDataset(str(dataset_path))

    # 3) Recompute normalisation
    data_processor = _build_normalizer_from_dataset(full_dataset)

    # 4) Rebuild model + load weights
    model = _build_model_from_config(model_cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 5) Device
    device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    data_processor = data_processor.to(device)

    # 6) Optional OOD subsampling
    if ood_fraction < 1.0:
        n_total = len(full_dataset)
        n_keep = max(1, int(ood_fraction * n_total))
        indices: list[int] = torch.randperm(n_total).tolist()[:n_keep]

        base_ds = cast("Dataset[dict[str, Tensor]]", full_dataset)
        dataset_eval: Dataset[dict[str, Tensor]] = Subset(base_ds, indices)
    else:
        dataset_eval = cast("Dataset[dict[str, Tensor]]", full_dataset)

    # 7) Evaluation DataLoader
    loader = _build_eval_loader(dataset_eval, batch_size=batch_size)

    return model, loader, data_processor, device
