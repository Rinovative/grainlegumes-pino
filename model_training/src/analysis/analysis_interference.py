"""
Inference utilities for PINO and FNO model evaluation.

This module reconstructs a complete, deterministic inference environment
that mirrors the training setup bit for bit. Instead of relying on
`neuralop.training.load_training_state`, all required components are
rebuilt explicitly and transparently. This ensures reproducibility and
consistent evaluation metrics across machines and training checkpoints.

The evaluation pipeline performs the following operations:

1. Load `config.json` from the run directory.
2. Rebuild the model architecture using stored hyperparameters.
3. Load the trained model weights from `best_model_state_dict.pt`.
4. Load and reconstruct the training normaliser (`normalizer.pt`)
   using the four stored tensors (flat NeuralOp format).
5. Load the simulation dataset for inference.
6. Create a deterministic evaluation DataLoader.

The main entry point:

    load_inference_context(...)

returns the tuple:

    (model, loader, processor, device)

which can be used directly for evaluation, visualisation, or downstream
postprocessing.
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
from torch.utils.data import DataLoader, Dataset

from src.dataset.dataset_simulation import PermeabilityFlowDataset

if TYPE_CHECKING:
    from torch import Tensor


# ======================================================================
# CONFIG LOADING
# ======================================================================
def _load_config(config_path: Path) -> dict[str, Any]:
    """
    Load a JSON configuration file generated during training.

    Args:
        config_path (Path): Path to the `config.json` file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# MODEL RECONSTRUCTION
# ======================================================================
def _build_model_from_config(model_cfg: dict[str, Any]) -> nn.Module:
    """
    Reconstruct an FNO model from stored hyperparameters.

    Args:
        model_cfg (dict): The `"model"` section of the configuration.

    Returns:
        nn.Module: Fully initialised FNO model.

    Raises:
        NotImplementedError: If the architecture type is unknown.

    """
    arch = model_cfg["architecture"]
    params = dict(model_cfg["model_params"])

    if arch != "FNO":
        msg = f"Unknown architecture: {arch}"
        raise NotImplementedError(msg)

    # Convert JSON single-element lists to scalars
    for key in ["channel_mlp_skip", "fno_skip"]:
        val = params.get(key)
        if isinstance(val, list) and len(val) == 1:
            params[key] = val[0]

    return FNO(**params)


# ======================================================================
# NORMALIZER LOADING
# ======================================================================
def _load_normalizer(normalizer_path: Path, *, device: torch.device) -> DefaultDataProcessor:
    """
    Load and reconstruct the NeuralOp normaliser used during training.

    The stored file contains four tensors:
        - in_normalizer.mean
        - in_normalizer.std
        - out_normalizer.mean
        - out_normalizer.std

    These tensors are assigned to a fresh `DefaultDataProcessor`, ensuring
    that the preprocessing pipeline matches the training setup exactly.

    Args:
        normalizer_path (Path): Path to `normalizer.pt`.
        device (torch.device): Target device for all tensors.

    Returns:
        DefaultDataProcessor: Fully reconstructed normalisation processor.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If expected keys are missing.

    """
    if not normalizer_path.exists():
        msg = f"Normalizer file not found: {normalizer_path}"
        raise FileNotFoundError(msg)

    state = torch.load(normalizer_path, map_location="cpu")

    required = {
        "in_normalizer.mean",
        "in_normalizer.std",
        "out_normalizer.mean",
        "out_normalizer.std",
    }

    if not required.issubset(state.keys()):
        msg = f"Invalid normaliser format. Missing keys. Found keys: {sorted(state.keys())}"
        raise RuntimeError(msg)

    processor = DefaultDataProcessor(
        in_normalizer=UnitGaussianNormalizer(dim=[0, 2, 3]),
        out_normalizer=UnitGaussianNormalizer(dim=[0, 2, 3]),
    )

    processor.in_normalizer.mean = state["in_normalizer.mean"].to(device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.in_normalizer.std = state["in_normalizer.std"].to(device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.out_normalizer.mean = state["out_normalizer.mean"].to(device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.out_normalizer.std = state["out_normalizer.std"].to(device)  # pyright: ignore[reportOptionalMemberAccess]

    processor.device = device
    return processor


# ======================================================================
# DATA LOADER
# ======================================================================
def _build_eval_loader(dataset: Dataset[Any], batch_size: int) -> DataLoader:
    """
    Build a deterministic evaluation DataLoader.

    Args:
        dataset (Dataset): Dataset containing simulation cases.
        batch_size (int): Evaluation batch size.

    Returns:
        DataLoader: Deterministic DataLoader with no shuffling.

    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# ======================================================================
# PUBLIC INFERENCE ENTRY POINT
# ======================================================================
def load_inference_context(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 1,
    prefer_cuda: bool = True,
) -> tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]:
    """
    Rebuild the complete inference context for a trained PINO or FNO model.

    This function reconstructs the model, normaliser, dataset, and DataLoader,
    ensuring that inference preprocessing matches the training phase exactly.

    Args:
        dataset_path (str | Path): Path to the evaluation dataset directory.
        checkpoint_path (str | Path): Path to `best_model_state_dict.pt`.
        batch_size (int, optional): Evaluation batch size. Default 1.
        prefer_cuda (bool, optional): Use CUDA if available. Default True.

    Returns:
        tuple:
            model (nn.Module): Loaded neural operator model.
            loader (DataLoader): Deterministic evaluation loader.
            processor (DefaultDataProcessor): Preprocessing pipeline.
            device (torch.device): Device used for inference.

    """
    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent

    device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")

    cfg = _load_config(run_dir / "config.json")
    model_cfg = cfg["model"]

    model = _build_model_from_config(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    processor = _load_normalizer(run_dir / "normalizer.pt", device=device)

    full_dataset = PermeabilityFlowDataset(str(dataset_path))
    dataset_eval = cast("Dataset[dict[str, Tensor]]", full_dataset)

    loader = _build_eval_loader(dataset_eval, batch_size=batch_size)

    return model, loader, processor, device
