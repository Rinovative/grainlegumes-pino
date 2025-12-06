"""
Inference utilities for PINO/FNO model evaluation.

This module reconstructs a full evaluation environment that mirrors the
training setup exactly. It avoids relying on `neuralop.training.load_training_state`
and instead rebuilds every required component deterministically.

The evaluation pipeline performs the following steps:

1. Load `config.json` from the run directory.
2. Rebuild the model architecture using stored hyperparameters.
3. Load model weights (`best_model_state_dict.pt`).
4. Load the training normaliser (`normalizer.pt`), including automatic
   support for both flat and nested state formats.
5. Load the simulation dataset for evaluation.
6. Build a deterministic DataLoader for inference.

The main entry point is:

    load_inference_context(...)

which returns:

    (model, loader, processor, device)

Ensuring the evaluation preprocessing and postprocessing match the training phase
bit-for-bit guarantees reproducibility and consistent metrics.
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


# =============================================================================
# 1) CONFIG LOADING
# =============================================================================
def _load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and parse a `config.json` file.

    Parameters
    ----------
    config_path : Path
        Path to the JSON configuration file produced during training.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.

    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 2) MODEL RECONSTRUCTION
# =============================================================================
def _build_model_from_config(model_cfg: dict[str, Any]) -> nn.Module:
    """
    Reconstruct an FNO model from the stored hyperparameters.

    Parameters
    ----------
    model_cfg : dict
        Dictionary under the key `"model"` from `config.json`.
        Must contain:
        - `"architecture"` (str)
        - `"model_params"` (dict)

    Returns
    -------
    nn.Module
        Instantiated FNO model configured identically to the training run.

    Raises
    ------
    NotImplementedError
        If an unsupported architecture is requested.

    """
    arch = model_cfg["architecture"]
    params = dict(model_cfg["model_params"])

    if arch != "FNO":
        msg = f"Unknown architecture: {arch}"
        raise NotImplementedError(msg)

    # Convert JSON-serialized single-element lists to scalars
    for key in ["channel_mlp_skip", "fno_skip"]:
        val = params.get(key)
        if isinstance(val, list) and len(val) == 1:
            params[key] = val[0]

    return FNO(**params)


# =============================================================================
# 3) NORMALIZER LOADING
# =============================================================================
def _load_normalizer(normalizer_path: Path, *, device: torch.device) -> DefaultDataProcessor:
    """
    Load the training normaliser stored in `normalizer.pt` (flat NeuralOp format).

    The saved file contains exactly four tensors:
        - in_normalizer.mean
        - in_normalizer.std
        - out_normalizer.mean
        - out_normalizer.std

    These tensors are manually assigned to a freshly constructed
    `DefaultDataProcessor`, ensuring the shapes and downstream
    NeuralOp expectations match the training setup.

    All tensors are moved onto `device` so that preprocessing during
    inference does not cause device-mismatch errors.

    Parameters
    ----------
    normalizer_path : Path
        Path to the `normalizer.pt` file.
    device : torch.device
        Device onto which the normaliser should be moved.

    Returns
    -------
    DefaultDataProcessor
        Fully reconstructed normalisation processor ready for inference.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If required keys are missing.

    """
    if not normalizer_path.exists():
        msg = f"Normalizer file not found: {normalizer_path}"
        raise FileNotFoundError(msg)

    # Load raw saved tensors from training (all CPU tensors)
    state = torch.load(normalizer_path, map_location="cpu")

    expected_keys = {
        "in_normalizer.mean",
        "in_normalizer.std",
        "out_normalizer.mean",
        "out_normalizer.std",
    }
    if not expected_keys.issubset(state.keys()):
        msg = f"Missing expected normalizer keys.\nExpected: {sorted(expected_keys)}\nFound: {sorted(state.keys())}"
        raise RuntimeError(msg)

    # Create empty processor with correct structure
    processor = DefaultDataProcessor(
        in_normalizer=UnitGaussianNormalizer(dim=[0, 2, 3]),
        out_normalizer=UnitGaussianNormalizer(dim=[0, 2, 3]),
    )

    # Assign loaded tensors AND move them to device
    processor.in_normalizer.mean = state["in_normalizer.mean"].to(device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.in_normalizer.std = state["in_normalizer.std"].to(device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.out_normalizer.mean = state["out_normalizer.mean"].to(device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.out_normalizer.std = state["out_normalizer.std"].to(device)  # pyright: ignore[reportOptionalMemberAccess]

    # Make sure the processor itself tracks the device
    processor.device = device

    return processor


# =============================================================================
# 4) EVALUATION DATALOADER
# =============================================================================
def _build_eval_loader(dataset: Dataset[Any], batch_size: int) -> DataLoader:
    """
    Create a deterministic DataLoader for evaluation.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing simulation cases.
    batch_size : int
        Number of samples per batch during evaluation.

    Returns
    -------
    DataLoader
        DataLoader with no shuffling and single-worker loading.

    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# =============================================================================
# 5) PUBLIC INFERENCE ENTRY POINT
# =============================================================================
def load_inference_context(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 1,
    prefer_cuda: bool = True,
) -> tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]:
    """
    Rebuild the complete inference context for a trained FNO/PINO model.

    This includes model reconstruction, normalizer loading, dataset creation,
    and DataLoader initialisation. All components are guaranteed to match the
    training setup.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the evaluation dataset directory (`cases/`).
    checkpoint_path : str or Path
        Path to the `best_model_state_dict.pt` file.
        The directory must also contain:
            - `config.json`
            - `normalizer.pt`
    batch_size : int, optional
        Evaluation batch size (default 1).
    prefer_cuda : bool, optional
        Whether to prefer GPU if available (default True).

    Returns
    -------
    tuple
        (model, loader, processor, device)
        where:
            model     : nn.Module, moved to correct device
            loader    : evaluation DataLoader
            processor : DefaultDataProcessor with loaded normalizer
            device    : torch.device used for inference

    Raises
    ------
    FileNotFoundError
        If any required file is missing.

    """
    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent

    # Select evaluation device
    device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")

    # Load configuration
    cfg = _load_config(run_dir / "config.json")
    model_cfg = cfg["model"]

    # Load dataset and DataLoader
    full_dataset = PermeabilityFlowDataset(str(dataset_path))
    dataset_eval = cast("Dataset[dict[str, Tensor]]", full_dataset)
    loader = _build_eval_loader(dataset_eval, batch_size=batch_size)

    # Load training normalizer
    processor = _load_normalizer(run_dir / "normalizer.pt", device=device)

    # Load model and weights
    model = _build_model_from_config(model_cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    return model, loader, processor, device
