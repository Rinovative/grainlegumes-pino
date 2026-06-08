"""
===============================================================================
learning_inference.py
===============================================================================
Rebuild deterministic model inference contexts from saved run artifacts.

Responsibilities:
  - Reconstruct saved model architecture and weights
  - Load saved normalizer state into a data processor
  - Build deterministic evaluation datasets and dataloaders
  - Validate domain field contracts against model and dataset channels

Design principles:
  - Inference mirrors the saved training configuration
  - Normalizers are reloaded instead of refit
  - Field order checks fail fast on incompatible artifacts

Boundaries:
  - Training and optimization belong to learning.training
  - Artifact generation belongs to analysis.artifacts
  - Run-directory creation belongs to experiments.cli

Notes:
  - This module assumes a specific directory structure for saved runs:
    run_dir/
  ├── config.json
  ├── normalizer.pt
  ├── best_model_state_dict.pt
  - The inference pipeline:
    1. Load config.json to get model architecture and hyperparameters
    2. Reconstruct the model and load weights from best_model_state_dict.pt
    3. Load normalizer state from normalizer.pt into a DefaultDataProcessor
    4. Build the evaluation dataset and DataLoader
    5. Return the model, DataLoader, processor, and device for inference
    6. Downstream code can then run inference with the loaded context, ensuring consistency with training.
===============================================================================

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

from src import datasets, domain

from ..models import learning_models_uno as uno

if TYPE_CHECKING:
    from torch import Tensor

# ======================================================================
# CHANNEL CONFIGURATION
# ======================================================================

INPUT_CHANNELS = domain.field_sets.DEFAULT_INPUTS_2D
OUTPUT_CHANNELS = domain.field_sets.DEFAULT_OUTPUTS_2D


# ======================================================================
# CONFIG LOADING
# ======================================================================
def _load_config(config_path: Path) -> dict[str, Any]:
    """
    Load a JSON configuration file generated during training.

    Parameters
    ----------
    config_path : Path
        Path to the `config.json` file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

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

    Parameters
    ----------
    model_cfg : dict[str, Any]
        The `"model"` section of the configuration.

    Returns
    -------
    nn.Module
        Fully initialised FNO model.

    Raises
    ------
    NotImplementedError
        If the architecture type is unknown.

    """
    arch = model_cfg["architecture"]
    params = dict(model_cfg["model_params"])
    # FIX: JSON -> int keys for horizontal_skips_map
    hsm = params.get("horizontal_skips_map")
    if isinstance(hsm, dict):
        params["horizontal_skips_map"] = {int(k): int(v) for k, v in hsm.items()}

    # Convert JSON single-element lists to scalars
    for key in ["channel_mlp_skip", "fno_skip"]:
        val = params.get(key)
        if isinstance(val, list) and len(val) == 1:
            params[key] = val[0]

    if arch == "FNO":
        return FNO(**params)

    if arch in {"UNO", "UNOWithCheckpoint"}:
        return uno.UNOWithCheckpoint(**params)

    msg = f"Unknown architecture: {arch}"
    raise NotImplementedError(msg)


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

    Parameters
    ----------
    normalizer_path : Path
        Path to `normalizer.pt`.
    device : torch.device
        Target device for all tensors.

    Returns
    -------
    DefaultDataProcessor
        Fully reconstructed normalisation processor.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If expected keys are missing.

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

    Parameters
    ----------
    dataset : Dataset
        Dataset containing simulation cases.
    batch_size : int
        Evaluation batch size.

    Returns
    -------
    DataLoader
        Deterministic DataLoader with no shuffling.

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

    Parameters
    ----------
    dataset_path : str | Path
        Path to the evaluation dataset directory.
    checkpoint_path : str | Path
        Path to `best_model_state_dict.pt`.
    batch_size : int, optional
        Evaluation batch size. Default is 1.
    prefer_cuda : bool, optional
        Use CUDA if available. Default is True.

    Returns
    -------
    tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]
        model : nn.Module
            Loaded neural operator model.
        loader : DataLoader
            Deterministic evaluation loader.
        processor : DefaultDataProcessor
            Preprocessing pipeline.
        device : torch.device
            Device used for inference.

    """
    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent

    device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")

    cfg = _load_config(run_dir / "config.json")
    model_cfg = cfg["model"]

    model = _build_model_from_config(model_cfg)
    # ------------------------------
    # HARD GUARDS: field contract <-> model
    # ------------------------------
    if getattr(model, "in_channels", None) is not None and model.in_channels != len(INPUT_CHANNELS):
        msg = f"in_channels mismatch: model.in_channels={model.in_channels} vs field contract={len(INPUT_CHANNELS)} ({INPUT_CHANNELS})"
        raise RuntimeError(msg)

    if getattr(model, "out_channels", None) is not None and model.out_channels != len(OUTPUT_CHANNELS):
        msg = f"out_channels mismatch: model.out_channels={model.out_channels} vs field contract={len(OUTPUT_CHANNELS)} ({OUTPUT_CHANNELS})"
        raise RuntimeError(msg)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    processor = _load_normalizer(run_dir / "normalizer.pt", device=device)

    full_dataset = datasets.simulation.PhysicsDataset(
        str(dataset_path),
        include_inputs=INPUT_CHANNELS,
        include_outputs=OUTPUT_CHANNELS,
    )
    # ------------------------------
    # HARD GUARDS: field contract <-> dataset
    # ------------------------------
    ds_in = getattr(full_dataset, "include_inputs", None) or getattr(full_dataset, "input_fields", None)
    ds_out = getattr(full_dataset, "include_outputs", None) or getattr(full_dataset, "output_fields", None)

    if ds_in is not None and list(ds_in) != list(INPUT_CHANNELS):
        msg = f"Dataset input field contract mismatch.\nExpected: {INPUT_CHANNELS}\nGot: {list(ds_in)}"
        raise RuntimeError(msg)

    if ds_out is not None and list(ds_out) != list(OUTPUT_CHANNELS):
        msg = f"Dataset output field contract mismatch.\nExpected: {OUTPUT_CHANNELS}\nGot: {list(ds_out)}"
        raise RuntimeError(msg)

    dataset_eval = cast("Dataset[dict[str, Tensor]]", full_dataset)
    loader = _build_eval_loader(dataset_eval, batch_size=batch_size)

    return model, loader, processor, device
