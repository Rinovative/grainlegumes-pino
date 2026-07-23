"""
===============================================================================
learning_inference.py
===============================================================================
Rebuild deterministic model inference contexts from saved run artifacts.

Responsibilities:
  - Reconstruct saved model architecture and weights
  - Load saved normalizer state into a data processor
  - Build deterministic split-aware evaluation datasets and dataloaders
  - Validate saved field contracts against model and dataset channels

Design principles:
  - Inference mirrors the saved training configuration
  - Normalizers are reloaded instead of refit
  - Saved split indices are applied before evaluation loaders are built
  - Field order checks fail fast on incompatible artifacts

Boundaries:
  - Training and optimization belong to learning.training
  - Artifact generation belongs to analysis.artifacts
  - Run-directory creation belongs to experiments.cli

Notes:
  - This module assumes the current saved-run contract:
    run_dir/
      config.yaml
      normalizer.pt
      best_checkpoint.pt
      split_indices.pt
      summary.json
  - The inference pipeline:
    1. Load config.yaml to get the resolved model architecture and parameters
    2. Reconstruct the model and load model_state_dict from best_checkpoint.pt
    3. Load normalizer state from normalizer.pt into a DefaultDataProcessor
    4. Load split_indices.pt and select an explicit train/eval/OOD role
    5. Apply saved split membership before building the DataLoader
    6. Return the model, DataLoader, processor, and device for inference
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import datasets, experiments, learning

if TYPE_CHECKING:
    from neuralop.data.transforms.data_processors import DefaultDataProcessor

# ======================================================================
# RUN CONTRACT
# ======================================================================

CONFIG_FILENAME = "config.yaml"
NORMALIZER_FILENAME = "normalizer.pt"
CHECKPOINT_FILENAME = "best_checkpoint.pt"
SPLIT_INDICES_FILENAME = "split_indices.pt"
SplitRole = Literal["train", "eval", "ood"]
_SPLIT_INDEX_KEYS: dict[str, str] = {
    "train": "train_indices",
    "eval": "eval_indices",
    "ood": "ood_indices",
}
_SPLIT_DATASET_METADATA_KEYS: dict[str, str] = {
    "train": "train_dataset",
    "eval": "train_dataset",
    "ood": "ood_dataset",
}
_SPLIT_COUNT_METADATA_KEYS: dict[str, str] = {
    "train": "n_train",
    "eval": "n_eval",
    "ood": "n_ood",
}
_SPLIT_FULL_COUNT_METADATA_KEYS: dict[str, str] = {
    "train": "n_train_full",
    "eval": "n_train_full",
    "ood": "n_ood_full",
}


@dataclass(frozen=True)
class RunArtifactPaths:
    """Paths for the current saved-run contract consumed by inference."""

    config: Path
    normalizer: Path
    checkpoint: Path
    split_indices: Path


@dataclass(frozen=True)
class SplitSelection:
    """Validated split role, dataset path and saved member indices."""

    role: SplitRole
    dataset_path: Path
    indices: torch.Tensor


class IndexedSubset(Dataset[dict[str, Any]]):
    """Select saved source indices while retaining explicit sample identity."""

    def __init__(self, dataset: Dataset[Any], source_indices: torch.Tensor) -> None:
        """Store a validated, ordered copy of the selected source indices."""
        if source_indices.ndim != 1:
            msg = f"source_indices must be one-dimensional, got shape {tuple(source_indices.shape)}."
            raise ValueError(msg)
        if source_indices.dtype == torch.bool or source_indices.is_floating_point() or source_indices.is_complex():
            msg = f"source_indices must contain integers, got dtype {source_indices.dtype}."
            raise TypeError(msg)
        if source_indices.numel() == 0:
            msg = "source_indices must not be empty."
            raise ValueError(msg)
        if torch.unique(source_indices).numel() != source_indices.numel():
            msg = "source_indices must not contain duplicates."
            raise ValueError(msg)

        normalized_indices = source_indices.to(dtype=torch.long, device="cpu").clone()
        min_index = int(normalized_indices.min().item())
        max_index = int(normalized_indices.max().item())
        if min_index < 0 or max_index >= len(dataset):
            msg = (
                f"source_indices are out of bounds for dataset size {len(dataset)}; "
                f"index range is {min_index}..{max_index}."
            )
            raise IndexError(msg)

        self.dataset = dataset
        self.source_indices = normalized_indices

    def __len__(self) -> int:
        """Return the number of selected samples."""
        return int(self.source_indices.numel())

    def __getitem__(self, split_local_index: int) -> dict[str, Any]:
        """Return a source sample with stable split-local and source indices."""
        if not isinstance(split_local_index, int) or isinstance(split_local_index, bool):
            msg = f"split_local_index must be an integer, got {type(split_local_index).__name__}."
            raise TypeError(msg)
        if split_local_index < 0 or split_local_index >= len(self):
            msg = f"split_local_index {split_local_index} is out of bounds for split size {len(self)}."
            raise IndexError(msg)

        source_index = int(self.source_indices[split_local_index].item())
        source_sample = self.dataset[source_index]
        if not isinstance(source_sample, Mapping):
            msg = f"IndexedSubset source samples must be mappings, got {type(source_sample).__name__}."
            raise TypeError(msg)
        reserved_keys = {"split_local_index", "source_index"}.intersection(source_sample)
        if reserved_keys:
            msg = f"Source sample contains reserved identity keys: {sorted(reserved_keys)}."
            raise KeyError(msg)

        sample = dict(source_sample)
        sample["split_local_index"] = split_local_index
        sample["source_index"] = source_index
        return sample


def _require_file(path: Path, *, label: str) -> Path:
    """Return an existing file path or raise a clear contract error."""
    if not path.is_file():
        msg = f"Required {label} file not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _run_artifact_paths(run_dir: Path) -> RunArtifactPaths:
    """Resolve and validate the saved-run files used by inference."""
    return RunArtifactPaths(
        config=_require_file(run_dir / CONFIG_FILENAME, label="run config"),
        normalizer=_require_file(run_dir / NORMALIZER_FILENAME, label="normalizer"),
        checkpoint=_require_file(run_dir / CHECKPOINT_FILENAME, label="checkpoint"),
        split_indices=_require_file(run_dir / SPLIT_INDICES_FILENAME, label="split indices"),
    )


# ======================================================================
# CONFIG AND SPLIT LOADING
# ======================================================================


def _normalize_path(path: Path) -> Path:
    """Return a comparable path without requiring the file to exist."""
    return path.expanduser().resolve(strict=False)


def _paths_match(left: Path, right: Path) -> bool:
    """Return whether two saved dataset paths identify the same location."""
    return _normalize_path(left) == _normalize_path(right)


def _load_split_indices(split_indices_path: Path) -> Mapping[str, Any]:
    """Load saved split indices from the current run contract."""
    split_indices = torch.load(split_indices_path, map_location="cpu")
    if not isinstance(split_indices, Mapping):
        msg = f"Split indices must be a mapping: {split_indices_path}"
        raise TypeError(msg)
    return split_indices


def _data_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the resolved data config section after validating its shape."""
    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        msg = "Run config must contain a mapping at data."
        raise TypeError(msg)
    return data_cfg


def _paths_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the resolved paths config section after validating its shape."""
    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, Mapping):
        msg = "Run config must contain a mapping at paths."
        raise TypeError(msg)
    return paths_cfg


def _configured_dataset_paths(config: Mapping[str, Any]) -> dict[SplitRole, Path]:
    """Resolve the merged train/eval/OOD dataset paths from saved config.yaml."""
    data_cfg = _data_section(config)
    paths_cfg = _paths_section(config)

    train_root_raw = paths_cfg.get("train_root")
    if not isinstance(train_root_raw, str) or not train_root_raw:
        msg = "Run config must contain paths.train_root as a non-empty string."
        raise TypeError(msg)
    train_root = Path(train_root_raw)

    train_dataset = data_cfg.get("train_dataset")
    if not isinstance(train_dataset, str) or not train_dataset:
        msg = "Run config must contain data.train_dataset as a non-empty string."
        raise TypeError(msg)

    ood_datasets = data_cfg.get("ood_datasets")
    if not isinstance(ood_datasets, list) or not ood_datasets:
        msg = "Run config must contain data.ood_datasets as a non-empty list."
        raise TypeError(msg)
    if not all(isinstance(name, str) and name for name in ood_datasets):
        msg = "Run config data.ood_datasets must contain non-empty strings."
        raise TypeError(msg)

    train_path = train_root / train_dataset / f"{train_dataset}.pt"
    ood_dataset = ood_datasets[0]
    ood_path = train_root / ood_dataset / f"{ood_dataset}.pt"

    return {
        "train": train_path,
        "eval": train_path,
        "ood": ood_path,
    }


def _split_metadata(split_indices: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return saved split metadata required for dataset identity checks."""
    metadata = split_indices.get("metadata")
    if not isinstance(metadata, Mapping):
        msg = "split_indices.pt must contain a metadata mapping with dataset identity."
        raise TypeError(msg)
    return metadata


def _validate_split_role(split: str) -> SplitRole:
    """Validate the requested split role."""
    if split not in _SPLIT_INDEX_KEYS:
        allowed = ", ".join(sorted(_SPLIT_INDEX_KEYS))
        msg = f"Unknown inference split {split!r}. Expected one of: {allowed}."
        raise ValueError(msg)
    return cast("SplitRole", split)


def _metadata_dataset_path(metadata: Mapping[str, Any], *, role: SplitRole) -> Path:
    """Return the dataset path saved for a split role in split_indices metadata."""
    metadata_key = _SPLIT_DATASET_METADATA_KEYS[role]
    value = metadata.get(metadata_key)
    if not isinstance(value, str) or not value:
        msg = f"split_indices.pt metadata missing non-empty {metadata_key!r} for split {role!r}."
        raise RuntimeError(msg)
    return Path(value)


def _select_split(
    *,
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    split: str,
    dataset_path: Path | None,
) -> SplitSelection:
    """Select a saved split and validate its dataset identity."""
    role = _validate_split_role(split)
    data_cfg = _data_section(config)
    run_cfg = config.get("run")
    if not isinstance(run_cfg, Mapping):
        msg = "Run config must contain a mapping at run."
        raise TypeError(msg)
    required_settings = {
        "data.train_ratio": (data_cfg, "train_ratio"),
        "data.ood_fraction": (data_cfg, "ood_fraction"),
        "run.seed": (run_cfg, "seed"),
    }
    missing_settings = [label for label, (section, key) in required_settings.items() if key not in section]
    if missing_settings:
        msg = f"Run config is missing saved split setting(s): {', '.join(missing_settings)}."
        raise KeyError(msg)

    validated_indices = datasets.base.validate_split_info(
        split_indices,
        expected_train_ratio=cast("float", data_cfg["train_ratio"]),
        expected_ood_fraction=cast("float", data_cfg["ood_fraction"]),
        expected_split_seed=cast("int", run_cfg["seed"]),
    )
    metadata = _split_metadata(split_indices)
    configured_paths = _configured_dataset_paths(config)

    metadata_path = _metadata_dataset_path(metadata, role=role)
    configured_path = configured_paths[role]
    if not _paths_match(metadata_path, configured_path):
        msg = f"Saved split metadata for split {role!r} does not match config.yaml.\nmetadata path: {metadata_path}\nconfig path:   {configured_path}"
        raise RuntimeError(msg)

    selected_path = dataset_path if dataset_path is not None else metadata_path
    if not _paths_match(selected_path, metadata_path):
        msg = (
            f"Requested dataset_path is incompatible with split {role!r}.\n"
            f"requested: {selected_path}\n"
            f"expected:  {metadata_path}\n"
            "Saved split indices are only valid for the merged dataset recorded in split_indices.pt."
        )
        raise RuntimeError(msg)

    return SplitSelection(
        role=role,
        dataset_path=selected_path,
        indices=validated_indices[_SPLIT_INDEX_KEYS[role]],
    )


def _validate_split_indices_for_dataset(
    *,
    selection: SplitSelection,
    dataset: Dataset[Any],
    split_metadata: Mapping[str, Any],
) -> None:
    """Validate saved split indices against the selected dataset length."""
    sized_dataset = cast("Sized", dataset)
    dataset_size = len(sized_dataset)
    full_count_key = _SPLIT_FULL_COUNT_METADATA_KEYS[selection.role]
    expected_full_count = split_metadata.get(full_count_key)
    if not isinstance(expected_full_count, int) or isinstance(expected_full_count, bool):
        msg = f"split_indices.pt metadata {full_count_key!r} must be an integer."
        raise TypeError(msg)
    if expected_full_count != dataset_size:
        msg = (
            f"split_indices.pt metadata {full_count_key!r}={expected_full_count!r} does not match "
            f"the selected {selection.role!r} dataset size {dataset_size}."
        )
        raise RuntimeError(msg)

    max_index = int(selection.indices.max().item())
    min_index = int(selection.indices.min().item())
    if min_index < 0 or max_index >= dataset_size:
        msg = (
            f"Saved indices for split {selection.role!r} are out of bounds for {selection.dataset_path}.\n"
            f"dataset size: {dataset_size}\n"
            f"index range:  {min_index}..{max_index}"
        )
        raise RuntimeError(msg)

    count_key = _SPLIT_COUNT_METADATA_KEYS[selection.role]
    expected_count = split_metadata.get(count_key)
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        msg = f"split_indices.pt metadata {count_key!r} must be an integer."
        raise TypeError(msg)
    if expected_count != int(selection.indices.numel()):
        msg = (
            f"split_indices.pt metadata {count_key!r}={expected_count!r} does not match "
            f"the {selection.role!r} index count {selection.indices.numel()}."
        )
        raise RuntimeError(msg)


def _load_config(config_path: Path) -> dict[str, Any]:
    """
    Load the resolved YAML configuration generated during training.

    Parameters
    ----------
    config_path : Path
        Path to the `config.yaml` file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    """
    return experiments.config.loader.load_yaml(config_path)


# ======================================================================
# MODEL RECONSTRUCTION
# ======================================================================
def _model_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the resolved model config section after validating its shape."""
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        msg = "Run config must contain a mapping at model."
        raise TypeError(msg)
    if "architecture" not in model_cfg:
        msg = "Run config missing model.architecture."
        raise KeyError(msg)
    params = model_cfg.get("params")
    if not isinstance(params, Mapping):
        msg = "Run config must contain a mapping at model.params."
        raise TypeError(msg)
    return model_cfg


def _field_contract(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return input and output field names from the resolved run config."""
    input_fields = config.get("input_fields")
    output_fields = config.get("output_fields")
    if not isinstance(input_fields, list) or not all(isinstance(name, str) for name in input_fields):
        msg = "Run config must contain input_fields as a list of strings."
        raise TypeError(msg)
    if not isinstance(output_fields, list) or not all(isinstance(name, str) for name in output_fields):
        msg = "Run config must contain output_fields as a list of strings."
        raise TypeError(msg)
    return input_fields, output_fields


def _with_inference_device(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Return a shallow config copy with run.device set for this inference process."""
    config_for_device = dict(config)
    run_cfg = dict(config.get("run", {}))
    run_cfg["device"] = str(device)
    config_for_device["run"] = run_cfg
    return config_for_device


def _build_model_from_config(config: dict[str, Any], *, device: torch.device) -> nn.Module:
    """
    Reconstruct a neural-operator model from the resolved run config.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved run configuration loaded from `config.yaml`.
    device : torch.device
        Device to place the model on.

    Returns
    -------
    nn.Module
        Fully initialized model.

    Raises
    ------
    ValueError
        If the architecture type is unknown.

    """
    _model_section(config)
    return learning.models.factory.build_model(_with_inference_device(config, device))


def _load_model_state_dict(checkpoint_path: Path) -> Mapping[str, Any]:
    """
    Load model weights from the current full checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to `best_checkpoint.pt`.

    Returns
    -------
    Mapping[str, Any]
        Model state dictionary stored under `model_state_dict`.

    Raises
    ------
    TypeError
        If the checkpoint or model state dictionary has an invalid type.
    RuntimeError
        If the checkpoint does not contain `model_state_dict`.

    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        msg = f"Checkpoint must be a mapping: {checkpoint_path}"
        raise TypeError(msg)

    if "model_state_dict" not in checkpoint:
        msg = f"Checkpoint must contain model_state_dict: {checkpoint_path}"
        raise RuntimeError(msg)

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        msg = f"Checkpoint model_state_dict must be a mapping: {checkpoint_path}"
        raise TypeError(msg)

    return state_dict


# ======================================================================
# NORMALIZER LOADING
# ======================================================================
def _load_normalizer(normalizer_path: Path, *, device: torch.device) -> DefaultDataProcessor:
    """
    Load and reconstruct the NeuralOp normalizer used during training.

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
        Fully reconstructed normalization processor.

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
    return datasets.base.data_processor_from_state(state, device=device)


# ======================================================================
# DATA LOADER
# ======================================================================
def _build_eval_loader(dataset: Dataset[Any], batch_size: int) -> DataLoader:
    """
    Build a deterministic evaluation DataLoader for a selected saved split.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the selected saved split membership.
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
    run_dir: str | Path,
    dataset_path: str | Path | None = None,
    split: SplitRole | str = "eval",
    batch_size: int = 1,
    prefer_cuda: bool = True,
) -> tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]:
    """
    Rebuild the complete split-aware inference context for a saved run.

    This function reconstructs the model from `config.yaml`, loads weights from
    `best_checkpoint.pt`, restores the saved `normalizer.pt` data processor
    state, loads `split_indices.pt`, and builds a deterministic loader for the
    requested saved split. It never refits preprocessing statistics during
    inference.

    Parameters
    ----------
    run_dir : str | Path
        Path to a saved run directory containing `config.yaml`,
        `normalizer.pt`, `best_checkpoint.pt`, and `split_indices.pt`.
    dataset_path : str | Path | None, optional
        Optional path to the merged dataset file to load. If omitted, the path
        recorded in `split_indices.pt` metadata is used. If provided, it must
        match the saved dataset identity for the requested split.
    split : {"train", "eval", "ood"}, optional
        Saved split role to load. `eval` and `train` use the saved training
        dataset membership; `ood` uses saved OOD membership against the OOD
        merged dataset recorded by training. Default is `eval`.
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
            Deterministic evaluation loader over the selected saved split.
        processor : DefaultDataProcessor
            Preprocessing pipeline loaded from `normalizer.pt`.
        device : torch.device
            Device used for inference.

    Raises
    ------
    RuntimeError
        If saved split metadata is missing, incompatible, or out of bounds.
    ValueError
        If the requested split role is unknown or has empty indices.

    """
    run_dir = Path(run_dir)
    requested_dataset_path = Path(dataset_path) if dataset_path is not None else None
    artifacts = _run_artifact_paths(run_dir)

    device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")

    cfg = _load_config(artifacts.config)
    input_channels, output_channels = _field_contract(cfg)
    split_indices = _load_split_indices(artifacts.split_indices)
    split_selection = _select_split(
        config=cfg,
        split_indices=split_indices,
        split=str(split),
        dataset_path=requested_dataset_path,
    )
    split_metadata = _split_metadata(split_indices)

    model = _build_model_from_config(cfg, device=device)
    # ------------------------------
    # HARD GUARDS: field contract <-> model
    # ------------------------------
    if getattr(model, "in_channels", None) is not None and model.in_channels != len(input_channels):
        msg = f"in_channels mismatch: model.in_channels={model.in_channels} vs field contract={len(input_channels)} ({input_channels})"
        raise RuntimeError(msg)

    if getattr(model, "out_channels", None) is not None and model.out_channels != len(output_channels):
        msg = f"out_channels mismatch: model.out_channels={model.out_channels} vs field contract={len(output_channels)} ({output_channels})"
        raise RuntimeError(msg)

    model.load_state_dict(_load_model_state_dict(artifacts.checkpoint))
    model = model.to(device)

    processor = _load_normalizer(artifacts.normalizer, device=device)

    source_dataset = datasets.simulation.PhysicsDataset(
        str(split_selection.dataset_path),
        include_inputs=input_channels,
        include_outputs=output_channels,
    )
    _validate_split_indices_for_dataset(
        selection=split_selection,
        dataset=source_dataset,
        split_metadata=split_metadata,
    )
    # ------------------------------
    # HARD GUARDS: field contract <-> dataset
    # ------------------------------
    ds_in = getattr(source_dataset, "include_inputs", None) or getattr(source_dataset, "input_fields", None)
    ds_out = getattr(source_dataset, "include_outputs", None) or getattr(source_dataset, "output_fields", None)

    if ds_in is not None and list(ds_in) != list(input_channels):
        msg = f"Dataset input field contract mismatch.\nExpected: {input_channels}\nGot: {list(ds_in)}"
        raise RuntimeError(msg)

    if ds_out is not None and list(ds_out) != list(output_channels):
        msg = f"Dataset output field contract mismatch.\nExpected: {output_channels}\nGot: {list(ds_out)}"
        raise RuntimeError(msg)

    selected_dataset = IndexedSubset(source_dataset, split_selection.indices)
    loader = _build_eval_loader(selected_dataset, batch_size=batch_size)

    return model, loader, processor, device
