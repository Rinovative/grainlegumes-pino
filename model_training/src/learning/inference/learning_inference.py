"""
===============================================================================
learning_inference.py
===============================================================================
Rebuild deterministic model inference contexts from saved run artifacts.

Responsibilities:
  - Resolve the saved task contract and reconstruct the semantic model kind
  - Load saved model weights
  - Load saved normalizer state into a data processor
  - Build deterministic split-aware evaluation datasets and dataloaders
  - Validate saved field contracts against model and dataset channels

Design principles:
  - Inference mirrors the saved training configuration
  - The exact normalizer state admitted by completed-run validation is reconstructed without refitting
  - Saved split indices are applied before evaluation loaders are built
  - Field order checks fail fast on incompatible artifacts

This module does NOT:
  - Train or optimize models; ``learning.training`` owns execution
  - Generate analysis artifacts; ``analysis.artifacts`` owns publication
  - Allocate or transition run directories; ``experiments.run`` owns lifecycle state

Notes:
  - This module assumes the current saved-run contract:
    run_dir/
      config.yaml
      normalizer.pt
      best_checkpoint.pt
      last_checkpoint.pt
      split_indices.pt
      summary.json
  - The inference pipeline:
    1. Load config.yaml to get the resolved task, model kind and parameters
    2. Reconstruct the model and load model_state_dict from best_checkpoint.pt
    3. Reconstruct a DefaultDataProcessor from the already validated normalizer state
    4. Load split_indices.pt and select an explicit train/eval/OOD role
    5. Apply saved split membership before building the DataLoader
    6. Return the model, DataLoader, processor, and device for inference
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import common, datasets, experiments, learning

if TYPE_CHECKING:
    from collections.abc import Sized

    from neuralop.data.transforms.data_processors import DefaultDataProcessor

# ======================================================================
# RUN CONTRACT
# ======================================================================

SplitRole = Literal["train", "eval", "ood"]
_SPLIT_INDEX_KEYS: dict[str, str] = {
    "train": "train_indices",
    "eval": "eval_indices",
    "ood": "ood_indices",
}


@dataclass(frozen=True)
class SplitSelection:
    """
    Carry one immutable saved-split selection into dataset reconstruction.

    Attributes
    ----------
    role : {"train", "eval", "ood"}
        Semantic saved membership being reconstructed.
    dataset_path : pathlib.Path
        Current dataset file resolved for the saved logical dataset identity.
    indices : torch.Tensor
        Ordered saved source indices for ``role``.

    """

    role: SplitRole
    dataset_path: Path
    indices: torch.Tensor


class IndexedSubset(Dataset[dict[str, Any]]):
    """
    Present an ordered saved-split view without losing source-case identity.

    Construction copies a unique, non-empty, in-bounds integer index vector to
    CPU. Each returned mapping preserves the underlying sample and adds its
    immutable ``source_index`` plus contiguous ``split_local_index``; callers can
    therefore distinguish dataset identity from evaluation order.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Source dataset already admitted against its saved fingerprint.
    source_indices : torch.Tensor
        Unique non-empty one-dimensional integer membership in desired order.

    Raises
    ------
    TypeError
        If indices are not integral or source samples are not mappings.
    ValueError
        If membership is empty, multidimensional, or contains duplicates.
    IndexError
        If any source index lies outside the dataset.

    """

    def __init__(self, dataset: Dataset[Any], source_indices: torch.Tensor) -> None:
        """
        Validate and copy membership before exposing the subset.

        The stored index vector is an owned CPU ``long`` clone, so later caller
        mutation cannot change saved-split membership or evaluation order.
        """
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
        dataset_size = len(cast("Sized", dataset))
        if min_index < 0 or max_index >= dataset_size:
            msg = f"source_indices are out of bounds for dataset size {dataset_size}; index range is {min_index}..{max_index}."
            raise IndexError(msg)

        self.dataset = dataset
        self.source_indices = normalized_indices

    def __len__(self) -> int:
        """Return the number of selected samples."""
        return int(self.source_indices.numel())

    def __getitem__(self, split_local_index: int) -> dict[str, Any]:
        """
        Return one source mapping with both evaluation and dataset identity.

        Type/bounds, mapping shape, and reserved-key ownership are checked before
        a copied sample receives ``split_local_index`` and ``source_index``.
        """
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


# ======================================================================
# CONFIG AND SPLIT LOADING
# ======================================================================


def _data_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the resolved data config section after validating its shape."""
    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        msg = "Run config must contain a mapping at data."
        raise TypeError(msg)
    return data_cfg


def _configured_dataset_ids(config: Mapping[str, Any]) -> dict[SplitRole, str]:
    """
    Resolve logical train, evaluation, and OOD dataset IDs from ``config.yaml``.

    Train and evaluation intentionally share the saved training dataset; OOD
    requires the sole current-contract logical ID. All names pass central path
    validation before they can participate in filesystem resolution.
    """
    data_cfg = _data_section(config)
    train_dataset = common.paths.validate_logical_name(
        data_cfg.get("train_dataset"),
        label="data.train_dataset",
    )
    ood_datasets = data_cfg.get("ood_datasets")
    if not isinstance(ood_datasets, list) or len(ood_datasets) != 1:
        msg = "Run config data.ood_datasets must contain exactly one logical dataset id."
        raise TypeError(msg)
    ood_dataset = common.paths.validate_logical_name(ood_datasets[0], label="data.ood_datasets[0]")
    return {"train": train_dataset, "eval": train_dataset, "ood": ood_dataset}


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


def _saved_dataset_id(split_indices: Mapping[str, Any], *, role: SplitRole) -> str:
    """
    Recover the logical dataset ID persisted for one saved split role.

    Train/eval use ``metadata.datasets.train`` and OOD uses its distinct identity.
    Missing or malformed nested identity fields fail before current-root lookup.
    """
    metadata = _split_metadata(split_indices)
    datasets_meta = metadata.get("datasets")
    if not isinstance(datasets_meta, Mapping):
        msg = "split_indices.pt metadata.datasets must be a mapping."
        raise TypeError(msg)
    identity_key = "ood" if role == "ood" else "train"
    saved_identity = datasets_meta.get(identity_key)
    if not isinstance(saved_identity, Mapping):
        msg = f"split_indices.pt metadata.datasets.{identity_key} must be a mapping."
        raise TypeError(msg)
    dataset_id = saved_identity.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        msg = f"split_indices.pt metadata.datasets.{identity_key}.dataset_id must be a non-empty string."
        raise TypeError(msg)
    return dataset_id


def _split_settings(config: Mapping[str, Any]) -> tuple[float, float, int]:
    """
    Recover the exact saved split ratios and stable split subseed.

    The subseed is re-derived from ``run.seed`` through the maintained seed plan;
    missing run/data mappings or split settings fail instead of defaulting.
    """
    data_cfg = _data_section(config)
    run_cfg = config.get("run")
    if not isinstance(run_cfg, Mapping):
        msg = "Run config must contain a mapping at run."
        raise TypeError(msg)
    required = ((data_cfg, "train_ratio"), (data_cfg, "ood_fraction"), (run_cfg, "seed"))
    if any(key not in section for section, key in required):
        msg = "Run config is missing train_ratio, ood_fraction, or run.seed split settings."
        raise KeyError(msg)
    split_seed = experiments.run.build_seed_plan(int(run_cfg["seed"]))["split"]
    return (
        cast("float", data_cfg["train_ratio"]),
        cast("float", data_cfg["ood_fraction"]),
        split_seed,
    )


def _select_split(
    *,
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    split: str,
    dataset_root: Path,
    dataset_path: Path | None,
) -> SplitSelection:
    """
    Bind one requested split role to saved membership and a current dataset path.

    Saved split ratios, subseed, and logical dataset identity must agree with
    ``config.yaml``. An explicit ``dataset_path`` changes location only; later
    fingerprint validation still prevents it from substituting different data.
    """
    role = _validate_split_role(split)
    train_ratio, ood_fraction, split_seed = _split_settings(config)
    validated_indices = datasets.base.validate_split_info(
        split_indices,
        expected_train_ratio=train_ratio,
        expected_ood_fraction=ood_fraction,
        expected_split_seed=split_seed,
    )
    configured_dataset_id = _configured_dataset_ids(config)[role]
    saved_dataset_id = _saved_dataset_id(split_indices, role=role)
    if saved_dataset_id != configured_dataset_id:
        msg = f"Saved split dataset id for {role!r} does not match config.yaml: {saved_dataset_id!r} != {configured_dataset_id!r}."
        raise RuntimeError(msg)
    selected_path = dataset_path or common.paths.resolve_dataset_path(
        configured_dataset_id,
        dataset_root=dataset_root,
    )
    return SplitSelection(
        role=role,
        dataset_path=selected_path,
        indices=validated_indices[_SPLIT_INDEX_KEYS[role]],
    )


def _validate_split_indices_for_dataset(
    *,
    selection: SplitSelection,
    dataset: Dataset[Any],
    split_indices: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """
    Bind saved membership to the loaded dataset fingerprint and ordered IDs.

    The dataset must expose a verified ``DatasetIdentity``. Validation replays
    the saved split contract against that identity and rejects any membership
    change before an ``IndexedSubset`` or loader is constructed.
    """
    dataset_identity = getattr(dataset, "identity", None)
    if not isinstance(dataset_identity, datasets.identity.DatasetIdentity):
        msg = "Inference dataset must expose a verified DatasetIdentity."
        raise TypeError(msg)
    train_ratio, ood_fraction, split_seed = _split_settings(config)
    validated = datasets.base.validate_split_info(
        split_indices,
        train_identity=dataset_identity if selection.role != "ood" else None,
        ood_identity=dataset_identity if selection.role == "ood" else None,
        expected_train_ratio=train_ratio,
        expected_ood_fraction=ood_fraction,
        expected_split_seed=split_seed,
    )
    if not torch.equal(validated[_SPLIT_INDEX_KEYS[selection.role]], selection.indices):
        msg = f"Validated {selection.role!r} membership changed during inference construction."
        raise RuntimeError(msg)


# ======================================================================
# MODEL RECONSTRUCTION
# ======================================================================
def _model_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Return the resolved model section after validating its required shape.

    ``model.kind`` and a mapping-valued ``model.params`` must already exist;
    reconstruction never supplies architecture defaults for a saved run.
    """
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        msg = "Run config must contain a mapping at model."
        raise TypeError(msg)
    if "kind" not in model_cfg:
        msg = "Run config missing model.kind."
        raise KeyError(msg)
    params = model_cfg.get("params")
    if not isinstance(params, Mapping):
        msg = "Run config must contain a mapping at model.params."
        raise TypeError(msg)
    return model_cfg


def _field_contract(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return exact fields from the validated registered task contract."""
    task = experiments.config.loader.validate_resolved_task_contract(config)
    return list(task.input_names), list(task.output_names)


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
    return learning.models.factory.build_model(config, device=device)


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
def load_inference_context_with_resolution(
    *,
    run_dir: str | Path,
    device_resolution: learning.device.DeviceResolution,
    dataset_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    split: SplitRole | str = "eval",
    batch_size: int = 1,
) -> tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]:
    """
    Rebuild a split-aware inference context on one service-resolved device.

    This lower-level service reconstructs the model from ``config.yaml``, loads
    weights from ``best_checkpoint.pt``, restores the saved ``normalizer.pt`` data processor
    state, loads `split_indices.pt`, and builds a deterministic loader for the
    requested saved split. It never refits preprocessing statistics during
    inference.

    Parameters
    ----------
    run_dir : str | Path
        Path to a saved run directory containing `config.yaml`,
        `normalizer.pt`, `best_checkpoint.pt`, and `split_indices.pt`.
    device_resolution : learning.device.DeviceResolution
        Immutable runtime decision resolved by the inference or artifact boundary.
    dataset_path : str | Path | None, optional
        Optional exact final-dataset file. Its fingerprint and ordered sample
        identity must match the saved split.
    dataset_root : str | Path | None, optional
        Current explicit dataset root used with the saved logical dataset id.
        Defaults to the current central dataset-root resolution.
    split : {"train", "eval", "ood"}, optional
        Saved split role to load. `eval` and `train` use the saved training
        dataset membership; `ood` uses saved OOD membership against the OOD
        final dataset recorded by training. Default is `eval`.
    batch_size : int, optional
        Evaluation batch size. Default is 1.

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
        If saved run, field, dataset, or split identities are incompatible.
    TypeError
        If the supplied resolution or saved artifact shapes have invalid types.
    ValueError
        If the requested split role is unknown or has invalid membership.
    FileNotFoundError
        If a required completed-run or dataset artifact is absent.

    """
    if not isinstance(device_resolution, learning.device.DeviceResolution):
        msg = f"Inference requires one DeviceResolution, got {device_resolution!r}."
        raise TypeError(msg)
    device = device_resolution.device
    run_dir = Path(run_dir)
    requested_dataset_path = Path(dataset_path).expanduser() if dataset_path is not None else None
    current_dataset_root = Path(dataset_root).expanduser() if dataset_root is not None else common.paths.get_dataset_root()
    completed_run = experiments.run.validate_completed_run(run_dir)

    cfg = completed_run["config"]
    split_indices = completed_run["split_indices"]
    input_channels, output_channels = _field_contract(cfg)
    seed_plan = experiments.run.configure_reproducibility(cfg, device=device)
    split_selection = _select_split(
        config=cfg,
        split_indices=split_indices,
        split=str(split),
        dataset_root=current_dataset_root,
        dataset_path=requested_dataset_path,
    )

    experiments.run.seed_process(seed_plan["model_init"], device=device)
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

    best_checkpoint = completed_run["best_checkpoint"]
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model = model.to(device)

    processor = datasets.base.data_processor_from_state(completed_run["normalizer_state"], device=device)

    task = experiments.config.loader.validate_resolved_task_contract(cfg)
    source_dataset = datasets.simulation.create_task_dataset(
        split_selection.dataset_path,
        task=task,
    )
    _validate_split_indices_for_dataset(
        selection=split_selection,
        dataset=source_dataset,
        split_indices=split_indices,
        config=cfg,
    )
    # ------------------------------
    # HARD GUARDS: field contract <-> dataset
    # ------------------------------
    ds_in = source_dataset.input_fields
    ds_out = source_dataset.output_fields

    if ds_in is not None and list(ds_in) != list(input_channels):
        msg = f"Dataset input field contract mismatch.\nExpected: {input_channels}\nGot: {list(ds_in)}"
        raise RuntimeError(msg)

    if ds_out is not None and list(ds_out) != list(output_channels):
        msg = f"Dataset output field contract mismatch.\nExpected: {output_channels}\nGot: {list(ds_out)}"
        raise RuntimeError(msg)

    selected_dataset = IndexedSubset(source_dataset, split_selection.indices)
    loader = _build_eval_loader(selected_dataset, batch_size=batch_size)

    return model, loader, processor, device


def load_inference_context(
    *,
    run_dir: str | Path,
    dataset_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    split: SplitRole | str = "eval",
    batch_size: int = 1,
    device_policy: str = "auto",
) -> tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]:
    """
    Resolve a device once and rebuild a complete saved-run inference context.

    Parameters
    ----------
    run_dir : str | Path
        Completed saved run directory.
    dataset_path : str | Path | None, optional
        Optional exact dataset file bound to saved identity.
    dataset_root : str | Path | None, optional
        Current independent dataset root.
    split : {"train", "eval", "ood"}, optional
        Saved split role.
    batch_size : int, optional
        Deterministic inference batch size.
    device_policy : {"auto", "cuda", "cpu"}, optional
        Runtime policy. Auto selects usable CUDA then CPU; CUDA is strict; CPU
        avoids CUDA queries.

    Returns
    -------
    tuple[nn.Module, DataLoader, DefaultDataProcessor, torch.device]
        Loaded model, deterministic loader, saved processor, and concrete device.

    """
    resolution = learning.device.resolve_device(
        device_policy,
        path="device_policy",
    )
    return load_inference_context_with_resolution(
        run_dir=run_dir,
        device_resolution=resolution,
        dataset_path=dataset_path,
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
    )
