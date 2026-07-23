"""
===============================================================================
dataset_base.py
===============================================================================
Provide base dataset loading, splitting and dataloader construction helpers.

Responsibilities:
  - Load serialized dataset dictionaries
  - Build deterministic train/eval/OOD splits
  - Construct data processors and dataloaders
  - Return split indices for persistence by callers

Design principles:
  - Split creation uses explicit seeds
  - Normalizers are fit from training data only
  - DataLoader settings are controlled by config values

Boundaries:
  - Split persistence belongs to experiments and training orchestration
  - Simulation-specific sample construction belongs to datasets.simulation
===============================================================================
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

if TYPE_CHECKING:
    from collections.abc import Callable


_SPLIT_INDEX_KEYS = ("train_indices", "eval_indices", "ood_indices")
_SPLIT_COUNT_METADATA_KEYS = {
    "train_indices": "n_train",
    "eval_indices": "n_eval",
    "ood_indices": "n_ood",
}
_REQUIRED_NORMALIZER_STATE_KEYS = (
    "in_normalizer.mean",
    "in_normalizer.std",
    "out_normalizer.mean",
    "out_normalizer.std",
)


class BaseDataset(Dataset[dict[str, Tensor]]):
    """
    Generic dataset base class for all simulation datasets.

    This class handles loading a `.pt` file and provides the standard
    dataset interface. It should be subclassed to implement `__getitem__`.
    """

    def __init__(self, data_path: str) -> None:
        """
        Load the dataset from a serialized PyTorch file.

        Parameters
        ----------
        data_path : str
            Path to a `.pt` file containing simulation data.

        """
        self.data = torch.load(data_path)

    def __len__(self) -> int:
        """Return the total number of samples (N)."""
        if "inputs" in self.data and isinstance(self.data["inputs"], torch.Tensor):
            return self.data["inputs"].shape[0]
        if "outputs" in self.data and isinstance(self.data["outputs"], torch.Tensor):
            return self.data["outputs"].shape[0]
        msg = "Dataset must contain 'inputs' or 'outputs' tensors."
        raise KeyError(msg)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """
        Return a single sample by index.

        Must be implemented in subclasses.
        """
        msg = "Implement in subclass."
        raise NotImplementedError(msg)


def _required_metadata_count(metadata: Mapping[str, Any], key: str) -> int:
    """Return a required positive integer count from split metadata."""
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"split_indices.pt metadata {key!r} must be an integer."
        raise TypeError(msg)
    if value <= 0:
        msg = f"split_indices.pt metadata {key!r} must be positive, got {value}."
        raise ValueError(msg)
    return value


def _normalized_fraction(value: Any, *, label: str, allow_one: bool) -> float:
    """Return a finite split fraction in the supported interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{label} must be numeric."
        raise TypeError(msg)
    fraction = float(value)
    upper_bound_valid = fraction <= 1.0 if allow_one else fraction < 1.0
    if not math.isfinite(fraction) or fraction <= 0.0 or not upper_bound_valid:
        interval = "(0, 1]" if allow_one else "(0, 1)"
        msg = f"{label} must be in {interval}, got {value!r}."
        raise ValueError(msg)
    return fraction


def _required_metadata_fraction(metadata: Mapping[str, Any], key: str, *, allow_one: bool) -> float:
    """Return a required split fraction from saved metadata."""
    return _normalized_fraction(
        metadata.get(key),
        label=f"split_indices.pt metadata {key!r}",
        allow_one=allow_one,
    )


def _normalized_seed(value: Any, *, label: str) -> int:
    """Return an integer split seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    return value


def _required_dataset_path(metadata: Mapping[str, Any], key: str) -> Path:
    """Return a required dataset path from split metadata."""
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        msg = f"split_indices.pt metadata {key!r} must be a non-empty string."
        raise TypeError(msg)
    return Path(value)


def _paths_match(left: str | Path, right: str | Path) -> bool:
    """Compare dataset paths without requiring them to exist."""
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def _validated_index_tensor(split_info: Mapping[str, Any], key: str) -> Tensor:
    """Return one non-empty, unique, one-dimensional integer index tensor."""
    value = split_info.get(key)
    if not isinstance(value, Tensor):
        msg = f"split_indices.pt key {key!r} must be a torch.Tensor."
        raise TypeError(msg)
    if value.ndim != 1:
        msg = f"split_indices.pt key {key!r} must be one-dimensional, got shape {tuple(value.shape)}."
        raise ValueError(msg)
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        msg = f"split_indices.pt key {key!r} must contain integer indices, got dtype {value.dtype}."
        raise TypeError(msg)
    if value.numel() == 0:
        msg = f"split_indices.pt key {key!r} must not be empty."
        raise ValueError(msg)
    if torch.unique(value).numel() != value.numel():
        msg = f"split_indices.pt key {key!r} must not contain duplicate indices."
        raise ValueError(msg)
    return value.to(dtype=torch.long, device="cpu").clone()


def _validate_index_bounds(indices: Tensor, *, key: str, full_count: int) -> None:
    """Reject negative or out-of-range saved indices."""
    min_index = int(indices.min().item())
    max_index = int(indices.max().item())
    if min_index < 0 or max_index >= full_count:
        msg = (
            f"split_indices.pt key {key!r} is out of bounds for full count {full_count}; "
            f"index range is {min_index}..{max_index}."
        )
        raise ValueError(msg)


def _validate_train_eval_partition(train_indices: Tensor, eval_indices: Tensor, *, n_train_full: int) -> None:
    """Require train/eval to be disjoint and cover the full training dataset."""
    overlap = train_indices[torch.isin(train_indices, eval_indices)]
    if overlap.numel():
        preview = overlap[:10].tolist()
        msg = f"Saved train/eval indices must be disjoint; overlapping indices include {preview}."
        raise ValueError(msg)

    combined = torch.cat((train_indices, eval_indices))
    expected = torch.arange(n_train_full, dtype=torch.long)
    if combined.numel() != n_train_full or not torch.equal(torch.sort(combined).values, expected):
        missing = expected[~torch.isin(expected, combined)]
        preview = missing[:10].tolist()
        msg = (
            "Saved train/eval indices must cover every source index exactly once in "
            f"0..{n_train_full - 1}; missing indices include {preview}."
        )
        raise ValueError(msg)


def validate_split_info(
    split_info: Mapping[str, Any],
    *,
    n_train_full: int | None = None,
    n_ood_full: int | None = None,
    path_train: str | Path | None = None,
    path_test_ood: str | Path | None = None,
    expected_train_ratio: float | None = None,
    expected_ood_fraction: float | None = None,
    expected_split_seed: int | None = None,
) -> dict[str, Tensor]:
    """
    Validate the complete saved split contract and return normalized indices.

    Train and eval membership must be duplicate-free, disjoint, and an exact
    partition of ``0..n_train_full-1``. OOD membership must be duplicate-free
    and within ``0..n_ood_full-1``. All saved per-split counts and generation
    settings are required. Optional expected settings bind a reused split to
    the effective run config.

    The current merged dataset format does not persist ordered case IDs or a
    producer fingerprint. Dataset path plus full sample count is therefore the
    strongest portable identity check available here; file mtimes and sizes are
    deliberately not used because they are unstable across copies.
    """
    if not isinstance(split_info, Mapping):
        msg = "split_indices.pt must contain a mapping."
        raise TypeError(msg)

    metadata = split_info.get("metadata")
    if not isinstance(metadata, Mapping):
        msg = "split_indices.pt must contain a metadata mapping."
        raise TypeError(msg)

    saved_train_path = _required_dataset_path(metadata, "train_dataset")
    saved_ood_path = _required_dataset_path(metadata, "ood_dataset")
    saved_n_train_full = _required_metadata_count(metadata, "n_train_full")
    saved_n_ood_full = _required_metadata_count(metadata, "n_ood_full")
    saved_train_ratio = _required_metadata_fraction(metadata, "train_ratio", allow_one=False)
    saved_ood_fraction = _required_metadata_fraction(metadata, "ood_fraction", allow_one=True)
    saved_split_seed = _normalized_seed(
        metadata.get("split_seed"),
        label="split_indices.pt metadata 'split_seed'",
    )

    if n_train_full is not None and saved_n_train_full != n_train_full:
        msg = (
            f"split_indices.pt metadata 'n_train_full'={saved_n_train_full} does not match "
            f"the training dataset size {n_train_full}."
        )
        raise ValueError(msg)
    if n_ood_full is not None and saved_n_ood_full != n_ood_full:
        msg = (
            f"split_indices.pt metadata 'n_ood_full'={saved_n_ood_full} does not match "
            f"the OOD dataset size {n_ood_full}."
        )
        raise ValueError(msg)
    if path_train is not None and not _paths_match(saved_train_path, path_train):
        msg = f"split_indices.pt train_dataset does not match the loaded dataset: {saved_train_path} != {path_train}"
        raise ValueError(msg)
    if path_test_ood is not None and not _paths_match(saved_ood_path, path_test_ood):
        msg = f"split_indices.pt ood_dataset does not match the loaded dataset: {saved_ood_path} != {path_test_ood}"
        raise ValueError(msg)
    if expected_train_ratio is not None:
        normalized_expected_train_ratio = _normalized_fraction(
            expected_train_ratio,
            label="Expected train_ratio",
            allow_one=False,
        )
        if saved_train_ratio != normalized_expected_train_ratio:
            msg = (
                f"split_indices.pt metadata 'train_ratio'={saved_train_ratio} does not match "
                f"the effective config value {normalized_expected_train_ratio}."
            )
            raise ValueError(msg)
    if expected_ood_fraction is not None:
        normalized_expected_ood_fraction = _normalized_fraction(
            expected_ood_fraction,
            label="Expected ood_fraction",
            allow_one=True,
        )
        if saved_ood_fraction != normalized_expected_ood_fraction:
            msg = (
                f"split_indices.pt metadata 'ood_fraction'={saved_ood_fraction} does not match "
                f"the effective config value {normalized_expected_ood_fraction}."
            )
            raise ValueError(msg)
    if expected_split_seed is not None:
        normalized_expected_split_seed = _normalized_seed(
            expected_split_seed,
            label="Expected split_seed",
        )
        if saved_split_seed != normalized_expected_split_seed:
            msg = (
                f"split_indices.pt metadata 'split_seed'={saved_split_seed} does not match "
                f"the effective config value {normalized_expected_split_seed}."
            )
            raise ValueError(msg)

    validated = {key: _validated_index_tensor(split_info, key) for key in _SPLIT_INDEX_KEYS}
    saved_counts: dict[str, int] = {}
    for index_key, count_key in _SPLIT_COUNT_METADATA_KEYS.items():
        expected_count = _required_metadata_count(metadata, count_key)
        saved_counts[count_key] = expected_count
        actual_count = int(validated[index_key].numel())
        if expected_count != actual_count:
            msg = (
                f"split_indices.pt metadata {count_key!r}={expected_count} does not match "
                f"{index_key!r} count {actual_count}."
            )
            raise ValueError(msg)

    expected_train_count = int(saved_train_ratio * saved_n_train_full)
    expected_eval_count = saved_n_train_full - expected_train_count
    expected_ood_count = int(saved_ood_fraction * saved_n_ood_full)
    derived_counts = {
        "n_train": expected_train_count,
        "n_eval": expected_eval_count,
        "n_ood": expected_ood_count,
    }
    for count_key, derived_count in derived_counts.items():
        if saved_counts[count_key] != derived_count:
            msg = (
                f"split_indices.pt metadata {count_key!r}={saved_counts[count_key]} is inconsistent "
                f"with the saved split fraction and full dataset count; expected {derived_count}."
            )
            raise ValueError(msg)

    _validate_index_bounds(validated["train_indices"], key="train_indices", full_count=saved_n_train_full)
    _validate_index_bounds(validated["eval_indices"], key="eval_indices", full_count=saved_n_train_full)
    _validate_index_bounds(validated["ood_indices"], key="ood_indices", full_count=saved_n_ood_full)
    _validate_train_eval_partition(
        validated["train_indices"],
        validated["eval_indices"],
        n_train_full=saved_n_train_full,
    )
    return validated


def data_processor_from_state(
    state: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> DefaultDataProcessor:
    """Reconstruct a data processor from the persisted normalizer tensors."""
    if not isinstance(state, Mapping):
        msg = "Saved normalizer state must be a mapping."
        raise TypeError(msg)

    tensors: dict[str, Tensor] = {}
    for key in _REQUIRED_NORMALIZER_STATE_KEYS:
        value = state.get(key)
        if not isinstance(value, Tensor):
            msg = f"Saved normalizer state {key!r} must be a torch.Tensor."
            raise TypeError(msg)
        tensors[key] = value

    if tensors["in_normalizer.mean"].shape != tensors["in_normalizer.std"].shape:
        msg = "Saved input normalizer mean/std shapes do not match."
        raise ValueError(msg)
    if tensors["out_normalizer.mean"].shape != tensors["out_normalizer.std"].shape:
        msg = "Saved output normalizer mean/std shapes do not match."
        raise ValueError(msg)

    target_device = torch.device(device)
    processor = DefaultDataProcessor(
        in_normalizer=UnitGaussianNormalizer(dim=[0, 2, 3]),
        out_normalizer=UnitGaussianNormalizer(dim=[0, 2, 3]),
    )
    processor.in_normalizer.mean = tensors["in_normalizer.mean"].detach().clone().to(target_device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.in_normalizer.std = tensors["in_normalizer.std"].detach().clone().to(target_device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.out_normalizer.mean = tensors["out_normalizer.mean"].detach().clone().to(target_device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.out_normalizer.std = tensors["out_normalizer.std"].detach().clone().to(target_device)  # pyright: ignore[reportOptionalMemberAccess]
    processor.device = target_device
    return processor


def _make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Create a worker_init_fn for deterministic DataLoader worker seeding.

    When num_workers > 0, PyTorch spawns worker processes. Each worker
    must have its RNG seeded independently but deterministically.

    Parameters
    ----------
    base_seed : int
        Base seed for the worker pool.

    Returns
    -------
    callable
        Function to pass as worker_init_fn to DataLoader.

    """

    def worker_init_fn(worker_id: int) -> None:
        """Seed the worker's random state."""
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        _ = np.random.default_rng(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def create_dataloaders(
    dataset_cls: type[BaseDataset],
    path_train: str,
    path_test_ood: str,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    ood_fraction: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    split_seed: int = 9,
    split_indices: Mapping[str, Any] | None = None,
    data_processor: DefaultDataProcessor | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor, dict[str, Any]]:
    """
    Create train, eval, and OOD dataloaders with deterministic splitting.

    Splitting is performed before normalizer fitting. Fresh normalizers are fit
    on the train split only. When a restored ``data_processor`` is supplied,
    fitting is skipped so resume uses the exact persisted transform state.

    Parameters
    ----------
    dataset_cls : type[BaseDataset]
        Dataset class to instantiate.
    path_train : str
        Path to the in-distribution training dataset ``.pt`` file.
    path_test_ood : str
        Path to the out-of-distribution dataset ``.pt`` file.
    batch_size : int, optional
        Batch size for all dataloaders (default: 16).
    train_ratio : float, optional
        Fraction of training samples used for training (default: 0.8).
    ood_fraction : float, optional
        Fraction of OOD samples used for evaluation (default: 0.2).
    num_workers : int, optional
        Number of parallel data loading workers (default: 4).
    pin_memory : bool, optional
        Use pinned memory for faster GPU transfer (default: True).
    persistent_workers : bool, optional
        Keep workers alive between epochs (default: True).
    split_seed : int, optional
        Random seed for split generation (default: 9).
    split_indices : Mapping[str, Any] | None, optional
        Complete saved split contract to reuse. It must include train/eval/OOD
        index tensors and the persisted metadata mapping. Reused membership is
        validated against the loaded datasets and is never regenerated.
    data_processor : DefaultDataProcessor | None, optional
        Restored processor to reuse without fitting. If omitted, normalizers
        are fit on the selected training samples.
    **kwargs : Any
        Additional keyword arguments passed to the dataset class.

    Returns
    -------
    tuple
        Training loader, eval/OOD loaders, processor, and validated split info.

    """
    if num_workers == 0:
        persistent_workers = False
    if num_workers > 0 and persistent_workers is None:
        persistent_workers = True

    full_train = dataset_cls(path_train, **kwargs)
    ood_full = dataset_cls(path_test_ood, **kwargs)
    n_train_full = len(full_train)
    n_ood_full = len(ood_full)

    if split_indices is None:
        n_train = int(train_ratio * n_train_full)
        n_eval = n_train_full - n_train
        train_random, eval_random = random_split(
            full_train,
            [n_train, n_eval],
            generator=torch.Generator().manual_seed(split_seed),
        )
        n_ood = int(ood_fraction * n_ood_full)
        ood_random, _ = random_split(
            ood_full,
            [n_ood, n_ood_full - n_ood],
            generator=torch.Generator().manual_seed(split_seed),
        )
        split_info: dict[str, Any] = {
            "train_indices": torch.tensor(train_random.indices, dtype=torch.long),
            "eval_indices": torch.tensor(eval_random.indices, dtype=torch.long),
            "ood_indices": torch.tensor(ood_random.indices, dtype=torch.long),
            "metadata": {
                "train_dataset": path_train,
                "ood_dataset": path_test_ood,
                "n_train_full": n_train_full,
                "n_train": n_train,
                "n_eval": n_eval,
                "n_ood_full": n_ood_full,
                "n_ood": n_ood,
                "train_ratio": train_ratio,
                "ood_fraction": ood_fraction,
                "split_seed": split_seed,
            },
        }
    else:
        split_info = dict(split_indices)

    validated_indices = validate_split_info(
        split_info,
        n_train_full=n_train_full,
        n_ood_full=n_ood_full,
        path_train=path_train,
        path_test_ood=path_test_ood,
        expected_train_ratio=train_ratio,
        expected_ood_fraction=ood_fraction,
        expected_split_seed=split_seed,
    )
    split_info.update(validated_indices)

    train_set = Subset(full_train, validated_indices["train_indices"].tolist())
    eval_set = Subset(full_train, validated_indices["eval_indices"].tolist())
    ood_subset = Subset(ood_full, validated_indices["ood_indices"].tolist())

    if data_processor is None:
        xs_train: list[Tensor] = []
        ys_train: list[Tensor] = []
        train_loader_for_norm = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        for batch in train_loader_for_norm:
            xs_train.append(batch["x"])
            ys_train.append(batch["y"])

        x_train = torch.cat(xs_train, dim=0)
        y_train = torch.cat(ys_train, dim=0)
        in_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
        in_norm.fit(x_train)
        out_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
        out_norm.fit(y_train)
        data_processor = DefaultDataProcessor(
            in_normalizer=in_norm,
            out_normalizer=out_norm,
        )

    generator = torch.Generator().manual_seed(split_seed)
    worker_init = _make_worker_init_fn(split_seed) if num_workers > 0 else None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    ood_loader = DataLoader(
        ood_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    return train_loader, {"eval": eval_loader, "ood": ood_loader}, data_processor, split_info
