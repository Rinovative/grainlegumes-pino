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

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

if TYPE_CHECKING:
    from collections.abc import Callable


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
    split_indices: dict[str, Tensor] | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor, dict[str, Any]]:
    """
    Create train, eval, and OOD dataloaders with deterministic splitting.

    Splitting is performed BEFORE normalizer fitting. Normalizers are fit
    on the train split only to avoid data leakage.

    Split indices are returned explicitly for the caller to persist.
    The training layer is responsible for saving split_indices.pt.

    Parameters
    ----------
    dataset_cls : type[BaseDataset]
        Dataset class to instantiate.
    path_train : str
        Path to the in-distribution training dataset `.pt` file.
    path_test_ood : str
        Path to the out-of-distribution dataset `.pt` file.
    batch_size : int, optional
        Batch size for all dataloaders (default: 16).
    train_ratio : float, optional
        Fraction of training samples used for training (default: 0.8).
    ood_fraction : float, optional
        Fraction of OOD samples used for evaluation (default: 0.2).
    num_workers : int, optional
        Number of parallel data loading workers (default: 4).
        If 0, workers are disabled (useful for strict debugging).
    pin_memory : bool, optional
        Use pinned memory for faster GPU transfer (default: True).
    persistent_workers : bool, optional
        Keep workers alive between epochs (default: True).
        Only enabled if num_workers > 0.
    split_seed : int, optional
        Random seed for split generation (default: 9).
    split_indices : dict[str, Tensor] | None, optional
        Pre-computed split indices to reuse. If provided, splits are not
        regenerated. Expected keys: "train_indices", "eval_indices",
        "ood_indices". If None, splits are generated deterministically.
    **kwargs : Any
        Additional keyword arguments passed to the dataset class.

    Returns
    -------
    tuple
        Tuple containing:
            - train_loader: DataLoader for the training subset.
            - test_loaders: Dict with "eval" and "ood" DataLoaders.
            - data_processor: Normalizer fitted on train split only.
            - split_info: Dict with split indices and metadata for persistence.

    Notes
    -----
    The split_info dict contains:
        - "train_indices": Tensor of training set indices
        - "eval_indices": Tensor of evaluation set indices
        - "ood_indices": Tensor of OOD set indices
        - "metadata": Dict with dataset name, sample counts, etc.

    The training layer should save this dict using torch.save(split_info, path).

    """
    # Validate num_workers and persistent_workers
    if num_workers == 0:
        persistent_workers = False
    if num_workers > 0 and persistent_workers is None:
        persistent_workers = True

    # Load full training dataset
    full_train = dataset_cls(path_train, **kwargs)
    n_train_full = len(full_train)
    n_train = int(train_ratio * n_train_full)
    n_eval = n_train_full - n_train

    # Generate or reuse split indices
    if split_indices is None:
        # Create splits deterministically
        train_set, eval_set = random_split(
            full_train,
            [n_train, n_eval],
            generator=torch.Generator().manual_seed(split_seed),
        )
        # Extract indices from Subset objects
        train_indices = torch.tensor(train_set.indices, dtype=torch.long)
        eval_indices = torch.tensor(eval_set.indices, dtype=torch.long)
    else:
        # Reuse provided indices
        train_indices = split_indices["train_indices"]
        eval_indices = split_indices["eval_indices"]
        train_set = Subset(full_train, train_indices.tolist())
        eval_set = Subset(full_train, eval_indices.tolist())

    # Load full OOD dataset
    ood_full = dataset_cls(path_test_ood, **kwargs)
    n_ood_full = len(ood_full)
    n_ood = int(ood_fraction * n_ood_full)

    if split_indices is None:
        # Create OOD split deterministically
        ood_subset, _ = random_split(
            ood_full,
            [n_ood, n_ood_full - n_ood],
            generator=torch.Generator().manual_seed(split_seed),
        )
        ood_indices = torch.tensor(ood_subset.indices, dtype=torch.long)
    else:
        # Reuse provided OOD indices
        ood_indices = split_indices["ood_indices"]
        ood_subset = Subset(ood_full, ood_indices.tolist())

    # ========================================================================
    # Phase 3: Fit normalizers on TRAIN split only (after split, before eval)
    # ========================================================================
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

    x_train = torch.cat(xs_train, dim=0)  # [N_train, C, H, W]
    y_train = torch.cat(ys_train, dim=0)

    # Normalizers fitted on train split only
    in_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
    in_norm.fit(x_train)

    out_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
    out_norm.fit(y_train)

    data_processor = DefaultDataProcessor(
        in_normalizer=in_norm,
        out_normalizer=out_norm,
    )

    # ========================================================================
    # Create dataloaders with explicit generator seeding
    # ========================================================================
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

    test_loaders = {
        "eval": eval_loader,
        "ood": ood_loader,
    }

    # Split info for persistence
    split_info = {
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "ood_indices": ood_indices,
        "metadata": {
            "train_dataset": path_train,
            "ood_dataset": path_test_ood,
            "n_train_full": n_train_full,
            "n_train": len(train_indices),
            "n_eval": len(eval_indices),
            "n_ood_full": n_ood_full,
            "n_ood": len(ood_indices),
            "train_ratio": train_ratio,
            "ood_fraction": ood_fraction,
            "split_seed": split_seed,
        },
    }

    return train_loader, test_loaders, data_processor, split_info
