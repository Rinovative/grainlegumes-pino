"""
Base dataset utilities for modular simulation datasets.

This module defines a generic dataset base class and a helper function
to create train, validation, and out-of-distribution (OOD) dataloaders
with consistent normalization based strictly on the full training set.
"""

from __future__ import annotations

from typing import Any

import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class BaseDataset(Dataset[dict[str, Tensor]]):
    """
    Generic dataset base class for all simulation datasets.

    This class handles loading a `.pt` file and provides the standard
    dataset interface. It should be subclassed to implement `__getitem__`.
    """

    def __init__(self, data_path: str) -> None:
        """
        Load the dataset from a serialized PyTorch file.

        Args:
            data_path: Path to a `.pt` file containing simulation data.

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
    **kwargs: Any,
) -> tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor]:
    """
    Create train, evaluation, and OOD dataloaders with global normalization.

    The normalization parameters (mean/std) are computed over *all*
    training samples to ensure stable scaling and reproducible FNO/PINO training.

    Args:
        dataset_cls: Dataset class to instantiate.
        path_train: Path to the in-distribution training dataset `.pt` file.
        path_test_ood: Path to the out-of-distribution dataset `.pt` file.
        batch_size: Batch size for all dataloaders.
        train_ratio: Fraction of training samples used for training.
        ood_fraction: Fraction of OOD samples used for evaluation.
        num_workers: Number of parallel data loading workers.
        pin_memory: Use pinned memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        **kwargs: Additional keyword arguments passed to the dataset class.

    Returns:
        Tuple containing:
            - train_loader: DataLoader for the training subset.
            - test_loaders: Dict with "eval" and "ood" DataLoaders.
            - data_processor: Normalization processor using train mean/std.

    """
    # ----------------------------------------------------------------------
    # Load full training data (for splitting and global normalization)
    # ----------------------------------------------------------------------
    full_train = dataset_cls(path_train, **kwargs)
    n_train = int(train_ratio * len(full_train))
    n_eval = len(full_train) - n_train

    train_set, eval_set = random_split(
        full_train,
        [n_train, n_eval],
        generator=torch.Generator().manual_seed(9),
    )

    # ----------------------------------------------------------------------
    # Build loaders
    # ----------------------------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # ----------------------------------------------------------------------
    # Out-of-distribution loader
    # ----------------------------------------------------------------------
    ood_full = dataset_cls(path_test_ood, **kwargs)
    n_ood = int(ood_fraction * len(ood_full))

    ood_subset, _ = random_split(
        ood_full,
        [n_ood, len(ood_full) - n_ood],
        generator=torch.Generator().manual_seed(9),
    )

    ood_loader = DataLoader(
        ood_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    xs: list[Tensor] = []
    ys: list[Tensor] = []

    full_train_loader = DataLoader(
        full_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    for batch in full_train_loader:
        xs.append(batch["x"])
        ys.append(batch["y"])

    x_all = torch.cat(xs, dim=0)  # [N, C, H, W]
    y_all = torch.cat(ys, dim=0)

    # Normalizer based on the entire training dataset
    in_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
    in_norm.fit(x_all)

    out_norm = UnitGaussianNormalizer(dim=[0, 2, 3])
    out_norm.fit(y_all)

    data_processor = DefaultDataProcessor(
        in_normalizer=in_norm,
        out_normalizer=out_norm,
    )

    test_loaders = {
        "eval": eval_loader,
        "ood": ood_loader,
    }

    return train_loader, test_loaders, data_processor
