"""
===============================================================================
 learning_models_uno.py
===============================================================================
U-shaped Neural Operator (UNO) model construction and checkpoint utilities.

Responsibilities:
  - Build U-NO models with configurable mode schedules and scalings
  - Provide UNOWithCheckpoint wrapper for deterministic inference checkpointing
  - Support both base UNO and physics-informed UNO variants

This module does NOT:
  - Handle training orchestration (see learning_training_loop)
  - Define loss functions (see learning_losses_*)
  - Manage data loading or normalization (see src.datasets)
  - Create experiment configurations (see src.experiments.config)
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import torch
from neuralop.models import UNO


class UNOWithCheckpoint(UNO):
    """
    U-NO model with added checkpoint saving functionality.

    Extends the neuraloperator UNO class to add a simple checkpoint save method
    for deterministic inference and model serialization.

    Methods
    -------
    save_checkpoint(save_dir, save_name)
        Save the model state dict and metadata to the specified directory.

    """

    def save_checkpoint(self, save_dir: str, save_name: str = "model") -> None:
        """
        Save the model checkpoint to the specified directory.

        Parameters
        ----------
        save_dir : str
            Directory path where the checkpoint will be saved.
        save_name : str, optional
            Base name for checkpoint files (default: "model").

        Notes
        -----
        Saves two files:
        - {save_name}_state_dict.pt: Model state dictionary
        - {save_name}_metadata.pkl: Model metadata (class name, architecture)

        """
        torch.save(self.state_dict(), Path(save_dir) / f"{save_name}_state_dict.pt")

        metadata = {"model_class": self.__class__.__name__, "architecture": "UNO"}
        torch.save(metadata, Path(save_dir) / f"{save_name}_metadata.pkl")
