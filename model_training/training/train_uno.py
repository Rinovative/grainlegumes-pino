"""
Model training script for U-shaped Neural Operator (U-NO).

This script sets up the U-NO model, optimizer, scheduler, loss functions,
and launches the training process using the base training function.
"""

from pathlib import Path

import torch
from neuralop import H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import UNO
from neuralop.training import AdamW
from src.learning.metrics import metrics as learning_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from training.tools.spectral_hook import SpectralEnergyHook
from training.train_base import train_base

# ================================================================
# ⚙️ 1) Base configuration
# ================================================================
CONFIG = {
    # --- General ---
    "run_suffix": "resumed_UNO_trial14_to_M",
    "seed": 9,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # --- Spectral diagnostics ---
    "enable_spectral_hooks": True,
    # --- Dataset ---
    "train_dataset_name": "lhs_var80_seed3001",
    "ood_dataset_name": "lhs_var120_seed4001",
    "train_ratio": 0.8,  # fraction of dataset used for training
    "ood_fraction": 0.2,  # fraction of OOD data for evaluation
    # --- Dataloader ---
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    # --- Training ---
    "n_epochs": 1_000,
    "eval_interval": 5,  # evaluate every N epochs
    "mixed_precision": False,  # enables AMP on modern GPUs
    # --- Checkpointing & Resume ---
    "resume_from_dir": "M_UNO_m64x64_h32_l7_s1-05-05-1-1-2-2_mr0p4951318201313778_trial14_20260116_022222",
    # "resume_from_dir": "latest",
    # --- Logging ---
    "save_best": "eval_overall_rmse",  # metric key to monitor for best checkpoint
    "log_physical_rmse": True,  # log RMSE in physical units for each channel
    "save_every": None,  # optional periodic checkpoint saving
}


def finalize_config(CONFIG: dict) -> None:
    """Finalize the configuration by setting safe default values for missing keys."""
    # --- Architecture defaults ---
    CONFIG.setdefault("n_layers", 7)
    CONFIG.setdefault("hidden_channels", 32)
    CONFIG.setdefault("modes_x", 64)
    CONFIG.setdefault("modes_y", 64)
    CONFIG.setdefault("mode_ratio", 0.4951318201313778)

    if "uno_scalings" not in CONFIG:
        if CONFIG["n_layers"] == 5:  # noqa: PLR2004
            CONFIG["uno_scalings"] = [
                [1.0, 1.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        elif CONFIG["n_layers"] == 7:  # noqa: PLR2004
            CONFIG["uno_scalings"] = [
                [1.0, 1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
class UNOWithCheckpoint(UNO):
    """U-NO model with added checkpoint saving functionality."""

    def save_checkpoint(self, save_dir: str, save_name: str = "model") -> None:
        """Save the model checkpoint to the specified directory."""
        torch.save(self.state_dict(), Path(save_dir) / f"{save_name}_state_dict.pt")

        metadata = {"model_class": self.__class__.__name__, "architecture": "UNO"}
        torch.save(metadata, Path(save_dir) / f"{save_name}_metadata.pkl")


def build_model(CONFIG: dict) -> UNOWithCheckpoint:
    """Build the U-NO model based on the configuration."""
    n_layers = CONFIG["n_layers"]
    hidden = CONFIG["hidden_channels"]
    uno_scalings = CONFIG["uno_scalings"]

    base_x = int(CONFIG["modes_x"])
    base_y = int(CONFIG["modes_y"])

    mode_ratio = float(CONFIG.get("mode_ratio", 0.5))

    mid_x = max(8, int(base_x * mode_ratio))
    mid_y = max(8, int(base_y * mode_ratio))

    if n_layers == 5:  # noqa: PLR2004
        uno_n_modes = [
            [base_x, base_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [base_x, base_y],
        ]
    elif n_layers == 7:  # noqa: PLR2004
        uno_n_modes = [
            [base_x, base_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [base_x, base_y],
        ]
    uno_out_channels = [hidden] * n_layers

    return UNOWithCheckpoint(
        in_channels=7,
        out_channels=3,
        hidden_channels=hidden,
        n_layers=n_layers,
        uno_out_channels=uno_out_channels,
        uno_n_modes=uno_n_modes,
        uno_scalings=uno_scalings,
        channel_mlp_skip="linear",
    ).to(CONFIG["device"])


# 🏷️ --- Model naming ---
def build_model_name(CONFIG: dict, model: UNO) -> str:
    """Build a descriptive model name based on the configuration and model."""
    scaling_tag = "-".join(str(int(s[0]) if float(s[0]).is_integer() else s[0]).replace(".", "") for s in model.uno_scalings)
    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    parts = [
        "UNO",
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"s{scaling_tag}",
        f"mr{CONFIG.get('mode_ratio', 0.5):.3g}".replace(".", "p"),
    ]
    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# --- Optimizer ---
def build_optimizer(CONFIG: dict, model: UNO) -> AdamW:
    """Build the AdamW optimizer for the model."""
    return AdamW(
        model.parameters(),
        lr=CONFIG.get("lr", 0.0005319714363424885),
        weight_decay=CONFIG.get("weight_decay", 0.0001),
    )


# --- Scheduler ---
def build_scheduler(optimizer: Optimizer) -> ReduceLROnPlateau:
    """Build the ReduceLROnPlateau scheduler for the optimizer."""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-5,
    )


# --- Losses ---
train_loss = H1Loss(d=2)
eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
    "overall_rmse": learning_metrics.RMSEOverall(),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
def run_uno(CONFIG: dict, manage_wandb: bool = True) -> None:
    """Run the U-NO training process."""
    finalize_config(CONFIG)
    model = build_model(CONFIG)
    CONFIG["model_name"] = build_model_name(CONFIG, model)

    spectral_hook = None
    if CONFIG.get("enable_spectral_hooks", False):
        spectral_hook = SpectralEnergyHook()
        for module in model.modules():
            if isinstance(module, SpectralConv):
                module.register_forward_hook(spectral_hook.hook)

    optimizer = build_optimizer(CONFIG, model)
    scheduler = build_scheduler(optimizer)

    train_base(
        CONFIG,
        model,
        optimizer,
        scheduler,
        train_loss,
        eval_losses,
        spectral_hook,
        manage_wandb=manage_wandb,
    )


if __name__ == "__main__":
    run_uno(CONFIG)
