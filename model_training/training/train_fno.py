"""
Model training script for Fourier Neural Operator (FNO).

This script sets up the FNO model, optimizer, scheduler, loss functions,
and launches the training process using the base training function.
"""

import torch
from neuralop import H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import FNO
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
    "run_suffix": None,
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
    "n_epochs": 600,
    "eval_interval": 5,  # evaluate every N epochs
    "mixed_precision": False,  # enables AMP on modern GPUs
    # --- Checkpointing & Resume ---
    # "resume_from_dir": "FNO_samples_uniform_var10_N1000_20251112_153012",
    # "resume_from_dir": "latest",
    # --- Logging ---
    "save_best": "eval_overall_rmse",  # metric key to monitor for best checkpoint
    "log_physical_rmse": True,  # log RMSE in physical units for each channel
    "save_every": None,  # optional periodic checkpoint saving
}


def finalize_config(CONFIG: dict) -> None:
    """Finalize and validate the configuration dictionary."""
    CONFIG.setdefault("modes_x", 128)
    CONFIG.setdefault("modes_y", 160)

    CONFIG.setdefault("hidden_channels", 64)
    CONFIG.setdefault("n_layers", 3)


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
# --- Model ---
def build_model(CONFIG: dict) -> FNO:
    """Build the FNO model based on the configuration."""
    CONFIG.setdefault("modes_x", 128)
    CONFIG.setdefault("modes_y", 160)

    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    CONFIG["n_modes_xy"] = (m_x, m_y)
    return FNO(
        n_modes=(m_x, m_y),
        hidden_channels=CONFIG["hidden_channels"],
        in_channels=7,
        out_channels=3,
        n_layers=CONFIG["n_layers"],
    ).to(CONFIG["device"])


# 🏷️ --- Model naming ---
def build_model_name(CONFIG: dict) -> str:
    """Build a descriptive model name based on the configuration."""
    m_x, m_y = CONFIG.get("n_modes_xy", (CONFIG["modes_x"], CONFIG["modes_y"]))
    hidden = CONFIG["hidden_channels"]
    layers = CONFIG["n_layers"]

    parts = [
        "FNO",
        f"m{m_x}x{m_y}",
        f"h{hidden}",
        f"l{layers}",
        CONFIG["train_dataset_name"],
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# --- Optimizer model ---
def build_optimizer(CONFIG: dict, model: FNO) -> AdamW:
    """Build the AdamW optimizer for the model."""
    return AdamW(
        model.parameters(),
        lr=CONFIG.get("lr", 0.006213783249197781),
        weight_decay=CONFIG.get("weight_decay", 1e-4),
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
def run_fno(CONFIG: dict, manage_wandb: bool = True) -> None:
    """Run the training process for the FNO model."""
    finalize_config(CONFIG)
    model = build_model(CONFIG)
    CONFIG["model_name"] = build_model_name(CONFIG)

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
        manage_wandb,
    )


if __name__ == "__main__":
    run_fno(CONFIG)
