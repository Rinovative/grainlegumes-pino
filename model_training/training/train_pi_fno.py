"""
Model training script for Physics-Informed Fourier Neural Operator (PI-FNO).

This script sets up the FNO model, optimizer, scheduler, physics-informed loss functions,
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

from training.tools.pino_brinkman_losses import (
    PINOPhysicalLossDiv,
    PINOPhysicalLossEps,
    PINOSpectralLossDiv,
    PINOSpectralLossEps,
)
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
    # --- Physics-informed training ---
    # Optionen:
    #   "ps_eps"  -> Physical derivatives + div(eps u)
    #   "ps_u"    -> Physical derivatives + div(u)
    #   "sp_eps"  -> Spectral derivatives + div(eps u)
    #   "sp_u"    -> Spectral derivatives + div(u)
    "pino_loss_type": "sp_eps",
    "grad_mode": "fft_reflect",  # only for spectral losses
    "interior_pad": 2,  # useful for sp_*, can be 0 for ps_*
    "lambda_phys": 1e-4,
    "lambda_p": 5e-4,
    "phys_warmup_epochs": 1,
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
    # -------------------------------
    # Defaults for PINO-loss selection
    # -------------------------------
    CONFIG.setdefault("pino_loss_type", "sp_eps")  # ps_eps | ps_u | sp_eps | sp_u
    CONFIG.setdefault("grad_mode", "fft_reflect")  # only for spectral losses
    CONFIG.setdefault("interior_pad", 2)  # useful for sp_*, can be 0 for ps_*

    # -------------------------------
    # Architecture
    # -------------------------------
    CONFIG.setdefault("modes_x", 48)
    CONFIG.setdefault("modes_y", 48)
    CONFIG.setdefault("hidden_channels", 64)
    CONFIG.setdefault("n_layers", 4)

    # -------------------------------
    # Physics-informed weights (nur Existenz erzwingen)
    # -------------------------------
    _ = CONFIG["lambda_phys"]
    _ = CONFIG["lambda_p"]
    _ = CONFIG["phys_warmup_epochs"]


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
# --- Model ---
def build_model(CONFIG: dict) -> FNO:
    """Build the FNO model based on the configuration."""
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
    """Build a descriptive PI-FNO model name based on the configuration."""
    m_x, m_y = CONFIG.get("n_modes_xy", (CONFIG["modes_x"], CONFIG["modes_y"]))

    loss_type = str(CONFIG.get("pino_loss_type", "sp_eps"))
    tag_map = {
        "ps_eps": "PI-FNO-PS-EPS",
        "ps_u": "PI-FNO-PS-DIV",
        "sp_eps": "PI-FNO-SP-EPS",
        "sp_u": "PI-FNO-SP-DIV",
    }
    if loss_type not in tag_map:
        msg = f"Unknown pino_loss_type: {loss_type}"
        raise ValueError(msg)

    parts = [
        tag_map[loss_type],
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
    ]

    # optional: grad mode nur bei spectral
    if loss_type.startswith("sp"):
        parts.append(f"grad{CONFIG.get('grad_mode', 'fft_reflect')}")

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# --- Optimizer ---
def build_optimizer(CONFIG: dict, model: FNO) -> AdamW:
    """Build the AdamW optimizer for the model."""
    return AdamW(
        model.parameters(),
        lr=CONFIG.get("lr", 0.01),
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
def build_train_loss(CONFIG: dict) -> PINOPhysicalLossDiv | PINOPhysicalLossEps | PINOSpectralLossDiv | PINOSpectralLossEps:
    """Build the physics-informed loss function for training."""
    loss_type = str(CONFIG.get("pino_loss_type", "sp_eps"))

    # optional: fuer spectral backend
    grad_mode = CONFIG.get("grad_mode", "fft_reflect")
    interior_pad = int(CONFIG.get("interior_pad", 2 if loss_type.startswith("sp") else 0))

    cls_map = {
        "ps_eps": PINOPhysicalLossEps,
        "ps_u": PINOPhysicalLossDiv,
        "sp_eps": PINOSpectralLossEps,
        "sp_u": PINOSpectralLossDiv,
    }
    if loss_type not in cls_map:
        msg = f"Unknown pino_loss_type: {loss_type}"
        raise ValueError(msg)

    LossCls = cls_map[loss_type]

    kwargs = {
        "data_loss": H1Loss(d=2),
        "lambda_phys": CONFIG["lambda_phys"],
        "lambda_p": CONFIG["lambda_p"],
        "in_normalizer": None,
        "out_normalizer": None,
        "interior_pad": interior_pad,
    }

    # spectral-only kwargs
    if loss_type.startswith("sp"):
        kwargs["grad_mode"] = grad_mode

    return LossCls(**kwargs)


eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
    "overall_rmse": learning_metrics.RMSEOverall(),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
def run_pi_fno(CONFIG: dict, manage_wandb: bool = True) -> None:
    """Run the training process for the PI-FNO model."""
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
    train_loss = build_train_loss(CONFIG)

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
    run_pi_fno(CONFIG)
