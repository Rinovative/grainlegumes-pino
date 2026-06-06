"""
Model training script for Physics-Informed U-shaped Neural Operator (PI-U-NO).

This script sets up the PI-U-NO model, optimizer, scheduler, loss functions,
and launches the training process using the base training function.
"""

from pathlib import Path

import torch
from neuralop import H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import UNO
from neuralop.training import AdamW
from src.learning.metrics import metrics as learning_metrics
from torch import nn
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
    "pino_loss_type": "sp_u",
    "grad_mode": "fft_reflect",  # only for spectral losses
    "interior_pad": 2,  # useful for sp_*, can be 0 for ps_*
    "lambda_phys": 0.0001,
    "lambda_p": 0.0005,
    "phys_warmup_epochs": 300,
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
    # "resume_from_dir": "latest",
    # --- Logging ---
    "save_best": "eval_overall_rmse",  # metric key to monitor for best checkpoint
    "log_physical_rmse": True,  # log RMSE in physical units for each channel
    "save_every": None,  # optional periodic checkpoint saving
}


def finalize_config(CONFIG: dict) -> None:
    """Finalize the configuration by setting default values for missing keys."""
    # Architecture
    CONFIG.setdefault("n_layers", 7)
    CONFIG.setdefault("hidden_channels", 32)
    CONFIG.setdefault("modes_x", 64)
    CONFIG.setdefault("modes_y", 64)
    CONFIG.setdefault("mode_ratio", 0.4951318201313778)
    CONFIG.setdefault(
        "uno_scalings",
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ],
    )

    # Loss selection defaults
    CONFIG.setdefault("pino_loss_type", "sp_eps")
    CONFIG.setdefault("grad_mode", "fft_reflect")
    # interior_pad default: spectral sinnvoll >0, physical sinnvoll 0
    lt = str(CONFIG.get("pino_loss_type", "sp_eps"))
    CONFIG.setdefault("interior_pad", 2 if lt.startswith("sp") else 0)

    # Physics-informed weights
    _ = CONFIG["lambda_phys"]
    _ = CONFIG["lambda_p"]
    _ = CONFIG["phys_warmup_epochs"]


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

    mode_ratio = float(CONFIG.get("mode_ratio", 0.4951318201313778))

    base_x = int(CONFIG["modes_x"])
    base_y = int(CONFIG["modes_y"])

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
    else:
        msg = f"Unsupported n_layers for UNO: {n_layers}"
        raise ValueError(msg)

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
    """Build a descriptive model name based on the configuration."""
    scaling_tag = "-".join(str(int(s[0]) if float(s[0]).is_integer() else s[0]).replace(".", "") for s in model.uno_scalings)

    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    loss_type = str(CONFIG.get("pino_loss_type", "sp_eps"))
    tag_map = {
        "ps_eps": "PI-UNO-PS-EPS",
        "ps_u": "PI-UNO-PS-DIV",
        "sp_eps": "PI-UNO-SP-EPS",
        "sp_u": "PI-UNO-SP-DIV",
    }
    if loss_type not in tag_map:
        msg = f"Unknown pino_loss_type: {loss_type}"
        raise ValueError(msg)

    parts = [
        tag_map[loss_type],
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"s{scaling_tag}",
        f"mr{CONFIG.get('mode_ratio', 0.5):.3g}".replace(".", "p"),
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
        CONFIG["train_dataset_name"],
    ]

    # optional: grad mode nur bei spectral
    if loss_type.startswith("sp"):
        parts.append(f"grad{CONFIG.get('grad_mode', 'fft_reflect')}")

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


def build_optimizer(CONFIG: dict, model: UNO) -> AdamW:
    """Build the AdamW optimizer for the model."""
    return AdamW(
        model.parameters(),
        lr=CONFIG.get("lr", 0.0085115429814798161),
        weight_decay=CONFIG.get("weight_decay", 1e-4),
    )


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
def build_train_loss(CONFIG: dict) -> nn.Module:
    """Build the training loss function based on the configuration."""
    loss_type = str(CONFIG.get("pino_loss_type", "sp_eps"))

    grad_mode = str(CONFIG.get("grad_mode", "fft_reflect"))
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

    kwargs: dict = {
        "data_loss": H1Loss(d=2),
        "lambda_phys": CONFIG["lambda_phys"],
        "lambda_p": CONFIG["lambda_p"],
        "in_normalizer": None,
        "out_normalizer": None,
        "interior_pad": interior_pad,
    }

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
def run_pi_uno(CONFIG: dict, manage_wandb: bool = True) -> None:
    """Run the PI-U-NO training process."""
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
    run_pi_uno(CONFIG)
