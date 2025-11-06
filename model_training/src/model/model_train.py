"""
============================================================
 PINO Training Framework
============================================================
Author:  Rino M. Albertin
Date:    2025-10-29
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Generic, high-stability training loop for Physics-Informed Neural Operators
(PINOs) using neuralp.

Features:
  AMP mixed precision
  CosineAnnealingWarmRestarts
  MSE + PDE residual loss
  Early stopping & checkpointing
  W&B logging (train/val/test)
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from torch.amp import GradScaler, autocast
from torcheval.metrics.functional import r2_score

# ============================================================
# --- Device & AMP setup
# ============================================================


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_amp_context(device):
    if device.type == "cuda":
        return autocast("cuda", dtype=torch.float16)
    elif device.type == "mps":
        return autocast("mps", dtype=torch.float16)
    elif device.type == "cpu":
        return autocast("cpu", dtype=torch.bfloat16)
    else:
        return nullcontext()


# ============================================================
# --- Combined PINO loss
# ============================================================


def pino_loss(pred, target, physics_residual=None, λ_phys: float = 1.0):
    """
    Combined data + physics loss.
    pred, target: [B, C, H, W]
    physics_residual: scalar tensor or None
    λ_phys: weighting for PDE residual
    """
    mse = torch.mean((pred - target) ** 2)
    if physics_residual is not None:
        loss_phys = torch.mean(physics_residual**2)
        total = mse + λ_phys * loss_phys
        return total, mse, loss_phys
    else:
        return mse, mse, torch.tensor(0.0, device=pred.device)


# ============================================================
# --- Training loop
# ============================================================


def train_pino(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    physics_fn=None,  # callable(model, batch) -> PDE residual
    λ_phys=1.0,
    patience=150,
    min_delta=1e-4,
):
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    amp_ctx = get_amp_context(device)

    # Cosine LR schedule (works well for PINOs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # --- wandb setup ---
    run = wandb.init(
        project="GrainLegumes_PINO",
        entity="Rinovative-Hub",
        config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": type(optimizer).__name__,
            "λ_phys": λ_phys,
            "early_stop_patience": patience,
            "min_delta": min_delta,
        },
    )
    wandb.watch(model, log="all", log_freq=20)

    best_val = np.inf
    epochs_no_improve = 0
    total_step = len(train_loader)

    model.to(device)
    print("🚀 Starting PINO training")

    # ============================================================
    # --- Epoch loop
    # ============================================================
    for epoch in range(num_epochs):
        model.train()
        train_losses, val_losses = [], []

        # -------------------------------
        # 🔹 Training phase
        # -------------------------------
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = batch["input_fields"].to(device), batch["output_fields"].to(device)

            with amp_ctx:
                pred = model(x)
                phys_res = physics_fn(model, batch) if physics_fn is not None else None
                loss_total, loss_data, loss_phys = pino_loss(pred, y, phys_res, λ_phys)

            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

            scheduler.step(epoch + step / total_step)
            train_losses.append(loss_total.item())

            if (step + 1) % 10 == 0:
                wandb.log(
                    {
                        "train/total_loss": loss_total.item(),
                        "train/data_loss": loss_data.item(),
                        "train/phys_loss": loss_phys.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    }
                )

        # -------------------------------
        # 🔹 Validation phase
        # -------------------------------
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["input_fields"].to(device), batch["output_fields"].to(device)
                with amp_ctx:
                    pred = model(x)
                    phys_res = physics_fn(model, batch) if physics_fn is not None else None
                    loss_total, loss_data, loss_phys = pino_loss(pred, y, phys_res, λ_phys)
                    val_losses.append(loss_total.item())

        val_mean = np.mean(val_losses)
        train_mean = np.mean(train_losses)
        wandb.log({"val/total_loss": val_mean, "val/epoch": epoch + 1})

        print(f"Epoch [{epoch + 1}/{num_epochs}] TrainLoss={train_mean:.4e}  ValLoss={val_mean:.4e}")

        # --- Early stopping ---
        if val_mean < best_val - min_delta:
            best_val = val_mean
            epochs_no_improve = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_mean),
            }
            os.makedirs("models", exist_ok=True)
            torch.save(checkpoint, "models/PINO_best_model.pth")
            print(f"💾 Saved new best model (val_loss={val_mean:.4e})")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch + 1}")
            break

    print(f"\n🏁 Training complete. Best validation loss: {best_val:.4e}")
    wandb.finish()
    return model


# ============================================================
# --- Evaluation helper
# ============================================================


def evaluate_pino(model, test_loader, device, physics_fn=None):
    model.eval()
    l2_errors, r2_scores = [], []
    with torch.no_grad():
        with get_amp_context(device):
            for batch in test_loader:
                x, y = batch["input_fields"].to(device), batch["output_fields"].to(device)
                pred = model(x)
                err = torch.mean((pred - y) ** 2)
                rel_l2 = torch.sqrt(err) / torch.sqrt(torch.mean(y**2))
                r2 = r2_score(pred.flatten(), y.flatten())
                l2_errors.append(rel_l2.item())
                r2_scores.append(r2.item())

    print(f"📊 Test Results → mean rel L2 = {np.mean(l2_errors):.4f}, mean R² = {np.mean(r2_scores):.4f}")
    wandb.log({"test/rel_l2": np.mean(l2_errors), "test/r2": np.mean(r2_scores)})
