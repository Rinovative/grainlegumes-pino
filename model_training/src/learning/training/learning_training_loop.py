"""
===============================================================================
learning_training_loop.py
===============================================================================
Run the custom training and evaluation loop with checkpoint support.

Responsibilities:
  - Execute training and evaluation epochs
  - Manage scheduler updates, checkpoints and best metrics
  - Track histories, RNG state and optional mixed precision
  - Invoke optional epoch-end callbacks for tuning

Design principles:
  - Reproducibility state is explicit in checkpoints
  - Data, model and loss objects are caller-provided
  - Controller behavior stays independent of model architecture

Boundaries:
  - Config loading belongs to experiments.config.loader
  - Model and loss construction belong to learning factories
  - CLI argument parsing belongs to experiments.cli
===============================================================================
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

TensorBatch = dict[str, torch.Tensor]
EpochEndCallback = Callable[[int, dict[str, float]], None]


def set_seed(seed: int = 9) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    _ = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _move_batch_to_device(batch: Any, device: torch.device) -> TensorBatch:
    """Move an existing dataset batch to the target device."""
    if not isinstance(batch, dict):
        msg = f"Expected dataloader batch to be a dict, got: {type(batch).__name__}"
        raise TypeError(msg)

    tensor_batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    if "x" not in tensor_batch or "y" not in tensor_batch:
        msg = "Training batches must contain tensor keys 'x' and 'y'."
        raise KeyError(msg)
    return tensor_batch


def _prepare_batch(
    raw_batch: Any,
    device: torch.device,
    data_processor: Any | None,
    *,
    training: bool,
) -> TensorBatch:
    """Prepare one raw dataloader batch for model execution."""
    if data_processor is None:
        return _move_batch_to_device(raw_batch, device)

    if training:
        data_processor.train()
    else:
        data_processor.eval()

    processed = data_processor.preprocess(dict(raw_batch))
    return _move_batch_to_device(processed, device)


def _compute_loss(loss_fn: nn.Module, pred: torch.Tensor, batch: TensorBatch) -> torch.Tensor:
    """Compute a supervised or PINO-style loss for one prepared batch."""
    if hasattr(loss_fn, "set_normalizers"):
        return loss_fn(pred, x=batch["x"], y=batch["y"])
    return loss_fn(pred, batch["y"])


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    data_processor: Any | None = None,
    scaler: Any | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    Execute one training epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    optimizer : Optimizer
        Optimizer
    loss_fn : nn.Module
        Loss function
    device : torch.device
        Compute device
    data_processor : Any | None
        Optional neuralop data processor
    scaler : Any | None
        GradScaler for mixed precision (optional)
    use_amp : bool
        Whether to use automatic mixed precision

    Returns
    -------
    dict[str, float]
        Dictionary with key "train_loss" (average loss over epoch)

    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for raw_batch in train_loader:
        batch = _prepare_batch(raw_batch, device, data_processor, training=True)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.autocast(device_type=device.type):
                pred = model(batch["x"])
                loss = _compute_loss(loss_fn, pred, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(batch["x"])
            loss = _compute_loss(loss_fn, pred, batch)
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"train_loss": avg_loss}


def eval_one_epoch(
    model: nn.Module,
    eval_loader: DataLoader,
    loss_fns: dict[str, nn.Module],
    device: torch.device,
    data_processor: Any | None = None,
) -> dict[str, float]:
    """
    Execute one evaluation epoch.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    eval_loader : DataLoader
        Evaluation data loader
    loss_fns : dict[str, nn.Module]
        Dictionary of named loss functions to evaluate
    device : torch.device
        Compute device
    data_processor : Any | None
        Optional neuralop data processor

    Returns
    -------
    dict[str, float]
        Dictionary with evaluation metrics

    """
    model.eval()
    metrics = dict.fromkeys(loss_fns, 0.0)
    n_batches = 0

    with torch.no_grad():
        for raw_batch in eval_loader:
            batch = _prepare_batch(raw_batch, device, data_processor, training=False)
            pred = model(batch["x"])
            if data_processor is not None:
                pred, processed_batch = data_processor.postprocess(pred, batch)
                batch = _move_batch_to_device(processed_batch, device)

            for name, loss_fn in loss_fns.items():
                metrics[name] += _compute_loss(loss_fn, pred, batch).detach().item()

            n_batches += 1

    for name in metrics:
        metrics[name] /= max(n_batches, 1)

    return {f"eval_{name}": val for name, val in metrics.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    save_dir: Path | str,
    filename: str = "best_checkpoint.pt",
) -> Path:
    """
    Save checkpoint with full state.

    Parameters
    ----------
    model : nn.Module
        Model to save
    optimizer : Optimizer
        Optimizer state
    scheduler : ReduceLROnPlateau | None
        Scheduler state (optional)
    epoch : int
        Current epoch
    best_metric : float
        Best metric value so far
    best_epoch : int
        Epoch with best metric
    save_dir : Path | str
        Directory to save checkpoint
    filename : str
        Checkpoint filename

    Returns
    -------
    Path
        Path to saved checkpoint

    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    return save_path


def train_loop(
    config: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    train_loss: nn.Module,
    eval_losses: dict[str, nn.Module],
    data_processor: Any | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    save_dir: Path | str | None = None,
    use_amp: bool = False,
    resume_from: Path | str | None = None,
    epoch_end_callback: EpochEndCallback | None = None,
) -> dict[str, Any]:
    """
    Run the main training loop.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    model : nn.Module
        Model to train
    optimizer : Optimizer
        Optimizer
    train_loader : DataLoader
        Training data loader
    eval_loader : DataLoader
        Evaluation data loader
    train_loss : nn.Module
        Training loss function
    eval_losses : dict[str, nn.Module]
        Evaluation loss functions
    data_processor : Any | None, optional
        Optional neuralop data processor used for normalization
    scheduler : ReduceLROnPlateau | None, optional
        Learning rate scheduler (optional)
    save_dir : Path | str | None, optional
        Directory to save checkpoints (optional)
    use_amp : bool, optional
        Use automatic mixed precision (default: False)
    resume_from : Path | str | None, optional
        Path to checkpoint to resume from (optional)
    epoch_end_callback : EpochEndCallback | None, optional
        Callback invoked after each evaluation epoch with epoch number and metrics

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "best_epoch": epoch with best metric
        - "best_metric": best metric value
        - "checkpoint_path": path to best checkpoint if saved
        - "status": "completed" or other status

    """
    device = torch.device(config["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(config["run"].get("seed", 9)))

    model = model.to(device)
    if data_processor is not None:
        data_processor.to(device)

    n_epochs = int(config["training"].get("n_epochs", 600))
    eval_interval = int(config["training"].get("eval_interval", 5))
    if eval_interval <= 0:
        msg = f"training.eval_interval must be positive, got: {eval_interval}"
        raise ValueError(msg)
    save_best_metric = str(config["training"].get("save_best_metric", "eval_overall_rmse"))

    best_metric = float("inf")
    best_epoch = 0
    scaler = None

    use_amp = use_amp and device.type == "cuda"
    if use_amp:
        torch_amp = importlib.import_module("torch.amp")
        scaler = torch_amp.GradScaler("cuda")

    save_dir = Path(save_dir) if save_dir else None

    start_epoch = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_metric = checkpoint.get("best_metric", float("inf"))
            best_epoch = checkpoint.get("best_epoch", 0)
            start_epoch = checkpoint.get("epoch", 0)

    for epoch in range(start_epoch, n_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_loss,
            device,
            data_processor,
            scaler,
            use_amp,
        )

        if (epoch + 1) % eval_interval == 0:
            eval_metrics = eval_one_epoch(model, eval_loader, eval_losses, device, data_processor)
            current_metric = eval_metrics.get(save_best_metric)
            if current_metric is None:
                msg = f"Configured save_best_metric {save_best_metric!r} was not produced by eval losses."
                raise KeyError(msg)

            if scheduler is not None:
                scheduler.step(current_metric)

            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1

                if save_dir is not None:
                    save_checkpoint(model, optimizer, scheduler, epoch + 1, best_metric, best_epoch, save_dir)

            if epoch_end_callback is not None:
                epoch_end_callback(epoch + 1, {**train_metrics, **eval_metrics})

            if (epoch + 1) % (eval_interval * 10) == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d} | train_loss: {train_metrics.get('train_loss', 0):.4f} | eval_loss: {eval_metrics.get('eval_h1', 0):.4f}"
                )

    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint_path": str(save_dir / "best_checkpoint.pt") if save_dir else None,
        "status": "completed",
    }
