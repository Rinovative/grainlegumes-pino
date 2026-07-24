"""
===============================================================================
learning_training_loop.py
===============================================================================
Run the custom training and evaluation loop with checkpoint support.

Responsibilities:
  - Execute training and evaluation epochs
  - Consume the resolved semantic objective for scheduler and best-metric updates
  - Manage checkpoints, histories, and reproducibility state
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
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from src import common

from . import learning_training_checkpoint as checkpoints

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

TensorBatch = dict[str, torch.Tensor]
EpochEndCallback = Callable[[int, dict[str, float]], None]


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
    """Compute one semantic composition or conventional supervised loss."""
    if hasattr(loss_fn, "compute_components"):
        return loss_fn(pred, x=batch["x"], y=batch["y"])
    return loss_fn(pred, batch["y"])


def _require_finite_training_loss(loss: torch.Tensor) -> None:
    """Reject a non-finite batch loss before backward or optimizer mutation."""
    if not bool(torch.isfinite(loss.detach()).all().item()):
        msg = f"Training loss is non-finite before backward: {loss.detach()}."
        raise FloatingPointError(msg)


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
    sample_count = 0
    optimizer_steps = 0

    for raw_batch in train_loader:
        batch = _prepare_batch(raw_batch, device, data_processor, training=True)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.autocast(device_type=device.type):
                pred = model(batch["x"])
                loss = _compute_loss(loss_fn, pred, batch)
            _require_finite_training_loss(loss)
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += int(float(scaler.get_scale()) >= scale_before)
        else:
            pred = model(batch["x"])
            loss = _compute_loss(loss_fn, pred, batch)
            _require_finite_training_loss(loss)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
        batch_samples = int(batch["y"].shape[0])
        total_loss += loss.detach().item() * batch_samples
        sample_count += batch_samples

    if sample_count == 0:
        msg = "Training loader produced no samples."
        raise RuntimeError(msg)
    avg_loss = total_loss / sample_count
    return {"train_loss": avg_loss, "optimizer_steps": float(optimizer_steps)}


def eval_one_epoch(
    model: nn.Module,
    eval_loader: Iterable[Any],
    eval_metrics: dict[str, Any],
    device: torch.device,
    data_processor: Any | None = None,
) -> dict[str, float]:
    """
    Execute evaluation with explicit tensor spaces and dataset accumulation.

    Normalized model outputs and targets remain available unchanged. When at
    least one physical metric is configured, each is inverse-normalized exactly
    once per batch before any metric sees it. Metrics accumulate sufficient
    statistics and finalize only after the complete loader.
    """
    model.eval()
    for metric_id, metric in eval_metrics.items():
        if getattr(metric, "id", metric_id) != metric_id:
            msg = f"Evaluation metric key {metric_id!r} does not match its resolved id."
            raise ValueError(msg)
        metric.reset()

    requires_physical = any(getattr(metric, "space", None) == "physical" for metric in eval_metrics.values())
    if requires_physical and data_processor is None:
        msg = "Physical evaluation metrics require a fitted data processor."
        raise RuntimeError(msg)

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(eval_loader):
            batch = _prepare_batch(raw_batch, device, data_processor, training=False)
            pred_normalized = model(batch["x"])
            target_normalized = batch["y"]
            views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "normalized": (pred_normalized, target_normalized),
            }
            if requires_physical:
                out_normalizer = getattr(data_processor, "out_normalizer", None)
                if out_normalizer is None:
                    msg = "Physical evaluation metrics require data_processor.out_normalizer."
                    raise RuntimeError(msg)
                pred_physical = out_normalizer.inverse_transform(pred_normalized)
                target_physical = out_normalizer.inverse_transform(target_normalized)
                views["physical"] = (pred_physical, target_physical)

            for metric in eval_metrics.values():
                space = str(metric.space)
                try:
                    pred_view, target_view = views[space]
                except KeyError as error:
                    msg = f"Metric {metric.id!r} requested unavailable tensor space {space!r}."
                    raise RuntimeError(msg) from error
                metric.update(
                    pred_view,
                    target_view,
                    space=space,
                    batch_index=batch_index,
                )

    return {metric_id: metric.compute() for metric_id, metric in eval_metrics.items()}


def train_loop(  # noqa: C901, PLR0912, PLR0915
    config: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    train_loss: nn.Module,
    eval_metrics: dict[str, Any],
    data_processor: Any | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    save_dir: Path | str | None = None,
    use_amp: bool = False,
    resume_from: Path | str | None = None,
    epoch_end_callback: EpochEndCallback | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """
    Train through exact completed-epoch checkpoints.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved runtime config.
    model, optimizer, train_loader, eval_loader, train_loss, eval_metrics : Any
        Already constructed runtime components. Fresh-run seeding must occur
        before their construction.
    data_processor : Any | None, optional
        Saved/fitted normalization processor.
    scheduler : ReduceLROnPlateau | None, optional
        Configured objective scheduler.
    save_dir : Path | str | None, optional
        Canonical run directory. Required for lifecycle execution.
    use_amp : bool, optional
        Request CUDA automatic mixed precision.
    resume_from : Path | str | None, optional
        Exact ``last_checkpoint.pt`` continuation source.
    epoch_end_callback : Callable | None, optional
        Callback invoked only after an evaluated epoch is safely checkpointed.
    checkpoint_identity : Mapping[str, Any] | None, optional
        Immutable task/config/dataset/split/objective identity.
    scaler : Any | None, optional
        Injected scaler for focused tests; normally constructed internally.

    Returns
    -------
    dict[str, Any]
        Completed progress, best/last paths, finite objective, and history.

    Raises
    ------
    ValueError
        If duration, objective, checkpoint identity, or finite-value contracts fail.

    """
    if save_dir is None:
        msg = "Canonical training requires save_dir for best and last checkpoints."
        raise ValueError(msg)
    run_dir = Path(save_dir)
    if not run_dir.is_dir():
        msg = f"Allocated run directory does not exist: {run_dir}"
        raise FileNotFoundError(msg)
    identity = dict(checkpoint_identity or {})
    if not identity:
        msg = "Canonical training requires a non-empty checkpoint_identity."
        raise ValueError(msg)

    device = torch.device(config["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    if data_processor is not None:
        data_processor.to(device)

    n_epochs = int(config["training"]["epochs"])
    eval_interval = int(config["training"]["evaluation_interval"])
    if n_epochs <= 0:
        msg = f"training.epochs must be positive, got: {n_epochs}"
        raise ValueError(msg)
    if eval_interval <= 0:
        msg = f"training.evaluation_interval must be positive, got: {eval_interval}"
        raise ValueError(msg)
    objective = config["evaluation"]["objective"]
    objective_id = str(objective["id"])
    objective_direction = str(objective["direction"])
    if objective_direction not in {"minimize", "maximize"}:
        msg = f"Unknown objective direction {objective_direction!r}."
        raise ValueError(msg)
    if objective_id not in eval_metrics:
        msg = f"Configured evaluation objective {objective_id!r} is absent from evaluation metrics."
        raise KeyError(msg)

    amp_enabled = bool(use_amp and device.type == "cuda")
    if amp_enabled and scaler is None:
        torch_amp = importlib.import_module("torch.amp")
        scaler = torch_amp.GradScaler("cuda")
    if not amp_enabled and scaler is not None:
        msg = "A scaler was supplied while CUDA AMP is inactive."
        raise ValueError(msg)

    best_metric: float | None = None
    best_epoch: int | None = None
    objective_history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch_index = 0

    if resume_from is not None:
        resume_path = Path(resume_from)
        last_payload = checkpoints.load_checkpoint(
            resume_path,
            expected_identity=identity,
            expected_role="last",
            scheduler_expected=scheduler is not None,
            amp_expected=amp_enabled,
            require_best=False,
        )
        restored = checkpoints.restore_checkpoint(
            last_payload,
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            loss=train_loss,
            train_loader=train_loader,
        )
        start_epoch_index = int(restored["next_epoch"]) - 1
        global_step = int(restored["global_step"])
        best_metric = restored["best_metric"]
        best_epoch = restored["best_epoch"]
        objective_history = list(restored["objective_history"])
        if start_epoch_index >= n_epochs:
            msg = (
                f"Resume checkpoint already completed epoch {restored['completed_epoch']}, "
                f"but runtime training.epochs is {n_epochs}. Increase the terminal duration deliberately."
            )
            raise ValueError(msg)

    best_path = common.paths.resolve_best_checkpoint_file(run_dir)
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)

    for epoch_index in range(start_epoch_index, n_epochs):
        completed_epoch = epoch_index + 1
        set_epoch = getattr(train_loss, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch_index)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_loss,
            device,
            data_processor,
            scaler,
            amp_enabled,
        )
        optimizer_steps = int(train_metrics.pop("optimizer_steps"))
        global_step += optimizer_steps

        evaluated: dict[str, float] = {}
        should_evaluate = completed_epoch % eval_interval == 0 or completed_epoch == n_epochs
        if should_evaluate:
            evaluated = eval_one_epoch(model, eval_loader, eval_metrics, device, data_processor)
            current_metric = evaluated.get(objective_id)
            if current_metric is None:
                msg = f"Configured evaluation objective {objective_id!r} was not produced by evaluation metrics."
                raise KeyError(msg)
            current_metric = float(current_metric)
            if not math.isfinite(current_metric):
                msg = f"Evaluation objective {objective_id!r} is non-finite at epoch {completed_epoch}: {current_metric}."
                raise FloatingPointError(msg)
            objective_history.append(
                {
                    "epoch": completed_epoch,
                    "objective_id": objective_id,
                    "value": current_metric,
                }
            )

            if scheduler is not None:
                scheduler.step(current_metric)

            is_better = best_metric is None or (current_metric < best_metric if objective_direction == "minimize" else current_metric > best_metric)
            if is_better:
                best_metric = current_metric
                best_epoch = completed_epoch
                best_payload = checkpoints.make_checkpoint(
                    role="best",
                    identity=identity,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    loss=train_loss,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    objective_history=objective_history,
                    train_loader=train_loader,
                    runtime_device=device,
                )
                checkpoints.save_checkpoint(best_payload, best_path)

        last_payload = checkpoints.make_checkpoint(
            role="last",
            identity=identity,
            completed_epoch=completed_epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            loss=train_loss,
            best_metric=best_metric,
            best_epoch=best_epoch,
            objective_history=objective_history,
            train_loader=train_loader,
            runtime_device=device,
        )
        checkpoints.save_checkpoint(last_payload, last_path)

        if should_evaluate and epoch_end_callback is not None:
            epoch_end_callback(completed_epoch, {**train_metrics, **evaluated})
        if should_evaluate and (completed_epoch % (eval_interval * 10) == 0 or completed_epoch in {1, n_epochs}):
            print(
                f"Epoch {completed_epoch:3d} | train_loss: {train_metrics.get('train_loss', 0):.4f} | {objective_id}: {evaluated[objective_id]:.4f}"
            )

    if best_metric is None or best_epoch is None:
        msg = "Training produced no finite objective and cannot be marked completed."
        raise RuntimeError(msg)
    best_checkpoint = checkpoints.load_checkpoint(
        best_path,
        expected_identity=identity,
        expected_role="best",
        scheduler_expected=scheduler is not None,
        amp_expected=amp_enabled,
        require_best=True,
    )
    last_checkpoint = checkpoints.load_checkpoint(
        last_path,
        expected_identity=identity,
        expected_role="last",
        scheduler_expected=scheduler is not None,
        amp_expected=amp_enabled,
        require_best=True,
    )
    if last_checkpoint["completed_epoch"] != n_epochs:
        msg = "Last checkpoint does not represent the configured terminal epoch."
        raise RuntimeError(msg)
    if best_checkpoint["best_metric"] != last_checkpoint["best_metric"] or best_checkpoint["best_epoch"] != last_checkpoint["best_epoch"]:
        msg = "Best and last checkpoints disagree about the selected objective state."
        raise RuntimeError(msg)

    return {
        "completed_epoch": n_epochs,
        "next_epoch": n_epochs + 1,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "objective": dict(objective),
        "objective_history": objective_history,
        "checkpoint_path": str(best_path),
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "status": "completed",
    }
