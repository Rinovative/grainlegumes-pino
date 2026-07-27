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
  - Invoke optional epoch-end callbacks for local lifecycle observers

Design principles:
  - Reproducibility state is explicit in checkpoints
  - Data, model and loss objects are caller-provided
  - Controller behavior stays independent of model architecture

This module does NOT:
  - Load or resolve configs; ``experiments.config.loader`` owns semantic admission
  - Construct models or losses; caller-selected learning factories own construction
  - Parse CLI arguments; ``experiments.cli`` owns command boundaries
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

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
    """
    Prepare one raw batch with the processor in the requested lifecycle mode.

    When present, the processor is switched to train/eval state and preprocesses
    a copied mapping before tensor transfer. The final batch must contain tensor
    ``x`` and ``y`` keys on the concrete device.
    """
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
    Execute one training epoch and sample-weight every named loss component.

    Batch means are multiplied by their actual sample counts before epoch
    reduction. Supervised runs publish only total/data loss; physics-informed
    runs additionally publish momentum, boundary, one formulation-qualified
    continuity contribution, applied weights, and warmup fractions.

    Parameters
    ----------
    model : torch.nn.Module
        Model mutated through training-mode forward/backward updates.
    train_loader : DataLoader
        Loader providing mapping batches with ``x`` and ``y`` tensors.
    optimizer : Optimizer
        Optimizer stepped once per finite batch, subject to AMP overflow skips.
    loss_fn : torch.nn.Module
        Conventional loss or semantic composition exposing named components.
    device : torch.device
        Concrete device receiving each prepared batch.
    data_processor : Any | None, optional
        Processor switched to training mode before preprocessing each batch.
    scaler : Any | None, optional
        CUDA gradient scaler required when ``use_amp`` is true.
    use_amp : bool, optional
        Whether to execute CUDA autocast and scaled optimization.

    Returns
    -------
    dict[str, float]
        Stable per-epoch training telemetry plus an internal optimizer-step
        count consumed by the checkpoint lifecycle.

    Raises
    ------
    ValueError
        If AMP is requested without CUDA or a scaler.
    FloatingPointError
        If any batch loss is non-finite before optimizer mutation.
    RuntimeError
        If the loader is empty or physics component telemetry is incomplete.

    """
    if use_amp and device.type != "cuda":
        msg = "Mixed-precision training requires a concrete CUDA device; CPU autocast is unsupported."
        raise ValueError(msg)
    if use_amp and scaler is None:
        msg = "Mixed-precision training requires a CUDA GradScaler."
        raise ValueError(msg)

    model.train()
    component_sums: dict[str, float] = {}
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
        raw_components = getattr(loss_fn, "last_components", {})
        components = (
            dict(raw_components) if isinstance(raw_components, Mapping) and raw_components else {"total": loss.detach(), "data": loss.detach()}
        )
        for name, value in components.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                msg = f"Training loss component {name!r} must be one scalar tensor."
                raise TypeError(msg)
            component_sums[name] = component_sums.get(name, 0.0) + float(value.item()) * batch_samples
        sample_count += batch_samples

    if sample_count == 0:
        msg = "Training loader produced no samples."
        raise RuntimeError(msg)

    averaged = {name: value / sample_count for name, value in component_sums.items()}
    result = {
        "train/loss_total": averaged["total"],
        "train/loss_data": averaged.get("data", averaged["total"]),
        "optimizer_steps": float(optimizer_steps),
    }
    if bool(getattr(loss_fn, "physics_enabled", False)):
        continuity = str(getattr(loss_fn, "continuity", ""))
        continuity_component = f"continuity_{continuity}"
        required = {"momentum", "boundary", continuity_component}
        missing = sorted(required.difference(averaged))
        if missing:
            msg = f"Physics-informed epoch aggregation is missing component(s): {missing}."
            raise RuntimeError(msg)
        result.update(
            {
                "train/loss_momentum": averaged["momentum"],
                "train/loss_boundary": averaged["boundary"],
                f"train/loss_continuity_{continuity}": averaged[continuity_component],
            }
        )
        telemetry_state = getattr(loss_fn, "telemetry_state", None)
        if not callable(telemetry_state):
            msg = "Physics-informed loss does not expose applied weight telemetry."
            raise RuntimeError(msg)
        telemetry = cast("Mapping[str, float]", telemetry_state())
        for name, value in telemetry.items():
            result[f"train/{name}"] = float(value)
    return result


def evaluate_physics_monitor(
    model: nn.Module,
    eval_loader: Iterable[Any],
    loss_fn: nn.Module,
    device: torch.device,
    data_processor: Any,
    *,
    max_cases: int,
) -> dict[str, float]:
    """
    Evaluate bounded deterministic physics monitors on the saved eval prefix.

    The caller persists the exact prefix membership. This function consumes no
    more than ``max_cases`` in loader order, reuses the semantic loss's domain
    physics evaluator and saved normalizers, and produces no artifact files.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model evaluated temporarily in evaluation mode.
    eval_loader : Iterable[Any]
        Deterministic saved-evaluation membership in source order.
    loss_fn : torch.nn.Module
        Semantic loss exposing ``compute_physics_diagnostics``.
    device : torch.device
        Concrete device for bounded inference.
    data_processor : Any
        Fitted processor owning the saved normalizers.
    max_cases : int
        Positive upper bound on prefix samples, not batches.

    Returns
    -------
    dict[str, float]
        Sample-weighted momentum, both continuity, and boundary monitor means.

    Raises
    ------
    TypeError
        If batches or the diagnostic interface violate their contracts.
    ValueError
        If ``max_cases`` is not a positive exact integer.
    FloatingPointError
        If a diagnostic scalar is non-finite.
    RuntimeError
        If the selected membership produces no samples.

    Notes
    -----
    The model's incoming training/evaluation state is restored even on failure.

    """
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = f"max_cases must be a positive integer, got {max_cases!r}."
        raise ValueError(msg)
    compute = getattr(loss_fn, "compute_physics_diagnostics", None)
    if not callable(compute):
        msg = "Physics monitoring requires the semantic physics diagnostic adapter."
        raise TypeError(msg)

    totals = {
        "monitor/momentum_residual_mse": 0.0,
        "monitor/div_velocity_mse": 0.0,
        "monitor/div_eps_velocity_mse": 0.0,
        "monitor/pressure_boundary_mse": 0.0,
    }
    sample_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for raw_batch in eval_loader:
                if not isinstance(raw_batch, Mapping):
                    msg = f"Expected monitor batch mapping, got {type(raw_batch).__name__}."
                    raise TypeError(msg)
                remaining = max_cases - sample_count
                if remaining <= 0:
                    break
                raw_y = raw_batch.get("y")
                raw_x = raw_batch.get("x")
                if not isinstance(raw_x, torch.Tensor) or not isinstance(raw_y, torch.Tensor):
                    msg = "Monitor batches must contain tensor keys 'x' and 'y'."
                    raise TypeError(msg)
                take = min(remaining, int(raw_y.shape[0]))
                batch = _prepare_batch(
                    {"x": raw_x[:take], "y": raw_y[:take]},
                    device,
                    data_processor,
                    training=False,
                )
                pred = model(batch["x"])
                diagnostics = cast("Any", compute(pred, x=batch["x"]))
                values = {
                    "monitor/momentum_residual_mse": diagnostics.momentum_residual_mse,
                    "monitor/div_velocity_mse": diagnostics.div_velocity_mse,
                    "monitor/div_eps_velocity_mse": diagnostics.div_eps_velocity_mse,
                    "monitor/pressure_boundary_mse": diagnostics.boundary_mse,
                }
                for name, value in values.items():
                    scalar = float(value.detach().item())
                    if not math.isfinite(scalar):
                        msg = f"Physics monitor {name!r} is non-finite: {scalar}."
                        raise FloatingPointError(msg)
                    totals[name] += scalar * take
                sample_count += take
    finally:
        model.train(was_training)

    if sample_count == 0:
        msg = "Physics monitor evaluation membership produced no samples."
        raise RuntimeError(msg)
    return {name: value / sample_count for name, value in totals.items()}


def eval_one_epoch(
    model: nn.Module,
    eval_loader: Iterable[Any],
    eval_metrics: dict[str, Any],
    device: torch.device,
    data_processor: Any | None = None,
) -> dict[str, float]:
    """
    Execute evaluation with explicit tensor spaces and dataset accumulation.

    Evaluation preprocessing normalizes model inputs while preserving physical
    targets. This function derives normalized targets once with the fitted
    output normalizer and derives physical predictions once by inverse transform.
    Metrics accumulate sufficient statistics and finalize only after the loader.

    Parameters
    ----------
    model : torch.nn.Module
        Model evaluated without gradients.
    eval_loader : Iterable[Any]
        Loader providing task ``x`` and ``y`` batches.
    eval_metrics : dict[str, Any]
        Metric-ID keyed explicit-space dataset accumulators.
    device : torch.device
        Concrete device receiving prepared batches.
    data_processor : Any | None, optional
        Fitted processor needed for normalized/physical view construction.

    Returns
    -------
    dict[str, float]
        One finalized dataset value per configured metric ID.

    Raises
    ------
    ValueError
        If metric IDs, spaces, or tensor views are inconsistent.
    RuntimeError
        If required normalizer state or a requested tensor space is unavailable.

    Notes
    -----
    Dataset accumulators are reset at entry and finalized once after all batches;
    no batch-level metric means are averaged.

    """
    model.eval()
    for metric_id, metric in eval_metrics.items():
        if getattr(metric, "id", metric_id) != metric_id:
            msg = f"Evaluation metric key {metric_id!r} does not match its resolved id."
            raise ValueError(msg)
        metric.reset()

    metric_spaces = {str(getattr(metric, "space", "")) for metric in eval_metrics.values()}
    requires_physical = "physical" in metric_spaces
    if requires_physical and data_processor is None:
        msg = "Physical evaluation metrics require a fitted data processor."
        raise RuntimeError(msg)
    out_normalizer = getattr(data_processor, "out_normalizer", None) if data_processor is not None else None
    if data_processor is not None and out_normalizer is None:
        msg = "Evaluation with a data processor requires data_processor.out_normalizer."
        raise RuntimeError(msg)

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(eval_loader):
            batch = _prepare_batch(raw_batch, device, data_processor, training=False)
            pred_normalized = model(batch["x"])
            if data_processor is None:
                target_normalized = batch["y"]
                target_physical = None
            else:
                if out_normalizer is None:
                    msg = "Evaluation normalized target construction requires an output normalizer."
                    raise RuntimeError(msg)
                target_physical = batch["y"]
                target_normalized = out_normalizer.transform(target_physical)
            views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "normalized": (pred_normalized, target_normalized),
            }
            if requires_physical:
                if out_normalizer is None:
                    msg = "Physical evaluation prediction construction requires an output normalizer."
                    raise RuntimeError(msg)
                pred_physical = out_normalizer.inverse_transform(pred_normalized)
                if target_physical is None:
                    msg = "Physical evaluation target construction requires a fitted data processor."
                    raise RuntimeError(msg)
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
    device: torch.device,
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
        Fully resolved runtime config retaining the requested device policy.
    device : torch.device
        Concrete indexed runtime device resolved at the service boundary.
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
        Callback invoked after every completed epoch is safely checkpointed.
        Evaluation metrics are present only on ordinary evaluation events.
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

    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Training requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda" and device.index is None:
        msg = "Training requires an indexed CUDA device resolved by the runtime boundary."
        raise ValueError(msg)
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

    if type(use_amp) is not bool:
        msg = f"use_amp must be boolean, got {use_amp!r}."
        raise TypeError(msg)
    if use_amp and device.type != "cuda":
        msg = "training.mixed_precision=true requires a resolved CUDA device; CPU autocast is unsupported."
        raise ValueError(msg)
    amp_enabled = use_amp
    if amp_enabled and scaler is None:
        scaler = GradScaler("cuda")
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
        epoch_started = time.perf_counter()
        parameter_groups = optimizer.param_groups
        if not parameter_groups:
            msg = "Optimizer has no parameter groups before a training epoch."
            raise RuntimeError(msg)
        epoch_learning_rate = float(parameter_groups[0]["lr"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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

        train_metrics["global_step"] = float(global_step)
        train_metrics["train/learning_rate"] = epoch_learning_rate
        train_metrics["train/epoch_duration_seconds"] = time.perf_counter() - epoch_started
        if device.type == "cuda":
            train_metrics["train/cuda_peak_memory_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
            train_metrics["train/cuda_peak_memory_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))

        if epoch_end_callback is not None:
            epoch_end_callback(completed_epoch, {**train_metrics, **evaluated})
        if should_evaluate and (completed_epoch % (eval_interval * 10) == 0 or completed_epoch in {1, n_epochs}):
            print(f"Epoch {completed_epoch:3d} | train_loss: {train_metrics['train/loss_total']:.4f} | {objective_id}: {evaluated[objective_id]:.4f}")

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
