"""
Persistent line-oriented console reporting for experiment lifecycles.

The reporter consumes the same resolved config and completed-epoch payloads as
checkpointing and W&B. It never recomputes scientific values and emits no
batch-level progress by default.
"""

from __future__ import annotations

import math
import re
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src import common, learning

from .config import experiments_config_loader as config_loader

if TYPE_CHECKING:
    from pathlib import Path

_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")


def _redact_secrets(value: object) -> str:
    """Redact credential-shaped values while preserving surrounding text."""
    return _SECRET_PATTERN.sub(r"\1\2<redacted>", str(value))


def _clean(value: object) -> str:
    """Return one bounded, single-line, secret-redacted display value."""
    text = _redact_secrets(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:600]


def _value(value: object) -> str:
    """Format one scalar without changing its stored authoritative value."""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.8g}"
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(_clean(item) for item in value)
    return _clean(value)


def _emit(event: str, /, **fields: object) -> None:
    """Emit and promptly flush one persistent lifecycle line."""
    raw_to_stderr = fields.pop("_to_stderr", False)
    if type(raw_to_stderr) is not bool:
        msg = "Console destination flag must be a boolean."
        raise TypeError(msg)
    parts = [f"event={event}"]
    parts.extend(f"{key}={_value(value)}" for key, value in fields.items() if value is not None)
    print(" ".join(parts), file=sys.stderr if raw_to_stderr else sys.stdout, flush=True)


def _metric_summary(
    metrics: Mapping[str, float],
    *,
    prefix: str,
    metric_ids: tuple[str, ...],
) -> str:
    """Format resolved evaluation metrics from an already computed payload."""
    return ",".join(f"{metric_id}:{_value(metrics[f'{prefix}{metric_id}'])}" for metric_id in metric_ids if f"{prefix}{metric_id}" in metrics)


def _loss_composition(config: Mapping[str, Any]) -> str:
    """Describe the implemented active total-loss composition."""
    loss = config["loss"]
    data = loss["data"]
    terms = [f"{data['weight']}*{data['kind']}[{data['space']}]"]
    physics = loss["physics"]
    if bool(physics["enabled"]):
        residual = physics["residual_weight"]
        boundary = physics["boundary_weight"]
        terms.extend(
            (
                f"residual_weight(epoch,target={residual['target']},warmup={residual['warmup']['epochs']})*momentum_and_{physics['continuity']}",
                f"boundary_weight(epoch,target={boundary['target']},warmup={boundary['warmup']['epochs']})*pressure_boundary",
            )
        )
    return "+".join(terms)


@dataclass(slots=True)
class ConsoleReporter:
    """Render startup, completed-epoch, final, and failure lifecycle events."""

    config: Mapping[str, Any]
    run_dir: Path
    resume: bool = False
    study_name: str | None = None
    trial_number: int | None = None
    _started: float = field(default_factory=time.perf_counter)

    @property
    def objective_id(self) -> str:
        """Return the explicitly resolved selection metric ID."""
        return str(self.config["evaluation"]["objective"]["id"])

    @property
    def metric_ids(self) -> tuple[str, ...]:
        """Return the objective first, followed by resolved diagnostic metrics."""
        declared = tuple(str(metric["id"]) for metric in self.config["evaluation"]["metrics"])
        return (self.objective_id, *(metric_id for metric_id in declared if metric_id != self.objective_id))

    def startup(self, *, resolved_device: str) -> None:
        """Emit one resolved-run line and one active-loss-composition line."""
        run = self.config["run"]
        data = self.config["data"]
        training = self.config["training"]
        wandb = self.config["tracking"]["wandb"]
        monitor = wandb["monitor"]
        physics = self.config["loss"]["physics"]
        objective = self.config["evaluation"]["objective"]
        model_variant = config_loader.resolved_model_variant(self.config)
        id_interval = int(training["evaluation_interval"])
        ood_interval = int(training["ood_evaluation_interval"])
        physics_interval = int(monitor["interval"])
        target = int(training["epochs"])
        _emit(
            "startup",
            run_name=run["name"],
            project=wandb["project"],
            task=self.config["task"],
            model_kind=self.config["model"]["kind"],
            architecture=model_variant,
            suffix=run.get("suffix"),
            device=resolved_device,
            seed=run["seed"],
            deterministic=run["deterministic"],
            train_dataset=data["train_dataset"],
            id_membership="saved_eval_split",
            id_interpretation="validation",
            ood_datasets=data["ood_datasets"],
            batch_size=data["batch_size"],
            workers=data["num_workers"],
            epochs=target,
            mixed_precision=training["mixed_precision"],
            optimizer=self.config["optimizer"]["kind"],
            initial_lr=self.config["optimizer"]["lr"],
            scheduler=None if self.config.get("scheduler") is None else self.config["scheduler"]["kind"],
            training_loss=self.config["loss"]["data"]["kind"],
            pi_enabled=physics["enabled"],
            continuity=physics["continuity"] if physics["enabled"] else None,
            derivatives=physics["derivatives"]["kind"] if physics["enabled"] else None,
            objective=self.objective_id,
            direction=objective["direction"],
            id_interval=id_interval,
            id_first=learning.training.events.first_completed_epoch_event(interval=id_interval, target_epoch=target),
            ood_interval=ood_interval,
            ood_first=learning.training.events.first_completed_epoch_event(interval=ood_interval, target_epoch=target),
            physics_interval=physics_interval,
            physics_first=learning.training.events.first_completed_epoch_event(interval=physics_interval, target_epoch=target),
            wandb_mode=wandb["mode"],
            workflow=wandb["workflow"],
            state="resume" if self.resume else "fresh",
            run_dir=self.run_dir,
            study=self.study_name,
            trial=self.trial_number,
        )
        _emit(
            "loss_composition",
            run_name=run["name"],
            formula=_loss_composition(self.config),
            continuity=physics["continuity"] if physics["enabled"] else None,
            derivative_kind=physics["derivatives"]["kind"] if physics["enabled"] else None,
            derivative_extension=physics["derivatives"]["extension"] if physics["enabled"] else None,
        )

    def epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Emit one training line and each due evaluation/checkpoint event."""
        target = int(self.config["training"]["epochs"])
        train_components = ",".join(
            f"{key.removeprefix('train/').removeprefix('physics/train/')}:{_value(value)}"
            for key, value in metrics.items()
            if key.startswith(("train/", "physics/train/"))
        )
        _emit(
            "training_epoch",
            run_name=self.config["run"]["name"],
            epoch=f"{epoch}/{target}",
            total_loss=metrics.get("train/loss_total"),
            components=train_components or None,
            learning_rate=metrics.get("optimization/learning_rate"),
            train_seconds=metrics.get("system/train_duration_seconds"),
            samples_per_second=metrics.get("system/train_samples_per_second"),
            epoch_seconds=metrics.get("system/epoch_duration_seconds"),
            elapsed_seconds=metrics.get("system/session_elapsed_seconds"),
            eta_seconds=metrics.get("system/estimated_remaining_seconds"),
            last_checkpoint="published" if metrics.get("checkpoint/last_published") == 1.0 else None,
        )

        id_key = f"id/{self.objective_id}"
        if id_key in metrics:
            _emit(
                "id_evaluation",
                run_name=self.config["run"]["name"],
                epoch=epoch,
                membership="id",
                interpretation="validation",
                cases=int(metrics.get("system/id_evaluation_case_count", 0.0)),
                objective_id=self.objective_id,
                objective=metrics[id_key],
                metrics=_metric_summary(metrics, prefix="id/", metric_ids=self.metric_ids),
                duration_seconds=metrics.get("system/id_evaluation_duration_seconds"),
                new_best=bool(metrics.get("checkpoint/new_best", 0.0)),
            )
        ood_key = f"ood/{self.objective_id}"
        if ood_key in metrics:
            _emit(
                "ood_evaluation",
                run_name=self.config["run"]["name"],
                epoch=epoch,
                membership="ood",
                dataset=self.config["data"]["ood_datasets"][0],
                cases=int(metrics.get("system/ood_evaluation_case_count", 0.0)),
                objective_id=self.objective_id,
                objective=metrics[ood_key],
                metrics=_metric_summary(metrics, prefix="ood/", metric_ids=self.metric_ids),
                duration_seconds=metrics.get("system/ood_evaluation_duration_seconds"),
            )
        physics_key = "physics/id/momentum_residual_mse"
        if physics_key in metrics:
            physics = self.config["loss"]["physics"]
            _emit(
                "physics_monitor",
                run_name=self.config["run"]["name"],
                epoch=epoch,
                membership="bounded_saved_id_prefix",
                cases=int(metrics.get("system/physics_monitor_case_count", 0.0)),
                derivative_kind=physics["derivatives"]["kind"],
                momentum_residual_mse=metrics[physics_key],
                continuity_div_velocity_mse=metrics["physics/id/continuity_div_velocity_mse"],
                continuity_div_epsilon_velocity_mse=metrics["physics/id/continuity_div_eps_velocity_mse"],
                pressure_boundary_mse=metrics["physics/id/pressure_boundary_mse"],
                duration_seconds=metrics.get("system/physics_monitor_duration_seconds"),
            )
        if metrics.get("checkpoint/new_best") == 1.0:
            _emit(
                "best_checkpoint",
                run_name=self.config["run"]["name"],
                epoch=epoch,
                objective_id=self.objective_id,
                objective=metrics[id_key],
                previous_best=metrics.get("checkpoint/previous_best_objective"),
                direction=self.config["evaluation"]["objective"]["direction"],
                path=common.paths.resolve_best_checkpoint_file(self.run_dir),
            )
        if "optimization/scheduler_new_learning_rate" in metrics:
            _emit(
                "scheduler_update",
                run_name=self.config["run"]["name"],
                epoch=epoch,
                scheduler=self.config["scheduler"]["kind"],
                objective_id=self.objective_id,
                objective=metrics.get(id_key),
                old_learning_rate=metrics["optimization/scheduler_old_learning_rate"],
                new_learning_rate=metrics["optimization/scheduler_new_learning_rate"],
            )

    def final(self, result: Mapping[str, Any], *, total_wall_seconds: float) -> None:
        """Emit final ownership from the reloaded selected best checkpoint."""
        selected = result["selected_metrics"]
        physics = ",".join(
            f"{key.removeprefix('selected/physics/')}:{_value(value)}" for key, value in selected.items() if key.startswith("selected/physics/")
        )
        _emit(
            "completed",
            run_name=self.config["run"]["name"],
            completed_epoch=result["completed_epoch"],
            selected_best_epoch=result["best_epoch"],
            selected_objective=result["best_metric"],
            selected_source="reloaded_best_checkpoint.pt",
            best_checkpoint=result["best_checkpoint_path"],
            last_checkpoint=result["last_checkpoint_path"],
            final_id_objective=selected[f"selected/id/{self.objective_id}"],
            final_ood_objective=selected[f"selected/ood/{self.objective_id}"],
            final_physics=physics,
            total_wall_seconds=total_wall_seconds,
            run_dir=self.run_dir,
        )

    def failure(self, error: BaseException, *, status: str, phase: str | None = None) -> None:
        """Emit sanitized phase context and retain the complete traceback."""
        active_phase = phase or getattr(error, "training_phase", "run_lifecycle")
        epoch = getattr(error, "completed_epoch", None)
        _emit(
            "failure",
            _to_stderr=True,
            run_name=self.config["run"]["name"],
            phase=active_phase,
            epoch=epoch,
            exception_type=type(error).__name__,
            message=_clean(error),
            last_checkpoint_exists=common.paths.resolve_last_checkpoint_file(self.run_dir).is_file(),
            run_dir=self.run_dir,
            wandb_status=status,
        )
        print_sanitized_traceback(error)


def sanitized_exception_message(error: BaseException) -> str:
    """Return one bounded, single-line, secret-redacted exception message."""
    return _clean(error)


def print_sanitized_traceback(error: BaseException) -> None:
    """Print the complete exception chain and frames with secrets redacted."""
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    sys.stderr.write(_redact_secrets(rendered))
    sys.stderr.flush()


def optuna_study_event(status: str, **fields: object) -> None:
    """Emit one bounded, stable, secret-safe Optuna study lifecycle line."""
    normalized = dict(fields)
    parameters = normalized.get("best_parameters")
    if isinstance(parameters, Mapping):
        normalized["best_parameters"] = ",".join(f"{key}:{_value(value)}" for key, value in sorted(parameters.items()))
    _emit("optuna_study", status=status, **normalized)


def optuna_trial_event(
    status: str,
    *,
    study: str,
    trial: int,
    run_name: str,
    sampled: Mapping[str, Any],
    study_role: str | None = None,
    objective_id: str | None = None,
    objective: float | None = None,
    best_trial_objective: float | None = None,
    best_study_objective: float | None = None,
    step: int | None = None,
    pruning: str | None = None,
    pruner: str | None = None,
    pruning_eligible: bool | None = None,
    report_duration_seconds: float | None = None,
    training_seed: int | None = None,
    sampler_seed: int | None = None,
    device: str | None = None,
    run_dir: object | None = None,
    max_epochs: int | None = None,
    selected_best_epoch: int | None = None,
    final_state: str | None = None,
    duration_seconds: float | None = None,
    checkpoint_state: str | None = None,
    wandb_url: str | None = None,
    phase: str | None = None,
    exception_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Emit one bounded Optuna trial lifecycle or genuine ID-report line."""
    sampled_text = ",".join(f"{key}:{_value(value)}" for key, value in sorted(sampled.items()))
    _emit(
        "optuna_trial",
        status=status,
        study=study,
        study_role=study_role,
        trial=trial,
        run_name=run_name,
        sampled=sampled_text,
        objective_id=objective_id,
        objective=objective,
        best_trial_objective=best_trial_objective,
        best_study_objective=best_study_objective,
        step=step,
        pruning=pruning,
        pruner=pruner,
        pruning_eligible=pruning_eligible,
        report_duration_seconds=report_duration_seconds,
        training_seed=training_seed,
        sampler_seed=sampler_seed,
        device=device,
        run_dir=run_dir,
        max_epochs=max_epochs,
        selected_best_epoch=selected_best_epoch,
        final_state=final_state,
        duration_seconds=duration_seconds,
        checkpoint_state=checkpoint_state,
        wandb_url=wandb_url,
        phase=phase,
        exception_type=exception_type,
        error_message=_clean(error_message) if error_message is not None else None,
    )
