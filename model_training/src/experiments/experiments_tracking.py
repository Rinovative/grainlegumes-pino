"""
Provide optional experiment tracking without coupling training to an SDK.

W&B is imported only for explicitly enabled runs. The SDK owns standard
``WANDB_API_KEY`` discovery; this module never reads credentials or probes
host/container secret files.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, Unpack, cast

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

EpochEndCallback = Callable[[int, dict[str, float]], None]


class _WandbRun(Protocol):
    """Describe the SDK run surface used by the lifecycle adapter."""

    summary: MutableMapping[str, Any]

    def log(self, data: Mapping[str, Any], *, step: int) -> None:
        """Log one evaluated epoch."""

    def finish(self, exit_code: int = 0) -> None:
        """Finalize the remote/local tracking run."""


class _WandbInitKwargs(TypedDict):
    """Type the exact W&B initialization keywords used by this adapter."""

    project: str
    entity: str | None
    group: str | None
    tags: list[str]
    mode: str
    name: str
    id: str
    resume: str
    job_type: str
    dir: str
    config: Mapping[str, Any]


class _WandbModule(Protocol):
    """Describe the lazily imported W&B module surface."""

    def init(self, **kwargs: Unpack[_WandbInitKwargs]) -> _WandbRun | None:
        """Initialize one W&B run."""


@dataclass(slots=True)
class WandbSession:
    """Own one optional W&B run and its idempotent lifecycle."""

    _run: _WandbRun | None
    objective_id: str
    _finished: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether this session owns an initialized SDK run."""
        return self._run is not None

    def log_epoch(
        self,
        epoch: int,
        metrics: Mapping[str, float],
        *,
        learning_rate: float,
    ) -> None:
        """Log one evaluated epoch using stable semantic key names."""
        if self._run is None:
            return
        payload: dict[str, float | int] = {
            "epoch": epoch,
            "optimizer/learning_rate": learning_rate,
        }
        for metric_id, value in metrics.items():
            key = "train/loss" if metric_id == "train_loss" else f"evaluation/{metric_id}"
            payload[key] = float(value)
        if self.objective_id in metrics:
            payload["objective/value"] = float(metrics[self.objective_id])
        self._run.log(payload, step=epoch)

    def finish(
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Publish terminal summary values and shut down exactly once."""
        if self._run is None or self._finished:
            return
        exit_code = 0 if status == "completed" else 1
        try:
            self._run.summary["status"] = status
            self._run.summary["objective/id"] = self.objective_id
            if result is not None:
                for key in ("best_epoch", "best_metric", "completed_epoch", "global_step"):
                    if key in result:
                        self._run.summary[key] = result[key]
            if error is not None:
                self._run.summary["error"] = error
            self._run.finish(exit_code=exit_code)
        finally:
            self._finished = True


def initialize_wandb(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str,
) -> WandbSession:
    """Initialize an enabled W&B run or return a no-op disabled session."""
    objective = cast("Mapping[str, Any]", config["evaluation"])["objective"]
    objective_id = str(cast("Mapping[str, Any]", objective)["id"])
    tracking = cast("Mapping[str, Any]", config["tracking"])
    settings = cast("Mapping[str, Any]", tracking["wandb"])
    if not bool(settings["enabled"]):
        return WandbSession(None, objective_id)

    try:
        wandb = cast("_WandbModule", importlib.import_module("wandb"))
    except ModuleNotFoundError as error:
        msg = "tracking.wandb.enabled=true requires the 'wandb' package."
        raise RuntimeError(msg) from error

    run_name = str(cast("Mapping[str, Any]", config["run"])["name"])
    task_id = str(config["task"])
    run_id = hashlib.sha256(f"{task_id}\0{run_name}".encode()).hexdigest()[:32]
    sdk_run = wandb.init(
        project=str(settings["project"]),
        entity=cast("str | None", settings["entity"]),
        group=cast("str | None", settings["group"]),
        tags=list(cast("list[str]", settings["tags"])),
        mode=str(settings["mode"]),
        name=run_name,
        id=run_id,
        resume="allow",
        job_type="training",
        dir=str(Path(run_dir)),
        config=copy.deepcopy(dict(config)),
    )
    if sdk_run is None:
        msg = "wandb.init() did not return a run for an enabled tracking configuration."
        raise RuntimeError(msg)
    return WandbSession(sdk_run, objective_id)


def epoch_callback(
    session: WandbSession,
    optimizer: Optimizer,
) -> EpochEndCallback | None:
    """Bind an enabled W&B session to optimizer learning-rate state."""
    if not session.enabled:
        return None

    def callback(epoch: int, metrics: dict[str, float]) -> None:
        parameter_groups = optimizer.param_groups
        if not parameter_groups:
            msg = "Cannot log W&B learning rate: optimizer has no parameter groups."
            raise RuntimeError(msg)
        session.log_epoch(
            epoch,
            metrics,
            learning_rate=float(parameter_groups[0]["lr"]),
        )

    return callback


def combine_epoch_callbacks(
    *callbacks: EpochEndCallback | None,
) -> EpochEndCallback | None:
    """Return one ordered callback over all non-null lifecycle consumers."""
    active = tuple(callback for callback in callbacks if callback is not None)
    if not active:
        return None

    def combined(epoch: int, metrics: dict[str, float]) -> None:
        for callback in active:
            callback(epoch, metrics)

    return combined
