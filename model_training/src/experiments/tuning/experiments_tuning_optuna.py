"""
===============================================================================
experiments_tuning_optuna.py
===============================================================================
Run reusable Optuna studies and trial objectives.

Responsibilities:
  - Load Optuna YAML files with inline base experiments
  - Resolve base experiment configs and search spaces
  - Apply trial overrides and run custom-loop trials
  - Store trial configs, summaries, split indices and normalizers

Design principles:
  - CLI code delegates reusable tuning behavior here
  - Search spaces are YAML-defined and auditable
  - Optuna imports stay lazy for help and dry-run validation

Boundaries:
  - Search-space parsing belongs to experiments.tuning.search_space
  - Training execution belongs to learning.training.loop
  - Repository cleanup belongs outside Optuna orchestration
===============================================================================
"""

from __future__ import annotations

import copy
import gc
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from src import experiments, learning

from . import experiments_tuning_search_space as search_space

N_MODES_2D_LENGTH = 2


class TrialProtocol(Protocol):
    """Minimal Optuna trial surface used by the objective."""

    number: int

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        """Suggest one value from categorical choices."""
        ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        """Suggest a floating-point value."""
        ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        """Suggest an integer value."""
        ...

    def report(self, value: float, step: int) -> None:
        """Report an intermediate metric value."""

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a serializable trial user attribute."""

    def should_prune(self) -> bool:
        """Return whether Optuna wants to prune this trial."""
        ...


@dataclass(frozen=True)
class OptunaStudyConfig:
    """
    Resolved Optuna study configuration.

    Parameters
    ----------
    path : Path
        Source Optuna YAML path
    study : dict[str, Any]
        Study-level settings such as name, metric, direction, and n_trials
    base_experiment : dict[str, Any]
        Inline base experiment block from the Optuna YAML
    base_config : dict[str, Any]
        Base experiment resolved through config defaults
    search_space : tuple[search_space.SearchSpaceParameter, ...]
        Parsed search-space parameters

    """

    path: Path
    study: dict[str, Any]
    base_experiment: dict[str, Any]
    base_config: dict[str, Any]
    search_space: tuple[search_space.SearchSpaceParameter, ...]


@dataclass
class OptunaEpochReporter:
    """
    Report evaluation metrics from the custom training loop to Optuna.

    Parameters
    ----------
    trial : TrialProtocol
        Optuna trial
    metric_name : str
        Metric key to report and prune on
    direction : str
        Study direction: minimize or maximize
    report_epochs : set[int]
        Epoch numbers to report. Empty means every evaluation epoch.
    hard_prune_spike : bool
        Whether to prune sudden metric explosions for minimize studies

    """

    trial: TrialProtocol
    metric_name: str
    direction: str
    report_epochs: set[int]
    hard_prune_spike: bool = True
    report_index: int = 0
    last_value: float | None = None
    best_value: float | None = None
    best_epoch: int | None = None

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Report one evaluated epoch to Optuna and prune if requested.

        Parameters
        ----------
        epoch : int
            Completed epoch number
        metrics : dict[str, float]
            Metrics produced by the training loop at this epoch

        """
        if self.report_epochs and epoch not in self.report_epochs:
            return

        value = metrics.get(self.metric_name)
        if value is None:
            msg = f"Optuna metric {self.metric_name!r} was not produced by train_loop"
            raise KeyError(msg)
        value = float(value)

        trial_pruned = _trial_pruned_error()
        if not math.isfinite(value):
            msg_0 = f"Non-finite Optuna metric {self.metric_name}: {value}"
            raise trial_pruned(msg_0)

        if self.best_value is None or _is_better(value, self.best_value, self.direction):
            self.best_value = value
            self.best_epoch = epoch

        previous = self.last_value
        if self.hard_prune_spike and self.direction == "minimize" and previous is not None:
            spike_factor = 8.0 if self.report_index == 0 else 4.0
            if value > spike_factor * previous:
                msg_1 = f"Stage metric exploded: {value:.3e} > {spike_factor:g} x {previous:.3e}"
                raise trial_pruned(msg_1)

        self.trial.report(value, step=self.report_index)
        self.trial.set_user_attr("last_reported_epoch", epoch)
        self.trial.set_user_attr("last_reported_metric", value)
        self.last_value = value
        self.report_index += 1

        if self.trial.should_prune():
            msg_2 = f"Pruned by Optuna at epoch {epoch}"
            raise trial_pruned(msg_2)


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Validate and return a mapping value."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping, got: {type(value).__name__}"
        raise TypeError(msg)
    return cast("Mapping[str, Any]", value)


def _trial_pruned_error() -> type[Exception]:
    """Return Optuna's TrialPruned exception class using a lazy import."""
    exceptions = importlib.import_module("optuna.exceptions")
    return cast("type[Exception]", exceptions.TrialPruned)


def _optuna_module() -> Any:
    """Import Optuna lazily for study creation and CLI help friendliness."""
    return importlib.import_module("optuna")


def _is_better(value: float, best: float, direction: str) -> bool:
    """Return whether value improves best under the study direction."""
    if direction == "maximize":
        return value > best
    return value < best


def _normalise_study(raw_study: Mapping[str, Any], base_config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Fill study defaults and validate study-level settings."""
    study = dict(copy.deepcopy(raw_study))
    study.setdefault("name", source_path.stem)
    study.setdefault("direction", "minimize")
    study.setdefault("metric", base_config.get("training", {}).get("save_best_metric", "eval_overall_rmse"))
    study.setdefault("seed", base_config.get("run", {}).get("seed", 9))
    study.setdefault("n_trials", 30)
    study.setdefault("pruner", "median")
    study.setdefault("sampler", "tpe")

    direction = str(study["direction"])
    if direction not in ("minimize", "maximize"):
        msg = f"study.direction must be 'minimize' or 'maximize', got: {direction!r}"
        raise ValueError(msg)
    study["direction"] = direction
    study["name"] = str(study["name"])
    study["metric"] = str(study["metric"])
    study["seed"] = int(study["seed"])
    study["n_trials"] = int(study["n_trials"])

    report_epochs = study.get("report_epochs", study.get("budgets"))
    if report_epochs is not None:
        if isinstance(report_epochs, (str, bytes)) or not isinstance(report_epochs, Sequence):
            msg = "study.report_epochs/budgets must be a sequence of positive integers"
            raise TypeError(msg)
        parsed_epochs = [int(epoch) for epoch in report_epochs]
        if any(epoch <= 0 for epoch in parsed_epochs):
            msg = f"study.report_epochs/budgets must be positive, got: {parsed_epochs!r}"
            raise ValueError(msg)
        study["report_epochs"] = parsed_epochs

    return study


def load_optuna_study_config(path: Path | str) -> OptunaStudyConfig:
    """
    Load an Optuna YAML file and resolve its base experiment config.

    Parameters
    ----------
    path : Path | str
        Path to an Optuna YAML file

    Returns
    -------
    OptunaStudyConfig
        Resolved study configuration

    Raises
    ------
    KeyError
        If required blocks are missing
    TypeError
        If required blocks have invalid types

    """
    source_path = Path(path)
    raw = experiments.config.loader.load_yaml(source_path)
    raw_mapping = _as_mapping(raw, label="Optuna YAML")

    if "experiment" not in raw_mapping:
        msg = "Optuna YAML must contain an experiment block"
        raise KeyError(msg)
    if "search_space" not in raw_mapping:
        msg = "Optuna YAML must contain a search_space block"
        raise KeyError(msg)

    base_experiment = dict(copy.deepcopy(_as_mapping(raw_mapping["experiment"], label="experiment")))
    base_config = experiments.config.loader.resolve_config(base_experiment)
    study = _normalise_study(_as_mapping(raw_mapping.get("study", {}), label="study"), base_config, source_path)
    search_parameters = search_space.parse_search_space(raw_mapping["search_space"])

    return OptunaStudyConfig(
        path=source_path,
        study=study,
        base_experiment=base_experiment,
        base_config=base_config,
        search_space=search_parameters,
    )


def describe_optuna_study_config(config: OptunaStudyConfig) -> dict[str, Any]:
    """
    Return a serializable summary for dry-run validation.

    Parameters
    ----------
    config : OptunaStudyConfig
        Resolved Optuna study config

    Returns
    -------
    dict[str, Any]
        Summary suitable for CLI output

    """
    return {
        "path": str(config.path),
        "study": config.study,
        "base_run_name": config.base_config["run"]["name"],
        "task": config.base_config["task"],
        "model_architecture": config.base_config["model"]["architecture"],
        "search_space": search_space.search_space_summary(config.search_space),
    }


def _study_dir(config: OptunaStudyConfig) -> Path:
    """Return the study directory under the configured train root."""
    train_root = Path(config.base_config["paths"]["train_root"])
    return train_root / config.base_config["task"] / "optuna" / str(config.study["name"])


def _build_pruner(study: Mapping[str, Any]) -> Any:
    """Build an Optuna pruner from study config."""
    optuna = _optuna_module()
    raw_pruner = study.get("pruner", "median")
    pruner_cfg = {"type": raw_pruner} if isinstance(raw_pruner, str) else dict(_as_mapping(raw_pruner, label="study.pruner"))
    pruner_type = str(pruner_cfg.get("type", "median")).lower()

    if pruner_type in ("none", "nop", "noop"):
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 0)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 0)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )
    msg = f"Unsupported Optuna pruner: {pruner_type!r}"
    raise ValueError(msg)


def _build_sampler(study: Mapping[str, Any]) -> Any:
    """Build an Optuna sampler from study config."""
    optuna = _optuna_module()
    raw_sampler = study.get("sampler", "tpe")
    sampler_cfg = {"type": raw_sampler} if isinstance(raw_sampler, str) else dict(_as_mapping(raw_sampler, label="study.sampler"))
    sampler_type = str(sampler_cfg.get("type", "tpe")).lower()
    raw_seed = sampler_cfg.get("seed")
    if raw_seed is None:
        raw_seed = study.get("seed", 9)
    if raw_seed is None:
        msg = "study.sampler.seed/study.seed must not be None"
        raise TypeError(msg)
    seed = int(raw_seed)

    if sampler_type in ("tpe", "tpemultivariate"):
        return optuna.samplers.TPESampler(seed=seed, multivariate=bool(sampler_cfg.get("multivariate", False)))
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    msg = f"Unsupported Optuna sampler: {sampler_type!r}"
    raise ValueError(msg)


def _report_epochs(study: Mapping[str, Any]) -> set[int]:
    """Return configured reporting epochs, or an empty set for every eval."""
    raw_epochs = study.get("report_epochs", ())

    if raw_epochs is None:
        return set()

    if isinstance(raw_epochs, (str, bytes)) or not isinstance(raw_epochs, Sequence):
        msg = "study.report_epochs must be a sequence of positive integers"
        raise TypeError(msg)

    return {int(epoch) for epoch in raw_epochs}


def _as_required_int(value: Any, *, label: str) -> int:
    """Return value as int, rejecting missing values explicitly."""
    if value is None:
        msg = f"{label} must not be None"
        raise TypeError(msg)
    return int(value)


def _trial_capacity(config: dict[str, Any]) -> int | None:
    """Estimate a simple architecture capacity attribute for trial metadata."""
    params = config.get("model", {}).get("params", {})
    if not isinstance(params, Mapping):
        return None

    hidden = params.get("hidden_channels")
    layers = params.get("n_layers")
    batch_size = config.get("data", {}).get("batch_size")

    if hidden is None or layers is None or batch_size is None:
        return None

    batch_size_int = _as_required_int(batch_size, label="data.batch_size")
    hidden_int = _as_required_int(hidden, label="model.params.hidden_channels")
    layers_int = _as_required_int(layers, label="model.params.n_layers")

    if "n_modes" in params:
        n_modes = params["n_modes"]
        if isinstance(n_modes, Sequence) and not isinstance(n_modes, (str, bytes)) and len(n_modes) >= N_MODES_2D_LENGTH:
            modes_x = _as_required_int(n_modes[0], label="model.params.n_modes[0]")
            modes_y = _as_required_int(n_modes[1], label="model.params.n_modes[1]")
            return batch_size_int * hidden_int * layers_int * modes_x * modes_y

    if "modes_x" in params and "modes_y" in params:
        modes_x = _as_required_int(params.get("modes_x"), label="model.params.modes_x")
        modes_y = _as_required_int(params.get("modes_y"), label="model.params.modes_y")
        return batch_size_int * hidden_int * layers_int * modes_x * modes_y

    return None


def _prepare_trial_config(study_config: OptunaStudyConfig, trial: TrialProtocol) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample a trial, apply overrides, and assign trial run metadata."""
    overrides = search_space.suggest_trial_overrides(trial, study_config.search_space)
    config = search_space.apply_trial_overrides(study_config.base_config, overrides)

    config.setdefault("run", {})
    config["run"]["suffix"] = f"trial{trial.number:04d}"
    config["run"].pop("name", None)
    config["run"]["name"] = experiments.config.loader.generate_run_name(config)
    config["optuna"] = {
        "study_name": study_config.study["name"],
        "trial_number": trial.number,
        "metric": study_config.study["metric"],
        "overrides": overrides,
    }

    capacity = _trial_capacity(config)
    if capacity is not None:
        trial.set_user_attr("capacity", capacity)
    trial.set_user_attr("run_name", config["run"]["name"])
    trial.set_user_attr("overrides", overrides)
    return config, overrides


def _trial_run_dir(config: dict[str, Any]) -> Path:
    """Return the configured run directory for a trial."""
    return Path(config["paths"]["train_root"]) / config["task"] / "runs" / config["run"]["name"]


def _finite_objective_value(result: Mapping[str, Any], trial_pruned: type[Exception]) -> float:
    """Return the finite Optuna objective value or raise TrialPruned."""
    objective_value = float(result["best_metric"])
    if not math.isfinite(objective_value):
        msg = "No finite objective value produced"
        raise trial_pruned(msg)
    return objective_value


def _write_summary(
    *,
    run_dir: Path,
    config: dict[str, Any],
    status: str,
    start_time: datetime,
    result: Mapping[str, Any] | None = None,
    reporter: OptunaEpochReporter | None = None,
    error: str | None = None,
) -> None:
    """Write a trial summary JSON file."""
    end_time = datetime.now(UTC)
    result = result or {}
    summary = {
        "task": config["task"],
        "model_architecture": config["model"]["architecture"],
        "study_name": config["optuna"]["study_name"],
        "trial_number": config["optuna"]["trial_number"],
        "metric_name": config["optuna"]["metric"],
        "best_epoch": result.get("best_epoch", reporter.best_epoch if reporter else None),
        "best_metric": result.get("best_metric", reporter.best_value if reporter else None),
        "checkpoint_path": result.get("checkpoint_path"),
        "status": status,
        "error": error,
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_trial(study_config: OptunaStudyConfig, trial: TrialProtocol) -> float:
    """
    Run one Optuna trial through the config-driven training stack.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Resolved Optuna study configuration
    trial : TrialProtocol
        Optuna trial

    Returns
    -------
    float
        Objective metric value

    """
    config, _overrides = _prepare_trial_config(study_config, trial)
    run_dir = _trial_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    experiments.config.loader.save_yaml(config, run_dir / "config.yaml")

    start_time = datetime.now(UTC)
    reporter = OptunaEpochReporter(
        trial=trial,
        metric_name=str(study_config.study["metric"]),
        direction=str(study_config.study["direction"]),
        report_epochs=_report_epochs(study_config.study),
        hard_prune_spike=bool(study_config.study.get("hard_prune_spike", True)),
    )
    trial_pruned = _trial_pruned_error()

    try:
        dataloaders = experiments.config.loader.create_dataloaders_from_config(config)
        data_processor = dataloaders["data_processor"]
        torch.save(data_processor.state_dict(), run_dir / "normalizer.pt")
        torch.save(dataloaders["split_indices"], run_dir / "split_indices.pt")

        model = learning.models.factory.build_model(config)
        train_loss = learning.losses.factory.build_training_loss(config)
        set_normalizers = getattr(train_loss, "set_normalizers", None)
        if callable(set_normalizers):
            set_normalizers(
                in_normalizer=data_processor.in_normalizer,
                out_normalizer=data_processor.out_normalizer,
            )
        eval_losses = learning.losses.factory.build_eval_losses(config, out_normalizer=data_processor.out_normalizer)
        optimizer = learning.training.optim.build_optimizer(model, config)
        scheduler = learning.training.optim.build_scheduler(optimizer, config)

        result = learning.training.loop.train_loop(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=dataloaders["train"],
            eval_loader=dataloaders["eval"],
            train_loss=train_loss,
            eval_losses=eval_losses,
            data_processor=data_processor,
            scheduler=scheduler,
            save_dir=run_dir,
            use_amp=config["training"].get("mixed_precision", False),
            epoch_end_callback=reporter,
        )
        objective_value = _finite_objective_value(result, trial_pruned)

    except trial_pruned as err:
        _write_summary(run_dir=run_dir, config=config, status="pruned", start_time=start_time, reporter=reporter, error=str(err))
        raise
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            _write_summary(run_dir=run_dir, config=config, status="oom_pruned", start_time=start_time, reporter=reporter, error=str(err))
            torch.cuda.empty_cache()
            msg_0 = "Trial pruned after out-of-memory error"
            raise trial_pruned(msg_0) from None
        raise
    else:
        _write_summary(
            run_dir=run_dir,
            config=config,
            status="completed",
            start_time=start_time,
            result=result,
            reporter=reporter,
        )
        return objective_value
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_objective(study_config: OptunaStudyConfig) -> Callable[[TrialProtocol], float]:
    """
    Create an Optuna objective callable for a resolved study config.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Resolved Optuna study configuration

    Returns
    -------
    Callable[[TrialProtocol], float]
        Objective function suitable for study.optimize

    """

    def objective(trial: TrialProtocol) -> float:
        """Run one trial for Optuna's optimize loop."""
        return run_trial(study_config, trial)

    return objective


def _with_runtime_overrides(
    config: OptunaStudyConfig,
    *,
    device: str | None = None,
    output_root: Path | str | None = None,
) -> OptunaStudyConfig:
    """Return a study config copy with CLI runtime overrides applied."""
    base_config = copy.deepcopy(config.base_config)
    if device:
        base_config["run"]["device"] = device
    if output_root:
        base_config["paths"]["train_root"] = str(Path(output_root))
    return replace(config, base_config=base_config)


def run_optuna_study(
    config: OptunaStudyConfig | Path | str,
    *,
    n_trials: int | None = None,
    device: str | None = None,
    output_root: Path | str | None = None,
    show_progress_bar: bool = False,
) -> Any:
    """
    Run an Optuna study from a resolved config or YAML path.

    Parameters
    ----------
    config : OptunaStudyConfig | Path | str
        Resolved Optuna config or path to YAML
    n_trials : int | None, optional
        Override number of trials
    device : str | None, optional
        Override run.device for all trials
    output_root : Path | str | None, optional
        Override train root for study DB and trial run directories
    show_progress_bar : bool, optional
        Whether Optuna should show a progress bar

    Returns
    -------
    Any
        Optuna Study object

    """
    study_config = load_optuna_study_config(config) if isinstance(config, (str, Path)) else config
    study_config = _with_runtime_overrides(study_config, device=device, output_root=output_root)

    optuna = _optuna_module()
    study_dir = _study_dir(study_config)
    study_dir.mkdir(parents=True, exist_ok=True)
    study_name = str(study_config.study["name"])
    storage = study_config.study.get("storage") or f"sqlite:///{study_dir / (study_name + '.db')}"

    study = optuna.create_study(
        study_name=study_name,
        direction=str(study_config.study["direction"]),
        pruner=_build_pruner(study_config.study),
        sampler=_build_sampler(study_config.study),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        create_objective(study_config),
        n_trials=int(n_trials if n_trials is not None else study_config.study["n_trials"]),
        show_progress_bar=show_progress_bar,
    )
    return study
