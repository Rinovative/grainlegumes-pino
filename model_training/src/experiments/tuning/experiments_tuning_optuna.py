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
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from src import common, experiments, learning

from . import experiments_tuning_search_space as search_space


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
        Study-level settings such as name, objective, derived direction, and n_trials
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
    objective_id : str
        Semantic objective metric id to report and prune on
    direction : str
        Study direction: minimize or maximize
    report_epochs : set[int]
        Epoch numbers to report. Empty means every evaluation epoch.

    """

    trial: TrialProtocol
    objective_id: str
    direction: str
    report_epochs: set[int]
    report_index: int = 0
    best_value: float | None = None
    best_epoch: int | None = None

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Track every objective evaluation and report configured epochs."""
        value = metrics.get(self.objective_id)
        if value is None:
            msg = f"Optuna objective {self.objective_id!r} was not produced by train_loop"
            raise KeyError(msg)
        value = float(value)

        trial_pruned = _trial_pruned_error()
        if not math.isfinite(value):
            msg = f"Non-finite Optuna objective {self.objective_id}: {value}"
            raise trial_pruned(msg)

        if self.best_value is None or _is_better(value, self.best_value, self.direction):
            self.best_value = value
            self.best_epoch = epoch

        if self.report_epochs and epoch not in self.report_epochs:
            return

        self.trial.report(value, step=self.report_index)
        self.trial.set_user_attr("last_reported_epoch", epoch)
        self.trial.set_user_attr("last_reported_objective", value)
        self.report_index += 1

        if self.trial.should_prune():
            msg = f"Pruned by Optuna at epoch {epoch}"
            raise trial_pruned(msg)


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
    """Return whether value improves best under the resolved direction."""
    if direction == "maximize":
        return value > best
    if direction == "minimize":
        return value < best
    msg = f"Unknown objective direction {direction!r}."
    raise ValueError(msg)


def _require_nonempty_string(value: Any, *, label: str) -> str:
    """Return a non-empty string without coercing another scalar type."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{label} must be a non-empty string, got: {value!r}"
        raise TypeError(msg)
    return value


def _require_exact_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    """Return one exact integer with an optional inclusive minimum."""
    if type(value) is not int:
        msg = f"{label} must be an integer, got: {value!r}"
        raise TypeError(msg)
    if minimum is not None and value < minimum:
        msg = f"{label} must be at least {minimum}, got: {value}"
        raise ValueError(msg)
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    """Return one exact boolean without truthiness coercion."""
    if type(value) is not bool:
        msg = f"{label} must be a boolean, got: {value!r}"
        raise TypeError(msg)
    return value


def _normalise_sampler_config(value: Any) -> dict[str, Any]:
    """Validate one exact Optuna sampler mapping."""
    if isinstance(value, str):
        msg = "study.sampler must be a mapping with a semantic kind; string shorthand is unsupported."
        raise TypeError(msg)
    sampler = dict(copy.deepcopy(_as_mapping(value, label="study.sampler")))
    allowed = {"kind", "multivariate"}
    unknown = sorted(set(sampler).difference(allowed))
    if unknown:
        msg = f"study.sampler contains unknown key(s): {unknown}."
        raise ValueError(msg)
    if "kind" not in sampler:
        msg = "study.sampler.kind is required"
        raise KeyError(msg)
    kind = _require_nonempty_string(sampler["kind"], label="study.sampler.kind")
    if kind not in {"random", "tpe"}:
        msg = f"Unsupported Optuna sampler: {kind!r}"
        raise ValueError(msg)
    if "multivariate" in sampler:
        sampler["multivariate"] = _require_bool(
            sampler["multivariate"],
            label="study.sampler.multivariate",
        )
        if kind != "tpe":
            msg = "study.sampler.multivariate is only valid for the tpe sampler"
            raise ValueError(msg)
    sampler["kind"] = kind
    return sampler


def _normalise_pruner_config(value: Any) -> dict[str, Any]:
    """Validate one exact Optuna pruner mapping."""
    if isinstance(value, str):
        msg = "study.pruner must be a mapping with a semantic kind; string shorthand is unsupported."
        raise TypeError(msg)
    pruner = dict(copy.deepcopy(_as_mapping(value, label="study.pruner")))
    allowed = {"kind", "n_startup_trials", "n_warmup_steps", "interval_steps"}
    unknown = sorted(set(pruner).difference(allowed))
    if unknown:
        msg = f"study.pruner contains unknown key(s): {unknown}."
        raise ValueError(msg)
    if "kind" not in pruner:
        msg = "study.pruner.kind is required"
        raise KeyError(msg)
    kind = _require_nonempty_string(pruner["kind"], label="study.pruner.kind")
    if kind not in {"median", "none"}:
        msg = f"Unsupported Optuna pruner: {kind!r}"
        raise ValueError(msg)
    tuning_keys = {"n_startup_trials", "n_warmup_steps", "interval_steps"}
    invalid_keys = sorted(tuning_keys.intersection(pruner)) if kind == "none" else []
    if invalid_keys:
        msg = f"study.pruner contains key(s) invalid for none: {invalid_keys}."
        raise ValueError(msg)
    for key in ("n_startup_trials", "n_warmup_steps"):
        if key in pruner:
            pruner[key] = _require_exact_int(
                pruner[key],
                label=f"study.pruner.{key}",
                minimum=0,
            )
    if "interval_steps" in pruner:
        pruner["interval_steps"] = _require_exact_int(
            pruner["interval_steps"],
            label="study.pruner.interval_steps",
            minimum=1,
        )
    pruner["kind"] = kind
    return pruner


def _normalise_report_epochs(value: Any) -> list[int] | None:
    """Validate optional exact, ordered, unique report epoch integers."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "study.report_epochs must be a sequence of positive integers"
        raise TypeError(msg)
    epochs = [_require_exact_int(epoch, label="study.report_epochs entry", minimum=1) for epoch in value]
    if not epochs:
        msg = "study.report_epochs must not be empty when provided"
        raise ValueError(msg)
    if epochs != sorted(set(epochs)):
        msg = "study.report_epochs must be unique and strictly increasing"
        raise ValueError(msg)
    return epochs


def _validate_report_schedule(study: Mapping[str, Any], base_config: Mapping[str, Any]) -> None:
    """Require explicit report epochs to coincide with reachable evaluations."""
    report_epochs = _normalise_report_epochs(study.get("report_epochs"))
    if report_epochs is None:
        return
    training = _as_mapping(base_config.get("training"), label="experiment.training")
    epochs = _require_exact_int(training.get("epochs"), label="training.epochs", minimum=1)
    interval = _require_exact_int(
        training.get("evaluation_interval"),
        label="training.evaluation_interval",
        minimum=1,
    )
    reachable = set(range(interval, epochs + 1, interval)) | {epochs}
    unreachable = [epoch for epoch in report_epochs if epoch not in reachable]
    if unreachable:
        msg = (
            "study.report_epochs must contain only reachable evaluation epochs; "
            f"got {unreachable}, training.epochs={epochs}, evaluation_interval={interval}."
        )
        raise ValueError(msg)


def _validate_study_settings(study: Mapping[str, Any]) -> None:
    """Validate normalized study settings without scalar coercion."""
    allowed = {
        "name",
        "objective",
        "direction",
        "seed",
        "n_trials",
        "report_epochs",
        "sampler",
        "pruner",
        "storage",
    }
    unknown = sorted(set(study).difference(allowed))
    if unknown:
        msg = f"Resolved study contains unknown key(s): {unknown}."
        raise ValueError(msg)
    common.paths.validate_logical_name(
        _require_nonempty_string(study.get("name"), label="study.name"),
        label="study.name",
    )
    _require_nonempty_string(study.get("objective"), label="study.objective")
    _require_exact_int(study.get("seed"), label="study.seed")
    _require_exact_int(study.get("n_trials"), label="study.n_trials", minimum=1)
    _normalise_report_epochs(study.get("report_epochs"))
    _normalise_sampler_config(study.get("sampler"))
    _normalise_pruner_config(study.get("pruner"))
    if "storage" in study:
        storage = study["storage"]
        if storage is not None:
            _require_nonempty_string(storage, label="study.storage")


def _normalise_study(raw_study: Mapping[str, Any], base_config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Validate study settings and derive direction from one objective."""
    allowed = {
        "name",
        "seed",
        "n_trials",
        "report_epochs",
        "sampler",
        "pruner",
        "storage",
    }
    unknown = sorted(set(raw_study).difference(allowed))
    if unknown:
        msg = f"study contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ValueError(msg)

    study = dict(copy.deepcopy(raw_study))
    study.setdefault("name", source_path.stem)
    study.setdefault("seed", base_config.get("run", {}).get("seed", 9))
    study.setdefault("n_trials", 30)
    study.setdefault("pruner", {"kind": "median"})
    study.setdefault("sampler", {"kind": "tpe"})

    objective_config = _as_mapping(base_config["evaluation"]["objective"], label="evaluation.objective")
    experiment_objective = _require_nonempty_string(
        objective_config["id"],
        label="evaluation.objective.id",
    )
    study["objective"] = experiment_objective
    study["direction"] = _require_nonempty_string(
        objective_config["direction"],
        label="evaluation.objective.direction",
    )
    study["name"] = common.paths.validate_logical_name(
        _require_nonempty_string(study["name"], label="study.name"),
        label="study.name",
    )
    study["seed"] = _require_exact_int(study["seed"], label="study.seed")
    study["n_trials"] = _require_exact_int(
        study["n_trials"],
        label="study.n_trials",
        minimum=1,
    )
    if "report_epochs" in study:
        study["report_epochs"] = _normalise_report_epochs(study["report_epochs"])
    study["sampler"] = _normalise_sampler_config(study["sampler"])
    study["pruner"] = _normalise_pruner_config(study["pruner"])
    if "storage" in study and study["storage"] is not None:
        study["storage"] = _require_nonempty_string(
            study["storage"],
            label="study.storage",
        )
    _validate_study_settings(study)
    _validate_report_schedule(study, base_config)
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
    allowed_root = {"study", "experiment", "search_space"}
    unknown_root = sorted(set(raw_mapping).difference(allowed_root))
    if unknown_root:
        msg = f"Optuna YAML contains unknown top-level key(s): {unknown_root}."
        raise ValueError(msg)

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
    search_space.validate_search_space_paths(base_config, search_parameters)

    return OptunaStudyConfig(
        path=source_path,
        study=study,
        base_experiment=base_experiment,
        base_config=base_config,
        search_space=search_parameters,
    )


def _validate_study_contract(config: OptunaStudyConfig) -> tuple[OptunaStudyConfig, dict[str, Any]]:
    """Revalidate a resolved study and return its full objective before output."""
    _validate_study_settings(config.study)
    base_config = experiments.config.loader.validate_resolved_config(config.base_config)
    _validate_report_schedule(config.study, base_config)
    objective = experiments.config.loader.get_resolved_objective(base_config)
    if config.study.get("objective") != objective["id"]:
        msg = "Resolved study objective id does not match its experiment objective."
        raise ValueError(msg)
    if config.study.get("direction") != objective["direction"]:
        msg = "Resolved study direction does not match its experiment objective."
        raise ValueError(msg)
    search_space.validate_search_space_paths(base_config, config.search_space)
    return replace(config, base_config=base_config), objective


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
        "model_kind": config.base_config["model"]["kind"],
        "objective": experiments.config.loader.get_resolved_objective(config.base_config),
        "search_space": search_space.search_space_summary(config.search_space),
    }


def _study_dir(config: OptunaStudyConfig) -> Path:
    """Return the study directory under the independent output root."""
    output_root = Path(config.base_config["paths"]["output_root"])
    study_name = common.paths.validate_logical_name(config.study["name"], label="study.name")
    return output_root / config.base_config["task"] / "optuna" / study_name


def _build_pruner(study: Mapping[str, Any]) -> Any:
    """Build an Optuna pruner from one validated semantic config."""
    pruner_cfg = _normalise_pruner_config(study.get("pruner"))
    optuna = _optuna_module()
    pruner_type = pruner_cfg["kind"]

    if pruner_type == "none":
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 0),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 0),
            interval_steps=pruner_cfg.get("interval_steps", 1),
        )
    msg = f"Unsupported Optuna pruner: {pruner_type!r}"
    raise ValueError(msg)


def _build_sampler(study: Mapping[str, Any]) -> Any:
    """Build an Optuna sampler from one validated semantic config."""
    sampler_cfg = _normalise_sampler_config(study.get("sampler"))
    sampler_type = sampler_cfg["kind"]
    raw_study_seed = _require_exact_int(study.get("seed"), label="study.seed")
    seed = experiments.run.derive_subseed(raw_study_seed, "optuna-sampler")
    optuna = _optuna_module()

    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=sampler_cfg.get("multivariate", False),
        )
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    msg = f"Unsupported Optuna sampler: {sampler_type!r}"
    raise ValueError(msg)


def _report_epochs(study: Mapping[str, Any]) -> set[int]:
    """Return configured reporting epochs, or an empty set for every eval."""
    return set(_normalise_report_epochs(study.get("report_epochs")) or ())


def _prepare_trial_config(study_config: OptunaStudyConfig, trial: TrialProtocol) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample and fully validate a generic resolved trial config."""
    overrides = search_space.suggest_trial_overrides(trial, study_config.search_space)
    config = search_space.apply_trial_overrides(study_config.base_config, overrides)
    config["run"]["seed"] = experiments.run.derive_subseed(
        int(study_config.study["seed"]),
        f"trial-{trial.number}",
    )
    config = experiments.config.loader.validate_resolved_config(config)
    _validate_report_schedule(study_config.study, config)

    base_objective = experiments.config.loader.get_resolved_objective(study_config.base_config)
    trial_objective = experiments.config.loader.get_resolved_objective(config)
    if trial_objective != base_objective:
        msg = "Sampled trial objective does not match the resolved study objective."
        raise ValueError(msg)

    config["run"]["suffix"] = f"trial{trial.number:04d}"
    config["run"].pop("name", None)
    config["run"]["name"] = experiments.config.loader.generate_run_name(config)
    config = experiments.config.loader.validate_resolved_config(config)
    context = {
        "study_name": str(study_config.study["name"]),
        "trial_number": int(trial.number),
        "overrides": overrides,
    }

    trial.set_user_attr("run_name", config["run"]["name"])
    trial.set_user_attr("run_seed", config["run"]["seed"])
    trial.set_user_attr("overrides", overrides)
    return config, context


def _trial_run_dir(config: Mapping[str, Any], context: Mapping[str, Any]) -> Path:
    """Return a study- and trial-qualified output directory."""
    return common.paths.resolve_optuna_trial_dir(
        str(config["task"]),
        str(context["study_name"]),
        int(context["trial_number"]),
        output_root=Path(config["paths"]["output_root"]),
    )


def _finite_objective_value(
    result: Mapping[str, Any],
    objective: Mapping[str, Any],
    trial_pruned: type[Exception],
) -> float:
    """Return the finite resolved objective value or raise TrialPruned."""
    if result.get("objective") != objective:
        msg = "Training result objective does not match the resolved trial objective."
        raise ValueError(msg)
    objective_value = float(result["best_metric"])
    if not math.isfinite(objective_value):
        msg = "No finite objective value produced"
        raise trial_pruned(msg)
    return objective_value


def _write_summary(
    *,
    run_dir: Path,
    config: dict[str, Any],
    context: Mapping[str, Any],
    status: str,
    start_time: datetime,
    result: Mapping[str, Any] | None = None,
    reporter: OptunaEpochReporter | None = None,
    error: str | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    amp_enabled: bool = False,
) -> None:
    """Write a trial summary bound to the resolved objective."""
    end_time = datetime.now(UTC)
    result = result or {}
    objective = experiments.config.loader.get_resolved_objective(config)
    summary = {
        "task": config["task"],
        "model_kind": config["model"]["kind"],
        "study_name": context["study_name"],
        "trial_number": context["trial_number"],
        "objective": objective,
        "best_epoch": result.get("best_epoch", reporter.best_epoch if reporter else None),
        "best_metric": result.get("best_metric", reporter.best_value if reporter else None),
        "checkpoint_path": result.get("checkpoint_path"),
        "status": status,
        "error": error,
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }
    if status == "completed":
        if checkpoint_identity is None:
            msg = "Completed Optuna summary requires checkpoint identity."
            raise ValueError(msg)
        summary.update(
            {
                "run_name": config["run"]["name"],
                "completed_epoch": result.get("completed_epoch"),
                "global_step": result.get("global_step"),
                "best_checkpoint": "best_checkpoint.pt",
                "last_checkpoint": "last_checkpoint.pt",
                "config_sha256": common.serialization.file_sha256(common.paths.resolve_run_config_path(run_dir)),
                "split_indices_sha256": common.serialization.file_sha256(common.paths.resolve_split_indices_path(run_dir)),
                "normalizer_sha256": common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
                "best_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_best_checkpoint_file(run_dir)),
                "last_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_last_checkpoint_file(run_dir)),
                "effective_config_digest": checkpoint_identity["effective_config_digest"],
                "amp_enabled": amp_enabled,
            }
        )
    experiments.run.transition_run_status(run_dir, status, updates=summary)


def run_trial(study_config: OptunaStudyConfig, trial: TrialProtocol) -> float:
    """Run one exclusively allocated, reproducible Optuna trial."""
    study_config, _ = _validate_study_contract(study_config)
    config, context = _prepare_trial_config(study_config, trial)
    objective = experiments.config.loader.get_resolved_objective(config)
    reporter = OptunaEpochReporter(
        trial=trial,
        objective_id=str(objective["id"]),
        direction=str(objective["direction"]),
        report_epochs=_report_epochs(study_config.study),
    )
    trial_pruned = _trial_pruned_error()
    requested_run_dir = _trial_run_dir(config, context)
    summary_extra = dict(context)
    run_dir = experiments.run.prepare_fresh_run(
        config,
        run_dir=requested_run_dir,
        summary_extra=summary_extra,
    )

    start_time = datetime.now(UTC)
    checkpoint_identity: dict[str, Any] | None = None
    amp_enabled = False
    run_started = False
    tracker: experiments.tracking.WandbSession | None = None
    tracking_status = "failed"
    tracking_result: Mapping[str, Any] | None = None
    tracking_error: str | None = None

    try:
        tracker = experiments.tracking.initialize_wandb(config, run_dir=run_dir)
        seed_plan = experiments.run.configure_reproducibility(config)
        dataloaders = experiments.config.loader.create_dataloaders_from_config(
            config,
            seed_plan=seed_plan,
        )
        data_processor = dataloaders["data_processor"]
        common.serialization.atomic_torch_save(
            data_processor.state_dict(),
            common.paths.resolve_normalizer_path(run_dir),
        )
        common.serialization.atomic_torch_save(
            dataloaders["split_indices"],
            common.paths.resolve_split_indices_path(run_dir),
        )

        experiments.run.seed_process(seed_plan["model_init"])
        model = learning.models.factory.build_model(config)
        train_loss = learning.losses.factory.build_training_loss(config)
        set_normalizers = getattr(train_loss, "set_normalizers", None)
        if callable(set_normalizers):
            set_normalizers(
                in_normalizer=data_processor.in_normalizer,
                out_normalizer=data_processor.out_normalizer,
            )
        eval_metrics = learning.losses.factory.build_eval_metrics(config)
        optimizer = learning.training.optim.build_optimizer(model, config)
        scheduler = learning.training.optim.build_scheduler(optimizer, config)
        checkpoint_identity = learning.training.checkpoint.build_checkpoint_identity(
            config,
            dataloaders["split_indices"],
            persisted_config=config,
        )
        amp_enabled = bool(config["training"].get("mixed_precision", False) and torch.device(config["run"]["device"]).type == "cuda")
        experiments.run.transition_run_status(
            run_dir,
            "running",
            updates={
                **summary_extra,
                "target_epochs": int(config["training"]["epochs"]),
                "seed_plan": seed_plan,
                "deterministic": bool(config["run"]["deterministic"]),
                "amp_enabled": amp_enabled,
            },
        )
        run_started = True

        result = learning.training.loop.train_loop(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=dataloaders["train"],
            eval_loader=dataloaders["eval"],
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            data_processor=data_processor,
            scheduler=scheduler,
            save_dir=run_dir,
            use_amp=config["training"].get("mixed_precision", False),
            epoch_end_callback=experiments.tracking.combine_epoch_callbacks(
                reporter,
                experiments.tracking.epoch_callback(tracker, optimizer),
            ),
            checkpoint_identity=checkpoint_identity,
        )
        objective_value = _finite_objective_value(result, objective, trial_pruned)
        tracking_status = "completed"
        tracking_result = result

    except KeyboardInterrupt as err:
        tracking_status = "interrupted"
        tracking_error = str(err)
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="interrupted",
            start_time=start_time,
            reporter=reporter,
            error=str(err),
        )
        raise
    except trial_pruned as err:
        tracking_status = "pruned"
        tracking_error = str(err)
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="pruned",
            start_time=start_time,
            reporter=reporter,
            error=str(err),
        )
        raise
    except FloatingPointError as err:
        tracking_status = "pruned" if run_started else "failed"
        tracking_error = str(err)
        if run_started:
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="pruned",
                start_time=start_time,
                reporter=reporter,
                error=str(err),
            )
            message = f"Trial pruned after non-finite objective: {err}"
            raise trial_pruned(message) from None
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="failed",
            start_time=start_time,
            reporter=reporter,
            error=str(err),
        )
        raise
    except RuntimeError as err:
        tracking_error = str(err)
        if "out of memory" in str(err).lower():
            tracking_status = "oom_pruned"
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="oom_pruned",
                start_time=start_time,
                reporter=reporter,
                error=str(err),
            )
            torch.cuda.empty_cache()
            message = "Trial pruned after out-of-memory error"
            raise trial_pruned(message) from None
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="failed",
            start_time=start_time,
            reporter=reporter,
            error=str(err),
        )
        raise
    except Exception as err:
        tracking_error = str(err)
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="failed",
            start_time=start_time,
            reporter=reporter,
            error=str(err),
        )
        raise
    else:
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="completed",
            start_time=start_time,
            result=result,
            reporter=reporter,
            checkpoint_identity=checkpoint_identity,
            amp_enabled=amp_enabled,
        )
        return objective_value
    finally:
        if tracker is not None:
            with suppress(Exception):
                tracker.finish(
                    status=tracking_status,
                    result=tracking_result,
                    error=tracking_error,
                )
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
        base_config["paths"]["output_root"] = str(Path(output_root).expanduser())
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
        Override only the study and trial run/output root
    show_progress_bar : bool, optional
        Whether Optuna should show a progress bar

    Returns
    -------
    Any
        Optuna Study object

    """
    study_config = load_optuna_study_config(config) if isinstance(config, (str, Path)) else config
    study_config = _with_runtime_overrides(study_config, device=device, output_root=output_root)
    study_config, objective = _validate_study_contract(study_config)

    raw_trial_count = n_trials if n_trials is not None else study_config.study["n_trials"]
    if type(raw_trial_count) is not int:
        msg = f"Optuna n_trials must be an integer, got: {raw_trial_count!r}"
        raise TypeError(msg)
    trial_count = raw_trial_count
    if trial_count <= 0:
        msg = f"Optuna n_trials must be positive, got {trial_count}."
        raise ValueError(msg)
    optuna = _optuna_module()
    pruner = _build_pruner(study_config.study)
    sampler = _build_sampler(study_config.study)

    study_name = common.paths.validate_logical_name(
        study_config.study["name"],
        label="study.name",
    )
    configured_storage = study_config.study.get("storage")
    if configured_storage:
        storage = configured_storage
    else:
        study_dir = _study_dir(study_config)
        study_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{study_dir / (study_name + '.db')}"

    study = optuna.create_study(
        study_name=study_name,
        direction=str(objective["direction"]),
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    actual_direction = str(study.direction.name).lower()
    if actual_direction != objective["direction"]:
        msg = f"Existing Optuna study direction {actual_direction!r} does not match resolved objective direction {objective['direction']!r}."
        raise ValueError(msg)
    stored_objective = study.user_attrs.get("resolved_objective")
    if stored_objective is None:
        if study.trials:
            msg = "Existing Optuna study has trials but no resolved objective signature."
            raise ValueError(msg)
        study.set_user_attr("resolved_objective", objective)
    elif stored_objective != objective:
        msg = "Existing Optuna study objective signature does not match the resolved experiment objective."
        raise ValueError(msg)

    study.optimize(
        create_objective(study_config),
        n_trials=trial_count,
        show_progress_bar=show_progress_bar,
    )
    return study
