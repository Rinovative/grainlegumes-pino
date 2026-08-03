"""
===============================================================================
experiments_tuning_optuna.py
===============================================================================
Run reusable Optuna studies and trial objectives.

Responsibilities:
  - Load strict study YAML with an inline resolved experiment and search space
  - Bind reusable studies to scientific semantic signatures before optimization
  - Run fresh numbered trials with actual-completed-epoch reporting and pruning
  - Persist local trial inputs, lifecycle summaries, splits, normalizers, and checkpoints
  - Classify non-finite, OOM, recoverable, interrupted, and unexpected outcomes explicitly

Design principles:
  - Reopening preserves history but each invocation allocates only additional fresh trials
  - Embedded requests use the generic experiment structure; wrappers own study/search policy
  - Wrapper objective and direction derive from the embedded resolved experiment objective
  - Device, output, tracking, and invocation count remain outside scientific identity
  - Local trial state is authoritative; W&B is an optional post-publication observer
  - Optuna imports stay lazy for help and dry-run validation

This module does NOT:
  - Parse search-space schemas; ``experiments.tuning.search_space`` owns admission
  - Execute training epochs; ``learning.training.loop`` owns model optimization
  - Clean repository or run state
===============================================================================
"""

from __future__ import annotations

import copy
import gc
import importlib
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from src import common, datasets, experiments, learning

from . import experiments_tuning_search_space as search_space

STUDY_SIGNATURE_SCHEMA_VERSION = 2
TRIAL_LIFECYCLE_SCHEMA_VERSION = 2
_STUDY_SIGNATURE_ATTR = "semantic_signature"
_STUDY_SIGNATURE_PAYLOAD_ATTR = "semantic_signature_payload"
_STUDY_SIGNATURE_SCHEMA_ATTR = "semantic_signature_schema_version"
_RESOLVED_OBJECTIVE_ATTR = "resolved_objective"
_SAMPLER_METADATA_ATTR = "sampler_metadata"
_STUDY_SUMMARY_FILENAME = "study_summary.json"
_PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS = 20
_STUDY_ROLES = frozenset({"production", "smoke"})
OPTUNA_ROOT_KEYS = frozenset({"study", "experiment", "search_space"})
_TRIAL_OUTCOMES = (
    "completed",
    "pruned",
    "nonfinite_pruned",
    "oom_pruned",
    "recoverable_failed",
    "failed",
    "interrupted",
)


class RecoverableTrialError(RuntimeError):
    """
    Represent an explicitly recoverable trial-local execution failure.

    Only this project exception is supplied to ``study.optimize(catch=...)``;
    programming, identity, storage, tracking, and arbitrary runtime errors remain
    fatal to the invocation.
    """


class NonFiniteTrialError(FloatingPointError):
    """
    Represent a non-finite objective produced after trial execution starts.

    The trial lifecycle classifies this separately from generic failure and
    converts it to pruning only after publishing truthful local terminal state.
    """


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
        Parsed search-space parameters.

    Notes
    -----
    The dataclass is frozen, but nested mappings are treated as owned resolved
    values and are defensively copied by runtime-override and trial preparation
    boundaries before mutation.

    """

    path: Path
    study: dict[str, Any]
    base_experiment: dict[str, Any]
    base_config: dict[str, Any]
    search_space: tuple[search_space.SearchSpaceParameter, ...]


@dataclass(frozen=True)
class _OptunaStudyPaths:
    """Pure resolved study paths shared by dry-run planning and execution."""

    study_dir: Path
    trial_root: Path
    storage: str
    local_storage_path: Path | None


@dataclass
class OptunaEpochReporter:
    """Report the exact held-out objective only at genuine completed-epoch events."""

    trial: TrialProtocol
    objective_id: str
    direction: str
    evaluation_interval: int = 1
    target_epoch: int | None = None
    pruner_config: Mapping[str, Any] | None = None
    best_value: float | None = None
    best_epoch: int | None = None
    last_reported_epoch: int | None = None
    last_reported_objective: float | None = None
    last_global_step: int | None = None
    last_metrics: dict[str, float] | None = None
    last_pruning_eligible: bool | None = None
    last_pruning_decision: str | None = None
    last_report_duration_seconds: float | None = None
    last_study_best_value: float | None = None

    def _expected_epoch(self) -> int:
        """Return the next interval-or-terminal completed epoch."""
        interval = _require_exact_int(
            self.evaluation_interval,
            label="Optuna reporter evaluation_interval",
            minimum=1,
        )
        if self.target_epoch is not None:
            target = _require_exact_int(
                self.target_epoch,
                label="Optuna reporter target_epoch",
                minimum=1,
            )
            if self.last_reported_epoch is None:
                return min(interval, target)
            return min(self.last_reported_epoch + interval, target)
        if self.last_reported_epoch is None:
            return interval
        return self.last_reported_epoch + interval

    def _pruning_eligible(self, epoch: int) -> bool | None:
        """Return exact MedianPruner eligibility when prior study state is observable."""
        pruner = dict(self.pruner_config or {})
        if not pruner or pruner.get("kind") == "none":
            return False
        warmup = int(pruner.get("n_warmup_steps", 0))
        if epoch < warmup:
            return False
        startup = int(pruner.get("n_startup_trials", 0))
        if startup == 0:
            return True
        study = getattr(self.trial, "study", None)
        get_trials = getattr(study, "get_trials", None)
        if not callable(get_trials):
            return None
        try:
            trials = cast("Sequence[Any]", get_trials(deepcopy=False))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        completed = sum(str(getattr(getattr(item, "state", None), "name", "")).lower() == "complete" for item in trials)
        return completed >= startup

    def _study_best_value(self) -> float | None:
        """Return the current completed-study best without inventing a value."""
        study = cast("Any", getattr(self.trial, "study", None))
        try:
            value = float(study.best_value)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Publish one actual ID evaluation at its completed epoch and ask the pruner once."""
        report_started = time.perf_counter()
        if type(epoch) is not int or epoch <= 0:
            msg = f"Completed Optuna epoch must be a positive integer, got {epoch!r}."
            raise TypeError(msg)
        expected_epoch = self._expected_epoch()
        if epoch != expected_epoch:
            msg = f"Optuna reports must follow interval-or-terminal completed epochs; expected {expected_epoch}, got {epoch}."
            raise ValueError(msg)
        metric_key = f"id/{self.objective_id}"
        if metric_key not in metrics:
            msg = f"Held-out Optuna objective {metric_key!r} is missing at completed epoch {epoch}."
            raise KeyError(msg)
        value = float(metrics[metric_key])
        if not math.isfinite(value):
            self.trial.set_user_attr("nonfinite_epoch", epoch)
            msg = f"Non-finite Optuna objective {self.objective_id} at epoch {epoch}: {value}"
            raise NonFiniteTrialError(msg)

        if self.best_value is None or _is_better(value, self.best_value, self.direction):
            self.best_value = value
            self.best_epoch = epoch

        raw_global_step = metrics.get("global_step")
        if raw_global_step is not None and math.isfinite(float(raw_global_step)):
            self.last_global_step = int(raw_global_step)
        self.trial.report(value, step=epoch)
        self.trial.set_user_attr("last_reported_epoch", epoch)
        self.trial.set_user_attr("last_reported_objective", value)
        if self.last_global_step is not None:
            self.trial.set_user_attr("last_global_step", self.last_global_step)
        self.last_reported_epoch = epoch
        self.last_reported_objective = value
        self.last_metrics = dict(metrics)
        self.last_pruning_eligible = self._pruning_eligible(epoch)
        self.last_study_best_value = self._study_best_value()
        should_prune = bool(self.trial.should_prune())
        self.last_pruning_decision = "prune" if should_prune else "continue"
        self.last_report_duration_seconds = time.perf_counter() - report_started

        if should_prune:
            msg = f"Pruned by Optuna at completed epoch {epoch}"
            raise _trial_pruned_error()(msg)


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
    """
    Validate and isolate one exact semantic sampler mapping.

    Only ``random`` and ``tpe`` are accepted; multivariate mode belongs only to
    TPE and must be an exact boolean. String shorthand, unknown keys, and scalar
    coercion are deliberately rejected.
    """
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
    """
    Validate and isolate one exact semantic pruner mapping.

    Median-pruner cadence receives explicit defaults and exact non-negative or
    positive integers. The ``none`` pruner rejects tuning-only keys rather than
    silently ignoring them.
    """
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
    if kind == "median":
        pruner.setdefault("n_startup_trials", 5)
        pruner.setdefault("n_warmup_steps", 25)
        pruner.setdefault("interval_steps", 5)
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


def _validate_study_settings(study: Mapping[str, Any]) -> None:
    """
    Revalidate the complete normalized study mapping without scalar coercion.

    Name, objective, direction, seed, additional-trial count, sampler, pruner,
    and optional storage are checked with an exact closed schema. No file,
    database, Optuna import, or runtime allocation occurs.
    """
    allowed = {
        "name",
        "role",
        "objective",
        "direction",
        "seed",
        "n_trials",
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
    role = _require_nonempty_string(study.get("role"), label="study.role")
    if role not in _STUDY_ROLES:
        msg = f"study.role must be one of {sorted(_STUDY_ROLES)}, got {role!r}."
        raise ValueError(msg)
    _require_nonempty_string(study.get("objective"), label="study.objective")
    direction = _require_nonempty_string(study.get("direction"), label="study.direction")
    if direction not in {"minimize", "maximize"}:
        msg = f"study.direction must be minimize or maximize, got {direction!r}."
        raise ValueError(msg)
    _require_exact_int(study.get("seed"), label="study.seed")
    _require_exact_int(study.get("n_trials"), label="study.n_trials", minimum=1)
    _normalise_sampler_config(study.get("sampler"))
    pruner = _normalise_pruner_config(study.get("pruner"))
    if pruner["kind"] == "median" and int(pruner["n_startup_trials"]) >= int(study["n_trials"]):
        msg = "study.pruner.n_startup_trials must be smaller than study.n_trials."
        raise ValueError(msg)
    _sampler_seed(study)
    if "storage" in study:
        storage = study["storage"]
        if storage is not None:
            _require_nonempty_string(storage, label="study.storage")


def _normalise_study(raw_study: Mapping[str, Any], base_config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """
    Normalize study policy and derive objective identity from the base config.

    Defaults are applied to an isolated copy, direction cannot be supplied
    independently from the resolved experiment objective, and all names and
    scalar types are validated before an ``OptunaStudyConfig`` is constructed.
    """
    allowed = {"name", "role", "seed", "n_trials", "sampler", "pruner", "storage"}
    unknown = sorted(set(raw_study).difference(allowed))
    if unknown:
        msg = f"study contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ValueError(msg)

    study = dict(copy.deepcopy(raw_study))
    study.setdefault("name", source_path.stem)
    study.setdefault("role", "production")
    study.setdefault("seed", base_config.get("run", {}).get("seed", 9))
    study.setdefault("n_trials", 30)
    study.setdefault(
        "pruner",
        {"kind": "median", "n_startup_trials": 5, "n_warmup_steps": 25, "interval_steps": 5},
    )
    study.setdefault("sampler", {"kind": "tpe"})

    objective_config = _as_mapping(base_config["evaluation"]["objective"], label="evaluation.objective")
    study["objective"] = _require_nonempty_string(objective_config["id"], label="evaluation.objective.id")
    study["direction"] = _require_nonempty_string(
        objective_config["direction"],
        label="evaluation.objective.direction",
    )
    study["name"] = common.paths.validate_logical_name(
        _require_nonempty_string(study["name"], label="study.name"),
        label="study.name",
    )
    study["role"] = _require_nonempty_string(study["role"], label="study.role")
    if study["role"] not in _STUDY_ROLES:
        msg = f"study.role must be one of {sorted(_STUDY_ROLES)}, got {study['role']!r}."
        raise ValueError(msg)
    study["seed"] = _require_exact_int(study["seed"], label="study.seed")
    study["n_trials"] = _require_exact_int(study["n_trials"], label="study.n_trials", minimum=1)
    study["sampler"] = _normalise_sampler_config(study["sampler"])
    study["pruner"] = _normalise_pruner_config(study["pruner"])
    if "storage" in study and study["storage"] is not None:
        study["storage"] = _require_nonempty_string(study["storage"], label="study.storage")
    _validate_study_settings(study)
    return study


def load_optuna_study_config(path: Path | str) -> OptunaStudyConfig:
    """
    Load and semantically validate one complete Optuna study recipe.

    The inline experiment is deep-copied, receives its wrapper-owned study
    tracking identity, and is resolved first. Study objective and direction are
    then derived from that resolved config before exact parameter schemas,
    approved paths, supported choices, and base-value containment are validated.

    Parameters
    ----------
    path : pathlib.Path | str
        Optuna YAML containing only ``study``, ``experiment``, and
        ``search_space`` top-level blocks.

    Returns
    -------
    OptunaStudyConfig
        Source path, normalized study policy, untouched raw experiment copy,
        resolved base config, and parsed search parameters.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not identify a readable YAML file.
    KeyError
        If ``experiment`` or ``search_space`` is absent, or a required nested
        semantic field is missing.
    TypeError
        If a mapping, scalar, or search parameter has the wrong exact type.
    ValueError
        If keys, study policy, resolved experiment semantics, search schemas,
        approved paths, supported choices, or base-value containment are invalid.

    Notes
    -----
    Loading is side-effect free: it does not import Optuna, create output paths,
    open study storage, allocate trial directories, or mutate the caller's YAML
    mappings. Every failure occurs before study or trial lifecycle state exists.

    """
    source_path = Path(path)
    raw = experiments.config.loader.load_yaml(source_path)
    raw_mapping = _as_mapping(raw, label="Optuna YAML")
    allowed_root = OPTUNA_ROOT_KEYS
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
    raw_run = _as_mapping(base_experiment.get("run"), label="experiment.run")
    if "seed" not in raw_run:
        msg = "experiment.run.seed must be explicit for an Optuna study."
        raise ValueError(msg)
    raw_study = _as_mapping(raw_mapping.get("study", {}), label="study")
    if "seed" not in raw_study:
        msg = "study.seed must be explicit for deterministic sampler reconstruction."
        raise ValueError(msg)
    study_name = common.paths.validate_logical_name(
        _require_nonempty_string(raw_study.get("name", source_path.stem), label="study.name"),
        label="study.name",
    )

    effective_experiment = copy.deepcopy(base_experiment)
    tracking = dict(_as_mapping(effective_experiment.get("tracking"), label="experiment.tracking"))
    wandb_settings = dict(_as_mapping(tracking.get("wandb"), label="experiment.tracking.wandb"))
    workflow = wandb_settings.get("workflow")
    if workflow != "optuna_trial":
        msg = "experiment.tracking.wandb.workflow must be 'optuna_trial' for an Optuna study."
        raise ValueError(msg)
    if "study" in wandb_settings:
        msg = "experiment.tracking.wandb.study is wrapper-owned and must not be supplied."
        raise ValueError(msg)
    wandb_settings["workflow"] = "optuna_trial"
    wandb_settings["study"] = study_name
    tracking["wandb"] = wandb_settings
    effective_experiment["tracking"] = tracking

    base_config = experiments.config.loader.resolve_config(effective_experiment)
    experiments.config.loader.validate_task_directory_identity(
        source_path,
        raw_task=base_experiment.get("task"),
        resolved_task=base_config.get("task"),
    )
    study = _normalise_study(raw_study, base_config, source_path)
    base_config = experiments.config.loader.validate_resolved_config(base_config)
    search_parameters = search_space.parse_search_space(raw_mapping["search_space"])
    search_space.validate_search_space_paths(base_config, search_parameters)
    _validate_search_space_choices(base_config, search_parameters)

    return OptunaStudyConfig(
        path=source_path,
        study=study,
        base_experiment=base_experiment,
        base_config=base_config,
        search_space=search_parameters,
    )


def _validate_reporting_contract(base_config: Mapping[str, Any]) -> int:
    """Require one shared positive ID/OOD/physics interval for Optuna trials."""
    training = _as_mapping(base_config.get("training"), label="experiment.training")
    wandb = _as_mapping(_as_mapping(base_config.get("tracking"), label="experiment.tracking").get("wandb"), label="tracking.wandb")
    monitor = _as_mapping(wandb.get("monitor"), label="tracking.wandb.monitor")
    intervals = {
        "ID": _require_exact_int(training.get("evaluation_interval"), label="training.evaluation_interval", minimum=1),
        "OOD": _require_exact_int(training.get("ood_evaluation_interval"), label="training.ood_evaluation_interval", minimum=1),
        "physics": _require_exact_int(monitor.get("interval"), label="tracking.wandb.monitor.interval", minimum=1),
    }
    if len(set(intervals.values())) != 1:
        msg = f"Optuna ID/OOD/physics intervals must share one completed-epoch cadence, got {intervals}."
        raise ValueError(msg)
    return intervals["ID"]


def _validate_search_space_choices(
    base_config: dict[str, Any],
    parameters: Sequence[search_space.SearchSpaceParameter],
) -> None:
    """
    Revalidate every declared endpoint, choice, or fixed value as a full config.

    Candidate values are applied one path at a time to the resolved base config.
    Any model, loss, device-policy, or task-contract violation fails before
    Optuna storage or a trial run is created.
    """
    for parameter in parameters:
        if parameter.kind == "categorical":
            candidates = parameter.values
        elif parameter.kind in {"float", "int"}:
            candidates = (parameter.low, parameter.high)
        else:
            candidates = (parameter.value,)
        for candidate in candidates:
            candidate_config = search_space.apply_trial_overrides(base_config, {parameter.path: candidate})
            candidate_config["run"].pop("name", None)
            candidate_config["run"]["name"] = experiments.config.loader.generate_run_name(candidate_config)
            try:
                experiments.config.loader.validate_resolved_config(candidate_config)
            except (KeyError, TypeError, ValueError) as error:
                msg = f"Search-space candidate {candidate!r} for {parameter.path!r} violates the resolved experiment contract."
                raise ValueError(msg) from error


def _validate_study_contract(config: OptunaStudyConfig) -> tuple[OptunaStudyConfig, dict[str, Any]]:
    """
    Revalidate a study and return its exact full objective before side effects.

    Study policy, base config, per-epoch reporting, search-path admission, and
    every declared candidate must agree. A defensive replacement containing the
    revalidated base config is returned with the canonical objective.
    """
    _validate_study_settings(config.study)
    base_config = experiments.config.loader.validate_resolved_config(config.base_config)
    reporting_interval = _validate_reporting_contract(base_config)
    pruner = _normalise_pruner_config(config.study["pruner"])
    if pruner["kind"] == "median":
        if int(pruner["interval_steps"]) != reporting_interval:
            msg = "study.pruner.interval_steps must equal the genuine ID evaluation interval."
            raise ValueError(msg)
        warmup = int(pruner["n_warmup_steps"])
        if warmup < reporting_interval or warmup % reporting_interval != 0:
            msg = "study.pruner.n_warmup_steps must be a positive whole number of ID evaluation intervals."
            raise ValueError(msg)
    objective = experiments.config.loader.get_resolved_objective(base_config)
    if config.study.get("objective") != objective["id"]:
        msg = "Resolved study objective id does not match its experiment objective."
        raise ValueError(msg)
    if config.study.get("direction") != objective["direction"]:
        msg = "Resolved study direction does not match its experiment objective."
        raise ValueError(msg)
    search_space.validate_search_space_paths(base_config, config.search_space)
    _validate_search_space_choices(base_config, config.search_space)
    return replace(config, base_config=base_config), objective


def _scientific_base_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Project the resolved base config into invocation-independent science identity.

    Paths, W&B policy, device, generated run name, prefix, and suffix are removed.
    Task, data, model, loss, objective, optimizer, scheduler, duration, seed, and
    deterministic semantics remain signature-bearing.
    """
    scientific = copy.deepcopy(dict(base_config))
    scientific.pop("paths", None)
    scientific.pop("tracking", None)
    run = dict(_as_mapping(scientific.get("run"), label="base_config.run"))
    for key in ("device", "name", "prefix", "suffix"):
        run.pop(key, None)
    scientific["run"] = run
    return scientific


def build_study_signature(config: OptunaStudyConfig) -> dict[str, Any]:
    """
    Build the canonical invocation-independent scientific study signature.

    Parameters
    ----------
    config : OptunaStudyConfig
        Loaded study with resolved base experiment and parsed search parameters.

    Returns
    -------
    dict[str, Any]
        Schema version, SHA-256 digest, and canonical payload covering task,
        scientific config, search space, objective, sampler/pruner, reporting, and
        trial lifecycle policy.

    Raises
    ------
    TypeError
        If an in-memory study component has an invalid exact type.
    ValueError
        If study, objective, reporting cadence, search policy, or resolved base
        config has drifted outside the admitted semantic contract.

    Notes
    -----
    Device/output location and W&B settings are operational and excluded; every
    model-selection or training semantic participates in the digest. Construction
    is deterministic and does not mutate ``config`` or touch study storage.

    """
    validated, objective = _validate_study_contract(config)
    sampler = _normalise_sampler_config(validated.study["sampler"])
    pruner = _normalise_pruner_config(validated.study["pruner"])
    sampler_seed = _sampler_seed(validated.study)
    reporting_interval = _validate_reporting_contract(validated.base_config)
    payload = {
        "schema_version": STUDY_SIGNATURE_SCHEMA_VERSION,
        "task": {
            "id": validated.base_config["task"],
            "schema_version": validated.base_config["task_contract"]["schema_version"],
            "contract_digest": validated.base_config["task_contract"]["digest"],
        },
        "scientific_base_config": _scientific_base_config(validated.base_config),
        "search_space": sorted(
            search_space.search_space_summary(validated.search_space),
            key=lambda item: (str(item["path"]), str(item["name"])),
        ),
        "study_role": validated.study["role"],
        "objective": objective,
        "direction": objective["direction"],
        "sampler": {**sampler, "seed": sampler_seed},
        "pruner": pruner,
        "reporting": {
            "metric_source": "held_out_evaluation",
            "evaluation_interval": reporting_interval,
            "event_model": "completed_epoch_interval_or_terminal",
            "step": "actual_completed_epoch",
            "pruning_subset": "id_only",
        },
        "trial_lifecycle": {
            "schema_version": TRIAL_LIFECYCLE_SCHEMA_VERSION,
            "run_summary_schema_version": experiments.run.RUN_SUMMARY_SCHEMA_VERSION,
            "outcomes": list(_TRIAL_OUTCOMES),
            "resume_policy": "new_trials_only",
            "trial_count_policy": "additional_fresh_trials_per_invocation",
            "training_seed_policy": "fixed_configured_run_seed_across_trials",
            "trial_identity_policy": "native_zero_based_optuna_number",
        },
    }
    return {
        "schema_version": STUDY_SIGNATURE_SCHEMA_VERSION,
        "digest": common.serialization.canonical_json_sha256(payload),
        "payload": payload,
    }


def describe_optuna_study_config(config: OptunaStudyConfig) -> dict[str, Any]:
    """
    Validate and describe a study without creating runtime state.

    Parameters
    ----------
    config : OptunaStudyConfig
        Loaded or programmatically modified study configuration.

    Returns
    -------
    dict[str, Any]
        Serializable source, study, paths, dataset/task/model identities, device
        policy, search space, objective, and complete semantic signature.

    Raises
    ------
    TypeError
        If an in-memory study or compact dataset-metadata component is invalid.
    ValueError
        If study policy, resolved science, dataset identity, reporting, or search
        semantics have drifted since loading.

    Notes
    -----
    Description reads compact model-training metadata but does not load dataset
    tensors, resolve hardware, create paths, open storage, allocate trials,
    initialize W&B, or import the Optuna SDK.

    """
    validated, objective = _validate_study_contract(config)
    signature = build_study_signature(validated)
    study_paths = _resolve_study_paths(validated)
    return {
        "path": str(validated.path),
        "study": validated.study,
        "base_run_name": validated.base_config["run"]["name"],
        "device_policy": validated.base_config["run"]["device"],
        "storage": study_paths.storage,
        "study_dir": str(study_paths.study_dir),
        "trial_root": str(study_paths.trial_root),
        "task": validated.base_config["task"],
        "model_kind": validated.base_config["model"]["kind"],
        "dataset_roles": _configured_dataset_identities(validated.base_config),
        "objective": objective,
        "search_space": search_space.search_space_summary(validated.search_space),
        "semantic_signature": signature,
    }


def _study_dir(config: OptunaStudyConfig) -> Path:
    """Return the task-owned study directory under the derived output root."""
    return common.paths.resolve_study_dir(
        str(config.base_config["task"]),
        str(config.study["name"]),
        output_root=Path(config.base_config["paths"]["output_root"]),
    )


def _resolve_study_paths(config: OptunaStudyConfig) -> _OptunaStudyPaths:
    """Resolve storage, study, and trial roots without creating any path."""
    study_name = common.paths.validate_logical_name(config.study["name"], label="study.name")
    study_dir = _study_dir(config)
    configured_storage = config.study.get("storage")
    local_storage_path = None if configured_storage is not None else study_dir / f"{study_name}.db"
    storage = str(configured_storage) if configured_storage is not None else f"sqlite:///{local_storage_path}"
    return _OptunaStudyPaths(study_dir, study_dir / "trials", storage, local_storage_path)


def _configured_dataset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the exact configured ID and OOD dataset roles."""
    data = _as_mapping(config.get("data"), label="experiment.data")
    train_dataset = _require_nonempty_string(data.get("train_dataset"), label="data.train_dataset")
    raw_ood = data.get("ood_datasets")
    if not isinstance(raw_ood, list) or len(raw_ood) != 1:
        msg = "data.ood_datasets must contain exactly one configured dataset id."
        raise ValueError(msg)
    ood_dataset = _require_nonempty_string(raw_ood[0], label="data.ood_datasets[0]")

    task = experiments.config.loader.validate_resolved_task_contract(config)
    paths = _as_mapping(config.get("paths"), label="experiment.paths")

    def summarize(dataset_id: str) -> dict[str, Any]:
        summary = datasets.metadata.load_dataset_metadata_summary(
            dataset_id,
            task=task,
            dataset_root=Path(paths["dataset_root"]),
            metadata_root=Path(paths["training_meta_root"]),
        )
        if not summary.dataset_exists:
            msg = f"Configured training dataset is not a regular file: {summary.dataset_path}"
            raise FileNotFoundError(msg)
        return {
            "dataset_id": summary.dataset_id,
            "dataset_path": str(summary.dataset_path),
            "metadata_dir": str(summary.metadata_directory),
            "validation": "metadata_package_and_artifact_stat",
            "task": summary.task_id,
            "task_contract_digest": summary.task_contract_digest,
            "fingerprint": summary.fingerprint,
            "sample_count": summary.sample_count,
        }

    return {
        "id": summarize(train_dataset),
        "ood": [summarize(ood_dataset)],
    }


def _build_pruner(study: Mapping[str, Any]) -> Any:
    """
    Build the exact Optuna pruner from validated semantic policy.

    Optuna is imported lazily only at construction. ``none`` maps to ``NopPruner``
    and ``median`` forwards the already normalized startup, warmup, and interval values.
    """
    pruner_cfg = _normalise_pruner_config(study.get("pruner"))
    optuna = _optuna_module()
    pruner_type = pruner_cfg["kind"]

    if pruner_type == "none":
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 25),
            interval_steps=pruner_cfg.get("interval_steps", 5),
        )
    msg = f"Unsupported Optuna pruner: {pruner_type!r}"
    raise ValueError(msg)


def _sampler_seed(study: Mapping[str, Any]) -> int:
    """Return the explicit persisted Optuna sampler seed without derivation."""
    seed = _require_exact_int(study.get("seed"), label="study.seed", minimum=0)
    if seed >= 2**32:
        msg = f"study.seed must fit Optuna/NumPy's 32-bit sampler domain, got {seed}."
        raise ValueError(msg)
    return seed


def _build_sampler(study: Mapping[str, Any]) -> Any:
    """
    Build a deterministic Optuna sampler from validated semantic policy.

    Both random and TPE receive the explicit persisted 32-bit study seed; only
    TPE accepts the normalized multivariate flag. Optuna remains lazily imported.
    """
    sampler_cfg = _normalise_sampler_config(study.get("sampler"))
    sampler_type = sampler_cfg["kind"]
    seed = _sampler_seed(study)
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


def _prepare_trial_config(study_config: OptunaStudyConfig, trial: TrialProtocol) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one trial while preserving the configured fixed training seed."""
    trial_number = _require_exact_int(trial.number, label="Optuna trial.number", minimum=0)
    overrides = search_space.suggest_trial_overrides(trial, study_config.search_space)
    config = search_space.apply_trial_overrides(study_config.base_config, overrides)
    training_seed = _require_exact_int(config["run"]["seed"], label="experiment.run.seed", minimum=0)
    sampler_seed = _sampler_seed(study_config.study)
    wandb_settings = config["tracking"]["wandb"]
    wandb_settings["workflow"] = "optuna_trial"
    wandb_settings["study"] = str(study_config.study["name"])
    wandb_settings.update(experiments.config.loader.derive_wandb_organization(config))
    config["run"]["suffix"] = f"optuna_trial_{trial_number:03d}"
    config["run"].pop("name", None)
    config["run"]["name"] = experiments.config.loader.generate_run_name(config)
    config = experiments.config.loader.validate_resolved_config(config)

    base_objective = experiments.config.loader.get_resolved_objective(study_config.base_config)
    trial_objective = experiments.config.loader.get_resolved_objective(config)
    if trial_objective != base_objective:
        msg = "Sampled trial objective does not match the resolved study objective."
        raise ValueError(msg)

    analysis_parameters = {
        parameter.name: copy.deepcopy(overrides[parameter.path]) for parameter in study_config.search_space if parameter.kind != "fixed"
    }
    context = {
        "study_name": str(study_config.study["name"]),
        "study_role": str(study_config.study["role"]),
        "trial_number": trial_number,
        "training_seed": training_seed,
        "sampler_seed": sampler_seed,
        "overrides": overrides,
        "analysis_parameters": analysis_parameters,
        "search_signature": build_study_signature(study_config)["digest"],
    }

    trial.set_user_attr("run_name", config["run"]["name"])
    trial.set_user_attr("run_seed", training_seed)
    trial.set_user_attr("sampler_seed", sampler_seed)
    trial.set_user_attr("study_role", context["study_role"])
    trial.set_user_attr("overrides", overrides)
    return config, context


def _trial_run_dir(config: Mapping[str, Any], context: Mapping[str, Any]) -> Path:
    """Return a study- and trial-qualified output directory."""
    return common.paths.resolve_optuna_trial_dir(
        str(config["task"]),
        str(context["study_name"]),
        str(config["run"]["name"]),
        output_root=Path(config["paths"]["output_root"]),
    )


def _finite_objective_value(result: Mapping[str, Any], objective: Mapping[str, Any]) -> float:
    """
    Admit the training result's objective identity and finite selected value.

    A mismatched full objective is a configuration/lifecycle error. A matching
    but NaN or infinite best metric becomes ``NonFiniteTrialError`` for distinct
    trial-outcome classification.
    """
    if result.get("objective") != objective:
        msg = "Training result objective does not match the resolved trial objective."
        raise ValueError(msg)
    objective_value = float(result["best_metric"])
    if not math.isfinite(objective_value):
        msg = "Training completed without a finite held-out objective."
        raise NonFiniteTrialError(msg)
    return objective_value


def _runtime_error_is_allocation_oom(error: RuntimeError) -> bool:
    """
    Classify only established CUDA or CPU allocator OOM ``RuntimeError`` text.

    Matching is intentionally narrow so arbitrary runtime, driver, shape, and
    programming errors remain fatal rather than being mislabeled as prunable
    resource exhaustion.
    """
    message = str(error).lower()
    return (
        message.startswith("cuda out of memory")
        or "cuda error: out of memory" in message
        or "defaultcpuallocator: can't allocate memory" in message
        or "defaultcpuallocator: not enough memory" in message
        or "mmap failed: cannot allocate memory" in message
    )


def _failure_epoch(error: BaseException, reporter: OptunaEpochReporter) -> int | None:
    """
    Recover an explicit failure epoch without inventing progress.

    A maintained ``" at epoch N:"`` message marker takes precedence; otherwise
    only the reporter's last successfully published completed epoch is returned.
    """
    marker = " at epoch "
    message = str(error)
    if marker in message:
        suffix = message.split(marker, 1)[1]
        digits = suffix.split(":", 1)[0].strip()
        if digits.isdigit():
            return int(digits)
    return reporter.last_reported_epoch


def _failure_context(
    *,
    error: BaseException,
    device_resolution: learning.device.DeviceResolution,
    reporter: OptunaEpochReporter,
) -> dict[str, Any]:
    """
    Build bounded trial-failure context without initializing a CUDA context.

    Requested/concrete device, explicit epoch, latest Optuna report, and global
    step are always safe. CUDA allocation counters are queried only when the
    resolved device is CUDA and Torch already initialized that runtime.
    """
    context: dict[str, Any] = {
        "error_type": type(error).__name__,
        "requested_device_policy": device_resolution.requested_policy,
        "resolved_device": str(device_resolution.device),
        "epoch": _failure_epoch(error, reporter),
        "last_reported_epoch": reporter.last_reported_epoch,
        "last_reported_objective": reporter.last_reported_objective,
        "last_global_step": reporter.last_global_step,
    }
    device = device_resolution.device
    if device.type == "cuda" and torch.cuda.is_initialized():
        with suppress(RuntimeError):
            context["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            context["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
            context["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            context["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return context


def _checkpoint_progress(
    run_dir: Path,
    reporter: OptunaEpochReporter,
) -> dict[str, Any]:
    """
    Bind every reported Optuna epoch to already durable checkpoint metadata.

    A reported epoch requires ``last_checkpoint.pt`` and its digest. A retained
    finite best epoch additionally requires ``best_checkpoint.pt``. Missing files
    are lifecycle violations rather than guessed or backfilled progress.
    """
    if reporter.last_reported_epoch is None:
        return {}
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    if not last_path.is_file():
        msg = "A reported Optuna epoch must already have a durable last checkpoint."
        raise experiments.run.RunLifecycleError(msg)
    progress = {
        "completed_epoch": reporter.last_reported_epoch,
        "global_step": reporter.last_global_step,
        "last_checkpoint": last_path.name,
        "last_checkpoint_sha256": common.serialization.file_sha256(last_path),
    }
    if reporter.best_epoch is not None:
        best_path = common.paths.resolve_best_checkpoint_file(run_dir)
        if not best_path.is_file():
            msg = "A finite best Optuna objective must already have a durable best checkpoint."
            raise experiments.run.RunLifecycleError(msg)
        progress.update(
            {
                "best_checkpoint": best_path.name,
                "best_checkpoint_sha256": common.serialization.file_sha256(best_path),
            }
        )
    return progress


def _write_summary(
    *,
    run_dir: Path,
    config: dict[str, Any],
    context: Mapping[str, Any],
    status: str,
    start_time: datetime,
    result: Mapping[str, Any] | None = None,
    reporter: OptunaEpochReporter | None = None,
    error: BaseException | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    amp_enabled: bool = False,
    failure_context: Mapping[str, Any] | None = None,
) -> None:
    """
    Atomically publish one terminal local trial outcome and provenance record.

    All statuses share objective, sampled parameters, reporting progress, timing,
    and bounded failure context. Completed trials additionally require checkpoint
    identity and all authoritative artifact digests; non-completed trials publish
    only checkpoints proven durable by the reporter. The run state machine owns
    final atomic summary replacement.
    """
    if status not in _TRIAL_OUTCOMES:
        msg = f"Unsupported Optuna trial outcome {status!r}."
        raise ValueError(msg)
    end_time = datetime.now(UTC)
    result = result or {}
    objective = experiments.config.loader.get_resolved_objective(config)
    summary: dict[str, Any] = {
        "task": config["task"],
        "model_kind": config["model"]["kind"],
        "study_name": context["study_name"],
        "trial_number": context["trial_number"],
        "sampled_parameters": context["overrides"],
        "search_signature": context["search_signature"],
        "objective": objective,
        "best_epoch": result.get("best_epoch", reporter.best_epoch if reporter else None),
        "best_metric": result.get("best_metric", reporter.best_value if reporter else None),
        "checkpoint_path": result.get("checkpoint_path"),
        "last_reported_epoch": reporter.last_reported_epoch if reporter else None,
        "last_reported_objective": reporter.last_reported_objective if reporter else None,
        "pruning_epoch": reporter.last_reported_epoch if status == "pruned" and reporter else None,
        "status": status,
        "error": str(error) if error is not None else None,
        "failure_context": dict(failure_context) if failure_context is not None else None,
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
                "selected_epoch": result.get("selected_epoch"),
                "selected_metrics": copy.deepcopy(result.get("selected_metrics", {})),
                "terminal_epoch": result.get("terminal_epoch"),
                "terminal_metrics": copy.deepcopy(result.get("terminal_metrics", {})),
                "duration_contract": copy.deepcopy(experiments.run.RUN_DURATION_CONTRACT),
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
    elif reporter is not None:
        summary.update(_checkpoint_progress(run_dir, reporter))
    experiments.run.transition_run_status(run_dir, status, updates=summary)


def _validate_completed_reporting(
    reporter: OptunaEpochReporter,
    result: Mapping[str, Any],
) -> None:
    """
    Require Optuna's latest durable report to equal training's terminal epoch.

    This closes the lifecycle only when actual completed-epoch pruning evidence
    is continuous through the same terminal checkpoint represented by the result.
    """
    if reporter.last_reported_epoch != result.get("completed_epoch"):
        msg = "Completed training did not report its actual terminal epoch to Optuna."
        raise RuntimeError(msg)


def run_trial(  # noqa: C901, PLR0912, PLR0915
    study_config: OptunaStudyConfig,
    trial: TrialProtocol,
) -> float:
    """
    Execute one exclusively allocated trial through the normal run lifecycle.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Validated study contract and runtime overrides.
    trial : TrialProtocol
        Optuna-compatible trial supplying suggestions, reports, pruning, and attrs.

    Returns
    -------
    float
        Finite terminal held-out value of the resolved primary objective.

    Raises
    ------
    optuna.TrialPruned
        For explicit pruning, non-finite science, or narrowly classified OOM.
    RecoverableTrialError
        For an explicitly recoverable trial-local failure.
    BaseException
        Interrupts and programming/identity errors propagate after local terminal
        status publication and guarded device-specific cleanup.

    Notes
    -----
    The trial gets a fresh study-qualified run leaf and never resumes partial
    training. Actual completed epochs are reported before optional W&B mirroring.

    """
    study_config, _ = _validate_study_contract(study_config)
    device_resolution = learning.device.resolve_device(
        study_config.base_config["run"]["device"],
        path="run.device",
    )
    learning.device.validate_mixed_precision_device(
        study_config.base_config["training"]["mixed_precision"],
        device_resolution,
    )
    config, context = _prepare_trial_config(study_config, trial)
    if config["run"]["device"] != device_resolution.requested_policy:
        msg = "Prepared trial changed the already resolved runtime device policy."
        raise ValueError(msg)
    experiments.run.validate_deterministic_model_device_policy(config, device_resolution)
    device = device_resolution.device
    objective = experiments.config.loader.get_resolved_objective(config)
    reporter = OptunaEpochReporter(
        trial=trial,
        objective_id=str(objective["id"]),
        direction=str(objective["direction"]),
        evaluation_interval=int(config["training"]["evaluation_interval"]),
        target_epoch=int(config["training"]["epochs"]),
        pruner_config=study_config.study["pruner"],
    )
    trial_pruned = _trial_pruned_error()
    requested_run_dir = _trial_run_dir(config, context)
    summary_extra = dict(context)
    run_dir = experiments.run.prepare_fresh_run(
        config,
        run_dir=requested_run_dir,
        summary_extra=summary_extra,
    )
    console_reporter = experiments.console.ConsoleReporter(
        config=config,
        run_dir=run_dir,
        study_name=str(context["study_name"]),
        trial_number=int(context["trial_number"]),
    )
    start_time = datetime.now(UTC)
    experiments.console.optuna_trial_event(
        "started",
        study=str(context["study_name"]),
        study_role=str(context["study_role"]),
        trial=int(context["trial_number"]),
        run_name=str(config["run"]["name"]),
        sampled=context["analysis_parameters"],
        objective_id=str(objective["id"]),
        training_seed=int(context["training_seed"]),
        sampler_seed=int(context["sampler_seed"]),
        device=str(device),
        run_dir=run_dir,
        max_epochs=int(config["training"]["epochs"]),
    )

    checkpoint_identity: dict[str, Any] | None = None
    amp_enabled = False
    run_started = False
    runtime_session_id = uuid.uuid4().hex
    tracker: experiments.tracking.WandbSession | None = None
    tracking_status = "failed"
    tracking_result: Mapping[str, Any] | None = None
    tracking_error: BaseException | None = None
    dataloaders: Any = None
    data_processor: Any = None
    model: Any = None
    train_loss: Any = None
    eval_metrics: Any = None
    optimizer: Any = None
    scheduler: Any = None
    wandb_epoch_callback: Callable[[int, dict[str, float]], None] | None = None
    result: Mapping[str, Any] | None = None

    try:
        seed_plan = experiments.run.configure_reproducibility(config, device=device)
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

        experiments.run.seed_process(seed_plan["model_init"], device=device)
        model = learning.models.factory.build_model(config, device=device)
        train_loss = learning.losses.factory.build_training_loss(config, device=device)
        set_normalizers = getattr(train_loss, "set_normalizers", None)
        if callable(set_normalizers):
            set_normalizers(
                in_normalizer=data_processor.in_normalizer,
                out_normalizer=data_processor.out_normalizer,
            )
        data_processor.to(device)
        eval_metrics = learning.metrics.metrics.build_evaluation_metrics(config, device=device)
        optimizer = learning.training.optim.build_optimizer(model, config)
        scheduler = learning.training.optim.build_scheduler(optimizer, config)
        checkpoint_identity = learning.training.checkpoint.build_checkpoint_identity(
            config,
            dataloaders["split_indices"],
            persisted_config=config,
        )
        amp_enabled = bool(config["training"]["mixed_precision"])
        experiments.run.transition_run_status(
            run_dir,
            "running",
            updates={
                **summary_extra,
                "target_epochs": int(config["training"]["epochs"]),
                "seed_plan": seed_plan,
                "deterministic": bool(config["run"]["deterministic"]),
                "amp_enabled": amp_enabled,
                **experiments.run.runtime_session_updates(
                    run_dir,
                    device_resolution,
                    started_at=start_time,
                    session_id=runtime_session_id,
                    tracking_state=experiments.run.initial_tracking_state(config),
                ),
            },
        )
        run_started = True

        def state_updater(updates: Mapping[str, Any]) -> None:
            """Persist trial-observer facts in the current run runtime session."""
            experiments.run.update_runtime_session(run_dir, runtime_session_id, updates)

        monitor_membership = experiments.tracking.build_monitor_membership(
            config,
            dataloaders["split_indices"],
        )
        if monitor_membership is not None:
            state_updater({"monitor": monitor_membership})
        semantic_config: Mapping[str, Any] | None = None
        monitor_settings = config["tracking"]["wandb"]["monitor"]
        if config["tracking"]["wandb"]["mode"] != "disabled":
            semantic_config = experiments.tracking.build_semantic_config(
                config,
                split_indices=dataloaders["split_indices"],
                split_indices_sha256=common.serialization.file_sha256(common.paths.resolve_split_indices_path(run_dir)),
                normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
                checkpoint_identity=checkpoint_identity,
                model=model,
                device_metadata=device_resolution.as_dict(),
                duration_contract=experiments.run.RUN_DURATION_CONTRACT,
                tuning_context={
                    "study_name": context["study_name"],
                    "study_role": context["study_role"],
                    "trial_number": context["trial_number"],
                    "training_seed": context["training_seed"],
                    "sampler_seed": context["sampler_seed"],
                    "search_signature": context["search_signature"],
                    "sampled_parameters": copy.deepcopy(context["analysis_parameters"]),
                    "objective": copy.deepcopy(objective),
                },
            )
        console_reporter.startup(resolved_device=str(device))
        tracker = experiments.tracking.initialize_wandb(
            config,
            run_dir=run_dir,
            semantic_config=semantic_config,
            state_updater=state_updater,
        )
        wandb_epoch_callback = experiments.tracking.epoch_callback(tracker)

        def mirror_epoch_to_wandb(epoch: int, metrics: dict[str, float]) -> None:
            """Mirror trial telemetry only when an optional W&B session is active."""
            if wandb_epoch_callback is not None:
                wandb_epoch_callback(epoch, metrics)

        def optuna_tracking_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
            """Mirror only the exact reported ID value and trial-best scalar under Optuna/."""
            payload = dict(metrics)
            if reporter.last_reported_objective is not None:
                payload["optuna/objective"] = reporter.last_reported_objective
            if reporter.best_value is not None:
                payload["optuna/best_objective_so_far"] = reporter.best_value
            return payload

        def trial_epoch_callback(epoch: int, metrics: dict[str, float]) -> None:
            """Log every epoch but report/prune only after one genuine ID evaluation."""
            console_reporter.epoch(epoch, metrics)
            id_key = f"id/{objective['id']}"
            if id_key not in metrics:
                mirror_epoch_to_wandb(epoch, metrics)
                return
            try:
                reporter(epoch, metrics)
            except trial_pruned:
                experiments.console.optuna_trial_event(
                    "observed",
                    study=str(context["study_name"]),
                    trial=int(context["trial_number"]),
                    run_name=str(config["run"]["name"]),
                    sampled=context["analysis_parameters"],
                    objective_id=str(objective["id"]),
                    objective=reporter.last_reported_objective,
                    best_trial_objective=reporter.best_value,
                    best_study_objective=reporter.last_study_best_value,
                    step=reporter.last_reported_epoch,
                    pruning="prune",
                    pruning_eligible=reporter.last_pruning_eligible,
                    report_duration_seconds=reporter.last_report_duration_seconds,
                )
                raise
            experiments.console.optuna_trial_event(
                "observed",
                study=str(context["study_name"]),
                trial=int(context["trial_number"]),
                run_name=str(config["run"]["name"]),
                sampled=context["analysis_parameters"],
                objective_id=str(objective["id"]),
                objective=reporter.last_reported_objective,
                best_trial_objective=reporter.best_value,
                best_study_objective=reporter.last_study_best_value,
                step=reporter.last_reported_epoch,
                pruning="continue",
                pruning_eligible=reporter.last_pruning_eligible,
                report_duration_seconds=reporter.last_report_duration_seconds,
            )
            mirror_epoch_to_wandb(epoch, optuna_tracking_metrics(metrics))

        result = learning.training.loop.train_loop(
            config=config,
            device=device,
            model=model,
            optimizer=optimizer,
            train_loader=dataloaders["train"],
            eval_loader=dataloaders["eval"],
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            ood_loader=dataloaders["ood"],
            data_processor=data_processor,
            scheduler=scheduler,
            save_dir=run_dir,
            use_amp=config["training"].get("mixed_precision", False),
            epoch_end_callback=trial_epoch_callback,
            checkpoint_identity=checkpoint_identity,
        )
        selected = learning.training.loop.evaluate_selected_checkpoint(
            config=config,
            model=model,
            train_loss=train_loss,
            eval_loader=dataloaders["eval"],
            ood_loader=dataloaders["ood"],
            eval_metrics=eval_metrics,
            device=device,
            data_processor=data_processor,
            checkpoint_identity=checkpoint_identity,
            best_checkpoint_path=result["best_checkpoint_path"],
            scheduler_expected=scheduler is not None,
            amp_expected=amp_enabled,
            max_physics_cases=int(monitor_settings["max_cases"]),
        )
        result.update(selected)
        objective_value = _finite_objective_value(result, objective)
        _validate_completed_reporting(reporter, result)
        tracking_status = "completed"
        tracking_result = result
        console_reporter.final(result, total_wall_seconds=(datetime.now(UTC) - start_time).total_seconds())

    except (KeyboardInterrupt, SystemExit) as error:
        tracking_status = "interrupted"
        tracking_error = error
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="interrupted",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        raise
    except trial_pruned as error:
        tracking_error = error
        if not run_started:
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="failed",
                start_time=start_time,
                reporter=reporter,
                error=error,
                failure_context=_failure_context(
                    error=error,
                    device_resolution=device_resolution,
                    reporter=reporter,
                ),
            )
            msg = "Optuna requested pruning before the trial reached training."
            raise RuntimeError(msg) from error
        tracking_status = "pruned"
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="pruned",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        pruning_epoch = reporter.last_reported_epoch
        if wandb_epoch_callback is not None and reporter.last_metrics is not None and pruning_epoch is not None:
            wandb_epoch_callback(pruning_epoch, optuna_tracking_metrics(reporter.last_metrics))
        raise
    except FloatingPointError as error:
        tracking_error = error
        if not run_started:
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="failed",
                start_time=start_time,
                reporter=reporter,
                error=error,
                failure_context=_failure_context(
                    error=error,
                    device_resolution=device_resolution,
                    reporter=reporter,
                ),
            )
            raise
        tracking_status = "nonfinite_pruned"
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="nonfinite_pruned",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        message = f"Trial pruned after non-finite value: {error}"
        raise trial_pruned(message) from None
    except (torch.cuda.OutOfMemoryError, MemoryError) as error:
        tracking_error = error
        if not run_started:
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="failed",
                start_time=start_time,
                reporter=reporter,
                error=error,
                failure_context=_failure_context(
                    error=error,
                    device_resolution=device_resolution,
                    reporter=reporter,
                ),
            )
            raise
        tracking_status = "oom_pruned"
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="oom_pruned",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        message = "Trial pruned after allocator out-of-memory failure"
        raise trial_pruned(message) from None
    except RecoverableTrialError as error:
        tracking_error = error
        if not run_started:
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="failed",
                start_time=start_time,
                reporter=reporter,
                error=error,
                failure_context=_failure_context(
                    error=error,
                    device_resolution=device_resolution,
                    reporter=reporter,
                ),
            )
            msg = "A recoverable trial error occurred before training started."
            raise RuntimeError(msg) from error
        tracking_status = "recoverable_failed"
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="recoverable_failed",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        raise
    except experiments.tracking.TrackingError as error:
        tracking_error = error
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="failed",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        raise
    except RuntimeError as error:
        tracking_error = error
        if run_started and _runtime_error_is_allocation_oom(error):
            tracking_status = "oom_pruned"
            _write_summary(
                run_dir=run_dir,
                config=config,
                context=context,
                status="oom_pruned",
                start_time=start_time,
                reporter=reporter,
                error=error,
                failure_context=_failure_context(
                    error=error,
                    device_resolution=device_resolution,
                    reporter=reporter,
                ),
            )
            message = "Trial pruned after allocator out-of-memory failure"
            raise trial_pruned(message) from None
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="failed",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
        )
        raise
    except Exception as error:
        tracking_error = error
        _write_summary(
            run_dir=run_dir,
            config=config,
            context=context,
            status="failed",
            start_time=start_time,
            reporter=reporter,
            error=error,
            failure_context=_failure_context(error=error, device_resolution=device_resolution, reporter=reporter),
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
        experiments.console.optuna_trial_event(
            "completed",
            study=str(context["study_name"]),
            study_role=str(context["study_role"]),
            trial=int(context["trial_number"]),
            run_name=str(config["run"]["name"]),
            sampled=context["analysis_parameters"],
            objective_id=str(objective["id"]),
            objective=objective_value,
            best_trial_objective=reporter.best_value,
            step=reporter.last_reported_epoch,
            pruning="complete",
            selected_best_epoch=int(result["best_epoch"]),
            final_state="completed",
            duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
            checkpoint_state="best_and_last_published",
            run_dir=run_dir,
            wandb_url=tracker.url if tracker is not None else None,
        )
        return objective_value
    finally:
        if tracking_error is not None:
            experiments.console.optuna_trial_event(
                tracking_status,
                study=str(context["study_name"]),
                study_role=str(context["study_role"]),
                trial=int(context["trial_number"]),
                run_name=str(config["run"]["name"]),
                sampled=context["analysis_parameters"],
                objective_id=str(objective["id"]),
                objective=reporter.last_reported_objective,
                best_trial_objective=reporter.best_value,
                step=reporter.last_reported_epoch,
                pruning="prune" if tracking_status.endswith("pruned") else None,
                pruner=str(study_config.study["pruner"]["kind"]) if tracking_status.endswith("pruned") else None,
                selected_best_epoch=reporter.best_epoch,
                final_state=tracking_status,
                duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                checkpoint_state="durable_report_checkpoint" if reporter.last_reported_epoch is not None else "none",
                run_dir=run_dir,
                wandb_url=tracker.url if tracker is not None else None,
                phase=str(getattr(tracking_error, "training_phase", "optuna_trial")),
                exception_type=type(tracking_error).__name__,
                error_message=str(tracking_error),
            )
            if tracking_status not in {"pruned", "nonfinite_pruned", "oom_pruned"}:
                console_reporter.failure(tracking_error, status=tracking_status, phase="optuna_trial")
        if tracker is not None:
            local_summary: Mapping[str, Any] | None = None
            with suppress(Exception):
                local_summary = experiments.run.read_run_summary(run_dir)
            terminal_tracking_result = tracking_result
            if terminal_tracking_result is None and reporter.last_reported_epoch is not None:
                terminal_tracking_result = {
                    "best_metric": reporter.best_value,
                    "best_epoch": reporter.best_epoch,
                    "completed_epoch": reporter.last_reported_epoch,
                    "global_step": reporter.last_global_step,
                    "terminal_epoch": reporter.last_reported_epoch,
                    "terminal_metrics": {key: value for key, value in (reporter.last_metrics or {}).items() if key.startswith("train/")},
                }
            try:
                tracker.finish(
                    status=tracking_status,
                    result=terminal_tracking_result,
                    local_summary=local_summary,
                    error=tracking_error,
                )
            except experiments.tracking.TrackingError:
                if tracking_error is None:
                    raise
        scheduler = None
        optimizer = None
        eval_metrics = None
        train_loss = None
        model = None
        data_processor = None
        dataloaders = None
        result = None
        tracker = None
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_initialized():
            with suppress(RuntimeError):
                torch.cuda.empty_cache()
            with suppress(RuntimeError):
                torch.cuda.reset_peak_memory_stats(device)


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


def with_runtime_overrides(
    config: OptunaStudyConfig,
    *,
    device: str | None = None,
    output_root: Path | str | None = None,
) -> OptunaStudyConfig:
    """
    Return a validated study copy with invocation-only runtime overrides.

    Parameters
    ----------
    config : OptunaStudyConfig
        Resolved study configuration to copy.
    device : str | None, optional
        Exact runtime policy applied to every trial for this invocation.
    output_root : pathlib.Path | str | None, optional
        Invocation-only study and trial output root.

    Returns
    -------
    OptunaStudyConfig
        Revalidated copy retaining unchanged scientific study semantics.

    Raises
    ------
    experiments.config.loader.ConfigError
        If the device override or resulting resolved config is invalid.

    Notes
    -----
    The input object is not mutated. This function validates requested policy but
    does not resolve hardware, create the output root, or alter the semantic study
    signature because both overrides are invocation-only.

    """
    base_config = copy.deepcopy(config.base_config)
    if device is not None:
        base_config["run"]["device"] = device
    if output_root is not None:
        base_config["paths"]["output_root"] = str(Path(output_root).expanduser())
    base_config = experiments.config.loader.validate_resolved_config(base_config)
    return replace(config, base_config=base_config)


def _publish_study_signature(study: Any, signature: Mapping[str, Any], objective: Mapping[str, Any]) -> None:
    """
    Attach the complete semantic signature and objective to a new study.

    Schema version, digest, canonical payload, and full objective are published
    together before optimization so later reopen admission can fail closed.
    """
    study.set_user_attr(_STUDY_SIGNATURE_SCHEMA_ATTR, signature["schema_version"])
    study.set_user_attr(_STUDY_SIGNATURE_ATTR, signature["digest"])
    study.set_user_attr(_STUDY_SIGNATURE_PAYLOAD_ATTR, signature["payload"])
    study.set_user_attr(_RESOLVED_OBJECTIVE_ATTR, dict(objective))
    study.set_user_attr(_SAMPLER_METADATA_ATTR, copy.deepcopy(signature["payload"]["sampler"]))


def _validate_existing_study(
    study: Any,
    *,
    signature: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> None:
    """
    Admit an existing study only when direction and semantic metadata match.

    Missing signature fields, payload drift, or objective drift are rejected;
    name and storage identity alone never authorize adding fresh trials.
    """
    actual_direction = str(study.direction.name).lower()
    if actual_direction != objective["direction"]:
        msg = f"Existing Optuna study direction {actual_direction!r} does not match resolved objective direction {objective['direction']!r}."
        raise ValueError(msg)
    required = {
        _STUDY_SIGNATURE_SCHEMA_ATTR: signature["schema_version"],
        _STUDY_SIGNATURE_ATTR: signature["digest"],
        _STUDY_SIGNATURE_PAYLOAD_ATTR: signature["payload"],
        _RESOLVED_OBJECTIVE_ATTR: dict(objective),
        _SAMPLER_METADATA_ATTR: copy.deepcopy(signature["payload"]["sampler"]),
    }
    missing = sorted(key for key in required if key not in study.user_attrs)
    if missing:
        msg = f"Existing Optuna study is missing required semantic metadata: {missing}."
        raise ValueError(msg)

    def require_schema_version(value: Any, expected: int, *, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            msg = f"Existing Optuna study {label} must be schema version {expected}."
            raise ValueError(msg)

    require_schema_version(
        study.user_attrs[_STUDY_SIGNATURE_SCHEMA_ATTR],
        STUDY_SIGNATURE_SCHEMA_VERSION,
        label=_STUDY_SIGNATURE_SCHEMA_ATTR,
    )
    actual_payload = study.user_attrs[_STUDY_SIGNATURE_PAYLOAD_ATTR]
    if not isinstance(actual_payload, Mapping):
        msg = "Existing Optuna study semantic signature payload must be a mapping."
        raise TypeError(msg)
    actual_task = actual_payload.get("task")
    actual_lifecycle = actual_payload.get("trial_lifecycle")
    if not isinstance(actual_task, Mapping) or not isinstance(actual_lifecycle, Mapping):
        msg = "Existing Optuna study semantic signature has invalid task or trial lifecycle metadata."
        raise TypeError(msg)
    require_schema_version(
        actual_payload.get("schema_version"),
        STUDY_SIGNATURE_SCHEMA_VERSION,
        label="signature payload",
    )
    expected_payload = signature["payload"]
    require_schema_version(
        actual_task.get("schema_version"),
        expected_payload["task"]["schema_version"],
        label="task contract",
    )
    require_schema_version(
        actual_lifecycle.get("schema_version"),
        TRIAL_LIFECYCLE_SCHEMA_VERSION,
        label="trial lifecycle",
    )
    require_schema_version(
        actual_lifecycle.get("run_summary_schema_version"),
        experiments.run.RUN_SUMMARY_SCHEMA_VERSION,
        label="run summary",
    )

    mismatched = sorted(key for key, value in required.items() if study.user_attrs.get(key) != value)
    if mismatched:
        msg = f"Existing Optuna study semantic signature mismatch in: {mismatched}."
        raise ValueError(msg)


def _create_or_load_study(
    *,
    optuna: Any,
    study_name: str,
    direction: str,
    pruner: Any,
    sampler: Any,
    storage: str,
) -> tuple[Any, bool]:
    """
    Create a study or explicitly load the exact duplicate-name study.

    Creation never uses ``load_if_exists``. Only Optuna's dedicated duplicate
    exception selects the load path, and the returned boolean requires callers
    to perform semantic-signature admission before optimization.
    """
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            storage=storage,
            load_if_exists=False,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            pruner=pruner,
            sampler=sampler,
        )
        return study, True
    return study, False


def _storage_console_location(paths: _OptunaStudyPaths) -> str:
    """Return a useful storage label without credentials or query parameters."""
    if paths.local_storage_path is not None:
        return str(paths.local_storage_path)
    scheme = paths.storage.split(":", 1)[0].strip().lower()
    return f"{scheme or 'external'}://<configured>"


def _trial_state_name(trial: Any) -> str:
    """Return one stable lowercase Optuna trial state name."""
    return str(getattr(getattr(trial, "state", None), "name", "unknown")).lower()


def _study_lifecycle_summary(
    study: Any,
    *,
    config: OptunaStudyConfig,
    objective: Mapping[str, Any],
    requested_trials: int,
    reopened: bool,
    status: str,
    elapsed_seconds: float,
    include_importance: bool,
) -> dict[str, Any]:
    """Build one bounded study inventory from persisted Optuna trial facts."""
    trials = list(study.trials)
    counts = {
        "completed": sum(_trial_state_name(trial) == "complete" for trial in trials),
        "pruned": sum(_trial_state_name(trial) == "pruned" for trial in trials),
        "failed": sum(_trial_state_name(trial) == "fail" for trial in trials),
        "running": sum(_trial_state_name(trial) == "running" for trial in trials),
        "waiting": sum(_trial_state_name(trial) == "waiting" for trial in trials),
    }
    completed = [trial for trial in trials if _trial_state_name(trial) == "complete" and trial.value is not None]
    reverse = objective["direction"] == "maximize"
    best = sorted(completed, key=lambda item: float(item.value), reverse=reverse)[0] if completed else None
    inventory: list[dict[str, Any]] = []
    for trial in trials:
        started = getattr(trial, "datetime_start", None)
        finished = getattr(trial, "datetime_complete", None)
        duration = (finished - started).total_seconds() if started is not None and finished is not None else None
        attrs = getattr(trial, "user_attrs", {})
        inventory.append(
            {
                "number": int(trial.number),
                "state": _trial_state_name(trial),
                "value": trial.value,
                "params": copy.deepcopy(dict(getattr(trial, "params", {}))),
                "run_name": attrs.get("run_name") if isinstance(attrs, Mapping) else None,
                "training_seed": attrs.get("run_seed") if isinstance(attrs, Mapping) else None,
                "sampler_seed": attrs.get("sampler_seed") if isinstance(attrs, Mapping) else None,
                "duration_seconds": duration,
            }
        )

    importance: dict[str, Any]
    if include_importance and len(completed) >= _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS and any(trial.params for trial in completed):
        try:
            values = _optuna_module().importance.get_param_importances(study)
        except (ImportError, RuntimeError, TypeError, ValueError) as error:
            importance = {"status": "unavailable", "reason": type(error).__name__, "values": {}}
        else:
            importance = {"status": "computed", "minimum_completed_trials": _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS, "values": values}
    else:
        importance = {
            "status": "not_computed",
            "reason": "insufficient_completed_trials",
            "minimum_completed_trials": _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS,
            "values": {},
        }

    return {
        "schema_version": 1,
        "status": status,
        "study_name": config.study["name"],
        "study_role": config.study["role"],
        "task": config.base_config["task"],
        "model_kind": config.base_config["model"]["kind"],
        "objective": copy.deepcopy(dict(objective)),
        "sampler": {**copy.deepcopy(config.study["sampler"]), "seed": _sampler_seed(config.study)},
        "pruner": copy.deepcopy(config.study["pruner"]),
        "training_seed": int(config.base_config["run"]["seed"]),
        "configured_trials": int(config.study["n_trials"]),
        "invocation_trials_requested": requested_trials,
        "continuation": "reopened" if reopened else "new",
        "attempted_trials": len(trials),
        **counts,
        "best_trial_number": int(best.number) if best is not None else None,
        "best_objective": float(best.value) if best is not None else None,
        "best_parameters": copy.deepcopy(dict(best.params)) if best is not None else {},
        "elapsed_seconds": elapsed_seconds,
        "parameter_importance": importance,
        "trials": inventory,
    }


def _publish_study_lifecycle_summary(paths: _OptunaStudyPaths, summary: Mapping[str, Any]) -> None:
    """Persist the bounded inventory only beside repository-owned local storage."""
    if paths.local_storage_path is not None:
        common.serialization.atomic_write_json(paths.study_dir / _STUDY_SUMMARY_FILENAME, dict(summary))


def _emit_study_summary(summary: Mapping[str, Any], *, storage: str, summary_file: str | None) -> None:
    """Emit one concise W&B-independent study terminal event."""
    experiments.console.optuna_study_event(
        str(summary["status"]),
        study=summary["study_name"],
        study_role=summary["study_role"],
        configured_trials=summary["configured_trials"],
        invocation_trials_requested=summary["invocation_trials_requested"],
        attempted_trials=summary["attempted_trials"],
        completed_trials=summary["completed"],
        pruned_trials=summary["pruned"],
        failed_trials=summary["failed"],
        best_trial=summary["best_trial_number"],
        best_objective=summary["best_objective"],
        best_parameters=summary["best_parameters"],
        duration_seconds=summary["elapsed_seconds"],
        storage=storage,
        continuation=summary["continuation"],
        summary_file=summary_file,
    )


def run_optuna_study(
    config: OptunaStudyConfig | Path | str,
    *,
    n_trials: int | None = None,
    device: str | None = None,
    output_root: Path | str | None = None,
    show_progress_bar: bool = False,
) -> Any:
    """
    Run additional fresh trials in a new or semantically identical study.

    Parameters
    ----------
    config : OptunaStudyConfig | pathlib.Path | str
        Loaded study or YAML path.
    n_trials : int | None, optional
        Positive number of additional trials for this invocation.
    device : str | None, optional
        Runtime-only ``auto``, ``cpu``, or strict ``cuda`` override.
    output_root : pathlib.Path | str | None, optional
        Invocation-only study database and trial-output root.
    show_progress_bar : bool, optional
        Forward Optuna's progress display choice.

    Returns
    -------
    Any
        Optimized Optuna study object.

    Raises
    ------
    ValueError
        If an existing study's direction, objective, or semantic signature differs.
    TypeError
        If the requested additional trial count is not an exact integer.

    Notes
    -----
    Reopening adds fresh trial numbers; it never resumes partial run directories.

    """
    study_config = load_optuna_study_config(config) if isinstance(config, (str, Path)) else config
    study_config = with_runtime_overrides(study_config, device=device, output_root=output_root)
    study_config, objective = _validate_study_contract(study_config)
    signature = build_study_signature(study_config)

    raw_trial_count = n_trials if n_trials is not None else study_config.study["n_trials"]
    if type(raw_trial_count) is not int:
        msg = f"Optuna n_trials must be an integer, got: {raw_trial_count!r}"
        raise TypeError(msg)
    trial_count = raw_trial_count
    if trial_count <= 0:
        msg = f"Optuna n_trials must be positive, got {trial_count}."
        raise ValueError(msg)

    study_paths = _resolve_study_paths(study_config)
    optuna = _optuna_module()
    pruner = _build_pruner(study_config.study)
    sampler = _build_sampler(study_config.study)
    study_name = common.paths.validate_logical_name(study_config.study["name"], label="study.name")
    if study_paths.local_storage_path is not None:
        study_paths.study_dir.mkdir(parents=True, exist_ok=True)

    study, reopened = _create_or_load_study(
        optuna=optuna,
        study_name=study_name,
        direction=str(objective["direction"]),
        pruner=pruner,
        sampler=sampler,
        storage=study_paths.storage,
    )
    if reopened:
        _validate_existing_study(study, signature=signature, objective=objective)
    else:
        _publish_study_signature(study, signature, objective)

    pruner_config = study_config.study["pruner"]
    reporting_interval = _validate_reporting_contract(study_config.base_config)
    storage_label = _storage_console_location(study_paths)
    invocation_started = time.perf_counter()
    experiments.console.optuna_study_event(
        "started",
        study=study_name,
        study_role=study_config.study["role"],
        task=study_config.base_config["task"],
        model_family=study_config.base_config["model"]["kind"],
        storage=storage_label,
        sampler=study_config.study["sampler"]["kind"],
        sampler_seed=_sampler_seed(study_config.study),
        pruner=pruner_config["kind"],
        direction=objective["direction"],
        objective=objective["id"],
        configured_trials=int(study_config.study["n_trials"]),
        invocation_trials=trial_count,
        timeout="none",
        parallelism=1,
        maximum_epochs=int(study_config.base_config["training"]["epochs"]),
        id_interval=reporting_interval,
        ood_interval=reporting_interval,
        physics_interval=reporting_interval,
        startup_trials=pruner_config.get("n_startup_trials"),
        warmup_epochs=pruner_config.get("n_warmup_steps"),
        pruning_interval_epochs=pruner_config.get("interval_steps"),
        wandb_project=study_config.base_config["tracking"]["wandb"]["project"],
        wandb_group=study_name,
        wandb_workflow=study_config.base_config["tracking"]["wandb"]["workflow"],
        continuation="reopened" if reopened else "new",
        existing_trials=len(study.trials),
    )
    try:
        study.optimize(
            create_objective(study_config),
            n_trials=trial_count,
            catch=(RecoverableTrialError,),
            show_progress_bar=show_progress_bar,
        )
    except BaseException:
        summary = _study_lifecycle_summary(
            study,
            config=study_config,
            objective=objective,
            requested_trials=trial_count,
            reopened=reopened,
            status="failed",
            elapsed_seconds=time.perf_counter() - invocation_started,
            include_importance=False,
        )
        _publish_study_lifecycle_summary(study_paths, summary)
        _emit_study_summary(
            summary,
            storage=storage_label,
            summary_file=_STUDY_SUMMARY_FILENAME if study_paths.local_storage_path is not None else None,
        )
        raise
    summary = _study_lifecycle_summary(
        study,
        config=study_config,
        objective=objective,
        requested_trials=trial_count,
        reopened=reopened,
        status="completed",
        elapsed_seconds=time.perf_counter() - invocation_started,
        include_importance=True,
    )
    _publish_study_lifecycle_summary(study_paths, summary)
    _emit_study_summary(
        summary,
        storage=storage_label,
        summary_file=_STUDY_SUMMARY_FILENAME if study_paths.local_storage_path is not None else None,
    )
    return study
