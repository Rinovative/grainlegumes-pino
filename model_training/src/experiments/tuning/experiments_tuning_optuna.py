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
import json
import math
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

STUDY_SIGNATURE_SCHEMA_VERSION = 1
TRIAL_LIFECYCLE_SCHEMA_VERSION = 1
_STUDY_SIGNATURE_ATTR = "semantic_signature"
_STUDY_SIGNATURE_PAYLOAD_ATTR = "semantic_signature_payload"
_STUDY_SIGNATURE_SCHEMA_ATTR = "semantic_signature_schema_version"
_RESOLVED_OBJECTIVE_ATTR = "resolved_objective"
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
    """
    Report one held-out objective at every actual completed training epoch.

    The adapter enforces continuous one-based epoch steps, finite values, the
    configured optimization direction, and immediate pruning after local epoch
    publication. It retains the last metrics/global step for truthful terminal
    summaries and never reconstructs reports from planned epochs.

    Attributes
    ----------
    trial : TrialProtocol
        Current Optuna trial receiving reports and user attributes.
    objective_id : str
        Exact held-out evaluation key required at every completed epoch.
    direction : {"minimize", "maximize"}
        Direction used to retain best value and epoch.
    last_reported_epoch : int | None
        Latest durable one-based step accepted by Optuna.
    last_metrics : dict[str, float] | None
        Complete telemetry for the latest accepted epoch, used for terminal mirrors.

    """

    trial: TrialProtocol
    objective_id: str
    direction: str
    best_value: float | None = None
    best_epoch: int | None = None
    last_reported_epoch: int | None = None
    last_reported_objective: float | None = None
    last_global_step: int | None = None
    last_metrics: dict[str, float] | None = None

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Publish one completed-epoch objective and honor the pruner immediately.

        Parameters
        ----------
        epoch : int
            Actual positive completed epoch; gaps and duplicates are rejected.
        metrics : dict[str, float]
            Completed-epoch telemetry containing the configured held-out
            objective and optionally ``global_step``.

        Raises
        ------
        NonFiniteTrialError
            If the objective is NaN or infinite.
        optuna.TrialPruned
            If the trial pruner rejects the newly reported epoch.

        """
        if type(epoch) is not int or epoch <= 0:
            msg = f"Completed Optuna epoch must be a positive integer, got {epoch!r}."
            raise TypeError(msg)
        expected_epoch = 1 if self.last_reported_epoch is None else self.last_reported_epoch + 1
        if epoch != expected_epoch:
            msg = f"Optuna reports must be continuous actual epochs; expected {expected_epoch}, got {epoch}."
            raise ValueError(msg)
        if self.objective_id not in metrics:
            msg = f"Held-out Optuna objective {self.objective_id!r} is missing at completed epoch {epoch}."
            raise KeyError(msg)
        value = float(metrics[self.objective_id])
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

        if self.trial.should_prune():
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
        pruner.setdefault("n_warmup_steps", 20)
        pruner.setdefault("interval_steps", 1)
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
    _require_nonempty_string(study.get("objective"), label="study.objective")
    direction = _require_nonempty_string(study.get("direction"), label="study.direction")
    if direction not in {"minimize", "maximize"}:
        msg = f"study.direction must be minimize or maximize, got {direction!r}."
        raise ValueError(msg)
    _require_exact_int(study.get("seed"), label="study.seed")
    _require_exact_int(study.get("n_trials"), label="study.n_trials", minimum=1)
    _normalise_sampler_config(study.get("sampler"))
    _normalise_pruner_config(study.get("pruner"))
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
    allowed = {"name", "seed", "n_trials", "sampler", "pruner", "storage"}
    unknown = sorted(set(raw_study).difference(allowed))
    if unknown:
        msg = f"study contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ValueError(msg)

    study = dict(copy.deepcopy(raw_study))
    study.setdefault("name", source_path.stem)
    study.setdefault("seed", base_config.get("run", {}).get("seed", 9))
    study.setdefault("n_trials", 30)
    study.setdefault(
        "pruner",
        {"kind": "median", "n_startup_trials": 5, "n_warmup_steps": 20, "interval_steps": 1},
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

    The inline experiment is deep-copied and resolved first. Study objective and
    direction are then derived from that resolved config before exact parameter
    schemas, approved paths, supported choices, and base-value containment are
    validated.

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
    _validate_search_space_choices(base_config, search_parameters)

    return OptunaStudyConfig(
        path=source_path,
        study=study,
        base_experiment=base_experiment,
        base_config=base_config,
        search_space=search_parameters,
    )


def _validate_reporting_contract(base_config: Mapping[str, Any]) -> None:
    """
    Require held-out objective evaluation after every completed trial epoch.

    Continuous one-based Optuna reports and immediate pruning are truthful only
    when ``training.evaluation_interval`` is exactly one; larger intervals fail
    before study creation or trial-directory allocation.
    """
    training = _as_mapping(base_config.get("training"), label="experiment.training")
    interval = _require_exact_int(
        training.get("evaluation_interval"),
        label="training.evaluation_interval",
        minimum=1,
    )
    if interval != 1:
        msg = "Optuna training.evaluation_interval must be 1 for continuous pruning."
        raise ValueError(msg)


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
    _validate_reporting_contract(base_config)
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
        "objective": objective,
        "direction": objective["direction"],
        "sampler": {**sampler, "seed": sampler_seed},
        "pruner": pruner,
        "reporting": {
            "metric_source": "held_out_evaluation",
            "evaluation_interval": 1,
            "step": "actual_completed_epoch",
            "pruning_subset": None,
        },
        "trial_lifecycle": {
            "schema_version": TRIAL_LIFECYCLE_SCHEMA_VERSION,
            "run_summary_schema_version": experiments.run.RUN_SUMMARY_SCHEMA_VERSION,
            "outcomes": list(_TRIAL_OUTCOMES),
            "resume_policy": "new_trials_only",
            "trial_count_policy": "additional_fresh_trials_per_invocation",
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


def _describe_dataset_identity(config: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    """Validate one configured dataset through its compact metadata package."""
    task = experiments.config.loader.validate_resolved_task_contract(config)
    paths = _as_mapping(config.get("paths"), label="experiment.paths")
    dataset_path = common.paths.resolve_dataset_path(dataset_id, dataset_root=Path(paths["dataset_root"]))
    metadata_dir = common.paths.resolve_dataset_metadata_dir(dataset_id, metadata_root=Path(paths["training_meta_root"]))
    if not dataset_path.is_file() or dataset_path.is_symlink():
        msg = f"Configured training dataset is not a regular file: {dataset_path}"
        raise FileNotFoundError(msg)

    metadata = _as_mapping(
        json.loads((metadata_dir / datasets.metadata.METADATA_FILENAME).read_text(encoding="utf-8")),
        label="dataset metadata",
    )
    manifest = _as_mapping(
        json.loads((metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME).read_text(encoding="utf-8")),
        label="source manifest snapshot",
    )
    scientific = _as_mapping(metadata.get("scientific_identity"), label="dataset metadata scientific_identity")
    identity = datasets.identity.DatasetIdentity(
        dataset_id=dataset_id,
        task=task.id,
        task_contract_digest=task.contract_digest,
        fingerprint=scientific["dataset_fingerprint"],
        sample_ids=tuple(manifest["intended_case_ids"]),
        sample_count=scientific["sample_count"],
        spatial_shape=tuple(scientific["spatial_shape"]),
        generated_batch_identity_sha256=scientific["generated_batch_identity_sha256"],
    )
    package = datasets.metadata.validate_dataset_metadata_directory(metadata_dir, dataset_identity=identity)
    artifact = package.metadata["artifacts"]["dataset"]
    if dataset_path.name != artifact["filename"] or dataset_path.stat().st_size != artifact["size_bytes"]:
        msg = "Configured training dataset name or size does not match its metadata package."
        raise ValueError(msg)
    return {
        "dataset_id": identity.dataset_id,
        "dataset_path": str(dataset_path),
        "metadata_dir": str(metadata_dir),
        "validation": "metadata_package_and_artifact_stat",
        "task": identity.task,
        "task_contract_digest": identity.task_contract_digest,
        "fingerprint": identity.fingerprint,
        "sample_count": identity.sample_count,
    }


def _configured_dataset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the exact configured ID and OOD dataset roles."""
    data = _as_mapping(config.get("data"), label="experiment.data")
    train_dataset = _require_nonempty_string(data.get("train_dataset"), label="data.train_dataset")
    raw_ood = data.get("ood_datasets")
    if not isinstance(raw_ood, list) or len(raw_ood) != 1:
        msg = "data.ood_datasets must contain exactly one configured dataset id."
        raise ValueError(msg)
    ood_dataset = _require_nonempty_string(raw_ood[0], label="data.ood_datasets[0]")
    return {
        "id": _describe_dataset_identity(config, train_dataset),
        "ood": [_describe_dataset_identity(config, ood_dataset)],
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
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 20),
            interval_steps=pruner_cfg.get("interval_steps", 1),
        )
    msg = f"Unsupported Optuna pruner: {pruner_type!r}"
    raise ValueError(msg)


def _sampler_seed(study: Mapping[str, Any]) -> int:
    """Return a stable subseed in Optuna/NumPy's accepted 32-bit domain."""
    raw_study_seed = _require_exact_int(study.get("seed"), label="study.seed")
    return experiments.run.derive_subseed(raw_study_seed, "optuna-sampler") % (2**32)


def _build_sampler(study: Mapping[str, Any]) -> Any:
    """
    Build a deterministic Optuna sampler from validated semantic policy.

    Both random and TPE receive the stable 32-bit study subseed; only TPE accepts
    the normalized multivariate flag. Optuna remains lazily imported.
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
    """
    Sample, identify, and fully validate one fresh trial config.

    Overrides are applied to an isolated base copy, a stable study/trial subseed
    replaces the base run seed, objective identity is required to stay unchanged,
    and a trial-qualified run name is regenerated. Run metadata is recorded on
    the Optuna trial only after final resolved-config validation.
    """
    overrides = search_space.suggest_trial_overrides(trial, study_config.search_space)
    config = search_space.apply_trial_overrides(study_config.base_config, overrides)
    config["run"]["seed"] = experiments.run.derive_subseed(
        int(study_config.study["seed"]),
        f"trial-{trial.number}",
    )
    config = experiments.config.loader.validate_resolved_config(config)

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
    device = device_resolution.device
    objective = experiments.config.loader.get_resolved_objective(config)
    reporter = OptunaEpochReporter(
        trial=trial,
        objective_id=str(objective["id"]),
        direction=str(objective["direction"]),
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
    monitor_evaluator: experiments.tracking.MonitorEvaluator | None = None
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
        eval_metrics = learning.losses.factory.build_eval_metrics(config, device=device)
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
        if bool(config["tracking"]["wandb"]["enabled"]):
            semantic_config = experiments.tracking.build_semantic_config(
                config,
                split_indices=dataloaders["split_indices"],
                normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
                checkpoint_identity=checkpoint_identity,
                model=model,
                device_metadata=device_resolution.as_dict(),
            )
        if bool(config["tracking"]["wandb"]["enabled"]) and bool(monitor_settings["enabled"]):
            max_monitor_cases = int(monitor_settings["max_cases"])

            def evaluate_monitor() -> Mapping[str, float]:
                """Evaluate physics on the fixed trial monitor membership."""
                return learning.training.loop.evaluate_physics_monitor(
                    model,
                    dataloaders["eval"],
                    train_loss,
                    device,
                    data_processor,
                    max_cases=max_monitor_cases,
                )

            monitor_evaluator = evaluate_monitor
        tracker = experiments.tracking.initialize_wandb(
            config,
            run_dir=run_dir,
            semantic_config=semantic_config,
            state_updater=state_updater,
            monitor_evaluator=monitor_evaluator,
        )
        wandb_epoch_callback = experiments.tracking.epoch_callback(tracker)

        def mirror_epoch_to_wandb(epoch: int, metrics: dict[str, float]) -> None:
            """Mirror trial telemetry only when an optional W&B session is active."""
            if wandb_epoch_callback is not None:
                wandb_epoch_callback(epoch, metrics)

        def trial_epoch_callback(epoch: int, metrics: dict[str, float]) -> None:
            """Report pruning evidence before optional observer mirroring."""
            reporter(epoch, metrics)
            mirror_epoch_to_wandb(epoch, metrics)

        result = learning.training.loop.train_loop(
            config=config,
            device=device,
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
            epoch_end_callback=trial_epoch_callback,
            checkpoint_identity=checkpoint_identity,
        )
        objective_value = _finite_objective_value(result, objective)
        _validate_completed_reporting(reporter, result)
        tracking_status = "completed"
        tracking_result = result

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
            try:
                wandb_epoch_callback(pruning_epoch, reporter.last_metrics)
            except experiments.tracking.TrackingError as mirror_error:
                trial.set_user_attr("wandb_prune_mirror_error", str(mirror_error))
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
        return objective_value
    finally:
        if tracker is not None:
            local_summary: Mapping[str, Any] | None = None
            with suppress(Exception):
                local_summary = experiments.run.read_run_summary(run_dir)
            try:
                tracker.finish(
                    status=tracking_status,
                    result=tracking_result,
                    local_summary=local_summary,
                    error=tracking_error,
                )
            except experiments.tracking.TrackingError:
                if tracking_error is None:
                    raise
        monitor_evaluator = None
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

    study.optimize(
        create_objective(study_config),
        n_trials=trial_count,
        catch=(RecoverableTrialError,),
        show_progress_bar=show_progress_bar,
    )
    return study
