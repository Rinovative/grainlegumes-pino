"""
===============================================================================
experiments_tracking.py
===============================================================================
Mirror authoritative local experiment state to optional W&B observability.

Responsibilities:
  - Keep disabled tracking free of SDK imports and tracking side effects
  - Persist opaque fresh identities and exact-resume identities locally first
  - Publish bounded semantic config, completed-epoch history, summaries, and curated media
  - Mirror authoritative ID, OOD, and physics event values without recomputation
  - Fail closed on requested online or offline observer failures
  - Keep built-in system telemetry secondary to scientific metrics

Design principles:
  - Local config, split, normalizer, checkpoints, summaries and artifacts win
  - W&B observes training and artifacts but never chooses or reconstructs them
  - Normal histories contain genuine completed epochs only. Final results are summaries
  - Fresh and exact-resume sessions have strict fail-closed identity semantics
  - Secrets, arbitrary environment state and incidental absolute paths are absent

This module does NOT:
  - Own training, checkpoint, scheduler, pruning, or local lifecycle decisions
  - Upload raw datasets, resume-only checkpoints, arbitrary files, or cache internals
  - Make remote availability or W&B state a prerequisite for local correctness
===============================================================================
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, Unpack, cast

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

EpochEndCallback = Callable[[int, dict[str, float]], None]
TrackingStateUpdater = Callable[[Mapping[str, Any]], None]

POST_ARTIFACT_MEDIA_KEYS = frozenset(
    {
        "run_summary_table",
        "accuracy_physics_pareto",
        "dual_continuity_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    }
)
_POST_ARTIFACT_FILE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".html", ".pdf"})
_SECRET_KEY_PATTERN = re.compile(r"(?i)(WANDB_API_KEY|api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")
_MAX_SAFE_ERROR_LENGTH = 600
AUTOMATIC_HISTORY_TOP_LEVEL_PREFIXES = (
    "Overview",
    "Accuracy",
    "Physics",
    "Diagnostics",
    "Optuna",
)


def _accuracy_history_metric_ids(
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
) -> tuple[str, ...]:
    """Select ordered predictive diagnostics from resolved metric semantics."""
    selected: list[str] = []
    for metric in evaluation_metrics:
        metric_id = str(metric["id"])
        if metric_id == objective_id:
            continue
        kind = str(metric["kind"])
        fields = metric["fields"]
        field_count = len(fields) if isinstance(fields, Sequence) and not isinstance(fields, str) else 0
        if kind in {"group_rmse", "vector_rmse", "relative_l2", "relative_h1"} or (kind == "rmse" and field_count == 1):
            selected.append(metric_id)
    return tuple(selected)


_PHYSICS_ID_SOURCE_KEYS = (
    "physics/id/momentum_residual_mse",
    "physics/id/continuity_div_velocity_mse",
    "physics/id/continuity_div_eps_velocity_mse",
    "physics/id/pressure_boundary_mse",
)


@dataclass(frozen=True, slots=True)
class HistoryMetricDefinition:
    """Document one authoritative source-to-W&B history mapping."""

    source_key: str
    wandb_key: str
    owner: str
    computation_cost: str
    scientific_question: str


def _definition(
    source_key: str,
    wandb_key: str,
    *,
    owner: str,
    computation_cost: str,
    scientific_question: str,
) -> HistoryMetricDefinition:
    """Build one compact immutable history definition."""
    return HistoryMetricDefinition(
        source_key=source_key,
        wandb_key=wandb_key,
        owner=owner,
        computation_cost=computation_cost,
        scientific_question=scientific_question,
    )


def automatic_history_metric_definitions(
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
    physics_training_enabled: bool,
    continuity: str,
    physics_monitor_enabled: bool,
    cuda_enabled: bool,
    optuna_trial: bool = False,
) -> tuple[HistoryMetricDefinition, ...]:
    """
    Return the ordered automatic-personal-workspace history contract.

    The source keys are authoritative local training-loop telemetry. Only this
    observer boundary renames them for W&B. Definitions follow the intended
    Overview, Accuracy/ID, Accuracy/OOD, Physics/ID, Physics/Training, and
    Diagnostics registration order. W&B itself does not guarantee that a
    personal workspace will preserve registration order in its UI.
    """
    metric_ids = frozenset(str(metric["id"]) for metric in evaluation_metrics)
    accuracy_metric_ids = _accuracy_history_metric_ids(evaluation_metrics, objective_id=objective_id)
    definitions: list[HistoryMetricDefinition] = []
    evaluation_owner = "src.learning.training.learning_training_loop.eval_one_epoch"
    training_owner = "src.learning.training.learning_training_loop.train_one_epoch"
    loop_owner = "src.learning.training.learning_training_loop.train_loop"
    monitor_owner = "src.learning.training.learning_training_loop.evaluate_physics_monitor"

    if objective_id in metric_ids:
        definitions.extend(
            (
                _definition(
                    f"id/{objective_id}",
                    f"Overview/ID/{objective_id}",
                    owner=evaluation_owner,
                    computation_cost="existing ID evaluation pass",
                    scientific_question="Is authoritative in-distribution validation performance improving?",
                ),
                _definition(
                    f"ood/{objective_id}",
                    f"Overview/OOD/{objective_id}",
                    owner=evaluation_owner,
                    computation_cost="existing OOD diagnostic pass",
                    scientific_question="Is out-of-distribution generalization improving or degrading?",
                ),
                _definition(
                    "generalization/objective_gap",
                    "Overview/generalization_gap",
                    owner=loop_owner,
                    computation_cost="one scalar subtraction",
                    scientific_question="How far does OOD error separate from ID error?",
                ),
            )
        )
    definitions.extend(
        (
            _definition(
                "train/loss_total",
                "Overview/train_loss_total",
                owner=training_owner,
                computation_cost="existing training reduction",
                scientific_question="Is the complete optimization objective converging?",
            ),
            _definition(
                "train/loss_data",
                "Overview/train_loss_data",
                owner=training_owner,
                computation_cost="existing training reduction",
                scientific_question="Is supervised fit improving independently of PI contributions?",
            ),
            _definition(
                "optimization/learning_rate",
                "Overview/learning_rate",
                owner=loop_owner,
                computation_cost="existing optimizer scalar",
                scientific_question="When and how does the scheduler react?",
            ),
        )
    )

    for role in ("ID", "OOD"):
        source_role = role.lower()
        pass_cost = "existing ID evaluation pass" if role == "ID" else "existing OOD diagnostic pass"
        for metric_id in accuracy_metric_ids:
            if metric_id not in metric_ids:
                continue
            definitions.append(
                _definition(
                    f"{source_role}/{metric_id}",
                    f"Accuracy/{role}/{metric_id}",
                    owner=evaluation_owner,
                    computation_cost=pass_cost,
                    scientific_question=f"What does {metric_id} reveal for {role} predictive accuracy?",
                )
            )

    if physics_monitor_enabled:
        for source_key in _PHYSICS_ID_SOURCE_KEYS:
            suffix = source_key.removeprefix("physics/id/")
            definitions.append(
                _definition(
                    source_key,
                    f"Physics/ID/{suffix}",
                    owner=monitor_owner,
                    computation_cost="existing bounded configured monitor pass",
                    scientific_question=f"Does the ID prediction satisfy {suffix}?",
                )
            )

    if physics_training_enabled:
        training_physics = (
            ("loss_momentum", "Does the weighted momentum contribution converge?"),
            ("loss_boundary", "Does the weighted pressure-boundary contribution converge?"),
            (f"loss_continuity_{continuity}", "Does the configured weighted continuity contribution converge?"),
            ("residual_weight", "What residual weight is actually applied during warmup?"),
            ("boundary_weight", "What boundary weight is actually applied during warmup?"),
        )
        for suffix, question in training_physics:
            definitions.append(
                _definition(
                    f"physics/train/{suffix}",
                    f"Physics/Training/{suffix}",
                    owner=training_owner,
                    computation_cost="existing PI loss telemetry",
                    scientific_question=question,
                )
            )

    if optuna_trial:
        definitions.extend(
            (
                _definition(
                    "optuna/objective",
                    "Optuna/objective",
                    owner="src.experiments.tuning.experiments_tuning_optuna.OptunaEpochReporter",
                    computation_cost="exact mirror of the existing ID objective payload",
                    scientific_question="What exact objective value did Optuna receive at this report step?",
                ),
                _definition(
                    "optuna/best_objective_so_far",
                    "Optuna/best_objective_so_far",
                    owner="src.experiments.tuning.experiments_tuning_optuna.OptunaEpochReporter",
                    computation_cost="one direction-aware scalar comparison",
                    scientific_question="What is the best reported objective reached by this trial?",
                ),
            )
        )

    definitions.extend(
        (
            _definition(
                "system/epoch_duration_seconds",
                "Diagnostics/epoch_duration_seconds",
                owner=loop_owner,
                computation_cost="one monotonic-clock subtraction",
                scientific_question="How long does each complete maintained epoch lifecycle take?",
            ),
            _definition(
                "system/train_duration_seconds",
                "Diagnostics/train_duration_seconds",
                owner=training_owner,
                computation_cost="one training-phase clock interval and one CUDA synchronization pair when applicable",
                scientific_question="How long does optimizer training take without evaluation or epoch finalization?",
            ),
            _definition(
                "system/train_samples_per_second",
                "Diagnostics/train_samples_per_second",
                owner=training_owner,
                computation_cost="one division of actual processed samples by training-phase duration",
                scientific_question="What actual training throughput is achieved at the visible configured batch size?",
            ),
        )
    )
    if cuda_enabled:
        definitions.append(
            _definition(
                "system/cuda_peak_memory_allocated_bytes",
                "Diagnostics/cuda_peak_memory_allocated_bytes",
                owner=loop_owner,
                computation_cost="existing CUDA allocator counter",
                scientific_question="What is the peak allocated CUDA memory per epoch?",
            )
        )

    source_keys = [definition.source_key for definition in definitions]
    wandb_keys = [definition.wandb_key for definition in definitions]
    if len(source_keys) != len(set(source_keys)) or len(wandb_keys) != len(set(wandb_keys)):
        msg = "Automatic W&B history definitions must have unique source and destination keys."
        raise RuntimeError(msg)
    return tuple(definitions)


class TrackingError(RuntimeError):
    """
    Base class for failures owned by the optional tracking boundary.

    These errors describe observer initialization, local offline persistence, or
    upload-admission failure. They never represent local scientific-run validity.
    """


class TrackingInitializationError(TrackingError):
    """
    Represent failure to initialize an explicitly enabled session before epoch one.

    Raised after sanitized failure facts are persisted locally. No training
    telemetry has been accepted by the observer at this boundary.
    """


class TrackingIOError(TrackingError):
    """Represent loss of records in an explicitly requested tracking mode."""


class TrackingCallbackError(TrackingError):
    """Represent a callback payload or orchestration contract violation."""


class TrackingUploadError(TrackingError):
    """
    Represent rejection by the run-file or curated-media upload allowlist.

    The error is raised before an SDK mutation for unsupported kinds, paths,
    formats, cache ownership, or disabled checkpoint publication.
    """


class _WandbRun(Protocol):
    """Describe the SDK run surface used by the lifecycle adapter."""

    summary: MutableMapping[str, Any]
    tags: Sequence[str] | None
    url: str | None

    def define_metric(self, name: str, **kwargs: Any) -> Any:
        """Define one supported metric family and its epoch step contract."""

    def log(self, data: Mapping[str, Any], *, step: int) -> None:
        """Log one completed epoch."""

    def finish(self, exit_code: int = 0) -> None:
        """Finalize the tracking run."""


class _WandbArtifact(Protocol):
    """Describe the bounded W&B artifact bundle surface."""

    def add_file(self, path: str, *, name: str) -> None:
        """Add one explicit rendered media file."""

    def add(self, value: Any, name: str) -> None:
        """Add one explicit prebuilt table object."""


class _WandbInitKwargs(TypedDict):
    """Type the exact W&B initialization keywords used by this adapter."""

    project: str
    entity: str | None
    tags: list[str] | None
    group: str | None
    job_type: str
    mode: str
    name: str
    id: str
    resume: str | None
    dir: str
    config: Mapping[str, Any]
    save_code: bool
    settings: Mapping[str, Any]


class _WandbModule(Protocol):
    """Describe the lazily imported W&B module surface."""

    def init(self, **kwargs: Unpack[_WandbInitKwargs]) -> _WandbRun | None:
        """Initialize one W&B run."""

    def Table(self, *, columns: Sequence[str], data: Sequence[Sequence[object]]) -> Any:  # noqa: N802
        """Build one W&B table from a neutral curated payload."""


def _require_initialized_run(run: _WandbRun | None) -> _WandbRun:
    """Return the SDK run or fail through the initialization wrapper."""
    if run is None:
        msg = "wandb.init() did not return a run."
        raise RuntimeError(msg)
    return run


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _safe_error(error: BaseException) -> dict[str, str]:
    """
    Return bounded exception context safe for local or remote tracking records.

    Active W&B keys, key-like assignments, and the current home path are
    redacted before truncation. Only the exception class and sanitized message
    are returned. Traceback, environment, and arbitrary object state are absent.
    """
    message = str(error)
    secret = os.environ.get("WANDB_API_KEY")
    if secret:
        message = message.replace(secret, "<redacted>")
    message = _SECRET_KEY_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", message)
    with suppress(RuntimeError):
        home = str(Path.home())
        if home and home != "/":
            message = message.replace(home, "<home>")
    return {
        "error_class": type(error).__name__,
        "error_message": message[:_MAX_SAFE_ERROR_LENGTH],
    }


def _sanitize_semantic_value(value: Any, *, key: str = "") -> Any:
    """
    Recursively copy a semantic value while excluding secrets and host paths.

    Secret-like mapping keys are removed, absolute paths retain only a basename,
    and unsupported objects fail instead of being stringified. The result is a
    bounded JSON-like structure suitable for W&B config, never scientific state.
    """
    lowered = key.lower()
    if any(marker in lowered for marker in ("api_key", "password", "secret", "token")):
        return None
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for child_key, child_value in value.items():
            name = str(child_key)
            if any(marker in name.lower() for marker in ("api_key", "password", "secret", "token")):
                continue
            sanitized[name] = _sanitize_semantic_value(child_value, key=name)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_semantic_value(item, key=key) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if Path(value).is_absolute():
            return Path(value).name or "<absolute-path-omitted>"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    msg = f"Semantic tracking payload contains unsupported value at {key or '<root>'}: {type(value).__name__}."
    raise TypeError(msg)


def _git_metadata() -> dict[str, Any]:
    """
    Return read-only commit and dirty-state facts without repository identity.

    Fixed local Git commands run with a short timeout. Missing Git, command
    errors, or timeouts return unknown values. No remote URL, branch, author,
    path, or environment state is collected.
    """
    project_root = Path(__file__).resolve().parents[3]
    git_path = shutil.which("git")
    if git_path is None:
        return {"commit": None, "dirty": None}
    try:
        revision = subprocess.run(  # noqa: S603 -- absolute git resolved from trusted PATH
            [git_path, "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(  # noqa: S603 -- fixed read-only git arguments
            [git_path, "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    return {
        "commit": commit or None,
        "dirty": None if status.returncode != 0 else bool(status.stdout),
    }


TRACKING_INTEGRATION_VERSION = 1


def model_parameter_counts(model: Any) -> dict[str, int]:
    """Count the exact instantiated total and trainable model parameters."""
    return {
        "total": sum(int(parameter.numel()) for parameter in model.parameters()),
        "trainable": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
    }


def _package_versions() -> dict[str, str | None]:
    """Return a bounded reproducibility package inventory without SDK imports."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for label, distribution in (
        ("numpy", "numpy"),
        ("neuraloperator", "neuraloperator"),
        ("optuna", "optuna"),
        ("wandb", "wandb"),
    ):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def build_semantic_config(
    config: Mapping[str, Any],
    *,
    split_indices: Mapping[str, Any],
    split_indices_sha256: str,
    normalizer_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    model: Any,
    device_metadata: Mapping[str, Any],
    duration_contract: Mapping[str, Any],
    tuning_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one complete, nested, path-free scientific W&B configuration."""
    task_contract = config.get("task_contract")
    split_metadata = split_indices.get("metadata")
    if not isinstance(task_contract, Mapping) or not isinstance(split_metadata, Mapping):
        msg = "Semantic tracking config requires resolved task and split metadata."
        raise TypeError(msg)
    datasets = split_metadata.get("datasets")
    membership = split_metadata.get("membership_digests")
    if not isinstance(datasets, Mapping) or not isinstance(membership, Mapping):
        msg = "Semantic tracking config requires dataset identities and membership digests."
        raise TypeError(msg)

    dataset_payload: dict[str, Any] = {}
    for saved_role, tracked_role in (("train", "id"), ("ood", "ood")):
        raw_identity = datasets.get(saved_role)
        if not isinstance(raw_identity, Mapping):
            msg = f"split_indices metadata dataset {saved_role!r} must be a mapping."
            raise TypeError(msg)
        dataset_payload[tracked_role] = {
            key: copy.deepcopy(raw_identity.get(key))
            for key in (
                "dataset_id",
                "fingerprint",
                "sample_count",
                "spatial_shape",
                "data_contract_digest",
            )
        }

    data_config = cast("Mapping[str, Any]", config["data"])
    loader_keys = ("batch_size", "num_workers", "pin_memory", "persistent_workers")
    split_keys = (
        "n_train_full",
        "n_train",
        "n_eval",
        "n_ood_full",
        "n_ood",
        "train_ratio",
        "ood_fraction",
        "split_seed",
    )
    preprocessing = task_contract.get("preprocessing")
    physics_contract = task_contract.get("physics")
    if not isinstance(preprocessing, Mapping) or not isinstance(physics_contract, Mapping):
        msg = "Semantic tracking config requires task preprocessing and physics semantics."
        raise TypeError(msg)

    parameter_counts = model_parameter_counts(model)
    model_payload = copy.deepcopy(dict(cast("Mapping[str, Any]", config["model"])))
    loss_config = cast("Mapping[str, Any]", config["loss"])
    physics_config = cast("Mapping[str, Any]", loss_config["physics"])
    derivative_config = physics_config.get("derivatives")
    if not isinstance(derivative_config, Mapping):
        msg = "Semantic tracking config requires physics derivative semantics."
        raise TypeError(msg)
    physics_enabled = bool(physics_config["enabled"])
    loss_payload = {
        "data": copy.deepcopy(dict(cast("Mapping[str, Any]", loss_config["data"]))),
        "physics": copy.deepcopy(dict(physics_config)) if physics_enabled else {"enabled": False},
    }
    wandb_config = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    monitor_config = cast("Mapping[str, Any]", wandb_config["monitor"])
    diagnostics_payload = {
        "physics_monitor": {
            "enabled": bool(monitor_config["enabled"]),
            "role": "id",
            "membership": "bounded_saved_evaluation_prefix",
            "interval_epochs": int(monitor_config["interval"]),
            "max_cases": int(monitor_config["max_cases"]),
            "physics_kind": physics_contract.get("kind"),
            "equation_set": physics_contract.get("equation_set"),
            "continuity_forms": copy.deepcopy(physics_contract.get("allowed_continuities")),
            "boundary": physics_contract.get("boundary"),
            "derivatives": copy.deepcopy(dict(derivative_config)),
            "interior_crop": int(physics_config["interior_crop"]),
            "metric_ids": [source.removeprefix("physics/id/") for source in _PHYSICS_ID_SOURCE_KEYS],
        }
    }
    model_kind = str(model_payload["kind"])
    model_payload["variant"] = f"pi-{model_kind}" if physics_enabled else model_kind
    model_payload["parameter_counts"] = parameter_counts
    evaluation_payload = copy.deepcopy(dict(cast("Mapping[str, Any]", config["evaluation"])))
    training_config = cast("Mapping[str, Any]", config["training"])
    evaluation_payload["roles"] = {
        "selection": "id",
        "id_interpretation": "in_distribution_validation",
        "diagnostic": ["ood", "physics"],
        "event_model": "completed_epoch_interval_or_terminal",
        "id_interval_epochs": int(training_config["evaluation_interval"]),
        "ood_interval_epochs": int(training_config["ood_evaluation_interval"]),
        "physics_interval_epochs": int(monitor_config["interval"]),
        "epoch_zero_evaluation": False,
    }
    source = _git_metadata()
    payload: dict[str, Any] = {
        "task": {
            "id": config["task"],
            "contract_digest": task_contract.get("digest"),
            "contract": copy.deepcopy(dict(task_contract)),
        },
        "data": {
            "datasets": dataset_payload,
            "loader": {key: copy.deepcopy(data_config.get(key)) for key in loader_keys},
            "split": {
                "schema_version": split_indices.get("schema_version"),
                "artifact": "split_indices.pt",
                "artifact_sha256": split_indices_sha256,
                **{key: copy.deepcopy(split_metadata.get(key)) for key in split_keys},
                "membership_digests": copy.deepcopy(dict(membership)),
            },
            "normalization": {
                **copy.deepcopy(dict(preprocessing)),
                "artifact": "normalizer.pt",
                "artifact_sha256": normalizer_sha256,
            },
        },
        "model": model_payload,
        "loss": loss_payload,
        "diagnostics": diagnostics_payload,
        "evaluation": evaluation_payload,
        "optimizer": copy.deepcopy(dict(cast("Mapping[str, Any]", config["optimizer"]))),
        "scheduler": copy.deepcopy(config.get("scheduler")),
        "training": copy.deepcopy(dict(cast("Mapping[str, Any]", config["training"]))),
        "run": copy.deepcopy(dict(cast("Mapping[str, Any]", config["run"]))),
        "provenance": {
            "repository": source,
            "config_digest": checkpoint_identity.get("effective_config_digest"),
            "task_contract_digest": task_contract.get("digest"),
            "dataset_fingerprints": {role: identity.get("fingerprint") for role, identity in dataset_payload.items()},
            "schema_versions": {
                "run": 1,
                "checkpoint": 1,
                "split": split_indices.get("schema_version"),
                "tracking_integration": TRACKING_INTEGRATION_VERSION,
            },
        },
        "runtime": {
            "device": copy.deepcopy(dict(device_metadata)),
            "packages": {**_package_versions(), "pytorch": device_metadata.get("pytorch_version")},
            "duration_contract": copy.deepcopy(dict(duration_contract)),
        },
    }
    if tuning_context is not None:
        payload["tuning"] = copy.deepcopy(dict(tuning_context))
    return cast("dict[str, Any]", _sanitize_semantic_value(payload))


def build_monitor_membership(
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    Build the exact saved-evaluation prefix identity used by physics monitors.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved experiment config with validated W&B monitor settings.
    split_indices : Mapping[str, Any]
        Persisted split artifact including ordered indices and dataset identity.

    Returns
    -------
    dict[str, Any] | None
        Source indices, sample IDs, membership digests, and configured bound when
        monitoring is enabled. Otherwise ``None``.

    Notes
    -----
    The membership is fixed before training telemetry and never resampled by epoch.

    """
    settings = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    monitor = cast("Mapping[str, Any]", settings["monitor"])
    if settings["mode"] == "disabled" or not bool(monitor["enabled"]):
        return None
    raw_indices = split_indices.get("eval_indices")
    metadata = split_indices.get("metadata")
    if not hasattr(raw_indices, "tolist") or not isinstance(metadata, Mapping):
        msg = "Physics monitor membership requires saved evaluation indices and metadata."
        raise TypeError(msg)
    datasets = metadata.get("datasets")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get("train"), Mapping):
        msg = "Physics monitor membership requires the saved training dataset identity."
        raise TypeError(msg)
    dataset_identity = cast("Mapping[str, Any]", datasets["train"])
    sample_ids = dataset_identity.get("sample_ids")
    fingerprint = dataset_identity.get("fingerprint")
    if not isinstance(sample_ids, list) or not isinstance(fingerprint, str):
        msg = "Physics monitor membership requires ordered sample IDs and fingerprint."
        raise TypeError(msg)
    selected = [int(value) for value in cast("Any", raw_indices).tolist()[: int(monitor["max_cases"])]]
    from src import datasets as dataset_package  # noqa: PLC0415

    digest = dataset_package.identity.membership_digest(
        role="wandb_physics_monitor",
        dataset_fingerprint=fingerprint,
        sample_ids=sample_ids,
        indices=selected,
    )
    membership_digests = metadata.get("membership_digests")
    saved_eval_digest = membership_digests.get("eval") if isinstance(membership_digests, Mapping) else None
    return {
        "source_indices": selected,
        "sample_ids": [sample_ids[index] for index in selected],
        "membership_digest": digest,
        "saved_eval_membership_digest": saved_eval_digest,
        "max_cases": int(monitor["max_cases"]),
    }


def persisted_wandb_identity(summary: Mapping[str, Any]) -> tuple[str, int | None]:
    """
    Recover the sole W&B identity and latest successful epoch from local sessions.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Authoritative run summary containing append-only runtime session records.

    Returns
    -------
    tuple[str, int | None]
        Persisted W&B run ID and maximum locally recorded logged epoch.

    Raises
    ------
    TrackingInitializationError
        If runtime sessions are malformed or contain zero/multiple run IDs.

    """
    raw_sessions = summary.get("runtime_sessions", [])
    if not isinstance(raw_sessions, list):
        msg = "Run summary runtime_sessions must be a list."
        raise TrackingInitializationError(msg)
    identities: set[str] = set()
    last_epoch: int | None = None
    for raw_session in raw_sessions:
        if not isinstance(raw_session, Mapping):
            continue
        state = raw_session.get("tracking")
        if not isinstance(state, Mapping):
            continue
        run_id = state.get("wandb_run_id")
        if isinstance(run_id, str) and run_id:
            identities.add(run_id)
        raw_epoch = state.get("last_logged_epoch")
        if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool):
            last_epoch = raw_epoch if last_epoch is None else max(last_epoch, raw_epoch)
    if len(identities) != 1:
        msg = f"Exact W&B resume requires one persisted run ID, found {len(identities)}."
        raise TrackingInitializationError(msg)
    return next(iter(identities)), last_epoch


@dataclass(slots=True)
class WandbSession:
    """Own one optional, fail-closed W&B mirror of authoritative local state."""

    _run: _WandbRun | None
    _wandb: Any | None
    objective_id: str
    objective_direction: str
    evaluation_metric_ids: frozenset[str]
    mode: str
    run_id: str | None
    run_dir: Path
    run_name: str
    task_id: str
    upload_settings: Mapping[str, Any]
    semantic_config: Mapping[str, Any] = field(default_factory=dict)
    history_metric_definitions: tuple[HistoryMetricDefinition, ...] = ()
    state_updater: TrackingStateUpdater | None = None
    _last_logged_epoch: int | None = None
    _finished: bool = False
    _observer_failed: bool = False
    _failed_operation: str | None = None
    _uploaded_media: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Return whether this session owns an initialized SDK run."""
        return self._run is not None

    @property
    def url(self) -> str | None:
        """Return the active W&B run URL when the SDK exposes one."""
        value = getattr(self._run, "url", None)
        return value if isinstance(value, str) and value else None

    @property
    def history_destination_by_source(self) -> dict[str, str]:
        """Return the unique admitted local-to-W&B history mapping."""
        return {definition.source_key: definition.wandb_key for definition in self.history_metric_definitions}

    def _persist(self, updates: Mapping[str, Any]) -> None:
        """Persist bounded observer state through the authoritative local writer."""
        if self.state_updater is not None:
            self.state_updater(copy.deepcopy(dict(updates)))

    def _operation_failure(self, error: BaseException, *, operation: str) -> TrackingIOError:
        """Persist sanitized context and return one fail-closed tracking error."""
        context = _safe_error(error)
        self._observer_failed = True
        self._failed_operation = operation
        self._persist({"status": "failed", "failed_operation": operation, **context})
        msg = f"Requested {self.mode} W&B {operation} failed: {context['error_class']}: {context['error_message']}"
        return TrackingIOError(msg)

    def _run_operation(self, operation: str, action: Callable[[], None]) -> bool:
        """Apply one SDK mutation or fail the explicitly requested observer."""
        if self._run is None or self._finished:
            return False
        try:
            action()
        except Exception as error:
            raise self._operation_failure(error, operation=operation) from error
        return True

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Log one strictly increasing completed epoch with only approved keys."""
        if self._run is None or self._finished:
            return
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            msg = "W&B completed-epoch history requires an integer epoch >= 1."
            raise TrackingError(msg)
        if self._last_logged_epoch is not None and epoch <= self._last_logged_epoch:
            msg = f"W&B completed-epoch history cannot rewrite epoch {epoch}. The last successful epoch is {self._last_logged_epoch}."
            raise TrackingError(msg)

        payload: dict[str, float | int] = {"epoch": epoch}
        destination_by_source = self.history_destination_by_source
        for key, value in metrics.items():
            destination = destination_by_source.get(key)
            if destination is not None:
                payload[destination] = float(value)

        if self._run_operation(
            "history",
            lambda: cast("_WandbRun", self._run).log(payload, step=epoch),
        ):
            self._last_logged_epoch = epoch
            self._persist({"last_logged_epoch": epoch})

    def _validate_run_file(self, kind: str, path: Path) -> None:
        """Admit only one completed artifact-provenance file under this run."""
        if kind != "artifact_provenance":
            msg = f"Unsupported tracked file kind {kind!r}."
            raise TrackingUploadError(msg)
        resolved = path.resolve()
        try:
            resolved.relative_to((self.run_dir / "analysis").resolve())
        except ValueError as error:
            msg = "Artifact provenance must be inside the current run analysis root."
            raise TrackingUploadError(msg) from error
        if resolved.name != "artifact_provenance.json":
            msg = "Artifact provenance upload requires artifact_provenance.json."
            raise TrackingUploadError(msg)
        if not resolved.is_file():
            msg = f"Tracked file is not complete: {resolved}"
            raise FileNotFoundError(msg)

    def upload_files(self, files: Mapping[str, Path]) -> None:
        """Upload only explicitly selected evaluation provenance files."""
        if self._run is None or self._finished or not files:
            return
        if not bool(self.upload_settings["evaluation_artifacts"]):
            msg = "Evaluation-artifact upload is disabled by configuration."
            raise TrackingUploadError(msg)
        candidates = [(kind, Path(path)) for kind, path in files.items()]
        for kind, candidate in candidates:
            self._validate_run_file(kind, candidate)
        raw_save = getattr(self._run, "save", None)
        if not callable(raw_save):
            error = TrackingUploadError("The active W&B run does not expose bounded file upload.")
            raise self._operation_failure(error, operation="evaluation_provenance_upload")
        save = cast("Callable[..., Any]", raw_save)
        for _kind, candidate in candidates:

            def upload_candidate(candidate: Path = candidate) -> None:
                """Upload one prevalidated evaluation provenance file."""
                save(
                    str(candidate),
                    base_path=str(self.run_dir),
                    policy="now",
                )

            self._run_operation(
                "evaluation_provenance_upload",
                upload_candidate,
            )
        self._persist(
            {"uploaded_provenance_files": sorted(str(candidate.resolve().relative_to(self.run_dir.resolve())) for _kind, candidate in candidates)}
        )

    def upload_post_artifact(
        self,
        *,
        artifact_root: Path,
        media_files: Mapping[str, Path] | None = None,
        tables: Mapping[str, Any] | None = None,
    ) -> None:
        """Upload the exact curated evaluation bundle without scanning a directory."""
        files = dict(media_files or {})
        table_values = dict(tables or {})
        names = set(files).union(table_values)
        unsupported = sorted(names.difference(POST_ARTIFACT_MEDIA_KEYS))
        if unsupported:
            msg = f"Unsupported post-artifact media key(s): {unsupported}."
            raise TrackingUploadError(msg)
        if names and not bool(self.upload_settings["evaluation_artifacts"]):
            msg = "Evaluation-artifact upload is disabled by configuration."
            raise TrackingUploadError(msg)
        missing = sorted(POST_ARTIFACT_MEDIA_KEYS.difference(names))
        if names and missing:
            msg = f"Curated evaluation artifact is missing required key(s): {missing}."
            raise TrackingUploadError(msg)
        if set(files).intersection(table_values):
            msg = "Post-artifact keys cannot identify both a file and a table."
            raise TrackingUploadError(msg)
        if "run_summary_table" in files:
            msg = "run_summary_table must be supplied as an already-built table object."
            raise TrackingUploadError(msg)
        if any(name != "run_summary_table" for name in table_values):
            msg = "Only run_summary_table accepts a table object."
            raise TrackingUploadError(msg)
        if not names or self._run is None or self._finished:
            return

        root = Path(artifact_root).resolve()
        resolved_files: dict[str, Path] = {}
        for name, raw_path in files.items():
            candidate = Path(raw_path).resolve()
            if candidate.is_relative_to(root):
                msg = f"Curated media {name!r} must be rendered outside the immutable artifact cache."
                raise TrackingUploadError(msg)
            if candidate.suffix.lower() not in _POST_ARTIFACT_FILE_SUFFIXES or not candidate.is_file():
                msg = f"Curated media {name!r} is not an allowed complete rendered file."
                raise TrackingUploadError(msg)
            resolved_files[name] = candidate

        normalized_tables: dict[str, Any] = {}
        for name, table in table_values.items():
            value = table
            if isinstance(table, Mapping) and set(table) == {"columns", "data"}:
                columns = table["columns"]
                data = table["data"]
                if (
                    not isinstance(columns, Sequence)
                    or isinstance(columns, (str, bytes))
                    or any(not isinstance(column, str) or not column for column in columns)
                    or not isinstance(data, Sequence)
                    or isinstance(data, (str, bytes))
                ):
                    msg = "Neutral run_summary_table payload must contain string columns and row data."
                    raise TrackingUploadError(msg)
                table_factory = getattr(self._wandb, "Table", None)
                if not callable(table_factory):
                    error = TrackingUploadError("The active W&B SDK cannot serialize the run summary table.")
                    raise self._operation_failure(error, operation="evaluation_artifact_upload")
                value = table_factory(columns=list(columns), data=list(data))
            normalized_tables[name] = value

        raw_artifact_factory = getattr(self._wandb, "Artifact", None)
        raw_log_artifact = getattr(self._run, "log_artifact", None)
        if not callable(raw_artifact_factory) or not callable(raw_log_artifact):
            error = TrackingUploadError("The active W&B SDK does not expose evaluation artifact upload.")
            raise self._operation_failure(error, operation="evaluation_artifact_upload")
        artifact_factory = cast("Callable[..., _WandbArtifact]", raw_artifact_factory)
        log_artifact = cast("Callable[..., None]", raw_log_artifact)

        def upload_bundle() -> None:
            """Create and publish the prevalidated bounded evaluation artifact."""
            bundle = artifact_factory(
                name=f"{self.task_id}-{self.run_name}-curated-media",
                type="evaluation",
                metadata={"wandb_run_id": self.run_id, "inventory": sorted(names)},
            )
            for name, path in resolved_files.items():
                bundle.add_file(str(path), name=f"{name}{path.suffix.lower()}")
            for name, table in normalized_tables.items():
                bundle.add(table, name)
            log_artifact(bundle, aliases=["latest"])

        if self._run_operation("evaluation_artifact_upload", upload_bundle):
            self._uploaded_media = sorted(names)
            self._persist({"uploaded_media": self._uploaded_media})

    def _terminal_summary(  # noqa: C901, PLR0912
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None,
        local_summary: Mapping[str, Any] | None,
        error: BaseException | str | None,
    ) -> dict[str, Any]:
        """Build one concise terminal summary from authoritative local facts."""
        summary: dict[str, Any] = {
            "run/status": status,
            "run/name": self.run_name,
            "objective/id": self.objective_id,
            "objective/direction": self.objective_direction,
            "objective/selection_role": "id_validation",
            "tracking/status": "failed" if self._observer_failed else "finished",
            "tracking/mode": self.mode,
            "tracking/run_id": self.run_id,
        }
        if self._failed_operation is not None:
            summary["tracking/failed_operation"] = self._failed_operation
        task = self.semantic_config.get("task")
        if isinstance(task, Mapping):
            summary["task/id"] = task.get("id")
            summary["task/contract_digest"] = task.get("contract_digest")
        provenance = self.semantic_config.get("provenance")
        if isinstance(provenance, Mapping):
            summary["config/digest"] = provenance.get("config_digest")
        model = self.semantic_config.get("model")
        if isinstance(model, Mapping) and isinstance(model.get("parameter_counts"), Mapping):
            counts = cast("Mapping[str, Any]", model["parameter_counts"])
            summary["model/variant"] = model.get("variant")
            summary["model/parameters_total"] = counts.get("total")
            summary["model/parameters_trainable"] = counts.get("trainable")
        data = self.semantic_config.get("data")
        if isinstance(data, Mapping):
            datasets = data.get("datasets")
            if isinstance(datasets, Mapping):
                for role in ("id", "ood"):
                    identity = datasets.get(role)
                    if isinstance(identity, Mapping):
                        summary[f"data/{role}/dataset_id"] = identity.get("dataset_id")
                        summary[f"data/{role}/fingerprint"] = identity.get("fingerprint")
            split = data.get("split")
            if isinstance(split, Mapping):
                summary["data/split_sha256"] = split.get("artifact_sha256")
            normalization = data.get("normalization")
            if isinstance(normalization, Mapping):
                summary["data/normalizer_sha256"] = normalization.get("artifact_sha256")
        tuning = self.semantic_config.get("tuning")
        if isinstance(tuning, Mapping):
            for source, target in (
                ("study_name", "tuning/study_name"),
                ("study_role", "tuning/study_role"),
                ("trial_number", "tuning/trial_number"),
                ("training_seed", "tuning/training_seed"),
                ("sampler_seed", "tuning/sampler_seed"),
                ("search_signature", "tuning/search_signature"),
                ("sampled_parameters", "tuning/sampled_parameters"),
            ):
                if source in tuning:
                    summary[target] = copy.deepcopy(tuning[source])
            summary["tuning/final_state"] = status
            summary["Optuna/trial_number"] = tuning.get("trial_number")
            summary["Optuna/state"] = status
            summary["Optuna/pruned"] = status in {"pruned", "nonfinite_pruned", "oom_pruned"}

        if result is not None:
            for source, target in (
                ("completed_epoch", "run/completed_epoch"),
                ("global_step", "run/global_step"),
                ("selected_epoch", "selected/epoch"),
                ("terminal_epoch", "terminal/epoch"),
            ):
                if source in result:
                    summary[target] = result[source]
            if tuning is not None and "best_metric" in result:
                summary["Optuna/objective"] = result["best_metric"]
            selected_metrics = result.get("selected_metrics")
            if isinstance(selected_metrics, Mapping):
                summary.update({key: value for key, value in selected_metrics.items() if isinstance(key, str) and key.startswith("selected/")})
            terminal_metrics = result.get("terminal_metrics")
            if isinstance(terminal_metrics, Mapping):
                for key, value in terminal_metrics.items():
                    if isinstance(key, str) and key.startswith("train/"):
                        summary[f"terminal/{key}"] = value
        if local_summary is not None:
            for source, target in (
                ("effective_config_digest", "config/digest"),
                ("split_indices_sha256", "data/split_sha256"),
                ("normalizer_sha256", "data/normalizer_sha256"),
                ("elapsed_seconds", "run/duration_seconds"),
            ):
                if source in local_summary:
                    summary[target] = local_summary[source]
            if isinstance(tuning, Mapping):
                if "elapsed_seconds" in local_summary:
                    summary["Optuna/trial_duration_seconds"] = local_summary["elapsed_seconds"]
                if "best_metric" in local_summary:
                    summary["Optuna/objective"] = local_summary["best_metric"]
            checkpoint_digest = local_summary.get("best_checkpoint_sha256")
            if isinstance(checkpoint_digest, str):
                summary["selected/checkpoint_sha256_short"] = checkpoint_digest[:16]
            sessions = local_summary.get("runtime_sessions")
            if isinstance(sessions, list):
                resume_count = 0
                for session in sessions:
                    if not isinstance(session, Mapping):
                        continue
                    state = session.get("tracking")
                    if isinstance(state, Mapping) and state.get("session_kind") == "resume":
                        resume_count += 1
                summary["run/resume_count"] = resume_count
        if error is not None:
            context = _safe_error(error if isinstance(error, BaseException) else RuntimeError(error))
            summary["run/error_class"] = context["error_class"]
            summary["run/error_message"] = context["error_message"]
        return summary

    def finish(
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        local_summary: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Publish terminal facts and finish exactly once, failing closed."""
        if self._run is None or self._finished:
            return
        exit_code = 0 if status == "completed" else 1
        try:
            terminal = self._terminal_summary(
                status=status,
                result=result,
                local_summary=local_summary,
                error=error,
            )
            for key, value in terminal.items():
                self._run.summary[key] = value
            self._run.finish(exit_code=exit_code)
        except Exception as failure:
            self._finished = True
            raise self._operation_failure(failure, operation="finish") from failure
        self._finished = True
        terminal_tracking_status = "failed" if self._observer_failed else "finished"
        self._persist({"status": terminal_tracking_status, "finished_at": _utc_now()})


def _runtime_wandb_tags(settings: Mapping[str, Any], semantic_config: Mapping[str, Any] | None) -> list[str]:
    """Keep role-less Optuna tags canonical and qualify only explicit study roles."""
    tags = [str(tag) for tag in cast("Sequence[str]", settings["tags"])]
    if settings.get("workflow") == "optuna_trial" and isinstance(semantic_config, Mapping):
        tuning = semantic_config.get("tuning")
        role = tuning.get("study_role") if isinstance(tuning, Mapping) else None
        if role in {"production", "smoke"}:
            tags = [f"optuna-{role}" if tag == "optuna" else tag for tag in tags]
    return list(dict.fromkeys(tags))


def initialize_wandb(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str,
    semantic_config: Mapping[str, Any] | None = None,
    resume: bool = False,
    persisted_run_id: str | None = None,
    previous_last_logged_epoch: int | None = None,
    state_updater: TrackingStateUpdater | None = None,
) -> WandbSession:
    """Initialize the configured W&B mode with strict identity and metric rules."""
    objective = cast("Mapping[str, Any]", config["evaluation"])["objective"]
    objective_mapping = cast("Mapping[str, Any]", objective)
    objective_id = str(objective_mapping["id"])
    objective_direction = str(objective_mapping["direction"])
    evaluation_metrics = cast(
        "Sequence[Mapping[str, Any]]",
        cast("Mapping[str, Any]", config["evaluation"])["metrics"],
    )
    metric_ids = frozenset(str(metric["id"]) for metric in evaluation_metrics)
    settings = cast(
        "Mapping[str, Any]",
        cast("Mapping[str, Any]", config["tracking"])["wandb"],
    )
    path = Path(run_dir)
    run_name = str(cast("Mapping[str, Any]", config["run"])["name"])
    task_id = str(config["task"])
    upload_settings = cast("Mapping[str, Any]", settings["upload"])
    monitor_settings = cast("Mapping[str, Any]", settings["monitor"])
    mode = str(settings["mode"])
    workflow = str(settings["workflow"])
    group = str(settings["study"]) if workflow == "optuna_trial" else None
    runtime_tags = _runtime_wandb_tags(settings, semantic_config)

    if mode == "disabled":
        return WandbSession(
            None,
            None,
            objective_id,
            objective_direction,
            metric_ids,
            mode,
            None,
            path,
            run_name,
            task_id,
            upload_settings,
            semantic_config=copy.deepcopy(dict(semantic_config or {})),
        )

    run_id = persisted_run_id if resume else uuid.uuid4().hex
    if not isinstance(run_id, str) or not run_id:
        msg = "Exact W&B resume requires the previously persisted non-empty run ID."
        raise TrackingInitializationError(msg)
    resume_policy: str | None
    offline_fallback: str | None = None
    if mode == "offline":
        resume_policy = None
        if resume:
            offline_fallback = "offline_same_persisted_id_segment"
    else:
        resume_policy = "must" if resume else "never"

    base_state: dict[str, Any] = {
        "requested_mode": mode,
        "wandb_run_id": run_id,
        "project": settings["project"],
        "entity": settings["entity"],
        "tags": runtime_tags,
        "group": group,
        "job_type": workflow,
        "session_started_at": _utc_now(),
        "session_kind": "resume" if resume else "fresh",
        "status": "offline" if mode == "offline" else "active",
    }
    if offline_fallback is not None:
        base_state["offline_resume_fallback"] = offline_fallback
    if state_updater is not None:
        state_updater(base_state)

    if mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        error = RuntimeError("WANDB_API_KEY is missing or blank. Online tracking requires non-interactive environment authentication.")
        context = _safe_error(error)
        if state_updater is not None:
            state_updater(
                {
                    "status": "failed_before_start",
                    "failed_operation": "authentication",
                    **context,
                }
            )
        message = f"tracking.wandb.mode='online' authentication failed before epoch 1: {context['error_message']}"
        raise TrackingInitializationError(message) from error

    physics_config = cast(
        "Mapping[str, Any]",
        cast("Mapping[str, Any]", config["loss"])["physics"],
    )
    runtime_config = semantic_config.get("runtime") if isinstance(semantic_config, Mapping) else None
    runtime_device = runtime_config.get("device") if isinstance(runtime_config, Mapping) else None
    resolved_device = runtime_device.get("resolved_device") if isinstance(runtime_device, Mapping) else None
    requested_device = cast("Mapping[str, Any]", config["run"]).get("device")
    cuda_enabled = (isinstance(resolved_device, str) and resolved_device.startswith("cuda")) or (
        resolved_device is None and isinstance(requested_device, str) and requested_device.startswith("cuda")
    )
    history_definitions = automatic_history_metric_definitions(
        evaluation_metrics,
        objective_id=objective_id,
        physics_training_enabled=bool(physics_config["enabled"]),
        continuity=str(physics_config["continuity"]),
        physics_monitor_enabled=bool(monitor_settings["enabled"]),
        cuda_enabled=cuda_enabled,
        optuna_trial=workflow == "optuna_trial",
    )

    sdk_run: _WandbRun | None = None
    try:
        wandb = cast("_WandbModule", importlib.import_module("wandb"))
        sdk_run = _require_initialized_run(
            wandb.init(
                project=str(settings["project"]),
                entity=cast("str | None", settings["entity"]),
                tags=None if resume and mode == "online" else runtime_tags,
                group=group,
                job_type=workflow,
                mode=mode,
                name=run_name,
                id=run_id,
                resume=resume_policy,
                dir=str(path),
                config=copy.deepcopy(dict(semantic_config or {})),
                save_code=False,
                settings={
                    "disable_git": True,
                    "disable_code": True,
                },
            )
        )
        required_tags = tuple(runtime_tags)
        existing_tags = tuple(str(tag) for tag in (sdk_run.tags or ()))
        merged_tags = (*existing_tags, *(tag for tag in required_tags if tag not in existing_tags))
        if merged_tags != existing_tags:
            sdk_run.tags = merged_tags

        sdk_run.define_metric("epoch", hidden=True, summary="none")
        for definition in history_definitions:
            sdk_run.define_metric(
                definition.wandb_key,
                step_metric="epoch",
                step_sync=False,
                summary="none",
            )
    except Exception as error:
        if sdk_run is not None:
            with suppress(Exception):
                sdk_run.finish(exit_code=1)
        context = _safe_error(error)
        if state_updater is not None:
            state_updater(
                {
                    "status": "failed_before_start",
                    "failed_operation": "initialization",
                    **context,
                }
            )
        message = (
            f"tracking.wandb.mode={mode!r} initialization failed before epoch 1: "
            f"{context['error_class']}: {context['error_message']}. "
            "Verify W&B installation, local write access, and online authentication."
        )
        raise TrackingInitializationError(message) from error

    return WandbSession(
        sdk_run,
        wandb,
        objective_id,
        objective_direction,
        metric_ids,
        mode,
        run_id,
        path,
        run_name,
        task_id,
        upload_settings,
        semantic_config=copy.deepcopy(dict(semantic_config or {})),
        history_metric_definitions=history_definitions,
        state_updater=state_updater,
        _last_logged_epoch=previous_last_logged_epoch,
    )


def epoch_callback(
    session: WandbSession,
    optimizer: Optimizer | None = None,
) -> EpochEndCallback | None:
    """
    Bind an enabled W&B session to completed-epoch telemetry.

    Parameters
    ----------
    session : WandbSession
        Initialized or disabled observer session.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose first parameter-group learning rate is added when absent.

    Returns
    -------
    EpochEndCallback | None
        Callback for the training loop, or ``None`` for a disabled session.

    """
    if not session.enabled:
        return None

    def callback(epoch: int, metrics: dict[str, float]) -> None:
        """Add the current optimizer rate and forward one completed-epoch payload."""
        values = dict(metrics)
        if "optimization/learning_rate" not in values and optimizer is not None:
            parameter_groups = optimizer.param_groups
            if not parameter_groups:
                msg = "Cannot log W&B learning rate: optimizer has no parameter groups."
                raise RuntimeError(msg)
            values["optimization/learning_rate"] = float(parameter_groups[0]["lr"])
        session.log_epoch(epoch, values)

    return callback


def combine_epoch_callbacks(
    *callbacks: EpochEndCallback | None,
) -> EpochEndCallback | None:
    """
    Combine non-null completed-epoch consumers without changing their order.

    Parameters
    ----------
    *callbacks : EpochEndCallback | None
        Scheduler-independent lifecycle callbacks such as Optuna reporting and W&B.

    Returns
    -------
    EpochEndCallback | None
        Ordered composite, or ``None`` when every input is disabled.

    Notes
    -----
    Exceptions propagate immediately, so later consumers do not observe an epoch
    rejected by an earlier authoritative consumer.

    """
    active = tuple(callback for callback in callbacks if callback is not None)
    if not active:
        return None

    def combined(epoch: int, metrics: dict[str, float]) -> None:
        """Invoke lifecycle consumers in caller-specified order for one epoch."""
        for callback in active:
            callback(epoch, metrics)

    return combined
