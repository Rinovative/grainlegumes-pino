"""
===============================================================================
experiments_config_loader.py
===============================================================================
Load and strictly resolve semantic experiment configurations.

Responsibilities:
  - Parse YAML mappings under the strict experiment schema
  - Reject unknown keys, identifiers, fields, and contradictory settings
  - Derive task-fixed channels, defaults, objective, and task-contract digest
  - Construct dataloaders from an already resolved configuration

Design principles:
  - Resolution is strict, path-aware, deterministic, and side-effect free
  - Task-fixed semantics and full metric definitions come only from domain.tasks
  - Executable YAMLs use one canonical section order and explicitly select their objective
  - Saved configuration identifiers never depend on Python class names

This module does NOT:
  - Define dataset storage or fingerprints; dataset objects enforce those contracts
  - Implement physics equations or metric mathematics; domain and learning modules do
  - Own checkpoint, resume, run-directory, or artifact lifecycle
===============================================================================
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from src import common, datasets, domain
from src.learning import learning_device_policy
from src.learning.losses import learning_losses_factory as loss_factory
from src.learning.metrics import learning_metrics as metric_registry
from src.learning.models import learning_models_factory as model_factory

from . import experiments_config_defaults as config_defaults


class ConfigError(ValueError):
    """
    Represent a path-qualified semantic configuration violation.

    This error is raised at raw-schema and resolved-config boundaries. Registry
    errors are wrapped as ``ConfigError`` when their meaning belongs to a YAML
    path, while unknown standalone registry lookups may still raise ``ValueError``.
    """


CANONICAL_EXPERIMENT_SECTION_ORDER = (
    "task",
    "run",
    "data",
    "model",
    "loss",
    "evaluation",
    "optimizer",
    "scheduler",
    "training",
    "tracking",
)
_ROOT_KEYS = frozenset(CANONICAL_EXPERIMENT_SECTION_ORDER)
EXPERIMENT_ROOT_KEYS = _ROOT_KEYS
_TASK_CONFIG_MARKER = ("configs", "tasks")
_TASK_FIXED_KEYS = frozenset(
    {
        "input_fields",
        "output_fields",
        "in_channels",
        "out_channels",
        "task_contract",
        "preprocessing",
        "physics",
        "paths",
    }
)
_ADAM_BETA_COUNT = 2
_RESOLVED_PATH_KEYS = frozenset(
    {
        "project_root",
        "model_training_data_root",
        "training_meta_root",
        "dataset_root",
        "output_root",
    }
)
_SECTION_KEYS = {
    "run": frozenset({"seed", "deterministic", "device", "suffix", "name"}),
    "data": frozenset(
        {
            "train_dataset",
            "ood_datasets",
            "train_ratio",
            "ood_fraction",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
        }
    ),
    "model": frozenset({"kind", "params"}),
    "loss": frozenset({"data", "physics"}),
    "evaluation": frozenset({"metrics", "objective"}),
    "optimizer": frozenset({"kind", "lr", "weight_decay", "betas", "second_moment_floor"}),
    "scheduler": frozenset({"kind", "factor", "patience", "min_lr"}),
    "training": frozenset({"epochs", "evaluation_interval", "ood_evaluation_interval", "mixed_precision"}),
    "tracking": frozenset({"wandb"}),
}


def _as_mapping(value: Any, *, path: str) -> dict[str, Any]:
    """Return a mutable mapping copy with path-rich type errors."""
    if not isinstance(value, Mapping):
        msg = f"{path} must be a mapping, got {type(value).__name__}."
        raise ConfigError(msg)
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    """Reject keys outside one strict schema node."""
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        msg = f"{path} contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ConfigError(msg)


def _validate_input_schema(user_config: Mapping[str, Any]) -> None:  # noqa: C901, PLR0912
    """
    Reject noncanonical task-fixed overrides and unknown nested keys.

    Validation walks every user-addressable schema node before defaults are
    merged. It admits only semantic selectors, rejects derived task contracts
    and channel counts, and reports the exact dotted path of an unsupported key.
    The supplied mapping is inspected without mutation.
    """
    fixed = sorted(set(user_config).intersection(_TASK_FIXED_KEYS))
    if fixed:
        msg = f"Task-fixed config key(s) cannot be overridden: {fixed}. Select a registered task instead."
        raise ConfigError(msg)
    _reject_unknown(user_config, _ROOT_KEYS, path="config")

    for section, allowed in _SECTION_KEYS.items():
        if section not in user_config or user_config[section] is None:
            continue
        section_mapping = _as_mapping(user_config[section], path=section)
        _reject_unknown(section_mapping, allowed, path=section)

    raw_run = _as_mapping(user_config.get("run"), path="run")
    if raw_run.get("name") is not None:
        msg = "run.name is derived and must not be supplied by an executable request."
        raise ConfigError(msg)

    model = _as_mapping(user_config.get("model"), path="model")
    params = _as_mapping(model.get("params"), path="model.params")
    fixed_channels = sorted({"in_channels", "out_channels"}.intersection(params))
    if fixed_channels:
        msg = f"model.params task-fixed channel key(s) cannot be overridden: {fixed_channels}."
        raise ConfigError(msg)

    if "loss" in user_config:
        loss = _as_mapping(user_config["loss"], path="loss")
        if "data" in loss:
            data_loss = _as_mapping(loss["data"], path="loss.data")
            _reject_unknown(data_loss, frozenset({"kind", "space", "weight"}), path="loss.data")
        if "physics" in loss:
            physics = _as_mapping(loss["physics"], path="loss.physics")
            _reject_unknown(
                physics,
                frozenset(
                    {
                        "enabled",
                        "continuity",
                        "derivatives",
                        "interior_crop",
                        "residual_weight",
                        "boundary_weight",
                    }
                ),
                path="loss.physics",
            )
            if "derivatives" in physics:
                derivatives = _as_mapping(physics["derivatives"], path="loss.physics.derivatives")
                _reject_unknown(
                    derivatives,
                    frozenset({"kind", "extension"}),
                    path="loss.physics.derivatives",
                )
            for weight_name in ("residual_weight", "boundary_weight"):
                if weight_name not in physics:
                    continue
                weight = _as_mapping(physics[weight_name], path=f"loss.physics.{weight_name}")
                _reject_unknown(
                    weight,
                    frozenset({"target", "warmup"}),
                    path=f"loss.physics.{weight_name}",
                )
                if "warmup" in weight:
                    warmup = _as_mapping(weight["warmup"], path=f"loss.physics.{weight_name}.warmup")
                    _reject_unknown(
                        warmup,
                        frozenset({"kind", "epochs"}),
                        path=f"loss.physics.{weight_name}.warmup",
                    )

    if "tracking" in user_config:
        tracking = _as_mapping(user_config["tracking"], path="tracking")
        if "wandb" in tracking:
            wandb = _as_mapping(tracking["wandb"], path="tracking.wandb")
            _reject_unknown(
                wandb,
                frozenset(
                    {
                        "mode",
                        "workflow",
                        "study",
                        "monitor",
                        "upload",
                    }
                ),
                path="tracking.wandb",
            )
            if "monitor" in wandb:
                monitor = _as_mapping(
                    wandb["monitor"],
                    path="tracking.wandb.monitor",
                )
                _reject_unknown(
                    monitor,
                    frozenset({"enabled", "interval", "max_cases"}),
                    path="tracking.wandb.monitor",
                )
            if "upload" in wandb:
                upload = _as_mapping(
                    wandb["upload"],
                    path="tracking.wandb.upload",
                )
                _reject_unknown(
                    upload,
                    frozenset({"evaluation_artifacts"}),
                    path="tracking.wandb.upload",
                )

    if "evaluation" not in user_config:
        msg = "evaluation.objective is required for every executable request."
        raise ConfigError(msg)
    evaluation = _as_mapping(user_config["evaluation"], path="evaluation")
    if "metrics" in evaluation:
        metrics = evaluation["metrics"]
        if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
            msg = "evaluation.metrics must be a list of metric mappings."
            raise ConfigError(msg)
        for index, raw_metric in enumerate(metrics):
            metric = _as_mapping(raw_metric, path=f"evaluation.metrics[{index}]")
            _reject_unknown(
                metric,
                frozenset({"id", "kind", "space", "fields", "reduction"}),
                path=f"evaluation.metrics[{index}]",
            )
    if "objective" not in evaluation:
        msg = "evaluation.objective is required for every executable request."
        raise ConfigError(msg)
    objective = _as_mapping(evaluation["objective"], path="evaluation.objective")
    _reject_unknown(
        objective,
        frozenset({"id"}),
        path="evaluation.objective",
    )
    objective_id = objective.get("id")
    if not isinstance(objective_id, str) or not objective_id:
        msg = "evaluation.objective.id must be a non-empty string."
        raise ConfigError(msg)


def task_directory_from_config_path(yaml_path: Path | str) -> str | None:
    """Return the task owned by a ``configs/tasks/<task>/`` source path."""
    parts = Path(yaml_path).expanduser().parts
    matches = [index for index in range(len(parts) - 1) if tuple(parts[index : index + len(_TASK_CONFIG_MARKER)]) == _TASK_CONFIG_MARKER]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Config path contains an ambiguous repeated configs/tasks marker: {yaml_path}"
        raise ConfigError(msg)
    task_index = matches[0] + len(_TASK_CONFIG_MARKER)
    if task_index >= len(parts) - 1:
        msg = f"Task-first config path must include configs/tasks/<task>/<workflow>/...: {yaml_path}"
        raise ConfigError(msg)
    try:
        return common.paths.validate_logical_name(parts[task_index], label="config directory task")
    except ValueError as error:
        raise ConfigError(str(error)) from error


def validate_task_directory_identity(
    yaml_path: Path | str,
    *,
    raw_task: object,
    resolved_task: object,
) -> None:
    """Require a task-first source directory to agree with raw and resolved task identity."""
    directory_task = task_directory_from_config_path(yaml_path)
    if directory_task is None:
        return
    if raw_task != directory_task:
        msg = (
            f"Task config path mismatch for {yaml_path}: directory task is {directory_task!r}, "
            f"but raw task is {raw_task!r}. Move the YAML or set its authoritative task to {directory_task!r}."
        )
        raise ConfigError(msg)
    if resolved_task != directory_task:
        msg = f"Task config path mismatch for {yaml_path}: directory task is {directory_task!r}, but resolved task is {resolved_task!r}."
        raise ConfigError(msg)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """
    Load one YAML experiment mapping under the strict schema.

    Parameters
    ----------
    path : Path or str
        YAML source path.

    Returns
    -------
    dict[str, Any]
        Raw semantic experiment mapping.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ConfigError
        If the YAML root is not a mapping.

    """
    source_path = Path(path)
    if not source_path.exists():
        msg = f"Config file not found: {source_path}"
        raise FileNotFoundError(msg)
    with source_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, Mapping):
        msg = f"YAML root must be a mapping: {source_path}"
        raise ConfigError(msg)
    return dict(payload)


def save_yaml(config: dict[str, Any], path: Path | str) -> None:
    """
    Save a resolved semantic config mapping.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved semantic configuration.
    path : Path or str
        Destination YAML path.

    Notes
    -----
    Serialization is assembled in memory and published through atomic text
    replacement; callers never observe a partially written config.

    """
    destination = Path(path)
    stream = StringIO()
    yaml.dump(config, stream, default_flow_style=False, sort_keys=False)
    common.serialization.atomic_write_text(destination, stream.getvalue())


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursively merge mappings while replacing scalar and list leaves.

    Parameters
    ----------
    base : dict[str, Any]
        Base mapping copied before merge.
    override : Mapping[str, Any]
        Values that override matching base paths.

    Returns
    -------
    dict[str, Any]
        Independent merged mapping.

    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_loss(config: dict[str, Any], *, task: domain.tasks.spec.TaskSpec) -> None:
    """
    Resolve loss semantics against the registered task contract in place.

    The helper validates supervised space/weight, continuity formulation,
    derivative/extension compatibility, crop size, and both linear warmup
    schedules. Canonical numeric values are written back into ``config`` only
    after their path-qualified constraints have been checked.
    """
    loss = _as_mapping(config["loss"], path="loss")
    data_loss = _as_mapping(loss["data"], path="loss.data")
    kind = str(data_loss["kind"])
    if kind not in task.data_losses:
        msg = f"loss.data.kind {kind!r} is not allowed by task {task.id!r}: {list(task.data_losses)}."
        raise ConfigError(msg)
    try:
        loss_factory.validate_data_loss_semantics(kind, space=str(data_loss["space"]))
    except ValueError as error:
        msg = f"loss.data: {error}"
        raise ConfigError(msg) from error
    weight = float(data_loss["weight"])
    if weight < 0:
        msg = f"loss.data.weight must be non-negative, got {weight}."
        raise ConfigError(msg)
    data_loss["weight"] = weight

    physics = _as_mapping(loss["physics"], path="loss.physics")
    if not isinstance(physics["enabled"], bool):
        msg = "loss.physics.enabled must be a boolean."
        raise ConfigError(msg)
    selected_physics = domain.tasks.registry.resolve_physics(task.physics.kind)
    if selected_physics != task.physics:
        msg = f"Task {task.id!r} physics registry entry does not match its task contract."
        raise ConfigError(msg)
    continuity = physics.get("continuity")
    if not isinstance(continuity, str) or not continuity:
        msg = "loss.physics.continuity must be a non-empty semantic identifier."
        raise ConfigError(msg)
    if continuity not in selected_physics.allowed_continuities:
        available = ", ".join(selected_physics.allowed_continuities)
        msg = (
            f"Unknown continuity identifier {continuity!r} at loss.physics.continuity "
            f"for task {task.id!r}. Available continuity formulations: {available}."
        )
        raise ConfigError(msg)
    try:
        domain.physics.contracts.validate_continuity_kind(continuity)
    except ValueError as error:
        msg = f"loss.physics.continuity: {error}"
        raise ConfigError(msg) from error
    physics["continuity"] = continuity
    derivatives = _as_mapping(physics["derivatives"], path="loss.physics.derivatives")
    try:
        domain.physics.contracts.validate_derivative_kind(
            str(derivatives["kind"]),
            extension=str(derivatives["extension"]),
        )
    except ValueError as error:
        msg = f"loss.physics.derivatives: {error}"
        raise ConfigError(msg) from error
    interior_crop = int(physics["interior_crop"])
    if interior_crop < 0:
        msg = f"loss.physics.interior_crop must be non-negative, got {interior_crop}."
        raise ConfigError(msg)
    physics["interior_crop"] = interior_crop

    for weight_name in ("residual_weight", "boundary_weight"):
        weight_config = _as_mapping(physics[weight_name], path=f"loss.physics.{weight_name}")
        target = float(weight_config["target"])
        if target < 0:
            msg = f"loss.physics.{weight_name}.target must be non-negative, got {target}."
            raise ConfigError(msg)
        weight_config["target"] = target
        warmup = _as_mapping(weight_config["warmup"], path=f"loss.physics.{weight_name}.warmup")
        if warmup["kind"] != "linear":
            msg = f"Unknown warmup identifier {warmup['kind']!r} at loss.physics.{weight_name}.warmup.kind; expected 'linear'."
            raise ConfigError(msg)
        epochs = int(warmup["epochs"])
        if epochs < 0:
            msg = f"loss.physics.{weight_name}.warmup.epochs must be non-negative, got {epochs}."
            raise ConfigError(msg)
        warmup["epochs"] = epochs
        weight_config["warmup"] = warmup
        physics[weight_name] = weight_config

    loss["data"] = data_loss
    loss["physics"] = physics
    config["loss"] = loss


def _metric_fields(
    metric: dict[str, Any],
    *,
    task: domain.tasks.spec.TaskSpec,
    path: str,
) -> tuple[str, ...]:
    """
    Validate and canonicalize one metric's output-field selection.

    ``all`` expands in exact TaskSpec output order. Empty, duplicate, or unknown
    fields raise ``ConfigError`` at ``path``.
    """
    raw_fields = metric.get("fields", "all")
    if raw_fields == "all":
        fields = task.output_names
        metric["fields"] = list(fields)
        return fields
    if isinstance(raw_fields, (str, bytes)) or not isinstance(raw_fields, Sequence):
        msg = f"{path}.fields must be 'all' or a non-empty list of output fields."
        raise ConfigError(msg)
    fields = tuple(str(field) for field in raw_fields)
    if not fields:
        msg = f"{path}.fields must not be empty."
        raise ConfigError(msg)
    if len(fields) != len(set(fields)):
        msg = f"{path}.fields contains duplicates: {list(fields)}."
        raise ConfigError(msg)
    unknown = [field for field in fields if field not in task.output_names]
    if unknown:
        msg = f"{path}.fields references unknown task output field(s): {unknown}. Available outputs: {list(task.output_names)}."
        raise ConfigError(msg)
    metric["fields"] = list(fields)
    return fields


def _validate_resolved_metric_keys(config: Mapping[str, Any]) -> None:
    """
    Require every resolved metric to use the exact six-field canonical schema.

    This fail-closed pass prevents raw-only aliases or omitted direction from
    entering saved configs before metric/objective revalidation.
    """
    evaluation = _as_mapping(config.get("evaluation"), path="evaluation")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        return
    expected_keys = {"id", "kind", "space", "fields", "reduction", "direction"}
    for index, raw_metric in enumerate(metrics):
        path = f"evaluation.metrics[{index}]"
        metric = _as_mapping(raw_metric, path=path)
        if set(metric) != expected_keys:
            msg = f"Resolved {path} must contain exactly {sorted(expected_keys)}, got {sorted(metric)}."
            raise ConfigError(msg)


def _validate_evaluation(config: dict[str, Any], *, task: domain.tasks.spec.TaskSpec) -> None:
    """
    Resolve metric declarations and materialize one complete objective in place.

    Each metric is bound to an exact tensor space, ordered field set, reduction,
    direction, and unit-compatible task contract. The selected objective becomes
    a full copy of exactly one declared metric; partial or contradictory resolved
    objective mappings fail closed.
    """
    evaluation = _as_mapping(config["evaluation"], path="evaluation")
    raw_metrics = evaluation["metrics"]
    if not isinstance(raw_metrics, list) or not raw_metrics:
        msg = "evaluation.metrics must be a non-empty list."
        raise ConfigError(msg)

    metrics: list[dict[str, Any]] = []
    metric_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_metric in enumerate(raw_metrics):
        path = f"evaluation.metrics[{index}]"
        metric = _as_mapping(raw_metric, path=path)
        metric_id = metric.get("id")
        if not isinstance(metric_id, str) or not metric_id:
            msg = f"{path}.id must be a non-empty string."
            raise ConfigError(msg)
        if metric_id in metric_by_id:
            msg = f"Duplicate evaluation metric id {metric_id!r} at {path}."
            raise ConfigError(msg)
        kind = str(metric.get("kind"))
        space = str(metric.get("space"))
        reduction = str(metric.get("reduction"))
        try:
            metric_kind = metric_registry.validate_metric_semantics(
                kind,
                space=space,
                reduction=reduction,
            )
        except ValueError as error:
            msg = f"{path}: {error}"
            raise ConfigError(msg) from error
        fields = _metric_fields(metric, task=task, path=path)
        if kind == "macro_rmse" and fields != task.output_names:
            msg = f"{path} macro_rmse must select every TaskSpec output field in declared order: {list(task.output_names)}."
            raise ConfigError(msg)
        if space == "physical" and len(fields) != 1:
            units = {task.field(field).unit for field in fields}
            if len(units) > 1:
                msg = f"{path} cannot aggregate physical fields with incompatible units: {sorted(units)}."
                raise ConfigError(msg)
            msg = f"{path} physical metrics must select exactly one output field."
            raise ConfigError(msg)
        requested_direction = metric.get("direction", metric_kind.direction)
        if requested_direction != metric_kind.direction:
            msg = f"{path}.direction {requested_direction!r} contradicts metric {kind!r} direction {metric_kind.direction!r}."
            raise ConfigError(msg)
        metric["direction"] = metric_kind.direction
        metrics.append(metric)
        metric_by_id[metric_id] = metric

    selection = _as_mapping(evaluation["objective"], path="evaluation.objective")
    objective_id = selection.get("id")
    if not isinstance(objective_id, str) or not objective_id:
        msg = "evaluation.objective.id must be a non-empty metric identifier."
        raise ConfigError(msg)
    if objective_id not in metric_by_id:
        msg = f"evaluation.objective.id {objective_id!r} is not a declared metric id. Available ids: {sorted(metric_by_id)}."
        raise ConfigError(msg)

    selected = metric_by_id[objective_id]
    objective = {
        "id": selected["id"],
        "kind": selected["kind"],
        "space": selected["space"],
        "fields": copy.deepcopy(selected["fields"]),
        "reduction": selected["reduction"],
        "direction": selected["direction"],
    }
    selection_keys = set(selection)
    if selection_keys != {"id"} and selection != objective:
        msg = f"Resolved evaluation.objective must exactly equal its selected metric definition; expected {objective!r}, got {selection!r}."
        raise ConfigError(msg)

    evaluation["metrics"] = metrics
    evaluation["objective"] = objective
    config["evaluation"] = evaluation


def get_resolved_objective(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the complete canonical objective from a resolved experiment config.

    Parameters
    ----------
    config : Mapping[str, Any]
        Candidate resolved config containing ``evaluation.objective`` and the
        corresponding metric declaration.

    Returns
    -------
    dict[str, Any]
        Isolated id, kind, space, ordered fields, reduction, and direction.

    Raises
    ------
    ConfigError
        If the objective is partial, unsupported, or inconsistent with metrics.

    """
    evaluation = _as_mapping(config.get("evaluation"), path="evaluation")
    objective = _as_mapping(evaluation.get("objective"), path="evaluation.objective")
    expected_keys = {"id", "kind", "space", "fields", "reduction", "direction"}
    if set(objective) != expected_keys:
        msg = f"Resolved evaluation.objective must contain exactly {sorted(expected_keys)}, got {sorted(objective)}."
        raise ConfigError(msg)
    for key in ("id", "kind", "space", "reduction"):
        if not isinstance(objective[key], str) or not objective[key]:
            msg = f"Resolved evaluation.objective.{key} must be a non-empty string."
            raise ConfigError(msg)
    fields = objective["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) and field for field in fields):
        msg = "Resolved evaluation.objective.fields must be a non-empty exact field list."
        raise ConfigError(msg)
    if len(fields) != len(set(fields)):
        msg = f"Resolved evaluation.objective.fields contains duplicates: {fields!r}."
        raise ConfigError(msg)
    if objective["direction"] not in {"minimize", "maximize"}:
        msg = "Resolved evaluation.objective.direction must be 'minimize' or 'maximize'."
        raise ConfigError(msg)

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        msg = "Resolved evaluation.metrics must be a list."
        raise ConfigError(msg)
    selected = [metric for metric in metrics if isinstance(metric, Mapping) and metric.get("id") == objective["id"]]
    if len(selected) != 1:
        msg = f"Resolved objective id {objective['id']!r} must select exactly one evaluation metric."
        raise ConfigError(msg)
    selected_objective = {key: copy.deepcopy(selected[0].get(key)) for key in expected_keys}
    if selected_objective != objective:
        msg = "Resolved evaluation.objective does not exactly match its evaluation metric definition."
        raise ConfigError(msg)
    return copy.deepcopy(objective)


def _single_ood_dataset(data: Mapping[str, Any], *, path: str) -> str:
    """Return the sole current-contract OOD dataset identifier."""
    value = data.get("ood_datasets")
    if not isinstance(value, list) or len(value) != 1:
        msg = f"{path}.ood_datasets must contain exactly one logical dataset id."
        raise ConfigError(msg)
    dataset_id = value[0]
    try:
        return common.paths.validate_logical_name(dataset_id, label=f"{path}.ood_datasets[0]")
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _model_variant(config: Mapping[str, Any]) -> str:
    """Return the exact human-facing network/loss variant."""
    model = _as_mapping(config.get("model"), path="model")
    model_kind = common.paths.validate_logical_name(model.get("kind"), label="model.kind")
    loss = _as_mapping(config.get("loss"), path="loss")
    physics = _as_mapping(loss.get("physics"), path="loss.physics")
    return f"pi-{model_kind}" if bool(physics.get("enabled")) else model_kind


def resolved_model_variant(config: Mapping[str, Any]) -> str:
    """Return the loader-owned architecture and PI identity segment."""
    task = str(config["task"])
    model = _as_mapping(config.get("model"), path="model")
    kind = str(model.get("kind"))
    params = _as_mapping(model.get("params"), path="model.params")
    variant = _model_variant(config)
    if kind == "fno":
        modes = params["n_modes"]
        return f"{variant}_m{modes[0]}x{modes[1]}_h{params['hidden_channels']}_l{params['n_layers']}"
    if kind == "uno":
        mode_ratio = format(float(params["mode_ratio"]), ".8g").replace(".", "p")
        return f"{variant}_m{params['modes_x']}x{params['modes_y']}_h{params['hidden_channels']}_l{params['n_layers']}_r{mode_ratio}"
    domain.tasks.registry.get_task(task)
    msg = f"Unknown model identifier {kind!r} while generating a run name."
    raise ConfigError(msg)


_RUN_SUFFIX_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")


def resolved_scientific_variant(config: Mapping[str, Any]) -> str | None:
    """Return the PI derivative/continuity identity from resolved physics."""
    loss = _as_mapping(config.get("loss"), path="loss")
    physics = _as_mapping(loss.get("physics"), path="loss.physics")
    enabled = physics.get("enabled")
    if type(enabled) is not bool:
        msg = "Resolved loss.physics.enabled must be boolean while deriving scientific run identity."
        raise ConfigError(msg)
    if not enabled:
        return None

    derivatives = _as_mapping(physics.get("derivatives"), path="loss.physics.derivatives")
    raw_kind = derivatives.get("kind")
    raw_extension = derivatives.get("extension")
    continuity = physics.get("continuity")
    if not isinstance(raw_kind, str) or not isinstance(raw_extension, str):
        msg = "PI run identity requires resolved derivative kind and extension identifiers."
        raise ConfigError(msg)
    if not isinstance(continuity, str) or not continuity:
        msg = "PI run identity requires a resolved continuity identifier."
        raise ConfigError(msg)
    try:
        kind, extension = domain.physics.contracts.validate_derivative_kind(
            raw_kind,
            extension=raw_extension,
        )
        resolved_continuity = domain.physics.contracts.validate_continuity_kind(continuity)
    except ValueError as error:
        msg = f"Cannot derive canonical PI scientific variant: {error}"
        raise ConfigError(msg) from error

    strategies = {
        ("physical", "none"): "physical",
        ("spectral", "reflect"): "spectral_reflect",
    }
    strategy = strategies.get((kind, extension))
    if strategy is None:
        msg = f"Unsupported resolved PI derivative strategy for run identity: kind={kind!r}, extension={extension!r}."
        raise ConfigError(msg)
    return f"{strategy}_{resolved_continuity}"


def _identity_tokens(value: object) -> tuple[str, ...]:
    """Split one derived identity at semantic token boundaries."""
    return tuple(token for token in str(value).lower().replace("-", "_").split("_") if token)


def _remove_identity_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    """Remove every exact token-bounded occurrence of one derived component."""
    found = False
    index = 0
    while sequence and index <= len(tokens) - len(sequence):
        if tuple(tokens[index : index + len(sequence)]) == sequence:
            del tokens[index : index + len(sequence)]
            found = True
        else:
            index += 1
    return found


def _derived_suffix_components(
    config: Mapping[str, Any],
    *,
    model_key: str,
    scientific_variant: str | None,
) -> list[tuple[str, str]]:
    """Return token-aware identities forbidden in manual experiment context."""
    model = _as_mapping(config.get("model"), path="model")
    run = _as_mapping(config.get("run"), path="run")
    components = [
        ("model architecture", model_key),
        ("task", str(config.get("task"))),
        ("model family", _model_variant(config)),
        ("model kind", str(model.get("kind"))),
        ("seed", f"s{run.get('seed')}"),
        ("seed", f"seed{run.get('seed')}"),
    ]
    components.extend(("architecture parameter", token) for token in model_key.split("_")[1:])
    if scientific_variant is not None:
        physics = _as_mapping(_as_mapping(config.get("loss"), path="loss").get("physics"), path="loss.physics")
        derivatives = _as_mapping(physics.get("derivatives"), path="loss.physics.derivatives")
        continuity = str(physics.get("continuity"))
        derivative_strategy = scientific_variant.removesuffix(f"_{continuity}")
        components.extend(
            [
                ("scientific variant", scientific_variant),
                ("derivative strategy", derivative_strategy),
                ("derivative kind", str(derivatives.get("kind"))),
                ("continuity formulation", continuity),
            ]
        )
        extension = str(derivatives.get("extension"))
        if extension != "none":
            components.append(("derivative extension", extension))

    unique: dict[tuple[str, ...], tuple[str, str]] = {}
    for label, value in components:
        tokens = _identity_tokens(value)
        if tokens:
            unique.setdefault(tokens, (label, value))
    return sorted(unique.values(), key=lambda item: len(_identity_tokens(item[1])), reverse=True)


def _validate_run_context(
    config: Mapping[str, Any],
    *,
    model_key: str | None = None,
    scientific_variant: str | None = None,
) -> None:
    """Require normalized suffix-only context without repeated derived identity."""
    run = _as_mapping(config.get("run"), path="run")
    suffix = run.get("suffix")
    if suffix is None:
        return
    if not isinstance(suffix, str) or not suffix:
        msg = f"run.suffix must be null or a non-empty normalized string, got {suffix!r}."
        raise ConfigError(msg)

    resolved_model_key = model_key or resolved_model_variant(config)
    resolved_science = resolved_scientific_variant(config) if scientific_variant is None else scientific_variant
    remaining = list(_identity_tokens(suffix))
    duplicated: list[tuple[str, str]] = []
    for label, value in _derived_suffix_components(
        config,
        model_key=resolved_model_key,
        scientific_variant=resolved_science,
    ):
        if _remove_identity_sequence(remaining, _identity_tokens(value)):
            duplicated.append((label, value))
    if duplicated:
        details = ", ".join(f"{label}={value!r}" for label, value in duplicated)
        suggestion = "_".join(remaining) or "null"
        msg = f"run.suffix {suffix!r} duplicates canonical derived identity ({details}). Suggested run.suffix: {suggestion}."
        raise ConfigError(msg)
    if _RUN_SUFFIX_PATTERN.fullmatch(suffix) is None:
        msg = f"run.suffix {suffix!r} must be lowercase underscore-separated tokens with no leading, trailing, or repeated underscore."
        raise ConfigError(msg)


def derive_wandb_organization(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the canonical project, optional study, and minimal base tags."""
    tracking = _as_mapping(config.get("tracking"), path="tracking")
    wandb = _as_mapping(tracking.get("wandb"), path="tracking.wandb")
    workflow = wandb.get("workflow")
    if workflow not in config_defaults.WANDB_WORKFLOWS:
        msg = f"tracking.wandb.workflow must be one of {list(config_defaults.WANDB_WORKFLOWS)}, got {workflow!r}."
        raise ConfigError(msg)

    task = common.paths.validate_logical_name(config.get("task"), label="task")
    variant = _model_variant(config)
    raw_study = wandb.get("study")
    if raw_study is not None:
        try:
            study = common.paths.validate_logical_name(raw_study, label="tracking.wandb.study")
        except ValueError as error:
            raise ConfigError(str(error)) from error
    else:
        study = None

    if workflow == "train":
        if study is not None:
            msg = "tracking.wandb.study is valid only for workflow='optuna_trial'."
            raise ConfigError(msg)
        tags = [variant]
    elif workflow == "optuna_trial":
        if study is None:
            msg = "tracking.wandb.study is required for workflow='optuna_trial'."
            raise ConfigError(msg)
        tags = [variant, "optuna"]
    else:
        if study is not None:
            msg = "Acceptance workflows do not accept tracking.wandb.study."
            raise ConfigError(msg)
        tags = []

    if len(tags) > config_defaults.WANDB_MAX_TAGS or len(tags) != len(set(tags)):
        msg = f"Derived W&B tags must be unique and contain at most {config_defaults.WANDB_MAX_TAGS} values."
        raise ConfigError(msg)
    return {
        "project": config_defaults.WANDB_TASK_PROJECTS[task],
        "entity": config_defaults.WANDB_ENTITY,
        "study": study,
        "tags": tags,
    }


def _validate_tracking(config: dict[str, Any], *, require_derived: bool) -> None:
    """Validate one central W&B mode and canonical derived organization."""
    tracking = _as_mapping(config["tracking"], path="tracking")
    _reject_unknown(tracking, frozenset({"wandb"}), path="tracking")
    wandb = _as_mapping(tracking.get("wandb"), path="tracking.wandb")
    _reject_unknown(
        wandb,
        frozenset(
            {
                "mode",
                "workflow",
                "study",
                "project",
                "entity",
                "tags",
                "monitor",
                "upload",
            }
        ),
        path="tracking.wandb",
    )
    mode = wandb.get("mode")
    if mode not in {"online", "offline", "disabled"}:
        msg = "tracking.wandb.mode must be 'online', 'offline', or 'disabled'."
        raise ConfigError(msg)

    monitor = _as_mapping(wandb.get("monitor"), path="tracking.wandb.monitor")
    _reject_unknown(monitor, frozenset({"enabled", "interval", "max_cases"}), path="tracking.wandb.monitor")
    if type(monitor.get("enabled")) is not bool:
        msg = "tracking.wandb.monitor.enabled must be boolean."
        raise ConfigError(msg)
    for key in ("interval", "max_cases"):
        value = monitor.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            msg = f"tracking.wandb.monitor.{key} must be a positive integer."
            raise ConfigError(msg)

    upload = _as_mapping(wandb.get("upload"), path="tracking.wandb.upload")
    _reject_unknown(upload, frozenset({"evaluation_artifacts"}), path="tracking.wandb.upload")
    if type(upload.get("evaluation_artifacts")) is not bool:
        msg = "tracking.wandb.upload.evaluation_artifacts must be boolean."
        raise ConfigError(msg)

    expected = derive_wandb_organization({**config, "tracking": {"wandb": wandb}})
    if require_derived:
        differences = sorted(key for key, value in expected.items() if wandb.get(key) != value)
        if differences:
            msg = f"Resolved W&B organization is noncanonical at key(s): {differences}."
            raise ConfigError(msg)
    else:
        wandb.update(expected)
    wandb["monitor"] = monitor
    wandb["upload"] = upload
    tracking["wandb"] = wandb
    config["tracking"] = tracking


def _validate_runtime_sections(config: dict[str, Any], *, require_derived_tracking: bool) -> None:
    """
    Validate all generic runtime sections after semantic resolution.

    This final pass checks optimizer/scheduler identifiers, positive duration,
    mixed-precision type, logical dataset names, optional tracking policy, and
    the exact requested device vocabulary. It validates policy only: concrete
    device resolution remains a top-level execution-service responsibility.
    """
    optimizer = _as_mapping(config["optimizer"], path="optimizer")
    if optimizer["kind"] != "adamw":
        msg = f"Unknown optimizer identifier {optimizer['kind']!r}. Available optimizers: adamw."
        raise ConfigError(msg)
    if "lr" not in optimizer:
        msg = "optimizer.lr is required."
        raise ConfigError(msg)
    betas = optimizer["betas"]
    if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence) or len(betas) != _ADAM_BETA_COUNT:
        msg = f"optimizer.betas must contain exactly two values, got {betas!r}."
        raise ConfigError(msg)

    scheduler = config.get("scheduler")
    if scheduler is not None:
        scheduler_mapping = _as_mapping(scheduler, path="scheduler")
        if scheduler_mapping["kind"] != "reduce_on_plateau":
            msg = f"Unknown scheduler identifier {scheduler_mapping['kind']!r}. Available schedulers: reduce_on_plateau."
            raise ConfigError(msg)

    training = _as_mapping(config["training"], path="training")
    if type(training["mixed_precision"]) is not bool:
        msg = f"training.mixed_precision must be boolean, got {training['mixed_precision']!r}."
        raise ConfigError(msg)
    if int(training["epochs"]) <= 0:
        msg = "training.epochs must be positive."
        raise ConfigError(msg)
    if int(training["evaluation_interval"]) <= 0:
        msg = "training.evaluation_interval must be positive."
        raise ConfigError(msg)
    if int(training["ood_evaluation_interval"]) <= 0:
        msg = "training.ood_evaluation_interval must be positive."
        raise ConfigError(msg)
    data = _as_mapping(config["data"], path="data")
    try:
        common.paths.validate_logical_name(data["train_dataset"], label="data.train_dataset")
    except ValueError as error:
        raise ConfigError(str(error)) from error
    _single_ood_dataset(data, path="data")

    _validate_tracking(config, require_derived=require_derived_tracking)

    run = _as_mapping(config["run"], path="run")
    try:
        run["device"] = learning_device_policy.validate_device_policy(
            run.get("device"),
            path="run.device",
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    for key in ("suffix", "name"):
        value = run.get(key)
        if value is None:
            continue
        try:
            common.paths.validate_logical_name(value, label=f"run.{key}")
        except ValueError as error:
            raise ConfigError(str(error)) from error
    _validate_run_context(config)


def generate_run_name(config: dict[str, Any]) -> str:
    """
    Generate a descriptive run name from semantic model and loss settings.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved experiment configuration.

    Returns
    -------
    str
        Deterministic architecture/science/seed name with an optional final
        experiment suffix.

    Raises
    ------
    ConfigError
        If required semantic model or scientific settings are invalid.

    """
    run = _as_mapping(config["run"], path="run")
    model_key = resolved_model_variant(config)
    scientific_variant = resolved_scientific_variant(config)
    _validate_run_context(
        config,
        model_key=model_key,
        scientific_variant=scientific_variant,
    )

    parts = [model_key]
    if scientific_variant is not None:
        parts.append(scientific_variant)
    parts.append(f"s{run['seed']}")
    if run.get("suffix"):
        parts.append(str(run["suffix"]))
    return "__".join(parts)


def resolve_config(user_config: dict[str, Any]) -> dict[str, Any]:
    """
    Strictly resolve one semantic experiment configuration.

    Parameters
    ----------
    user_config : dict[str, Any]
        Raw semantic experiment mapping.

    Returns
    -------
    dict[str, Any]
        Fully resolved configuration with task contract, digest, channels, and paths.

    Raises
    ------
    ConfigError
        If the schema, identifiers, fields, or settings violate the contract.
    ValueError
        If a referenced semantic identifier is not registered.

    """
    if not isinstance(user_config, Mapping):
        msg = "Experiment config must be a mapping."
        raise ConfigError(msg)
    _validate_input_schema(user_config)
    task_id = user_config.get("task")
    if not isinstance(task_id, str) or not task_id:
        msg = "Missing required non-empty config task identifier."
        raise ConfigError(msg)
    try:
        task = domain.tasks.registry.get_task(task_id)
    except ValueError as error:
        msg = f"config.task: {error}"
        raise ConfigError(msg) from error

    effective = deep_merge(config_defaults.get_task_defaults(task_id), user_config)
    effective["task"] = task_id

    model = _as_mapping(effective["model"], path="model")
    kind = model.get("kind")
    if not isinstance(kind, str):
        msg = "model.kind is required and must be a semantic string identifier."
        raise ConfigError(msg)
    params = _as_mapping(model.get("params"), path="model.params")
    try:
        params = deep_merge(model_factory.model_defaults(kind), params)
        model_factory.validate_model_params(
            kind,
            params,
            require_channels=False,
            operator_dimensionality=task.operator_dimensionality,
        )
    except ValueError as error:
        msg = f"model: {error}"
        raise ConfigError(msg) from error
    params["in_channels"] = task.in_channels
    params["out_channels"] = task.out_channels
    model["params"] = params
    effective["model"] = model

    optimizer = _as_mapping(effective["optimizer"], path="optimizer")
    optimizer_kind = str(optimizer.get("kind"))
    if optimizer_kind not in config_defaults.OPTIMIZER_DEFAULTS:
        msg = f"Unknown optimizer identifier {optimizer_kind!r}. Available optimizers: {sorted(config_defaults.OPTIMIZER_DEFAULTS)}."
        raise ConfigError(msg)
    effective["optimizer"] = deep_merge(config_defaults.OPTIMIZER_DEFAULTS[optimizer_kind], optimizer)

    scheduler = effective.get("scheduler")
    if scheduler is not None:
        scheduler_mapping = _as_mapping(scheduler, path="scheduler")
        scheduler_kind = str(scheduler_mapping.get("kind"))
        if scheduler_kind not in config_defaults.SCHEDULER_DEFAULTS:
            msg = f"Unknown scheduler identifier {scheduler_kind!r}. Available schedulers: {sorted(config_defaults.SCHEDULER_DEFAULTS)}."
            raise ConfigError(msg)
        effective["scheduler"] = deep_merge(config_defaults.SCHEDULER_DEFAULTS[scheduler_kind], scheduler_mapping)

    _validate_loss(effective, task=task)
    _validate_evaluation(effective, task=task)
    _validate_runtime_sections(effective, require_derived_tracking=False)

    effective["task_contract"] = task.resolved_contract()
    effective["paths"] = {
        "project_root": str(common.paths.get_project_root()),
        "model_training_data_root": str(common.paths.get_model_training_data_root()),
        "training_meta_root": str(common.paths.get_training_meta_root()),
        "dataset_root": str(common.paths.get_training_raw_root()),
        "output_root": str(common.paths.get_training_processed_root()),
    }
    run = _as_mapping(effective["run"], path="run")
    run["name"] = generate_run_name(effective)
    try:
        common.paths.validate_logical_name(run["name"], label="run.name")
    except ValueError as error:
        raise ConfigError(str(error)) from error
    effective["run"] = run
    return effective


def validate_resolved_task_contract(config: Mapping[str, Any]) -> domain.tasks.spec.TaskSpec:
    """
    Validate the persisted task contract in an effective configuration.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved or saved semantic configuration.

    Returns
    -------
    domain.tasks.spec.TaskSpec
        Registered task matching the saved identifier and digest.

    Raises
    ------
    ConfigError
        If the complete task contract does not exactly match the registered task.
    ValueError
        If the task identifier is unknown.

    """
    task_id = config.get("task")
    if not isinstance(task_id, str):
        msg = "Resolved config must contain a string task identifier."
        raise ConfigError(msg)
    task = domain.tasks.registry.get_task(task_id)
    contract = config.get("task_contract")
    if not isinstance(contract, Mapping):
        msg = "Resolved config must contain the current task_contract."
        raise ConfigError(msg)
    expected_contract = task.resolved_contract()
    schema_version = contract.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != domain.tasks.spec.TASK_SCHEMA_VERSION:
        msg = (
            f"Resolved task contract does not exactly match registered task {task_id!r}: "
            f"schema_version must be integer {domain.tasks.spec.TASK_SCHEMA_VERSION}."
        )
        raise ConfigError(msg)
    if dict(contract) != expected_contract:
        msg = f"Resolved task contract does not exactly match registered task {task_id!r}."
        raise ConfigError(msg)
    return task


def validate_resolved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate and return an isolated canonical resolved experiment config.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved candidate, commonly loaded from a saved ``config.yaml``.

    Returns
    -------
    dict[str, Any]
        Deep-copied canonical config whose task contract, paths, model, loss,
        metrics, objective, runtime sections, and tracking policy all agree.

    Raises
    ------
    ConfigError
        If required derived values are absent or any semantic section drifts.

    Notes
    -----
    Validation is read-only: it allocates no run, dataset loader, tracker, or
    device and never rewrites the supplied mapping.

    """
    if not isinstance(config, Mapping):
        msg = "Resolved experiment config must be a mapping."
        raise ConfigError(msg)
    effective = copy.deepcopy(dict(config))
    allowed = _ROOT_KEYS.union({"task_contract", "paths"})
    missing = sorted(allowed.difference(effective))
    unknown = sorted(set(effective).difference(allowed))
    if missing or unknown:
        msg = f"Resolved config keys do not match. Missing: {missing}; unknown: {unknown}."
        raise ConfigError(msg)

    task = validate_resolved_task_contract(effective)
    model = _as_mapping(effective["model"], path="model")
    kind = model.get("kind")
    if not isinstance(kind, str) or not kind:
        msg = "Resolved model.kind must be a non-empty semantic identifier."
        raise ConfigError(msg)
    params = _as_mapping(model.get("params"), path="model.params")
    try:
        model_factory.validate_model_params(
            kind,
            params,
            require_channels=True,
            operator_dimensionality=task.operator_dimensionality,
        )
    except ValueError as error:
        msg = f"model: {error}"
        raise ConfigError(msg) from error
    if params.get("in_channels") != task.in_channels or params.get("out_channels") != task.out_channels:
        msg = "Resolved model channels do not match the task contract."
        raise ConfigError(msg)
    model["params"] = params
    effective["model"] = model

    _validate_loss(effective, task=task)
    _validate_resolved_metric_keys(effective)
    _validate_evaluation(effective, task=task)
    _validate_runtime_sections(effective, require_derived_tracking=True)
    get_resolved_objective(effective)
    expected_run_name = generate_run_name(effective)
    if effective["run"]["name"] != expected_run_name:
        msg = f"Resolved run.name must equal the canonical generated leaf {expected_run_name!r}."
        raise ConfigError(msg)
    paths = _as_mapping(effective["paths"], path="paths")
    missing_paths = sorted(_RESOLVED_PATH_KEYS.difference(paths))
    unknown_paths = sorted(set(paths).difference(_RESOLVED_PATH_KEYS))
    if missing_paths or unknown_paths:
        msg = f"Resolved paths do not match the two-domain contract. Missing: {missing_paths}; unknown: {unknown_paths}."
        raise ConfigError(msg)
    invalid_paths = sorted(key for key, value in paths.items() if not isinstance(value, str) or not value)
    if invalid_paths:
        msg = f"Resolved paths must contain non-empty strings; invalid key(s): {invalid_paths}."
        raise ConfigError(msg)
    effective["paths"] = paths
    return effective


def load_and_resolve_config(yaml_path: Path | str) -> dict[str, Any]:
    """
    Load and strictly resolve one experiment YAML.

    Parameters
    ----------
    yaml_path : Path or str
        Semantic experiment YAML path.

    Returns
    -------
    dict[str, Any]
        Fully resolved semantic configuration.

    """
    raw = load_yaml(yaml_path)
    resolved = resolve_config(raw)
    validate_task_directory_identity(
        yaml_path,
        raw_task=raw.get("task"),
        resolved_task=resolved.get("task"),
    )
    return resolved


def create_dataloaders_from_config(
    config: dict[str, Any],
    *,
    split_indices: dict[str, Any] | None = None,
    data_processor: Any | None = None,
    seed_plan: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """
    Create current dataloaders after validating the resolved task contract.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved experiment configuration.
    split_indices : dict[str, Any] or None, optional
        Existing split membership to reuse.
    data_processor : Any or None, optional
        Existing data processor passed through to the current loader.
    seed_plan : Mapping[str, int] | None, optional
        Stable labeled ``split``, ``loader``, and ``worker`` seeds. Defaults to
        ``run.seed`` for isolated direct loader callers.

    Returns
    -------
    dict[str, Any]
        Train/evaluation loaders, data processor, and resolved split membership.

    Raises
    ------
    ConfigError
        If the task contract or required data settings are invalid.

    Notes
    -----
    The shared task dataset factory validates schema and fingerprint before
    splitting. Loader construction may read dataset files, fit preprocessing
    state for a fresh run, or reuse caller-supplied split and processor state;
    persistence remains the run lifecycle's responsibility.

    """
    task = validate_resolved_task_contract(config)
    data_cfg = _as_mapping(config.get("data"), path="data")
    dataset_root = Path(config["paths"]["dataset_root"])

    train_dataset_name = common.paths.validate_logical_name(data_cfg["train_dataset"], label="data.train_dataset")
    ood_dataset_name = _single_ood_dataset(data_cfg, path="data")

    path_train = common.paths.resolve_dataset_path(train_dataset_name, dataset_root=dataset_root)
    path_test_ood = common.paths.resolve_dataset_path(ood_dataset_name, dataset_root=dataset_root)
    seeds = dict(seed_plan or {})
    run_seed = int(config["run"]["seed"])
    train_loader, test_loaders, normalizer, split_indices = datasets.base.create_dataloaders(
        dataset_factory=datasets.simulation.create_task_dataset,
        path_train=str(path_train),
        path_test_ood=str(path_test_ood),
        task=task,
        train_dataset_id=train_dataset_name,
        ood_dataset_id=ood_dataset_name,
        train_ratio=data_cfg["train_ratio"],
        ood_fraction=data_cfg["ood_fraction"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg["persistent_workers"],
        split_seed=seeds.get("split", run_seed),
        loader_seed=seeds.get("loader", run_seed),
        worker_seed=seeds.get("worker", run_seed),
        split_indices=split_indices,
        data_processor=data_processor,
    )
    eval_loader = test_loaders.get("eval")
    if eval_loader is None:
        msg = "No evaluation dataloader was created."
        raise ConfigError(msg)
    return {
        "train": train_loader,
        "eval": eval_loader,
        "ood": test_loaders["ood"],
        "data_processor": normalizer,
        "split_indices": split_indices,
    }
