"""Classify and strictly validate executable config workflows without side effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src import common

from . import experiments_config_loader as loader

EXPERIMENT_FAMILY = "experiment"
OPTUNA_FAMILY = "optuna"
_WORKFLOW_FAMILY = {"train": EXPERIMENT_FAMILY, "optuna": OPTUNA_FAMILY}


@dataclass(frozen=True, slots=True)
class ConfigPreflight:
    """Describe one strictly validated executable config without runtime allocation."""

    family: str
    task: str
    model_kind: str
    physics_enabled: bool
    source_path: Path
    canonical_path: str


class WorkflowMismatchError(ValueError):
    """Report a valid config supplied to the wrong executable workflow."""

    def __init__(self, message: str, *, result: ConfigPreflight) -> None:
        """Initialize the mismatch with its validated supplied-config result."""
        super().__init__(message)
        self.result = result


def _canonical_config_path(path: Path) -> str:
    """Return a stable repository-relative config path when one is available."""
    source = path.expanduser().resolve()
    project_root = common.paths.get_project_root().expanduser().resolve()
    try:
        return source.relative_to(project_root).as_posix()
    except ValueError:
        return str(source)


def _classify_root(raw: dict[str, object]) -> str:
    """Classify strict schema ownership from parsed root keys, never path names."""
    from src.experiments.tuning import experiments_tuning_optuna as optuna  # noqa: PLC0415

    keys = set(raw)
    experiment_keys = keys.intersection(loader.EXPERIMENT_ROOT_KEYS)
    optuna_keys = keys.intersection(optuna.OPTUNA_ROOT_KEYS)
    if experiment_keys and optuna_keys:
        msg = (
            "Configuration root mixes normal experiment and Optuna wrapper sections; "
            f"experiment-owned={sorted(experiment_keys)}, optuna-owned={sorted(optuna_keys)}."
        )
        raise loader.ConfigError(msg)
    if optuna_keys:
        return OPTUNA_FAMILY
    if experiment_keys:
        return EXPERIMENT_FAMILY
    msg = f"Configuration family could not be identified from strict root ownership. Found root keys: {sorted(keys)}."
    raise loader.ConfigError(msg)


def inspect_config(path: Path | str) -> ConfigPreflight:
    """Classify and fully validate one config without GPU, queue, run, or study state."""
    source = Path(path).expanduser()
    raw = loader.load_yaml(source)
    family = _classify_root(raw)
    if family == EXPERIMENT_FAMILY:
        resolved = loader.load_and_resolve_config(source)
    else:
        from src.experiments.tuning import experiments_tuning_optuna as optuna  # noqa: PLC0415

        resolved = optuna.load_optuna_study_config(source).base_config
    model = resolved.get("model")
    loss = resolved.get("loss")
    if not isinstance(model, dict) or not isinstance(loss, dict):
        msg = "Resolved executable config must contain model and loss mappings."
        raise loader.ConfigError(msg)
    physics = loss.get("physics")
    if not isinstance(physics, dict) or type(physics.get("enabled")) is not bool:
        msg = "Resolved executable config must contain boolean loss.physics.enabled."
        raise loader.ConfigError(msg)
    task = resolved.get("task")
    model_kind = model.get("kind")
    if not isinstance(task, str) or not isinstance(model_kind, str):
        msg = "Resolved executable config must contain string task and model.kind values."
        raise loader.ConfigError(msg)
    return ConfigPreflight(
        family=family,
        task=task,
        model_kind=model_kind,
        physics_enabled=physics["enabled"],
        source_path=source.resolve(),
        canonical_path=_canonical_config_path(source),
    )


def matching_optuna_configs(result: ConfigPreflight) -> tuple[ConfigPreflight, ...]:
    """Return valid task/model-family wrappers matching one normal experiment."""
    if result.family != EXPERIMENT_FAMILY:
        return ()
    root = common.paths.get_project_root().expanduser().resolve() / "model_training/configs/tasks"
    candidates: list[ConfigPreflight] = []
    for path in sorted(root.glob(f"{result.task}/optuna/**/*.yaml")):
        try:
            candidate = inspect_config(path)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if (
            candidate.family == OPTUNA_FAMILY
            and candidate.task == result.task
            and candidate.model_kind == result.model_kind
            and candidate.physics_enabled is result.physics_enabled
        ):
            candidates.append(candidate)
    return tuple(candidates)


def _quoted(value: str) -> str:
    """Single-quote one shell argument without relying on caller interpolation."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def mismatch_message(result: ConfigPreflight, *, requested_workflow: str) -> str:
    """Build the actionable wrong-workflow message for a validated config."""
    if requested_workflow == "train":
        return "\n".join(
            (
                "Configuration workflow mismatch.",
                f"Supplied config family: {result.family}",
                "Requested workflow: train",
                "",
                "Use:",
                f"./scripts/docker_job.sh optuna {_quoted(result.canonical_path)}",
            )
        )
    lines = [
        "Configuration workflow mismatch.",
        f"Supplied config family: {result.family}",
        "Requested workflow: optuna",
        "",
        "This file is a normal training experiment.",
        "",
        "Train it with:",
        f"./scripts/docker_job.sh train {_quoted(result.canonical_path)}",
    ]
    matches = matching_optuna_configs(result)
    if len(matches) == 1:
        lines.extend(
            (
                "",
                "Matching Optuna study:",
                f"./scripts/docker_job.sh optuna {_quoted(matches[0].canonical_path)}",
            )
        )
    return "\n".join(lines)


def validate_workflow(path: Path | str, *, requested_workflow: str) -> ConfigPreflight:
    """Strictly validate one config and reject workflow-family mismatches."""
    if requested_workflow not in _WORKFLOW_FAMILY:
        msg = f"Unsupported config workflow preflight: {requested_workflow!r}."
        raise ValueError(msg)
    result = inspect_config(path)
    if result.family != _WORKFLOW_FAMILY[requested_workflow]:
        raise WorkflowMismatchError(
            mismatch_message(result, requested_workflow=requested_workflow),
            result=result,
        )
    return result
