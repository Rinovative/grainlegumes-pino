"""Discover task-first executable YAML fixtures by role and resolved intent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
TASKS_CONFIG_ROOT = _CONFIG_ROOT / "tasks"
_MIN_EXECUTABLE_PATH_PARTS = 2
_MIN_EXPERIMENT_PATH_PARTS = 4


def task_config_directories() -> tuple[Path, ...]:
    """Return every non-empty task directory without freezing task names or count."""
    directories = tuple(
        sorted(path for path in TASKS_CONFIG_ROOT.iterdir() if path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*.yaml")))
    )
    if not directories:
        message = f"Task-first config root contains no executable task directories: {TASKS_CONFIG_ROOT}"
        raise AssertionError(message)
    return directories


def _role_paths(*relative_parts: str) -> tuple[Path, ...]:
    """Discover one executable role beneath every current task directory."""
    paths: list[Path] = []
    role = "/".join(relative_parts)
    for task_dir in task_config_directories():
        role_root = task_dir.joinpath(*relative_parts)
        if not role_root.is_dir():
            message = f"Task-first config role directory is missing for {task_dir.name}: {role_root}"
            raise AssertionError(message)
        task_paths = tuple(sorted(path for path in role_root.rglob("*.yaml") if path.is_file()))
        if not task_paths:
            message = f"Task-first config role contains no YAML files for {task_dir.name}: {role}"
            raise AssertionError(message)
        paths.extend(task_paths)
    return tuple(sorted(paths))


def executable_config_paths() -> tuple[Path, ...]:
    """Discover every task-specific executable YAML across all maintained roles."""
    return tuple(
        sorted(
            {
                *experiment_config_paths(),
                *optuna_config_paths(),
                *acceptance_config_paths(),
            }
        )
    )


def experiment_config_paths() -> tuple[Path, ...]:
    """Discover every task-local normal experiment regardless of mutable subcategory."""
    return _role_paths("experiments")


def experiment_category(path: Path) -> str:
    """Return the required category below one task-local experiments directory."""
    relative = path.resolve().relative_to(TASKS_CONFIG_ROOT.resolve())
    if len(relative.parts) < _MIN_EXPERIMENT_PATH_PARTS or relative.parts[1] != "experiments":
        message = f"Normal experiment YAML must be below configs/tasks/<task>/experiments/<category>/: {path}"
        raise AssertionError(message)
    return relative.parts[2]


def acceptance_config_paths() -> tuple[Path, ...]:
    """Discover every task-local bounded acceptance recipe."""
    return _role_paths("acceptance")


def optuna_config_paths() -> tuple[Path, ...]:
    """Discover every task-local Optuna study wrapper."""
    return _role_paths("optuna")


def directory_task(path: Path) -> str:
    """Return the task directory component for one task-first executable path."""
    relative = path.resolve().relative_to(TASKS_CONFIG_ROOT.resolve())
    if len(relative.parts) < _MIN_EXECUTABLE_PATH_PARTS:
        message = f"Executable config path has no task/workflow components: {path}"
        raise ValueError(message)
    return relative.parts[0]


def executable_role(path: Path) -> str:
    """Return the directory-owned role for one task-first executable path."""
    relative = path.resolve().relative_to(TASKS_CONFIG_ROOT.resolve())
    role_parts = relative.parts[1:-1]
    if role_parts[:1] == ("experiments",):
        return "experiments"
    if role_parts[:1] == ("optuna",):
        return "optuna"
    if role_parts[:1] == ("acceptance",):
        return "acceptance"
    message = f"Unknown task-first executable role for {path}: {role_parts}"
    raise ValueError(message)


def _raw_mapping(path: Path) -> Mapping[str, Any]:
    """Load enough raw YAML structure to choose a representative semantic recipe."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        message = f"Experiment YAML root is not a mapping: {path}"
        raise TypeError(message)
    return payload


def experiment_config_path(*, model_kind: str, physics_enabled: bool) -> Path:
    """Return one deterministic normal recipe matching model and physics intent."""
    for path in experiment_config_paths():
        raw = _raw_mapping(path)
        model = raw.get("model")
        loss = raw.get("loss")
        if not isinstance(model, Mapping) or not isinstance(loss, Mapping):
            continue
        physics = loss.get("physics")
        if not isinstance(physics, Mapping):
            continue
        if model.get("kind") == model_kind and physics.get("enabled") is physics_enabled:
            return path
    message = f"No production experiment recipe matches model_kind={model_kind!r}, physics_enabled={physics_enabled!r}"
    raise AssertionError(message)


def optuna_config_path(*, model_kind: str, physics_enabled: bool, role: str = "production") -> Path:
    """Return one deterministic Optuna recipe matching model, physics, and study role."""
    for path in optuna_config_paths():
        raw = _raw_mapping(path)
        study = raw.get("study")
        experiment = raw.get("experiment")
        if not isinstance(study, Mapping) or not isinstance(experiment, Mapping):
            continue
        model = experiment.get("model")
        loss = experiment.get("loss")
        if not isinstance(model, Mapping) or not isinstance(loss, Mapping):
            continue
        physics = loss.get("physics")
        if not isinstance(physics, Mapping):
            continue
        if study.get("role") == role and model.get("kind") == model_kind and physics.get("enabled") is physics_enabled:
            return path
    message = f"No Optuna recipe matches model_kind={model_kind!r}, physics_enabled={physics_enabled!r}, role={role!r}"
    raise AssertionError(message)


def acceptance_config_path() -> Path:
    """Return one deterministic maintained acceptance recipe for generic config fixtures."""
    return acceptance_config_paths()[0]
