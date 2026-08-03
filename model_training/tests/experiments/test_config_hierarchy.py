# ruff: noqa: S101
"""Protect the project-wide task-first executable-config hierarchy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import yaml
from src import experiments
from support import configs

if TYPE_CHECKING:
    from pathlib import Path

_REPOSITORY_ROOT = configs.TASKS_CONFIG_ROOT.parents[2]
_STEADY_FLOW_CATEGORIES = {"capacity_and_physics", "best_of_class", "model_selection"}
_FORBIDDEN_EXPERIMENT_CATEGORIES = {"low_capacity", "study", "studies", "optuna_study", "search"}
_MAINTAINED_PATH_SOURCES = (
    _REPOSITORY_ROOT / "README.md",
    _REPOSITORY_ROOT / "scripts/docker_job.sh",
    _REPOSITORY_ROOT / "model_training/notebooks/training_pipeline.ipynb",
)
_CONFIG_PATH_PATTERN = re.compile(r"model_training/configs/tasks/[A-Za-z0-9_./-]+\.yaml")


def test_dynamic_discovery_covers_every_task_first_executable_once() -> None:
    """Classify every YAML below configs/tasks without fixed tasks, names, or counts."""
    discovered = configs.executable_config_paths()
    actual = tuple(sorted(path for path in configs.TASKS_CONFIG_ROOT.rglob("*.yaml") if path.is_file()))

    assert discovered == actual
    assert len(discovered) == len(set(discovered))
    assert not any(path.is_symlink() for path in configs.TASKS_CONFIG_ROOT.rglob("*"))
    for path in discovered:
        role = configs.executable_role(path)
        relative = path.relative_to(configs.TASKS_CONFIG_ROOT)
        assert relative.parts[0] == configs.directory_task(path)
        assert relative.parts[1] == role
        if role == "experiments":
            assert configs.experiment_category(path) == relative.parts[2]


def test_steady_flow_experiment_categories_and_roles_are_semantically_distinct() -> None:
    """Protect the report-aligned categories without freezing files or counts."""
    paths = tuple(path for path in configs.experiment_config_paths() if configs.directory_task(path) == "steady_flow")
    by_category = {category: tuple(path for path in paths if configs.experiment_category(path) == category) for category in _STEADY_FLOW_CATEGORIES}

    assert {configs.experiment_category(path) for path in paths} == _STEADY_FLOW_CATEGORIES
    assert all(by_category.values())
    experiments_root = configs.TASKS_CONFIG_ROOT / "steady_flow/experiments"
    assert not any((experiments_root / category).exists() for category in _FORBIDDEN_EXPERIMENT_CATEGORIES)

    capacity = [experiments.config.loader.load_and_resolve_config(path) for path in by_category["capacity_and_physics"]]
    controlled_params = capacity[0]["model"]["params"]
    assert all(config["model"]["kind"] == "fno" for config in capacity)
    assert all(config["model"]["params"] == controlled_params for config in capacity)
    assert {config["loss"]["physics"]["enabled"] for config in capacity} == {False, True}
    for config in capacity:
        assert str(config["run"]["suffix"]).startswith("low_capacity")
        if config["loss"]["physics"]["enabled"]:
            assert config["loss"]["physics"]["derivatives"]["kind"] == "physical"
            assert config["loss"]["physics"]["continuity"] in {"div_velocity", "div_eps_velocity"}

    best_of_class = [experiments.config.loader.load_and_resolve_config(path) for path in by_category["best_of_class"]]
    assert all(config["run"]["suffix"] == "best_of_class" for config in best_of_class)

    for path in by_category["model_selection"]:
        raw = experiments.config.loader.load_yaml(path)
        resolved = experiments.config.loader.load_and_resolve_config(path)
        assert "study" not in raw
        assert resolved["tracking"]["wandb"]["workflow"] == "train"


def test_normal_experiment_discovery_rejects_missing_role_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail actionably instead of returning empty discovery for an invalid task tree."""
    tasks_root = tmp_path / "configs/tasks"
    acceptance = tasks_root / "sample_task/acceptance/smoke.yaml"
    acceptance.parent.mkdir(parents=True)
    acceptance.write_text("task: sample_task\n", encoding="utf-8")
    monkeypatch.setattr(configs, "TASKS_CONFIG_ROOT", tasks_root)

    with pytest.raises(AssertionError, match=r"role directory is missing.*experiments"):
        configs.experiment_config_paths()


def test_category_paths_do_not_change_resolved_experiment_semantics(tmp_path: Path) -> None:
    """Resolve identical bytes from different named categories with identical results."""
    source = configs.experiment_config_paths()[0]
    payload = source.read_bytes()
    destinations = (
        tmp_path / "configs/tasks/steady_flow/experiments/category_a/request.yaml",
        tmp_path / "configs/tasks/steady_flow/experiments/category_b/request.yaml",
    )
    for destination in destinations:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)

    resolved = [experiments.config.loader.load_and_resolve_config(destination) for destination in destinations]

    assert resolved[0] == resolved[1]
    assert resolved[0]["run"]["name"] == resolved[1]["run"]["name"]
    assert all(destination.read_bytes() == payload for destination in destinations)


def test_maintained_command_examples_reference_existing_categorized_configs() -> None:
    """Keep scripts, notebook commands, and documentation on canonical live paths."""
    referenced: set[str] = set()
    for source in _MAINTAINED_PATH_SOURCES:
        referenced.update(_CONFIG_PATH_PATTERN.findall(source.read_text(encoding="utf-8")))

    assert referenced
    for relative in referenced:
        path = _REPOSITORY_ROOT / relative
        assert path.is_file(), f"Maintained command references missing config: {relative}"
        if "/experiments/" in relative:
            assert configs.experiment_category(path) in _STEADY_FLOW_CATEGORIES
    assert not any("/experiments/low_capacity/" in relative for relative in referenced)


def test_directory_task_equals_raw_and_resolved_task_for_every_executable() -> None:
    """Require path ownership to agree with authoritative YAML and TaskSpec resolution."""
    optuna_paths = set(configs.optuna_config_paths())
    before = {path: path.read_bytes() for path in configs.executable_config_paths()}
    for path in configs.executable_config_paths():
        directory_task = configs.directory_task(path)
        raw = experiments.config.loader.load_yaml(path)
        if path in optuna_paths:
            resolved = experiments.tuning.optuna.load_optuna_study_config(path).base_config
            raw_task = raw["experiment"]["task"]
        else:
            resolved = experiments.config.loader.load_and_resolve_config(path)
            raw_task = raw["task"]
        assert raw_task == directory_task
        assert resolved["task"] == directory_task
    assert {path: path.read_bytes() for path in configs.executable_config_paths()} == before


def test_task_identity_is_not_repeated_in_task_local_filenames() -> None:
    """Keep the task-owned directory identity out of task-local filenames."""
    for path in configs.executable_config_paths():
        task = configs.directory_task(path)
        assert task not in path.stem


def test_no_task_specific_yaml_remains_in_workflow_first_roots() -> None:
    """Reject parallel old/new executable trees and fallback copies."""
    config_root = configs.TASKS_CONFIG_ROOT.parent
    for old_role in ("experiments", "optuna", "acceptance"):
        old_root = config_root / old_role
        assert not old_root.exists() or not tuple(old_root.rglob("*.yaml"))


@pytest.mark.parametrize("family", ["experiment", "optuna"])
def test_task_path_mismatch_is_rejected_actionably(tmp_path: Path, family: str) -> None:
    """Reject raw task identity that disagrees with configs/tasks/<directory_task>."""
    if family == "experiment":
        source = configs.experiment_config_paths()[0]
        raw = experiments.config.loader.load_yaml(source)
        destination = tmp_path / "configs/tasks/not_the_task/experiments/best_of_class/request.yaml"
    else:
        source = configs.optuna_config_paths()[0]
        raw = experiments.config.loader.load_yaml(source)
        destination = tmp_path / "configs/tasks/not_the_task/optuna/request.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    loader = experiments.config.loader.load_and_resolve_config if family == "experiment" else experiments.tuning.optuna.load_optuna_study_config
    with pytest.raises(ValueError, match="Task config path mismatch"):
        loader(destination)
