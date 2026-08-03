# ruff: noqa: S101
"""Validate strict side-effect-free config-family classification and workflow dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from src import experiments
from src.experiments.config import experiments_config_preflight as preflight
from support import configs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("path", [*configs.experiment_config_paths(), *configs.acceptance_config_paths()])
def test_direct_experiments_preflight_as_train(path: Path) -> None:
    """Admit every direct experiment and acceptance request through train only."""
    result = preflight.validate_workflow(path, requested_workflow="train")
    assert result.family == preflight.EXPERIMENT_FAMILY
    assert result.task == configs.directory_task(path)
    assert result.canonical_path.startswith("model_training/configs/tasks/")


@pytest.mark.parametrize("path", configs.optuna_config_paths())
def test_optuna_wrappers_preflight_as_optuna(path: Path) -> None:
    """Admit every wrapper only through the maintained Optuna workflow."""
    result = preflight.validate_workflow(path, requested_workflow="optuna")
    assert result.family == preflight.OPTUNA_FAMILY
    assert result.task == configs.directory_task(path)
    assert result.canonical_path.startswith("model_training/configs/tasks/")


@pytest.mark.parametrize("path", [*configs.experiment_config_paths(), *configs.acceptance_config_paths()])
def test_every_direct_experiment_is_rejected_by_the_optuna_wrapper_schema(path: Path) -> None:
    """Keep direct requests out of the strict wrapper schema for every discovered YAML."""
    with pytest.raises((KeyError, TypeError, ValueError)):
        experiments.tuning.optuna.load_optuna_study_config(path)


@pytest.mark.parametrize("path", configs.optuna_config_paths())
def test_every_wrapper_is_rejected_by_the_normal_experiment_schema(path: Path) -> None:
    """Keep wrappers out of normal resolution for every discovered YAML."""
    with pytest.raises((KeyError, TypeError, ValueError)):
        experiments.config.loader.load_and_resolve_config(path)


def test_train_with_wrapper_fails_with_exact_corrected_command() -> None:
    """Reject inverse misuse without silently rerouting the valid wrapper."""
    path = configs.optuna_config_paths()[0]
    with pytest.raises(preflight.WorkflowMismatchError) as captured:
        preflight.validate_workflow(path, requested_workflow="train")

    message = str(captured.value)
    assert "Supplied config family: optuna" in message
    assert "Requested workflow: train" in message
    assert f"./scripts/docker_job.sh optuna '{captured.value.result.canonical_path}'" in message


def test_optuna_with_experiment_fails_with_unique_matching_study() -> None:
    """Reject a plain experiment and suggest only its unique task/model-family wrapper."""
    path = configs.experiment_config_path(model_kind="uno", physics_enabled=False)
    with pytest.raises(preflight.WorkflowMismatchError) as captured:
        preflight.validate_workflow(path, requested_workflow="optuna")

    message = str(captured.value)
    matches = preflight.matching_optuna_configs(captured.value.result)
    assert len(matches) == 1
    assert "Supplied config family: experiment" in message
    assert "Requested workflow: optuna" in message
    assert f"./scripts/docker_job.sh train '{captured.value.result.canonical_path}'" in message
    assert f"./scripts/docker_job.sh optuna '{matches[0].canonical_path}'" in message


def test_mixed_root_is_ambiguous_and_never_classified_by_filename(tmp_path: Path) -> None:
    """Fail closed when a root mixes schema ownership, regardless of its path name."""
    source = configs.experiment_config_paths()[0]
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["study"] = {}
    path = tmp_path / "looks_like_search.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="mixes normal experiment and Optuna wrapper"):
        preflight.inspect_config(path)
