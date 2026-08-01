# ruff: noqa: S101
"""Protect the single maintained ownership and accepted version of W&B."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from src import common, experiments

_ACCEPTED_WANDB_VERSION = "0.28.1"
_REPOSITORY_ROOT = Path(__file__).parents[3]


def _pip_dependencies(environment_path: Path) -> list[str]:
    payload = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    pip_sections = [entry["pip"] for entry in payload["dependencies"] if isinstance(entry, dict) and "pip" in entry]
    assert len(pip_sections) == 1
    return pip_sections[0]


def test_wandb_dependency_has_one_exact_maintained_version_contract() -> None:
    """Require both runtime environments and the installed SDK to agree exactly."""
    expected = f"wandb == {_ACCEPTED_WANDB_VERSION}"
    for filename in ("environment.yml", "environment-dev.yml"):
        dependencies = _pip_dependencies(_REPOSITORY_ROOT / filename)
        declarations = [dependency for dependency in dependencies if Requirement(dependency).name == "wandb"]
        assert declarations == [expected]
        assert str(Requirement(declarations[0]).specifier) == f"=={_ACCEPTED_WANDB_VERSION}"

    project = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wheel_dependencies = project.get("dependencies", [])
    assert all(Requirement(dependency).name != "wandb" for dependency in wheel_dependencies)
    assert importlib.metadata.version("wandb") == _ACCEPTED_WANDB_VERSION


def test_managed_wandb_workspace_implementation_is_completely_absent() -> None:
    """Keep dependencies, modules, exports, state paths, and imports removed."""
    for filename in ("environment.yml", "environment-dev.yml"):
        dependencies = _pip_dependencies(_REPOSITORY_ROOT / filename)
        assert all(Requirement(dependency).name != "wandb-workspaces" for dependency in dependencies)

    project = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wheel_dependencies = project.get("dependencies", [])
    assert all(Requirement(dependency).name != "wandb-workspaces" for dependency in wheel_dependencies)

    removed_paths = (
        _REPOSITORY_ROOT / "model_training/src/experiments/experiments_workspaces.py",
        _REPOSITORY_ROOT / "model_training/src/experiments/cli/cli_sync_wandb_workspaces.py",
        _REPOSITORY_ROOT / "model_training/tests/experiments/test_wandb_workspaces.py",
    )
    assert all(not path.exists() for path in removed_paths)
    maintained_source = tuple((_REPOSITORY_ROOT / "model_training/src").rglob("*.py"))
    assert maintained_source
    for source_path in maintained_source:
        source = source_path.read_text(encoding="utf-8")
        assert "wandb_workspaces" not in source
        assert "experiments_workspaces" not in source
        assert "workspace_contract" not in source

    assert not hasattr(common.paths, "get_wandb_state_root")
    assert not hasattr(common.paths, "resolve_wandb_workspace_contract_path")
    assert not hasattr(common.paths, "resolve_wandb_workspace_lock_path")
    assert not hasattr(experiments, "workspaces")
