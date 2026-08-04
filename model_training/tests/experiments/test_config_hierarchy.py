# ruff: noqa: S101
"""Protect path-sensitive config semantics with temporary artificial files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src import experiments
from support import configs

if TYPE_CHECKING:
    from pathlib import Path


def test_category_path_does_not_change_resolved_semantics(tmp_path: Path) -> None:
    """Resolve identical test-owned requests from two arbitrary category names."""
    raw = configs.direct_config()
    paths = (
        tmp_path / "configs/tasks/steady_flow/experiments/category_a/request.yaml",
        tmp_path / "configs/tasks/steady_flow/experiments/renamed_category/request.yaml",
    )
    for path in paths:
        configs.write_yaml(path, raw)

    resolved = [experiments.config.loader.load_and_resolve_config(path) for path in paths]

    assert resolved[0] == resolved[1]
    assert resolved[0]["run"]["name"] == resolved[1]["run"]["name"]


@pytest.mark.parametrize("family", ["direct", "optuna"])
def test_directory_task_mismatch_is_rejected(
    tmp_path: Path,
    family: str,
) -> None:
    """Reject a task-owned path whose artificial request declares another task."""
    if family == "direct":
        payload = configs.direct_config()
        path = tmp_path / "configs/tasks/not_the_task/experiments/category/request.yaml"
        loader = experiments.config.loader.load_and_resolve_config
    else:
        payload = configs.optuna_config()
        path = tmp_path / "configs/tasks/not_the_task/optuna/request.yaml"
        loader = experiments.tuning.optuna.load_optuna_study_config
    configs.write_yaml(path, payload)

    with pytest.raises(ValueError, match="Task config path mismatch"):
        loader(path)
