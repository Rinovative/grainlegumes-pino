# ruff: noqa: S101
"""Validate side-effect-free workflow classification on test-owned YAMLs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.experiments.config import experiments_config_preflight as preflight
from support import configs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("workflow", "expected_family"),
    [
        ("train", preflight.EXPERIMENT_FAMILY),
        ("optuna", preflight.OPTUNA_FAMILY),
    ],
)
def test_matching_workflow_is_admitted(
    tmp_path: Path,
    workflow: str,
    expected_family: str,
) -> None:
    """Classify one complete artificial request for each public workflow."""
    payload = configs.direct_config() if workflow == "train" else configs.optuna_config()
    path = configs.write_yaml(tmp_path / f"{workflow}.yaml", payload)

    result = preflight.validate_workflow(path, requested_workflow=workflow)

    assert result.family == expected_family
    assert result.task == "steady_flow"


@pytest.mark.parametrize(
    ("requested_workflow", "payload_family"),
    [("train", "optuna"), ("optuna", "direct")],
)
def test_wrong_workflow_is_rejected(
    tmp_path: Path,
    requested_workflow: str,
    payload_family: str,
) -> None:
    """Reject cross-family dispatch without depending on correction wording."""
    payload = configs.optuna_config() if payload_family == "optuna" else configs.direct_config()
    path = configs.write_yaml(tmp_path / "request.yaml", payload)

    with pytest.raises(preflight.WorkflowMismatchError) as captured:
        preflight.validate_workflow(
            path,
            requested_workflow=requested_workflow,
        )

    assert captured.value.result.family != requested_workflow


def test_mixed_root_is_never_classified_by_filename(tmp_path: Path) -> None:
    """Fail closed when one YAML root mixes direct and Optuna ownership."""
    payload = configs.direct_config()
    payload["study"] = {}
    path = configs.write_yaml(tmp_path / "looks_like_search.yaml", payload)

    with pytest.raises(ValueError, match="mixes normal experiment and Optuna wrapper"):
        preflight.inspect_config(path)
