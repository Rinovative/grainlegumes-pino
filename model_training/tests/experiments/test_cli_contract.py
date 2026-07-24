# ruff: noqa: S101, S603
"""Verify command modules stay lightweight and expose material failures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from src import analysis, experiments
from src.experiments.cli import cli_build_artifacts, cli_optuna, cli_train

_CLI_MODULES = (
    "src.experiments.cli.cli_train",
    "src.experiments.cli.cli_optuna",
    "src.experiments.cli.cli_build_artifacts",
)
_FORBIDDEN_IMPORTS = ("torch", "optuna", "pandas")


def _subprocess_environment() -> dict[str, str]:
    """Return an isolated environment with the maintained package on PYTHONPATH."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])
    return environment


@pytest.mark.parametrize("module_name", _CLI_MODULES, ids=("train", "optuna", "artifacts"))
def test_cli_module_import_is_lightweight(module_name: str) -> None:
    """Importing a command module does not initialize runtime dependencies."""
    expression = f"import sys; import {module_name}; assert not {set(_FORBIDDEN_IMPORTS)!r}.intersection(sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", expression],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert completed.returncode == 0, completed.stderr


def test_training_cli_returns_nonzero_on_material_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training lifecycle failures are observable to shell callers."""

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        message = "training failed"
        raise RuntimeError(message)

    monkeypatch.setattr(experiments.run, "run_experiment", fail_run)
    assert cli_train.main(["unused.yaml"]) == 1


def test_optuna_cli_returns_nonzero_on_material_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Study validation failures are observable to shell callers."""

    def fail_load(_path: str) -> object:
        message = "study invalid"
        raise ValueError(message)

    monkeypatch.setattr(experiments.tuning.optuna, "load_optuna_study_config", fail_load)
    assert cli_optuna.main(["unused.yaml", "--dry-run"]) == 1


def test_artifact_cli_returns_nonzero_on_material_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact generation failures are observable to shell callers."""

    def fail_build(**_kwargs: object) -> dict[str, object]:
        message = "generation failed"
        raise RuntimeError(message)

    monkeypatch.setattr(analysis.artifact_service, "build_artifacts", fail_build)
    assert cli_build_artifacts.main(["--runs-root", str(tmp_path)]) == 1
