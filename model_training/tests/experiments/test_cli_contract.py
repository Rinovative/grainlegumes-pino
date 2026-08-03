# ruff: noqa: S101, S603, SLF001
"""
Protect the thin command modules and their public argument contracts.

Parser-level tests cover lightweight imports, exact device vocabulary, positive
artifact batch sizes, override forwarding, required arguments, and material
failure exit codes. Training, Optuna, and artifact services are stubbed here;
their lifecycle semantics are tested in dedicated modules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from src import analysis, experiments
from src.experiments.cli import cli_build_artifacts, cli_config_preflight, cli_optuna, cli_train
from support import configs

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec

_CLI_MODULES = (
    "src.experiments.cli.cli_train",
    "src.experiments.cli.cli_optuna",
    "src.experiments.cli.cli_config_preflight",
    "src.experiments.cli.cli_build_artifacts",
)
_FORBIDDEN_IMPORTS = ("torch", "optuna", "pandas", "wandb")


def _subprocess_environment() -> dict[str, str]:
    """Return an isolated environment with the maintained package on PYTHONPATH."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])
    return environment


def test_cli_modules_and_parsers_are_lightweight_consistent_and_strict() -> None:
    """Probe all thin imports and both valid and invalid device parser paths."""
    for module_name in _CLI_MODULES:
        expression = f"import sys; import {module_name}; assert not {set(_FORBIDDEN_IMPORTS)!r}.intersection(sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", expression],
            check=False,
            capture_output=True,
            text=True,
            env=_subprocess_environment(),
        )
        assert completed.returncode == 0, completed.stderr

    parser_cases = (
        (cli_train._build_parser, ["experiment.yaml", "--device", "cpu"]),
        (cli_optuna._build_parser, ["study.yaml", "--device", "cpu", "--dry-run"]),
        (cli_build_artifacts._build_parser, ["--runs-root", "runs", "--device", "cpu"]),
    )
    invalid_cases = (
        (cli_train._build_parser, ["experiment.yaml", "--device", "unsupported"]),
        (cli_optuna._build_parser, ["study.yaml", "--device", "unsupported"]),
        (cli_build_artifacts._build_parser, ["--runs-root", "runs", "--device", "unsupported"]),
    )
    for parser_builder, arguments in parser_cases:
        parser = parser_builder()
        assert parser.parse_args(arguments).device == "cpu"
        help_text = " ".join(parser.format_help().split())
        assert "--device {auto,cuda,cpu}" in help_text
        assert "auto chooses CUDA when usable, otherwise CPU" in help_text
        assert "cuda is strict and never falls back" in help_text
        assert "cpu avoids CUDA use" in help_text

    preflight_parser = cli_config_preflight._build_parser()
    parsed_preflight = preflight_parser.parse_args(["train", "experiment.yaml"])
    assert parsed_preflight.workflow == "train"
    assert parsed_preflight.config_path == "experiment.yaml"
    assert "{train,optuna}" in " ".join(preflight_parser.format_help().split())

    for parser_builder, arguments in invalid_cases:
        with pytest.raises(SystemExit):
            parser_builder().parse_args(arguments)


def test_artifact_cli_accepts_any_positive_processing_batch_size() -> None:
    """
    Parse a positive multi-case artifact batch and then the zero boundary.

    Any positive size must be retained while zero fails, preserving chunking as
    operational policy rather than a one-case scientific restriction.
    """
    parser = cli_build_artifacts._build_parser()
    requested_batch_size = 3
    parsed = parser.parse_args(["--runs-root", "runs", "--batch-size", str(requested_batch_size)])
    assert parsed.batch_size == requested_batch_size
    with pytest.raises(SystemExit):
        parser.parse_args(["--runs-root", "runs", "--batch-size", "0"])


def test_cli_services_receive_exact_device_and_path_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward train and artifact policies without resolving them in the parser."""
    training_call: dict[str, object] = {}
    artifact_call: dict[str, object] = {}

    def capture_run(*_args: object, **kwargs: object) -> dict[str, object]:
        training_call.update(kwargs)
        return {
            "run_dir": Path("run"),
            "result": {"best_epoch": 1, "best_metric": 0.5},
        }

    def capture_build(**kwargs: object) -> dict[str, object]:
        artifact_call.update(kwargs)
        return {}

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", capture_build)
    assert cli_train.main(["experiment.yaml", "--device", "cpu"]) == 0
    assert (
        cli_build_artifacts.main(
            [
                "--runs-root",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path / "raw"),
                "--metadata-root",
                str(tmp_path / "meta"),
                "--device",
                "cuda",
            ],
        )
        == 0
    )

    assert training_call["device"] == "cpu"
    assert artifact_call["device_policy"] == "cuda"
    assert artifact_call["dataset_root"] == tmp_path / "raw"
    assert artifact_call["metadata_root"] == tmp_path / "meta"


def test_optuna_dry_run_applies_device_override_without_side_effects(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Print a fully resolved CPU plan without SDK, training, tracking, or writes."""
    config_path = configs.optuna_config_path(model_kind="fno", physics_enabled=False)
    study_name = experiments.tuning.optuna.load_optuna_study_config(config_path).study["name"]
    output_root = tmp_path / "must-not-exist"
    described_datasets: list[str] = []

    def describe_dataset(
        dataset_id: str,
        *,
        task: TaskSpec,
        dataset_root: Path,
        metadata_root: Path,
    ) -> SimpleNamespace:
        """Return a bounded validated summary while recording both configured roles."""
        described_datasets.append(dataset_id)
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_path=dataset_root / f"{dataset_id}.pt",
            metadata_directory=metadata_root / dataset_id,
            dataset_exists=True,
            task_id=task.id,
            task_contract_digest=task.contract_digest,
            fingerprint=dataset_id + "-fingerprint",
            sample_count=4,
        )

    def reject_side_effect(*_args: object, **_kwargs: object) -> object:
        """Fail if dry-run reaches an SDK, run allocator, or W&B initializer."""
        pytest.fail("dry-run reached a side-effecting runtime boundary")

    optuna_runtime = experiments.tuning.optuna
    monkeypatch.setattr(optuna_runtime.datasets.metadata, "load_dataset_metadata_summary", describe_dataset)
    monkeypatch.setattr(optuna_runtime, "_optuna_module", reject_side_effect)
    monkeypatch.setattr(experiments.run, "prepare_fresh_run", reject_side_effect)
    monkeypatch.setattr(experiments.tracking, "initialize_wandb", reject_side_effect)

    assert (
        cli_optuna.main(
            [
                str(config_path),
                "--dry-run",
                "--device",
                "cpu",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    study_dir = output_root / "steady_flow" / "studies" / study_name
    assert plan["device_policy"] == "cpu"
    assert plan["study_dir"] == str(study_dir)
    assert plan["trial_root"] == str(study_dir / "trials")
    assert plan["storage"] == f"sqlite:///{study_dir / f'{study_name}.db'}"
    assert plan["task"] == "steady_flow"
    assert plan["model_kind"] == "fno"
    assert plan["dataset_roles"]["id"]["dataset_id"] == "lhs_var80_seed3001"
    assert plan["dataset_roles"]["ood"][0]["dataset_id"] == "lhs_var120_seed4001"
    assert described_datasets == ["lhs_var80_seed3001", "lhs_var120_seed4001"]
    assert "semantic_signature" in plan
    assert not output_root.exists()


def test_optuna_dry_run_dataset_failure_is_nonzero_without_output(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject one missing configured dataset before creating study state."""
    config_path = configs.optuna_config_path(model_kind="fno", physics_enabled=False)
    output_root = tmp_path / "must-not-exist"

    def reject_dataset(dataset_id: str, **_kwargs: object) -> None:
        """Stand in for metadata admission of an absent configured dataset."""
        msg = f"configured dataset missing: {dataset_id}"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(
        experiments.tuning.optuna.datasets.metadata,
        "load_dataset_metadata_summary",
        reject_dataset,
    )
    assert (
        cli_optuna.main(
            [
                str(config_path),
                "--dry-run",
                "--device",
                "cpu",
                "--output-root",
                str(output_root),
            ]
        )
        == 1
    )
    assert "configured dataset missing: lhs_var80_seed3001" in capsys.readouterr().err
    assert not output_root.exists()


def test_training_cli_retains_a_sanitized_full_traceback(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep admission failures diagnosable without disclosing credential-shaped text."""

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        message = "api_key=never-disclose"
        raise RuntimeError(message)

    monkeypatch.setattr(experiments.run, "run_experiment", fail_run)
    assert cli_train.main(["unused.yaml"]) == 1
    error = capsys.readouterr().err
    assert "Training failed; sanitized traceback follows." in error
    assert "Traceback (most recent call last):" in error
    assert "RuntimeError: api_key=<redacted>" in error
    assert "never-disclose" not in error


def test_all_clis_return_nonzero_on_material_service_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert delegated training, study, and artifact failures to nonzero status."""

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        message = "training failed"
        raise RuntimeError(message)

    def fail_load(_path: str) -> object:
        message = "study invalid"
        raise ValueError(message)

    def fail_build(**_kwargs: object) -> dict[str, object]:
        message = "generation failed"
        raise RuntimeError(message)

    monkeypatch.setattr(experiments.run, "run_experiment", fail_run)
    monkeypatch.setattr(experiments.tuning.optuna, "load_optuna_study_config", fail_load)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", fail_build)

    assert cli_train.main(["unused.yaml"]) == 1
    assert cli_optuna.main(["unused.yaml", "--dry-run"]) == 1
    assert cli_build_artifacts.main(["--runs-root", str(tmp_path)]) == 1
