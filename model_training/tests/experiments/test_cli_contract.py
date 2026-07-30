# ruff: noqa: S101, S603, SLF001
"""
Protect the three thin command modules and their shared public argument contract.

Parser-level tests cover lightweight imports, exact device vocabulary, positive
artifact batch sizes, override forwarding, required arguments, and material
failure exit codes. Training, Optuna, and artifact services are stubbed here;
their lifecycle semantics are tested in dedicated modules.
"""

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
    """
    Import each CLI module in a fresh subprocess with the maintained package path.

    Torch, Optuna, and pandas must remain absent, protecting thin ``--help`` and
    parser use from expensive runtime initialization.
    """
    expression = f"import sys; import {module_name}; assert not {set(_FORBIDDEN_IMPORTS)!r}.intersection(sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", expression],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("parser", "arguments"),
    [
        (cli_train._build_parser, ["experiment.yaml", "--device", "cpu"]),
        (cli_optuna._build_parser, ["study.yaml", "--device", "cpu", "--dry-run"]),
        (cli_build_artifacts._build_parser, ["--runs-root", "runs", "--device", "cpu"]),
    ],
    ids=("train", "optuna", "artifacts"),
)
def test_all_cli_parsers_share_exact_device_vocabulary_and_help(
    parser: object,
    arguments: list[str],
) -> None:
    """
    Parse one valid CPU invocation through each train, study, and artifact parser.

    All three parameter families must expose the same exact device choices and
    strict/fallback help, preventing boundary-specific policy drift.
    """
    command_parser = parser()  # type: ignore[operator]
    assert command_parser.parse_args(arguments).device == "cpu"
    help_text = " ".join(command_parser.format_help().split())
    assert "--device {auto,cuda,cpu}" in help_text
    assert "auto chooses CUDA when usable, otherwise CPU" in help_text
    assert "cuda is strict and never falls back" in help_text
    assert "cpu avoids CUDA use" in help_text


@pytest.mark.parametrize(
    ("parser", "arguments"),
    [
        (cli_train._build_parser, ["experiment.yaml", "--device", "cuda:0"]),
        (cli_optuna._build_parser, ["study.yaml", "--device", "gpu"]),
        (cli_build_artifacts._build_parser, ["--runs-root", "runs", "--device", "AUTO"]),
    ],
    ids=("train", "optuna", "artifacts"),
)
def test_cli_invalid_device_values_fail_during_argument_parsing(
    parser: object,
    arguments: list[str],
) -> None:
    """
    Send an indexed CUDA token, a GPU alias, and capitalization drift to the three parsers.

    Every family must exit during argument parsing, before delegated services can
    resolve hardware or mutate lifecycle state.
    """
    with pytest.raises(SystemExit):
        parser().parse_args(arguments)  # type: ignore[operator]


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


def test_artifact_cli_removes_cpu_flag() -> None:
    """
    Pass the unsupported ``--cpu`` flag to the artifact parser.

    Parsing must fail so all runtime commands expose only the shared ``--device``
    vocabulary.
    """
    with pytest.raises(SystemExit):
        cli_build_artifacts._build_parser().parse_args(["--runs-root", "runs", "--cpu"])


def test_training_cli_forwards_device_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Stub training orchestration and invoke the CLI with explicit CPU policy.

    The exact token must be forwarded and success returned; the thin command must
    not pre-resolve or reinterpret hardware ownership.
    """
    captured: dict[str, object] = {}

    def capture_run(*_args: object, **kwargs: object) -> dict[str, object]:
        """Capture service keywords and return the smallest CLI-shaped success."""
        captured.update(kwargs)
        return {
            "run_dir": Path("run"),
            "result": {"best_epoch": 1, "best_metric": 0.5},
        }

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    assert cli_train.main(["experiment.yaml", "--device", "cpu"]) == 0
    assert captured["device"] == "cpu"


def test_artifact_cli_forwards_device_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Stub artifact orchestration and invoke the CLI with strict CUDA policy.

    The service must receive the exact policy token under its public argument name,
    proving the parser owns no runtime resolution.
    """
    captured: dict[str, object] = {}

    def capture_build(**kwargs: object) -> dict[str, object]:
        """Capture artifact-service keywords without generating any files."""
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(analysis.artifact_service, "build_artifacts", capture_build)
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
    assert captured["device_policy"] == "cuda"
    assert captured["dataset_root"] == tmp_path / "raw"
    assert captured["metadata_root"] == tmp_path / "meta"


def test_optuna_dry_run_applies_device_override_without_side_effects(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """
    Execute an Optuna CPU dry-run against a deliberately absent output root.

    Output must expose the policy and semantic signature while leaving all study,
    trial, and tracking storage absent, preserving dry-run as read-only validation.
    """
    config_path = Path(__file__).parents[2] / "configs/optuna/steady_flow_fno_search.yaml"
    output_root = tmp_path / "must-not-exist"
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
    output = capsys.readouterr().out
    assert '"device_policy": "cpu"' in output
    assert '"semantic_signature"' in output
    assert not output_root.exists()


def test_training_cli_returns_nonzero_on_material_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make delegated training raise one material runtime failure.

    The CLI must convert it to a nonzero return code so shell and queue callers do
    not mistake a failed lifecycle for success.
    """

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        """Stand in for a material failure owned by training orchestration."""
        message = "training failed"
        raise RuntimeError(message)

    monkeypatch.setattr(experiments.run, "run_experiment", fail_run)
    assert cli_train.main(["unused.yaml"]) == 1


def test_optuna_cli_returns_nonzero_on_material_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make delegated study loading raise one validation failure.

    The Optuna command must return nonzero even in dry-run mode, preserving error
    observability before any study starts.
    """

    def fail_load(_path: str) -> object:
        """Stand in for invalid study configuration at the service boundary."""
        message = "study invalid"
        raise ValueError(message)

    monkeypatch.setattr(experiments.tuning.optuna, "load_optuna_study_config", fail_load)
    assert cli_optuna.main(["unused.yaml", "--dry-run"]) == 1


def test_artifact_cli_returns_nonzero_on_material_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Make delegated artifact generation raise one material runtime failure.

    The command must return nonzero so wrapper logs and queue exit status remain
    authoritative for failed publication.
    """

    def fail_build(**_kwargs: object) -> dict[str, object]:
        """Stand in for material artifact-publication failure."""
        message = "generation failed"
        raise RuntimeError(message)

    monkeypatch.setattr(analysis.artifact_service, "build_artifacts", fail_build)
    assert cli_build_artifacts.main(["--runs-root", str(tmp_path)]) == 1
