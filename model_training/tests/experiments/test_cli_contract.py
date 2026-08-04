# ruff: noqa: EM101, EM102, S101, SLF001, TRY003
"""Exercise CLI parsing, delegation, dry-run isolation, and failure status."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from src import analysis, experiments, learning
from src.experiments.cli import cli_build_artifacts, cli_config_preflight, cli_optuna, cli_train
from support import configs

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


def test_cli_parsers_accept_supported_devices_and_reject_unknown_values() -> None:
    """Keep device selection strict across the three runtime entry points."""
    valid_cases = (
        (cli_train._build_parser, ["experiment.yaml", "--device", "cpu"]),
        (cli_optuna._build_parser, ["study.yaml", "--device", "cpu", "--dry-run"]),
        (cli_build_artifacts._build_parser, ["--runs-root", "runs", "--device", "cpu"]),
    )
    invalid_cases = (
        (cli_train._build_parser, ["experiment.yaml", "--device", "unsupported"]),
        (cli_optuna._build_parser, ["study.yaml", "--device", "unsupported"]),
        (
            cli_build_artifacts._build_parser,
            ["--runs-root", "runs", "--device", "unsupported"],
        ),
    )
    for parser_builder, arguments in valid_cases:
        assert parser_builder().parse_args(arguments).device == "cpu"
    for parser_builder, arguments in invalid_cases:
        with pytest.raises(SystemExit):
            parser_builder().parse_args(arguments)

    preflight = cli_config_preflight._build_parser().parse_args(
        ["train", "experiment.yaml"],
    )
    assert preflight.workflow == "train"
    assert preflight.config_path == "experiment.yaml"


def test_artifact_cli_requires_run_root_and_forwards_rebuild_flag() -> None:
    """Represent artifact membership through a run root and explicit rebuild flag."""
    parser = cli_build_artifacts._build_parser()
    parsed = parser.parse_args(["--runs-root", "runs", "--rebuild"])
    assert parsed.runs_root == Path("runs")
    assert parsed.rebuild is True
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--task", "steady_flow", "--runs-root", "runs"])
    with pytest.raises(SystemExit):
        cli_optuna._build_parser().parse_args(["study.yaml", "--no-build-artifacts"])


def _cuda_zero_resolution() -> learning.device.DeviceResolution:
    """Return a CUDA decision without querying test-host hardware."""
    device = torch.device("cuda", 0)
    metadata = learning.device.DeviceRuntimeMetadata(
        requested_policy="cuda",
        resolved_device=str(device),
        device_type="cuda",
        pytorch_version=str(torch.__version__),
        cuda_index=0,
        cuda_device_name="restricted fixture GPU",
    )
    return learning.device.DeviceResolution(
        requested_policy="cuda",
        device=device,
        device_type="cuda",
        cuda_index=0,
        metadata=metadata,
    )


def test_direct_cli_builds_artifacts_after_strict_completion_on_training_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the exact completed path and CUDA decision after releasing training."""
    run_dir = tmp_path / "renamed completed bundle"
    resolution = _cuda_zero_resolution()
    events: list[str] = []
    training_call: dict[str, object] = {}
    post_training_call: dict[str, object] = {}
    artifact_cli_call: dict[str, object] = {}

    def capture_run(*_args: object, **kwargs: object) -> dict[str, object]:
        events.append("training returned")
        training_call.update(kwargs)
        return {
            "run_dir": run_dir,
            "result": {"best_epoch": 1, "best_metric": 0.5},
            "device_resolution": resolution,
        }

    def validate_completed(selected: object) -> dict[str, object]:
        assert selected is run_dir
        events.append("strict completion validated")
        return {}

    def cleanup(device: torch.device) -> None:
        assert device is resolution.device
        events.append("training resources cleaned")

    def prepare(selected: object, **kwargs: object) -> SimpleNamespace:
        assert selected is run_dir
        post_training_call.update(kwargs)
        events.append("artifacts prepared")
        return SimpleNamespace(role_actions={"id": "reused", "ood": "generated"})

    def capture_build(**kwargs: object) -> dict[str, object]:
        artifact_cli_call.update(kwargs)
        return {}

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    monkeypatch.setattr(experiments.run, "validate_completed_run", validate_completed)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", cleanup)
    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", prepare)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", capture_build)

    assert cli_train.main(["experiment.yaml", "--device", "cpu"]) == 0
    assert events == [
        "training returned",
        "strict completion validated",
        "training resources cleaned",
        "artifacts prepared",
    ]
    assert training_call["device"] == "cpu"
    assert post_training_call == {"device_resolution": resolution}

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
    assert artifact_cli_call["device_policy"] == "cuda"
    assert artifact_cli_call["dataset_root"] == tmp_path / "raw"
    assert artifact_cli_call["metadata_root"] == tmp_path / "meta"


def test_direct_cli_opt_out_skips_only_post_training_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed training unchanged when the trailing invocation flag opts out."""
    resolution = _cuda_zero_resolution()
    calls: list[str] = []

    def capture_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("training")
        return {
            "run_dir": Path("run"),
            "result": {"best_epoch": 1, "best_metric": 0.5},
            "device_resolution": resolution,
        }

    def reject_post_training(*_args: object, **_kwargs: object) -> object:
        pytest.fail("opt-out reached post-training artifact processing")

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    monkeypatch.setattr(experiments.run, "validate_completed_run", reject_post_training)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", reject_post_training)
    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", reject_post_training)

    assert cli_train.main(["experiment.yaml", "--no-build-artifacts"]) == 0
    assert calls == ["training"]


def test_artifact_failure_is_nonzero_without_relabeling_completed_training(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report recoverable post-processing failure without a lifecycle transition."""
    resolution = _cuda_zero_resolution()

    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": Path("completed-run"),
            "result": {"best_epoch": 1, "best_metric": 0.5},
            "device_resolution": resolution,
        },
    )
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: {})
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)
    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_or_build_run_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("artifact cache is incompatible")),
    )
    monkeypatch.setattr(
        experiments.run,
        "transition_run_status",
        lambda *_args, **_kwargs: pytest.fail("artifact failure changed training lifecycle"),
    )

    assert cli_train.main(["experiment.yaml"]) == 1
    captured = capsys.readouterr()
    assert "Training status: completed" in captured.out
    assert "Post-training artifacts: failed" in captured.err
    assert "artifact CLI or evaluation notebook later" in captured.err


def _describe_artificial_dataset(
    seen: list[str],
    dataset_id: str,
    *,
    task: TaskSpec,
    dataset_root: Path,
    metadata_root: Path,
) -> SimpleNamespace:
    """Return a valid bounded summary for a test-owned dataset identifier."""
    seen.append(dataset_id)
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_path=dataset_root / f"{dataset_id}.pt",
        metadata_directory=metadata_root / dataset_id,
        dataset_exists=True,
        task_id=task.id,
        data_contract_digest=task.data_contract_digest,
        fingerprint=f"{dataset_id}-fingerprint",
        sample_count=4,
    )


def test_optuna_dry_run_resolves_artificial_plan_without_side_effects(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a test-owned study without SDK, allocator, tracking, or output writes."""
    request = configs.optuna_config()
    config_path = configs.write_yaml(tmp_path / "artificial-study.yaml", request)
    study_name = str(request["study"]["name"])
    output_root = tmp_path / "must-not-exist"
    described: list[str] = []

    def describe(dataset_id: str, **kwargs: object) -> SimpleNamespace:
        return _describe_artificial_dataset(described, dataset_id, **kwargs)  # type: ignore[arg-type]

    def reject_side_effect(*_args: object, **_kwargs: object) -> object:
        pytest.fail("dry-run reached a side-effecting runtime boundary")

    runtime = experiments.tuning.optuna
    monkeypatch.setattr(runtime.datasets.metadata, "load_dataset_metadata_summary", describe)
    monkeypatch.setattr(runtime, "_optuna_module", reject_side_effect)
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
            ],
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    study_dir = output_root / "steady_flow" / "studies" / study_name
    assert plan["device_policy"] == "cpu"
    assert plan["study_dir"] == str(study_dir)
    assert plan["trial_root"] == str(study_dir / "trials")
    assert plan["storage"] == f"sqlite:///{study_dir / f'{study_name}.db'}"
    assert plan["study"]["role"] is None
    assert plan["dataset_roles"]["id"]["dataset_id"] == "synthetic_train"
    assert plan["dataset_roles"]["ood"][0]["dataset_id"] == "synthetic_ood"
    assert described == ["synthetic_train", "synthetic_ood"]
    assert "semantic_signature" in plan
    assert not output_root.exists()


def test_optuna_dry_run_dataset_failure_is_nonzero_without_output(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a missing artificial dataset before allocating study state."""
    config_path = configs.write_yaml(
        tmp_path / "artificial-study.yaml",
        configs.optuna_config(),
    )
    output_root = tmp_path / "must-not-exist"

    def reject_dataset(dataset_id: str, **_kwargs: object) -> None:
        raise FileNotFoundError(f"configured dataset missing: {dataset_id}")

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
            ],
        )
        == 1
    )
    assert "synthetic_train" in capsys.readouterr().err
    assert not output_root.exists()


def test_cli_failures_are_nonzero_and_redact_credentials(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return failure status while keeping credential-shaped values out of errors."""

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("api_key=never-disclose")

    def fail_load(_path: str) -> object:
        raise ValueError("study invalid")

    def fail_build(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(experiments.run, "run_experiment", fail_run)
    monkeypatch.setattr(experiments.tuning.optuna, "load_optuna_study_config", fail_load)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", fail_build)

    assert cli_train.main(["unused.yaml"]) == 1
    training_error = capsys.readouterr().err
    assert "RuntimeError" in training_error
    assert "api_key=<redacted>" in training_error
    assert "never-disclose" not in training_error
    assert cli_optuna.main(["unused.yaml", "--dry-run"]) == 1
    assert cli_build_artifacts.main(["--runs-root", str(tmp_path)]) == 1
