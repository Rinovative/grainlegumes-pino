# ruff: noqa: PLR2004, S101, S603
"""Verify the cluster queue wrappers without invoking runTSGPU, Docker, or training."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[3]


@dataclass(frozen=True)
class _Harness:
    """Hold one isolated repository copy and command-capture environment."""

    repository: Path
    environment: dict[str, str]
    runtsgpu_capture: Path
    docker_capture: Path
    docker_environment_capture: Path


def _write_executable(path: Path, content: str) -> None:
    """Write one isolated command stub with executable permissions."""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _capture_arguments(path: Path) -> list[str]:
    """Decode one NUL-delimited command capture."""
    return [part.decode() for part in path.read_bytes().split(b"\0") if part]


def _harness(
    tmp_path: Path,
    *,
    docker_exit_code: int = 0,
    exported_key: str | None = "mock API key with spaces",
    file_key: str | None = None,
) -> _Harness:
    """Create safe nvidia-smi, runTSGPU, and Docker stubs around copied scripts."""
    repository = tmp_path / "repository with spaces"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (repository.parent / "storage").mkdir()
    for name in ("docker_job.sh", "_docker_run.sh", "docker_dev.sh"):
        shutil.copy2(_REPOSITORY_ROOT / "scripts" / name, scripts / name)

    binary_dir = tmp_path / "stub commands"
    binary_dir.mkdir()
    _write_executable(
        binary_dir / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *"--query-gpu=index,name,utilization.gpu,memory.used,memory.total"*)
    printf '0, Cluster GPU A, 20, 7000, 24000\n2, Cluster GPU B, 5, 1000, 24000\n'
    ;;
  *"--query-gpu=index,memory.used"*)
    printf '0, 7000\n2, 1000\n'
    ;;
  *"--query-gpu=index --format=csv,noheader,nounits"*)
    printf '0\n2\n'
    ;;
  *)
    printf 'unexpected nvidia-smi arguments: %s\n' "$*" >&2
    exit 64
    ;;
esac
""",
    )
    _write_executable(
        binary_dir / "runTSGPU.py",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" > "${RUNTSGPU_CAPTURE}"
while (( $# > 0 )); do
  if [[ "$1" == "--" ]]; then
    shift
    exec "$@"
  fi
  shift
done
echo 'runTSGPU stub did not receive --' >&2
exit 65
""",
    )
    _write_executable(
        binary_dir / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == "ps" ]]; then
  exit 0
fi
printf '%s\\0' "$@" > "${DOCKER_CAPTURE}"
printf '%s' "${WANDB_API_KEY-<unset>}" > "${DOCKER_ENV_CAPTURE}"
printf 'captured Docker output with spaces\n'
exit "${DOCKER_EXIT_CODE:-0}"
""",
    )

    home = tmp_path / "home without ssh"
    home.mkdir()
    if file_key is not None:
        (home / "wandb_key.txt").write_text(file_key, encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{binary_dir}{os.pathsep}{environment['PATH']}",
            "HOME": str(home),
            "STORAGE_ROOT": str(tmp_path / "storage root with spaces"),
            "RUNTSGPU_CAPTURE": str(tmp_path / "runtsgpu.args"),
            "DOCKER_CAPTURE": str(tmp_path / "docker.args"),
            "DOCKER_ENV_CAPTURE": str(tmp_path / "docker.env"),
            "DOCKER_EXIT_CODE": str(docker_exit_code),
        }
    )
    if exported_key is None:
        environment.pop("WANDB_API_KEY", None)
    else:
        environment["WANDB_API_KEY"] = exported_key
    return _Harness(
        repository=repository,
        environment=environment,
        runtsgpu_capture=Path(environment["RUNTSGPU_CAPTURE"]),
        docker_capture=Path(environment["DOCKER_CAPTURE"]),
        docker_environment_capture=Path(environment["DOCKER_ENV_CAPTURE"]),
    )


def _run_job(
    harness: _Harness,
    *arguments: str,
    selection: str,
) -> subprocess.CompletedProcess[str]:
    """Run the copied submission wrapper against isolated command stubs."""
    return subprocess.run(
        [str(harness.repository / "scripts" / "docker_job.sh"), *arguments],
        cwd=harness.repository,
        env=harness.environment,
        input=selection,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_dev(harness: _Harness) -> subprocess.CompletedProcess[str]:
    """Run the copied interactive launcher against the isolated Docker stub."""
    return subprocess.run(
        [str(harness.repository / "scripts" / "docker_dev.sh")],
        cwd=harness.repository,
        env=harness.environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_wandb_forwarding(
    harness: _Harness,
    result: subprocess.CompletedProcess[str],
    *,
    expected_key: str | None,
    possible_keys: tuple[str | None, ...],
) -> None:
    """Verify name-only Docker forwarding without exposing credential values."""
    docker = _capture_arguments(harness.docker_capture)
    expected_capture = expected_key if expected_key is not None else "<unset>"
    assert harness.docker_environment_capture.read_text(encoding="utf-8") == expected_capture
    assert ("WANDB_API_KEY" in docker) is (expected_key is not None)
    assert not any(argument.startswith("WANDB_API_KEY=") for argument in docker)

    log_text = "\n".join(log_path.read_text(encoding="utf-8") for log_path in (harness.repository / "logs").glob("*.log"))
    visible_text = "\n".join(("\0".join(docker), result.stdout, result.stderr, log_text))
    for key in possible_keys:
        if key:
            assert key not in visible_text


def _assert_common_chain(
    harness: _Harness,
    result: subprocess.CompletedProcess[str],
    *,
    gpu: str,
    job_type: str,
    module: str,
    command_arguments: list[str],
) -> str:
    """Verify the queue, Docker, logging, storage, W&B, and CLI command layers."""
    assert result.returncode == 0, result.stderr
    runtsgpu = _capture_arguments(harness.runtsgpu_capture)
    assert runtsgpu[:3] == [f"-g{gpu}", "--", str(harness.repository / "scripts" / "_docker_run.sh")]
    assert runtsgpu[3:5] == [gpu, job_type]
    log_basename = runtsgpu[5]
    assert re.fullmatch(rf"\d{{8}}_\d{{6}}__{job_type}__gpu{gpu}__[A-Za-z0-9]{{6}}\.log", log_basename)
    assert runtsgpu[6:] == command_arguments

    docker = _capture_arguments(harness.docker_capture)
    assert docker[:2] == ["run", "--rm"]
    assert docker[docker.index("--gpus") + 1] == f"device={gpu}"
    assert docker[docker.index("--workdir") + 1] == "/workspace/repo/model_training"
    assert "STORAGE_ROOT=/workspace/storage" in docker
    assert "WANDB_API_KEY" in docker
    project_mount = f"{harness.repository}:/workspace/repo:rw"
    storage_mount = f"{harness.environment['STORAGE_ROOT']}:/workspace/storage:rw"
    assert project_mount in docker
    assert storage_mount in docker
    assert not any("/workspace/repo/data" in argument for argument in docker)
    assert not any("DATA_ROOT" in argument or "GEN_ROOT" in argument or "TRAIN_ROOT" in argument for argument in docker)
    assert docker[-(len(command_arguments) + 3) :] == [
        "python",
        "-m",
        module,
        *command_arguments,
    ]
    assert harness.docker_environment_capture.read_text(encoding="utf-8") == "mock API key with spaces"

    log_path = harness.repository / "logs" / log_basename
    assert log_path.read_text(encoding="utf-8") == "captured Docker output with spaces\n"
    assert "Current GPU usage:" in result.stdout
    assert f"Queue: runTSGPU.py -g{gpu} -s" in result.stdout
    assert f"Tail:  tail -F {log_path}" in result.stdout
    return log_basename


def test_automatic_gpu_queues_training_with_exact_arguments(tmp_path: Path) -> None:
    """The least-used proposed GPU reaches runTSGPU, Docker, and the training CLI."""
    harness = _harness(tmp_path)
    arguments = [
        "configs/experiments/config with spaces.yaml",
        "--resume",
        "/workspace/storage/run with spaces",
        "--device",
        "cuda",
    ]
    result = _run_job(harness, "train", *arguments, selection="\n")

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="train",
        module="src.experiments.cli.cli_train",
        command_arguments=arguments,
    )


def test_explicit_gpu_queues_optuna_with_runtime_overrides(tmp_path: Path) -> None:
    """An explicit available GPU and all Optuna overrides are preserved exactly."""
    harness = _harness(tmp_path)
    arguments = [
        "configs/optuna/steady_flow_fno_search.yaml",
        "--n-trials",
        "3",
        "--output-root",
        "/workspace/storage/output root",
        "--show-progress-bar",
    ]
    result = _run_job(harness, "optuna", *arguments, selection="0\n")

    _assert_common_chain(
        harness,
        result,
        gpu="0",
        job_type="optuna",
        module="src.experiments.cli.cli_optuna",
        command_arguments=arguments,
    )


def test_artifact_arguments_keep_duplicates_and_spaces(tmp_path: Path) -> None:
    """Artifact options, repeated selectors, and spaces cross every shell boundary."""
    harness = _harness(tmp_path)
    arguments = [
        "--task",
        "steady_flow",
        "--run-name",
        "first run",
        "--run-name",
        "second run",
        "--cpu",
    ]
    result = _run_job(harness, "artifacts", *arguments, selection="2\n")

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="artifacts",
        module="src.experiments.cli.cli_build_artifacts",
        command_arguments=arguments,
    )


def test_repeated_submissions_allocate_distinct_logs(tmp_path: Path) -> None:
    """Even same-second submissions receive separate host-visible log files."""
    harness = _harness(tmp_path)
    command = ("train", "configs/experiments/steady_flow_fno.yaml")

    first = _run_job(harness, *command, selection="\n")
    second = _run_job(harness, *command, selection="\n")

    assert first.returncode == 0
    assert second.returncode == 0
    logs = sorted((harness.repository / "logs").glob("*.log"))
    assert len(logs) == 2
    assert logs[0].name != logs[1].name


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("train",),
        ("optuna",),
        ("unsupported",),
    ],
)
def test_missing_or_unknown_job_arguments_fail_before_submission(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    """Missing commands/configs and unknown job types never reach runTSGPU."""
    harness = _harness(tmp_path)
    result = _run_job(harness, *arguments, selection="")

    assert result.returncode == 2
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


def test_invalid_explicit_gpu_fails_before_submission(tmp_path: Path) -> None:
    """A GPU outside the reported index set is rejected before queue submission."""
    harness = _harness(tmp_path)
    result = _run_job(
        harness,
        "train",
        "configs/experiments/steady_flow_fno.yaml",
        selection="7\n",
    )

    assert result.returncode == 2
    assert "not one of the available indices" in result.stderr
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


def test_inner_docker_failure_reaches_the_queue_process_and_log(tmp_path: Path) -> None:
    """A non-zero Docker/inner-command result is preserved by both wrappers."""
    harness = _harness(tmp_path, docker_exit_code=37)
    result = _run_job(
        harness,
        "train",
        "configs/experiments/steady_flow_fno.yaml",
        selection="\n",
    )

    assert result.returncode == 37
    runtsgpu = _capture_arguments(harness.runtsgpu_capture)
    log_path = harness.repository / "logs" / runtsgpu[5]
    assert log_path.read_text(encoding="utf-8") == "captured Docker output with spaces\n"
    assert "Queued train job" not in result.stdout


@pytest.mark.parametrize(
    ("exported_key", "file_key", "expected_key"),
    [
        ("exported secret with spaces", "ignored file secret", "exported secret with spaces"),
        (None, "file secret with spaces\n", "file secret with spaces"),
        (None, None, None),
    ],
)
def test_wandb_credentials_are_resolved_and_forwarded_without_disclosure(
    tmp_path: Path,
    exported_key: str | None,
    file_key: str | None,
    expected_key: str | None,
) -> None:
    """Both launchers apply precedence and omit credential values from arguments."""
    possible_keys = (exported_key, file_key.rstrip("\n") if file_key else file_key)

    queued = _harness(tmp_path / "queued", exported_key=exported_key, file_key=file_key)
    queued_result = _run_job(
        queued,
        "train",
        "configs/experiments/steady_flow_fno.yaml",
        selection="\n",
    )
    assert queued_result.returncode == 0, queued_result.stderr
    _assert_wandb_forwarding(
        queued,
        queued_result,
        expected_key=expected_key,
        possible_keys=possible_keys,
    )

    interactive = _harness(tmp_path / "interactive", exported_key=exported_key, file_key=file_key)
    interactive_result = _run_dev(interactive)
    assert interactive_result.returncode == 0, interactive_result.stderr
    _assert_wandb_forwarding(
        interactive,
        interactive_result,
        expected_key=expected_key,
        possible_keys=possible_keys,
    )
