# ruff: noqa: PLR2004, S101, S603, S606
"""
Verify Docker and cluster launchers through isolated command stubs, never submission.

The harness covers canonical mounts/environment roots, GPU discovery/selection,
quoting, queue arguments, logging, wrapper validation, and early rejection of
invalid CPU/fallback requests. It deliberately does not run Docker, Slurm,
``runTSGPU.py``, training, or a real GPU workload.
"""

from __future__ import annotations

import os
import pty
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from support import configs

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PRODUCTION_CONFIG = configs.experiment_config_path(
    model_kind="fno",
    physics_enabled=False,
)
_OPTUNA_CONFIG = configs.optuna_config_path(
    model_kind="fno",
    physics_enabled=False,
)
_OPTUNA_SMOKE_CONFIG = configs.optuna_config_path(
    model_kind="fno",
    physics_enabled=False,
    role="smoke",
)
_CONFIG_ROOT_RELATIVE = _PRODUCTION_CONFIG.relative_to(
    _REPOSITORY_ROOT / "model_training" / "configs",
).as_posix()
_CONFIG_MODEL_TRAINING_RELATIVE = _PRODUCTION_CONFIG.relative_to(
    _REPOSITORY_ROOT / "model_training",
).as_posix()
_CONFIG_REPOSITORY_RELATIVE = _PRODUCTION_CONFIG.relative_to(_REPOSITORY_ROOT).as_posix()
_OPTUNA_MODEL_TRAINING_RELATIVE = _OPTUNA_CONFIG.relative_to(
    _REPOSITORY_ROOT / "model_training",
).as_posix()
_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE = _OPTUNA_SMOKE_CONFIG.relative_to(
    _REPOSITORY_ROOT / "model_training",
).as_posix()
_OPTUNA_SMOKE_REPOSITORY_RELATIVE = _OPTUNA_SMOKE_CONFIG.relative_to(_REPOSITORY_ROOT).as_posix()
_SPACED_CONFIG_RELATIVE = "configs/tasks/steady_flow/experiments/best_of_class/config with spaces.yaml"
_FNO_SEARCH_REPOSITORY_RELATIVE = "model_training/configs/tasks/steady_flow/optuna/fno_search.yaml"
_UNO_REPOSITORY_RELATIVE = "model_training/configs/tasks/steady_flow/experiments/best_of_class/uno_m64x64_h32_l7_mr0p495.yaml"
_UNO_CONFIG = _REPOSITORY_ROOT / _UNO_REPOSITORY_RELATIVE
_UNO_MODEL_TRAINING_RELATIVE = _UNO_CONFIG.relative_to(_REPOSITORY_ROOT / "model_training").as_posix()


@dataclass(frozen=True)
class _Harness:
    """
    Hold one immutable isolated launcher-test environment.

    Attributes
    ----------
    repository : pathlib.Path
        Temporary repository copy containing only the launcher scripts and configs.
    environment : dict[str, str]
        Subprocess environment that routes external commands to local stubs.
    binary_dir : pathlib.Path
        Directory containing the fake ``docker``, ``nvidia-smi``, and queue commands.
    home : pathlib.Path
        Isolated home used for optional credential-file precedence.
    launcher capture paths : pathlib.Path
        NUL-delimited argument/environment and invocation capture files written by stubs.

    """

    repository: Path
    environment: dict[str, str]
    binary_dir: Path
    home: Path
    runtsgpu_capture: Path
    docker_capture: Path
    preflight_docker_capture: Path
    docker_environment_capture: Path
    tail_capture: Path
    nvidia_capture: Path
    host_python_capture: Path


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
    gpu_report: str | None = None,
) -> _Harness:
    """
    Create safe command stubs around copied launcher scripts and minimal configs.

    The returned harness records argv, environment, output streams, and exit codes
    without invoking Docker, a scheduler, or GPU hardware. Each call owns an isolated
    repository, storage root, home, and PATH.
    """
    repository = tmp_path / "repository with spaces"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    for name in ("docker_job.sh", "_docker_run.sh", "docker_dev.sh", "config_preflight_runtime.py"):
        shutil.copy2(_REPOSITORY_ROOT / "scripts" / name, scripts / name)

    model_training = repository / "model_training"
    shutil.copytree(_REPOSITORY_ROOT / "model_training" / "src", model_training / "src")
    config_payloads = {
        model_training / _CONFIG_MODEL_TRAINING_RELATIVE: _PRODUCTION_CONFIG.read_bytes(),
        model_training / _SPACED_CONFIG_RELATIVE: _PRODUCTION_CONFIG.read_bytes(),
        model_training / _OPTUNA_MODEL_TRAINING_RELATIVE: _OPTUNA_CONFIG.read_bytes(),
        model_training / _OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE: _OPTUNA_SMOKE_CONFIG.read_bytes(),
        model_training / _UNO_MODEL_TRAINING_RELATIVE: _UNO_CONFIG.read_bytes(),
    }
    for config_path, payload in config_payloads.items():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(payload)

    binary_dir = tmp_path / "stub commands"
    binary_dir.mkdir()
    _write_executable(
        binary_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" > "${HOST_PYTHON_CAPTURE}"
if [[ "${1-}" == "-c" ]]; then
  printf '%s\\n' "${HOST_PYTHON_STUB_VERSION:-3.9.19}"
  exit "${HOST_PYTHON_VERSION_EXIT_CODE:-0}"
fi
echo 'host Python was asked to import project code directly' >&2
exit 97
""",
    )
    report = gpu_report if gpu_report is not None else ("0, Cluster GPU A, 20, 7000, 24000\n2, Cluster GPU B, 5, 1000, 24000\n")
    _write_executable(
        binary_dir / "nvidia-smi",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'called\n' > "${{NVIDIA_CAPTURE}}"
if [[ "${{1-}}" == "-L" ]]; then
  printf 'GPU 0: Cluster GPU A\\nGPU 2: Cluster GPU B\\n'
elif [[ "$*" == *"--query-gpu=index,name,utilization.gpu,memory.used,memory.total"* ]]; then
  printf '%b' {report!r}
else
  printf 'unexpected nvidia-smi arguments: %s\\n' "$*" >&2
  exit 64
fi
""",
    )
    _write_executable(
        binary_dir / "runTSGPU.py",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" > "${RUNTSGPU_CAPTURE}"
if (( ${QUEUE_SUBMISSION_EXIT:-0} != 0 )); then
  printf '%b' "${QUEUE_PARTIAL_OUTPUT-}"
  echo 'stub queue submission refused' >&2
  exit "${QUEUE_SUBMISSION_EXIT}"
fi
while (( $# > 0 )); do
  if [[ "$1" == "--" ]]; then
    shift
    set +e
    "$@"
    set -e
    if [[ -n "${QUEUE_SUBMISSION_OUTPUT+x}" ]]; then
      printf '%b' "${QUEUE_SUBMISSION_OUTPUT}"
    else
      printf 'TS socket: %s\n' "${TS_SOCKET:-/etc/ts/socket_unknown}"
      printf '%s\n' "${QUEUE_JOB_ID:-25}"
    fi
    exit 0
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
case "${1-}" in
  info)
    if [[ "$*" == *"--format"* ]]; then
      printf '{"nvidia": {}}\\n'
    fi
    exit "${DOCKER_INFO_EXIT_CODE:-0}"
    ;;
  image)
    if [[ "${DOCKER_IMAGE_AVAILABLE:-true}" == true ]]; then
      exit 0
    fi
    exit 1
    ;;
  ps)
    if [[ "$*" == *" -a "* || "$*" == *"ps -a"* ]]; then
      printf '%s' "${DOCKER_ALL_NAMES-}"
    else
      printf '%s' "${DOCKER_RUNNING_NAMES-}"
    fi
    exit 0
    ;;
  start)
    exit 0
    ;;
  run)
    arguments=("$@")
    bootstrap_index=-1
    for index in "${!arguments[@]}"; do
      if [[ "${arguments[index]}" == "/workspace/repo/scripts/config_preflight_runtime.py" ]]; then
        bootstrap_index="${index}"
        break
      fi
    done
    if (( bootstrap_index >= 0 )); then
      printf '%s\\0' "$@" > "${PREFLIGHT_DOCKER_CAPTURE}"
      if (( ${PREFLIGHT_CONTAINER_EXIT_CODE:-0} != 0 )); then
        printf '%s' "${PREFLIGHT_CONTAINER_STDOUT-}"
        printf '%s' "${PREFLIGHT_CONTAINER_STDERR-}" >&2
        exit "${PREFLIGHT_CONTAINER_EXIT_CODE}"
      fi
      workflow="${arguments[bootstrap_index + 1]}"
      config="${arguments[bootstrap_index + 2]}"
      case "${config}" in
        configs/*)
          host_config="${PROJECT_ROOT}/model_training/${config}"
          ;;
        /workspace/repo/model_training/data/*)
          host_config="${STORAGE_ROOT}/data_training/${config#/workspace/repo/model_training/data/}"
          ;;
        /workspace/repo/data_generation/data/*)
          host_config="${STORAGE_ROOT}/data_generation/${config#/workspace/repo/data_generation/data/}"
          ;;
        /workspace/repo/*)
          host_config="${PROJECT_ROOT}/${config#/workspace/repo/}"
          ;;
        *)
          host_config="${config}"
          ;;
      esac
      (
        cd "${PROJECT_ROOT}/model_training"
        export PYTHONPATH="${PROJECT_ROOT}/model_training"
        export PROJECT_ROOT
        export GENERATED_DATA_ROOT="${STORAGE_ROOT}/data_generation"
        export MODEL_TRAINING_DATA_ROOT="${STORAGE_ROOT}/data_training"
        "${CONTAINER_PYTHON}" "${PROJECT_ROOT}/scripts/config_preflight_runtime.py" "${workflow}" "${host_config}"
      )
      exit $?
    fi
    printf '%s\\0' "$@" > "${DOCKER_CAPTURE}"
    printf '%s' "${WANDB_API_KEY-<unset>}" > "${DOCKER_ENV_CAPTURE}"
    printf 'captured Docker stdout with spaces\\n'
    printf 'captured Docker stderr with spaces\\n' >&2
    exit "${DOCKER_EXIT_CODE:-0}"
    ;;
  build)
    exit 0
    ;;
  *)
    printf 'unexpected Docker command: %s\\n' "$*" >&2
    exit 66
    ;;
esac
""",
    )

    _write_executable(
        binary_dir / "tail",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" > "${TAIL_CAPTURE}"
case "${TAIL_MODE:-return}" in
  return)
    exit 0
    ;;
  interrupt)
    kill -INT "$PPID"
    exit 130
    ;;
  fail)
    exit 42
    ;;
  *)
    echo "unknown TAIL_MODE: ${TAIL_MODE}" >&2
    exit 64
    ;;
esac
""",
    )

    home = tmp_path / "home without ssh"
    home.mkdir()
    if file_key is not None:
        (home / "wandb_key.txt").write_text(file_key, encoding="utf-8")

    fallback_binary_dir = tmp_path / "fallback commands"
    fallback_binary_dir.mkdir()
    for command in (
        "bash",
        "basename",
        "cat",
        "chmod",
        "date",
        "dirname",
        "env",
        "grep",
        "id",
        "mkdir",
        "mktemp",
        "realpath",
    ):
        resolved = shutil.which(command)
        assert resolved is not None
        (fallback_binary_dir / command).symlink_to(resolved)

    storage_root = tmp_path / "storage root with spaces"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{binary_dir}{os.pathsep}{fallback_binary_dir}",
            "HOME": str(home),
            "PROJECT_ROOT": str(repository),
            "STORAGE_ROOT": str(storage_root),
            "RUNTSGPU_CAPTURE": str(tmp_path / "runtsgpu.args"),
            "DOCKER_CAPTURE": str(tmp_path / "docker.args"),
            "PREFLIGHT_DOCKER_CAPTURE": str(tmp_path / "preflight-docker.args"),
            "DOCKER_ENV_CAPTURE": str(tmp_path / "docker.env"),
            "TAIL_CAPTURE": str(tmp_path / "tail.args"),
            "NVIDIA_CAPTURE": str(tmp_path / "nvidia.called"),
            "HOST_PYTHON_CAPTURE": str(tmp_path / "host-python.args"),
            "HOST_PYTHON_STUB_VERSION": "3.9.19",
            "CONTAINER_PYTHON": sys.executable,
            "QUEUE_JOB_ID": "25",
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
        binary_dir=binary_dir,
        home=home,
        runtsgpu_capture=Path(environment["RUNTSGPU_CAPTURE"]),
        docker_capture=Path(environment["DOCKER_CAPTURE"]),
        preflight_docker_capture=Path(environment["PREFLIGHT_DOCKER_CAPTURE"]),
        docker_environment_capture=Path(environment["DOCKER_ENV_CAPTURE"]),
        tail_capture=Path(environment["TAIL_CAPTURE"]),
        nvidia_capture=Path(environment["NVIDIA_CAPTURE"]),
        host_python_capture=Path(environment["HOST_PYTHON_CAPTURE"]),
    )


def _run_job(
    harness: _Harness,
    *arguments: str,
    selection: str = "",
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


def _run_job_tty(
    harness: _Harness,
    *arguments: str,
    selection: bytes = b"\n",
) -> subprocess.CompletedProcess[str]:
    """Run the wrapper with a controlling pseudo-terminal and scripted input."""
    command = [str(harness.repository / "scripts" / "docker_job.sh"), *arguments]
    pid, controller = pty.fork()
    if pid == 0:
        os.chdir(harness.repository)
        os.execve(command[0], command, harness.environment)

    output = bytearray()
    selection_sent = False
    try:
        while True:
            try:
                chunk = os.read(controller, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            if not selection_sent and b"Select GPU (" in output:
                os.write(controller, selection)
                selection_sent = True
    finally:
        os.close(controller)
    _, wait_status = os.waitpid(pid, 0)
    return subprocess.CompletedProcess(
        command,
        os.waitstatus_to_exitcode(wait_status),
        output.decode(errors="replace").replace("\r\n", "\n"),
        "",
    )


def _run_dev(harness: _Harness) -> subprocess.CompletedProcess[str]:
    """Run the copied development launcher against the isolated Docker stub."""
    return subprocess.run(
        [str(harness.repository / "scripts" / "docker_dev.sh")],
        cwd=harness.repository,
        env=harness.environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _queue_log_dir(harness: _Harness) -> Path:
    """Return the host-visible processed-domain queue-log directory."""
    return Path(harness.environment["STORAGE_ROOT"]) / "data_training" / "processed" / "steady_flow" / "logs" / "queue"


def _log_path(harness: _Harness) -> Path:
    """Return the single queue log created by one harness invocation."""
    logs = list(_queue_log_dir(harness).glob("*.log"))
    assert len(logs) == 1
    return logs[0]


def _assert_preflight_container(
    harness: _Harness,
    *,
    workflow: str,
    config_path: str,
) -> None:
    """Verify one read-only, CPU-only, network-disabled authoritative preflight."""
    arguments = _capture_arguments(harness.preflight_docker_capture)
    assert arguments[:2] == ["run", "--rm"]
    assert arguments[arguments.index("--network") + 1] == "none"
    assert arguments[arguments.index("--workdir") + 1] == "/workspace/repo/model_training"
    assert "--gpus" not in arguments
    assert "WANDB_API_KEY" not in arguments
    assert not any(argument.startswith("WANDB_") for argument in arguments)
    assert f"type=bind,source={harness.repository},target=/workspace/repo,readonly" in arguments
    assert arguments[-5:] == [
        "grainlegumes-pino-airflow",
        "python",
        "/workspace/repo/scripts/config_preflight_runtime.py",
        workflow,
        config_path,
    ]


def _without_device(arguments: list[str]) -> list[str]:
    """
    Remove the already-validated semantic device option from forwarded arguments.

    Split and equals spellings are handled here; wrapper validation owns malformed
    or duplicate cases before the inner command is normalized to strict CUDA.
    """
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        if arguments[index] == "--device":
            index += 2
        elif arguments[index].startswith("--device="):
            index += 1
        else:
            cleaned.append(arguments[index])
            index += 1
    return cleaned


def _assert_wandb_forwarding(
    harness: _Harness,
    result: subprocess.CompletedProcess[str],
    *,
    expected_key: str | None,
    possible_keys: tuple[str | None, ...],
) -> None:
    """
    Verify name-only Docker credential forwarding and output redaction.

    Captured stub environment may contain the selected key for assertion, but queue,
    Docker argv, launcher streams, and logs must never expose any candidate value.
    """
    docker = _capture_arguments(harness.docker_capture)
    expected_capture = expected_key if expected_key is not None else "<unset>"
    assert harness.docker_environment_capture.read_text(encoding="utf-8") == expected_capture
    assert ("WANDB_API_KEY" in docker) is (expected_key is not None)
    assert not any(argument.startswith("WANDB_API_KEY=") for argument in docker)

    log_text = "\n".join(path.read_text(encoding="utf-8") for path in _queue_log_dir(harness).glob("*.log"))
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
    """
    Verify the complete stubbed queue-to-semantic-CLI argument chain.

    The helper asserts GPU binding, unique log identity, canonical storage mounts,
    environment roots, W&B name forwarding, exact module selection, and inner
    strict-CUDA normalization; it returns only the allocated log basename.
    """
    assert result.returncode == 0, result.stderr
    runtsgpu = _capture_arguments(harness.runtsgpu_capture)
    assert runtsgpu[:3] == [f"-g{gpu}", "--", str(harness.repository / "scripts" / "_docker_run.sh")]
    assert runtsgpu[3:5] == [gpu, job_type]
    log_basename = runtsgpu[5]
    assert re.fullmatch(rf"\d{{8}}_\d{{6}}__{job_type}__gpu{gpu}__[A-Za-z0-9]{{6}}\.log", log_basename)
    assert runtsgpu[6:] == command_arguments
    if job_type in {"train", "optuna"}:
        _assert_preflight_container(
            harness,
            workflow=job_type,
            config_path=command_arguments[0],
        )

    docker = _capture_arguments(harness.docker_capture)
    assert docker[:2] == ["run", "--rm"]
    assert docker[docker.index("--gpus") + 1] == f"device={gpu}"
    assert docker[docker.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert docker[docker.index("--workdir") + 1] == "/workspace/repo/model_training"
    for expected in (
        "PROJECT_ROOT=/workspace/repo",
        "GENERATED_DATA_ROOT=/workspace/repo/data_generation/data",
        "MODEL_TRAINING_DATA_ROOT=/workspace/repo/model_training/data",
    ):
        assert expected in docker
    storage_root = Path(harness.environment["STORAGE_ROOT"])
    project_mount = f"{harness.repository}:/workspace/repo:rw"
    generated_mount = f"{storage_root / 'data_generation'}:/workspace/repo/data_generation/data:ro"
    training_mount = f"{storage_root / 'data_training'}:/workspace/repo/model_training/data:rw"
    docker_home_mount = f"{storage_root / '.docker_home'}:/workspace/storage/.docker_home:rw"
    passwd_mount = f"{storage_root / '.docker_home' / 'passwd'}:/etc/passwd:ro"
    group_mount = f"{storage_root / '.docker_home' / 'group'}:/etc/group:ro"
    assert project_mount in docker
    assert generated_mount in docker
    assert training_mount in docker
    assert docker_home_mount in docker
    assert passwd_mount in docker
    assert group_mount in docker
    assert (harness.repository / "data_generation" / "data").is_dir()
    assert (harness.repository / "model_training" / "data").is_dir()
    assert f"{storage_root}:/workspace/storage:rw" not in docker
    assert not any(
        argument.startswith(("STORAGE_ROOT=", "DATA_ROOT=", "DATASET_ROOT=", "OUTPUT_ROOT=", "GEN_ROOT=", "TRAIN_ROOT=")) for argument in docker
    )
    inner_arguments = [*_without_device(command_arguments), "--device", "cuda"]
    assert docker[-(len(inner_arguments) + 3) :] == ["python", "-m", module, *inner_arguments]

    log_path = _queue_log_dir(harness) / log_basename
    log_text = log_path.read_text(encoding="utf-8")
    assert "captured Docker stdout with spaces" in log_text
    assert "captured Docker stderr with spaces" in log_text
    assert "Docker exit status: 0" in log_text
    assert f"Selected host GPU: {gpu}" in log_text
    assert f"CUDA_VISIBLE_DEVICES: {gpu}" in log_text
    assert "Container CUDA device: 0" in log_text
    assert f"Task-spooler socket: /etc/ts/socket_{gpu}" in log_text
    assert "Current GPU usage:" in result.stdout
    assert "Queue job ID: 25" in result.stdout
    assert f"Workflow: {job_type}" in result.stdout
    assert "Task: steady_flow" in result.stdout
    assert f"Selected host GPU: {gpu}" in result.stdout
    assert f"CUDA_VISIBLE_DEVICES: {gpu}" in result.stdout
    assert "Container CUDA device: 0" in result.stdout
    assert f"Task-spooler socket: /etc/ts/socket_{gpu}" in result.stdout
    assert "Queued command:" in result.stdout
    assert f"Host log: {log_path}" in result.stdout
    assert "Follow manually:" in result.stdout
    assert "tail -n +1 -F" in result.stdout
    assert not harness.tail_capture.exists()
    return log_basename


def test_default_tty_enter_accepts_proposal_and_preserves_training_arguments(tmp_path: Path) -> None:
    """Prompt in a TTY and accept the least-memory proposal with Enter."""
    harness = _harness(tmp_path)
    arguments = [
        _SPACED_CONFIG_RELATIVE,
        "--resume",
        "/workspace/repo/model_training/data/processed/steady_flow/runs/run with spaces",
        "--device",
        "cuda",
    ]
    result = _run_job_tty(harness, "train", *arguments)

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="train",
        module="src.experiments.cli.cli_train",
        command_arguments=arguments,
    )
    assert "Proposed GPU: 2" in result.stdout
    assert "Select GPU (0,2; Enter for proposed 2):" in result.stdout
    assert "Automatically selected GPU" not in result.stdout


def test_default_tty_manual_override_and_invalid_reprompt(tmp_path: Path) -> None:
    """Reject malformed/unavailable choices, then accept a different reported GPU."""
    harness = _harness(tmp_path)
    result = _run_job_tty(
        harness,
        "optuna",
        _OPTUNA_SMOKE_REPOSITORY_RELATIVE,
        selection=b"not-a-gpu\n7\n 0 \n",
    )
    _assert_common_chain(
        harness,
        result,
        gpu="0",
        job_type="optuna",
        module="src.experiments.cli.cli_optuna",
        command_arguments=[_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE],
    )
    assert "Invalid GPU selection" in result.stdout
    assert "not one of the reported indices" in result.stdout
    assert result.stdout.count("Select GPU") == 3
    assert "Selected GPU: 0" in result.stdout


def test_default_noninteractive_requires_explicit_gpu_mode(tmp_path: Path) -> None:
    """Fail closed after GPU discovery when stdin is not a TTY and no mode is explicit."""
    harness = _harness(tmp_path)
    result = _run_job(harness, "train", _CONFIG_MODEL_TRAINING_RELATIVE)

    assert result.returncode == 2
    assert "GPU selection requires an explicit option in non-interactive mode." in result.stderr
    assert "--queue-gpu auto" in result.stderr
    assert "--queue-gpu INDEX" in result.stderr
    assert harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_default_tty_eof_fails_without_submission(tmp_path: Path) -> None:
    """Treat terminal EOF as cancellation rather than accepting the proposal."""
    harness = _harness(tmp_path)
    result = _run_job_tty(harness, "optuna", _OPTUNA_SMOKE_REPOSITORY_RELATIVE, selection=b"\x04")

    assert result.returncode == 2
    assert "input closed" in result.stdout
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_default_tty_ctrl_c_submits_nothing(tmp_path: Path) -> None:
    """Abort an active selection prompt on Ctrl+C without allocating queue state."""
    harness = _harness(tmp_path)
    result = _run_job_tty(harness, "train", _CONFIG_MODEL_TRAINING_RELATIVE, selection=b"\x03")

    assert result.returncode == 130
    assert "selection cancelled" in result.stdout
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_explicit_gpu_after_config_queues_optuna_with_runtime_overrides(tmp_path: Path) -> None:
    """
    Select a reported GPU explicitly after an Optuna config with runtime overrides.

    The wrapper option is stripped while every semantic argument reaches the maintained CLI.
    """
    harness = _harness(tmp_path)
    arguments = [
        _OPTUNA_MODEL_TRAINING_RELATIVE,
        "--n-trials",
        "3",
        "--output-root",
        "/workspace/repo/model_training/data/processed/output root",
        "--show-progress-bar",
    ]
    result = _run_job(harness, "optuna", *arguments, "--queue-gpu", "0")

    _assert_common_chain(
        harness,
        result,
        gpu="0",
        job_type="optuna",
        module="src.experiments.cli.cli_optuna",
        command_arguments=arguments,
    )
    assert "Selection source: explicit --queue-gpu" in result.stdout
    assert "Automatically selected GPU" not in result.stdout
    assert "Select GPU" not in result.stdout


@pytest.mark.parametrize(
    ("workflow", "repository_config", "container_config", "module"),
    [
        (
            "optuna",
            _FNO_SEARCH_REPOSITORY_RELATIVE,
            _OPTUNA_MODEL_TRAINING_RELATIVE,
            "src.experiments.cli.cli_optuna",
        ),
        (
            "train",
            _UNO_REPOSITORY_RELATIVE,
            _UNO_MODEL_TRAINING_RELATIVE,
            "src.experiments.cli.cli_train",
        ),
    ],
)
def test_exact_production_commands_reach_mocked_queue(
    tmp_path: Path,
    workflow: str,
    repository_config: str,
    container_config: str,
    module: str,
) -> None:
    """Admit the documented production search and UNO train commands through stubs."""
    harness = _harness(tmp_path)
    result = _run_job_tty(harness, workflow, repository_config)

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type=workflow,
        module=module,
        command_arguments=[container_config],
    )


def test_exact_fno_smoke_detached_command_parses_without_real_submission(tmp_path: Path) -> None:
    """Parse the documented detached smoke command through isolated queue/Docker stubs."""
    harness = _harness(tmp_path)
    result = _run_job_tty(harness, "optuna", _OPTUNA_SMOKE_REPOSITORY_RELATIVE)

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="optuna",
        module="src.experiments.cli.cli_optuna",
        command_arguments=[_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE],
    )


def test_exact_fno_smoke_follow_command_keeps_follow_host_only(tmp_path: Path) -> None:
    """Parse the documented followed smoke command without forwarding its wrapper flag."""
    harness = _harness(tmp_path)
    result = _run_job(
        harness,
        "optuna",
        _OPTUNA_SMOKE_REPOSITORY_RELATIVE,
        "--queue-gpu",
        "auto",
        "--follow",
    )

    assert result.returncode == 0, result.stderr
    runtsgpu = _capture_arguments(harness.runtsgpu_capture)
    docker = _capture_arguments(harness.docker_capture)
    assert runtsgpu[6:] == [_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE]
    assert docker[-6:] == [
        "python",
        "-m",
        "src.experiments.cli.cli_optuna",
        _OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE,
        "--device",
        "cuda",
    ]
    assert "--follow" not in runtsgpu
    assert "--follow" not in docker
    _assert_preflight_container(
        harness,
        workflow="optuna",
        config_path=_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE,
    )
    assert "--follow" not in _capture_arguments(harness.preflight_docker_capture)
    assert _capture_arguments(harness.tail_capture) == ["-n", "+1", "-F", str(_log_path(harness))]


def test_exact_uno_follow_command_keeps_follow_host_only(tmp_path: Path) -> None:
    """Admit the exact UNO command while retaining follow entirely on the host."""
    harness = _harness(tmp_path)
    result = _run_job(harness, "train", _UNO_REPOSITORY_RELATIVE, "--queue-gpu", "auto", "--follow")

    assert result.returncode == 0, result.stderr
    _assert_preflight_container(
        harness,
        workflow="train",
        config_path=_UNO_MODEL_TRAINING_RELATIVE,
    )
    assert _capture_arguments(harness.runtsgpu_capture)[6:] == [_UNO_MODEL_TRAINING_RELATIVE]
    assert "--follow" not in _capture_arguments(harness.docker_capture)
    assert _capture_arguments(harness.tail_capture) == ["-n", "+1", "-F", str(_log_path(harness))]


@pytest.mark.parametrize(
    ("wrapper_arguments", "expected_gpu"),
    [
        (["--queue-gpu", "auto"], "2"),
        (["--queue-gpu", "0"], "0"),
    ],
    ids=("auto", "explicit"),
)
def test_noninteractive_gpu_modes_preserve_artifact_arguments(
    tmp_path: Path,
    wrapper_arguments: list[str],
    expected_gpu: str,
) -> None:
    """
    Vary noninteractive queue selection between ``auto`` and an explicit reported index.

    Both wrapper modes must avoid prompting and preserve repeated artifact arguments,
    while only the expected scheduler GPU changes.
    """
    harness = _harness(tmp_path)
    arguments = ["--task", "steady_flow", "--run-name", "first run", "--run-name", "second run"]
    result = _run_job(harness, *wrapper_arguments, "artifacts", *arguments)

    _assert_common_chain(
        harness,
        result,
        gpu=expected_gpu,
        job_type="artifacts",
        module="src.experiments.cli.cli_build_artifacts",
        command_arguments=arguments,
    )
    assert "Select GPU" not in result.stdout
    if wrapper_arguments[-1] == "auto":
        assert f"Automatically selected GPU: {expected_gpu}" in result.stdout
    else:
        assert "Selection source: explicit --queue-gpu" in result.stdout


def test_least_memory_tie_uses_lowest_reported_index(tmp_path: Path) -> None:
    """
    Report two GPUs with equal allocated memory in reverse numeric order.

    Automatic selection must choose the lowest reported index, providing a stable
    tie-break independent of report order or utilization percentage.
    """
    report = "7, GPU Seven, 1, 1000, 24000\n3, GPU Three, 9, 1000, 24000\n"
    harness = _harness(tmp_path, gpu_report=report)
    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _CONFIG_MODEL_TRAINING_RELATIVE,
    )

    assert result.returncode == 0, result.stderr
    assert "Automatically selected GPU: 3" in result.stdout
    assert "Reason: least allocated memory; lowest index breaks ties" in result.stdout
    assert _capture_arguments(harness.runtsgpu_capture)[0] == "-g3"


def test_repository_relative_config_path_is_validated_and_mapped(tmp_path: Path) -> None:
    """
    Submit a repository-root-relative experiment config that exists on the host.

    The wrapper must validate it and map it to the model-training container cwd,
    avoiding both host-path leakage and a duplicated ``model_training`` prefix.
    """
    harness = _harness(tmp_path)
    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _CONFIG_REPOSITORY_RELATIVE,
    )

    assert result.returncode == 0, result.stderr
    runtsgpu = _capture_arguments(harness.runtsgpu_capture)
    assert runtsgpu[6] == _CONFIG_MODEL_TRAINING_RELATIVE


@pytest.mark.parametrize("host_version", ["3.8.20", "3.9.19"])
def test_old_host_python_only_reports_version_and_uses_container(
    tmp_path: Path,
    host_version: str,
) -> None:
    """Never import project modules with old host Python; use the guarded image runtime."""
    harness = _harness(tmp_path)
    harness.environment["HOST_PYTHON_STUB_VERSION"] = host_version
    result = _run_job(harness, "--queue-gpu", "auto", "optuna", _OPTUNA_SMOKE_REPOSITORY_RELATIVE)

    assert result.returncode == 0, result.stderr
    host_arguments = _capture_arguments(harness.host_python_capture)
    assert host_arguments[0] == "-c"
    assert len(host_arguments) == 2
    assert "src" not in host_arguments[1]
    assert "below the project minimum" in result.stderr
    assert "maintained CPU-only project container" in result.stderr
    assert "dataclass() got an unexpected keyword argument 'slots'" not in result.stdout + result.stderr
    _assert_preflight_container(
        harness,
        workflow="optuna",
        config_path=_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE,
    )


def test_supported_host_python_is_still_not_an_implicit_project_runtime(tmp_path: Path) -> None:
    """Use the maintained image even when arbitrary host Python happens to be new enough."""
    harness = _harness(tmp_path)
    harness.environment["HOST_PYTHON_STUB_VERSION"] = "3.11.15"
    result = _run_job(harness, "--queue-gpu", "auto", "train", _UNO_REPOSITORY_RELATIVE)

    assert result.returncode == 0, result.stderr
    assert "below the project minimum" not in result.stderr
    assert _capture_arguments(harness.host_python_capture)[0] == "-c"
    _assert_preflight_container(
        harness,
        workflow="train",
        config_path=_UNO_MODEL_TRAINING_RELATIVE,
    )


def test_host_python_is_optional_when_container_runtime_is_available(tmp_path: Path) -> None:
    """Require neither a host installation nor shell activation for project validation."""
    harness = _harness(tmp_path)
    (harness.binary_dir / "python").unlink()
    result = _run_job(harness, "--queue-gpu", "auto", "optuna", _OPTUNA_SMOKE_REPOSITORY_RELATIVE)

    assert result.returncode == 0, result.stderr
    assert not harness.host_python_capture.exists()
    _assert_preflight_container(
        harness,
        workflow="optuna",
        config_path=_OPTUNA_SMOKE_MODEL_TRAINING_RELATIVE,
    )


def test_missing_container_runtime_has_actionable_version_error_before_gpu(tmp_path: Path) -> None:
    """Fail with host/project/runtime facts when the approved image is unavailable."""
    harness = _harness(tmp_path)
    harness.environment["DOCKER_IMAGE_AVAILABLE"] = "false"
    result = _run_job(harness, "train", _UNO_REPOSITORY_RELATIVE)

    assert result.returncode == 1
    assert "Configuration preflight could not start" in result.stderr
    assert "Host Python: 3.9.19" in result.stderr
    assert "Required Python: >=3.11" in result.stderr
    assert "image 'grainlegumes-pino-airflow' is missing" in result.stderr
    assert "slots" not in result.stderr
    assert not harness.preflight_docker_capture.exists()
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_container_preflight_status_and_streams_propagate_before_gpu(tmp_path: Path) -> None:
    """Return an explicit container failure status and both streams without queue state."""
    harness = _harness(tmp_path)
    harness.environment.update(
        {
            "PREFLIGHT_CONTAINER_EXIT_CODE": "73",
            "PREFLIGHT_CONTAINER_STDOUT": "preflight stdout marker\n",
            "PREFLIGHT_CONTAINER_STDERR": "preflight stderr marker\n",
        }
    )
    result = _run_job(harness, "optuna", _OPTUNA_SMOKE_REPOSITORY_RELATIVE)

    assert result.returncode == 73
    assert "preflight stdout marker" in result.stdout
    assert "preflight stderr marker" in result.stderr
    assert harness.preflight_docker_capture.exists()
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_malformed_yaml_fails_in_authoritative_container_before_gpu(tmp_path: Path) -> None:
    """Reject malformed YAML through project validation without worker-side state."""
    harness = _harness(tmp_path)
    relative = "configs/tasks/steady_flow/experiments/malformed.yaml"
    path = harness.repository / "model_training" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("task: [unterminated\n", encoding="utf-8")

    result = _run_job(harness, "train", relative)

    assert result.returncode == 2
    assert "YAML" in result.stderr or "while parsing" in result.stderr
    _assert_preflight_container(harness, workflow="train", config_path=relative)
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_obsolete_optuna_path_remains_missing_before_any_runtime(tmp_path: Path) -> None:
    """Keep the removed pre-task-first Optuna path absent without an alias."""
    harness = _harness(tmp_path)
    old_path = "model_training/configs/optuna/steady_flow_fno_search.yaml"
    result = _run_job(harness, "optuna", old_path)

    assert result.returncode == 2
    assert "Config path does not exist" in result.stderr
    assert not (harness.repository / old_path).exists()
    assert not harness.host_python_capture.exists()
    assert not harness.preflight_docker_capture.exists()
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not _queue_log_dir(harness).exists()


def test_obsolete_low_capacity_path_remains_missing_before_any_runtime(tmp_path: Path) -> None:
    """Keep the migrated experiment path absent without a compatibility alias."""
    harness = _harness(tmp_path)
    old_path = "model_training/configs/tasks/steady_flow/experiments/low_capacity/fno_low_capacity_m12x12_h24_l4.yaml"
    result = _run_job(harness, "train", old_path)

    assert result.returncode == 2
    assert "Config path does not exist" in result.stderr
    assert not (harness.repository / old_path).exists()
    assert not harness.host_python_capture.exists()
    assert not harness.preflight_docker_capture.exists()
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not _queue_log_dir(harness).exists()


@pytest.mark.parametrize(
    ("workflow", "config_path", "detected_family", "corrected_workflow"),
    [
        ("train", _OPTUNA_MODEL_TRAINING_RELATIVE, "optuna", "optuna"),
        ("optuna", _CONFIG_MODEL_TRAINING_RELATIVE, "experiment", "train"),
    ],
)
def test_wrong_workflow_fails_before_gpu_queue_and_log_allocation(
    tmp_path: Path,
    workflow: str,
    config_path: str,
    detected_family: str,
    corrected_workflow: str,
) -> None:
    """Reject both inverse schema misuses before any GPU, queue, Docker, or log state."""
    harness = _harness(tmp_path)
    result = _run_job(harness, workflow, config_path)

    assert result.returncode == 2
    assert f"Supplied config family: {detected_family}" in result.stderr
    assert f"Requested workflow: {workflow}" in result.stderr
    assert f"./scripts/docker_job.sh {corrected_workflow} 'model_training/configs/tasks/" in result.stderr
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    _assert_preflight_container(harness, workflow=workflow, config_path=config_path)
    assert not _queue_log_dir(harness).exists()


@pytest.mark.parametrize(
    ("workflow", "config_path"),
    [
        ("train", _CONFIG_MODEL_TRAINING_RELATIVE),
        ("optuna", _OPTUNA_MODEL_TRAINING_RELATIVE),
    ],
)
def test_follow_is_host_only_and_targets_the_exact_delayed_log(
    tmp_path: Path,
    workflow: str,
    config_path: str,
) -> None:
    """Follow the exact host log with tail -n +1 -F without forwarding the flag."""
    harness = _harness(tmp_path)
    result = _run_job(harness, workflow, config_path, "--queue-gpu", "auto", "--follow")

    assert result.returncode == 0, result.stderr
    assert "Following host log. Press Ctrl+C to stop following; the queue job continues." in result.stdout
    assert "Log following ended. Queue job 25 continues independently." in result.stdout
    assert _capture_arguments(harness.tail_capture) == ["-n", "+1", "-F", str(_log_path(harness))]
    assert "--follow" not in _capture_arguments(harness.runtsgpu_capture)
    assert "--follow" not in _capture_arguments(harness.docker_capture)


def test_follow_ctrl_c_stops_only_tail_and_returns_cleanly(tmp_path: Path) -> None:
    """A follower interrupt must leave the already submitted queue job independent."""
    harness = _harness(tmp_path)
    harness.environment["TAIL_MODE"] = "interrupt"
    result = _run_job(
        harness,
        "train",
        _CONFIG_MODEL_TRAINING_RELATIVE,
        "--queue-gpu",
        "auto",
        "--follow",
    )

    assert result.returncode == 0, result.stderr
    assert "Log following stopped. Queue job 25 continues independently." in result.stdout
    assert harness.runtsgpu_capture.exists()
    assert harness.docker_capture.exists()
    assert _capture_arguments(harness.tail_capture) == ["-n", "+1", "-F", str(_log_path(harness))]
    visible = result.stdout + result.stderr
    assert "cancel" not in visible.lower()
    assert "remove" not in visible.lower()


def test_help_documents_detached_workflows_and_follow_without_wait_mode(tmp_path: Path) -> None:
    """Expose train/Optuna/follow semantics without advertising completion waiting."""
    harness = _harness(tmp_path)
    result = _run_job(harness, "--help")

    assert result.returncode == 0
    help_text = result.stdout + result.stderr
    assert "train <experiment_config>" in help_text
    assert "optuna <optuna_config>" in help_text
    assert "--follow" in help_text
    assert "press Enter to accept the least-memory proposal" in help_text
    assert "Non-interactive callers must provide --queue-gpu auto or --queue-gpu INDEX." in help_text
    assert "Ctrl+C during GPU selection submits nothing" in help_text
    assert "Ctrl+C during log following stops only" in help_text
    assert "No --wait mode" in help_text
    assert "exists; the wrapper never polls" in help_text
    assert "\n  --wait" not in help_text
    assert not harness.nvidia_capture.exists()
    assert not harness.runtsgpu_capture.exists()


def test_queue_submission_errors_are_not_swallowed(tmp_path: Path) -> None:
    """Propagate scheduler submission failure without pretending a job was admitted."""
    harness = _harness(tmp_path)
    harness.environment["QUEUE_SUBMISSION_EXIT"] = "37"
    harness.environment["QUEUE_PARTIAL_OUTPUT"] = "partial scheduler diagnostic\n"
    result = _run_job(harness, "train", _CONFIG_MODEL_TRAINING_RELATIVE, "--queue-gpu", "auto")

    assert result.returncode == 37
    assert "partial scheduler diagnostic" in result.stderr
    assert "stub queue submission refused" in result.stderr
    assert "Queue submission failed for train workflow." in result.stderr
    assert "Queue job ID:" not in result.stdout
    assert harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


def test_queue_output_with_whitespace_and_diagnostics_keeps_authoritative_id(tmp_path: Path) -> None:
    """Extract one standalone numeric ID while preserving unrelated diagnostics."""
    harness = _harness(tmp_path)
    harness.environment["QUEUE_SUBMISSION_OUTPUT"] = "  TS socket: /etc/ts/socket_2  \nscheduler diagnostic with number 99 embedded\n  41  \n"
    result = _run_job(harness, "--queue-gpu", "auto", "optuna", _OPTUNA_SMOKE_REPOSITORY_RELATIVE)

    assert result.returncode == 0, result.stderr
    assert "Queue job ID: 41" in result.stdout
    assert "Queue submission diagnostics:" in result.stdout
    assert "scheduler diagnostic with number 99 embedded" in result.stdout
    assert "Queue submission output: 41" not in result.stdout
    assert "Task-spooler socket: /etc/ts/socket_2" in result.stdout


def test_generic_helper_socket_never_replaces_selected_gpu_socket(tmp_path: Path) -> None:
    """Retain the selected GPU-specific socket when helper output reports a stale generic alias."""
    harness = _harness(tmp_path)
    harness.environment["QUEUE_SUBMISSION_OUTPUT"] = "TS socket: /etc/ts/socket\n52\n"
    result = _run_job(harness, "--queue-gpu", "2", "train", _CONFIG_MODEL_TRAINING_RELATIVE)

    assert result.returncode == 0, result.stderr
    assert "Queue job ID: 52" in result.stdout
    assert "did not match selected GPU socket" in result.stdout
    assert "Task-spooler socket: /etc/ts/socket_2" in result.stdout
    assert "Task-spooler socket: /etc/ts/socket\n" not in result.stdout


def test_ambiguous_queue_output_never_invents_job_id(tmp_path: Path) -> None:
    """Keep raw multiline output when more than one standalone numeric candidate exists."""
    harness = _harness(tmp_path)
    harness.environment["QUEUE_SUBMISSION_OUTPUT"] = "TS socket: /etc/ts/socket_0\n12\n13\nambiguous response\n"
    result = _run_job(harness, "--queue-gpu", "0", "train", _CONFIG_MODEL_TRAINING_RELATIVE)

    assert result.returncode == 0, result.stderr
    assert "Queue submission output:" in result.stdout
    assert "12" in result.stdout
    assert "13" in result.stdout
    assert "Queue job ID: unavailable" in result.stdout
    assert "Queue job ID: 12" not in result.stdout


def test_malformed_explicit_gpu_values_fail_before_submission(tmp_path: Path) -> None:
    """Reject strings, negatives, and multiple indices without calling the queue helper."""
    for value in ("gpu0", "-1", "0,2", "2 0"):
        harness = _harness(tmp_path / value.replace("/", "_"))
        result = _run_job(harness, "--queue-gpu", value, "artifacts")

        assert result.returncode == 2
        assert "--queue-gpu must be auto" in result.stderr
        assert not harness.runtsgpu_capture.exists()
        assert not harness.docker_capture.exists()
        assert not _queue_log_dir(harness).exists()


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("train",),
        ("optuna",),
        ("unsupported",),
        ("train", "configs/tasks/steady_flow/experiments/best_of_class/missing.yaml"),
        ("--queue-gpu",),
        ("--queue-gpu=auto", "artifacts"),
        ("train", _CONFIG_MODEL_TRAINING_RELATIVE, "--queue-gpu", "auto", "--queue-gpu", "0"),
        ("train", _CONFIG_MODEL_TRAINING_RELATIVE, "--wait"),
        ("optuna", _OPTUNA_MODEL_TRAINING_RELATIVE, "--resume", "run"),
    ],
)
def test_invalid_job_or_wrapper_arguments_fail_before_submission(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    """
    Vary missing values, unknown jobs, duplicate wrapper options, waiting, and cross-workflow resume.

    Every wrapper-syntax family must exit with usage status before queue capture,
    Docker execution, or log allocation, isolating validation from side effects.
    """
    harness = _harness(tmp_path)
    result = _run_job(harness, *arguments)

    assert result.returncode == 2
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()
    assert not _queue_log_dir(harness).exists()


@pytest.mark.parametrize(
    "semantic_arguments",
    [
        ("--device", "cpu"),
        ("--device", "auto"),
        ("--device=cpu",),
        ("--device", "cuda", "--device", "cuda"),
        ("--device",),
    ],
)
def test_invalid_queued_device_requests_fail_before_submission(
    tmp_path: Path,
    semantic_arguments: tuple[str, ...],
) -> None:
    """
    Vary queued semantic options across CPU, auto, duplicate, and missing device values.

    Every family must fail before submission because GPU queue placement always
    normalizes to one strict inner ``--device cuda`` request.
    """
    harness = _harness(tmp_path)
    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "artifacts",
        *semantic_arguments,
    )

    assert result.returncode == 2
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


def test_invalid_explicit_gpu_fails_before_submission(tmp_path: Path) -> None:
    """
    Request a well-formed non-negative GPU index absent from the stub report.

    The wrapper must reject it with the reported-set context before queue or Docker
    invocation, preventing scheduler/device disagreement.
    """
    harness = _harness(tmp_path)
    result = _run_job(
        harness,
        "--queue-gpu",
        "7",
        "train",
        _CONFIG_MODEL_TRAINING_RELATIVE,
    )

    assert result.returncode == 2
    assert "not one of the reported indices" in result.stderr
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


@pytest.mark.parametrize(
    ("missing_command", "expected_message"),
    [
        ("nvidia-smi", "nvidia-smi is required"),
        ("runTSGPU.py", "runTSGPU.py is required"),
    ],
)
def test_missing_infrastructure_fails_clearly_before_submission(
    tmp_path: Path,
    missing_command: str,
    expected_message: str,
) -> None:
    """
    Remove either GPU discovery or queue submission from the isolated PATH.

    Each infrastructure family must return a clear pre-submit failure and never
    invoke the remaining queue boundary.
    """
    harness = _harness(tmp_path)
    (harness.binary_dir / missing_command).unlink()
    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _CONFIG_MODEL_TRAINING_RELATIVE,
    )

    assert result.returncode == 1
    assert expected_message in result.stderr
    assert not harness.runtsgpu_capture.exists()


@pytest.mark.parametrize(
    "gpu_report",
    [
        "",
        "garbage\n",
        "0, GPU, N/A, 1, 100\n",
        "0, GPU, 1, 101, 100\n",
        "0, GPU, 1, 1, 100\n0, GPU duplicate, 2, 2, 100\n",
    ],
)
def test_malformed_or_empty_gpu_output_never_submits(tmp_path: Path, gpu_report: str) -> None:
    """
    Vary GPU reports across empty, malformed, nonnumeric, impossible, and duplicate records.

    Every malformed family must fail before queue/Docker invocation, keeping automatic
    selection grounded in complete validated device facts.
    """
    harness = _harness(tmp_path, gpu_report=gpu_report)
    result = _run_job(harness, "--queue-gpu", "auto", "artifacts")

    assert result.returncode == 1
    assert not harness.runtsgpu_capture.exists()
    assert not harness.docker_capture.exists()


def test_repeated_submissions_allocate_distinct_logs(tmp_path: Path) -> None:
    """
    Submit the same validated job twice within one timestamp window.

    Both must succeed with distinct log basenames, preventing concurrent/repeated
    queue requests from overwriting one another's authoritative output.
    """
    harness = _harness(tmp_path)
    command = ("--queue-gpu", "auto", "train", _CONFIG_MODEL_TRAINING_RELATIVE)

    first = _run_job(harness, *command)
    second = _run_job(harness, *command)

    assert first.returncode == 0
    assert second.returncode == 0
    logs = sorted(_queue_log_dir(harness).glob("*.log"))
    assert len(logs) == 2
    assert logs[0].name != logs[1].name


def test_later_worker_failure_stays_in_log_after_detached_submission(tmp_path: Path) -> None:
    """
    Make the queued Docker worker emit both streams and exit with status 37.

    Successful submission remains detached; later worker status is preserved only
    in the host log and is never propagated back to the submit shell.
    """
    harness = _harness(tmp_path, docker_exit_code=37)
    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        _CONFIG_MODEL_TRAINING_RELATIVE,
    )

    assert result.returncode == 0
    assert "Queue job ID: 25" in result.stdout
    log_text = _log_path(harness).read_text(encoding="utf-8")
    assert "captured Docker stdout with spaces" in log_text
    assert "captured Docker stderr with spaces" in log_text
    assert "Docker exit status: 37" in log_text


@pytest.mark.parametrize(
    ("exported_key", "file_key", "expected_key"),
    [
        ("exported secret with spaces", "ignored file secret", "exported secret with spaces"),
        (None, "file secret with spaces\n", "file secret with spaces"),
        ("", "fallback file secret\n", "fallback file secret"),
        (None, "  \n\t", None),
        (None, None, None),
    ],
)
def test_wandb_credentials_are_resolved_and_forwarded_without_disclosure(
    tmp_path: Path,
    exported_key: str | None,
    file_key: str | None,
    expected_key: str | None,
) -> None:
    """
    Cross exported, file, blank, and absent W&B credential sources in queue and dev launchers.

    Both paths must apply exported-over-file precedence, trim file whitespace, forward
    only the variable name to Docker, and redact every possible secret value.
    """
    possible_keys = (exported_key, file_key.strip() if file_key else file_key)

    queued = _harness(tmp_path / "queued", exported_key=exported_key, file_key=file_key)
    queued_result = _run_job(queued, "--queue-gpu", "auto", "artifacts")
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


def test_storage_backed_config_maps_to_logical_training_domain(tmp_path: Path) -> None:
    """An acceptance config under physical training storage reaches the mounted logical path."""
    harness = _harness(tmp_path)
    config = Path(harness.environment["STORAGE_ROOT"]) / "data_training" / "processed" / "steady_flow" / "acceptance" / "bounded" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(_PRODUCTION_CONFIG.read_bytes())

    result = _run_job(harness, "--queue-gpu", "auto", "train", str(config))

    assert result.returncode == 0, result.stderr
    logical_path = "/workspace/repo/model_training/data/processed/steady_flow/acceptance/bounded/config.yaml"
    assert _capture_arguments(harness.runtsgpu_capture)[6] == logical_path
    preflight = _capture_arguments(harness.preflight_docker_capture)
    storage_root = Path(harness.environment["STORAGE_ROOT"])
    assert f"type=bind,source={storage_root / 'data_training'},target=/workspace/repo/model_training/data,readonly" in preflight
    assert preflight[-1] == logical_path


def test_train_translates_split_host_paths_including_new_output_destinations(tmp_path: Path) -> None:
    """Train host paths map to logical mounts even when an output path does not exist."""
    harness = _harness(tmp_path)
    resume = harness.repository / "model_training" / "data" / "processed" / "steady_flow" / "runs" / "run with spaces"
    resume.mkdir(parents=True)
    output = Path(harness.environment["STORAGE_ROOT"]) / "data_training" / "processed" / "new output"
    arguments = [
        _CONFIG_MODEL_TRAINING_RELATIVE,
        "--resume",
        str(resume),
        "--output-root",
        str(output),
    ]
    expected = [
        arguments[0],
        "--resume",
        "/workspace/repo/model_training/data/processed/steady_flow/runs/run with spaces",
        "--output-root",
        "/workspace/repo/model_training/data/processed/new output",
    ]

    result = _run_job(harness, "--queue-gpu", "auto", "train", *arguments)

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="train",
        module="src.experiments.cli.cli_train",
        command_arguments=expected,
    )


def test_optuna_translates_equals_host_output_path_without_requiring_existence(tmp_path: Path) -> None:
    """Equals-form Optuna output paths map without pre-creating the destination."""
    harness = _harness(tmp_path)
    output = Path(harness.environment["STORAGE_ROOT"]) / "data_training" / "processed" / "future study"
    arguments = [
        _OPTUNA_MODEL_TRAINING_RELATIVE,
        f"--output-root={output}",
    ]
    expected = [
        arguments[0],
        "--output-root=/workspace/repo/model_training/data/processed/future study",
    ]

    result = _run_job(harness, "--queue-gpu", "auto", "optuna", *arguments)

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="optuna",
        module="src.experiments.cli.cli_optuna",
        command_arguments=expected,
    )


def test_artifacts_translate_all_supported_host_path_options(tmp_path: Path) -> None:
    """Artifact input roots translate in split and equals forms without host leakage."""
    harness = _harness(tmp_path)
    training = Path(harness.environment["STORAGE_ROOT"]) / "data_training"
    arguments = [
        "--runs-root",
        str(training / "processed" / "steady_flow" / "runs"),
        f"--dataset-root={training / 'raw'}",
        "--metadata-root",
        str(training / "meta" / "not built yet"),
    ]
    expected = [
        "--runs-root",
        "/workspace/repo/model_training/data/processed/steady_flow/runs",
        "--dataset-root=/workspace/repo/model_training/data/raw",
        "--metadata-root",
        "/workspace/repo/model_training/data/meta/not built yet",
    ]

    result = _run_job(harness, "--queue-gpu", "auto", "artifacts", *arguments)

    _assert_common_chain(
        harness,
        result,
        gpu="2",
        job_type="artifacts",
        module="src.experiments.cli.cli_build_artifacts",
        command_arguments=expected,
    )


def test_development_launcher_uses_same_two_domain_mount_contract(tmp_path: Path) -> None:
    """Interactive development exposes the same logical roots and permissions as queued jobs."""
    harness = _harness(tmp_path)

    result = _run_dev(harness)

    assert result.returncode == 0, result.stderr
    docker = _capture_arguments(harness.docker_capture)
    storage_root = Path(harness.environment["STORAGE_ROOT"])
    assert docker[docker.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert docker[docker.index("--workdir") + 1] == "/workspace/repo"
    assert "PROJECT_ROOT=/workspace/repo" in docker
    assert "GENERATED_DATA_ROOT=/workspace/repo/data_generation/data" in docker
    assert "MODEL_TRAINING_DATA_ROOT=/workspace/repo/model_training/data" in docker
    assert f"{storage_root / 'data_generation'}:/workspace/repo/data_generation/data:ro" in docker
    assert f"{storage_root / 'data_training'}:/workspace/repo/model_training/data:rw" in docker
    assert f"{storage_root / '.docker_home' / 'passwd'}:/etc/passwd:ro" in docker
    assert f"{storage_root / '.docker_home' / 'group'}:/etc/group:ro" in docker
    assert f"{storage_root}:/workspace/storage:rw" not in docker
    assert not any(".state" in argument for argument in docker)
    assert (harness.repository / "data_generation" / "data").is_dir()
    assert (harness.repository / "model_training" / "data").is_dir()


def test_development_launcher_refuses_silent_container_reuse(tmp_path: Path) -> None:
    """An existing named container must be stopped rather than silently reused."""
    harness = _harness(tmp_path)
    harness.environment["DOCKER_RUNNING_NAMES"] = "grainlegumes-pino-airflow-dev\n"

    result = _run_dev(harness)

    assert result.returncode == 1
    assert "stale image or mount contract" in result.stderr
    assert "docker stop grainlegumes-pino-airflow-dev" in result.stderr
    assert not harness.docker_capture.exists()


def test_dockerfile_exports_only_two_application_data_roots() -> None:
    """Image defaults match direct, queued, and development logical paths."""
    dockerfile = (_REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV GENERATED_DATA_ROOT=/workspace/repo/data_generation/data" in dockerfile
    assert "ENV MODEL_TRAINING_DATA_ROOT=/workspace/repo/model_training/data" in dockerfile
    for forbidden in (
        "ENV STORAGE_ROOT=",
        "ENV DATA_ROOT=",
        "ENV DATASET_ROOT=",
        "ENV OUTPUT_ROOT=",
        "ENV TRAINING_STATE_ROOT=",
        "ENV DATASET_LOCK_ROOT=",
        "ENV RUN_LOCK_ROOT=",
    ):
        assert forbidden not in dockerfile
    assert ".state" not in dockerfile


def test_two_data_domains_are_excluded_from_git_and_docker_context() -> None:
    """Both public data-domain trees remain runtime-only regardless of contained artifacts."""
    gitignore = (_REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore = (_REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    expected = {"/data_generation/data/", "/model_training/data/"}
    assert expected.issubset(gitignore)
    assert expected.issubset(dockerignore)
    assert not any(line.startswith("!data_generation/data/") for line in gitignore)
    assert not any(line.startswith("!model_training/data/") for line in gitignore)


def test_custom_storage_root_maps_two_domains_with_distinct_permissions(tmp_path: Path) -> None:
    """Host storage locates two targeted mounts without becoming an application root."""
    harness = _harness(tmp_path)
    result = _run_job(harness, "--queue-gpu", "auto", "artifacts")

    assert result.returncode == 0, result.stderr
    docker = _capture_arguments(harness.docker_capture)
    storage_root = Path(harness.environment["STORAGE_ROOT"])
    assert f"{storage_root / 'data_generation'}:/workspace/repo/data_generation/data:ro" in docker
    assert f"{storage_root / 'data_training'}:/workspace/repo/model_training/data:rw" in docker
    assert f"{storage_root}:/workspace/storage:rw" not in docker
    assert "GENERATED_DATA_ROOT=/workspace/repo/data_generation/data" in docker
    assert "MODEL_TRAINING_DATA_ROOT=/workspace/repo/model_training/data" in docker
