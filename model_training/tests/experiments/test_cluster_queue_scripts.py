# ruff: noqa: PLR2004, S101, S603
"""
Verify Docker and cluster launchers through isolated command stubs, never submission.

The harness covers canonical mounts/environment roots, GPU discovery/selection,
quoting, queue arguments, logging, wrapper validation, and early rejection of
invalid CPU/fallback requests. It deliberately does not run Docker, Slurm,
``runTSGPU.py``, training, or a real GPU workload.
"""

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
    runtsgpu_capture, docker_capture, docker_environment_capture : pathlib.Path
        NUL-delimited argument and environment capture files written by stubs.

    """

    repository: Path
    environment: dict[str, str]
    binary_dir: Path
    home: Path
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
    for name in ("docker_job.sh", "_docker_run.sh", "docker_dev.sh"):
        shutil.copy2(_REPOSITORY_ROOT / "scripts" / name, scripts / name)

    config_root = repository / "model_training" / "configs"
    for relative in (
        "experiments/steady_flow_fno.yaml",
        "experiments/config with spaces.yaml",
        "optuna/steady_flow_fno_search.yaml",
    ):
        config_path = config_root / relative
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("task: steady_flow\n", encoding="utf-8")

    binary_dir = tmp_path / "stub commands"
    binary_dir.mkdir()
    report = gpu_report if gpu_report is not None else ("0, Cluster GPU A, 20, 7000, 24000\n2, Cluster GPU B, 5, 1000, 24000\n")
    _write_executable(
        binary_dir / "nvidia-smi",
        f"""#!/usr/bin/env bash
set -euo pipefail
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
case "${1-}" in
  info)
    if [[ "$*" == *"--format"* ]]; then
      printf '{"nvidia": {}}\\n'
    fi
    exit 0
    ;;
  image)
    exit 0
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
            "STORAGE_ROOT": str(storage_root),
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
        binary_dir=binary_dir,
        home=home,
        runtsgpu_capture=Path(environment["RUNTSGPU_CAPTURE"]),
        docker_capture=Path(environment["DOCKER_CAPTURE"]),
        docker_environment_capture=Path(environment["DOCKER_ENV_CAPTURE"]),
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
    assert "Current GPU usage:" in result.stdout
    assert f"Selected GPU: {gpu}" in result.stdout
    assert f"Log:   {log_path}" in result.stdout
    assert "Tail:  tail -F " in result.stdout
    return log_basename


def test_prompted_default_gpu_queues_training_with_exact_arguments(tmp_path: Path) -> None:
    """
    Submit training interactively by accepting the least-memory GPU proposal.

    Config/resume values containing spaces must survive queue and Docker argv exactly,
    proving default selection does not alter semantic arguments.
    """
    harness = _harness(tmp_path)
    arguments = [
        "configs/experiments/config with spaces.yaml",
        "--resume",
        "/workspace/repo/model_training/data/processed/steady_flow/runs/run with spaces",
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
    assert "Proposed GPU: 2" in result.stdout


def test_prompted_explicit_gpu_queues_optuna_with_runtime_overrides(tmp_path: Path) -> None:
    """
    Select a reported GPU explicitly for an Optuna job with several runtime overrides.

    The chosen index and every spaced/flag argument must reach the correct inner
    module unchanged, protecting interactive selection from consuming CLI options.
    """
    harness = _harness(tmp_path)
    arguments = [
        "configs/optuna/steady_flow_fno_search.yaml",
        "--n-trials",
        "3",
        "--output-root",
        "/workspace/repo/model_training/data/processed/output root",
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
        "configs/experiments/steady_flow_fno.yaml",
    )

    assert result.returncode == 0, result.stderr
    assert "Selected GPU: 3" in result.stdout
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
        "model_training/configs/experiments/steady_flow_fno.yaml",
    )

    assert result.returncode == 0, result.stderr
    runtsgpu = _capture_arguments(harness.runtsgpu_capture)
    assert runtsgpu[6] == "configs/experiments/steady_flow_fno.yaml"


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("train",),
        ("optuna",),
        ("unsupported",),
        ("train", "configs/experiments/missing.yaml"),
        ("--queue-gpu",),
        ("--queue-gpu=auto", "artifacts"),
        ("train", "configs/experiments/steady_flow_fno.yaml", "--queue-gpu", "auto"),
    ],
)
def test_invalid_job_or_wrapper_arguments_fail_before_submission(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    """
    Vary missing job/config values, unknown jobs, and misplaced queue options.

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
        ("--cpu",),
    ],
)
def test_invalid_queued_device_requests_fail_before_submission(
    tmp_path: Path,
    semantic_arguments: tuple[str, ...],
) -> None:
    """
    Vary queued semantic options across CPU, auto, duplicates, missing values, and unsupported flags.

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
        "configs/experiments/steady_flow_fno.yaml",
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
        "configs/experiments/steady_flow_fno.yaml",
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
    command = ("--queue-gpu", "auto", "train", "configs/experiments/steady_flow_fno.yaml")

    first = _run_job(harness, *command)
    second = _run_job(harness, *command)

    assert first.returncode == 0
    assert second.returncode == 0
    logs = sorted(_queue_log_dir(harness).glob("*.log"))
    assert len(logs) == 2
    assert logs[0].name != logs[1].name


def test_inner_docker_failure_reaches_queue_process_and_log(tmp_path: Path) -> None:
    """
    Make the Docker stub emit both streams and exit with status 37.

    The queue process must propagate that exact status and preserve stdout/stderr
    in its log without printing a misleading successful-submission message.
    """
    harness = _harness(tmp_path, docker_exit_code=37)
    result = _run_job(
        harness,
        "--queue-gpu",
        "auto",
        "train",
        "configs/experiments/steady_flow_fno.yaml",
    )

    assert result.returncode == 37
    log_text = _log_path(harness).read_text(encoding="utf-8")
    assert "captured Docker stdout with spaces" in log_text
    assert "captured Docker stderr with spaces" in log_text
    assert "Docker exit status: 37" in log_text
    assert "Queued train job" not in result.stdout


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
    config.write_text("task: steady_flow\n", encoding="utf-8")

    result = _run_job(harness, "--queue-gpu", "auto", "train", str(config))

    assert result.returncode == 0, result.stderr
    assert _capture_arguments(harness.runtsgpu_capture)[6] == (
        "/workspace/repo/model_training/data/processed/steady_flow/acceptance/bounded/config.yaml"
    )


def test_train_translates_split_host_paths_including_new_output_destinations(tmp_path: Path) -> None:
    """Train host paths map to logical mounts even when an output path does not exist."""
    harness = _harness(tmp_path)
    resume = harness.repository / "model_training" / "data" / "processed" / "steady_flow" / "runs" / "run with spaces"
    resume.mkdir(parents=True)
    output = Path(harness.environment["STORAGE_ROOT"]) / "data_training" / "processed" / "new output"
    arguments = [
        "configs/experiments/steady_flow_fno.yaml",
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
        "configs/optuna/steady_flow_fno_search.yaml",
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
    for forbidden in ("ENV STORAGE_ROOT=", "ENV DATA_ROOT=", "ENV DATASET_ROOT=", "ENV OUTPUT_ROOT="):
        assert forbidden not in dockerfile


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
