#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 <gpu-id> <train|optuna|artifacts> <log-scope> <log-basename> [semantic CLI arguments...]" >&2
  exit 2
fi

GPU_ID="$1"
JOB_TYPE="$2"
LOG_SCOPE="$3"
LOG_BASENAME="$4"
shift 4
SEMANTIC_ARGS=("$@")

if [[ ! "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "GPU ID must be a non-negative integer, got: ${GPU_ID@Q}" >&2
  exit 2
fi
TRIMMED_LOG_SCOPE="${LOG_SCOPE#"${LOG_SCOPE%%[![:space:]]*}"}"
TRIMMED_LOG_SCOPE="${TRIMMED_LOG_SCOPE%"${TRIMMED_LOG_SCOPE##*[![:space:]]}"}"
if [[ -z "${LOG_SCOPE}" || "${TRIMMED_LOG_SCOPE}" != "${LOG_SCOPE}" || "${LOG_SCOPE}" == */* || "${LOG_SCOPE}" == *\\* || "${LOG_SCOPE}" == "." || "${LOG_SCOPE}" == ".." ]]; then
  echo "Log scope must be one safe non-empty path component, got: ${LOG_SCOPE@Q}" >&2
  exit 2
fi
if [[ -z "${LOG_BASENAME}" || "${LOG_BASENAME}" == */* || "${LOG_BASENAME}" == "." || "${LOG_BASENAME}" == ".." ]]; then
  echo "Log name must be one non-empty basename, got: ${LOG_BASENAME@Q}" >&2
  exit 2
fi

case "${JOB_TYPE}" in
  train)
    MODULE="src.experiments.cli.cli_train"
    ;;
  optuna)
    MODULE="src.experiments.cli.cli_optuna"
    ;;
  artifacts)
    MODULE="src.experiments.cli.cli_build_artifacts"
    ;;
  *)
    echo "Unsupported job type: ${JOB_TYPE}" >&2
    exit 2
    ;;
esac

# Normalize the worker boundary to exactly one strict semantic CUDA request.
CLI_ARGS=()
DEVICE_COUNT=0
INDEX=0
while (( INDEX < ${#SEMANTIC_ARGS[@]} )); do
  ARGUMENT="${SEMANTIC_ARGS[INDEX]}"
  case "${ARGUMENT}" in
    --queue-gpu|--queue-gpu=*)
      echo "--queue-gpu is wrapper-only and must not reach the queue worker." >&2
      exit 2
      ;;
    --follow|--follow=*|--wait|--wait=*|--follow-and-wait|--follow-and-wait=*)
      echo "Log-following and completion-wait options must not reach the queue worker." >&2
      exit 2
      ;;
    --device)
      if (( INDEX + 1 >= ${#SEMANTIC_ARGS[@]} )); then
        echo "--device requires one of auto, cuda, or cpu." >&2
        exit 2
      fi
      DEVICE_COUNT=$((DEVICE_COUNT + 1))
      DEVICE_VALUE="${SEMANTIC_ARGS[INDEX + 1]}"
      INDEX=$((INDEX + 2))
      ;;
    --device=*)
      DEVICE_COUNT=$((DEVICE_COUNT + 1))
      DEVICE_VALUE="${ARGUMENT#--device=}"
      INDEX=$((INDEX + 1))
      ;;
    *)
      CLI_ARGS+=("${ARGUMENT}")
      INDEX=$((INDEX + 1))
      continue
      ;;
  esac
  if (( DEVICE_COUNT > 1 )); then
    echo "Duplicate or conflicting --device options are not allowed for queued jobs." >&2
    exit 2
  fi
  if [[ "${DEVICE_VALUE}" != "cuda" ]]; then
    echo "Queued jobs require explicit --device cuda. Received --device ${DEVICE_VALUE@Q}." >&2
    exit 2
  fi
done
CLI_ARGS+=("--device" "cuda")

IMAGE_NAME="grainlegumes-pino-airflow"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
mkdir -p "${HOST_STORAGE_ROOT}"
STORAGE_DIR="$(cd "${HOST_STORAGE_ROOT}" && pwd -P)"
HOST_GENERATED_DATA_ROOT="${STORAGE_DIR}/data_generation"
HOST_MODEL_TRAINING_DATA_ROOT="${STORAGE_DIR}/data_training"
DOCKER_HOME="${STORAGE_DIR}/.docker_home"
QUEUE_LOG_DIR="${HOST_MODEL_TRAINING_DATA_ROOT}/processed/${LOG_SCOPE}/logs/queue"
LOG_FILE="${QUEUE_LOG_DIR}/${LOG_BASENAME}"
mkdir -p \
  "${PROJECT_DIR}/data_generation/data" \
  "${PROJECT_DIR}/model_training/data" \
  "${HOST_GENERATED_DATA_ROOT}" \
  "${HOST_MODEL_TRAINING_DATA_ROOT}" \
  "${DOCKER_HOME}" \
  "${QUEUE_LOG_DIR}"

# Capture both streams and the final Docker status in the unique host-visible log.
exec > "${LOG_FILE}" 2>&1
printf 'Queue worker job: %s\n' "${JOB_TYPE}"
printf 'Selected host GPU: %s\n' "${GPU_ID}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
printf 'Container CUDA device: 0\n'
printf 'Task-spooler socket: %s\n' "${TS_SOCKET:-/etc/ts/socket_${GPU_ID}}"
printf 'Host log: %s\n' "${LOG_FILE}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required by the queue worker but was not found on PATH."
  echo "Docker exit status: 127"
  exit 127
fi

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

# Create runtime user mapping for the non-root container process.
cat > "${DOCKER_HOME}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
EOF
cat > "${DOCKER_HOME}/group" <<EOF
root:x:0:
rino:x:$(id -g):
EOF
chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

# Resolve optional W&B authentication without printing or embedding the value.
WANDB_ENV_ARGS=()
if [[ -z "${WANDB_API_KEY:-}" && -r "${HOME}/wandb_key.txt" ]]; then
  FILE_WANDB_KEY="$(trim_whitespace "$(< "${HOME}/wandb_key.txt")")"
  if [[ -n "${FILE_WANDB_KEY}" ]]; then
    WANDB_API_KEY="${FILE_WANDB_KEY}"
    export WANDB_API_KEY
  else
    unset WANDB_API_KEY
  fi
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ENV_ARGS=(-e WANDB_API_KEY)
fi

SSH_ARGS=()
if [[ -d "${HOME}/.ssh" ]]; then
  SSH_ARGS=(-v "${HOME}/.ssh:/workspace/storage/.docker_home/.ssh:ro")
fi

set +e
docker run --rm \
  --gpus "device=${GPU_ID}" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo/model_training \
  -e HOME=/workspace/storage/.docker_home \
  -e PROJECT_ROOT=/workspace/repo \
  -e GENERATED_DATA_ROOT=/workspace/repo/data_generation/data \
  -e MODEL_TRAINING_DATA_ROOT=/workspace/repo/model_training/data \
  "${WANDB_ENV_ARGS[@]}" \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${HOST_GENERATED_DATA_ROOT}:/workspace/repo/data_generation/data:ro" \
  -v "${HOST_MODEL_TRAINING_DATA_ROOT}:/workspace/repo/model_training/data:rw" \
  -v "${DOCKER_HOME}:/workspace/storage/.docker_home:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  python -m "${MODULE}" "${CLI_ARGS[@]}"
DOCKER_STATUS=$?
set -e

printf 'Docker exit status: %s\n' "${DOCKER_STATUS}"
exit "${DOCKER_STATUS}"
