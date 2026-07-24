#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 <gpu-id> <train|optuna|artifacts> <log-basename> [command arguments...]" >&2
  exit 2
fi

GPU_ID="$1"
JOB_TYPE="$2"
LOG_BASENAME="$3"
shift 3

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU ID must be a non-negative integer, got: ${GPU_ID@Q}" >&2
  exit 2
fi
if [ -z "${LOG_BASENAME}" ] || [[ "${LOG_BASENAME}" == */* ]] || [[ "${LOG_BASENAME}" == "." || "${LOG_BASENAME}" == ".." ]]; then
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

IMAGE_NAME="grainlegumes-pino-airflow"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
mkdir -p "${PROJECT_DIR}/logs" "${HOST_STORAGE_ROOT}"
STORAGE_DIR="$(cd "${HOST_STORAGE_ROOT}" && pwd)"
DOCKER_HOME="${STORAGE_DIR}/.docker_home"
LOG_FILE="${PROJECT_DIR}/logs/${LOG_BASENAME}"
mkdir -p "${DOCKER_HOME}"

# ----------------------------------------------------------------------
# Create runtime user mapping for container
# ----------------------------------------------------------------------
cat > "${DOCKER_HOME}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
EOF

cat > "${DOCKER_HOME}/group" <<EOF
root:x:0:
rino:x:$(id -g):
EOF

chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

# ----------------------------------------------------------------------
# Resolve optional W&B authentication for standard Docker env pass-through
# ----------------------------------------------------------------------
WANDB_ENV_ARGS=()
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "${HOME}/wandb_key.txt" ]; then
  WANDB_API_KEY="$(< "${HOME}/wandb_key.txt")"
  if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY
  else
    unset WANDB_API_KEY
  fi
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ENV_ARGS=(-e WANDB_API_KEY)
fi

# ----------------------------------------------------------------------
# Optional SSH mount for Git operations
# ----------------------------------------------------------------------
SSH_ARGS=()
if [ -d "${HOME}/.ssh" ]; then
  SSH_ARGS=(-v "${HOME}/.ssh:/workspace/storage/.docker_home/.ssh:ro")
fi

# ----------------------------------------------------------------------
# Run the selected semantic CLI inside Docker and preserve its exit code
# ----------------------------------------------------------------------
docker run --rm \
  --gpus "device=${GPU_ID}" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo/model_training \
  -e HOME=/workspace/storage/.docker_home \
  -e STORAGE_ROOT=/workspace/storage \
  "${WANDB_ENV_ARGS[@]}" \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  python -m "${MODULE}" "$@" > "${LOG_FILE}" 2>&1