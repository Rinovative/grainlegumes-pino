#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="grainlegumes-pino-airflow"
CONTAINER_NAME="grainlegumes-pino-airflow-dev"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
mkdir -p "${HOST_STORAGE_ROOT}"
STORAGE_DIR="$(cd "${HOST_STORAGE_ROOT}" && pwd -P)"
HOST_GENERATED_DATA_ROOT="${STORAGE_DIR}/data_generation"
HOST_MODEL_TRAINING_DATA_ROOT="${STORAGE_DIR}/data_training"
DOCKER_HOME="${STORAGE_DIR}/.docker_home"
mkdir -p \
  "${PROJECT_DIR}/data_generation/data" \
  "${PROJECT_DIR}/model_training/data" \
  "${HOST_GENERATED_DATA_ROOT}" \
  "${HOST_MODEL_TRAINING_DATA_ROOT}" \
  "${DOCKER_HOME}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH. Install Docker and retry." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "The Docker daemon is unavailable. Start Docker and retry." >&2
  exit 1
fi
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image '${IMAGE_NAME}' is missing. Build it with ./scripts/docker_build.sh." >&2
  exit 1
fi

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

# Create runtime user mapping for the intended non-root container process.
cat > "${DOCKER_HOME}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
EOF
cat > "${DOCKER_HOME}/group" <<EOF
root:x:0:
rino:x:$(id -g):
EOF
chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

# Resolve optional W&B authentication without printing the value.
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

if docker ps --format '{{.Names}}' | grep -Fqx -- "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' already exists and may use a stale image or mount contract." >&2
  echo "Stop it with: docker stop ${CONTAINER_NAME}; then rerun this script to recreate it." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fqx -- "${CONTAINER_NAME}"; then
  echo "Stopped container '${CONTAINER_NAME}' may use a stale image or mount contract." >&2
  echo "Remove it with: docker rm ${CONTAINER_NAME}; then rerun this script to recreate it." >&2
  exit 1
fi

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  DOCKER_RUNTIMES="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  if [[ "${DOCKER_RUNTIMES}" == *nvidia* ]]; then
    GPU_ARGS=(--gpus all)
    echo "NVIDIA runtime detected; exposing all host GPUs to the development container."
  else
    echo "NVIDIA GPUs are present but the Docker NVIDIA runtime is unavailable." >&2
    echo "Starting CPU-only; install NVIDIA Container Toolkit to enable direct GPU work." >&2
  fi
else
  echo "No usable host NVIDIA GPU was detected; starting the development container CPU-only."
fi

docker run -d --rm \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo \
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
  bash -lc "sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
echo "Repository mount:          /workspace/repo"
echo "Generated data mount (ro): /workspace/repo/data_generation/data"
echo "Training data mount (rw):  /workspace/repo/model_training/data"
echo "Attach with VS Code: Remote Explorer -> Containers -> ${CONTAINER_NAME}"
echo "Stop with: docker stop ${CONTAINER_NAME}"
