#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="grainlegumes-pino-airflow"
CONTAINER_NAME="grainlegumes-pino-airflow-dev"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="$(cd "${PROJECT_DIR}/../storage" && pwd)"
DOCKER_HOME="${STORAGE_DIR}/.docker_home"

mkdir -p \
  "${PROJECT_DIR}/logs" \
  "${DOCKER_HOME}"

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
# Prevent duplicate dev container
# ----------------------------------------------------------------------
if docker ps --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' is already running."
  echo "Attach with VS Code or stop it with:"
  echo "  docker stop ${CONTAINER_NAME}"
  exit 0
fi

if docker ps -a --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing stopped container '${CONTAINER_NAME}'."
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# ----------------------------------------------------------------------
# Start dev container
# ----------------------------------------------------------------------
docker run -d --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo \
  -e HOME=/workspace/storage/.docker_home \
  "${WANDB_ENV_ARGS[@]}" \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash -lc "sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
echo "Attach with VS Code: Remote Explorer -> Containers -> ${CONTAINER_NAME}"
echo "Stop with: docker stop ${CONTAINER_NAME}"