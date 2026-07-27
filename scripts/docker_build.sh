#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="grainlegumes-pino-airflow"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH. Install Docker and retry." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "The Docker daemon is unavailable. Start Docker and retry." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .
