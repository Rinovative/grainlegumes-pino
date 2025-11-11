#!/usr/bin/env bash
# ============================================================
# scripts/run_auto.sh – Unified Runner (local + cluster)
# ============================================================

SCRIPT_PATH=$1
CONTAINER_NAME="grainlegumes-pino"

if hostname | grep -qiE "hpc|cluster|hpc115"; then
    echo "🧠 HPC mode – queueing job via gpucommand inside running container"
    CMD="docker exec $CONTAINER_NAME python3 ${SCRIPT_PATH}"
    gpucommand "${CMD}"
else
    echo "💻 Local mode – running directly"
    docker exec -it $CONTAINER_NAME python3 "${SCRIPT_PATH}"
fi
