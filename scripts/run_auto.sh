#!/bin/bash
# ============================================================
# scripts/run_auto.sh – Unified Docker Runner (local + cluster)
# ============================================================

SCRIPT_PATH=$1
CONTAINER_NAME="vsc-grainlegumes-pino"

if hostname | grep -qiE "hpc|cluster|hpc115"; then
    echo "🧠 HPC mode - job queued via gpucommand"
    SHM_SIZE=32G
    CMD="docker run --rm --gpus all --shm-size=${SHM_SIZE} \
        -v $(pwd):/home/mambauser/workspace \
        -w /home/mambauser/workspace \
        ${CONTAINER_NAME} python3 ${SCRIPT_PATH}"
    gpucommand ${CMD}
else
    echo "💻 Local mode - running directly"
    SHM_SIZE=16G
    docker run --rm --gpus all --shm-size=${SHM_SIZE} \
        -v $(pwd):/home/mambauser/workspace \
        -w /home/mambauser/workspace \
        ${CONTAINER_NAME} python3 "${SCRIPT_PATH}"
fi
