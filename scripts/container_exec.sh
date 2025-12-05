#!/usr/bin/env bash
set -e

# ============================================================================
# container_exec.sh
# Execute a training script inside the Docker container.
# Only the filename is required, not the full path.
# ============================================================================

if [ -z "$1" ]; then
    echo "Usage: bash container_exec.sh <script.py>"
    exit 1
fi

SCRIPT_FILE="$1"
SCRIPT_PATH="/home/mambauser/workspace/model_training/training/${SCRIPT_FILE}"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "▶️  Running inside container: ${SCRIPT_PATH}"

cd /home/mambauser/workspace
/opt/micromamba/envs/grainlegumes-pino/bin/python3 "$SCRIPT_PATH"
