#!/usr/bin/env bash
set -e

# ============================================================================
# hpc_train.sh
# Queue-based training launcher for HPC using runTSGPU.py + docker.
# Only the filename is required, not the full path.
# ============================================================================

if [ -z "$1" ]; then
    echo "Usage: bash hpc_train.sh <script.py>"
    exit 1
fi

SCRIPT_FILE="$1"
TRAIN_SCRIPT="model_training/training/${SCRIPT_FILE}"

echo "📄 Training script: ${TRAIN_SCRIPT}"
echo ""

# ------------------------------------------------------------
# Show GPU status
# ------------------------------------------------------------
echo "📊 Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
           --format=csv,noheader,nounits
echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Select GPU manually or automatically
# ------------------------------------------------------------
auto_gpu=$(nvidia-smi --query-gpu=index,memory.used \
                      --format=csv,noheader,nounits \
                      | sort -t, -k2 -n | head -n1 | cut -d',' -f1)

read -p "Select GPU (0–3, press Enter for ${auto_gpu}): " GPU_ID
GPU_ID=${GPU_ID:-$auto_gpu}

echo ""
echo "➡️  Starting training on GPU $GPU_ID (queued automatically)"
echo ""

# ------------------------------------------------------------
# Launch container via runTSGPU.py
# ------------------------------------------------------------
runTSGPU.py -g$GPU_ID -- docker run --rm \
    --gpus "\"device=$GPU_ID\"" \
    --shm-size=16G \
    -e WANDB_API_KEY=$(cat ~/wandb_key.txt) \
    -v ~/workspace/grainlegumes-pino:/home/mambauser/workspace \
    -v ~/workspace/data:/home/mambauser/workspace/data \
    -v ~/workspace/data_generation:/home/mambauser/workspace/data_generation/data \
    -v ~/workspace/data_training:/home/mambauser/workspace/model_training/data \
    grainlegumes-pino \
    bash /home/mambauser/workspace/scripts/container_exec.sh "$SCRIPT_FILE"
