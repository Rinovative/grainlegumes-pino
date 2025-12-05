#!/usr/bin/env bash
set -e

# ============================================================================
# hpc_start_container.sh
# Launch the GrainLegumes_PINO Docker container on an HPC node.
# Supports interactive mode (-i) or detached/background mode (default).
#
# Important:
# - This script is executed on the HPC host (not inside the container).
# - It starts a long-running development container with proper GPU access.
# ============================================================================

# Load WANDB key from private file
WANDB_API_KEY=$(cat ~/wandb_key.txt)

CONTAINER_NAME="grainlegumes-pino"
IMAGE_NAME="grainlegumes-pino"

# ============================================================================
# Mode selection
# ============================================================================
# Default: detached mode
# Usage:
#   ./scripts/hpc_start_container.sh        → detached
#   ./scripts/hpc_start_container.sh -i     → interactive
# ============================================================================
MODE="detached"
if [[ "$1" == "-i" || "$1" == "--interactive" ]]; then
  MODE="interactive"
fi

# ============================================================================
# Prevent duplicate container
# ============================================================================
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  Container '$CONTAINER_NAME' is already running."
    echo "👉  Attach using: docker exec -it $CONTAINER_NAME bash"
    exit 0
fi

# ============================================================================
# Start container
# ============================================================================
echo "🚀 Starting container '$CONTAINER_NAME' in $MODE mode..."

if [ "$MODE" == "interactive" ]; then

    docker run -it --rm \
      --name $CONTAINER_NAME \
      --gpus all \
      --shm-size=16G \
      -e WANDB_API_KEY="$WANDB_API_KEY" \
      -v ~/.ssh:/home/mambauser/.ssh:rw \
      -v ~/workspace/grainlegumes-pino:/home/mambauser/workspace:rw \
      -v ~/workspace/data:/home/mambauser/workspace/data:rw \
      -v ~/workspace/data_generation:/home/mambauser/workspace/data_generation/data:rw \
      -v ~/workspace/data_training:/home/mambauser/workspace/model_training/data:rw \
      $IMAGE_NAME bash

else

    docker run -d --rm \
      --name $CONTAINER_NAME \
      --gpus all \
      --shm-size=16G \
      -e WANDB_API_KEY="$WANDB_API_KEY" \
      -v ~/.ssh:/home/mambauser/.ssh:rw \
      -v ~/workspace/grainlegumes-pino:/home/mambauser/workspace:rw \
      -v ~/workspace/data:/home/mambauser/workspace/data:rw \
      -v ~/workspace/data_generation:/home/mambauser/workspace/data_generation/data:rw \
      -v ~/workspace/data_training:/home/mambauser/workspace/model_training/data:rw \
      $IMAGE_NAME bash -lc "sleep infinity"

    echo ""
    echo "✅ Container '$CONTAINER_NAME' is now running in detached mode."
    echo "👉 Attach using: docker exec -it $CONTAINER_NAME bash"
    echo ""
    echo "🧩 Or attach via VS Code:"
    echo "    Remote SSH → Attach to Running Container → $CONTAINER_NAME"
fi
