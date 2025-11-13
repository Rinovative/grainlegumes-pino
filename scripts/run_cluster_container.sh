#!/usr/bin/env bash
set -e

# ============================================================
# 🚀 Run GrainLegumes_PINO Docker Container
# ============================================================

CONTAINER_NAME="grainlegumes-pino"
IMAGE_NAME="grainlegumes-pino"

# ============================================================
# 🧩 Mode selection
# ============================================================
MODE="detached"
if [[ "$1" == "-i" || "$1" == "--interactive" ]]; then
  MODE="interactive"
fi

# ============================================================
# 🛑 Prevent duplicate container
# ============================================================
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  Container '$CONTAINER_NAME' is already running."
    echo "👉  Attach with: docker exec -it $CONTAINER_NAME bash"
    exit 0
fi

# ============================================================
# 🧠 Start container
# ============================================================
echo "🧠 Starting container '$CONTAINER_NAME' in $MODE mode..."

SSH_PATH="/home/rino.albertin/.ssh"

if [ "$MODE" == "interactive" ]; then
    docker run -it --rm \
      --name $CONTAINER_NAME \
      --gpus all \
      --shm-size=16G \
      -v $SSH_PATH:/home/mambauser/.ssh:ro \
      -v ~/workspace/grainlegumes-pino:/workspace:rw \
      -v ~/workspace/data:/workspace/data:rw \
      -v ~/workspace/data_generation:/workspace/data_generation/data:rw \
      -v ~/workspace/data_training:/workspace/model_training/data:rw \
      $IMAGE_NAME bash
else
    docker run -d --rm \
      --name $CONTAINER_NAME \
      --gpus all \
      --shm-size=16G \
      -v $SSH_PATH:/home/mambauser/.ssh:ro \
      -v ~/workspace/grainlegumes-pino:/workspace:rw \
      -v ~/workspace/data:/workspace/data:rw \
      -v ~/workspace/data_generation:/workspace/data_generation/data:rw \
      -v ~/workspace/data_training:/workspace/model_training/data:rw \
      $IMAGE_NAME tail -f /dev/null

    echo ""
    echo "✅ Container '$CONTAINER_NAME' is now running in detached mode."
    echo "👉 Attach via: docker exec -it $CONTAINER_NAME bash"
    echo ""
    echo "🧩 Or attach with VS Code:"
    echo "    Remote SSH → Attach to Running Container → $CONTAINER_NAME"
fi
