#!/usr/bin/env bash
# ============================================================
# 📤 Export container data FROM Docker volumes back to host
# ============================================================

set -e
CONTAINER=$(docker ps --filter "name=grainlegumes-pino" -q)

if [ -z "$CONTAINER" ]; then
  echo "❌ No running container found (grainlegumes-pino). Start Devcontainer first."
  exit 1
fi

echo "⬇️ Copying data from container to host folders ..."

docker cp "$CONTAINER":/home/mambauser/workspace/data/. ./data/
docker cp "$CONTAINER":/home/mambauser/workspace/data_generation/data/. ./data_generation/data/
docker cp "$CONTAINER":/home/mambauser/workspace/model_training/data/. ./model_training/data/

echo "✅ Export complete — host directories now contain copies of the volume data."
