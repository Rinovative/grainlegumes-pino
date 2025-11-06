#!/usr/bin/env bash
# ============================================================
# 📥 Import existing host data INTO Docker volumes
# ============================================================

set -e
CONTAINER=$(docker ps --filter "name=grainlegumes-pino" -q)

if [ -z "$CONTAINER" ]; then
  echo "❌ No running container found (grainlegumes-pino). Start Devcontainer first."
  exit 1
fi

echo "➡️ Copying host data into container volumes ..."

docker cp ./data/. "$CONTAINER":/home/mambauser/workspace/data/
docker cp ./data_generation/data/. "$CONTAINER":/home/mambauser/workspace/data_generation/data/
docker cp ./model_training/data/. "$CONTAINER":/home/mambauser/workspace/model_training/data/

echo "✅ Import finished — data now lives inside Docker volumes."
