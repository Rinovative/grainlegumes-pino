# ============================================================
# 📤 Export container data FROM Docker volumes back to host
# ============================================================

# Stoppe Skript bei Fehlern
$ErrorActionPreference = "Stop"

# Container-ID finden
$container = docker ps --filter "name=grainlegumes-pino" -q

if (-not $container) {
    Write-Host "❌ No running container found (grainlegumes-pino). Start Devcontainer first." -ForegroundColor Red
    exit 1
}

Write-Host "⬇️ Copying data from container to host folders ..." -ForegroundColor Cyan

# Daten aus den Volumes kopieren
docker cp "$container`:/home/mambauser/workspace/data/." "./data/"
docker cp "$container`:/home/mambauser/workspace/data_generation/data/." "./data_generation/data/"
docker cp "$container`:/home/mambauser/workspace/model_training/data/." "./model_training/data/"

Write-Host "✅ Export complete - host directories now contain copies of the volume data." -ForegroundColor Green
