# ============================================================
# 📥 Import existing host data INTO Docker volumes
# ============================================================

# Stoppe Skript bei Fehlern
$ErrorActionPreference = "Stop"

# Container-ID finden
$container = docker ps --filter "name=vsc-grainlegumes-pino" -q

if (-not $container) {
    Write-Host "❌ No running container found (vsc-grainlegumes-pino). Start Devcontainer first." -ForegroundColor Red
    exit 1
}

Write-Host "➡️ Copying host data into container volumes ..." -ForegroundColor Cyan

# Daten vom Host in die Container-Volumes kopieren
docker cp "./data/." "$container`:/home/mambauser/workspace/data/"
docker cp "./data_generation/data/." "$container`:/home/mambauser/workspace/data_generation/data/"
docker cp "./model_training/data/." "$container`:/home/mambauser/workspace/model_training/data/"

Write-Host "✅ Import finished - data now lives inside Docker volumes." -ForegroundColor Green
