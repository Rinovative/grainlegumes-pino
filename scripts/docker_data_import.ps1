# ============================================================
# 📥 Import existing host data INTO Docker volumes
# ============================================================

$ErrorActionPreference = "Stop"

# Container-ID finden
$container = docker ps --filter "name=grainlegumes-pino" -q

if (-not $container) {
    Write-Host "No running container found (grainlegumes-pino). Start Devcontainer first." -ForegroundColor Red
    exit 1
}

Write-Host "⬇️ Copying host data into container volumes ..." -ForegroundColor Cyan

# Daten vom Host in die Container-Volumes kopieren
docker cp "./data/." "$container`:/home/mambauser/workspace/data/"
docker cp "./data_generation/data/." "$container`:/home/mambauser/workspace/data_generation/data/"
docker cp "./model_training/data/." "$container`:/home/mambauser/workspace/model_training/data/"

# 🔧 Correct Ownership after import
Write-Host "Fixing permissions inside container ..." -ForegroundColor Yellow
docker exec -u root $container sh -c 'chown -R 1000:1000 /home/mambauser/workspace/data /home/mambauser/workspace/data_generation/data /home/mambauser/workspace/model_training/data ; chmod -R a+rwX /home/mambauser/workspace/data /home/mambauser/workspace/data_generation/data /home/mambauser/workspace/model_training/data'

Write-Host ('✅ Import finished and permissions fixed.') -ForegroundColor Green
