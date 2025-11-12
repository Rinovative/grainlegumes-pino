#!/usr/bin/env bash
# ============================================================
# scripts/run_auto.sh – Cluster-Docker + GPU-Freiheitscheck + interaktiver Startmodus
# ============================================================

SCRIPT_PATH=$1
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/$(basename "${SCRIPT_PATH%.*}")_$(date +%Y%m%d_%H%M%S).out"

# ============================================================
# 🧠 1. Umgebungserkennung
# ============================================================
if [ -f "/.dockerenv" ] && command -v nvidia-smi &>/dev/null; then
    ENV="cluster_docker"
else
    ENV="local"
fi

# ============================================================
# 📊 2. GPU-Statusanzeige
# ============================================================
show_gpu_status() {
    echo "📊 Aktuelle GPU-Auslastung:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader,nounits
    else
        echo "⚠️  Kein nvidia-smi verfügbar."
    fi
    echo "------------------------------------------------------------"
}

# ============================================================
# 🤖 3. Automatische GPU-Auswahl (Default)
# ============================================================
auto_select_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo 0
        return
    fi
    nvidia-smi --query-gpu=index,memory.used \
               --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -n1 | cut -d',' -f1 | xargs
}

# ============================================================
# 🚀 4. GPU-Verfügbarkeit prüfen
# ============================================================
check_gpu_free() {
    local gpu_id=$1
    local active
    active=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader | grep -v "^$" || true)
    if [ -n "$active" ]; then
        echo "⚠️  GPU $gpu_id hat laufende Prozesse. Trotzdem fortfahren? [y/N]"
        read -r cont
        if [[ "$cont" != "y" && "$cont" != "Y" ]]; then
            echo "❌ Abbruch."
            exit 1
        fi
    fi
}

# ============================================================
# 📦 5. Logging-Verzeichnis
# ============================================================
mkdir -p "$LOG_DIR"

# ============================================================
# 🏃 6. Ausführungslogik
# ============================================================
if [ "$ENV" = "cluster_docker" ]; then
    echo "🧠🐋 Cluster-Container erkannt – GPU-Freiheitsprüfung aktiv"
    show_gpu_status

    DEFAULT_GPU=$(auto_select_gpu)
    read -p "Welche GPU soll verwendet werden? (0–3, Enter für ${DEFAULT_GPU}): " GPU_ID
    GPU_ID=${GPU_ID:-$DEFAULT_GPU}

    check_gpu_free "$GPU_ID"
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo ""
    read -p "Im Hintergrund starten? (Enter = Ja, n = direkt im Terminal): " RUN_MODE
    echo ""

    if [[ "$RUN_MODE" == "n" || "$RUN_MODE" == "N" ]]; then
        echo "🧩 Starte interaktiv auf GPU ${GPU_ID}: ${SCRIPT_PATH}"
        echo "------------------------------------------------------------"
        python3 "${SCRIPT_PATH}"
    else
        echo "🚀 Starte detached auf GPU ${GPU_ID}: ${SCRIPT_PATH}"
        echo "📝 Logs: ${LOG_FILE}"
        nohup python3 "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
        echo "✅ Training läuft im Hintergrund (PID $!)"
        echo "👉 Log live ansehen mit: tail -f ${LOG_FILE}"
    fi
else
    echo "💻 Lokaler Modus – läuft direkt"
    python3 "${SCRIPT_PATH}"
fi
