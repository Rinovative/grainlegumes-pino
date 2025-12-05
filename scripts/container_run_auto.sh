#!/usr/bin/env bash
# ============================================================================
# scripts/container_run_auto.sh
# Auto GPU selection + interactive/local execution inside a Docker container.
# This script is intended to run *inside* a container or local machine,
# not on HPC queue systems.
# ============================================================================

SCRIPT_PATH=$1
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/$(basename "${SCRIPT_PATH%.*}")_$(date +%Y%m%d_%H%M%S).out"

# ============================================================================
# 1. Environment detection
# ============================================================================
if [ -f "/.dockerenv" ] && command -v nvidia-smi &>/dev/null; then
    ENV="container_gpu"
else
    ENV="local"
fi

# ============================================================================
# 2. Show GPU status
# ============================================================================
show_gpu_status() {
    echo "📊 Current GPU usage:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader,nounits
    else
        echo "⚠️  nvidia-smi not available."
    fi
    echo "------------------------------------------------------------"
}

# ============================================================================
# 3. Auto-select GPU with lowest memory load
# ============================================================================
auto_select_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo 0
        return
    fi
    nvidia-smi --query-gpu=index,memory.used \
               --format=csv,noheader,nounits \
        | sort -t, -k2 -n \
        | head -n1 \
        | cut -d',' -f1 \
        | xargs
}

# ============================================================================
# 4. Check if GPU is already running processes (local/container mode only)
# ============================================================================
check_gpu_free() {
    local gpu_id=$1
    local active
    active=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader | grep -v "^$" || true)

    if [ -n "$active" ]; then
        echo "⚠️  GPU $gpu_id has active processes."
        read -r -p "Still continue? [y/N]: " cont
        if [[ "$cont" != "y" && "$cont" != "Y" ]]; then
            echo "❌ Aborted."
            exit 1
        fi
    fi
}

# ============================================================================
# 5. Create log directory
# ============================================================================
mkdir -p "$LOG_DIR"

# ============================================================================
# 6. Execution logic
# ============================================================================
if [ "$ENV" = "container_gpu" ]; then
    echo "🐋 Container environment detected - GPU availability check enabled."
    show_gpu_status

    DEFAULT_GPU=$(auto_select_gpu)
    read -p "Select GPU (0–3, press Enter for ${DEFAULT_GPU}): " GPU_ID
    GPU_ID=${GPU_ID:-$DEFAULT_GPU}

    check_gpu_free "$GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    echo ""
    read -p "Start in background? (Enter = Yes, n = run in terminal): " RUN_MODE
    echo ""

    if [[ "$RUN_MODE" == "n" || "$RUN_MODE" == "N" ]]; then
        echo "▶️  Running interactively on GPU ${GPU_ID}: ${SCRIPT_PATH}"
        echo "------------------------------------------------------------"
        python3 "${SCRIPT_PATH}"
    else
        echo "🚀 Running in background on GPU ${GPU_ID}: ${SCRIPT_PATH}"
        echo "📝 Logs: ${LOG_FILE}"
        nohup python3 "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
        echo "✅ Background job started (PID $!)"
        echo "👉 View logs with: tail -f ${LOG_FILE}"
    fi

else
    echo "💻 Local environment detected – running directly."
    python3 "${SCRIPT_PATH}"
fi
