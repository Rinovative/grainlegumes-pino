#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"

usage() {
  cat >&2 <<EOF
Usage:
  $0 train <config.yaml> [train arguments...]
  $0 optuna <config.yaml> [Optuna arguments...]
  $0 artifacts [artifact arguments...]
EOF
}

if (( $# == 0 )); then
  usage
  exit 2
fi

JOB_TYPE="$1"
shift
case "${JOB_TYPE}" in
  train|optuna)
    if (( $# == 0 )); then
      echo "${JOB_TYPE} requires a YAML config path." >&2
      usage
      exit 2
    fi
    ;;
  artifacts)
    ;;
  *)
    echo "Unsupported job type: ${JOB_TYPE}" >&2
    usage
    exit 2
    ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for GPU selection." >&2
  exit 1
fi
if ! command -v runTSGPU.py >/dev/null 2>&1; then
  echo "runTSGPU.py is required on PATH." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
echo "------------------------------------------------------------"

GPU_IDS="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | sed 's/[[:space:]]//g;/^$/d'
)"
if [ -z "${GPU_IDS}" ]; then
  echo "No GPUs were reported by nvidia-smi." >&2
  exit 1
fi

AUTO_GPU="$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2,2n \
    | head -n 1 \
    | cut -d, -f1 \
    | xargs
)"
if ! printf '%s\n' "${GPU_IDS}" | grep -Fqx -- "${AUTO_GPU}"; then
  echo "Could not determine a valid automatic GPU proposal." >&2
  exit 1
fi

GPU_LIST="$(printf '%s\n' "${GPU_IDS}" | paste -sd, -)"
read -r -p "Select GPU (${GPU_LIST}; Enter for proposed ${AUTO_GPU}): " GPU_ID
GPU_ID="${GPU_ID:-${AUTO_GPU}}"
if ! printf '%s\n' "${GPU_IDS}" | grep -Fqx -- "${GPU_ID}"; then
  echo "GPU ${GPU_ID@Q} is not one of the available indices: ${GPU_LIST}." >&2
  exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$(mktemp --suffix=.log "${LOG_DIR}/${TIMESTAMP}__${JOB_TYPE}__gpu${GPU_ID}__XXXXXX")"
LOG_BASENAME="$(basename "${LOG_PATH}")"

cd "${PROJECT_DIR}"
runTSGPU.py -g"${GPU_ID}" -- "${PROJECT_DIR}/scripts/_docker_run.sh" \
  "${GPU_ID}" \
  "${JOB_TYPE}" \
  "${LOG_BASENAME}" \
  "$@"

echo "Queued ${JOB_TYPE} job on GPU ${GPU_ID}."
echo "Queue: runTSGPU.py -g${GPU_ID} -s"
echo "Log:   ${LOG_PATH}"
echo "Tail:  tail -F ${LOG_PATH}"