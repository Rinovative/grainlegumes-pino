#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
MODEL_DIR="${PROJECT_DIR}/model_training"
LOG_DIR="${PROJECT_DIR}/logs"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--queue-gpu auto|INDEX] train <experiment.yaml> [semantic CLI options...]
  $0 [--queue-gpu auto|INDEX] optuna <study.yaml> [semantic CLI options...]
  $0 [--queue-gpu auto|INDEX] artifacts [semantic CLI options...]

GPU selection:
  omit --queue-gpu  display utilization, propose the least-memory GPU, and prompt
  --queue-gpu auto  select the least-memory GPU without prompting
  --queue-gpu INDEX select one reported GPU index without prompting
EOF
}

fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

resolve_config_argument() {
  local requested="$1"
  local candidate=""
  local candidate_dir
  local resolved
  local storage_dir=""

  if [[ "${requested}" == /* ]]; then
    candidate="${requested}"
  elif [[ -e "${PWD}/${requested}" ]]; then
    candidate="${PWD}/${requested}"
  elif [[ -e "${PROJECT_DIR}/${requested}" ]]; then
    candidate="${PROJECT_DIR}/${requested}"
  elif [[ -e "${MODEL_DIR}/${requested}" ]]; then
    candidate="${MODEL_DIR}/${requested}"
  else
    fail 2 "Config path does not exist: ${requested}"
  fi

  if [[ ! -f "${candidate}" ]]; then
    fail 2 "Config path is not a regular file: ${requested}"
  fi
  if [[ ! -r "${candidate}" ]]; then
    fail 2 "Config path is not readable: ${requested}"
  fi

  candidate_dir="$(cd "$(dirname "${candidate}")" && pwd -P)"
  resolved="${candidate_dir}/$(basename "${candidate}")"
  if [[ "${resolved}" == "${MODEL_DIR}/"* ]]; then
    printf '%s' "${resolved#"${MODEL_DIR}/"}"
    return
  fi
  if [[ "${resolved}" == "${PROJECT_DIR}/"* ]]; then
    printf '/workspace/repo/%s' "${resolved#"${PROJECT_DIR}/"}"
    return
  fi
  if [[ -d "${HOST_STORAGE_ROOT}" ]]; then
    storage_dir="$(cd "${HOST_STORAGE_ROOT}" && pwd -P)"
  fi
  if [[ -n "${storage_dir}" && "${resolved}" == "${storage_dir}/"* ]]; then
    printf '/workspace/storage/%s' "${resolved#"${storage_dir}/"}"
    return
  fi
  fail 2 "Config path must be inside the repository or configured STORAGE_ROOT: ${requested}"
}

validate_semantic_device_arguments() {
  local arguments=("$@")
  local count=0
  local index=0
  local value

  while (( index < ${#arguments[@]} )); do
    case "${arguments[index]}" in
      --queue-gpu|--queue-gpu=*)
        fail 2 "--queue-gpu is a wrapper option and must appear before the job type."
        ;;
      --cpu|--cpu=*)
        fail 2 "Obsolete --cpu is unsupported; queued jobs always use --device cuda."
        ;;
      --device)
        if (( index + 1 >= ${#arguments[@]} )); then
          fail 2 "--device requires one of auto, cuda, or cpu."
        fi
        count=$((count + 1))
        value="${arguments[index + 1]}"
        index=$((index + 2))
        ;;
      --device=*)
        count=$((count + 1))
        value="${arguments[index]#--device=}"
        index=$((index + 1))
        ;;
      *)
        index=$((index + 1))
        ;;
    esac
    if (( count > 1 )); then
      fail 2 "Duplicate or conflicting --device options are not allowed for queued jobs."
    fi
  done

  if (( count == 1 )) && [[ "${value}" != "cuda" ]]; then
    fail 2 "Queued jobs require strict --device cuda; received --device ${value@Q}."
  fi
}

QUEUE_GPU_REQUEST="prompt"
if (( $# >= 1 )) && [[ "$1" == "--queue-gpu" ]]; then
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    fail 2 "--queue-gpu requires auto or one reported GPU index."
  fi
  QUEUE_GPU_REQUEST="$2"
  shift 2
elif (( $# >= 1 )) && [[ "$1" == --queue-gpu=* ]]; then
  fail 2 "Use the documented form: --queue-gpu auto|INDEX before the job type."
fi

if (( $# == 0 )); then
  usage
  exit 2
fi

JOB_TYPE="$1"
shift
SEMANTIC_ARGS=("$@")
case "${JOB_TYPE}" in
  train|optuna)
    if (( ${#SEMANTIC_ARGS[@]} == 0 )); then
      fail 2 "${JOB_TYPE} requires a YAML config path."
    fi
    SEMANTIC_ARGS[0]="$(resolve_config_argument "${SEMANTIC_ARGS[0]}")"
    ;;
  artifacts)
    ;;
  *)
    usage
    fail 2 "Unsupported job type: ${JOB_TYPE}"
    ;;
esac
validate_semantic_device_arguments "${SEMANTIC_ARGS[@]}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail 1 "nvidia-smi is required for GPU selection but was not found on PATH."
fi
if ! command -v runTSGPU.py >/dev/null 2>&1; then
  fail 1 "runTSGPU.py is required for queue submission but was not found on PATH."
fi

if ! GPU_REPORT="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"; then
  fail 1 "nvidia-smi failed while querying GPU utilization."
fi
if [[ -z "$(trim_whitespace "${GPU_REPORT}")" ]]; then
  fail 1 "nvidia-smi returned no GPUs."
fi

GPU_IDS=()
GPU_NAMES=()
GPU_UTILIZATIONS=()
GPU_MEMORY_USED=()
GPU_MEMORY_TOTAL=()
AUTO_GPU=""
AUTO_MEMORY_USED=""
while IFS= read -r line || [[ -n "${line}" ]]; do
  IFS=',' read -r raw_index raw_name raw_utilization raw_used raw_total extra <<< "${line}"
  gpu_index="$(trim_whitespace "${raw_index:-}")"
  gpu_name="$(trim_whitespace "${raw_name:-}")"
  gpu_utilization="$(trim_whitespace "${raw_utilization:-}")"
  gpu_used="$(trim_whitespace "${raw_used:-}")"
  gpu_total="$(trim_whitespace "${raw_total:-}")"
  if [[ -n "${extra:-}" || ! "${gpu_index}" =~ ^(0|[1-9][0-9]*)$ || -z "${gpu_name}" \
      || ! "${gpu_utilization}" =~ ^[0-9]+$ || ! "${gpu_used}" =~ ^[0-9]+$ \
      || ! "${gpu_total}" =~ ^[0-9]+$ ]]; then
    fail 1 "Malformed nvidia-smi GPU record: ${line@Q}"
  fi
  if (( gpu_utilization > 100 || gpu_total == 0 || gpu_used > gpu_total )); then
    fail 1 "Invalid nvidia-smi utilization values for GPU ${gpu_index}."
  fi
  for reported_index in "${GPU_IDS[@]}"; do
    if [[ "${reported_index}" == "${gpu_index}" ]]; then
      fail 1 "nvidia-smi reported duplicate GPU index ${gpu_index}."
    fi
  done

  GPU_IDS+=("${gpu_index}")
  GPU_NAMES+=("${gpu_name}")
  GPU_UTILIZATIONS+=("${gpu_utilization}")
  GPU_MEMORY_USED+=("${gpu_used}")
  GPU_MEMORY_TOTAL+=("${gpu_total}")
  if [[ -z "${AUTO_GPU}" ]] || (( gpu_used < AUTO_MEMORY_USED )) \
      || (( gpu_used == AUTO_MEMORY_USED && gpu_index < AUTO_GPU )); then
    AUTO_GPU="${gpu_index}"
    AUTO_MEMORY_USED="${gpu_used}"
  fi
done <<< "${GPU_REPORT}"

if (( ${#GPU_IDS[@]} == 0 )) || [[ -z "${AUTO_GPU}" ]]; then
  fail 1 "No valid GPU was reported by nvidia-smi."
fi

printf 'Current GPU usage:\n'
for index in "${!GPU_IDS[@]}"; do
  printf '  GPU %s: %s | utilization %s%% | memory %s/%s MiB\n' \
    "${GPU_IDS[index]}" "${GPU_NAMES[index]}" "${GPU_UTILIZATIONS[index]}" \
    "${GPU_MEMORY_USED[index]}" "${GPU_MEMORY_TOTAL[index]}"
done
printf 'Proposed GPU: %s (least allocated memory; lowest index breaks ties)\n' "${AUTO_GPU}"

GPU_LIST="$(IFS=,; printf '%s' "${GPU_IDS[*]}")"
case "${QUEUE_GPU_REQUEST}" in
  prompt)
    if ! IFS= read -r -p "Select GPU (${GPU_LIST}; Enter for proposed ${AUTO_GPU}): " GPU_ID; then
      fail 2 "GPU selection input closed before a choice was received."
    fi
    GPU_ID="${GPU_ID:-${AUTO_GPU}}"
    ;;
  auto)
    GPU_ID="${AUTO_GPU}"
    ;;
  *)
    if [[ ! "${QUEUE_GPU_REQUEST}" =~ ^(0|[1-9][0-9]*)$ ]]; then
      fail 2 "--queue-gpu must be auto or one non-negative reported GPU index."
    fi
    GPU_ID="${QUEUE_GPU_REQUEST}"
    ;;
esac

GPU_IS_REPORTED=false
for reported_index in "${GPU_IDS[@]}"; do
  if [[ "${reported_index}" == "${GPU_ID}" ]]; then
    GPU_IS_REPORTED=true
    break
  fi
done
if [[ "${GPU_IS_REPORTED}" != true ]]; then
  fail 2 "GPU ${GPU_ID@Q} is not one of the reported indices: ${GPU_LIST}."
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$(mktemp --suffix=.log "${LOG_DIR}/${TIMESTAMP}__${JOB_TYPE}__gpu${GPU_ID}__XXXXXX")"
LOG_BASENAME="$(basename "${LOG_PATH}")"

printf 'Selected GPU: %s\n' "${GPU_ID}"
printf 'Submitting validated %s job through runTSGPU.py.\n' "${JOB_TYPE}"
cd "${PROJECT_DIR}"
runTSGPU.py -g"${GPU_ID}" -- "${PROJECT_DIR}/scripts/_docker_run.sh" \
  "${GPU_ID}" \
  "${JOB_TYPE}" \
  "${LOG_BASENAME}" \
  "${SEMANTIC_ARGS[@]}"

printf 'Queued %s job on GPU %s.\n' "${JOB_TYPE}" "${GPU_ID}"
printf 'Queue: runTSGPU.py -g%s -- %s/scripts/_docker_run.sh [validated %s arguments]\n' \
  "${GPU_ID}" "${PROJECT_DIR}" "${JOB_TYPE}"
printf 'Log:   %s\n' "${LOG_PATH}"
printf 'Tail:  tail -F %q\n' "${LOG_PATH}"
