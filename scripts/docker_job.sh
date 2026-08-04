#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
MODEL_DIR="${PROJECT_DIR}/model_training"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${PROJECT_DIR}/../storage}"
HOST_GENERATED_DATA_ROOT="${HOST_STORAGE_ROOT}/data_generation"
HOST_MODEL_TRAINING_DATA_ROOT="${HOST_STORAGE_ROOT}/data_training"
IMAGE_NAME="grainlegumes-pino-airflow"
PREFLIGHT_RUNTIME="/workspace/repo/scripts/config_preflight_runtime.py"
PROJECT_PYTHON_MINIMUM="3.11"
DETECTED_HOST_PYTHON_EXECUTABLE="not found"
DETECTED_HOST_PYTHON_VERSION="unavailable"

usage() {
  cat >&2 <<EOF
Usage:
  $0 train <experiment_config> [training options...] [--queue-gpu auto|INDEX] [--follow]
  $0 optuna <optuna_config> [Optuna options...] [--queue-gpu auto|INDEX] [--follow]
  $0 artifacts (--task TASK | --runs-root PATH) [artifact options...] [--queue-gpu auto|INDEX]

Workflows:
  train <experiment_config>
      Submit one direct run that builds or reuses bundle-local artifacts after completion.
      Add --no-build-artifacts after the config to skip only post-training artifacts.
      Resume is explicit.
  optuna <optuna_config>
      Submit one Optuna study and return immediately. Persistent continuation uses study storage.
  artifacts (--task TASK | --runs-root PATH)
      Generate or validate analysis artifacts for completed runs and return immediately.

GPU selection:
  omit --queue-gpu  in an interactive TTY, show usage, and prompt for one host GPU.
                    Press Enter to accept the least-memory proposal
  --queue-gpu auto  select the least-memory host GPU without prompting
  --queue-gpu INDEX select one reported physical host GPU without prompting

Non-interactive callers must provide --queue-gpu auto or --queue-gpu INDEX.
--follow is independent of GPU selection and follows the detached worker host log.
Ctrl+C during GPU selection submits nothing. Ctrl+C during log following stops only
the follower. The queue job continues with no worker input channel. No --wait mode
exists. The wrapper never polls or waits for worker completion.
EOF
}

fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

detect_host_python() {
  local executable=""
  local version=""

  if executable="$(command -v python 2>/dev/null)" && [[ -n "${executable}" ]]; then
    DETECTED_HOST_PYTHON_EXECUTABLE="${executable}"
    if version="$("${executable}" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))' 2>/dev/null)" \
        && [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      DETECTED_HOST_PYTHON_VERSION="${version}"
    fi
  fi
}

host_python_is_below_project_minimum() {
  if [[ ! "${DETECTED_HOST_PYTHON_VERSION}" =~ ^([0-9]+)\.([0-9]+)\.[0-9]+$ ]]; then
    return 1
  fi
  (( BASH_REMATCH[1] < 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] < 11) ))
}

preflight_runtime_failure() {
  local detail="$1"
  printf 'Configuration preflight could not start.\n' >&2
  printf 'Host Python: %s (%s)\n' "${DETECTED_HOST_PYTHON_VERSION}" "${DETECTED_HOST_PYTHON_EXECUTABLE}" >&2
  printf 'Required Python: >=%s\n' "${PROJECT_PYTHON_MINIMUM}" >&2
  printf 'Maintained preflight container: %s\n' "${detail}" >&2
  exit 1
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

resolve_host_config_argument() {
  local requested="$1"
  local candidate=""
  local candidate_dir

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
  printf '%s/%s' "${candidate_dir}" "$(basename "${candidate}")"
}

translate_config_argument() {
  local resolved="$1"
  local generated_dir=""
  local training_dir=""

  if [[ "${resolved}" == "${MODEL_DIR}/"* ]]; then
    printf '%s' "${resolved#"${MODEL_DIR}/"}"
    return
  fi
  if [[ "${resolved}" == "${PROJECT_DIR}/"* ]]; then
    printf '/workspace/repo/%s' "${resolved#"${PROJECT_DIR}/"}"
    return
  fi
  if [[ -d "${HOST_GENERATED_DATA_ROOT}" ]]; then
    generated_dir="$(cd "${HOST_GENERATED_DATA_ROOT}" && pwd -P)"
  fi
  if [[ -n "${generated_dir}" && "${resolved}" == "${generated_dir}/"* ]]; then
    printf '/workspace/repo/data_generation/data/%s' "${resolved#"${generated_dir}/"}"
    return
  fi
  if [[ -d "${HOST_MODEL_TRAINING_DATA_ROOT}" ]]; then
    training_dir="$(cd "${HOST_MODEL_TRAINING_DATA_ROOT}" && pwd -P)"
  fi
  if [[ -n "${training_dir}" && "${resolved}" == "${training_dir}/"* ]]; then
    printf '/workspace/repo/model_training/data/%s' "${resolved#"${training_dir}/"}"
    return
  fi
  fail 2 "Config path must be inside the repository or one configured data domain: ${resolved}"
}

translate_semantic_path() {
  local requested="$1"
  local resolved
  local generated_root
  local training_root
  local project_root

  if [[ -z "${requested}" ]]; then
    fail 2 "Path-valued semantic options require a non-empty value."
  fi
  if [[ "${requested}" != /* ]]; then
    printf '%s' "${requested}"
    return
  fi
  if [[ "${requested}" == "/workspace/repo" || "${requested}" == "/workspace/repo/"* ]]; then
    printf '%s' "${requested}"
    return
  fi
  if ! command -v realpath >/dev/null 2>&1; then
    fail 1 "realpath is required for host-to-container path translation but was not found on PATH."
  fi

  resolved="$(realpath -m -- "${requested}")"
  generated_root="$(realpath -m -- "${HOST_GENERATED_DATA_ROOT}")"
  training_root="$(realpath -m -- "${HOST_MODEL_TRAINING_DATA_ROOT}")"
  project_root="$(realpath -m -- "${PROJECT_DIR}")"

  if [[ "${resolved}" == "${generated_root}" ]]; then
    printf '/workspace/repo/data_generation/data'
  elif [[ "${resolved}" == "${generated_root}/"* ]]; then
    printf '/workspace/repo/data_generation/data/%s' "${resolved#"${generated_root}/"}"
  elif [[ "${resolved}" == "${training_root}" ]]; then
    printf '/workspace/repo/model_training/data'
  elif [[ "${resolved}" == "${training_root}/"* ]]; then
    printf '/workspace/repo/model_training/data/%s' "${resolved#"${training_root}/"}"
  elif [[ "${resolved}" == "${project_root}" ]]; then
    printf '/workspace/repo'
  elif [[ "${resolved}" == "${project_root}/"* ]]; then
    printf '/workspace/repo/%s' "${resolved#"${project_root}/"}"
  else
    printf '%s' "${requested}"
  fi
}

run_config_preflight() {
  local workflow="$1"
  local container_config="$2"
  local output=""
  local status=0
  local mount_args=(
    --mount "type=bind,source=${PROJECT_DIR},target=/workspace/repo,readonly"
  )
  local command=()

  case "${container_config}" in
    /workspace/repo/data_generation/data/*)
      mount_args+=(
        --mount "type=bind,source=${HOST_GENERATED_DATA_ROOT},target=/workspace/repo/data_generation/data,readonly"
      )
      ;;
    /workspace/repo/model_training/data/*)
      mount_args+=(
        --mount "type=bind,source=${HOST_MODEL_TRAINING_DATA_ROOT},target=/workspace/repo/model_training/data,readonly"
      )
      ;;
  esac

  detect_host_python
  if host_python_is_below_project_minimum; then
    printf '%s\n' \
      "Host Python ${DETECTED_HOST_PYTHON_VERSION} at ${DETECTED_HOST_PYTHON_EXECUTABLE} is below the project minimum >=${PROJECT_PYTHON_MINIMUM}. Running configuration preflight in the maintained CPU-only project container." \
      >&2
  fi

  if ! command -v docker >/dev/null 2>&1; then
    preflight_runtime_failure "Docker was not found on PATH."
  fi
  if ! docker info >/dev/null 2>&1; then
    preflight_runtime_failure "the Docker daemon is unavailable."
  fi
  if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    preflight_runtime_failure "image '${IMAGE_NAME}' is missing. Build it with ./scripts/docker_build.sh."
  fi

  command=(
    docker run --rm
    --network none
    --workdir /workspace/repo/model_training
    --tmpfs /tmp:rw,nosuid,nodev,size=64m
    -e HOME=/tmp
    -e PYTHONDONTWRITEBYTECODE=1
    -e PYTHONNOUSERSITE=1
    -e PYTHONPATH=/workspace/repo/model_training
    -e PROJECT_ROOT=/workspace/repo
    -e GENERATED_DATA_ROOT=/workspace/repo/data_generation/data
    -e MODEL_TRAINING_DATA_ROOT=/workspace/repo/model_training/data
    "${mount_args[@]}"
    "${IMAGE_NAME}"
    python "${PREFLIGHT_RUNTIME}" "${workflow}" "${container_config}"
  )
  if output="$("${command[@]}")"; then
    printf '%s' "${output}"
    return 0
  else
    status=$?
  fi
  if [[ -n "${output}" ]]; then
    printf '%s\n' "${output}"
  fi
  return "${status}"
}

is_semantic_path_option() {
  local job_type="$1"
  local option="$2"

  case "${job_type}:${option}" in
    train:--resume|train:--output-root|optuna:--output-root|artifacts:--runs-root|artifacts:--dataset-root|artifacts:--metadata-root)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

translate_semantic_path_options() {
  local job_type="$1"
  shift
  local arguments=("$@")
  local translated=()
  local index=0
  local argument
  local option
  local value

  while (( index < ${#arguments[@]} )); do
    argument="${arguments[index]}"
    if is_semantic_path_option "${job_type}" "${argument}"; then
      if (( index + 1 >= ${#arguments[@]} )) || [[ -z "${arguments[index + 1]}" ]]; then
        fail 2 "${argument} requires a non-empty path value."
      fi
      value="$(translate_semantic_path "${arguments[index + 1]}")"
      translated+=("${argument}" "${value}")
      index=$((index + 2))
      continue
    fi

    if [[ "${argument}" == *=* ]]; then
      option="${argument%%=*}"
      if is_semantic_path_option "${job_type}" "${option}"; then
        value="${argument#*=}"
        if [[ -z "${value}" ]]; then
          fail 2 "${option} requires a non-empty path value."
        fi
        translated+=("${option}=$(translate_semantic_path "${value}")")
        index=$((index + 1))
        continue
      fi
    fi

    translated+=("${argument}")
    index=$((index + 1))
  done

  SEMANTIC_ARGS=("${translated[@]}")
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
    fail 2 "Queued jobs require explicit --device cuda. Received --device ${value@Q}."
  fi
}

resolve_artifact_selection() {
  local arguments=("$@")
  local index=0
  local argument
  local value
  local task_seen=false
  local runs_root_seen=false
  local artifact_task=""

  while (( index < ${#arguments[@]} )); do
    argument="${arguments[index]}"
    case "${argument}" in
      --task)
        if (( index + 1 >= ${#arguments[@]} )) || [[ -z "${arguments[index + 1]}" || "${arguments[index + 1]}" == --* ]]; then
          fail 2 "--task requires a non-empty registered task identifier."
        fi
        artifact_task="${arguments[index + 1]}"
        task_seen=true
        index=$((index + 2))
        ;;
      --task=*)
        value="${argument#--task=}"
        if [[ -z "${value}" ]]; then
          fail 2 "--task requires a non-empty registered task identifier."
        fi
        artifact_task="${value}"
        task_seen=true
        index=$((index + 1))
        ;;
      --runs-root)
        if (( index + 1 >= ${#arguments[@]} )) || [[ -z "${arguments[index + 1]}" || "${arguments[index + 1]}" == --* ]]; then
          fail 2 "--runs-root requires a non-empty path value."
        fi
        runs_root_seen=true
        index=$((index + 2))
        ;;
      --runs-root=*)
        value="${argument#--runs-root=}"
        if [[ -z "${value}" ]]; then
          fail 2 "--runs-root requires a non-empty path value."
        fi
        runs_root_seen=true
        index=$((index + 1))
        ;;
      *)
        index=$((index + 1))
        ;;
    esac
  done

  if [[ "${task_seen}" == "${runs_root_seen}" ]]; then
    fail 2 "Artifact jobs require exactly one of --task TASK or --runs-root PATH."
  fi
  if [[ "${task_seen}" == true ]]; then
    if [[ "$(trim_whitespace "${artifact_task}")" != "${artifact_task}" || "${artifact_task}" == */* || "${artifact_task}" == *\\* || "${artifact_task}" == "." || "${artifact_task}" == ".." ]]; then
      fail 2 "--task must be one safe registered task identifier."
    fi
    RESOLVED_TASK="${artifact_task}"
    LOG_SCOPE="${artifact_task}"
  else
    RESOLVED_TASK="not supplied"
    LOG_SCOPE="artifacts"
  fi
}

QUEUE_GPU_REQUEST=""
QUEUE_GPU_SEEN=false
FOLLOW_LOG=false
JOB_TYPE=""
SEMANTIC_ARGS=()

while (( $# > 0 )); do
  argument="$1"
  if [[ -z "${JOB_TYPE}" ]]; then
    case "${argument}" in
      -h|--help)
        usage
        exit 0
        ;;
      --follow)
        if [[ "${FOLLOW_LOG}" == true ]]; then
          fail 2 "Duplicate --follow options are not allowed."
        fi
        FOLLOW_LOG=true
        shift
        ;;
      --queue-gpu)
        if [[ "${QUEUE_GPU_SEEN}" == true ]]; then
          fail 2 "Duplicate --queue-gpu options are not allowed."
        fi
        if (( $# < 2 )) || [[ -z "$2" ]]; then
          fail 2 "--queue-gpu requires auto or one reported GPU index."
        fi
        QUEUE_GPU_REQUEST="$2"
        QUEUE_GPU_SEEN=true
        shift 2
        ;;
      --queue-gpu=*)
        fail 2 "Use the documented form: --queue-gpu auto|INDEX."
        ;;
      train|optuna|artifacts)
        JOB_TYPE="${argument}"
        shift
        ;;
      *)
        usage
        fail 2 "Unsupported job type or wrapper option: ${argument}"
        ;;
    esac
    continue
  fi

  case "${argument}" in
    --follow)
      if [[ "${FOLLOW_LOG}" == true ]]; then
        fail 2 "Duplicate --follow options are not allowed."
      fi
      FOLLOW_LOG=true
      shift
      ;;
    --queue-gpu)
      if [[ "${QUEUE_GPU_SEEN}" == true ]]; then
        fail 2 "Duplicate --queue-gpu options are not allowed."
      fi
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        fail 2 "--queue-gpu requires auto or one reported GPU index."
      fi
      QUEUE_GPU_REQUEST="$2"
      QUEUE_GPU_SEEN=true
      shift 2
      ;;
    --queue-gpu=*)
      fail 2 "Use the documented form: --queue-gpu auto|INDEX."
      ;;
    --wait|--wait=*|--follow-and-wait|--follow-and-wait=*)
      fail 2 "Queue completion waiting is unsupported. Submission is detached."
      ;;
    *)
      SEMANTIC_ARGS+=("${argument}")
      shift
      ;;
  esac
done

if [[ -z "${JOB_TYPE}" ]]; then
  usage
  exit 2
fi

RESOLVED_TASK="not supplied"
LOG_SCOPE=""
CANONICAL_CONFIG_PATH="not applicable"
case "${JOB_TYPE}" in
  train|optuna)
    if (( ${#SEMANTIC_ARGS[@]} == 0 )); then
      fail 2 "${JOB_TYPE} requires a YAML config path."
    fi
    if [[ "${JOB_TYPE}" == "optuna" ]]; then
      for argument in "${SEMANTIC_ARGS[@]}"; do
        case "${argument}" in
          --resume|--resume=*)
            fail 2 "--resume is a training checkpoint option and is unsupported for Optuna study continuation."
            ;;
        esac
      done
    fi
    HOST_CONFIG_PATH="$(resolve_host_config_argument "${SEMANTIC_ARGS[0]}")"
    SEMANTIC_ARGS[0]="$(translate_config_argument "${HOST_CONFIG_PATH}")"
    if PREFLIGHT_OUTPUT="$(run_config_preflight "${JOB_TYPE}" "${SEMANTIC_ARGS[0]}")"; then
      :
    else
      PREFLIGHT_STATUS=$?
      if [[ -n "${PREFLIGHT_OUTPUT}" ]]; then
        printf '%s\n' "${PREFLIGHT_OUTPUT}"
      fi
      exit "${PREFLIGHT_STATUS}"
    fi
    IFS=$'\t' read -r SUPPLIED_CONFIG_FAMILY RESOLVED_TASK CANONICAL_CONFIG_PATH PREFLIGHT_EXTRA <<< "${PREFLIGHT_OUTPUT}"
    if [[ -z "${SUPPLIED_CONFIG_FAMILY}" || -z "${RESOLVED_TASK}" || -z "${CANONICAL_CONFIG_PATH}" || -n "${PREFLIGHT_EXTRA:-}" ]]; then
      fail 1 "Configuration preflight returned a malformed container summary."
    fi
    LOG_SCOPE="${RESOLVED_TASK}"
    ;;
  artifacts)
    if [[ "${FOLLOW_LOG}" == true ]]; then
      fail 2 "--follow is supported only for train and optuna workflows."
    fi
    resolve_artifact_selection "${SEMANTIC_ARGS[@]}"
    ;;
esac

if [[ "${FOLLOW_LOG}" == true ]] && ! command -v tail >/dev/null 2>&1; then
  fail 1 "tail is required for --follow but was not found on PATH."
fi
translate_semantic_path_options "${JOB_TYPE}" "${SEMANTIC_ARGS[@]}"
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
printf 'Proposed GPU: %s\n' "${AUTO_GPU}"
printf 'Proposal reason: least allocated memory. Lowest index breaks ties.\n'

GPU_LIST="$(IFS=,; printf '%s' "${GPU_IDS[*]}")"
gpu_is_reported() {
  local candidate="$1"
  local reported_index
  for reported_index in "${GPU_IDS[@]}"; do
    if [[ "${reported_index}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

case "${QUEUE_GPU_REQUEST}" in
  "")
    if [[ ! -t 0 ]]; then
      printf '%s\n\n' "GPU selection requires an explicit option in non-interactive mode." >&2
      printf '%s\n' "Use automatic selection:" "  --queue-gpu auto" "" >&2
      printf '%s\n' "Or select one physical host GPU:" "  --queue-gpu INDEX" >&2
      exit 2
    fi

    selection_interrupted() {
      printf '\nGPU selection cancelled. No queue job was submitted.\n' >&2
      exit 130
    }
    trap selection_interrupted INT

    GPU_ID=""
    SELECTION_ATTEMPTS=0
    while (( SELECTION_ATTEMPTS < 10 )); do
      printf 'Select GPU (%s. Enter for proposed %s): ' "${GPU_LIST}" "${AUTO_GPU}"
      if ! IFS= read -r GPU_INPUT; then
        trap - INT
        unset -f selection_interrupted
        fail 2 "GPU selection input closed before a choice was received. No queue job was submitted."
      fi
      GPU_INPUT="$(trim_whitespace "${GPU_INPUT}")"
      if [[ -z "${GPU_INPUT}" ]]; then
        GPU_ID="${AUTO_GPU}"
        break
      fi
      if [[ ! "${GPU_INPUT}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        printf 'Invalid GPU selection %s. Enter one reported index or press Enter.\n' "${GPU_INPUT@Q}" >&2
      elif ! gpu_is_reported "${GPU_INPUT}"; then
        printf 'GPU %s is not one of the reported indices: %s.\n' "${GPU_INPUT@Q}" "${GPU_LIST}" >&2
      else
        GPU_ID="${GPU_INPUT}"
        break
      fi
      SELECTION_ATTEMPTS=$((SELECTION_ATTEMPTS + 1))
    done
    trap - INT
    unset -f selection_interrupted
    if [[ -z "${GPU_ID}" ]]; then
      fail 2 "GPU selection failed after 10 invalid attempts. No queue job was submitted."
    fi
    printf 'Selected GPU: %s\n' "${GPU_ID}"
    ;;
  auto)
    GPU_ID="${AUTO_GPU}"
    printf 'Automatically selected GPU: %s\n' "${GPU_ID}"
    printf 'Reason: least allocated memory. Lowest index breaks ties.\n'
    ;;
  *)
    if [[ ! "${QUEUE_GPU_REQUEST}" =~ ^(0|[1-9][0-9]*)$ ]]; then
      fail 2 "--queue-gpu must be auto or one non-negative reported GPU index."
    fi
    GPU_ID="${QUEUE_GPU_REQUEST}"
    if ! gpu_is_reported "${GPU_ID}"; then
      fail 2 "GPU ${GPU_ID@Q} is not one of the reported indices: ${GPU_LIST}."
    fi
    printf 'Selected GPU: %s\n' "${GPU_ID}"
    printf 'Selection source: explicit --queue-gpu\n'
    ;;
esac

TASK_SPOOLER_SOCKET="/etc/ts/socket_${GPU_ID}"
LOG_DIR="${HOST_MODEL_TRAINING_DATA_ROOT}/processed/${LOG_SCOPE}/logs/queue"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$(mktemp --suffix=.log "${LOG_DIR}/${TIMESTAMP}__${JOB_TYPE}__gpu${GPU_ID}__XXXXXX")"
LOG_BASENAME="$(basename "${LOG_PATH}")"
QUEUE_COMMAND=(
  runTSGPU.py
  "-g${GPU_ID}"
  --
  "${PROJECT_DIR}/scripts/_docker_run.sh"
  "${GPU_ID}"
  "${JOB_TYPE}"
  "${LOG_SCOPE}"
  "${LOG_BASENAME}"
  "${SEMANTIC_ARGS[@]}"
)

cd "${PROJECT_DIR}"
if QUEUE_OUTPUT="$(TS_SOCKET="${TASK_SPOOLER_SOCKET}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${QUEUE_COMMAND[@]}")"; then
  :
else
  QUEUE_STATUS=$?
  if [[ -n "${QUEUE_OUTPUT}" ]]; then
    printf 'Queue submission output:\n%s\n' "${QUEUE_OUTPUT}" >&2
  fi
  fail "${QUEUE_STATUS}" "Queue submission failed for ${JOB_TYPE} workflow."
fi

QUEUE_JOB_IDS=()
REPORTED_QUEUE_SOCKETS=()
QUEUE_DIAGNOSTICS=()
while IFS= read -r queue_line || [[ -n "${queue_line}" ]]; do
  trimmed_queue_line="$(trim_whitespace "${queue_line}")"
  if [[ -z "${trimmed_queue_line}" ]]; then
    continue
  fi
  if [[ "${trimmed_queue_line}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    QUEUE_JOB_IDS+=("${trimmed_queue_line}")
  elif [[ "${trimmed_queue_line}" =~ ^TS[[:space:]]socket:[[:space:]]+(/[^[:space:]]+)$ ]]; then
    REPORTED_QUEUE_SOCKETS+=("${BASH_REMATCH[1]}")
  else
    QUEUE_DIAGNOSTICS+=("${trimmed_queue_line}")
  fi
done <<< "${QUEUE_OUTPUT}"

if (( ${#QUEUE_JOB_IDS[@]} == 1 )); then
  QUEUE_JOB_ID="${QUEUE_JOB_IDS[0]}"
else
  QUEUE_JOB_ID="unavailable"
  TRIMMED_QUEUE_OUTPUT="$(trim_whitespace "${QUEUE_OUTPUT}")"
  if [[ -n "${TRIMMED_QUEUE_OUTPUT}" ]]; then
    printf 'Queue submission output:\n%s\n' "${TRIMMED_QUEUE_OUTPUT}"
  fi
fi

if (( ${#REPORTED_QUEUE_SOCKETS[@]} == 1 )); then
  if [[ "${REPORTED_QUEUE_SOCKETS[0]}" != "${TASK_SPOOLER_SOCKET}" ]]; then
    QUEUE_DIAGNOSTICS+=(
      "queue helper socket report ${REPORTED_QUEUE_SOCKETS[0]@Q} did not match selected GPU socket ${TASK_SPOOLER_SOCKET@Q}"
    )
  fi
elif (( ${#REPORTED_QUEUE_SOCKETS[@]} > 1 )); then
  QUEUE_DIAGNOSTICS+=("queue helper reported multiple task-spooler sockets")
fi

if (( ${#QUEUE_DIAGNOSTICS[@]} > 0 )); then
  printf 'Queue submission diagnostics:\n'
  printf '  %s\n' "${QUEUE_DIAGNOSTICS[@]}"
fi

printf 'Queue job ID: %s\n' "${QUEUE_JOB_ID}"
printf 'Workflow: %s\n' "${JOB_TYPE}"
printf 'Task: %s\n' "${RESOLVED_TASK}"
printf 'Config: %s\n' "${CANONICAL_CONFIG_PATH}"
printf 'Selected host GPU: %s\n' "${GPU_ID}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${GPU_ID}"
printf 'Container CUDA device: 0\n'
printf 'Task-spooler socket: %s\n' "${TASK_SPOOLER_SOCKET}"
printf 'Queued command:'
printf ' %q' "${QUEUE_COMMAND[@]}"
printf '\n'
printf 'Host log: %s\n' "${LOG_PATH}"
printf 'Follow manually:\n'
printf '  tail -n +1 -F %q\n' "${LOG_PATH}"

if [[ "${FOLLOW_LOG}" != true ]]; then
  exit 0
fi

printf 'Following host log. Press Ctrl+C to stop following. The queue job continues.\n'
FOLLOW_INTERRUPTED=false
follow_interrupt() {
  FOLLOW_INTERRUPTED=true
}
trap follow_interrupt INT
set +e
tail -n +1 -F "${LOG_PATH}"
TAIL_STATUS=$?
set -e
trap - INT
unset -f follow_interrupt
if [[ "${FOLLOW_INTERRUPTED}" == true ]] || (( TAIL_STATUS == 130 )); then
  printf 'Log following stopped. Queue job %s continues independently.\n' "${QUEUE_JOB_ID}"
  exit 0
fi
if (( TAIL_STATUS != 0 )); then
  fail "${TAIL_STATUS}" "Host log following failed. Queue job ${QUEUE_JOB_ID} continues independently."
fi
printf 'Log following ended. Queue job %s continues independently.\n' "${QUEUE_JOB_ID}"
