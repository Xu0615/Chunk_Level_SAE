#!/usr/bin/env bash
# Shared latest Chunk-SAE pipeline for model-specific run_scripts entrypoints.
#
# EleutherAI/the_pile_deduplicated is read directly through HF streaming.
# There is no local canonical-corpus, catalog, dedup, or sample-plan stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"
TORCHRUN=("${PY}" -m torch.distributed.run)
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
SAE_CPU_THREADS="${SAE_CPU_THREADS:-4}"
export SAE_CPU_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SAE_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SAE_CPU_THREADS}}"

PROTOCOL_VERSION="${PROTOCOL_VERSION:-pile_proportional_exact_occurrence}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"

: "${MODEL:?MODEL must be set by the model-specific run_scripts launcher}"
: "${MODEL_NAME:?MODEL_NAME must be set by the model-specific launcher}"
: "${NUM_LAYERS:?NUM_LAYERS must be set by the model-specific launcher}"
: "${LAYER:?LAYER must be set by the model-specific launcher}"
: "${FORWARD_BATCH_SIZE:?FORWARD_BATCH_SIZE must be set by the launcher}"
EVIDENCE_FORWARD_BATCH_SIZE="${EVIDENCE_FORWARD_BATCH_SIZE:-8}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-8}"
GPU_COUNT="${GPU_COUNT:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

HF_DATASET="${HF_DATASET:-EleutherAI/the_pile_deduplicated}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb}"
HF_DATASET_PHYSICAL_SPLIT="${HF_DATASET_PHYSICAL_SPLIT:-train}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DOCUMENT_SHUFFLE_SEED="${DOCUMENT_SHUFFLE_SEED:-71}"
STREAM_SHUFFLE_BUFFER="${STREAM_SHUFFLE_BUFFER:-16384}"
CORPUS_SOURCE="${CORPUS_SOURCE:-Pile-Deduplicated}"
export HF_HOME="${HF_HOME:-${TMPDIR:-${REPO_ROOT}/.cache}/chunk-saes-huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Standard HTTP(S)_PROXY environment variables are honored by dependencies.

TARGET_TRAIN_TOKENS="${TARGET_TRAIN_TOKENS:-1000000000}"
TARGET_VALIDATION_TOKENS="${TARGET_VALIDATION_TOKENS:-10000128}"
TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-43}"
VALIDATION_SAMPLE_SEED="${VALIDATION_SAMPLE_SEED:-44}"
TRAIN_SEED="${TRAIN_SEED:-42}"
CHUNK_LENGTHS="${CHUNK_LENGTHS:-32,64,128,256,512}"
MAX_DOCUMENT_REUSES="${MAX_DOCUMENT_REUSES:-64}"

STREAM_TOKENIZER_THREADS="${STREAM_TOKENIZER_THREADS:-16}"
STREAM_TOKENIZER_BATCH_SIZE="${STREAM_TOKENIZER_BATCH_SIZE:-1024}"
STREAM_TOKENIZER_BATCH_CHARS="${STREAM_TOKENIZER_BATCH_CHARS:-8000000}"
STREAM_SCHEDULER_WINDOW_PAIRS="${STREAM_SCHEDULER_WINDOW_PAIRS:-8192}"
STREAM_PLANNER_MODE="${STREAM_PLANNER_MODE:-central}"
STREAM_DISPATCH_QUEUE_DEPTH="${STREAM_DISPATCH_QUEUE_DEPTH:-4}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/qwen/${MODEL_NAME}}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${OUTPUT_ROOT}/${PROTOCOL_VERSION}}"
PERSISTENT_ACTIVATION_CACHE_ROOT="${PERSISTENT_ACTIVATION_CACHE_ROOT:-${PIPELINE_ROOT}/data}"
PERSISTENT_TRAIN_CACHE_DIR="${PERSISTENT_TRAIN_CACHE_DIR:-${CEPH_TRAIN_CACHE_DIR:-${PERSISTENT_ACTIVATION_CACHE_ROOT}/layer${LAYER}_train_cache_exact${TARGET_TRAIN_TOKENS}}}"
PERSISTENT_VALIDATION_CACHE_DIR="${PERSISTENT_VALIDATION_CACHE_DIR:-${CEPH_VALIDATION_CACHE_DIR:-${PERSISTENT_ACTIVATION_CACHE_ROOT}/layer${LAYER}_validation_cache_exact${TARGET_VALIDATION_TOKENS}}}"

# Activation caches are durable outputs, so the default read/write paths live
# directly under <PIPELINE_ROOT>/data.  LOCAL_ACTIVATION_CACHE_ROOT remains an
# explicit opt-in for runs that need local-NVMe staging followed by mirroring.
LOCAL_ACTIVATION_CACHE_ROOT="${LOCAL_ACTIVATION_CACHE_ROOT:-}"
if [[ -n "${LOCAL_ACTIVATION_CACHE_ROOT}" ]]; then
  TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${LOCAL_ACTIVATION_CACHE_ROOT}/train}"
  VALIDATION_CACHE_DIR="${VALIDATION_CACHE_DIR:-${LOCAL_ACTIVATION_CACHE_ROOT}/validation}"
else
  TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${PERSISTENT_TRAIN_CACHE_DIR}}"
  VALIDATION_CACHE_DIR="${VALIDATION_CACHE_DIR:-${PERSISTENT_VALIDATION_CACHE_DIR}}"
fi

# Backward-compatible aliases for older launch environments.
CEPH_TRAIN_CACHE_DIR="${PERSISTENT_TRAIN_CACHE_DIR}"
CEPH_VALIDATION_CACHE_DIR="${PERSISTENT_VALIDATION_CACHE_DIR}"

FORWARD_TOKEN_BUDGET="${FORWARD_TOKEN_BUDGET:-65536}"
CACHE_SHARD_TOKENS="${CACHE_SHARD_TOKENS:-131072}"
ASYNC_WRITE_BATCHES="${ASYNC_WRITE_BATCHES:-2}"
SHARD_WRITE_WORKERS="${SHARD_WRITE_WORKERS:-2}"
MAX_PENDING_SHARDS="${MAX_PENDING_SHARDS:-4}"
WRITER_BATCH_TOKENS="${WRITER_BATCH_TOKENS:-131072}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CACHE_MIRROR_WORKERS="${CACHE_MIRROR_WORKERS:-8}"
if [[ -z "${CACHE_MIRROR_POLICY:-}" ]]; then
  if [[ "${TRAIN_CACHE_DIR}" == "${PERSISTENT_TRAIN_CACHE_DIR}" &&
        "${VALIDATION_CACHE_DIR}" == "${PERSISTENT_VALIDATION_CACHE_DIR}" ]]; then
    CACHE_MIRROR_POLICY="disabled"
  else
    CACHE_MIRROR_POLICY="after_training"
  fi
fi
CACHE_MIRROR_VERIFY_DESTINATION="${CACHE_MIRROR_VERIFY_DESTINATION:-0}"

SAE_MODES="${SAE_MODES:-token,temporal,mean,cross}"
JOINT_CHUNK_ALPHA="${JOINT_CHUNK_ALPHA:-0.25}"
JOINT_CROSS_PREFIX="${JOINT_CROSS_PREFIX:-0}"
DIRECTION_POLICY="${DIRECTION_POLICY:-both}"
SAE_WIDTH="${SAE_WIDTH:-65536}"
K="${K:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32000}"
TRAIN_STEPS="${TRAIN_STEPS:-31250}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
SAE_IMPLEMENTATION_TAG="${SAE_IMPLEMENTATION_TAG:-batchtopk_separate_centers_preemptive_auxk_lr_floor_v3_latest}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_layer${LAYER}_w${SAE_WIDTH}_k${K}_train${TARGET_TRAIN_TOKENS}_seed${TRAIN_SEED}_${PROTOCOL_VERSION}_saeimpl_${SAE_IMPLEMENTATION_TAG}}"
SAE_ROOT="${SAE_ROOT:-${PIPELINE_ROOT}/checkpoints/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PIPELINE_ROOT}/runs}"
ATTAINABLE_REFERENCE_FVES="${ATTAINABLE_REFERENCE_FVES:-${FIDELITY_REFERENCE_FVES:-token=1.0,temporal=1.0,mean=1.0}}"
LOCAL_TRAINING_CACHE_DIR="${LOCAL_TRAINING_CACHE_DIR:-${TMPDIR:-${REPO_ROOT}/.cache}/chunk-saes-cache-${MODEL_NAME}-${PROTOCOL_VERSION}}"
CHECKPOINT_STAGING_DIR="${CHECKPOINT_STAGING_DIR:-${TMPDIR:-${REPO_ROOT}/.cache}/chunk-saes-checkpoints-${MODEL_NAME}-${PROTOCOL_VERSION}}"
RESUME="${RESUME:-0}"
OVERWRITE_CACHES="${OVERWRITE_CACHES:-0}"
OVERWRITE_TRAINING="${OVERWRITE_TRAINING:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
LOADER_PREFETCH_SHARDS="${LOADER_PREFETCH_SHARDS:-16}"
LOADER_PREFETCH_WORKERS="${LOADER_PREFETCH_WORKERS:-4}"
LOADER_PREFETCH_BATCHES="${LOADER_PREFETCH_BATCHES:-8}"
LOADER_GPU_SHARDS="${LOADER_GPU_SHARDS:-16}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
DECODER_BACKEND="${DECODER_BACKEND:-sparse}"
SAE_PARALLELISM="${SAE_PARALLELISM:-ddp}"
AUXK_ALPHA="${AUXK_ALPHA:-0.0625}"
AUXK_ACTIVATION_AGE="${AUXK_ACTIVATION_AGE:-5000000}"
AUXK="${AUXK:-512}"
AUXK_CANDIDATE_FEATURES="${AUXK_CANDIDATE_FEATURES:-4096}"
TEMPORAL_HIGH_FRACTION="${TEMPORAL_HIGH_FRACTION:-0.2}"
TEMPORAL_HIGH_RECONSTRUCTION_WEIGHT="${TEMPORAL_HIGH_RECONSTRUCTION_WEIGHT:-0.2}"
TEMPORAL_FULL_RECONSTRUCTION_WEIGHT="${TEMPORAL_FULL_RECONSTRUCTION_WEIGHT:-0.8}"
TEMPORAL_ALPHA="${TEMPORAL_ALPHA:-1.0}"
TEMPORAL_TEMPERATURE="${TEMPORAL_TEMPERATURE:-0.1}"
TEMPORAL_CONTRASTIVE_BLOCK_SIZE="${TEMPORAL_CONTRASTIVE_BLOCK_SIZE:-1024}"

mkdir -p \
  "${PIPELINE_ROOT}/data" \
  "${PIPELINE_ROOT}/logs" \
  "${TENSORBOARD_DIR}"
PIPELINE_TIMING_LOG="${PIPELINE_TIMING_LOG:-${PIPELINE_ROOT}/logs/pipeline_timings.tsv}"
if [[ ! -f "${PIPELINE_TIMING_LOG}" ]]; then
  printf 'started_at\telapsed_seconds\tstatus\tcommand\n' >"${PIPELINE_TIMING_LOG}"
fi

run() {
  printf '[streaming] +'
  printf ' %q' "$@"
  printf '\n'
  local started elapsed status
  started="$(date --iso-8601=seconds)"
  local start_epoch
  start_epoch="$(date +%s)"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  elapsed=$(( $(date +%s) - start_epoch ))
  {
    printf '%s\t%s\t%s\t' "${started}" "${elapsed}" "${status}"
    printf '%q ' "$@"
    printf '\n'
  } >>"${PIPELINE_TIMING_LOG}"
  return "${status}"
}

stage_enabled() {
  local wanted="$1"
  [[ "${PIPELINE_STAGE}" == "all" || "${PIPELINE_STAGE}" == "${wanted}" ]]
}

verify_hf_stream() {
  run "${PY}" - \
    "${HF_DATASET}" \
    "${HF_DATASET_REVISION}" \
    "${HF_DATASET_PHYSICAL_SPLIT}" \
    "${SPLIT_SEED}" \
    "${DOCUMENT_SHUFFLE_SEED}" \
    "${STREAM_SHUFFLE_BUFFER}" <<'PY'
import json
import os
import sys
from chunk_saes.hf_corpus import HFStreamingCorpus, verify_hf_stream

spec = HFStreamingCorpus(
    dataset=sys.argv[1],
    revision=sys.argv[2],
    physical_split=sys.argv[3],
    logical_split="train",
    split_seed=int(sys.argv[4]),
    shuffle_seed=int(sys.argv[5]),
    shuffle_buffer=int(sys.argv[6]),
)
print(json.dumps(verify_hf_stream(spec), sort_keys=True), flush=True)
os._exit(0)
PY
}

extract_one_cache() {
  local logical_split="$1"
  local cache="$2"
  local target="$3"
  local seed="$4"
  local -a args=(
    "${TORCHRUN[@]}" --standalone --nproc_per_node "${GPU_COUNT}" --max_restarts 0
    "${SCRIPT_DIR}/extract_streaming_chunk_activations.py"
    --model "${MODEL}"
    --hf-dataset "${HF_DATASET}"
    --hf-dataset-revision "${HF_DATASET_REVISION}"
    --hf-physical-split "${HF_DATASET_PHYSICAL_SPLIT}"
    --logical-split "${logical_split}"
    --split-seed "${SPLIT_SEED}"
    --document-shuffle-seed "${DOCUMENT_SHUFFLE_SEED}"
    --stream-shuffle-buffer "${STREAM_SHUFFLE_BUFFER}"
    --corpus-source "${CORPUS_SOURCE}"
    --cache-dir "${cache}"
    --layer "${LAYER}"
    --target-tokens "${target}"
    --chunk-lengths "${CHUNK_LENGTHS}"
    --sample-seed "${seed}"
    --max-document-reuses "${MAX_DOCUMENT_REUSES}"
    --tokenizer-threads "${STREAM_TOKENIZER_THREADS}"
    --tokenizer-batch-size "${STREAM_TOKENIZER_BATCH_SIZE}"
    --tokenizer-batch-chars "${STREAM_TOKENIZER_BATCH_CHARS}"
    --forward-batch-size "${FORWARD_BATCH_SIZE}"
    --forward-token-budget "${FORWARD_TOKEN_BUDGET}"
    --writer-batch-tokens "${WRITER_BATCH_TOKENS}"
    --forward-scheduler length
    --planner-mode "${STREAM_PLANNER_MODE}"
    --dispatch-queue-depth "${STREAM_DISPATCH_QUEUE_DEPTH}"
    --scheduler-window-pairs "${STREAM_SCHEDULER_WINDOW_PAIRS}"
    --async-write-batches "${ASYNC_WRITE_BATCHES}"
    --shard-write-workers "${SHARD_WRITE_WORKERS}"
    --max-pending-shards "${MAX_PENDING_SHARDS}"
    --cache-shard-tokens "${CACHE_SHARD_TOKENS}"
    --model-dtype "${MODEL_DTYPE}"
    --activation-dtype "${ACTIVATION_DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
  )
  [[ "${OVERWRITE_CACHES}" == "1" ]] && args+=(--overwrite)
  run "${args[@]}"
}

mirror_cache() {
  local source="$1"
  local destination="$2"
  local source_path destination_path
  source_path="$(readlink -m -- "${source}")"
  destination_path="$(readlink -m -- "${destination}")"
  if [[ "${source_path}" == "${destination_path}" ]]; then
    "${PY}" - "${source}" <<'PY'
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
manifest_path = cache / "manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(f"activation cache manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("complete") is not True:
    raise ValueError(f"activation cache is incomplete: {cache}")
print(f"[streaming] cache already uses persistent path: {cache}", flush=True)
PY
    return
  fi
  local -a args=(
    "${PY}" "${SCRIPT_DIR}/mirror_activation_cache.py"
    --source "${source}" \
    --destination "${destination}" \
    --workers "${CACHE_MIRROR_WORKERS}"
  )
  [[ "${CACHE_MIRROR_VERIFY_DESTINATION}" == "1" ]] &&
    args+=(--verify-destination)
  run "${args[@]}"
}

train_saes() {
  local lock_file="${PIPELINE_ROOT}/logs/${RUN_NAME}.train.lock"
  local gpu_lock_id
  gpu_lock_id="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -c '[:alnum:]' '_')"
  gpu_lock_id="${gpu_lock_id%_}"
  local gpu_lock_file="${TMPDIR:-${REPO_ROOT}/.cache}/chunk-saes-gpus-${gpu_lock_id}.train.lock"
  mkdir -p "$(dirname "${gpu_lock_file}")"
  exec {train_lock_fd}>"${lock_file}"
  if ! flock -n "${train_lock_fd}"; then
    echo "another training process already owns ${lock_file}" >&2
    return 1
  fi
  exec {gpu_lock_fd}>"${gpu_lock_file}"
  if ! flock -n "${gpu_lock_fd}"; then
    echo "another training process already owns GPU set ${CUDA_VISIBLE_DEVICES}" >&2
    flock -u "${train_lock_fd}"
    exec {train_lock_fd}>&-
    return 1
  fi
  local -a args=(
    "${TORCHRUN[@]}" --standalone --nproc_per_node "${GPU_COUNT}" --max_restarts 0
    "${SCRIPT_DIR}/train_chunk_saes.py"
    --activation-cache-dir "${TRAIN_CACHE_DIR}"
    --validation-cache-dir "${VALIDATION_CACHE_DIR}"
    --output-dir "${SAE_ROOT}"
    --model "${MODEL}"
    --layer "${LAYER}"
    --modes "${SAE_MODES}"
    --joint-chunk-alpha "${JOINT_CHUNK_ALPHA}"
    --joint-cross-prefix "${JOINT_CROSS_PREFIX}"
    --direction-policy "${DIRECTION_POLICY}"
    --dict-size "${SAE_WIDTH}"
    --k "${K}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --steps "${TRAIN_STEPS}"
    --lr "${LR}"
    --warmup-steps "${WARMUP_STEPS}"
    --min-lr-ratio "${MIN_LR_RATIO}"
    --threshold-beta 0.99
    --gradient-clip 1.0
    --seed "${TRAIN_SEED}"
    --autocast-dtype bfloat16
    --loader-prefetch-shards "${LOADER_PREFETCH_SHARDS}"
    --loader-prefetch-workers "${LOADER_PREFETCH_WORKERS}"
    --loader-prefetch-batches "${LOADER_PREFETCH_BATCHES}"
    --loader-gpu-shards "${LOADER_GPU_SHARDS}"
    --local-cache-dir "${LOCAL_TRAINING_CACHE_DIR}"
    --validation-cache-max-bytes 1073741824
    --batch-topk-candidate-multiplier 1.25
    --max-activation-norm-multiple 10.0
    --dead-feature-threshold 10000000
    --auxk-alpha "${AUXK_ALPHA}"
    --auxk-activation-age "${AUXK_ACTIVATION_AGE}"
    --auxk "${AUXK}"
    --auxk-candidate-features "${AUXK_CANDIDATE_FEATURES}"
    --temporal-high-fraction "${TEMPORAL_HIGH_FRACTION}"
    --temporal-high-reconstruction-weight "${TEMPORAL_HIGH_RECONSTRUCTION_WEIGHT}"
    --temporal-full-reconstruction-weight "${TEMPORAL_FULL_RECONSTRUCTION_WEIGHT}"
    --temporal-alpha "${TEMPORAL_ALPHA}"
    --temporal-temperature "${TEMPORAL_TEMPERATURE}"
    --temporal-contrastive-block-size "${TEMPORAL_CONTRASTIVE_BLOCK_SIZE}"
    --normalization-samples 65536
    --log-every 10
    --validate-every "${VALIDATE_EVERY}"
    --validation-samples 262144
    --final-validation-samples 0
    --validation-batch-size 0
    --save-every "${SAVE_EVERY}"
    --best-metric nmse
    --run-name "${RUN_NAME}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --tensorboard-flush-secs 30
    --tensorboard-max-queue 100
    --attainable-reference-fves "${ATTAINABLE_REFERENCE_FVES}"
    --checkpoint-staging-dir "${CHECKPOINT_STAGING_DIR}"
    --require-exact-coverage
    --normalize-activations
    --no-batch-topk-bf16-histogram
    --decoder-backend "${DECODER_BACKEND}"
    --parallelism "${SAE_PARALLELISM}"
    --fused-adam
    --deduplicate-chunk-inputs
    --joint-modes
    --loader-materialize-shards
    --loader-pin-memory
    --cache-periodic-validation
    --defer-best-checkpoint
    --async-latest-checkpoint
    --bind-cpu-affinity
  )
  [[ "${RESUME}" == "1" ]] && args+=(--resume)
  [[ "${OVERWRITE_TRAINING}" == "1" ]] && args+=(--overwrite-output)
  local status=0
  run "${args[@]}" || status=$?
  flock -u "${gpu_lock_fd}"
  exec {gpu_lock_fd}>&-
  flock -u "${train_lock_fd}"
  exec {train_lock_fd}>&-
  return "${status}"
}

TRAIN_MIRROR_PID=""
VALIDATION_MIRROR_PID=""
DEFER_TRAIN_MIRROR=0
cleanup() {
  local status=$?
  set +e
  [[ -z "${TRAIN_MIRROR_PID}" ]] || wait "${TRAIN_MIRROR_PID}"
  [[ -z "${VALIDATION_MIRROR_PID}" ]] || wait "${VALIDATION_MIRROR_PID}"
  trap - EXIT
  exit "${status}"
}
trap cleanup EXIT

case "${PIPELINE_STAGE}" in
  all|corpus|cache|train|eval) ;;
  *)
    echo "PIPELINE_STAGE must be all, corpus, cache, train, or eval" >&2
    exit 2
    ;;
esac

if stage_enabled corpus || [[ "${PIPELINE_STAGE}" == "all" ]]; then
  verify_hf_stream
fi

if stage_enabled cache || [[ "${PIPELINE_STAGE}" == "all" ]]; then
  extract_one_cache \
    "train" \
    "${TRAIN_CACHE_DIR}" \
    "${TARGET_TRAIN_TOKENS}" \
    "${TRAIN_SAMPLE_SEED}"
  case "${CACHE_MIRROR_POLICY}" in
    background)
      mirror_cache "${TRAIN_CACHE_DIR}" "${PERSISTENT_TRAIN_CACHE_DIR}" &
      TRAIN_MIRROR_PID=$!
      ;;
    after_training)
      DEFER_TRAIN_MIRROR=1
      ;;
    disabled) ;;
    *)
      echo "CACHE_MIRROR_POLICY must be background, after_training, or disabled" >&2
      exit 2
      ;;
  esac
  extract_one_cache \
    "validation" \
    "${VALIDATION_CACHE_DIR}" \
    "${TARGET_VALIDATION_TOKENS}" \
    "${VALIDATION_SAMPLE_SEED}"
  if [[ "${CACHE_MIRROR_POLICY}" != "disabled" ]]; then
    mirror_cache "${VALIDATION_CACHE_DIR}" "${PERSISTENT_VALIDATION_CACHE_DIR}" &
    VALIDATION_MIRROR_PID=$!
  fi
fi

if stage_enabled train || [[ "${PIPELINE_STAGE}" == "all" ]]; then
  train_saes
fi

if [[ "${CACHE_MIRROR_POLICY}" == "after_training" ]] &&
  [[ "${DEFER_TRAIN_MIRROR}" == "1" || "${PIPELINE_STAGE}" == "train" ]]; then
  mirror_cache "${TRAIN_CACHE_DIR}" "${PERSISTENT_TRAIN_CACHE_DIR}" &
  TRAIN_MIRROR_PID=$!
fi

if [[ "${RUN_EVALUATION}" == "1" ]] &&
  { stage_enabled eval || [[ "${PIPELINE_STAGE}" == "all" ]]; }; then
  run env \
    MODEL="${MODEL}" \
    MODEL_NAME="${MODEL_NAME}" \
    NUM_LAYERS="${NUM_LAYERS}" \
    LAYER="${LAYER}" \
    EVIDENCE_FORWARD_BATCH_SIZE="${EVIDENCE_FORWARD_BATCH_SIZE}" \
    PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE}" \
    PROTOCOL_VERSION="${PROTOCOL_VERSION}" \
    PIPELINE_ROOT="${PIPELINE_ROOT}" \
    SAE_ROOT="${SAE_ROOT}" \
    VALIDATION_CACHE_DIR="${PERSISTENT_VALIDATION_CACHE_DIR}" \
    TARGET_VALIDATION_TOKENS="${TARGET_VALIDATION_TOKENS}" \
    HF_DATASET="${HF_DATASET}" \
    HF_DATASET_REVISION="${HF_DATASET_REVISION}" \
    HF_DATASET_PHYSICAL_SPLIT="${HF_DATASET_PHYSICAL_SPLIT}" \
    SPLIT_SEED="${SPLIT_SEED}" \
    DOCUMENT_ORDER_SEED="${DOCUMENT_SHUFFLE_SEED}" \
    STREAM_SHUFFLE_BUFFER="${STREAM_SHUFFLE_BUFFER}" \
    CORPUS_SOURCE="${CORPUS_SOURCE}" \
    ATTAINABLE_REFERENCE_FVES="${ATTAINABLE_REFERENCE_FVES}" \
    CROSS_REFERENCE_MANIFEST="${CROSS_REFERENCE_MANIFEST:-}" \
    ALLOW_AUDIT_CROSS_SCALAR="${ALLOW_AUDIT_CROSS_SCALAR:-0}" \
    bash "${SCRIPT_DIR}/eval_chunk_saes_common.sh"
fi

[[ -z "${TRAIN_MIRROR_PID}" ]] || wait "${TRAIN_MIRROR_PID}"
TRAIN_MIRROR_PID=""
[[ -z "${VALIDATION_MIRROR_PID}" ]] || wait "${VALIDATION_MIRROR_PID}"
VALIDATION_MIRROR_PID=""
trap - EXIT

echo "[streaming] pipeline complete: ${PIPELINE_ROOT}"
