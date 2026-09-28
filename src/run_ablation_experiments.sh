#!/usr/bin/env bash
# Reproducible E1-E4 runner for ablation.md.
#
# The script is intentionally resumable: a complete.json artifact is treated
# as finished, while an incomplete directory is resumed when the underlying
# trainer supports it.  Each stage writes a status line before and after every
# seed so progress can be tailed without attaching to torchrun.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Qwen3.5-9B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen35-9b-base}"
LAYER="${LAYER:-21}"
GPU_COUNT="${GPU_COUNT:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

PIPELINE_ROOT="${ROOT}/qwen/${MODEL_NAME}/pile_proportional_exact_occurrence"
TRAIN_CACHE="${PIPELINE_ROOT}/data/layer21_train_cache_exact1000000000"
VAL_CACHE="${PIPELINE_ROOT}/data/layer21_validation_cache_exact10000128"
ABLATION_ROOT="${PIPELINE_ROOT}/ablation"
LOG_ROOT="${ABLATION_ROOT}/logs"
STATUS_FILE="${ABLATION_ROOT}/status.tsv"
mkdir -p "${ABLATION_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${TRAIN_CACHE}/manifest.json" || ! -f "${VAL_CACHE}/manifest.json" ]]; then
  echo "activation caches are required: ${TRAIN_CACHE} and ${VAL_CACHE}" >&2
  exit 2
fi

timestamp() { date --iso-8601=seconds; }
status() {
  local id="$1" stage="$2" seed="$3" state="$4" detail="${5:-}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "$id" "$stage" "$seed" "$state${detail:+ $detail}" >>"${STATUS_FILE}"
  printf '[ablation] %s id=%s stage=%s seed=%s %s\n' "$(timestamp)" "$id" "$stage" "$seed" "$state${detail:+ $detail}" | tee -a "${LOG_ROOT}/progress.log"
}

run_cross() {
  local id="$1" policy="$2" stage="$3" seed="$4" steps="$5" exact="$6"
  local out="${ABLATION_ROOT}/${id}/${stage}/seed${seed}"
  # Keep the launcher log outside the trainer output directory: the trainer
  # removes that directory when --overwrite-output is used.
  local log="${LOG_ROOT}/${id}_${stage}_seed${seed}.log"
  mkdir -p "${out}"
  if [[ -f "${out}/cross/complete.json" ]]; then
    status "$id" "$stage" "$seed" SKIP "complete"
    return 0
  fi
  status "$id" "$stage" "$seed" START "steps=${steps}"
  local -a args=(
    "${PY}" -m torch.distributed.run --standalone --nproc-per-node "${GPU_COUNT}" --max_restarts 0
    "${ROOT}/src/train_chunk_saes.py"
    --activation-cache-dir "${TRAIN_CACHE}"
    --validation-cache-dir "${VAL_CACHE}"
    --output-dir "${out}"
    --model "${MODEL}"
    --layer "${LAYER}"
    --modes cross
    --direction-policy "${policy}"
    --target-granularity mean
    --loss-mask partner-only
    --mask-representation norm-matched-mean
    --dict-size 65536 --k 128 --global-batch-size 32000
    --steps "${steps}" --seed "${seed}"
    --validation-samples 262144 --final-validation-samples 0
    --validate-every 250 --log-every 50 --save-every 1000
    --attainable-reference-fves cross=0.6272699794963396
    --run-name "${id}_${stage}_seed${seed}"
    --loader-prefetch-shards 16 --loader-prefetch-workers 4 --loader-prefetch-batches 8
    --loader-gpu-shards 2 --decoder-backend sparse
    --tensorboard-dir "${PIPELINE_ROOT}/ablation/tensorboard"
  )
  if [[ "${exact}" == "1" ]]; then
    args+=(--require-exact-coverage)
  else
    args+=(--no-joint-modes)
  fi
  if [[ -f "${out}/cross/checkpoints/latest/checkpoint_manifest.json" && "${exact}" == "1" ]]; then
    args+=(--resume)
  else
    args+=(--overwrite-output)
  fi
  if "${args[@]}" >"${log}" 2>&1; then
    status "$id" "$stage" "$seed" DONE "$(tail -n 1 "${log}" | tr '\t' ' ')"
  else
    status "$id" "$stage" "$seed" FAIL "see ${log}"
    return 1
  fi
}

run_sequence() {
  local id="$1" loss_mask="$2" stage="$3" seed="$4" target="$5" val_target="$6"
  local out="${ABLATION_ROOT}/${id}/${stage}/seed${seed}"
  # Keep the launcher log outside the trainer output directory: the trainer
  # removes that directory when --overwrite-output is used.
  local log="${LOG_ROOT}/${id}_${stage}_seed${seed}.log"
  mkdir -p "${out}"
  if [[ -f "${out}/complete.json" ]]; then
    status "$id" "$stage" "$seed" SKIP "complete"
    return 0
  fi
  status "$id" "$stage" "$seed" START "target=${target}"
  local -a args=(
    "${PY}" -m torch.distributed.run --standalone --nproc-per-node "${GPU_COUNT}" --max_restarts 0
    "${ROOT}/src/train_sequence_cross_ablation.py"
    --activation-cache-dir "${TRAIN_CACHE}"
    --validation-cache-dir "${VAL_CACHE}"
    --output-dir "${out}"
    --model "${MODEL}" --layer "${LAYER}"
    --dict-size 65536 --k 128 --global-batch-size 32000
    --target-occurrences "${target}" --validation-occurrences "${val_target}"
    --periodic-validation-occurrences 262144
    --examples-per-batch 20 --max-chunk-length 512
    --context-dim 256 --context-heads 8
    --loss-mask "${loss_mask}" --mask-representation norm-matched-mean
    --seed "${seed}" --log-every 50 --validate-every 250 --save-every 1000
    --decoder-backend sparse --final-validation
    --overwrite-output
  )
  if "${args[@]}" >"${log}" 2>&1; then
    status "$id" "$stage" "$seed" DONE "$(tail -n 1 "${log}" | tr '\t' ' ')"
  else
    status "$id" "$stage" "$seed" FAIL "see ${log}"
    return 1
  fi
}

run_direction_family() {
  local id="$1" policy="$2"
  local -a pilot=(42 43) confirm=(42 43 44) full=(42 43)
  local seed
  for seed in "${pilot[@]}"; do run_cross "$id" "$policy" pilot "$seed" 313 0; done
  for seed in "${confirm[@]}"; do run_cross "$id" "$policy" confirm "$seed" 3125 0; done
  for seed in "${full[@]}"; do run_cross "$id" "$policy" full "$seed" 31250 1; done
}

run_sequence_family() {
  local id="$1" loss="$2"
  local -a pilot=(42 43) confirm=(42 43 44) full=(42 43)
  local seed
  for seed in "${pilot[@]}"; do run_sequence "$id" "$loss" pilot "$seed" 10016000 262144; done
  for seed in "${confirm[@]}"; do run_sequence "$id" "$loss" confirm "$seed" 100000000 262144; done
  for seed in "${full[@]}"; do run_sequence "$id" "$loss" full "$seed" 1000000000 10000128; done
}

run_sequence_ddp_smoke() {
  local out="${ABLATION_ROOT}/_ddp_smoke_balanced_sequence_v2"
  local log="${out}/run.log"
  mkdir -p "${out}"
  status SMOKE sequence ddp START "balanced E4 target=1000"
  if "${PY}" -m torch.distributed.run --standalone --nproc-per-node "${GPU_COUNT}" --max_restarts 0 \
    "${ROOT}/src/train_sequence_cross_ablation.py" \
    --activation-cache-dir "${TRAIN_CACHE}" --validation-cache-dir "${VAL_CACHE}" \
    --output-dir "${out}" --model "${MODEL}" --layer "${LAYER}" \
    --dict-size 128 --k 4 --global-batch-size 256 \
    --target-occurrences 1000 --validation-occurrences 1000 \
    --periodic-validation-occurrences 1000 --examples-per-batch 4 \
    --max-chunk-length 512 --context-dim 32 --context-heads 8 \
    --loss-mask all --mask-representation norm-matched-mean \
    --seed 20260819 --log-every 1 --validate-every 1 --save-every 0 \
    --decoder-backend dense --final-validation --overwrite-output >"${log}" 2>&1 \
    && "${PY}" -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["validation_metrics"]; assert d["world_size"] == 8 and d["target_coverage"] == 1.0; assert abs(m["partner_effective_l0"] - 4.0) < 1e-6 and abs(m["self_effective_l0"] - 4.0) < 1e-6' "${out}/complete.json"; then
    status SMOKE sequence ddp DONE "$(tail -n 1 "${log}" | tr '\t' ' ')"
  else
    status SMOKE sequence ddp FAIL "see ${log}"
    return 1
  fi
}

main() {
  touch "${STATUS_FILE}"
  status E0 baseline baseline INFO "reuse=${PIPELINE_ROOT}/checkpoints/*/cross"
  run_direction_family E1 a-to-b
  run_direction_family E2 b-to-a
  run_sequence_ddp_smoke
  run_sequence_family E3 partner-only
  run_sequence_family E4 all
  "${PY}" "${ROOT}/src/evaluate_ablation_metrics.py" \
    --ablation-root "${ABLATION_ROOT}" \
    --train-cache-dir "${TRAIN_CACHE}" \
    --validation-cache-dir "${VAL_CACHE}" \
    --feature-dir "${PIPELINE_ROOT}/eval/layer21_w65536_k128_1b/document_linking" \
    --output "${ABLATION_ROOT}/invariant_evaluations.json" \
    --sample-pairs 2048 --feature-sample-size 8192 --bootstrap-samples 10000 \
    --sequence-sample-pairs 512 \
    --device auto
  "${PY}" "${ROOT}/src/summarize_ablation_results.py" \
    --ablation-root "${ABLATION_ROOT}" \
    --output "${ABLATION_ROOT}/results.md"
  status ALL all all DONE "training families complete"
}

main "$@"
