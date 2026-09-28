#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE="${EVAL9_ROOT:-${ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/eval/layer21_w65536_k128_1b/code_page_retrieval}"
MODEL="${MODEL:-${ROOT}/models/Qwen3.5-9B-Base}"
SAE_ROOT="${SAE_ROOT:-${ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/checkpoints/qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_pile_proportional_exact_occurrence_saeimpl_batchtopk_separate_centers_preemptive_auxk_lr_floor_v3_latest}"
PY="${PY:-python}"
SPARK_SUBMIT="${SPARK_SUBMIT:-spark-submit}"
CODE_PAGE_INPUT="${CODE_PAGE_INPUT:-${ROOT}/data/code_page_v5}"
STAGE="${1:-all}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${BASE}/data" "${BASE}/demo" "${BASE}/logs"

run_sample() {
  if [[ -n "${SPARK_ENV_FILE:-}" ]]; then
    source "${SPARK_ENV_FILE}" >/dev/null
  fi
  "${SPARK_SUBMIT}" --master yarn --deploy-mode client \
    --conf spark.app.name=eval9-code-page-random-10k \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.executor.instances="${SPARK_EXECUTORS:-160}" \
    --conf spark.executor.cores=1 \
    --conf spark.executor.memory=4g \
    --conf spark.driver.memory=16g \
    --conf spark.driver.maxResultSize=4g \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.speculation=false \
    --conf spark.eventLog.enabled=false \
    "${ROOT}/src/evals/code_page/sample_code_page_spark.py" \
    --input-path "${CODE_PAGE_INPUT}" \
    --output-dir "${BASE}/data" --sample-fraction 0.001 --write-jsonl
}

run_profile() {
  "${PY}" "${ROOT}/src/evals/code_page/profile_code_page_sample.py" \
    --input-parquet "${BASE}/data/code_page_hqs8_9_10k.parquet" \
    --model "${MODEL}" --output-dir "${BASE}/data" --overwrite
}

run_features() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  set +e
  "${PY}" -m torch.distributed.run --standalone --nproc-per-node="${GPU_COUNT:-8}" --max-restarts=0 \
    "${ROOT}/src/evals/code_page/extract_code_page_sae_features.py" \
    --input-parquet "${BASE}/data/code_page_hqs8_9_10k.parquet" \
    --seeds-jsonl "${BASE}/demo/seeds.jsonl" \
    --model "${MODEL}" --layer 21 --sae-root "${SAE_ROOT}" \
    --checkpoint-selection best --output-dir "${BASE}/data" \
    --batch-size 8 --documents-per-shard 64 --length-bucket-size 64 \
    --model-dtype bfloat16 --attn-implementation sdpa
  local status=$?
  set -e
  return "${status}"
}

run_demo() {
  "${PY}" "${ROOT}/src/evals/code_page/build_code_page_retrieval_demo.py" \
    --parquet "${BASE}/data/code_page_hqs8_9_10k.parquet" \
    --merged-features "${BASE}/data/features/merged_features.npz" \
    --seed-features "${BASE}/data/features/seed_features.npz" \
    --queries "${BASE}/demo/queries.locked.json" \
    --seeds-jsonl "${BASE}/demo/seeds.jsonl" \
    --output-dir "${BASE}/demo"

  # Publish one self-contained, clearly named interface and remove the old
  # split index/pages HTML files so relative-path 404s cannot reappear.
  "${PY}" "${ROOT}/src/evals/code_page/build_eval9_review_dashboard.py" \
    --base "${BASE}" --top-k 20

  mkdir -p "${BASE}/figures"
  "${PY}" "${ROOT}/src/evals/code_page/plot_core.py" \
    --summary "${BASE}/retrieval_summary.json" \
    --output "${BASE}/figures/code_page_retrieval.png"
}

case "${STAGE}" in
  sample) run_sample ;;
  profile) run_profile ;;
  features) run_features ;;
  demo|retrieve) run_demo ;;
  all) run_sample; run_profile; run_features; run_demo ;;
  *) echo "usage: $0 [sample|profile|features|retrieve|demo|all]" >&2; exit 2 ;;
esac
