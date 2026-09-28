#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PY:-${ROOT}/.venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/eval/layer21_w65536_k128_1b}"
MODEL="${EVAL8_MODEL:-${ROOT}/models/Qwen3.5-9B-Base}"
SOURCE_DATA_DIR="${EVAL8_SOURCE_DATA_DIR:-${EVAL_ROOT}/shared/uniform_feature_activation_cache_exact1000}"
JOINT_CACHE_DIR="${EVAL8_JOINT_CACHE_DIR:-${EVAL_ROOT}/shared/native_high_level_joint_cache_exact1000}"
CHECKPOINT_ROOT="${ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/checkpoints"
JOINT_ALPHA0P25_ROOT="${EVAL8_JOINT_ALPHA0P25_ROOT:-${CHECKPOINT_ROOT}/qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_joint_chunk_nested_h32768_alpha0p25}"
JOINT_ALPHA0P5_ROOT="${EVAL8_JOINT_ALPHA0P5_ROOT:-${CHECKPOINT_ROOT}/qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_joint_chunk_nested_h32768_alpha0p5}"
JOINT_ALPHA1_ROOT="${EVAL8_JOINT_ALPHA1_ROOT:-${CHECKPOINT_ROOT}/qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_joint_chunk_nested_h32768_alpha1}"
JOINT_ALPHA1P5_ROOT="${EVAL8_JOINT_ALPHA1P5_ROOT:-${CHECKPOINT_ROOT}/qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_joint_chunk_nested_h32768_alpha1p5}"

exec "${PY}" \
  "${ROOT}/src/evals/semantic_invariance/materialize_joint_semantic_invariance_cache.py" \
  --source-data-dir "${SOURCE_DATA_DIR}" \
  --output-dir "${JOINT_CACHE_DIR}" \
  --model "${MODEL}" \
  --layer "${EVAL8_LAYER:-21}" \
  --joint-alpha0p25-root "${JOINT_ALPHA0P25_ROOT}" \
  --joint-alpha0p5-root "${JOINT_ALPHA0P5_ROOT}" \
  --joint-alpha1-root "${JOINT_ALPHA1_ROOT}" \
  --joint-alpha1p5-root "${JOINT_ALPHA1P5_ROOT}" \
  --devices "${EVAL8_JOINT_CACHE_DEVICES:-0,1,2,3,4,5,6,7}" \
  --max-batch-size "${EVAL8_JOINT_CACHE_BATCH_SIZE:-2048}" \
  --forward-token-budget "${EVAL8_JOINT_CACHE_TOKEN_BUDGET:-98304}" \
  "$@"
