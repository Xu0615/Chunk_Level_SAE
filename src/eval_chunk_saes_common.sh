#!/usr/bin/env bash
# Shared fail-closed Chunk-SAE evaluation pipeline.

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${COMMON_DIR}/.." && pwd)}"
SRC_DIR="${REPO_ROOT}/src"
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"
TORCHRUN=("${PY}" -m torch.distributed.run)

: "${MODEL:?MODEL must be set by the model-specific launcher}"
: "${MODEL_NAME:?MODEL_NAME must be set by the model-specific launcher}"
: "${NUM_LAYERS:?NUM_LAYERS must be set by the model-specific launcher}"
: "${LAYER:?LAYER must be set by the model-specific launcher}"

MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SAE_WIDTH="${SAE_WIDTH:-65536}"
K="${K:-128}"
TARGET_TRAIN_TOKENS="${TARGET_TRAIN_TOKENS:-1000000000}"
TARGET_VALIDATION_TOKENS="${TARGET_VALIDATION_TOKENS:-10000128}"
TRAIN_SEED="${TRAIN_SEED:-42}"
PROTOCOL_VERSION="${PROTOCOL_VERSION:-pile_proportional_exact_occurrence}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/qwen/${MODEL_NAME}}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${OUTPUT_ROOT}/${PROTOCOL_VERSION}}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_layer${LAYER}_w${SAE_WIDTH}_k${K}_train${TARGET_TRAIN_TOKENS}_seed${TRAIN_SEED}_${PROTOCOL_VERSION}}"
SAE_ROOT="${SAE_ROOT:-${PIPELINE_ROOT}/checkpoints/${RUN_NAME}}"
EVAL_CHECKPOINT_SELECTION="${EVAL_CHECKPOINT_SELECTION:-best}"
EVAL_ROOT="${EVAL_ROOT:-${PIPELINE_ROOT}/eval/layer${LAYER}_w${SAE_WIDTH}_k${K}_1b}"
LOG_DIR="${LOG_DIR:-${PIPELINE_ROOT}/logs}"
VALIDATION_CACHE_DIR="${VALIDATION_CACHE_DIR:-${PIPELINE_ROOT}/data/layer${LAYER}_validation_cache_exact${TARGET_VALIDATION_TOKENS}}"
CROSS_REFERENCE_MANIFEST="${CROSS_REFERENCE_MANIFEST:-}"
JOINT_SAE_ROOT="${JOINT_SAE_ROOT:-}"
ALLOW_AUDIT_CROSS_SCALAR="${ALLOW_AUDIT_CROSS_SCALAR:-0}"

HF_DATASET="${HF_DATASET:-EleutherAI/the_pile_deduplicated}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb}"
HF_DATASET_PHYSICAL_SPLIT="${HF_DATASET_PHYSICAL_SPLIT:-train}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DOCUMENT_ORDER_SEED="${DOCUMENT_ORDER_SEED:-71}"
STREAM_SHUFFLE_BUFFER="${STREAM_SHUFFLE_BUFFER:-16384}"
CORPUS_SOURCE="${CORPUS_SOURCE:-Pile-Deduplicated}"

CHUNK_LENGTHS="${CHUNK_LENGTHS:-32,64,128,256,512}"
EVIDENCE_SEED="${EVIDENCE_SEED:-52}"
EVIDENCE_CHUNKS="${EVIDENCE_CHUNKS:-20000}"
FEATURE_SAMPLE_SIZE="${FEATURE_SAMPLE_SIZE:-1000}"
TOP_ACTIVATING_CHUNKS="${TOP_ACTIVATING_CHUNKS:-20}"
EVIDENCE_CHUNKS_PER_DOCUMENT="${EVIDENCE_CHUNKS_PER_DOCUMENT:-8}"
EVIDENCE_MAX_PASSES="${EVIDENCE_MAX_PASSES:-100}"
EVIDENCE_FORWARD_BATCH_SIZE="${EVIDENCE_FORWARD_BATCH_SIZE:-8}"

BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
HIGH_LEVEL_ANALYSIS_SEED="${HIGH_LEVEL_ANALYSIS_SEED:-62}"

ARXIV_DATASET="${ARXIV_DATASET:-librarian-bots/arxiv-metadata-snapshot}"
OOD_YEAR="${OOD_YEAR:-2023}"
PROBE_TRAIN_PER_CLASS="${PROBE_TRAIN_PER_CLASS:-1024}"
PROBE_VALIDATION_PER_CLASS="${PROBE_VALIDATION_PER_CLASS:-256}"
PROBE_TEST_PER_CLASS="${PROBE_TEST_PER_CLASS:-256}"
PROBE_OOD_PER_CLASS="${PROBE_OOD_PER_CLASS:-256}"
PROBE_SEED="${PROBE_SEED:-72}"
PROBE_MAX_LENGTH="${PROBE_MAX_LENGTH:-512}"
PROBE_CHUNK_LENGTH="${PROBE_CHUNK_LENGTH:-128}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-8}"
SPARSE_STORAGE_K="${SPARSE_STORAGE_K:-2048}"
TOKEN_CANDIDATE_K="${TOKEN_CANDIDATE_K:-2048}"
TOKEN_MATCH_REFERENCE="${TOKEN_MATCH_REFERENCE:-cross}"
TOKEN_MATCH_K="${TOKEN_MATCH_K:-0}"
FEATURE_BUDGETS="${FEATURE_BUDGETS:-16,64,256}"
LOW_LABEL_BUDGETS="${LOW_LABEL_BUDGETS:-1,2,4,8,16,64,256}"
LOW_LABEL_SEEDS="${LOW_LABEL_SEEDS:-0,1,2,3,4}"
CLASSIFIER_C="${CLASSIFIER_C:-1.0}"
CLASSIFIER_MAX_ITER="${CLASSIFIER_MAX_ITER:-2000}"
CLASSIFIER_N_JOBS="${CLASSIFIER_N_JOBS:-1}"
ATTAINABLE_REFERENCE_FVES="${ATTAINABLE_REFERENCE_FVES:-token=1.0,temporal=1.0,mean=1.0}"
ADJACENT_SAMPLE_PAIRS="${ADJACENT_SAMPLE_PAIRS:-2048}"
ADJACENT_FEATURE_SAMPLE_SIZE="${ADJACENT_FEATURE_SAMPLE_SIZE:-8192}"
ADJACENT_MIN_FEATURE_SUPPORT="${ADJACENT_MIN_FEATURE_SUPPORT:-8}"
ADJACENT_UTILIZATION_K="${ADJACENT_UTILIZATION_K:-8}"
ADJACENT_TOKEN_BATCH_SIZE="${ADJACENT_TOKEN_BATCH_SIZE:-256}"
ADJACENT_MEAN_BATCH_SIZE="${ADJACENT_MEAN_BATCH_SIZE:-64}"
ADJACENT_DEVICE="${ADJACENT_DEVICE:-auto}"

GPU_COUNT="${GPU_COUNT:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

OVERWRITE_EVIDENCE="${OVERWRITE_EVIDENCE:-0}"
OVERWRITE_BENCHMARK="${OVERWRITE_BENCHMARK:-0}"
OVERWRITE_PROBE_FEATURES="${OVERWRITE_PROBE_FEATURES:-0}"
OVERWRITE_PROBES="${OVERWRITE_PROBES:-0}"
RUN_DOCUMENT_LINKING="${RUN_DOCUMENT_LINKING:-1}"
OVERWRITE_DOCUMENT_LINKING="${OVERWRITE_DOCUMENT_LINKING:-0}"
DOCUMENT_LINKING_JACCARD_MAX="${DOCUMENT_LINKING_JACCARD_MAX:-0.10}"
DOCUMENT_LINKING_MIN_WORD_LENGTH="${DOCUMENT_LINKING_MIN_WORD_LENGTH:-3}"
DOCUMENT_LINKING_TOP_K="${DOCUMENT_LINKING_TOP_K:-128}"
DOCUMENT_LINKING_BATCH_SIZE="${DOCUMENT_LINKING_BATCH_SIZE:-8}"
DOCUMENT_LINKING_MAX_LENGTH="${DOCUMENT_LINKING_MAX_LENGTH:-512}"
DOCUMENT_LINKING_DEVICE="${DOCUMENT_LINKING_DEVICE:-cuda:0}"
DOCUMENT_LINKING_EXAMPLES="${DOCUMENT_LINKING_EXAMPLES:-3}"

RFVE_DIR="${EVAL_ROOT}/rfve"
DICTIONARY_UTILIZATION_DIR="${EVAL_ROOT}/dictionary_utilization"
SHARED_DIR="${EVAL_ROOT}/shared"
SEMANTIC_GEOMETRY_DIR="${EVAL_ROOT}/semantic_geometry"
LABEL_EFFICIENCY_DIR="${EVAL_ROOT}/label_efficiency"
TEMPORAL_ROBUSTNESS_DIR="${EVAL_ROOT}/temporal_robustness"
FEATURE_DYNAMICS_DIR="${EVAL_ROOT}/feature_dynamics"
AUTOINTERP_DIR="${EVAL_ROOT}/autointerp"
DOCUMENT_LINKING_DIR="${EVAL_ROOT}/document_linking"
REASONING_DIR="${EVAL_ROOT}/reasoning"
SEMANTIC_INVARIANCE_DIR="${EVAL_ROOT}/semantic_invariance"

TRAINING_FIDELITY_RESULT="${RFVE_DIR}/training_fidelity.json"
RECONSTRUCTION_COSINE_RESULT="${RFVE_DIR}/reconstruction_cosine.json"
EVIDENCE_DIR="${DICTIONARY_UTILIZATION_DIR}/feature_evidence"
HIGH_LEVEL_FEATURE_RESULT="${SHARED_DIR}/high_level_feature_analysis/high_level_feature_results.json"
ADJACENT_CONSISTENCY_DIR="${SHARED_DIR}/feature_consistency/adjacent_consistency"
DICTIONARY_UTILIZATION_RESULT="${DICTIONARY_UTILIZATION_DIR}/dictionary_utilization.json"
BENCHMARK_DIR="${SHARED_DIR}/downstream_transfer/arxiv_benchmark"
PROBE_FEATURE_DIR="${SHARED_DIR}/downstream_transfer/probe_features"
PROBE_RESULT="${LABEL_EFFICIENCY_DIR}/linear_probe_results.json"
REPRESENTATION_GEOMETRY_DIR="${SEMANTIC_GEOMETRY_DIR}/representation_geometry"
LOG_FILE="${LOG_DIR}/eval_layer${LAYER}_w${SAE_WIDTH}_k${K}_1b.log"
mkdir -p \
  "${EVAL_ROOT}" \
  "${RFVE_DIR}" \
  "${DICTIONARY_UTILIZATION_DIR}" \
  "${SHARED_DIR}" \
  "${SEMANTIC_GEOMETRY_DIR}" \
  "${LABEL_EFFICIENCY_DIR}" \
  "${TEMPORAL_ROBUSTNESS_DIR}" \
  "${FEATURE_DYNAMICS_DIR}" \
  "${AUTOINTERP_DIR}" \
  "${DOCUMENT_LINKING_DIR}" \
  "${REASONING_DIR}" \
  "${SEMANTIC_INVARIANCE_DIR}" \
  "${LOG_DIR}"

run_evaluation() {
  echo "[chunk-saes eval] model=${MODEL} layer=${LAYER}/${NUM_LAYERS} sae=${SAE_ROOT} selection=${EVAL_CHECKPOINT_SELECTION}"
  local -a fidelity_args=(
    "${PY}" "${SRC_DIR}/evals/rfve/analyze_training_fidelity.py"
    --sae-root "${SAE_ROOT}" \
    --checkpoint-selection "${EVAL_CHECKPOINT_SELECTION}" \
    --attainable-reference-fves "${ATTAINABLE_REFERENCE_FVES}" \
    --output "${TRAINING_FIDELITY_RESULT}"
  )
  [[ -z "${CROSS_REFERENCE_MANIFEST}" ]] ||
    fidelity_args+=(--cross-reference-manifest "${CROSS_REFERENCE_MANIFEST}")
  [[ -z "${JOINT_SAE_ROOT}" ]] ||
    fidelity_args+=(--joint-sae-root "${JOINT_SAE_ROOT}")
  [[ "${ALLOW_AUDIT_CROSS_SCALAR}" != "1" ]] ||
    fidelity_args+=(--allow-audit-cross-scalar)
  "${fidelity_args[@]}"

  local -a evidence_args=(
    "${TORCHRUN[@]}" --standalone --nproc_per_node "${GPU_COUNT}" --max_restarts 0
    "${SRC_DIR}/evals/dictionary_utilization/extract_feature_evidence.py"
    --model "${MODEL}"
    --hf-dataset "${HF_DATASET}"
    --hf-dataset-revision "${HF_DATASET_REVISION}"
    --hf-physical-split "${HF_DATASET_PHYSICAL_SPLIT}"
    --document-shuffle-seed "${DOCUMENT_ORDER_SEED}"
    --stream-shuffle-buffer "${STREAM_SHUFFLE_BUFFER}"
    --corpus-source "${CORPUS_SOURCE}"
    --sae-root "${SAE_ROOT}"
    --checkpoint-selection "${EVAL_CHECKPOINT_SELECTION}"
    --output-dir "${EVIDENCE_DIR}"
    --layer "${LAYER}"
    --split-seed "${SPLIT_SEED}"
    --sample-seed "${EVIDENCE_SEED}"
    --chunk-lengths "${CHUNK_LENGTHS}"
    --evidence-chunks "${EVIDENCE_CHUNKS}"
    --feature-sample-size "${FEATURE_SAMPLE_SIZE}"
    --top-n "${TOP_ACTIVATING_CHUNKS}"
    --forward-batch-size "${EVIDENCE_FORWARD_BATCH_SIZE}"
    --model-dtype "${MODEL_DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --chunks-per-document "${EVIDENCE_CHUNKS_PER_DOCUMENT}"
    --max-passes "${EVIDENCE_MAX_PASSES}"
  )
  [[ "${OVERWRITE_EVIDENCE}" == "1" ]] && evidence_args+=(--overwrite)
  "${evidence_args[@]}"

  local -a benchmark_args=(
    "${PY}" "${SRC_DIR}/evals/supervised_transfer/prepare_arxiv_benchmark.py"
    --dataset "${ARXIV_DATASET}"
    --output-dir "${BENCHMARK_DIR}"
    --ood-year "${OOD_YEAR}"
    --train-per-class "${PROBE_TRAIN_PER_CLASS}"
    --validation-per-class "${PROBE_VALIDATION_PER_CLASS}"
    --test-per-class "${PROBE_TEST_PER_CLASS}"
    --ood-per-class "${PROBE_OOD_PER_CLASS}"
    --seed "${PROBE_SEED}"
  )
  [[ "${OVERWRITE_BENCHMARK}" == "1" ]] && benchmark_args+=(--overwrite)
  "${benchmark_args[@]}"

  local -a feature_args=(
    "${TORCHRUN[@]}" --standalone --nproc_per_node "${GPU_COUNT}" --max_restarts 0
    "${SRC_DIR}/evals/supervised_transfer/extract_probe_features.py"
    --model "${MODEL}"
    --benchmark-dir "${BENCHMARK_DIR}"
    --benchmark-manifest "${BENCHMARK_DIR}/manifest.json"
    --sae-root "${SAE_ROOT}"
    --checkpoint-selection "${EVAL_CHECKPOINT_SELECTION}"
    --output-dir "${PROBE_FEATURE_DIR}"
    --layer "${LAYER}"
    --batch-size "${PROBE_BATCH_SIZE}"
    --storage-k "${SPARSE_STORAGE_K}"
    --token-candidate-k "${TOKEN_CANDIDATE_K}"
    --chunk-length "${PROBE_CHUNK_LENGTH}"
    --max-length "${PROBE_MAX_LENGTH}"
    --model-dtype "${MODEL_DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
  )
  [[ "${OVERWRITE_PROBE_FEATURES}" == "1" ]] && feature_args+=(--overwrite)
  "${feature_args[@]}"

  local -a probe_args=(
    "${PY}" "${SRC_DIR}/evals/supervised_transfer/run_linear_probes.py"
    --features-dir "${PROBE_FEATURE_DIR}"
    --feature-manifest "${PROBE_FEATURE_DIR}/feature_manifest.json"
    --output "${PROBE_RESULT}"
    --dict-size "${SAE_WIDTH}"
    --feature-budgets "${FEATURE_BUDGETS}"
    --low-label-budgets "${LOW_LABEL_BUDGETS}"
    --low-label-seeds "${LOW_LABEL_SEEDS}"
    --classifier-c "${CLASSIFIER_C}"
    --max-iter "${CLASSIFIER_MAX_ITER}"
    --n-jobs "${CLASSIFIER_N_JOBS}"
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
    --bootstrap-seed "${PROBE_SEED}"
    --token-match-reference "${TOKEN_MATCH_REFERENCE}"
    --token-match-k "${TOKEN_MATCH_K}"
  )
  [[ "${OVERWRITE_PROBES}" == "1" ]] && probe_args+=(--overwrite)
  "${probe_args[@]}"

  "${PY}" "${SRC_DIR}/evals/dictionary_utilization/analyze_high_level_features.py" \
    --evidence-dir "${EVIDENCE_DIR}" \
    --evidence-manifest "${EVIDENCE_DIR}/evidence_manifest.json" \
    --output "${HIGH_LEVEL_FEATURE_RESULT}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --seed "${HIGH_LEVEL_ANALYSIS_SEED}"

  "${PY}" "${SRC_DIR}/evals/dictionary_utilization/analyze_adjacent_feature_consistency.py" \
    --activation-cache-dir "${VALIDATION_CACHE_DIR}" \
    --sae-root "${SAE_ROOT}" \
    --checkpoint-selection "${EVAL_CHECKPOINT_SELECTION}" \
    --output-dir "${ADJACENT_CONSISTENCY_DIR}" \
    --dictionary-utilization-output "${DICTIONARY_UTILIZATION_RESULT}" \
    --sample-pairs "${ADJACENT_SAMPLE_PAIRS}" \
    --feature-sample-size "${ADJACENT_FEATURE_SAMPLE_SIZE}" \
    --min-feature-support "${ADJACENT_MIN_FEATURE_SUPPORT}" \
    --utilization-k "${ADJACENT_UTILIZATION_K}" \
    --token-batch-size "${ADJACENT_TOKEN_BATCH_SIZE}" \
    --mean-batch-size "${ADJACENT_MEAN_BATCH_SIZE}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --seed "${PROBE_SEED}" \
    --device "${ADJACENT_DEVICE}"

  "${PY}" "${SRC_DIR}/evals/supervised_transfer/analyze_representation_geometry.py" \
    --features-dir "${PROBE_FEATURE_DIR}" \
    --feature-manifest "${PROBE_FEATURE_DIR}/feature_manifest.json" \
    --probe-results "${PROBE_RESULT}" \
    --probe-manifest "${LABEL_EFFICIENCY_DIR}/linear_probe_manifest.json" \
    --output-dir "${REPRESENTATION_GEOMETRY_DIR}" \
    --dict-size "${SAE_WIDTH}" \
    --seed "${PROBE_SEED}"

  "${PY}" "${SRC_DIR}/evals/visualization/plot_task_figures.py" \
    --training-fidelity-results "${TRAINING_FIDELITY_RESULT}" \
    --probe-results "${PROBE_RESULT}" \
    --adjacent-consistency-results "${ADJACENT_CONSISTENCY_DIR}/adjacent_feature_consistency.json" \
    --dictionary-utilization-results "${DICTIONARY_UTILIZATION_RESULT}" \
    --representation-geometry-results "${REPRESENTATION_GEOMETRY_DIR}/representation_geometry.json" \
    --representation-embeddings "${REPRESENTATION_GEOMETRY_DIR}/representation_embeddings.npz" \
    --training-fidelity-manifest "${RFVE_DIR}/training_fidelity_manifest.json" \
    --probe-manifest "${LABEL_EFFICIENCY_DIR}/linear_probe_manifest.json" \
    --adjacent-consistency-manifest "${ADJACENT_CONSISTENCY_DIR}/adjacent_feature_consistency_manifest.json" \
    --representation-geometry-manifest "${REPRESENTATION_GEOMETRY_DIR}/representation_geometry_manifest.json" \
    --rfve-figure-dir "${RFVE_DIR}/figures" \
    --dictionary-utilization-figure-dir "${DICTIONARY_UTILIZATION_DIR}/figures" \
    --semantic-geometry-figure-dir "${SEMANTIC_GEOMETRY_DIR}/figures" \
    --label-efficiency-figure-dir "${LABEL_EFFICIENCY_DIR}/figures" \
    --temporal-robustness-figure-dir "${TEMPORAL_ROBUSTNESS_DIR}/figures"
  local -a training_health_args=(
    "${PY}" "${SRC_DIR}/evals/rfve/plot_training_health.py"
    --training-fidelity-results "${TRAINING_FIDELITY_RESULT}"
    --sae-root "${SAE_ROOT}"
    --output-dir "${RFVE_DIR}/figures"
  )
  [[ -z "${JOINT_SAE_ROOT}" ]] ||
    training_health_args+=(--joint-sae-root "${JOINT_SAE_ROOT}")
  "${training_health_args[@]}"
  local -a dictionary_health_args=(
    "${PY}" "${SRC_DIR}/evals/rfve/evaluate_dictionary_health.py"
    --training-fidelity-results "${TRAINING_FIDELITY_RESULT}"
    --sae-root "${SAE_ROOT}"
    --validation-cache-dir "${VALIDATION_CACHE_DIR}"
    --output "${RFVE_DIR}/dictionary_health.json"
    --figure-base "${RFVE_DIR}/figures/dictionary_health"
    --device "${ADJACENT_DEVICE}"
  )
  [[ -z "${JOINT_SAE_ROOT}" ]] ||
    dictionary_health_args+=(--joint-sae-root "${JOINT_SAE_ROOT}")
  "${dictionary_health_args[@]}"
  local -a reconstruction_cosine_args=(
    "${TORCHRUN[@]}" --standalone --nproc_per_node "${GPU_COUNT}" --max_restarts 0
    "${SRC_DIR}/evals/rfve/evaluate_reconstruction_cosine.py"
    --training-fidelity-results "${TRAINING_FIDELITY_RESULT}"
    --sae-root "${SAE_ROOT}"
    --validation-cache-dir "${VALIDATION_CACHE_DIR}"
    --output "${RECONSTRUCTION_COSINE_RESULT}"
    --figure-base "${RFVE_DIR}/figures/reconstruction_cosine"
  )
  [[ -z "${JOINT_SAE_ROOT}" ]] ||
    reconstruction_cosine_args+=(--joint-sae-root "${JOINT_SAE_ROOT}")
  "${reconstruction_cosine_args[@]}"
  if [[ "${RUN_DOCUMENT_LINKING}" == "1" ]]; then
    local -a linking_args=(
      "${PY}" "${SRC_DIR}/evals/document_linking/evaluate_lexical_controlled_document_linking.py"
      --model "${MODEL}"
      --layer "${LAYER}"
      --sae-root "${SAE_ROOT}"
      --checkpoint-selection "${EVAL_CHECKPOINT_SELECTION}"
      --evidence-dir "${EVIDENCE_DIR}"
      --evidence-manifest "${EVIDENCE_DIR}/evidence_manifest.json"
      --output-dir "${DOCUMENT_LINKING_DIR}"
      --lexical-jaccard-max "${DOCUMENT_LINKING_JACCARD_MAX}"
      --min-word-length "${DOCUMENT_LINKING_MIN_WORD_LENGTH}"
      --top-k "${DOCUMENT_LINKING_TOP_K}"
      --batch-size "${DOCUMENT_LINKING_BATCH_SIZE}"
      --max-length "${DOCUMENT_LINKING_MAX_LENGTH}"
      --model-dtype "${MODEL_DTYPE}"
      --attn-implementation "${ATTN_IMPLEMENTATION}"
      --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
      --permutation-samples 100000
      --seed 20260817
      --device "${DOCUMENT_LINKING_DEVICE}"
      --example-count "${DOCUMENT_LINKING_EXAMPLES}"
    )
    [[ "${OVERWRITE_DOCUMENT_LINKING}" == "1" ]] &&
      linking_args+=(--overwrite)
    "${linking_args[@]}"
  fi
  "${PY}" "${SRC_DIR}/evals/visualization/plot_paper_figures.py" \
    --eval-root "${EVAL_ROOT}"
  echo "[chunk-saes eval] complete: probes=${PROBE_RESULT}"
}

run_evaluation 2>&1 | tee -a "${LOG_FILE}"
