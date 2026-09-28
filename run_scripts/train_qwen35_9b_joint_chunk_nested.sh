#!/usr/bin/env bash
# Train the h=32768 nested-prefix Joint Chunk SAE on the exact 1B occurrence cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SAE_MODES="joint_chunk"
export JOINT_CHUNK_ALPHA="${JOINT_CHUNK_ALPHA:-0.25}"
export JOINT_CROSS_PREFIX="${JOINT_CROSS_PREFIX:-32768}"
export DIRECTION_POLICY="both"
export GPU_COUNT="8"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GLOBAL_BATCH_SIZE="32000"
export TRAIN_STEPS="31250"
export AUXK_ALPHA="0.0625"
export SAE_PARALLELISM="ddp"
export DECODER_BACKEND="sparse"
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-250}"
export RUN_EVALUATION="0"
export RUN_NAME="${RUN_NAME:-qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_joint_chunk_nested_h32768_alpha0p25}"

exec bash "${SCRIPT_DIR}/train_qwen35_9b_chunk_saes.sh" train
