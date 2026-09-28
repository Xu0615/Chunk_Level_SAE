#!/usr/bin/env bash
# Sequentially train the requested h=32768 nested Joint Chunk SAE alpha sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_ROOT="${REPO_ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/checkpoints"
LOG_ROOT="${REPO_ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/logs"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"
mkdir -p "${LOG_ROOT}"

summarize_progress() {
  local mode_dir="$1"
  "${PY}" - "${mode_dir}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

mode_dir = Path(sys.argv[1])
metrics_path = mode_dir / "metrics.jsonl"
if not metrics_path.is_file():
    print("[nested-sweep] metrics not available yet", flush=True)
    raise SystemExit

rows = []
for line in metrics_path.read_text(encoding="utf-8").splitlines():
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
train = [row for row in rows if row.get("split") == "train"]
validation = [row for row in rows if row.get("split") == "validation"]
if train:
    row = train[-1]
    recent = train[-min(25, len(train)):]
    k_values = [float(item["train/joint_k_prefix"]) for item in recent]
    print(
        "[nested-sweep] "
        f"train step={row['step']}/31250 "
        f"mean_fve={row['train/joint_mean_fve']:.5f} "
        f"cross_fve={row['train/joint_cross_fve']:.5f} "
        f"k_prefix={row['train/joint_k_prefix']:.3f} "
        f"recent_k=[{min(k_values):.3f},{max(k_values):.3f}] "
        f"dead={row['sparsity/dead_features']} "
        f"grad={row['optimizer/grad_norm']:.5f} "
        f"speed={row['system/samples_per_second']:.0f}/s",
        flush=True,
    )
if validation:
    row = validation[-1]
    print(
        "[nested-sweep] "
        f"validation step={row['step']} "
        f"mean_fve={row['validation/joint_mean_fve']:.5f} "
        f"cross_fve={row['validation/joint_cross_fve']:.5f} "
        f"k_prefix={row['validation/joint_k_prefix']:.3f} "
        f"a2b={row['validation/joint_cross/a_to_b/fve']:.5f} "
        f"b2a={row['validation/joint_cross/b_to_a/fve']:.5f} "
        f"best={row['is_best']}",
        flush=True,
    )
PY
}

verify_complete() {
  local mode_dir="$1"
  local expected_alpha="$2"
  "${PY}" - "${mode_dir}" "${expected_alpha}" <<'PY'
import json
import math
import sys
from pathlib import Path

mode_dir = Path(sys.argv[1])
expected_alpha = float(sys.argv[2])
complete = json.loads((mode_dir / "complete.json").read_text(encoding="utf-8"))
config = json.loads((mode_dir / "config.json").read_text(encoding="utf-8"))
assert complete["complete"] is True
assert complete["exact_coverage"] is True
assert int(complete["steps"]) == 31_250
assert int(complete["samples_seen"]) == 1_000_000_000
assert int(complete["accepted_training_rows"]) == 1_000_000_000
assert float(complete["coverage_fraction"]) == 1.0
assert complete["validation_skipped"] is False
assert isinstance(complete["final_full_validation_metrics"], dict)
assert config["joint_chunk_layout"] == "nested_prefix"
assert int(config["joint_cross_prefix"]) == 32_768
assert math.isclose(float(config["joint_chunk_alpha"]), expected_alpha)
for checkpoint in ("best", "latest"):
    assert (
        mode_dir / "checkpoints" / checkpoint / "checkpoint_manifest.json"
    ).is_file()
print(f"[nested-sweep] verified complete: {mode_dir}", flush=True)
PY
}

active_pid=""
cleanup() {
  local status=$?
  if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
    kill -TERM "${active_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup INT TERM

for spec in "0.5:0p5" "1.0:1" "1.5:1p5"; do
  alpha="${spec%%:*}"
  tag="${spec##*:}"
  run_name="qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_joint_chunk_nested_h32768_alpha${tag}"
  mode_dir="${CHECKPOINT_ROOT}/${run_name}/joint_chunk"
  log_path="${LOG_ROOT}/${run_name}.log"

  attempt=0
  while [[ ! -f "${mode_dir}/complete.json" ]]; do
    attempt=$((attempt + 1))
    resume=0
    overwrite=0
    if [[ -d "${mode_dir}/checkpoints/latest" ]]; then
      resume=1
      echo "[nested-sweep] resuming alpha=${alpha} attempt=${attempt} from latest"
    elif [[ -d "${mode_dir}" ]]; then
      overwrite=1
      echo "[nested-sweep] restarting alpha=${alpha} attempt=${attempt}; no resumable checkpoint"
    else
      echo "[nested-sweep] starting alpha=${alpha} attempt=${attempt}"
    fi
    echo "[nested-sweep] launch time $(date --iso-8601=seconds)"

    env \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      RUN_NAME="${run_name}" \
      JOINT_CHUNK_ALPHA="${alpha}" \
      JOINT_CROSS_PREFIX=32768 \
      VALIDATE_EVERY=250 \
      SAVE_EVERY=2000 \
      RESUME="${resume}" \
      OVERWRITE_TRAINING="${overwrite}" \
      MODEL="${MODEL:-${REPO_ROOT}/../models/Qwen3.5-9B-Base}" \
      bash "${SCRIPT_DIR}/train_qwen35_9b_joint_chunk_nested.sh" \
      >>"${log_path}" 2>&1 &
    active_pid=$!

    while kill -0 "${active_pid}" 2>/dev/null; do
      sleep 300
      summarize_progress "${mode_dir}" || {
        echo "[nested-sweep] progress read failed; training process remains under supervision" >&2
      }
    done
    set +e
    wait "${active_pid}"
    status=$?
    set -e
    active_pid=""
    if [[ "${status}" -ne 0 ]]; then
      echo "[nested-sweep] alpha=${alpha} attempt=${attempt} failed with status ${status}; retrying" >&2
      tail -n 100 "${log_path}" >&2 || true
      sleep 60
    fi
  done

  until verify_complete "${mode_dir}" "${alpha}"; do
    echo "[nested-sweep] completion verification temporarily failed; retrying" >&2
    sleep 30
  done
  summarize_progress "${mode_dir}"
  echo "[nested-sweep] completed alpha=${alpha} at $(date --iso-8601=seconds)"
done

trap - INT TERM
echo "[nested-sweep] all requested runs complete at $(date --iso-8601=seconds)"
