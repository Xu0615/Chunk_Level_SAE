#!/usr/bin/env bash
# Switch an already-running ablation orchestrator to the current runner at the
# E3 boundary. E1/E2 artifacts are complete and therefore skipped on restart.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS="${ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/ablation/status.tsv"
LOG_DIR="${ROOT}/qwen/qwen35-9b-base/pile_proportional_exact_occurrence/ablation/logs"
RUN_LOG="${LOG_DIR}/orchestrator.log"
RELAUNCH_LOG="${LOG_DIR}/relauncher.log"
OLD_PID="${OLD_ABLATION_PID:-3128845}"

children_of() {
  local parent="$1" child
  for child in $(pgrep -P "${parent}" 2>/dev/null || true); do
    printf '%s\n' "${child}"
    children_of "${child}"
  done
}

mkdir -p "${LOG_DIR}"
while kill -0 "${OLD_PID}" 2>/dev/null; do
  if awk -F $'\t' '$2 == "E3" && $3 == "pilot" && $4 == "42" && $5 ~ /^START/' "${STATUS}" 2>/dev/null | tail -n 1 | grep -q .; then
    printf '%s old orchestrator reached E3; stopping stale sequence launch\n' "$(date --iso-8601=seconds)" >>"${RELAUNCH_LOG}"
    for pid in $(children_of "${OLD_PID}"); do
      kill -TERM "${pid}" 2>/dev/null || true
    done
    kill -TERM "${OLD_PID}" 2>/dev/null || true
    sleep 15
    # The old sequence torchrun may have reparented its workers; constrain the
    # cleanup to this repository's exact trainer path.
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      kill -TERM "${pid}" 2>/dev/null || true
    done < <(pgrep -f "${ROOT}/src/train_sequence_cross_ablation.py" 2>/dev/null || true)
    sleep 5
    nohup bash "${ROOT}/src/run_ablation_experiments.sh" >>"${RUN_LOG}" 2>&1 &
    printf '%s started latest orchestrator pid=%s\n' "$(date --iso-8601=seconds)" "$!" >>"${RELAUNCH_LOG}"
    exit 0
  fi
  sleep 20
done
printf '%s old orchestrator exited before E3 boundary\n' "$(date --iso-8601=seconds)" >>"${RELAUNCH_LOG}"
