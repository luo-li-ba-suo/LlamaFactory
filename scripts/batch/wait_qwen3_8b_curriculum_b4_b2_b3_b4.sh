#!/usr/bin/env bash
# Wait for every GPU to be idle, then run the selected remaining curriculum jobs.
set -euo pipefail

cd "$(dirname "$0")/../.."

while true; do
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F ',' '$1 + 0 > 100 || $2 + 0 > 0 { exit 1 }'; then
    echo "[$(date -Is)] All GPUs are idle; starting B4 and B2+B3+B4."
    if [[ -n "${PAUSED_QUEUE_PID:-}" ]]; then
      kill -KILL "$PAUSED_QUEUE_PID" 2>/dev/null || true
    fi
    exec bash scripts/batch/rm_qwen3_8b_curriculum_b4_b2_b3_b4.sh
  fi

  echo "[$(date -Is)] GPUs are busy; retrying in 60 seconds."
  sleep 60
done
