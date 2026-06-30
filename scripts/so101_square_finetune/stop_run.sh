#!/usr/bin/env bash
# Stop one named G0.5 SO101 square fine-tuning run and its data-loader workers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/so101_square_finetune"
RUN_NAME="${1:?Usage: bash scripts/so101_square_finetune/stop_run.sh <run-name>}"

PIDS="$(pgrep -f "scripts/finetune.py.*logger.experiment_name=${RUN_NAME}" || true)"
if [[ -z "$PIDS" ]]; then
  echo "No live finetune process found for $RUN_NAME"
  exit 0
fi

echo "Stopping run $RUN_NAME (PIDs: $PIDS)"
kill -TERM $PIDS
