#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"

if [[ -f "$SCRIPT_DIR/latest_supervisor_log.txt" ]]; then
  log_path="$(cat "$SCRIPT_DIR/latest_supervisor_log.txt")"
elif [[ -f "$SCRIPT_DIR/latest_run.txt" ]]; then
  run_dir="$(cat "$SCRIPT_DIR/latest_run.txt")"
  log_path="$run_dir/train.log"
else
  echo "No latest log found."
  exit 1
fi

echo "Tailing: $log_path"
tail -f "$log_path"

