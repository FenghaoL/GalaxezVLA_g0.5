#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/so101_square_finetune"

if [[ $# -gt 0 ]]; then
  RUN_DIR="$SCRIPT_DIR/runs/so100/$1"
else
  RUN_DIR="$(cat "$SCRIPT_DIR/latest_run.txt")"
fi

LOG="$RUN_DIR/train.log"
[[ -f "$LOG" ]] || LOG="$RUN_DIR/launcher.log"
tail -n 120 -f "$LOG"
