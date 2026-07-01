#!/usr/bin/env bash
# Start a background supervisor for the current GRAPE-style window AR-DPO run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"
LOG_DIR="$SCRIPT_DIR/runs/background"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUPERVISOR_LOG="$LOG_DIR/supervisor_$STAMP.log"
SUPERVISOR_PID_FILE="$SCRIPT_DIR/supervisor.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$SUPERVISOR_PID_FILE" ]]; then
  old_pid="$(cat "$SUPERVISOR_PID_FILE" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Supervisor is already running: PID $old_pid"
    echo "Log: $(cat "$SCRIPT_DIR/latest_supervisor_log.txt" 2>/dev/null || true)"
    exit 1
  fi
fi

setsid bash -c "
  set -euo pipefail
  cd '$ROOT'
  echo '[supervisor] start: '\"\$(date)\"
  echo '[supervisor] running window AR-DPO'
  RUN_NAME='ar_dpo_window_$STAMP' bash '$SCRIPT_DIR/train_ar_dpo.sh'
  echo '[supervisor] AR-DPO finished: '\"\$(date)\"
  echo '[supervisor] all done: '\"\$(date)\"
" > "$SUPERVISOR_LOG" 2>&1 &

pid="$!"
printf '%s\n' "$pid" > "$SUPERVISOR_PID_FILE"
printf '%s\n' "$SUPERVISOR_LOG" > "$SCRIPT_DIR/latest_supervisor_log.txt"
printf '%s\n' "$SCRIPT_DIR/runs/so100/ar_dpo_window_$STAMP" > "$SCRIPT_DIR/latest_dpo_run.txt"
printf '%s\n' "$SCRIPT_DIR/runs/so100/ar_dpo_window_$STAMP" > "$SCRIPT_DIR/latest_run.txt"

echo "Started supervisor PID $pid"
echo "Log: $SUPERVISOR_LOG"
echo "Order: window AR-DPO only"
