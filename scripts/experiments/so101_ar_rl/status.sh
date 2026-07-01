#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"

show_pid() {
  local label="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$label PID: $pid (running)"
    else
      echo "$label PID: ${pid:-none} (not running)"
    fi
  else
    echo "$label PID: none"
  fi
}

show_run() {
  local label="$1"
  local file="$2"
  echo
  echo "[$label]"
  if [[ ! -f "$file" ]]; then
    echo "No latest run file: $file"
    return
  fi
  local run_dir
  run_dir="$(cat "$file")"
  echo "Run dir: $run_dir"
  if [[ -d "$run_dir/checkpoints" ]]; then
    find "$run_dir/checkpoints" -maxdepth 1 -type f -name '*.pt' | sort | tail -5
  else
    echo "No checkpoints yet."
  fi
  if [[ -f "$run_dir/train.log" ]]; then
    echo "--- train.log tail ---"
    tail -80 "$run_dir/train.log"
  else
    echo "No train.log yet."
  fi
}

show_pid "supervisor" "$SCRIPT_DIR/supervisor.pid"
if [[ -f "$SCRIPT_DIR/latest_supervisor_log.txt" ]]; then
  echo "Supervisor log: $(cat "$SCRIPT_DIR/latest_supervisor_log.txt")"
fi

show_run "DPO" "$SCRIPT_DIR/latest_dpo_run.txt"
show_run "SFT" "$SCRIPT_DIR/latest_sft_run.txt"

