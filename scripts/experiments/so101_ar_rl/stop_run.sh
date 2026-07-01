#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"
SUPERVISOR_PID_FILE="$SCRIPT_DIR/supervisor.pid"

if [[ ! -f "$SUPERVISOR_PID_FILE" ]]; then
  echo "No supervisor.pid found."
  exit 0
fi

pid="$(cat "$SUPERVISOR_PID_FILE" || true)"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "Supervisor is not running."
  exit 0
fi

echo "Stopping supervisor PID $pid"
pkill -TERM -P "$pid" 2>/dev/null || true
kill -TERM "$pid" 2>/dev/null || true
sleep 2
if kill -0 "$pid" 2>/dev/null; then
  echo "Supervisor still alive; sending KILL."
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
fi
echo "Stopped."

