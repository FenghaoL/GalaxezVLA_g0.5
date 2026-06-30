#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/so101_square_finetune"
RUN_NAME="${1:-so101_square_4ep_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$SCRIPT_DIR/runs/so100/$RUN_NAME"

mkdir -p "$RUN_DIR"
# ``setsid -f`` detaches from the invoking terminal/process group.  Plain
# ``nohup ... &`` is not sufficient under some remote-job launchers, which
# clean up children when the command-return channel closes.
setsid -f "$SCRIPT_DIR/train_4epochs.sh" "$RUN_NAME" \
  > "$RUN_DIR/launcher.log" 2>&1 < /dev/null
sleep 1
PID="$(pgrep -f "scripts/finetune.py.*${RUN_NAME}" | head -n 1 || true)"
if [[ -z "$PID" ]]; then
  echo "Training process did not remain alive; inspect $RUN_DIR/launcher.log" >&2
  exit 1
fi
printf '%s\n' "$PID" > "$RUN_DIR/launcher.pid"
printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/latest_run.txt"

echo "Started PID $PID"
echo "Run directory: $RUN_DIR"
echo "Log: $RUN_DIR/launcher.log"
