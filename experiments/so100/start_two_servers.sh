#!/usr/bin/env bash
# Start two SO100/G0.5 policy servers on two GPUs.
#
# Usage:
#   bash experiments/so100/start_two_servers.sh /path/to/ckpt_a.pt /path/to/ckpt_b.pt
#
# Optional environment variables:
#   GPU_A=0 GPU_B=1 PORT_A=8765 PORT_B=8766
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT="$HERE/../.."

if [[ $# -lt 2 ]]; then
  echo "Usage: bash experiments/so100/start_two_servers.sh /path/to/ckpt_a.pt /path/to/ckpt_b.pt" >&2
  exit 2
fi

CKPT_A="$1"
CKPT_B="$2"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
PORT_A="${PORT_A:-8765}"
PORT_B="${PORT_B:-8766}"
LOG_DIR="$PROJECT/experiments/so100/server_logs"

mkdir -p "$LOG_DIR"

echo "Starting server A: GPU=$GPU_A PORT=$PORT_A CKPT=$CKPT_A"
CUDA_VISIBLE_DEVICES="$GPU_A" PORT="$PORT_A" \
  bash "$HERE/start_server.sh" "$CKPT_A" \
  > "$LOG_DIR/server_a_port${PORT_A}.log" 2>&1 &
PID_A="$!"

echo "Starting server B: GPU=$GPU_B PORT=$PORT_B CKPT=$CKPT_B"
CUDA_VISIBLE_DEVICES="$GPU_B" PORT="$PORT_B" \
  bash "$HERE/start_server.sh" "$CKPT_B" \
  > "$LOG_DIR/server_b_port${PORT_B}.log" 2>&1 &
PID_B="$!"

printf '%s\n' "$PID_A" > "$LOG_DIR/server_a.pid"
printf '%s\n' "$PID_B" > "$LOG_DIR/server_b.pid"

echo "Server A PID: $PID_A  URI: ws://<server-ip>:$PORT_A"
echo "Server B PID: $PID_B  URI: ws://<server-ip>:$PORT_B"
echo "Logs:"
echo "  $LOG_DIR/server_a_port${PORT_A}.log"
echo "  $LOG_DIR/server_b_port${PORT_B}.log"
echo
echo "To stop:"
echo "  kill $PID_A $PID_B"
