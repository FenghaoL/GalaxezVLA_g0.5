#!/usr/bin/env bash
# Fast sanity check: data prep, pair build, then 2 optimizer steps for window AR-DPO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"

bash "$SCRIPT_DIR/prepare_data.sh"
bash "$SCRIPT_DIR/build_pairs.sh"

echo "[smoke] window AR-DPO 2 steps"
RUN_NAME="smoke_window_dpo_$STAMP" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_STEPS=2 \
CHECKPOINTING_STEPS=100000 \
EVAL_STEPS=100000 \
LOGGER_MODE=offline \
DPO_SAMPLE_MODE=window \
WINDOW_SIZE=8 \
WINDOW_STRIDE=4 \
WINDOW_ALIGN=relative \
MAX_WINDOWS_PER_PAIR=1 \
LOGP_MICRO_BATCH_SIZE=1 \
bash "$SCRIPT_DIR/train_ar_dpo.sh"

echo "[smoke] done"
