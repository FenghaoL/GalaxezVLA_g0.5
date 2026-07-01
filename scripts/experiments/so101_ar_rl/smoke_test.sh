#!/usr/bin/env bash
# Fast sanity check: data prep, pair build, then 2 optimizer steps for DPO and SFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"

bash "$SCRIPT_DIR/prepare_data.sh"
bash "$SCRIPT_DIR/build_pairs.sh"

echo "[smoke] AR-DPO 2 steps"
RUN_NAME="smoke_ar_dpo_$STAMP" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_STEPS=2 \
CHECKPOINTING_STEPS=1 \
EVAL_STEPS=100000 \
LOGGER_MODE=offline \
bash "$SCRIPT_DIR/train_ar_dpo.sh"

echo "[smoke] success-only AR-SFT 2 steps"
RUN_NAME="smoke_ar_sft_$STAMP" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_STEPS=2 \
CHECKPOINTING_STEPS=1 \
EVAL_STEPS=100000 \
LOGGER_MODE=offline \
bash "$SCRIPT_DIR/train_ar_sft_success.sh"

echo "[smoke] done"

