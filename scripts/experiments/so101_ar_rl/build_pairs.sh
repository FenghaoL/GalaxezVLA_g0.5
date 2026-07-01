#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"
CACHE_DIR="$SCRIPT_DIR/cache"

cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$CACHE_DIR"

python "$SCRIPT_DIR/tools/build_rl_pairs.py" \
  --labels-root "$ROOT/data/g05_rl_prepared/so101_g05_rl_pick_white_v1" \
  --output "$CACHE_DIR/pairs.jsonl" \
  --exclude-episode-uid 20260701_114422_ep00003 \
  --expect-pairs 85

