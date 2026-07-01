#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"

cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

bash "$SCRIPT_DIR/prepare_data.sh"
bash "$SCRIPT_DIR/build_pairs.sh"

python "$SCRIPT_DIR/tools/check_data.py" \
  --prepared-root "$ROOT/data/g05_rl_prepared/so101_g05_rl_pick_white_v1" \
  --pairs "$SCRIPT_DIR/cache/pairs.jsonl"

