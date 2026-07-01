#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"

cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python "$SCRIPT_DIR/tools/prepare_g05_rl_data.py" \
  --raw-root "$ROOT/data/g05_rl_raw/so101_g05_rl_pick_white_v1" \
  --prepared-root "$ROOT/data/g05_rl_prepared/so101_g05_rl_pick_white_v1"

