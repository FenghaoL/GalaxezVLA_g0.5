#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

python "$ROOT/scripts/so101_square_finetune/prepare_g05_model_frame.py" "$@"
