#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/scripts/so101_square_finetune:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export G05_FORCE_VIDEO_BACKEND=pyav

python "$ROOT/scripts/so101_square_finetune/validate_dataset.py" "$@"
