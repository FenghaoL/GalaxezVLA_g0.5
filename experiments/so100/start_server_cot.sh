#!/usr/bin/env bash
# Experimental SO100/SO101 server that asks the AR head for a subtask before
# producing the action.  It leaves start_server.sh unchanged.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash experiments/so100/start_server_cot.sh /path/to/step_xxx.pt
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT="$HERE/../.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/so100/start_server_cot.sh /path/to/checkpoint.pt" >&2
  exit 2
fi

CKPT="$1"

export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"

echo "Starting experimental CoT SO100 policy server on $(hostname) ..."
echo "  ckpt = $CKPT"
echo "  CoT  = SubtaskCoTBuilder (predict subtask before action)"
echo "  CoT token budget = 64"
echo "  note = inspect cot_text before using this server to control the arm"
cd "$PROJECT"
python "$PROJECT/scripts/serve_policy.py" \
  --ckpt_path "$CKPT" \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda \
  --action_steps 32 \
  eval_embodiment=so100 \
  model.model_weights_to_bf16=true \
  model.use_torch_compile=true \
  model.model_arch.attn_implementation=sdpa \
  model.model_arch.predict_cot=true \
  model.model_arch.ar.max_new_tokens=64 \
  model.processor.samples_builder._target_=g05.data_processor.processor.samples_builder.SubtaskCoTBuilder \
  model.processor.samples_builder._partial_=true \
  model.processor.samples_builder._recursive_=false
