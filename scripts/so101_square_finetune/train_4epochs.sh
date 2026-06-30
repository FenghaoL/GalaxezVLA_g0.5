#!/usr/bin/env bash
# Run directly in the foreground, or use start_background.sh for the normal case.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/so101_square_finetune"
RUN_NAME="${1:-so101_square_4ep_$(date +%Y%m%d_%H%M%S)}"
# configs/train.yaml places each task beneath G05_OUTPUT_DIR.
RUN_DIR="$SCRIPT_DIR/runs/so100/$RUN_NAME"

cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/scripts/so101_square_finetune:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export G05_OUTPUT_DIR="$SCRIPT_DIR/runs"
export EXP_NAME="$RUN_NAME"
export OVERRIDE_DATASET="$SCRIPT_DIR/so101_square_data.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export G05_FORCE_VIDEO_BACKEND=pyav
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/latest_run.txt"

# Notes on the intentionally small set of overrides:
# - New statistics must be computed from the model-frame data, never copied from g05-so101.
# - BF16 AMP keeps activations compact; weights remain FP32 because this repo's
#   Liger CE kernel requires FP32 hidden states and output projection weights.
# - Batch 2/GPU plus full gradient checkpointing fits FP32 weights on 40 GB A100s.
#   Batch 4 OOMs in the Liger fused-linear-CE head (it holds an FP32 vocab x hidden
#   grad_weight); the A100 vision SDPA fallback also costs more than flash-attn.
# - The source checkpoint remains untouched; all train products live in RUN_DIR.
exec python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/finetune.py \
  task=so100 \
  datastatics_path=null \
  data.use_weight_for_sampling=true \
  model.pretrained_ckpt=checkpoints/g05-so101/checkpoints/model_state_dict.pt \
  model.use_pretrained_norm_stats=false \
  model.use_8bit_optimizer=true \
  model.batch_size=2 \
  model.num_workers=4 \
  model.max_epochs=4 \
  model.max_steps=null \
  model.learning_rate=2.0e-5 \
  model.warmup_steps=200 \
  model.model_arch.checkpoint_vision=true \
  model.model_arch.checkpoint_vlm=true \
  model.model_arch.checkpoint_action_expert=true \
  checkpointing_steps=500 \
  eval_steps=500 \
  logger.project=g05-so101-square \
  logger.workspace=null \
  logger.experiment_name="$RUN_NAME"
