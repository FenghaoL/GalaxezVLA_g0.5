#!/usr/bin/env bash
# Foreground success-only AR-SFT baseline. Run with no arguments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/experiments/so101_ar_rl"
RUN_NAME="${RUN_NAME:-ar_sft_success_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$SCRIPT_DIR/runs/so100/$RUN_NAME"
BASE_CKPT="$ROOT/scripts/so101_square_finetune/runs/so100/so101_square_4ep_20260623_205729/checkpoints/step_9740.pt"
DATA_YAML="$SCRIPT_DIR/configs/so101_ar_rl_data.yaml"

cd "$ROOT"
source "$ROOT/.venv/bin/activate"

export PYTHONPATH="$ROOT/scripts/so101_square_finetune:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export G05_OUTPUT_DIR="$SCRIPT_DIR/runs"
export EXP_NAME="$RUN_NAME"
export OVERRIDE_DATASET="$DATA_YAML"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export G05_FORCE_VIDEO_BACKEND=pyav
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IFS=',' read -r -a _g05_gpus <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#_g05_gpus[@]}}"
MAX_STEPS="${MAX_STEPS:-600}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-100000}"
LOGGER_MODE="${LOGGER_MODE:-online}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LOGP_MICRO_BATCH_SIZE="${LOGP_MICRO_BATCH_SIZE:-1}"
SFT_ACTION_ONLY="${SFT_ACTION_ONLY:-false}"

bash "$SCRIPT_DIR/prepare_data.sh"

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/latest_run.txt"
printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/latest_sft_run.txt"

python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  scripts/finetune.py \
  task=so100 \
  datastatics_path=null \
  data.use_weight_for_sampling=false \
  model.pretrained_ckpt="$BASE_CKPT" \
  model.use_pretrained_norm_stats=true \
  model.use_8bit_optimizer=true \
  model.find_unused_parameters=true \
  model.batch_size="$BATCH_SIZE" \
  model.grad_accumulation_steps="$GRAD_ACCUMULATION_STEPS" \
  model.num_workers=2 \
  model.max_epochs=null \
  model.max_steps="$MAX_STEPS" \
  model.learning_rate=5.0e-6 \
  model.warmup_steps=20 \
  model.model_arch.discrete_action=true \
  model.model_arch.continuous_action=false \
  model.model_arch.return_continuous_action=false \
  model.model_arch.checkpoint_vision=true \
  model.model_arch.checkpoint_vlm=true \
  model.model_arch.checkpoint_action_expert=false \
  checkpointing_steps="$CHECKPOINTING_STEPS" \
  eval_steps="$EVAL_STEPS" \
  logger.project=g05-so101-ar-rl \
  logger.workspace=null \
  logger.mode="$LOGGER_MODE" \
  logger.experiment_name="$RUN_NAME" \
  +rl.mode=ar_sft_success \
  +rl.labels_root="$ROOT/data/g05_rl_prepared/so101_g05_rl_pick_white_v1" \
  +rl.action_only="$SFT_ACTION_ONLY" \
  +rl.logp_micro_batch_size="$LOGP_MICRO_BATCH_SIZE" \
  '+rl.exclude_episode_uids=[20260701_114422_ep00003]'
