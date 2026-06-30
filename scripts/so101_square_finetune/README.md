# SO101 square-camera G0.5 fine-tuning

This directory is self-contained: scripts, data configuration, logs, W&B
artifacts, resolved Hydra config, fresh `dataset_stats.json`, and checkpoints
are kept here.  The prepared datasets are deliberately siblings of the raw
recordings under `data/`, so the raw recordings remain immutable.

Run the pipeline in order:

```bash
bash scripts/so101_square_finetune/prepare_data.sh
bash scripts/so101_square_finetune/validate_data.sh
bash scripts/so101_square_finetune/start_background.sh
```

Monitor the current run with:

```bash
bash scripts/so101_square_finetune/tail_log.sh
```

To stop a specific run, use only its run name (not its full path):

```bash
bash scripts/so101_square_finetune/stop_run.sh <run-name>
```

`prepare_data.sh` applies the G0.5 SO101 coordinate transform to **both**
`action` and `observation.state` parquet columns. It writes a
`*_g05_model_frame` sibling for each timestamp run. Videos are symlinked, not
copied or re-encoded.

The reset footage between episodes needs no trimming. The repository's V3
reader uses `meta/episodes.dataset_from_index/dataset_to_index` to clamp every
32-step action chunk within its episode, and uses only each episode's video
`from_timestamp` plus the selected parquet-row timestamp to retrieve frames.
It does not use `to_timestamp` to construct training samples.

The data configuration preserves the recording camera contract exactly:

- `fixed` → `exterior`, raw `3×480×480`
- `wrist` → `wrist_right`, raw `3×480×640`
- `wrist_left` → black padding

It intentionally disables the generic SO100 camera swapping augmentation,
because it would violate that deployment-time mapping. The processor then
resizes images to G0.5's `256×256` model input.

The server has an installed but unusable TorchCodec shared library, so these
run scripts force the already-installed PyAV decoder only for this training
process and its data-loader workers. No global package is changed.

The four task groups are sampled equally by using inverse-frame-count weights;
this is necessary in this repo because group sampling is proportional to
`frames × weight`. The 5% validation split remains per task group.

The training command starts from
`checkpoints/g05-so101/checkpoints/model_state_dict.pt`, runs four epochs on
four visible GPUs with BF16 AMP (and FP32 model weights, required by the
repository's Liger CE kernel), and computes a new `dataset_stats.json` in the run
directory. Never deploy this fine-tuned checkpoint with the original
`checkpoints/g05-so101/dataset_stats.json`.

After training, deploy the final file with the matching run directory sidecars:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/so100/start_server.sh \
  scripts/so101_square_finetune/runs/so100/<run-name>/checkpoints/step_<final-step>.pt
```

The server resolves `dataset_stats.json` and `.hydra/config.yaml` from that
run directory automatically. Keep the copied `action_tokenizer.pt` and any
`hf_processor/` sidecars in the same run directory as well.
