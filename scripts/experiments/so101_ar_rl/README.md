# G0.5 SO101 AR-SFT vs AR-DPO 实验说明

这个目录用于在同一个 base checkpoint 上做两个公平对照：

- `ar_dpo_pairs`：用同 bucket 内 success/failure 轨迹 pair 做 AR-DPO。
- `ar_sft_success`：只用 success episode 继续做 AR-SFT。

两者都从同一个 checkpoint 开始：

`scripts/so101_square_finetune/runs/so100/so101_square_4ep_20260623_205729/checkpoints/step_9740.pt`

## 数据链路

原始数据在：

`data/g05_rl_raw/so101_g05_rl_pick_white_v1`

运行脚本时会先准备成 G0.5 训练使用的 model frame：

`data/g05_rl_prepared/so101_g05_rl_pick_white_v1`

准备阶段会复用 SO101 的关节坐标转换：

\[
q_{\mathrm{model}} = [1,-1,1,1,1,1] \cdot q_{\mathrm{raw}} + [0,90,90,0,0,0]
\]

坏 episode `20260701_114422_ep00003` 不会被物理删除，只在 label、manifest、pair 构建阶段过滤。

当前 label / pair 统计：

- raw labels: 53
- 过滤坏 episode 后 labels: 52
- success episodes: 27
- failure episodes: 25
- 原始 pair 数: 85
- 坏 episode 不出现在 success manifest 或 pairs

训练实例化时还发现 3 个尾部 episode 在当前 LeRobot dataset 里没有可索引训练帧：

- `20260701_130619_ep00009`
- `20260701_130619_ep00010`
- `20260701_130619_ep00011`

所以实际训练样本是：

- DPO: 78 trainable pairs
- SFT: 26 trainable success episodes，共 4605 frames

这个不是额外删除 raw 数据，而是 wrapper 防止训练时访问不可索引帧。

## 算法

DPO 使用 trajectory-level preference pair，但优化目标只看 AR action-token likelihood，不把 prompt/static text 算入 preference loss。

DPO loss 是：

\[
L_{\mathrm{DPO}}
=
-\log \sigma
\left(
\beta
[
(\log \pi_\theta(y^+) - \log \pi_{\mathrm{ref}}(y^+))
-
(\log \pi_\theta(y^-) - \log \pi_{\mathrm{ref}}(y^-))
]
\right)
\]

最终训练 loss：

\[
L = L_{\mathrm{DPO}} + 0.2 \cdot L_{\mathrm{CE}}(y^+)
\]

其中 `0.2 * CE(chosen)` 用来防止模型在做 preference alignment 时把成功轨迹本身的 action-token 概率压低。

为了适配 A100 40GB，DPO 训练使用 memory-light backward：先 no-grad 算当前 margin 和 DPO 梯度系数，再分别对 chosen / rejected 做两次小 forward/backward。这和当前参数点上的 DPO 一阶梯度等价，但不会同时保留 chosen 和 rejected 两张大图。

当前默认还做了两个稳定性取舍：

- `ANCHOR_COUNT=1`：每个 pair 默认取 1 个固定 anchor。
- DPO 分支关闭 activation checkpoint：避免 no-grad 预计算和 checkpoint recompute 的 metadata mismatch。

如果以后换更大显存或改成更细的 backward，可以再试：

```bash
ANCHOR_COUNT=2 bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

## 直接运行

先检查数据：

```bash
bash scripts/experiments/so101_ar_rl/check_data.sh
```

单独跑 DPO：

```bash
bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

单独跑 success-only SFT：

```bash
bash scripts/experiments/so101_ar_rl/train_ar_sft_success.sh
```

按顺序后台跑完整实验：

```bash
bash scripts/experiments/so101_ar_rl/run_all_background.sh
```

顺序是先 DPO，DPO 正常结束后再跑 SFT。

查看状态：

```bash
bash scripts/experiments/so101_ar_rl/status.sh
```

看最新日志：

```bash
bash scripts/experiments/so101_ar_rl/tail_log.sh
```

停止后台 supervisor：

```bash
bash scripts/experiments/so101_ar_rl/stop_run.sh
```

## 关键默认参数

DPO:

- `learning_rate=5e-6`
- `beta=0.1`
- `chosen_ce_weight=0.2`
- `ANCHOR_COUNT=1`
- `model.batch_size=1`
- `max_steps=600`
- `checkpointing_steps=200`
- `eval_steps=100000`
- `model.use_pretrained_norm_stats=true`
- `discrete_action=true`
- `continuous_action=false`
- `return_continuous_action=false`

SFT:

- `learning_rate=5e-6`
- `model.batch_size=2`
- `max_steps=600`
- `checkpointing_steps=200`
- `eval_steps=100000`
- `model.use_pretrained_norm_stats=true`
- `discrete_action=true`
- `continuous_action=false`
- `return_continuous_action=false`

## 输出位置

主要输出都在：

`scripts/experiments/so101_ar_rl`

重要文件：

- `cache/pairs.jsonl`
- `cache/pairs.jsonl.summary.json`
- `cache/ref_logps_step9740_anchors1.jsonl`
- `runs/so100/ar_dpo_pairs_<timestamp>/train.log`
- `runs/so100/ar_sft_success_<timestamp>/train.log`
- `latest_dpo_run.txt`
- `latest_sft_run.txt`
- `latest_run.txt`

## 观察指标

DPO 重点看：

- `rl/dpo_loss`
- `rl/chosen_ce_loss`
- `rl/total_loss`
- `rl/reward_margin`
- `rl/reward_accuracy`
- `rl/policy_chosen_logp`
- `rl/policy_rejected_logp`

直觉上，`reward_margin` 越往上越好，`reward_accuracy` 高于 0.5 是积极信号；但最终是否更稳定，必须用真机 rollout 评估，不能只看训练集日志。

SFT 重点看：

- `ce_loss`
- checkpoint 是否正常写出
- 后续真机加载后动作是否稳定

## 已验证

已经通过：

- `bash -n scripts/experiments/so101_ar_rl/*.sh`
- Python import/compile 检查
- `git diff --check`
- `bash scripts/experiments/so101_ar_rl/check_data.sh`
- `bash scripts/experiments/so101_ar_rl/smoke_test.sh`

smoke test 中 DPO 和 SFT 都完成了 2 step，并写出了 checkpoint。
