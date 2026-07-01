# G0.5 SO101 GRAPE-style Window AR-DPO 实验说明

这个目录现在的主实验是：从上一次 SO101 SFT checkpoint 出发，只做 **AR action-token DPO**，并把原来的稀疏单帧 anchor 改成更接近 GRAPE/TPO 的 **滑动窗口轨迹偏好训练**。

当前不再默认跑 SFT 对照；`run_all_background.sh` 只会启动 DPO。

base checkpoint：

`scripts/so101_square_finetune/runs/so100/so101_square_4ep_20260623_205729/checkpoints/step_9740.pt`

## 数据链路

原始数据：

`data/g05_rl_raw/so101_g05_rl_pick_white_v1`

脚本会先准备成 G0.5 训练使用的 model frame：

`data/g05_rl_prepared/so101_g05_rl_pick_white_v1`

准备阶段会复用 SO101 的关节坐标转换：

\[
q_{\mathrm{model}} = [1,-1,1,1,1,1] \cdot q_{\mathrm{raw}} + [0,90,90,0,0,0]
\]

坏 episode `20260701_114422_ep00003` 不会被物理删除，只在 label、manifest、pair 构建阶段过滤。

label / pair 统计：

- raw labels: 53
- 过滤坏 episode 后 labels: 52
- success episodes: 27
- failure episodes: 25
- 原始 pair 数: 85
- 坏 episode 不出现在 pairs

训练实例化时还发现 3 个尾部 episode 在当前 LeRobot dataset 里没有可索引训练帧：

- `20260701_130619_ep00009`
- `20260701_130619_ep00010`
- `20260701_130619_ep00011`

所以实际可训练 base pair 是 78 个。这个不是额外删除 raw 数据，而是 wrapper 防止训练时访问不可索引帧。

## 为什么改成 window

之前的 anchor 方案每个 pair 只拿少量固定帧，确实容易出现你指出的问题：成功轨迹和失败轨迹进度不完全同步时，单帧可能刚好错开关键阶段。

现在的 window 方案是：

- 每个 success/failure pair 仍然要求同 bucket，也就是同一类初始化状态。
- 对每条 episode 生成连续窗口，例如 8 帧一个窗口。
- 一个训练样本不是单帧，而是：

\[
(y^+_{t:t+w-1}, y^-_{t':t'+w-1})
\]

其中 \(w\) 是 `WINDOW_SIZE`。

窗口内的 AR action logprob 会求和：

\[
\log \pi(y_{t:t+w-1}\mid x)
=
\sum_{i=0}^{w-1}
\log \pi(a_{t+i}\mid o_{t+i}, instruction)
\]

这样一条 episode 会被多个重叠窗口覆盖，比单帧 anchor 更能利用整段轨迹。

## DPO 目标

这里只优化 AR action-token likelihood，不把 prompt/static text 算入 preference loss。

DPO loss：

\[
L_{\mathrm{DPO}}
=
-\log \sigma
\left(
\beta
\left[
(\log \pi_\theta(y^+) - \log \pi_{\mathrm{ref}}(y^+))
-
(\log \pi_\theta(y^-) - \log \pi_{\mathrm{ref}}(y^-))
\right]
\right)
\]

最终训练 loss：

\[
L = L_{\mathrm{DPO}} + 0.2 \cdot L_{\mathrm{CE}}(y^+)
\]

其中 `0.2 * CE(chosen)` 用来防止模型在做 preference alignment 时把成功轨迹本身的 action-token 概率压低。

## 和 GRAPE/TPO 的关系

GRAPE/TPO 的核心思路是 trajectory-level preference alignment：用成功轨迹作为 chosen，用失败轨迹作为 rejected，并在一致任务/一致初始化状态下比较轨迹片段。

当前实现没有直接搬 GRAPE 的 OpenVLA/RLDS 训练框架，而是在 G0.5 现有 LeRobot/G05Policy 训练栈里复现这个思想：

- 同 bucket 内 success/failure 全交叉配对。
- 轨迹 pair 展开成滑动窗口 pair。
- 对窗口内多个 AR action chunk 的 logprob 求和后做 DPO。
- 使用 reference checkpoint 的 cached logprob，避免同时加载两份 G0.5。

## 显存处理

`WINDOW_SIZE=8` 时，一个样本会包含 chosen 8 帧和 rejected 8 帧。A100 40GB 上直接整窗反传会 OOM。

现在训练用 memory-light backward：

1. 先 no-grad 算完整窗口的 DPO margin 和一阶梯度系数。
2. 再对 chosen / rejected 分侧反传。
3. 每侧窗口内部用 `LOGP_MICRO_BATCH_SIZE=1` 分块 backward。

因为 surrogate loss 对 action logprob 是线性的，窗口内分块反传和整窗反传在当前参数点上的一阶梯度等价；它只改变显存调度，不改变 DPO 目标。

## 直接运行

检查数据：

```bash
bash scripts/experiments/so101_ar_rl/check_data.sh
```

前台跑 DPO：

```bash
bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

后台跑 DPO：

```bash
bash scripts/experiments/so101_ar_rl/run_all_background.sh
```

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

DPO 默认：

- `DPO_SAMPLE_MODE=window`
- `WINDOW_SIZE=8`
- `WINDOW_STRIDE=4`
- `WINDOW_ALIGN=relative`
- `MAX_WINDOWS_PER_PAIR=0`
- `LOGP_MICRO_BATCH_SIZE=1`
- `learning_rate=5e-6`
- `beta=0.1`
- `chosen_ce_weight=0.2`
- `model.batch_size=1`
- `max_steps=600`
- `checkpointing_steps=200`
- `eval_steps=100000`
- `model.use_pretrained_norm_stats=true`
- `discrete_action=true`
- `continuous_action=false`
- `return_continuous_action=false`

`WINDOW_STRIDE=4` 表示每 4 帧取一个长度为 8 的重叠窗口。它会覆盖整段 episode，但不会像 `stride=1` 那样生成所有可能窗口。

如果想最接近 GRAPE 的 all sliding windows，可以运行：

```bash
WINDOW_STRIDE=1 bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

但 `stride=1` 训练样本会从约 2.6k 增到约 10k，reference cache 和训练都会明显更慢。

## 输出位置

主要输出都在：

`scripts/experiments/so101_ar_rl`

重要文件：

- `cache/pairs.jsonl`
- `cache/pairs.jsonl.summary.json`
- `cache/ref_logps_step9740_window8_stride4_relative_max0.jsonl`
- `runs/so100/ar_dpo_window_<timestamp>/train.log`
- `latest_dpo_run.txt`
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

## 已验证

已经通过：

- `bash -n scripts/experiments/so101_ar_rl/train_ar_dpo.sh`
- `bash -n scripts/experiments/so101_ar_rl/cache_ref_logps.sh`
- Python compile 检查
- `git diff --check`
- window-DPO 2 step smoke training

本次 smoke 使用：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 LOGGER_MODE=offline \
RUN_NAME=smoke_window_dpo_micro_<timestamp> \
MAX_STEPS=2 CHECKPOINTING_STEPS=100000 \
DPO_SAMPLE_MODE=window WINDOW_SIZE=8 WINDOW_STRIDE=4 \
WINDOW_ALIGN=relative MAX_WINDOWS_PER_PAIR=1 \
LOGP_MICRO_BATCH_SIZE=1 \
bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

结果：成功完成 2 step，并写出 `checkpoints/step_2.pt`。
