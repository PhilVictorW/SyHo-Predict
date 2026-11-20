# Stage I 后验教师后训练：2026-09-21

训练根目录：`runs/goal_local/stage_I_20260921/`。

```bash
tmux attach -t lawam_aba2_change_stageI
```

默认进入 `progress` 窗口，每 2 秒刷新进度条、当前优化器更新/梯度累积微步、action loss、最近 100 步均值、EMA、固定验证 loss、相对初始验证变化、ETA 与检查点路径。

按 `Ctrl+b`，松开后按 `d` 可离开，训练保持运行。关闭 VSCode 或 SSH 不会终止 tmux 中的训练。tmux 不能抵御机器重启；这种情况应从已完成检查点恢复。

## 本次设置

- 原模块：严格从原 `lawam_libero_sft_release/final_model/pytorch_model.pt` 加载。
- 新模块：随机初始化两路未来 Resampler、view/horizon embedding、Plan、角色动作投影与 cross-attention；Stage I 没有 Student Q_G/Q_L 或 gate。
- 初始化权重将单独保存在 `initial/`，不覆盖原始 SFT。
- 物理 GPU 5；进程内通过 `CUDA_VISIBLE_DEVICES=5` 映射为 `cuda:0`。按用户最新要求改用 GPU 5；GPU 6 上本次正式训练尚未更新参数时已停止，未停止其他任务。
- 目标 10,000 个**优化器更新步**，micro batch=2，累积 16 次，有效 batch=32；每个样本动作噪声重复 2 次，不将它计作新增数据样本。
- 采用 BF16 前向/反向和 FP32 主权重 AdamW；主权重、momentum、variance 位于 CPU，以降低共享 GPU 显存占用，并保留 BF16 量化间隔以下的权重更新。
- 基础 LR=1e-5，warmup=500 更新步，然后 cosine 衰减；梯度裁剪=1。
- 每 250 更新步，使用固定的 128 个跨任务验证样本、固定随机种子计算 action flow loss；初始 step=0 使用同一验证集合。
- 验证前 16 样本额外比较“正常角色条件”和“关闭 Goal/Local 注入”；两者使用同一噪声与流时间，记录未来条件是否影响动作预测。该诊断没有删除 Plan，不等同于完整 aba2 消融。
- 每 1000 步保存模型与完整优化器状态，保留最近 2 份完整检查点；验证创新低时另存 `best/` 模型权重。`best/` 用于教师选型，不含优化器；恢复训练使用 `step_XXXXXX/trainer.pt`。
- 10,000×32=320,000 个训练 anchor；相当于约 1.2 遍当前训练集，精确比例记录在 `dataset_audit.json`。这是首轮预算，是否延长应根据验证 loss 与未来条件诊断决定，不承诺固定步数必然收敛。

## 数据采样

数据源保持只读：`/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero_merged_no_noops_20hz`。

- 所有训练 episode 中的每一帧都作为候选 anchor，按帧均匀 shuffle；不因缺失远期 Goal 丢弃尾部样本。
- 当前：t 的 main+wrist；Local：t+8 的 main+wrist；Goal：t+[24,40,56,72,88] 的 main。
- main/wrist 都按 episode metadata 的视频偏移加逐帧 timestamp 解码。不会把 episode 相对下标误用作拼接视频全局下标。
- Local mask=[B]，Goal frame mask=[B,5]；无效未来不解码、不编码、不用末帧伪造。Stage I 对无效角色关闭整条动作注入残差。
- 动作 a_t…a_(t+7)，归一化后放入 50×32 buffer；监督只包括前 min(8,T−t) 步的前 7 维。
- 按任务分层划分约 5% 验证 episode，与训练 episode 无交叉。验证清单固定保存，禁止从训练 episode 抽相邻帧充当独立验证。
- 注意：初始化 SFT 曾使用该数据集，所以这里的验证用于本次后训练趋势与过拟合诊断，不是从未被原模型见过的独立泛化测试。

## 日志与启动状态

后台脚本先在 `runs/goal_local/preflight_20260921/` 完成真实数据、完整 FP32 主权重 AdamW 更新与保存预检；只有预检成功才从原 SFT **重新初始化**正式训练。预检参数不会作为正式训练起点。

- `train.log`：完整输出与报错。
- `metrics.jsonl`：每个优化器更新和固定验证的 loss，持久保留。
- `status.json`：面板当前状态，原子更新。
- `dataset_audit.json`、`validation_samples.json`、`parameter_audit.json`：可复核的数据、参数与采样记录。
- `initialization.json`：原权重迁移清单。

恢复示例（替换为实际最近完成检查点）：

```bash
CUDA_VISIBLE_DEVICES=5 /home/wangchaoyang/envs/LaWAM/bin/python -m goal_local.train \
  --config configs/goal_local/stage_I.json --stage I \
  --output runs/goal_local/stage_I_20260921 --steps 10000 \
  --batch-size 2 --accumulation 16 --workers 4 --cpu-threads 4 \
  --device cuda:0 --eval-batches 64 --diagnostic-batches 8 \
  --save-every 1000 --eval-every 250 --keep-checkpoints 2 --optimizer-device cpu \
  --resume runs/goal_local/stage_I_20260921/step_001000/trainer.pt
```

恢复模型、FP32 主权重、优化器、scheduler 与 RNG；数据迭代器从新一轮 shuffle 开始，不保证逐样本位级重放。
