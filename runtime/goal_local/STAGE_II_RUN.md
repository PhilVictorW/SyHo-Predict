# Stage II：固定 Stage-I 第 2000 步教师，训练独立学生

后台会话：

```bash
tmux attach -t lawam_aba2_change_stageII
```

默认显示 `progress` 窗口，每 2 秒刷新。退出查看：`Ctrl+b` 后按 `d`。训练日志与状态位于 `runs/goal_local/stage_II_teacher2000_20260921/`。

## 初始化与冻结

- 原始教师来源：`runs/goal_local/stage_I_20260921/step_002000/`。
- 固定教师快照：`runs/goal_local/teachers/stage_I_step_002000/`。模型文件通过硬链接固定，同一内容不重复占磁盘空间，processor 单独复制；Stage I 清理旧检查点不会删除这份快照。
- 教师完整后验链固定：当前 embedding、原 action/flow queries、Qwen 16 层、Future Resamplers、view/horizon embedding 全部 `requires_grad=false`、eval、no_grad；不执行教师动作头。
- 学生复制该教师检查点中的公共模块与动作头，Q_G/Q_L 和深层 gate 新初始化。师生参数存储独立。
- 只有学生可训练参数进入优化器。每步检查教师无梯度、参数版本不变、始终 eval；结果写入 `teacher_freeze_audit.json`。
- 原 Stage I 继续原任务。Stage II 不追踪其后续检查点，不用 EMA 更新教师。

## 损失与数据

```text
loss = flow + 1.0 * local_distill + 1.0 * goal_distill
```

Local/Goal 各在第 4/8/12/16 层进行 cosine + Gram 蒸馏，四层等权；Gram 系数 1。无效未来只屏蔽相应教师蒸馏项，不关闭学生角色、gate 或动作条件。

采样与 Stage I 相同：当前双视角；Local=t+8 双视角；Goal=t+[24,40,56,72,88] 主视角。无效未来不解码，不用最后帧伪造；动作尾部 min(8,T−t)×7 有效，其余 mask=false。训练与验证按 episode 隔离。

## 本次运行参数

- 物理 GPU 0，`CUDA_VISIBLE_DEVICES=0`，进程内为 `cuda:0`。
- 目标 10,000 优化器更新步；micro batch=2，梯度累积=16，有效 batch=32。
- LR=1e−5，warmup=500，cosine 衰减，clip=1；BF16 计算、FP32 主权重 AdamW。
- FP32 主权重与优化器状态卸载到 CPU，模型前反向位于 GPU 0。首次在 GPU 1 的尝试因其他进程新增约 30 GiB 占用，在第一个优化器更新分配状态时 OOM，未完成学生更新；失败记录单独保留在 `_attempt1_gpu1/`。正式重启使用相同教师与随机种子。该放置不改变架构或损失。
- 每 250 步用固定 128 个跨任务样本验证，分别记录 flow/local_distill/goal_distill/total。
- 每 1000 步保存完整恢复检查点，保留最近 2 份；action validation loss 创新低时另存 best 模型。
- 保存 `initial/` 学生起点；部署用保存的学生，不需要教师文件。

`teacher_source.json` 记录实际读到的源步数与路径，`parameter_audit.json` 记录优化器参数数量，`metrics.jsonl` 持续保存分项损失。

## 恢复示例

先确认没有同一训练任务仍在运行，然后在 tmux 中使用实际最近完整检查点替换下列 step：

```bash
CUDA_VISIBLE_DEVICES=0 /home/wangchaoyang/envs/LaWAM/bin/python -m goal_local.train \
  --config configs/goal_local/stage_II.json --stage II \
  --stage-one runs/goal_local/teachers/stage_I_step_002000 \
  --output runs/goal_local/stage_II_teacher2000_20260921 \
  --steps 10000 --batch-size 2 --accumulation 16 --workers 4 --cpu-threads 4 \
  --device cuda:0 --eval-batches 64 --diagnostic-batches 0 \
  --save-every 1000 --eval-every 250 --keep-checkpoints 2 --optimizer-device cpu \
  --resume runs/goal_local/stage_II_teacher2000_20260921/step_001000/trainer.pt
```

恢复时仍须使用同一固定教师快照。模型、FP32 主权重、优化器、scheduler 和 RNG 会恢复，数据迭代器重新建立，不保证逐样本位级重放。
