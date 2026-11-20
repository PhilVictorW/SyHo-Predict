# 实施验证记录（2026-09-20）

本轮已完成新架构实现与连通性验证。**未开展正式训练，未测新架构 LIBERO 成功率；下述 loss 不能解释为任务成功率。**

## 已通过

1. **数据契约**：逐 episode 审核 1693 条演示、273465 帧、40 个任务；真实解码 19 个边界样本。核对未来存在 mask、每路视频偏移、动作尾部有效数。完整结果：`data_audit.json`。
2. **结构测试**：7 项全部通过。涵盖角色 mask 与旧 query 移除、未来缺失 Resampler、masked cosine/Gram、Teacher Goal/Local 隔离、Student gate 梯度与独立冻结教师、动作注入层号、原动作头旁路、尾部 mask。日志：`unit_tests.log`。
3. **原权重加载**：Qwen 493 个 tensor、DINO 211 个 tensor 严格加载；原 Flow 245 个 tensor 加载，仅允许新增角色模块缺失。未加载旧 Q-Former、LAM encoder 或 LaWM decoder。
4. **当前 Qwen 基线对照**：FP32 最大绝对误差 `5.9127808e-05`，通过设定容差。BF16 的扩展序列与 attention kernel 会产生更大舍入差异，不宣称 BF16 逐位相同。保留 aba2 的实际顺序位置编号。
5. **真实全尺寸 Stage I**：flow loss `0.030668`，有限。R_G/R_L、原两组 query、两路角色投影和 Plan 都有非零梯度；冻结 Vision/DINO 没有梯度。
6. **真实全尺寸 Stage II**：学生动作与双路蒸馏 loss 均有限，Q_G/Q_L、relevance 投影、NULL key、gate scale 梯度均非零。学生小步更新后 Teacher 输出逐位一致。
7. **所有未来均缺失**：蒸馏两项恰为 0，学生动作 loss 有限；Student 编码在移除全部未来字段后输出逐位一致。
8. **检查点与部署**：后验保存/重载通过；Student 独立保存/重载后，固定噪声的 10 步 ODE 动作逐位一致，输出 `[1,8,7]` 且有限。Student 无 Future Resampler、Teacher 或旧 LaWM 计算路径。
9. **原代码边界**：复用的 5 个关键 baseline 文件内容一致，本轮逻辑新增在 `goal_local/`。见 `baseline_source_check.json`。

## 真实样本尺寸

```text
原当前序列 L=219
Qwen 输入新增角色后：235 × 2048
移除原 16 个 query 后的当前条件：203 × 2048
DINO：256 × 768
当前 base memory：459 × 768
Goal / Local：各 8 × 2048 → 各 8 × 1024
Plan + 动作 token：54 × 1024
动作 buffer / 速度：50 × 32
推理输出：8 × 7
```

验证使用物理 GPU 2；本次完整 smoke 的 PyTorch 峰值已分配显存约 `12.64 GiB`。这是 batch=1 连通性测试，**不包含完整 AdamW 优化器状态**，不可直接当正式训练显存预算。

## 产物与后续

- `smoke/report.json`：机器可读详细数值和梯度。
- `smoke/posterior/`、`smoke/student/`：只用于检查点往返测试，step=0、`smoke_only=true`；正式 Stage-II 入口拒绝这种 step=0 Teacher。
- 实现与使用方法：`../../IMPLEMENTATION.md`（从本报告位置实际可用链接见下文）。
- 下一步是正式训练新的后验 Teacher，检查验证集与未来条件利用，再执行学生蒸馏与 LIBERO 评测。学习率、蒸馏权重和训练步数尚未通过性能实验调优。

[完整实现说明](../../IMPLEMENTATION.md)

独立命令 `python -m goal_local.predict` 也已通过：仅加载保存的 Student 和当前 main/wrist PNG，成功生成 `predicted_actions.json`（8×7、全部有限），进程已正常退出。
