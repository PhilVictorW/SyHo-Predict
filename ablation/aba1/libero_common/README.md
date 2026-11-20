# A0–A3 LIBERO 实现与交付状态

本目录提供 A0–A3 共用实现；四组各自的 `libero` 目录提供独立配置、训练/推理入口和启动说明。原 `goal_local`、原权重、数据及历史结果均未修改。本次停在正式训练与实验推理之前。

| 实验 | 启动说明 | 结构 |
|---|---|---|
| A0 | [A0/libero/README.md](../A0/libero/README.md) | W0 → current-only；双角色 8+8；有 gate；flow-only；默认 500 epoch |
| A1 | [A1/libero/README.md](../A1/libero/README.md) | 独立 Local-only Teacher → 16 槽 Student；无 Goal |
| A2 | [A2/libero/README.md](../A2/libero/README.md) | 独立 Goal-only Teacher → 16 槽 Student；无 Local |
| A3 | [A3/libero/README.md](../A3/libero/README.md) | 完整 Teacher → 双角色 Student；Stage II 全程无 gate |

`model.py`、`roles.py`、`action.py`、`data.py` 从已审计实现保留公共数值路径并支持结构消融；`training.py` 实现真实 epoch、尾部累积和完整恢复；`server.py` / `evaluation.py` 加载本组配置并复用原 LIBERO episode core；`preflight.py` 仅做只读准备检查。

四层蒸馏、未来缺失规则、动作条件位置、旧 Query、当前输入和视觉冻结范围遵循 guide。单尺度永久删除角色，不通过假 validity 或仅关闭 loss 冒充结构消融。A0 不构造 Teacher；A3 不实例化 gate。共用层保留 A4 完整拓扑以做兼容性检查和将来配对初始化，本次不新增 A4 实验入口。

运行离线检查：

```bash
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
/home/wangchaoyang/envs/LaWAM/bin/python -m unittest ablation.aba1.libero_common.test_contracts -v
```

测试使用 CPU 小模型和合成 batch，不启动正式训练，不加载完整 W0，不运行 LIBERO 环境。它们验证动作梯度、单尺度蒸馏与冻结、A3 隔离、完整拓扑与原代码的一致性、Query/Resampler 形状、未来采样、epoch 覆盖、尾部梯度归一化、优化器/RNG 恢复、checkpoint 读写与形状校验、配对初始化、评估分母。

`upstream_hashes.json` 记录本次开始实现时原 `goal_local/*.py` 的哈希；`validation_report.json` 记录最终检查结果。预检查报告位于各组目录中，不是实验结果。

实现约束：单进程训练，默认 micro-batch=2 / accumulation=16；不支持 DDP。完整推理以 CUDA 为运行目标。A1–A3 必须显式提供阶段 epoch 数，Stage II 必须显式提供对应 Teacher；不自动选择历史权重或推断公平预算。真正的 GPU/W0 smoke 与真实环境 rollout 尚未执行，后续启动前应保留这一区别。
