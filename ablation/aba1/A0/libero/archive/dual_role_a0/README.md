# A0 LIBERO：纯动作监督，W0 直接初始化 Student

保留 Local 8 + Goal 8 槽、第 9–16 层 gate、4 个 Plan 和双路径动作条件。没有 Teacher、Future Resampler 或真实未来图像，只有 action flow loss。默认 500 个完整 epoch。

**当前交付状态：实现和离线检查已完成；未启动正式训练、未启动 LIBERO rollout。** 代码依据 [guide.md](../../guide.md) 和 [abalation.md](../../../../reference/abalation.md)。

## 文件与实现

- `config.json`：本组固定结构、权重路径、数据路径及优化参数。
- `train.py` / `train.sh`：训练入口；支持 `--dry-run` 和完整 `--resume`。
- `evaluate.py` / `evaluate.sh`：quick 400 / full 2000 评估入口，按本组 checkpoint 配置加载。
- `preflight.py` / `preflight.sh`：只读资源和依赖检查，不加载完整模型、不初始化 GPU、不启动环境。
- `preflight_report.json`：本次资源检查报告。
- [libero_common](../../libero_common/README.md)：共用模型、角色 mask、数据、动作头、训练、服务和检查实现。原 `goal_local` 源码未修改。

在本目录执行以下命令。训练和评估命令是后续启动用示例，**本次未执行**。

## 先检查

```bash
bash preflight.sh
bash train.sh --help
bash evaluate.sh --help
```

设置自己选定的 `TRAIN_GPU` 和 `RUN_ID`；A1–A3 还需明确阶段预算。代码不自动挑选 GPU，也不预设 A1–A3 的 epoch 数。

## 正式训练的最后一条命令

```bash
CUDA_VISIBLE_DEVICES="${TRAIN_GPU:?请先设置训练 GPU}" bash train.sh \
  --epochs 500 \
  --output "$PWD/runs/${RUN_ID:?请先设置新 run 名称}/action_only"
```

在命令末尾追加 `--dry-run` 只打印预算，不创建 run、不训练。默认 micro-batch=2、accumulation=16；每 epoch 完整遍历 260726 anchors，尾部 11 个 micro-batches 单独更新，总计 8148 次更新；500 epoch 为 4074000 次更新。

仅在动作 loss 训练满预算后，用 `action_only/final.json` 指向的最终权重推理。`best_flow.json` 是离线 flow 选择的附加候选，不能替代 epoch 500 主结果。

## 完整恢复与输出

原命令追加 `--resume "$RESUME_DIR"`，同时保持原 output、stage、epoch 总预算、batch、seed 和 Teacher 不变。`RESUME_DIR` 必须是同一 output 下 `checkpoints/` 中带完成标记的目录；可从 `latest.json` 读取。恢复包含 FP32 master 参数、Adam、LR 进度、RNG、epoch 和采样游标；不提供会悄悄重置 Adam 的 warm-start 替代。

每个 epoch 结束 flush 尾部梯度、固定离线验证并保存完整状态；可通过 `--save-every-updates N` 增加中途恢复点。默认保留最近两份完整 checkpoint 及 best-flow 对应 checkpoint。中断时如果存在 `.incomplete` 目录，先检查并保留/移走该目录，再从最近完整 checkpoint 恢复；程序不会覆盖不完整文件。

输出含配置、来源哈希、数据划分/版本、参数清单、逐更新指标、验证指标、`latest.json`、`best_flow.json`、`final.json`。失败或中断后 `final.json` 不会伪装成已完成结果。

## 最终 Student 推理

将 `CHECKPOINT_DIR` 设为最终 `final.json` 的 `path`。先 quick，再 full：

```bash
CUDA_VISIBLE_DEVICES="${EVAL_GPU:?请先设置推理 GPU}" \
MUJOCO_EGL_DEVICE_ID="${EGL_GPU:?请先设置 EGL GPU}" \
bash evaluate.sh --checkpoint "${CHECKPOINT_DIR:?请先指定最终 Student}" \
  --protocol quick --output "$PWD/evaluation/${RUN_ID:?}/quick_40x10"

CUDA_VISIBLE_DEVICES="${EVAL_GPU:?}" MUJOCO_EGL_DEVICE_ID="${EGL_GPU:?}" \
bash evaluate.sh --checkpoint "${CHECKPOINT_DIR:?}" \
  --protocol full --output "$PWD/evaluation/${RUN_ID:?}/full_40x50"
```

`EGL_GPU` 是渲染设备索引，应按本机 EGL 映射设置；策略 `--device` 默认使用可见设备中的 `cuda:0`。可追加 `--first-success-video` 保存每任务第一个成功视频。`--dry-run` 仅打印协议，允许最终权重尚未产生，用于启动前检查。

评估使用独立服务和原 LIBERO episode core，40 个任务顺序运行；固定初态、seed、8 步 action chunk、10 次 Euler。不同组和不同 checkpoint 使用独立输出目录，服务不会跨结构复用模型缓存。中断后重复同一命令，保留已完成的成功和失败 episode，仅继续未完成项；基础设施错误在日志中保留，不算成功。

Quick 使用初态 0–9，full 使用 0–49，两者重叠。若根据 quick 调参或选模型，需要另定独立最终评估集合。A 组没有 Stage III Actor 或在线更新。

## 正式比较前还需明确

A0 已指定 500 epoch，A1–A3 阶段预算尚未指定。若主表按 500 epoch 的离线 anchor 总预算比较，应令各单/双尺度 Teacher 的 E_I 一致，各 Student 的 E_II 一致，且 E_I+E_II=500。示例 167/333 未被自动采用。历史 Teacher/Student 结果需另列协议，不能直接标为新的等预算消融结果。

本次检查覆盖小模型 CPU 前向/反向、角色隔离、数据采样、checkpoint、恢复、命令行和资源路径；未加载完整 W0 做 GPU 前向，也未运行真实 LIBERO rollout。因此没有新成功率或实测显存/吞吐数据。
