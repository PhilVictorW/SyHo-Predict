# A1 LIBERO-Plus：GPU 2，默认六进程

本次执行用户最新要求：只用物理 GPU 2，六个进程同步训练一个 Local-only 模型。Stage I 训练3000次 optimizer update，结束后自动以其 step_003000 权重初始化 Stage II，再训练2500次 update。

每进程 micro-batch=1，六个 rank 每次更新分别累积 [6,6,5,5,5,5] 个样本，合计32。每个样本的 loss 在 backward 前乘 world_size/32；Gloo 将梯度跨 rank 求平均后，得到32样本均值。不是六次独立实验。

## 模型与数据

Stage I 从 config.json 指定的原始 LaWAM LIBERO SFT 权重初始化。当前输入保留主视角、腕部视角及指令；未来只读取 t+8 的主、腕部两张图，Local Resampler 输出16×2048。唯一损失为 action flow matching loss。

Stage II 从本次教师3000步复制公共及动作分支权重，新增随机 Local query [16,2048]，教师冻结。学生只看当前输入，损失为 action flow loss + Local 蒸馏；在第4、8、12、16层计算 cosine+Gram 并对四层取平均。

两阶段都没有 Goal 分支或 Goal→Local gate。保留 Current/DINO、原8+8查询、4个Plan及完整DiT，Local在16层注入Action。轨迹尾部未来无效时仍保留动作监督，屏蔽教师未来条件及相应蒸馏。

数据根为 LaWAM_without_subgoal/libero-plus/data/indexed_rlds。沿用 seed2026 的历史划分：训练13630段/2125407帧，验证717段/112629帧。验证 loss 是拟合诊断，不是推理成功率。

## 训练和保存

LR=1e-5、warmup=500，保持原16000/10000步余弦调度长度，实际停止为3000/2500步。Stage II 新建优化器、随机数状态和采样器。

- 教师只保存 step_003000，没有 step0 或中间教师权重。
- 学生每500步保存，只保留最新两份；正常结束后为 step_002000、step_002500。
- 保存模型、processor、动作统计和校验元数据，不保存 Adam/FP32 master 恢复状态，因此不能用这些模型快照精确恢复中断的优化器状态。
- Stage II 的教师、学生只共享数值完全相同且永久冻结的 Qwen visual 和 DINO 参数；可训练参数独立。

Gloo 只在 CPU 同步梯度，rank0 维护 FP32 master AdamW。累积梯度通过 post-accumulate hook 即时转存 CPU；启用激活检查点与 saved-tensor CPU offload，关闭 autocast 权重缓存，以降低显存峰值。计算精度仍为原 BF16/FP32 协议。

数据加载器 IPC 临时目录使用 /tmp，避免工作目录过深导致 AF_UNIX path too long。权重、配置、日志及结果都留在本目录。

## 入口与状态

不带 --execute 只打印方案：

    bash start_training.sh
    bash start_training.sh --execute

输出为 protocol_plus_v1/seed_2026/I/ 和 II/。主要文件：

- logs/train_gpu2.log：控制台日志。
- 各阶段 metrics.jsonl：训练及验证 loss、真实 step、耗时。
- 各阶段 status_rank*.json、parameter_audit.json、sync_audit.json：进程、参数和同步状态。
- logs/launch_record.json：本次启动记录。
- logs/validation_gpu2_restart.log：本次7项验证全部通过，包括六进程加权梯度、3000步教师到学生初始化及全量评测触发。
- startup_attempts/interrupted_gpu4_step424_20261022_100648/logs/validation_loader_ipc.log：此前通过的真实 Local-only 数据加载器 IPC 检查。

2026-10-22 14:24 已按用户要求改用 `nohup` 后台运行，并脱离终端会话。训练启动器 PID 为3908190，自动评测监控 PID 为3908191；进程号同时保存在 `logs/nohup_training.pid` 和 `logs/nohup_auto_eval.pid`，启动记录为 `logs/launch_record.json`。`tmux` 会话 `a1_plus_gpu2` 仅用于查看进度和日志。本次固定6进程，不启用旧的自动降进程数 OOM 守护。

当前请求保存在 config.json 的 a1_training 中；实际进程数以每阶段 arguments.json / contract.json 为准。startup_attempts/ 保留此前尚无训练更新的启动尝试。没有改动其他消融组代码或权重。Stage II 完成后自动推理的设置见下文。

## 2026-10-21 非有限梯度修复

首轮六进程在第一次更新前退出，非 OOM。使用相同原始权重、seed 和真实数据的独立反向检查，在 rank0 第二个微批次复现 `ScaledDotProductEfficientAttentionBackward0` 输出 NaN。A1 本地运行时禁用 memory-efficient/CuDNN SDPA，保留 math/Flash；带 mask 的 attention 使用 math 回退。结构、mask、损失、数据与训练预算不变，仍严格检查非有限 loss/gradient。实际后端记录在每阶段 contract.json 的 attention_backends 中。原始失败证据在 logs/diagnose_backward.log 和 startup_attempts/nonfinite_six_before_step1_*。

## Stage II 结束后自动全量评测

用户于 2026-10-21 授权自动衔接评测。`after_training_full_eval.py --execute` 通过 `nohup` 后台运行，每10秒监控训练，不停止或重启训练。只有 pipeline 正常完成、退出码为0、两阶段训练进程退出，且教师 I/step_003000、学生 II/step_002500 的完整标记与权重校验通过，才启动评测。

使用本次自己的 Student2500，在 GPU2 上启动6组独立 policy server / simulator。沿用完整10030任务清单（Spatial2402、Object2518、Goal2591、Long2519），每任务1个episode；chunk8、flow10、固定seed和初态0，不保存视频。学生推理不加载教师或真实未来。基础设施错误按既有协议最多重试3次，普通任务失败不重试；有未完成任务时不报告完整成功率。

A1 的评测入口为 `evaluate.py` / `a1_eval/`，仅将本次正式终点适配到2500步，并沿用训练修复后的 math/Flash SDPA、CPU临时目录 `/tmp`。没有改动共用评测或正在运行的训练代码。手动查看方案可执行 `bash evaluate.sh --scope full --gpus 2 --workers 6`，不带 `--execute` 不启动推理。

- 自动监控状态：`logs/after_training_full_eval_status.json`
- 自动监控日志：`logs/after_training_full_eval.log`
- 正式评测日志：`logs/full_eval.log`
- 校验交接记录：`logs/final_checkpoint_handoff.json`（训练完成后生成）
- 最终结果：`protocol_plus_v1/seed_2026/evaluation/full_II_step_002500/results.json`
- 分类矩阵：同目录 `matrix.csv`；逐任务结果：`episodes/`；错误证据：`errors/`。

`python -B progress.py --once` 查看当前进度；`tail -f logs/train_gpu2.log` 查看训练日志，`tail -f logs/nohup_auto_eval.log` 查看自动评测监控输出。也可通过 `tmux attach -t a1_plus_gpu2` 查看已有监控窗口。首次真实模型/仿真 rollout 在最终学生权重就绪后才执行；当前只验证触发逻辑、全量清单与仿真依赖/任务元数据。

## 2026-10-22 中断后重新开始

原 GPU4 训练最后完成424步（01:38:57），之后 Gloo collective 超时，于05:00退出。没有中间教师模型可恢复。旧配置、代码、日志及训练状态完整归档至 `startup_attempts/interrupted_gpu4_step424_20261022_100648/`。本轮从原始 baseline 重新初始化，在物理GPU2使用6进程，Stage I 3000步，Stage II 2500步；有效batch32及原学习率调度保持。教师保存为 `I/step_003000`，学生从该教师初始化，最终评测校验累计步数5500。当时的重启记录已随该轮日志归档到 `startup_attempts/gpu2_no_updates_before_nohup_20261022_142324/logs/restart_record.json`。

## 2026-10-22 nohup 后台重启

14:23 检查时，上一轮GPU2训练进程已不存在，尚无optimizer update或教师checkpoint；退出原因没有可靠日志可确认。按用户指令，原运行目录及日志归档至 `startup_attempts/gpu2_no_updates_before_nohup_20261022_142324/`，14:24 使用 `nohup` 重新启动。训练与自动评测监控各自建立独立session，stdin为 `/dev/null`，忽略SIGHUP；实际控制台输出为 `logs/train_gpu2.log`，启动器输出为 `logs/nohup_gpu2.log`。当前阶段预算仍为I 3000 / II 2500，物理GPU2上6个rank。此次归档记录为 `logs/nohup_restart_record.json`，启动核验为 `logs/nohup_verification.json`。
