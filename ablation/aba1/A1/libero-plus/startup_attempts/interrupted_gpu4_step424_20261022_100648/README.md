# A1 LIBERO-Plus：GPU 4，默认六进程

本次执行用户最新要求：只用物理 GPU 4，六个进程同步训练一个 Local-only 模型。Stage I 训练5000次 optimizer update，结束后自动以其 step_005000 权重初始化 Stage II，再训练2500次 update。

每进程 micro-batch=1，六个 rank 每次更新分别累积 [6,6,5,5,5,5] 个样本，合计32。每个样本的 loss 在 backward 前乘 world_size/32；Gloo 将梯度跨 rank 求平均后，得到32样本均值。不是六次独立实验。

## 模型与数据

Stage I 从 config.json 指定的原始 LaWAM LIBERO SFT 权重初始化。当前输入保留主视角、腕部视角及指令；未来只读取 t+8 的主、腕部两张图，Local Resampler 输出16×2048。唯一损失为 action flow matching loss。

Stage II 从本次教师5000步复制公共及动作分支权重，新增随机 Local query [16,2048]，教师冻结。学生只看当前输入，损失为 action flow loss + Local 蒸馏；在第4、8、12、16层计算 cosine+Gram 并对四层取平均。

两阶段都没有 Goal 分支或 Goal→Local gate。保留 Current/DINO、原8+8查询、4个Plan及完整DiT，Local在16层注入Action。轨迹尾部未来无效时仍保留动作监督，屏蔽教师未来条件及相应蒸馏。

数据根为 LaWAM_without_subgoal/libero-plus/data/indexed_rlds。沿用 seed2026 的历史划分：训练13630段/2125407帧，验证717段/112629帧。验证 loss 是拟合诊断，不是推理成功率。

## 训练和保存

LR=1e-5、warmup=500，保持原16000/10000步余弦调度长度，实际停止为5000/2500步。Stage II 新建优化器、随机数状态和采样器。

- 教师只保存 step_005000，没有 step0 或中间教师权重。
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

- logs/train_gpu4.log：控制台日志。
- 各阶段 metrics.jsonl：训练及验证 loss、真实 step、耗时。
- 各阶段 status_rank*.json、parameter_audit.json、sync_audit.json：进程、参数和同步状态。
- logs/launch_record.json：本次启动记录。
- logs/validation_six_rank.log：11项测试结果，包括六进程加权梯度及CPU累积。
- logs/validation_loader_ipc.log：真实 Local-only 数据加载器 IPC 检查。

tmux 会话为 a1_plus_gpu4；启动期 OOM 守护会话为 a1_plus_gpu4_guard。
不以空闲显存阈值阻止启动。若某阶段首次记录训练更新前出现实际 CUDA OOM，守护程序按用户授权依次降为4/2/1进程，并保持有效 batch32；已产生训练更新后不自动重置优化器或丢弃该训练历程。

初始请求保存在 config.json 的 a1_training 中；实际进程数以每阶段 arguments.json / contract.json 为准。startup_attempts/ 保留此前尚无训练更新的启动尝试。没有改动其他消融组代码或权重。Stage II 完成后自动推理的设置见下文。

## 2026-10-21 非有限梯度修复

首轮六进程在第一次更新前退出，非 OOM。使用相同原始权重、seed 和真实数据的独立反向检查，在 rank0 第二个微批次复现 `ScaledDotProductEfficientAttentionBackward0` 输出 NaN。A1 本地运行时禁用 memory-efficient/CuDNN SDPA，保留 math/Flash；带 mask 的 attention 使用 math 回退。结构、mask、损失、数据与训练预算不变，仍严格检查非有限 loss/gradient。实际后端记录在每阶段 contract.json 的 attention_backends 中。原始失败证据在 logs/diagnose_backward.log 和 startup_attempts/nonfinite_six_before_step1_*。

## Stage II 结束后自动全量评测

用户于 2026-10-21 授权自动衔接评测。`after_training_full_eval.py --execute` 在 tmux 的 `auto-eval` 窗口每10秒监控原训练，不停止或重启训练。只有 pipeline 正常完成、退出码为0、两阶段训练进程退出，且教师 I/step_005000、学生 II/step_002500 的完整标记与权重校验通过，才启动评测。

使用本次自己的 Student2500，在 GPU4 上启动6组独立 policy server / simulator。沿用完整10030任务清单（Spatial2402、Object2518、Goal2591、Long2519），每任务1个episode；chunk8、flow10、固定seed和初态0，不保存视频。学生推理不加载教师或真实未来。基础设施错误按既有协议最多重试3次，普通任务失败不重试；有未完成任务时不报告完整成功率。

A1 的评测入口为 `evaluate.py` / `a1_eval/`，仅将本次正式终点适配到2500步，并沿用训练修复后的 math/Flash SDPA、CPU临时目录 `/tmp`。没有改动共用评测或正在运行的训练代码。手动查看方案可执行 `bash evaluate.sh --scope full --gpus 4 --workers 6`，不带 `--execute` 不启动推理。

- 自动监控状态：`logs/after_training_full_eval_status.json`
- 自动监控日志：`logs/after_training_full_eval.log`
- 正式评测日志：`logs/full_eval.log`
- 校验交接记录：`logs/final_checkpoint_handoff.json`（训练完成后生成）
- 最终结果：`protocol_plus_v1/seed_2026/evaluation/full_II_step_002500/results.json`
- 分类矩阵：同目录 `matrix.csv`；逐任务结果：`episodes/`；错误证据：`errors/`。

`tmux attach -t a1_plus_gpu4` 查看进度；`auto-eval` 窗口显示监控状态，`eval-log` 窗口显示推理日志。首次真实模型/仿真 rollout 在最终学生权重就绪后才执行；当前只验证触发逻辑、全量清单与仿真依赖/任务元数据。
