# A1 LIBERO：Local-only，教师 1000 步 → 学生 2000 步

本目录使用 `config.json` 定义模型、`run_plan.json` 定义本次步数与运行参数。原始 LaWAM 权重与数据绝对路径见 config.json。

- Stage I：从原始 LaWAM LIBERO SFT 权重构建 Local-only 后验教师；未来仅采样同一 episode 的 t+8 主视角/腕部视角，现有冻结视觉编码器与 Resampler 输出 16 个 Local token。仅 action flow matching loss；保留现有 DINO、Plan 及 DiT 结构，没有 Goal 分支或两组 query 的交互。尾部缺失未来时使用有效性掩码，不越界、不筛掉整个样本。
- Stage II：只继承本次 Stage I 第 1000 步教师的共享网络和动作头权重，新建随机的 16×2048 Local query。教师独立冻结；学生优化 action flow matching + 第 4/8/12/16 层 Local 蒸馏（cosine + Gram），没有 Goal loss 或跨角色交互。
- GPU 5 初始启动四个同步训练 rank，共同训练一个模型；按用户最新授权取消资源阈值等待，实际 CUDA OOM 时依次减少至 3/2/1 rank 重试当前阶段。CPU Gloo 汇总梯度，rank 0 使用 CPU FP32 master AdamW 并广播更新；四个进程均映射到物理 GPU 5。micro-batch=1，有效 batch 始终为 32；4/3/2/1 rank 的样本份额分别为 [8,8,8,8]/[11,11,10]/[16,16]/[32]。每个 rank 另有 1 个数据读取 worker。
- 教师 1000 个 optimizer steps，仅导出 step_001000；随后自动退出教师四进程并启动学生四进程。学生 2000 个 optimizer steps，每 500 步导出，保留最新两个，最终为 step_001500 和 step_002000。不导出初始化或 best 权重。
- 保存的是模型、processor、统计量和来源记录，不保存 Adam 状态；因此不能用这些导出执行精确断点续训。运行目录已存在时拒绝覆盖。
- 峰值学习率 1e-5，Stage I/II warmup 分别为 100/200 步，余弦下降到 5%；实际预算以 run_plan.json 为准。每 250 步对固定的 128 个验证 anchor 计算损失，不额外导出权重。
- 训练/验证按 episode 分离，split_seed=2026；训练 260726 个 anchor。本次教师消费 32000 个、学生消费 64000 个打乱后的训练 anchor。

运行 `bash launch.sh --dry-run` 只显示配置与资源检查；`bash run_authorized.sh` 立即正式运行两阶段，`queue_status.json` 标记实际阶段和进程数。本次用户已授权检查通过后直接启动，并将进程数改为 4。

运行输出：`runs/local16_teacher1000_student2000_gpu5x4_20261021/`。`pipeline.json` 标记阶段，`stage_I/status.json`、`stage_II/status.json` 标记已完成步数，`metrics.jsonl` 保存训练/验证损失，`heartbeat_*.json` 标记各 rank 当前 micro-step，`logs/` 保存四个 rank 的日志。入口日志为 `launcher.log`，tmux 会话为 `a1_libero_gpu5x4`。

已完成的检查记录位于 `checks/`：真实视频采样、单尺度蒸馏与冻结检查、四进程 CPU 梯度等价性、采样分片及 checkpoint 保留规则。正式启动后再由首个真实训练更新验证 CUDA 执行。原显存阈值已按用户最新要求取消，资源数据仅作记录。OOM 的失败尝试移入 failed_attempts/；当前阶段从原始初始化重新计步，Stage II 仍使用已完成的教师 step1000。

训练后全量推理现由独立的 `watch_full_eval.py --execute` 负责，运行在同一 tmux 会话的 `auto-eval` 窗口。它每 5 秒读取两阶段进度，只有训练 launcher 标记整体完成、Stage I=1000 / Stage II=2000、最终权重与 SHA256 校验均通过后，才在 GPU 5 启动现有 `evaluate.py --protocol full`。这不修改当前训练进程、配置或原 launcher；run_plan.json 中的 automatic_evaluation=false 仅描述原训练 launcher，外部自动评测任务以 auto_eval/<run_name>/plan.json 为准。

全量协议是 libero_spatial / libero_object / libero_goal / libero_10 各 10 个任务，每个任务 50 回合，共 2000 回合。仅加载学生 step_002000，不加载教师、不输入未来图像。评测按任务顺序执行，遇到错误最多重试 3 次并复用已完成的 episode 记录；无视频输出。结果目录为 `evaluation/<run_name>_full40x50_step2000/`，其中 summary.json 为总结果，task_results.csv 为 40 个任务结果。监控状态、日志分别保存在 `auto_eval/<run_name>/status.json`、`watcher.log` 和 `evaluation.log`。训练尚未完成时不启动任何推理 GPU 进程。
