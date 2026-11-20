实时进度窗口：`tmux attach -t a1_libero_gpu013`，每 5 秒刷新。窗口 0 显示教师/学生进度、GPU0/1/3 每个 rank 的样本进度、存活/心跳/推进状态、错峰加载阶段和显存；窗口 1 显示训练日志，窗口 2 显示自动推理日志。按 Ctrl+b 再按 d 可离开，后台任务继续运行。

当前运行：`local16_teacher1000_student2000_gpu013_stagger_nohup_20261023_004550`。按 2026-10-23 最新要求，GPU0、1、3 各运行一个同步训练 rank（总计 3 个），nohup 后台启动，effective batch=32，rank 样本份额为 [11,11,10]。教师 1000 步完成并保存后自动启动学生 2000 步，再由后台 watcher 在 GPU0 执行 40 tasks × 50 episodes 全量推理。三卡布局保持固定，不自动减少 rank。按 GPU0 → 1 → 3 顺序错峰加载，前一个模型完成加载后间隔 10 秒再加载下一个；三个模型全部就绪后建立 Gloo 进程组并同步训练。Stage II 使用同样的错峰加载。启动信息与日志入口：`nohup/current.json`。

# A1 LIBERO：Local-only，教师 1000 步 → 学生 2000 步

上次 GPU0/1/3 运行因服务器重启而中断。本次不从中断状态恢复，使用新的运行目录，从原始 LaWAM 权重重新开始 Stage I。

本目录使用 `config.json` 定义模型、`run_plan.json` 定义本次步数与运行参数。原始 LaWAM 权重与数据绝对路径见 config.json。

- Stage I：从原始 LaWAM LIBERO SFT 权重构建 Local-only 后验教师；未来仅采样同一 episode 的 t+8 主视角/腕部视角，现有冻结视觉编码器与 Resampler 输出 16 个 Local token。仅 action flow matching loss；保留现有 DINO、Plan 及 DiT 结构，没有 Goal 分支或两组 query 的交互。尾部缺失未来时使用有效性掩码，不越界、不筛掉整个样本。
- Stage II：只继承本次 Stage I 第 1000 步教师的共享网络和动作头权重，新建随机的 16×2048 Local query。教师独立冻结；学生优化 action flow matching + 第 4/8/12/16 层 Local 蒸馏（cosine + Gram），没有 Goal loss 或跨角色交互。
- GPU0、1、3 各一个同步训练 rank，共同训练一个模型。CPU Gloo 经 lo 回环接口汇总梯度，rank 0 使用 CPU FP32 master AdamW 并广播更新。micro-batch=1，有效 batch=32，三个 rank 每步样本份额为 [11,11,10]，每个 rank 另有 1 个数据读取 worker。三卡运行保持三个 rank；若 OOM 则记录失败，不静默减少 GPU 数。
- 教师 1000 个 optimizer steps，仅导出 step_001000；随后自动退出教师训练进程并启动学生训练进程。学生 2000 个 optimizer steps，每 500 步导出，保留最新两个，最终为 step_001500 和 step_002000。不导出初始化或 best 权重。
- 保存的是模型、processor、统计量和来源记录，不保存 Adam 状态；因此不能用这些导出执行精确断点续训。运行目录已存在时拒绝覆盖。
- 峰值学习率 1e-5，Stage I/II warmup 分别为 100/200 步，余弦下降到 5%；实际预算以 run_plan.json 为准。每 250 步对固定的 128 个验证 anchor 计算损失，不额外导出权重。
- 训练/验证按 episode 分离，split_seed=2026；训练 260726 个 anchor。本次教师消费 32000 个、学生消费 64000 个打乱后的训练 anchor。

运行 `bash launch.sh --dry-run` 显示配置与资源检查。当前 nohup 启动进程、PID、启动时间及日志路径记录在 `nohup/current.json`；训练从 `launch.py --execute` 启动，`queue_status.json` 标记实际阶段和进程数。

运行输出：`runs/local16_teacher1000_student2000_gpu013_stagger_nohup_20261023_004550/`。`pipeline.json` 标记阶段，`stage_I/status.json`、`stage_II/status.json` 标记已完成步数，`metrics.jsonl` 保存训练/验证损失，`heartbeat_*.json` 标记各 rank 当前 micro-step，`logs/` 保存各 rank 日志。nohup 控制目录含 `training.log`、`progress.log` 和 `auto_eval.log`。

检查记录位于 `checks/`，包括采样分片、学习率与 checkpoint 保留规则、自动评估条件、GPU 映射和三卡 CUDA 梯度更新与全 batch AdamW 的数值等价性。资源阈值保持不强制等待，GPU 和主机资源仅作记录。

训练后全量推理由独立 nohup 进程 `watch_full_eval.py --execute` 负责。它每 5 秒读取两阶段进度，只有训练 launcher 标记整体完成、Stage I=1000 / Stage II=2000、最终权重与 SHA256 校验均通过后，才在 GPU0 启动 `evaluate.py --protocol full`。run_plan.json 中 automatic_evaluation=false 仅描述训练 launcher；外部自动评测任务以 `auto_eval/<run_name>/plan.json` 为准。

全量协议是 libero_spatial / libero_object / libero_goal / libero_10 各 10 个任务，每个任务 50 回合，共 2000 回合。仅加载学生 step_002000，不加载教师、不输入未来图像。评测按任务顺序执行，遇到错误最多重试 3 次并复用已完成的 episode 记录；无视频输出。结果目录为 `evaluation/<run_name>_full40x50_step2000/`，其中 summary.json 为总结果，task_results.csv 为 40 个任务结果。监控状态、日志分别保存在 `auto_eval/<run_name>/status.json`、`watcher.log` 和 `evaluation.log`。训练尚未完成时不启动任何推理 GPU 进程。

2026-10-22 重启记录：旧运行 `runs/local16_teacher1000_student2000_gpu5x4_20261021/` 保留，最后完成 step 881，step 882 在 Gloo 广播阶段超时，于 05:00 失败退出。旧运行未到 step1000，未导出教师权重。本轮从原始 LaWAM 权重重新训练，不是从 step881 续训。修改前的配置和源码及失败摘要保存在 `checks/relaunch_gpu4x3_20261022/`。新运行在根目录保存固定的 run_plan.json，两个训练阶段都读取该快照；失败状态保留已完成步数。
