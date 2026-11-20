# A1 LIBERO-Plus：GPU5，六进程两阶段训练与全量推理

当前在物理 GPU5 上以 nohup 后台运行6个同步训练进程，Stage I 训练3000次 optimizer update。按2026-10-22后续要求，已为正在运行的 Stage I 添加独立后台续接：Stage I 3000步正常结束并校验教师 → Stage II 六进程2500步 → 六个推理 worker 完成 LIBERO-Plus 全量10030个任务。现有 Stage I 没有重启。

每进程 micro-batch=1，六个 rank 每次更新分别累积 [6,6,5,5,5,5] 个样本，有效 batch32；这是六个 rank 同步训练一个 Local-only 模型。训练结构、数据、loss 和学习率调度沿用此前已验证协议。

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

## 当前入口与状态

不带 `--execute` 只打印方案：

    bash start_training.sh
    bash start_training.sh --execute

当前已有后台训练，不要重复执行启动命令。启动器 PID 为360006，实际存活状态请检查 `/proc` 或进度面板。

    /home/wangchaoyang/envs/LaWAM/bin/python -B progress.py --once
    tail -f logs/train_gpu5.log

- `logs/launch_record.json`：nohup 启动记录，GPU5、world_size6、Stage I 3000。
- `logs/nohup_training.pid`：后台启动器 PID。
- `logs/nohup_gpu5.log`：启动器输出；`logs/train_gpu5.log`：训练控制台输出。
- `logs/exit_code.txt`：原 Stage I 启动器的退出码；续接时另存 `logs/stage_I_exit_code.txt`。
- `logs/stage_II_exit_code.txt`：Stage II 独立退出码，只有此文件为0且学生最终权重完整，才允许全量推理。
- `protocol_plus_v1/seed_2026/I/metrics.jsonl`：真实 optimizer update、loss、梯度范数和每步时间。
- 同阶段的 `status_rank*.json`、`contract.json`、`sync_audit.json`：各 rank 状态、运行协议和同步检查。
- 教师最终权重保存在 `protocol_plus_v1/seed_2026/I/step_003000`；不保存中间教师 checkpoint 或 optimizer 恢复状态。
- `logs/restart_gpu5_record.json`：上一轮归档位置及此次重启原因。

训练使用 nohup、独立 session 和 `/dev/null` stdin，可脱离终端运行；服务器重启会结束进程。本轮不启用自动减少 rank 的旧 OOM 守护。

## 重启记录与历史验证

服务器启动时间为2026-10-22 16:07:03 +08。17:15 检查时，上一轮GPU2训练和监控进程均已不存在，旧运行没有记录到optimizer update或教师checkpoint。原日志、seed_2026、配置、启动脚本、本地运行代码及原README已归档到 `startup_attempts/gpu2_server_reboot_before_gpu5_20261022_171641/`。

本轮从原始baseline重新初始化，仅调整运行GPU、Stage I入口和日志路径；pipeline的完成状态按实际执行阶段报告。配置与Python/shell语法检查通过。此前通过的7项验证（六进程加权梯度、3000步教师初始化、全量评测触发等）记录随历史日志保存在 `startup_attempts/`；更早的GPU4运行在424步后因Gloo超时停止，归档也保留在其中。

A1 的数值修复继续生效：禁用 memory-efficient/CuDNN SDPA，保留 math/Flash；保留 loss/gradient 非有限值检查。GPU2两阶段流水线与自动全量评测的历史配置和说明见本轮归档内的README。本轮现在由 `continue_training.py` 接管阶段续接；`after_training_full_eval.py` 会在 Stage II 正常结束后自动启动。

## 本轮启动验证

2026-10-22 18:14 已观察到GPU5六个rank连续完成前5个训练步，loss和梯度范数均为有限值，第1步同步抽查max_difference为0，六个rank已进入第6步。训练继续通过nohup后台运行，目标仍为Stage I 3000步。验证结果为 `logs/stage_I_stability_result.json` 和 `logs/nohup_verification.json`，动态轮询记录为 `logs/stage_I_stability_poll.jsonl`。

## tmux 六进程动态进度面板

执行 `tmux attach -t a1_gpu5_progress` 查看GPU5上实际运行的6个训练rank，每2秒刷新。每行显示PID、已完成步数、micro-batch、距上次进度变化的时间、进程存活状态，以及CPU/I/O是否有活动。推进依据为训练步数、micro或阶段变化；仅CPU/I/O活动不会被当作训练步已推进。超过max(180秒, 最近20步耗时中位数的3倍)无进度变化时提示“长时间无进度”，不直接认定死锁；退出、失败、完成分别显示。

`Ctrl-b d` 脱离，`Ctrl-b n` 在进度、训练日志、自动续接日志之间切换。进度面板底部显示续接程序在线状态和当前阶段；Stage II 开始后自动切换为其2500步进度。面板脚本为 `watch_rank_progress.py`，只读监控，不影响nohup训练；单次查看可用 `/home/wangchaoyang/envs/LaWAM/bin/python -B watch_rank_progress.py --once`。

## 自动训练与推理续接

后台续接程序为 `continue_training.py --execute --poll-seconds 10`，由 nohup 启动。它每10秒读取当前状态；只有 Stage I 最终 checkpoint 完整、3000步教师哈希与配置校验通过、训练进程全部退出且退出码为0，才运行 `run_stage_ii_logged.sh`。Stage II 延用 config.json 的 GPU5、6个 rank、2500步设置，并从本次教师 `I/step_003000` 初始化。

Stage II 退出后使用独立退出码和 `II/step_002500` 完成标记作为交接条件，再启动已有的全量评测程序。评测前再次校验教师/学生文件哈希、3000+2500=5500累计步数及教师来源；评测命令为 `evaluate.py --scope full --checkpoint protocol_plus_v1/seed_2026/II/step_002500 --gpus 5 --workers 6 --execute`。全量范围是清单中的10030个任务，每个任务一次 episode。

- `logs/auto_continuation_status.json`：续接程序PID、心跳、当前阶段、步数/评测计数。
- `logs/auto_continuation_launch.json`、`logs/auto_continuation.pid`：本次后台续接的启动记录。
- `logs/auto_continuation.log`：自动续接日志；tmux 的 `auto-flow` 窗口可直接查看。
- `logs/train_gpu5.log`：两阶段训练输出；`logs/stage_II_teacher_handoff.json`：教师交接校验。
- `logs/after_training_full_eval_status.json`、`logs/full_eval.log`：后续全量推理状态和控制台输出。
- `protocol_plus_v1/seed_2026/evaluation/full_II_step_002500/results.json`：全量推理汇总。

后台续接持有排他锁，重复启动会退出；训练失败、权重不完整或已有中断的 Stage II 时，不会覆盖或跳过错误进入推理。nohup 可跨SSH/终端断联继续运行，服务器重启后需要检查并重新启动。仅查看续接计划可运行 `/home/wangchaoyang/envs/LaWAM/bin/python -B continue_training.py`。
