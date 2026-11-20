# A1 LIBERO-Plus：GPU0、1、2、3、5、6，每卡一进程

当前按最新要求在物理 GPU0、1、2、3、5、6 上运行，每卡一个同步训练 rank，共6个进程。2026-10-23 20:31通过nohup从baseline重新启动 Stage I，目标3000步；Stage I正常完成并校验后自动续接 Stage II 2500步，再使用同样六张卡各一个推理worker完成 LIBERO-Plus 全量10030个任务。模型通过文件锁依次加载，随后六个rank同步训练。

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

当前已有后台训练，不要重复执行启动命令。本轮启动器 PID 为37625，实际存活状态请检查 `/proc` 或进度面板。

    /home/wangchaoyang/envs/LaWAM/bin/python -B watch_rank_progress.py --once
    tail -f logs/train_gpu012356.log

- `logs/launch_record.json`：nohup 启动记录，GPU0、1、2、3、5、6，world_size6、每卡一个rank、Stage I 3000。
- `logs/nohup_training.pid`：后台启动器 PID。
- `logs/nohup_gpu012356.log`：启动器输出；`logs/train_gpu012356.log`：训练控制台输出。
- `logs/exit_code.txt`：原 Stage I 启动器的退出码；续接时另存 `logs/stage_I_exit_code.txt`。
- `logs/stage_II_exit_code.txt`：Stage II 独立退出码，只有此文件为0且学生最终权重完整，才允许全量推理。
- `protocol_plus_v1/seed_2026/I/metrics.jsonl`：真实 optimizer update、loss、梯度范数和每步时间。
- 同阶段的 `status_rank*.json`、`contract.json`、`sync_audit.json`：各 rank 状态、运行协议和同步检查。
- 教师最终权重保存在 `protocol_plus_v1/seed_2026/I/step_003000`；不保存中间教师 checkpoint 或 optimizer 恢复状态。
- `logs/restart_gpu012356_record.json`：上一轮归档位置及此次重启原因。

训练使用 nohup、独立 session 和 `/dev/null` stdin，可脱离终端运行；服务器重启会结束进程。本轮不启用自动减少 rank 的旧 OOM 守护。

## 重启记录与历史验证

服务器于2026-10-22 22:33重启，之前GPU5运行止于143步，归档于 `startup_attempts/gpu5_reboot_step143_20261022_224943/`。22:51启动的GPU5单卡六进程在用户纠正GPU分配后已停止（仍在加载阶段），完整日志、运行目录和代码归档于 `startup_attempts/gpu5_loading_before_gpu012356_20261022_225526/`。当前六卡运行重新从baseline初始化。

本次已验证六个rank的设备映射为0、1、2、3、5、6，有效batch32、3000/2500步预算保持；11项自动续接与全量评测检查通过。错峰加载锁的CPU进程测试已确认模型加载区间不会重叠。历史数值修复仍生效：禁用memory-efficient/CuDNN SDPA，保留math/Flash，保留非有限loss/gradient检查。早期训练验证与历史日志见对应归档。

## tmux 六进程动态进度面板

执行 `tmux attach -t a1_gpu012356_progress` 查看六张卡上的6个训练rank，每2秒刷新。每行显示物理GPU编号、PID、已完成步数、micro-batch、距上次进度变化的时间、进程存活状态，以及CPU/I/O是否有活动。推进依据为训练步数、micro或阶段变化；仅CPU/I/O活动不会被当作训练步已推进。超过max(180秒, 最近20步耗时中位数的3倍)无进度变化时提示“长时间无进度”，不直接认定死锁；退出、失败、完成分别显示。

`Ctrl-b d` 脱离，`Ctrl-b n` 在进度、训练日志、自动续接日志之间切换。进度面板底部显示续接程序在线状态和当前阶段；Stage II 开始后自动切换为其2500步进度。面板脚本为 `watch_rank_progress.py`，只读监控，不影响nohup训练；单次查看可用 `/home/wangchaoyang/envs/LaWAM/bin/python -B watch_rank_progress.py --once`。

## 自动训练与推理续接

后台续接程序为 `continue_training.py --execute --poll-seconds 10`，由 nohup 启动。它每10秒读取当前状态；只有 Stage I 最终 checkpoint 完整、3000步教师哈希与配置校验通过、训练进程全部退出且退出码为0，才运行 `run_stage_ii_logged.sh`。Stage II 延用 config.json 的 GPU0、1、2、3、5、6各一个rank、2500步设置，并从本次教师 `I/step_003000` 初始化。

Stage II 退出后使用独立退出码和 `II/step_002500` 完成标记作为交接条件，再启动已有的全量评测程序。评测前再次校验教师/学生文件哈希、3000+2500=5500累计步数及教师来源；评测命令为 `evaluate.py --scope full --checkpoint protocol_plus_v1/seed_2026/II/step_002500 --gpus 0,1,2,3,5,6 --workers 6 --execute`。全量范围是清单中的10030个任务，每个任务一次 episode。

- `logs/auto_continuation_status.json`：续接程序PID、心跳、当前阶段、步数/评测计数。
- `logs/auto_continuation_launch.json`、`logs/auto_continuation.pid`：本次后台续接的启动记录。
- `logs/auto_continuation.log`：自动续接日志；tmux 的 `auto-flow` 窗口可直接查看。
- `logs/train_gpu012356.log`：两阶段训练输出；`logs/stage_II_teacher_handoff.json`：教师交接校验。
- `logs/after_training_full_eval_status.json`、`logs/full_eval.log`：后续全量推理状态和控制台输出。
- `protocol_plus_v1/seed_2026/evaluation/full_II_step_002500/results.json`：全量推理汇总。

后台续接持有排他锁，重复启动会退出；训练失败、权重不完整或已有中断的 Stage II 时，不会覆盖或跳过错误进入推理。nohup 可跨SSH/终端断联继续运行，服务器重启后需要检查并重新启动。仅查看续接计划可运行 `/home/wangchaoyang/envs/LaWAM/bin/python -B continue_training.py`。


2026-10-22 23:27 已持续轮询确认本轮六卡训练连续完成前5步，并进入第6步；五步loss与梯度范数均为有限值，第1步同步抽查max_difference=0。六个进程实际分别位于GPU0、1、2、3、5、6，nohup与自动续接程序均存活。完整证据见 `logs/stage_I_stability_result.json`、`logs/stage_I_stability_poll.jsonl`、`logs/nohup_verification.json`。

2026-10-23服务器20:19重启后，本轮20:31重新从baseline启动。此前Stage I训练到967步，但原配置仅在3000步保存教师，且不保存优化器，因此无可恢复checkpoint。旧运行完整归档于 `startup_attempts/gpu012356_reboot_step967_20261023_203142`。本轮将持续主动轮询至少6个完整训练步，目标为Stage I 3000 → Stage II 2500 → 全量10030任务推理。
