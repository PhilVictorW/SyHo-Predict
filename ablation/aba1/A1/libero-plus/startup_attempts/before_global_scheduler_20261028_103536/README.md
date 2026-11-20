# A1 LIBERO-Plus：Stage II 在 GPU0 上按实测容量并发

Stage I 已于2026-10-28 05:02完成3000步，最终教师 `I/step_003000` 完整保留。原六卡自动续接的 Stage II 在05:09加载时因物理GPU6显存不足退出，未完成任何训练更新。失败记录归档于 `startup_attempts/stageII_gpu6_oom_before_gpu0_20261028_100314/`。

按最新要求，Stage II 改用GPU0：先以单进程完成验证及两个真实optimizer更新，测量加载、反向传播和优化器更新的显存峰值；实测记录位于 `capacity_profiles/`，不计入正式2500步。根据实测峰值、GPU可用显存和主机可用内存确定正式并发数，保留显存余量并用文件锁逐个加载。正式进程共同训练一个模型，有效batch始终为32；每个rank的累积次数为 `len(range(rank,32,world_size))`。

模型、数据、loss、学习率调度、教师配置和每50步保存一次的机制保持。GPU位置和进程数通过Stage II运行参数传入，与checkpoint中的模型配置分开记录。正式进程数及显存预算见 `logs/gpu0_capacity_selection.json`（实测完成后生成）。Stage II成功完成2500步并校验后，自动使用原配置GPU0、1、2、3、5、6的六个推理worker完成LIBERO-Plus全量10030个任务。

## 模型与数据

Stage I 最初从 config.json 指定的原始 LaWAM LIBERO SFT 权重初始化；当前恢复第150步模型、优化器以及六个rank的随机数和数据游标。当前输入保留主视角、腕部视角及指令；未来只读取 t+8 的主、腕部两张图，Local Resampler 输出16×2048。唯一损失为 action flow matching loss。

Stage II 从本次教师3000步复制公共及动作分支权重，新增随机 Local query [16,2048]，教师冻结。学生只看当前输入，损失为 action flow loss + Local 蒸馏；在第4、8、12、16层计算 cosine+Gram 并对四层取平均。

两阶段都没有 Goal 分支或 Goal→Local gate。保留 Current/DINO、原8+8查询、4个Plan及完整DiT，Local在16层注入Action。轨迹尾部未来无效时仍保留动作监督，屏蔽教师未来条件及相应蒸馏。

数据根为 LaWAM_without_subgoal/libero-plus/data/indexed_rlds。沿用 seed2026 的历史划分：训练13630段/2125407帧，验证717段/112629帧。验证 loss 是拟合诊断，不是推理成功率。

## 训练和保存

LR=1e-5、warmup=500，保持原16000/10000步余弦调度长度，实际停止为3000/2500步。Stage II 新建优化器、随机数状态和采样器。

- Stage I 与 Stage II 均每50个完整 optimizer update 保存一次，各阶段只保留最新一份：50 → 100 → 150。
- 先写入临时目录，保存模型、processor、动作统计、FP32 master/AdamW、学习率调度及各rank的随机数/采样游标；文件落盘并通过SHA256校验后原子发布，再删除本阶段旧 checkpoint。保存失败保留上一份完整 checkpoint。
- 正常结束保留教师 `I/step_003000` 和学生 `II/step_002500` 各一份。学生滚动保存不会删除教师。每次保存期间需要同时容纳旧、新两份，落盘前检查模型和优化器实际大小。
- `latest.json` 指向最近已完成保存的 checkpoint。完整恢复入口为 `a1_local.training --resume <checkpoint目录>`；恢复会核验原配置、代码、数据、模型来源与六卡映射。
- Stage II 的教师、学生只共享数值完全相同且永久冻结的 Qwen visual 和 DINO 参数；可训练参数独立。

Gloo 只在 CPU 同步梯度，rank0 维护 FP32 master AdamW。累积梯度通过 post-accumulate hook 即时转存 CPU；启用激活检查点与 saved-tensor CPU offload，关闭 autocast 权重缓存，以降低显存峰值。计算精度仍为原 BF16/FP32 协议。

数据加载器 IPC 临时目录使用 /tmp，避免工作目录过深导致 AF_UNIX path too long。权重、配置、日志及结果都留在本目录。

## 当前入口与状态

不带 `--execute` 只打印方案：

    bash start_training.sh
    bash start_training.sh --execute

完整恢复方案预览（不启动）：

    /home/wangchaoyang/envs/LaWAM/bin/python -B resume_stage_i.py --resume protocol_plus_v1/seed_2026/I/step_000150

当前已有后台训练，不要重复执行启动命令。本轮启动器 PID 见 `logs/nohup_training.pid`，实际存活状态请检查 `/proc` 或进度面板。

    /home/wangchaoyang/envs/LaWAM/bin/python -B watch_rank_progress.py --once
    tail -f logs/train_gpu012356.log

- `logs/launch_record.json`：nohup 启动记录，GPU0、1、2、3、5、6，world_size6、每卡一个rank、Stage I 3000。
- `logs/nohup_training.pid`：后台启动器 PID。
- `logs/nohup_gpu012356.log`：启动器输出；`logs/train_gpu012356.log`：训练控制台输出。
- `logs/exit_code.txt`：原 Stage I 启动器的退出码；续接时另存 `logs/stage_I_exit_code.txt`。
- `logs/stage_II_exit_code.txt`：Stage II 独立退出码，只有此文件为0且学生最终权重完整，才允许全量推理。
- `protocol_plus_v1/seed_2026/I/metrics.jsonl`：真实 optimizer update、loss、梯度范数和每步时间。
- 同阶段的 `status_rank*.json`、`contract.json`、`sync_audit.json`：各 rank 状态、运行协议和同步检查。
- 教师最终权重保存在 `protocol_plus_v1/seed_2026/I/step_003000`；训练期间每50步滚动保存一份可恢复 checkpoint。
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

2026-10-23按要求新增每50步完整保存、每阶段保留最新一份。原20:31加载进程在第0步停止，归档于 `startup_attempts/before_save50_step0_20261023_204042`；之前967步没有历史快照，无法补存。

2026-10-23 21:21已主动轮询确认新保存配置下连续完成前6个完整更新，六个rank进入第7步；loss和梯度均有限，第1步同步抽查差异为0。每阶段每50步保存一份完整恢复checkpoint，新checkpoint落盘校验后再删除旧checkpoint。17项相关检查通过；正式训练首份checkpoint将在第50步生成。证据见 `logs/checkpoint_policy_and_training_verified.json` 和 `logs/stage_I_stability_result.json`，训练与自动续接继续通过nohup运行。

2026-10-24 从第150步恢复：上一轮完成第200步计算后，在保存200步期间 rank0 收到 SIGKILL，其余rank随后因连接关闭退出；本次未发现服务器重启，SIGKILL来源尚未确认。按用户要求从已完成的 `I/step_000150` 恢复，旧日志及未发布的200步文件归档于 `startup_attempts/resume150_after_sigkill_20261024_101041/`。10:11 启动六卡各一个rank，使用文件锁错峰加载；Stage I → Stage II → 全量推理续接及每50步保留最新一份的配置保持。恢复入口为 `run_logged.sh --resume <完整checkpoint目录>`，由 `resume_stage_i.py` 协调并写入原有续接程序所需状态。启动验证见 `logs/resume_launch_validation.json`、`logs/nohup_verification.json`；本次已于2026-10-24 11:11确认连续完成151–156共6个新增完整更新，六个rank均进入157步；loss与梯度均为有限值，第151步同步抽查差异为0。最终核验见 `logs/resume150_training_verified.json`，主动轮询结束后nohup训练和自动续接继续运行。

## 2026-10-28 GPU0 Stage II 启动记录

- `run_gpu0_profile.py`：nohup单进程显存实测；`logs/gpu0_capacity_profile_status.json` 和 `logs/gpu0_capacity_profile_result.json` 记录状态与最终峰值。
- `run_stage_ii_gpu0.py --capacity-report logs/gpu0_capacity_selection.json`：仅打印正式方案；加 `--execute` 才启动。已有后台任务时不要重复执行。
- 正式训练日志为 `logs/train_stageII_gpu0.log`。继续通过原tmux会话 `a1_gpu012356_progress` 查看，面板读取实际GPU和进程数。
- 正式训练入口持有流水线与续接排他锁；每个进程加载前检查显存下限。最终学生权重校验、独立退出码为0后才允许启动全量推理。
- 历史六卡入口与下方历史记录保留用于溯源，当前执行以本节和最新启动记录为准。
