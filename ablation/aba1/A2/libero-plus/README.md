# A2 LIBERO-Plus：Stage I GPU6；Stage II GPU0/2/7各一进程、GPU5按实测增加进程

本组独立执行 Goal-only Teacher3000 → Goal-only Student2500 → 全量10030任务推理。所有代码改动、日志、权重、队列、缓存和临时文件保存在本目录。A1与公共核心实现仅只读复用；原脚手架归档在 setup_archive，A1运行文件来源哈希见 setup_source_manifest.json。

## 模型与训练

Stage I 从 config.json 指定的原始 LaWAM LIBERO SFT 权重初始化，只读取 t+[24,40,56,72,88] 五张有序未来主视角，经冻结 Qwen visual、可训练 horizon embedding 与 Goal Resampler 得到16个Goal槽。保留当前主/腕视角、指令、DINO、原查询和4个Plan。Goal仅在DiT第1/5/9/13层注入Plan，再经原self-attention影响Action。没有Local Resampler、Local Query、Local→Action或跨角色gate。

Stage I仅action flow loss，训练3000个optimizer updates。Stage II严格继承自己的 I/step_003000 公共参数与动作分支，新建 Q_Goal[16,2048]。教师完全冻结，学生只看当前输入，loss为flow+Goal蒸馏；第4/8/12/16语言层cosine+Gram四层平均，权重均1。Stage II新建优化器与RNG/采样器，训练2500步。

Stage I八rank共同训练一个模型，物理映射[6,6,6,6,6,6,6,6]。micro-batch=1，每rank累积4次。Stage II先逐个加载并实测前向/反向内存，再选定同步rank数N；每rank累积len(range(rank,32,N))次，总有效batch仍为32，与A1相同。LR1e-5、warmup500，scheduler总长16000/10000，flow repeat2。Stage I/II分别消耗96000/80000个anchor；固定seed2026与历史数据划分。不能与5000/4000的旧完整模型声称等预算。

保留A1的CPU Gloo梯度同步、rank0 FP32 master AdamW、梯度/激活CPU offload、冻结视觉编码器共享和错峰加载。禁用memory-efficient/CuDNN SDPA，启用math/Flash。当前观测、数据和训练目标不因运行布局变化而改动。

## 保存与恢复

Stage I每50步保存；Stage II按用户2026-10-29的新指令改为每100步保存：100、200、300……2500。200完成后删100，300完成后删200，以此类推。先写临时目录，落盘并验证模型/恢复状态SHA256，再原子发布并更新latest.json，最后只删除本阶段旧完整checkpoint。保存失败保留旧文件。完整保存包含processor、动作统计、FP32 master/Adam、调度合同和全部rank的RNG/采样游标。Gloo路径按step计算学习率，不需要有状态scheduler。

最终保留 protocol_plus_v1/seed_2026/I/step_003000 与 II/step_002500。学生滚动保存不删除教师。写入前检查空间；新旧快照同时占用磁盘。教师首份恢复点为step50；本次学生重启首份恢复点为step100，step0不保存。Stage II写盘预检除本次checkpoint与1GiB写盘余量外，额外保留24GiB共享磁盘余量，供同时运行的A1滚动保存，避免A2占满磁盘导致A1下一次保存失败。空间不足时所有rank保持心跳并每30秒重查，保留内存中的更新，空间足够后继续保存，不再因空间预检失败直接退出。其他任务在实际写入时抢占磁盘仍可能导致写入失败。

已完成教师的config.json及a2_local源码保持原样用于严格匹配来源；Stage II实际策略由a2_stage2/runtime.py的CHECKPOINT_POLICY覆盖，写入学生arguments.json、contract.json及checkpoint_complete.json/recovery_retention.json。上次Stage II更新到50但保存前因空间不足退出，无学生checkpoint可恢复；故障记录归档于logs/restart_stage_II_save100_*，本次从同一Teacher3000重新训练2500步。

Stage I恢复入口为 resume_stage_i.py --resume <本阶段完整checkpoint> --execute；新阶段初始化与同阶段断点恢复不同。恢复核对配置、代码、数据和设备布局。

## 启动、续接与监控

启动预览：bash launch_background.sh。明确执行：bash launch_background.sh --execute。已有任务时拒绝重复启动。

nohup后台运行Stage I及continue_training.py。教师3000完整校验且Stage I退出码为0后才启动Stage II；学生2500完整校验且独立退出码为0后才启动全量评测。失败不伪装为完成。nohup可跨终端断联，服务器重启后需检查并恢复。

动态进度：bash watch_progress.sh；单次快照：bash watch_progress.sh --once。

- logs/train_gpu6.log：训练输出。
- logs/auto_continuation_status.json：续接状态与心跳。
- logs/launch_record.json：GPU映射和后台PID。
- protocol_plus_v1/seed_2026/I 或 II：metrics、rank状态、参数与数据审计、latest.json。
- logs/full_eval.log：后续推理日志。

TMPDIR使用/proc/self/cwd/.t这个短别名，实际文件在本目录的.t内；训练和评测子进程cwd均为本目录，避免AF_UNIX路径过长。CUDA、HF、Matplotlib等缓存也位于本目录。

## 全量评测

最终只加载自己的Student2500，无Teacher/未来图像/Actor。GPU6上总共八个policy worker，模型在同一GPU上串行错峰加载，加载完成的worker并行推理；每worker另有CPU模拟器进程。固定10030项，suite数2402/2518/2591/2519，每项1episode，初始状态0、环境seed0、chunk8、flow10步，保持A1夹爪转换与噪声种子规则。

输出 protocol_plus_v1/seed_2026/evaluation/full_II_step_002500/results.json、matrix.csv和逐episode记录。训练未完成前没有本组成功率，不引用历史89.62%。

## 验证记录

logs/preflight.log、preflight_report.json：数据与预算预检。
logs/validate_cpu.log 与 logs/a2_targeted_tests.log：角色路径、梯度、初始化和八rank同步；最初的设备映射测试期望值已修正，以针对性重测为准。
logs/runtime_tests.log：滚动保存、恢复和自动交接测试。
logs/full_eval_simulator_audit.json：模拟器依赖与任务/初始状态审计。
正式启动及稳定性结果以logs中实际记录为准。

2026-10-25 按最新要求从三卡六rank切换为GPU6单卡八rank。旧运行仅加载、零训练更新，日志和配置归档于 startup_attempts/gpu123_loading_before_gpu6_20261025_172619。阶段预算、结构、损失、保存策略与全量推理保持。

## Stage II 自适应启动（2026-10-28）

Stage I代码、config.json和恢复合同保持原样。Stage I完成3000步并成功退出后，自动进入start_stage_ii_adaptive.sh。入口默认只显示计划，--execute才启动。

Stage II允许物理GPU6和GPU0，优先选择有安全余量且已有rank较少的卡，每次只启动一个模型进程。每个进程只暴露其选定的物理卡（进程内cuda:0），加载自己的Teacher3000和Student，使用四条较长指令、完整五帧未来图进行验证前向与训练前向/反向，不做optimizer.step。记录峰值显存、进程显存和CPU内存，保留模型以及分配器缓存，观察20秒后才决定是否再启动一个。每rank另保留1GiB工作空间；下一进程按已测占用的1.15倍估算，GPU全局保留至少4GiB。也预留CPU优化器和加载内存。首次资源不足时等待，后续容量不足时停止增加rank。探测失败的候选进程不会加入训练。

确认N后，原进程直接组成Gloo组开始2500步蒸馏；不重复加载模型，不把容量探测计入训练步数。最大候选数32来自有效batch32每rank至少一条样本的约束，实际N以当时容量为准，不固定8。正式更新开始后N不动态变化。仅启动阶段避免主动超额分配；其他任务后续抢占资源仍可能引发运行错误，错误时保留完整checkpoint并停止后续推理。

日志：logs/stage_II_adaptive_status.json；逐rank加载日志：protocol_plus_v1/seed_2026/II/startup/；最终并行数与累积次数：II/selected_runtime.json。完整恢复使用bash start_stage_ii_adaptive.sh --resume <II/step_XXXXXX> --execute，要求相同N及原始合同。恢复入口不会自动清除失败退出回执，后续评测仍需成功退出门禁。全量推理保持GPU6上8个policy worker和10030项。

2026-10-29 按用户要求将Stage II允许设备改为[6,0]，分别检查每卡余量并保存rank_to_gpu。原GPU6单卡尝试完成1步，尚无学生检查点，已归档在logs/restart_stage_II_gpu06_*；本次从同一Teacher3000重新开始。冻结并行数后同步训练，保持有效batch32。Stage I和config.json不改，全量推理仍使用GPU6八worker。

双卡启动时，某卡因瞬时余量不足被撤回后，后续候选的完整20秒观察窗口若确认该卡重新满足准入条件，则允许再次尝试；每卡连续容量拒绝最多允许两次，以免反复争抢。尚未发生正式更新的首次双卡准入归档在logs/restart_stage_II_gpu06_retry_*。

## 最新运行布局（2026-10-29 19:39 用户修正）

Stage II改为GPU0、2、7各一个进程；GPU6先一个，再按前向/反向实测和共享显存余量逐个增加，不固定为两个。加载顺序为[0,2,7,6]后追加GPU6。前三张卡各一个，不增加第二个；必需进程的显存不足时等待，不降低必需卡数。正式更新开始后固定本轮进程组；总有效batch32、Teacher3000、Student2500和每100步完整滚动保存均保持。GPU6后续显存被其他任务占用仍可能改变运行条件。Stage II结束后的全量推理仍为GPU6八worker。

旧三进程运行已完成39步但未到100步，无学生checkpoint可恢复；记录归档在logs/restart_stage_II_fixed5_gpu0267_20261029_193814/previous_stage_II_step39。本次按要求重新从Teacher3000初始化学生，不是从第39步恢复。fixed5归档名反映用户修正前的意图，未实际启动五进程；最新策略见logs/stage_II_gpu0267_restart.json。

## 2026-10-29 20:18 Stage II 的 GPU6 替换为 GPU5

当前生效设备为 GPU0、2、5、7：GPU0、2、7 各一个进程，先按 [0,2,7,5] 顺序逐个加载，再仅在 GPU5 上按实测显存和主机内存余量增加进程。上一轮仍处于加载检查阶段，完成 0 次参数更新，归档于 logs/switch_stage_II_gpu0257_20261029_201838。本轮仍从 Teacher3000 开始 Stage II 2500 步；全局有效 batch32，每100步完整保存，新权重校验完成后才删除上一个100步权重。原始 config.json、a2_local 和 Stage I 权重均不改；后续全量推理仍遵循此前 GPU6 八 worker 设置。本次仅切换当前 Stage II 训练设备。

进度面板：`tmux attach -t a2_plus_gpu0257_progress`。最终进程数及映射以 `protocol_plus_v1/seed_2026/II/selected_runtime.json` 为准。
