# A2 LIBERO-Plus：GPU1、2、3，每卡两个同步训练进程

本组独立执行 Goal-only Teacher3000 → Goal-only Student2500 → 全量10030任务推理。所有代码改动、日志、权重、队列、缓存和临时文件保存在本目录。A1与公共核心实现仅只读复用；原脚手架归档在 setup_archive，A1运行文件来源哈希见 setup_source_manifest.json。

## 模型与训练

Stage I 从 config.json 指定的原始 LaWAM LIBERO SFT 权重初始化，只读取 t+[24,40,56,72,88] 五张有序未来主视角，经冻结 Qwen visual、可训练 horizon embedding 与 Goal Resampler 得到16个Goal槽。保留当前主/腕视角、指令、DINO、原查询和4个Plan。Goal仅在DiT第1/5/9/13层注入Plan，再经原self-attention影响Action。没有Local Resampler、Local Query、Local→Action或跨角色gate。

Stage I仅action flow loss，训练3000个optimizer updates。Stage II严格继承自己的 I/step_003000 公共参数与动作分支，新建 Q_Goal[16,2048]。教师完全冻结，学生只看当前输入，loss为flow+Goal蒸馏；第4/8/12/16语言层cosine+Gram四层平均，权重均1。Stage II新建优化器与RNG/采样器，训练2500步。

六rank共同训练一个模型，物理映射[1,1,2,2,3,3]。micro-batch=1，累积[6,6,5,5,5,5]，有效batch32，与A1相同。LR1e-5、warmup500，scheduler总长16000/10000，flow repeat2。Stage I/II分别消耗96000/80000个anchor；固定seed2026与历史数据划分。不能与5000/4000的旧完整模型声称等预算。

保留A1的CPU Gloo梯度同步、rank0 FP32 master AdamW、梯度/激活CPU offload、冻结视觉编码器共享和错峰加载。禁用memory-efficient/CuDNN SDPA，启用math/Flash。当前观测、数据和训练目标不因运行布局变化而改动。

## 保存与恢复

两个阶段都每50个完整optimizer update保存。先写临时目录，落盘并验证模型/恢复状态SHA256，再原子发布并更新latest.json，最后只删除本阶段旧完整checkpoint。保存失败保留旧文件。完整保存包含processor、动作统计、FP32 master/Adam、调度合同和全部rank的RNG/采样游标。Gloo路径按step计算学习率，不需要有状态scheduler。

最终保留 protocol_plus_v1/seed_2026/I/step_003000 与 II/step_002500。学生滚动保存不删除教师。写入前检查空间；新旧快照同时占用磁盘。正式首份恢复点为step50，step0不保存。

Stage I恢复入口为 resume_stage_i.py --resume <本阶段完整checkpoint> --execute；新阶段初始化与同阶段断点恢复不同。恢复核对配置、代码、数据和设备布局。

## 启动、续接与监控

启动预览：bash launch_background.sh。明确执行：bash launch_background.sh --execute。已有任务时拒绝重复启动。

nohup后台运行Stage I及continue_training.py。教师3000完整校验且Stage I退出码为0后才启动Stage II；学生2500完整校验且独立退出码为0后才启动全量评测。失败不伪装为完成。nohup可跨终端断联，服务器重启后需检查并恢复。

动态进度：bash watch_progress.sh；单次快照：bash watch_progress.sh --once。

- logs/train_gpu123.log：训练输出。
- logs/auto_continuation_status.json：续接状态与心跳。
- logs/launch_record.json：GPU映射和后台PID。
- protocol_plus_v1/seed_2026/I 或 II：metrics、rank状态、参数与数据审计、latest.json。
- logs/full_eval.log：后续推理日志。

TMPDIR使用/proc/self/cwd/.t这个短别名，实际文件在本目录的.t内；训练和评测子进程cwd均为本目录，避免AF_UNIX路径过长。CUDA、HF、Matplotlib等缓存也位于本目录。

## 全量评测

最终只加载自己的Student2500，无Teacher/未来图像/Actor。GPU1/2/3各两个policy worker，总共六worker；每worker另有CPU模拟器进程。固定10030项，suite数2402/2518/2591/2519，每项1episode，初始状态0、环境seed0、chunk8、flow10步，保持A1夹爪转换与噪声种子规则。

输出 protocol_plus_v1/seed_2026/evaluation/full_II_step_002500/results.json、matrix.csv和逐episode记录。训练未完成前没有本组成功率，不引用历史89.62%。

## 验证记录

logs/preflight.log、preflight_report.json：数据与预算预检。
logs/validate_cpu.log 与 logs/a2_targeted_tests.log：角色路径、梯度、初始化和六rank同步；最初的设备映射测试期望值已修正，以针对性重测为准。
logs/runtime_tests.log：滚动保存、恢复和自动交接测试。
logs/full_eval_simulator_audit.json：模拟器依赖与任务/初始状态审计。
正式启动及稳定性结果以logs中实际记录为准。
