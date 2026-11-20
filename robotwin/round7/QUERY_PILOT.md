# 第七轮后续：仅训练学生query的200步小试验

触发依据：固定query主候选已在新的32条轨迹×2噪声上通过预先设定的筛查；学生推理只读取当前图像和指令，未来图片编码路径阻断检查通过，角色有效性固定开启。

从audit/initializers.pt中的fixed_queries以及round6/control/step_000200的冻结动作策略开始，仅训练q_goal和q_local，共32768个参数。原始骨干、动作头、Plan/gate、角色注入权重、新增RelevanceGate全部冻结，新gate scale始终为0。不是旧Stage II3000续训，也不是10000步正式训练。

训练目标仅为监督action flow-matching loss；batch4、lr1e-4、warmup20、总200步，每50步保存query、CPU master优化器与RNG。Local/Goal蒸馏损失在每一步第一个microbatch上作为诊断记录，权重为0；教师未来信息只用于这项无梯度诊断，不进入学生条件和优化目标。先隔离检验学生动作学习，避免再次用强蒸馏梯度扰动已保住的动作表现。

训练前要求推理路径数值一致、两组query梯度有限且非零、冻结骨干及gate无梯度。每个优化器更新后检查冻结参数版本。冻结DiT持续eval，学生语言和flow保持所需的activation checkpointing。

固定16个旧验证样本做step0及每50步验证；第七轮新的32条轨迹不用于该pilot选步或停止。记录相对初始化动作的输出漂移。任一物理平均误差超过初始值10%，连续两次则保存后结束本小试验。需20GiB空余启动，保存保留16GiB。本轮检查点只有query及优化器，冻结actor和gate由源文件恢复。

启动：bash robotwin/round7/launch_pilot.sh
进度：tmux attach -t robotwin_round7_progress（query_pilot窗口）
输出：query_pilot/status.json、metrics.jsonl、step_*/student.pt、validation_*.json。

GPU0低CPU/I/O优先级，25%显存预算；不操作其他运行进程。长训前还需专门末段压力测试和新的独立确认。本次没有恢复在线评测。
