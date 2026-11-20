# 第八轮：学生 query 与 RelevanceGate 的受控短训

上一轮已证明兼容初始化和200步query-only学生在离线动作误差上没有明显退化，但没有证明短训有额外收益。本轮只改变学生RelevanceGate是否可训练，进行同初始化、同样本、同噪声、同学习率的配对试验。

正式对照改用物理GPU0，CUDA可见设备只设为0，显存上限20GiB；CPU线程2，nice10，ionice普通类最低优先级。不停止任何其他进程。原始LaWM条件、旧query和兼容actor保留；这不是纯aba2消融。

- 两组从第七轮fixed_queries初始权重重新开始，不继承学生200步，不恢复旧StageII3000。
- query_only：仅q_goal/q_local可训练，RelevanceGate固定为零门控。
- query_gate：q_goal/q_local及完整RelevanceGate可训练；其他actor、Qwen、LaWM、视觉编码器和教师参数全部冻结。
- 每组200步，有效batch4，LR1e-4、warmup20；两组交替处理同一microbatch，flow noise seed固定为280920+4*step+micro，避免先后执行带来的随机数偏差。
- 训练索引使用第七轮已固定的800个训练锚点，验证使用旧固定16个诊断锚点；不调用上一轮新确认集。
- 学生只读取当前图像与指令。真实未来仅供冻结教师生成无梯度的Local/Goal KD诊断，KD权重均为0。
- 开始前验证零门控逐元素等价、零scale时梯度只流向scale/query、临时打开scale后消息和相关性模块能收到梯度。探针随后恢复到零，不保留任何优化器更新。
- 第2步检查实际门控参数已经变化；每步检查1234个冻结教师参数和对照组门控版本未变化。
- 每50步保存两组完整可训练参数与优化器，再做固定16样本验证；任一组某指标连续两次超过其初始误差10%，两组同时停止并保留已保存checkpoint。
- 门控完整保存及优化器会增加磁盘占用；开始时检查至少32GiB可用，每次保存检查至少24GiB，原子写入并生成完成标记。

运行：bash robotwin/round8/launch.sh。后台工作会话robotwin_round8_train；看板tmux attach -t robotwin_round8_progress。

完成后先审查两组固定诊断误差，再决定是否进行新的独立确认；没有自动开启长训或在线rollout。旧StageII3000及在线评测继续暂停。

## GPU5第0步预检异常

三次预检尝试均在优化器更新前停止，原始记录保留在attempt_001/002/003目录。两组所有初始参数完全相同，同一组重复前向仍变化；无CUDA随机数消耗，教师所有模块均为eval。独立BF16线性层在GPU5重复输出发生少量元素变化，确定性算法、cuBLAS workspace配置、关闭BF16 reduced precision reduction、切换cuBLAS/cuBLASLt均未消除；FP32及单独SDPA测试稳定。GPU0相同BF16测试六次重复差值均为0。ECC计数为0，尚不能归因于硬件故障。正式训练转到GPU0，上限20GiB、nice10，不停止已有任务。
