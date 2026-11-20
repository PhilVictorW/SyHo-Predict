# GPU5重复计算差异：本轮第0步发现

本轮在GPU5完成了3次第0步预检，均未执行优化器更新。两组初始化所有参数相同，但同一组重复前向也有差异；CUDA随机数状态没有变化，教师所有模块均为eval，关闭autocast缓存和改用SDPA math后仍存在差异。详见各attempt目录的zero_gate_diagnostic.json。

最小复现使用固定输入[1,280,2048]和权重[2048,2048]执行BF16线性层。GPU5每次约573440个输出中的1到10个元素变化，最大绝对差可达0.5；平均绝对差约1e-6量级。开启确定性算法及CUBLAS_WORKSPACE_CONFIG=:4096:8、关闭BF16 reduced-precision reduction、切换cuBLAS/cuBLASLt，未消除该现象。GPU5独立SDPA和FP32线性层测试均稳定。GPU0相同BF16测试六次重复差均为0。

GPU5 ECC各项计数为0。目前不能断言硬件故障，也不能把这少量低位差异直接认定为此前成功率大幅下降的主因。本轮为避免混入对照误差，改用GPU0重新做全部预检与训练。没有更改驱动、系统CUDA或全局环境，没有停止任何其他任务。

可复现的轻量探针numeric_probe.py只进行少量计算，显存限额1.5%；由调用者设置CUDA_VISIBLE_DEVICES、CUDA_DEVICE_ORDER=PCI_BUS_ID、CUBLAS_WORKSPACE_CONFIG=:4096:8。探针不会启动训练。已观测数据保存在gpu_repeatability_probe.json。
