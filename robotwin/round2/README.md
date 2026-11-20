# 第二轮：36步兼容初始化与教师有效性预实验

本目录是独立实验版本，不覆盖正在运行的模型、训练配置或权重。

- GPU0，200次有效更新，有效batch=4（micro batch 1，累积4）。2个CPU线程，低CPU/IO优先级，显存上限25%。GPU2、5原Stage II继续。
- 每50步保存adapter及完整optimizer，随后做固定16样本离线验证。没有自动在线评测，也没有自动开始新的10000步学生训练。
- 初始化来自原始发布的RoboTwin SFT权重；不是旧Stage I 1000步或已删除的Stage II检查点。
- 保留并冻结原始Qwen、DINO、动作头、LaWM mapper/decoder及两组旧query。只训练新增Goal/Local、Plan、投影/注入模块及全局残差门控。
- 新模块学习率1e-4、预热10步；CPU FP32 master AdamW，clip=1。这是预实验配置，不是已验证的最优学习率。

## 兼容初始化

`v = v_original + tanh(g) * (v_augmented - v_original)`，g初值为0。

原始路径没有Plan，增强路径加入Plan与Goal/Local，二者使用相同的冻结原始骨干及完整原始条件。零门控时恢复原始输出；门控具有梯度，打开后新增分支开始学习。

本版本恢复了原始LaWM/query条件，**不等同于移除这些条件的aba2定义**。它用于建立兼容锚点；移除旧条件应另做消融变量。

## 数据与维度

复用robotwin_merged轨迹级训练/验证划分，只读源数据及缓存。训练预生成800个随机anchor，记录samples.json；验证沿用上一轮固定索引的前16个，不用验证集训练。

动作标签[B,36,16]装入[B,50,32]缓冲，前36步及16维EEF有效，轨迹尾部按剩余动作屏蔽。预测36/30=1.2秒。

Local未来偏移保持+8帧的三视角；Goal保持+[24,40,56,72,88]帧主视角，分别对应0.267秒和0.8/1.333/1.867/2.4/2.933秒。未来偏移与动作预测长度分开配置，未直接按36/8倍增。两组角色均为[B,8,2048]，注入动作头时投影为1024维。

## 启动与查看

从LaWAM_aba2_change目录执行：

```bash
bash robotwin/round2/launch.sh
tmux attach -t robotwin_round2_progress
```

后台训练会话为robotwin_round2_train；运行目录为robotwin/round2/runs/teacher_compat36_20260925。启动脚本通过文件锁防止重复运行。

- preflight.json：原始头FP32一致性、零门控采样一致性、门控非零有限梯度、冻结骨干无梯度、轨迹边界mask检查。必须通过才优化。
- manifest.json / samples.json / dataset_audit.json：配置、代码hash、训练和验证采样、数据统计。
- status.json / metrics.jsonl：阶段、有效更新步数、action loss、gate和验证指标。
- step_000050/adapter.pt等：新增模块、optimizer和RNG。它们依赖manifest指定的原始冻结权重，不能直接交给旧在线服务器。
- validation_*.json：逐样本完整36步的位置、姿态、夹爪误差；与上一轮只统计前8步的均值不可直接比较。

恢复需在相同环境、相同源码和输出目录，用原参数运行python -m robotwin.round2.train并增加--resume <adapter.pt>。程序先重建并检查原始基线，再恢复adapter、optimizer和RNG；源码不一致会拒绝恢复。

梯度范数用FP64累加，非有限梯度不会更新参数，会重试当前步；三次异常则记录失败并停止这个独立实验。检查点以临时文件原子替换，完成后才生成完成标记；不无限自动重启错误进程。

## 验收和后续

固定样本、固定噪声对比四组：完整教师、原始参考路径、关闭角色注入但保留Plan、错配其他轨迹未来。同轨迹donor在逐样本结果中标记，解读错配效应时须排除。

200步完成后复核真实未来相对关闭/错配未来是否有一致收益、是否恶化原始参考动作；微小变化应扩大样本确认。预实验不证明闭环成功率。

教师通过后，再从相同初始化、相同anchor/种子建立学生action-only与低权重渐进蒸馏对照，限制对继承骨干的修改，并监测动作/蒸馏梯度。这一步尚未启动，不能自动蒸馏未经验证的教师。

最后再用相同任务、模式、episode种子比较预测36/执行36与预测36/执行8。旧的每500步在线队列继续暂停；本轮尚未启动在线测试。
