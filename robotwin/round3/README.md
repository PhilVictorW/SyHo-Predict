# 第三轮：修正轨迹尾部窗口，保持其他条件不变

问题不是归一化数值，而是轨迹末尾的有效动作token数量与原始训练不一致。

原始LaWAM数据加载器对absolute=true的EEF动作使用末值补齐；加载36步后，collator将其补到50步缓冲，前36步都有效。旧自定义数据集在尾部仅保留remaining个有效token，继承动作头因此面对改变了的自注意力上下文。

本轮TerminalHoldDataset恢复末值补齐，TerminalHoldFlow保留36步的监督与注意力窗口。同时保存observed_actions_mask，评估物理误差只用真实帧，不用虚拟补齐帧。原始8步训练和第二轮旧文件均保留，不修改正在运行的GPU2、5进程。

## 验证与实验

1. diagnose_tail.py直接调用原始加载器的retrieve_data_and_pad及原始collator工具，验证20个轨迹边界案例。
2. 24个训练anchor（12尾部、12非尾部），每个4个去噪时间点：同图像、同条件、同噪声，比较旧窗口与修正窗口；指标统一使用原本存在的动作帧。
3. 确认尾部改善且非尾部输出保持一致后，启动200步匹配重跑。初始化seed=92026、800个训练anchor、16个固定验证样本、学习率1e-4、batch4、每50步保存和验证，均与第二轮相同。只改变尾部动作处理。
4. Goal/Local是否真正有益仍由真实/关闭/错配未来对照决定，不自动启动学生或在线评测。

启动：bash robotwin/round3/launch.sh。
进度：tmux attach -t robotwin_round3_progress。
运行目录：robotwin/round3/runs/teacher_terminal_hold36_20260926。
配对结果：COMPARISON.md及comparison.json，每30秒更新。

新检查点格式robotwin_terminal_hold36_adapter_v1，包含新增模块、optimizer和RNG，依赖原始冻结SFT权重，不能直接交给旧的在线推理服务器。其余恢复与鲁棒性策略沿用第二轮。保存前检查剩余空间，若少于24GiB则终止独立实验以保留现有训练的保存空间。

这些实验仅验证离线行为和监督契约，不等于闭环成功率已恢复。
