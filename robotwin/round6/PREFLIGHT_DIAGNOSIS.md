# 第六轮预检：冻结 DiT 的 dropout

2026-09-27 修复，仅限 robotwin/round6。

- 首次训练预检在优化器更新前失败：初始动作保持 MSE=0.00013281700375955552。失败产物保存在 runs/failed_preflight_control_dropout_20260926，未覆盖日志。
- 122 个参考参数与实时参数逐个完全一致；六项条件张量在 grad/no_grad、autocast cache 开/关、functional_call 替换参数下也完全一致。
- 原始 DiT 默认 dropout=0.2，共64个非零 dropout 模块。冻结参数没有关闭训练模式，导致动作速度重复前向不同；同条件 grad/no_grad 最大差值0.07347548。
- 修复：RetainedTeacher.train() 在递归设置训练状态后固定 flow.DiT.eval()；flow 本身仍 train，保留 activation checkpointing，训练未来 resampler、Goal/Local 注入与辅助头。dropout 状态在反向重算时保持一致。
- 修复后的只读实测：参考/live、grad/no_grad、cache 开/关、参数替换各项条件与动作速度最大差值均为0。详见 reference_diagnosis.json 与 debug_reference_fixed.log。

这能确认首次保持损失预检失败的原因；不能单独证明历史闭环精度下降由 dropout 引起。两个第六轮对照组都应用同一修复，仅保持约束系数不同；跨第五/六轮比较同时包含冻结骨干运行模式变化。

训练启动另有梯度预检：零差异、扰动有响应且梯度有限非零、恢复原参数、冻结参数无梯度；任意失败即停止。旧 Stage II 与在线评测继续暂停。

正式梯度预检已通过：初始保持损失0；Local scale 加0.002后保持损失3.006975930475164e-07、梯度有限且非零；冻结骨干无梯度。训练前16样本验证复现源策略：位置0.3572507056cm、姿态0.7156830337°、夹爪MAE0.0024839889。control已进入实际优化器更新，tmux为robotwin_round6_train，进度面板robotwin_round6_progress。完整对照结果仍待训练和新轨迹评估，暂不宣告效果修复。
