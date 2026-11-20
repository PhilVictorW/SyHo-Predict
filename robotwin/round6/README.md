# 第六轮：显式保持已验证的动作策略

目标：检验采样轨迹上的动作保持约束能否减少第五轮出现的动作回退，同时继续学习未来辅助预测。不是旧Stage II续训。

两组均由round4/runs/future_aux/step_000200/adapter.pt初始化（累计教师400步），固定Plan0/Role3、原始骨干、Plan token及总门控；训练future resampler、Goal/Local注入参数和辅助头。数据仍为robotwin_merged，36步绝对动作窗口末值补齐，未来标签越界屏蔽。

- control：L = L_flow + lambda_aux * L_aux。
- retained：L = L_flow + lambda_aux * L_aux + 10 * L_keep。
- 两组辅助分项均等权，不再同时改变辅助权重或学习率；batch4、lr1e-4、warmup10、同一随机种子和训练anchor顺序，各追加200步，每50步保存、验证。

保持参考是初始化时的不可训练适配参数副本，与实时模型共享未更新的原始骨干，避免复制整个大模型。用torch.func.functional_call临时指定参考参数，不原地覆盖实时参数。参考计算禁用autocast缓存，不向参考传播梯度；每次更新检查参考张量版本和冻结骨干版本。

每个训练microbatch使用独立固定种子，从参考策略10步Euler轨迹中选择一个时间点。比较新旧策略在相同观测、未来条件、噪声状态与时刻下的36×16有效动作速度，MSE作为L_keep。源轨迹生成不改变主action loss的随机数流，两组均计算该指标，只改变其损失系数。它是局部速度保持约束，并非闭环成功率保证，必须通过后续采样动作验证。

预检要求：新实现重现此前已验证候选、关闭角色回到原始基线；参考初始loss接近零；扰动实时Local scale后保持loss变为正且产生有限非零梯度，恢复原值后才允许优化器更新。预检失败则停止，保留日志。

评估：固定16样本每50步验证；各组结束后使用新的32条轨迹×2个噪声，排除先前112条调参/诊断/确认轨迹。除原始基线、正确/错误未来外，同步报告冻结参考策略在相同输入和噪声下的结果，避免相对原始基线看似尚可、实际却比训练起点回退。

输出在robotwin/round6；GPU0低CPU/IO优先级、25%显存预算。启动需40GiB空余，保存保留32GiB；完整模型适配参数、CPU master optimizer、RNG和辅助梯度上限原子保存。参考副本不重复存储，可由源检查点重建。新检查点格式robotwin_retained_centered36_adapter_v1。

启动：bash robotwin/round6/launch.sh
进度：tmux attach -t robotwin_round6_progress

旧Stage II3000步不恢复，在线评测不启动。兼容教师保留了原始LaWM及旧query，不是纯aba2消融；离线误差与bootstrap区间不等于RoboTwin闭环成功率。

预检修复（2026-09-27）：原始 DiT 构造器默认 dropout=0.2；此前冻结权重但仍调用 train()，使相同输入的动作输出随机变化。首轮零步预检因此失败，未执行优化器更新。第六轮现在在 control/retained 两组都让冻结 DiT 持续 eval，保留 flow 的 train 模式和 activation checkpointing，确保反向重算时也关闭 dropout；未来 resampler、角色注入和辅助头仍训练。两组只在保持约束系数上不同，但与第五轮相比同时修正了冻结骨干的运行模式，不能把跨轮差异全部归因于保持约束。此问题与历史在线成功率下降的因果关系尚未确认。

2026-09-27 后续：两组200步及各64次离线配对评估已完成，见DECISION.md、paired_review.json。新增只读对照由launch_dependency.sh启动，进度位于robotwin_round6_progress的dependency窗口；结果写入dependency_audit/REPORT.md。该诊断仅占GPU0，CPU nice=10、磁盘best-effort最低优先级7、显存预算25%。仍不启动学生训练或在线评测。
