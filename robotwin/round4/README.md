# 第四轮：赋予Goal/Local明确的未来动作监督

第三轮已修正轨迹尾部token窗口。200步教师在32条新验证轨迹×2个噪声种子上保持了原始基线附近的动作表现，但真实未来相对错配未来的收益区间跨零，尚不适合直接蒸馏。

换掉未来图像后，Goal/Local第16层表示平均相对变化约55%/31%，动作仅约0.19%。因此不再假设未来信息在Qwen里完全丢失；本实验测试这些表示是否缺少与动作相关的明确监督。

## 两组同条件追加训练

源检查点：round3/runs/teacher_terminal_hold36_20260926/step_000200/adapter.pt。

- action_only：只优化36步flow action loss。
- future_aux：相同初始化、训练anchor、噪声流程、学习率和更新数，增加未来动作辅助头。
- 每组追加200步，即累计等效教师400步；有效batch4，学习率1e-4/预热10步，每50步保存并验证。
- GPU0顺序执行两组及各自32轨迹×2噪声的评估；2 CPU线程、显存上限25%、低CPU/IO优先级。
- 原始Qwen/DINO/动作头/LaWM和旧query继续冻结。保留原始条件的兼容候选不等同于去掉LaWM/query的aba2定义。

## 辅助头与目标

Local的8×2048表示取均值，经LayerNorm和2048→256→16 MLP预测t+8的动作目标；Goal同样经2048→256→80 MLP预测t+[24,40,56,72,88]的五个16维动作目标。

目标来自各自轨迹的真实action数组，是未来时刻的动作指令，不是额外输入的机器人状态。越界目标被屏蔽，不重复补齐未来标签。当前36步动作窗口仍按原始绝对动作规则末值补齐。

辅助loss取三个分量平均：归一化位置MSE、夹爪MSE、四元数符号不变的1-dot(q_pred,q_target)^2。四元数先单位化。辅助头只用于教师训练，不用于在线动作执行，也不给学生引入未来输入。

辅助系数上限1e-3，前20步逐渐引入；每50步抽查Local/Goal resampler query的动作梯度和辅助梯度，按其范数限制辅助系数。该限制仅针对抽查参数，不声称覆盖全模型梯度。原始骨干仍完全冻结。

## 校验、进度与判定

cpu_checks.json验证未来标签的轨迹边界、四元数正负号等价、全无效目标的loss和梯度为零。每组preflight验证继承200步后的物理结果一致、临时关闭新路径时恢复原始动作头、冻结骨干无梯度。

启动：bash robotwin/round4/launch.sh。
进度：tmux attach -t robotwin_round4_progress。
动态对照：REPORT.md / summary.json。

每个checkpoint的step表示本轮追加步数，metrics同时记录累计教师步数。新格式robotwin_future_aux36_adapter_v1，依赖原始冻结权重，不能直接交给旧推理服务器。不会自动启动学生或在线评测。

验收同时看：相对原始基线的动作保持、真实未来相对错配未来的配对收益、辅助组相对action-only组的配对收益。按轨迹bootstrap，两个噪声种子不当成两条独立轨迹。辅助loss下降本身不构成成功。原始SFT可能见过示范数据，仍需后续闭环验证。

旧GPU2、5 Stage II因磁盘空间不足于3314步退出。3000步模型和optimizer可读取，恢复点保留；未在已知旧数据规则下自动重启。当前实验使用独立目录，不改旧检查点和线上队列的暂停标记。

## 训练后只读排查

- `audit_auxiliary.py`：同一批32条验证轨迹，检查追加0/50/100/200步的辅助预测，分别替换正确/错误未来，保持接收样本的有效性掩码。只有31条有Local目标、30条有Goal目标；全无效样本不计入均值。
- `auxiliary_audit/REPORT.md` 与 `component_benefit.json`：辅助头整体及位置、旋转、夹爪分项的配对结果。所有区间是探索性、未作多重比较校正。
- `audit_gate.py`：固定200步教师，取上述验证集前16条轨迹和一个固定噪声，内存中把残差门控扩大到1/3/10倍，同时比较正确未来、错误未来、关闭角色和原始基线。它不更新优化器，不写回检查点，也不运行闭环测试。
- 两个审计都只用GPU0，不启动旧Stage II，不更改在线评测暂停标记。源检查点依赖的原始骨干保持不变。

## Plan与Goal/Local分离对照

`audit_split_paths.py` 对固定200步辅助教师执行只读分离：

`v = v_base + g_plan*(v_plan_off-v_base) + g_role*(v_plan_role-v_plan_off)`

抵消Plan独立增量不等于删除Plan token；Goal条件仍可通过Plan内部结构影响动作。原始门控、全零门控和只有Plan的三个等价性检查，动作输出最大差异均为0。速度输出实际为FP32，没有证据把本次问题归因于残差合成时的BF16舍入。

诊断结果见 `split_path_audit/REPORT.md`。选定Plan0/Role3后，`confirm_split_paths.py` 固定此候选，使用另外32条验证轨迹、每条两个噪声种子复核。与之前32条诊断轨迹和16条调参轨迹无重合。候选、样本和判据在评估前写入 `split_path_confirmation/manifest.json`，结果见同目录 `REPORT.md`。

进度：`tmux attach -t robotwin_round4_progress`，其中 `split_paths` 和 `confirmation` 窗口分别展示两项检查。没有新增正式训练、学生续训或在线测评。
