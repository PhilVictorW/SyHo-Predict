# 第七轮：学生无需真实未来的迁移预检

依据第六轮结果，暂不继续强化后验蒸馏，也不恢复旧Stage II3000步。先验证已改善动作的教师能否转换为仅依赖当前观测的学生初始化。本轮没有优化器更新和在线评测。

共享冻结来源：原始RoboTwin SFT + round4辅助教师200步的Plan/gate + round6/control追加200步适配参数。保留恢复的原始LaWM与旧query，动作输出36×16，兼容候选不是纯aba2消融结构。

学生输入API严格限制为当前三视角图像与指令。prepare_current只编码当前图像并由原始LaWM预测未来；不调用真实未来图像编码或未来resampler。Goal/Local有效性恒为真，不依赖轨迹剩余长度、未来帧有效性和动作标签。未来图像编码函数与resampler在学生前向期间被替换为报错钩子。

学生新增两组8×2048的全局query和后8层RelevanceGate。Gate scale初始化为0，要求相同query时逐元素复现教师的语言路径。此处只评估初始化，不训练query或gate；动作头和原始骨干全部冻结。

候选预先固定：主候选为固定训练轨迹生成的query；次要候选为32条完全有效训练轨迹生成query的均值；零query为对照。固定/均值query离线生成后导出，推理时不再访问这些训练图像；所有测试样本使用同一份query。禁止根据fresh结果择优后仍将fresh视为独立验收。

评估包括此前32条诊断轨迹×2噪声，以及排除此前144条轨迹的新32条×2噪声，总计128组。每组比较teacher/base/fixed_queries/mean_queries/zero_queries。诊断集必须复现第六轮teacher和base指标，误差<1e-6；当前条件与零gate语言路径要求逐元素一致。

探索性门槛预先记录：新轨迹上每项物理平均误差不超过教师及原始基线的1.05倍，并且相对教师标准化误差增量的配对bootstrap 95%上界<=0.05。该门槛不是部署许可或闭环成功率结论。所有冻结参数版本逐样本检查。

输出：audit/initializers.pt（全局query与新gate初始化，不是训练检查点）、audit/results.json、audit/REPORT.md。GPU0，25%显存预算、CPU nice10、磁盘优先级best-effort7。不操作其他人的进程或现有GPU2/5任务。

启动：bash robotwin/round7/launch.sh
进度：tmux attach -t robotwin_round7_progress
