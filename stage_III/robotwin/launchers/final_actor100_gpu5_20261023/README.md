# RoboTwin Stage III 最终推理

GPU5，两个独立模型服务与两个并行仿真通道。nohup 调度器独立于 tmux 看板。

- 模型：原91.45%兼容基座与适配器 + 冻结 student_000200.pt + actor_0100.pt。
- 推理模式：actor_mean，动态 Goal→Local 交互强度；不更新任何参数。
- 范围：50任务 × clean/randomized × 50 episode，总计5000。
- 91个任务模式复用原始评测完整50条的种子列表；其余9个从标准种子100000起寻找50个专家可完成场景。历史指令与噪声未严格配对，不宣称严格配对改善。
- 环境：复用 Round10 标准评测流程，使用已修复桌面重置偏移的 simulator_reset_v2。专家试跑用于有效场景和任务元数据；策略只接收当前三视角图像与指令。
- 每通道权重加载完成后，40分钟没有真实场景/动作/episode进展，终止该通道当前任务模式，保留部分结果并继续队列。普通日志写入不延长超时。
- 每个完成的 episode 立即保存；异常、超时与缺失覆盖单独统计；成功率只按已经完成的episode计算，同时展示完成覆盖。
- 每个通道独立模型端口，超时服务只恢复本通道；不管理其他用户/任务的进程。

查看进度：

```bash
tmux attach -t robotwin_stageIII_eval_progress
```

结果目录：/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/robotwin/evaluation/final_actor100_gpu5_20261023_014920

关键文件：manifest.json（配置与身份）、nohup.log（调度日志）、status.json（实时进度）、jobs/<任务模式>/result.json（任务结果及跳过原因）、summary.json（队列结束后汇总）。

启动信息保存在 launch.json；调度器代码为 eval_dispatcher.py。

2026-10-23 服务器重启后，从802条已保存结果续跑；中断任务 handover_block/randomized 15条、handover_mic/clean 10条保留。局部 resumable_round10_bridge.py 继承原始环境与模型调用，仅增加结果和种子游标恢复；每条结果原子写入并fsync。启动记录见 restart_launch.json，恢复验证见 restart_preflight.json；原文件快照见 restarts/。
