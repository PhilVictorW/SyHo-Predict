# Goal/Local LIBERO-Plus Stage I

本实验从原始 LaWAM LIBERO SFT 权重干净初始化，复用父目录 `goal_local` 架构。完整继承关系见 [guide.md](../guide.md)。不继承旧 Goal/Local 教师或学生，也不继承 `LaWAM_without_subgoal` 的模型/state token。

## 数据

源目录只读：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_without_subgoal/libero-plus/data/indexed_rlds`。

- 1024 分片，14347 episodes，2238036 steps；读取完整 manifest。
- `cache/index` 只保存约 130 MiB 的动作及 JPEG 偏移 mmap 和少量 episode 元数据。图像不复制、不重新编码。
- 每个训练样本：当前双视角+语言，Local t+8 双视角，Goal t+[24,40,56,72,88] 主视角，8×7 动作；按自然 episode 尾部 mask。
- 按指令分层、episode 级划分 5% 验证集；全训练帧均匀 shuffle。
- 夹爪已是 1=开、0=闭；保留原 LIBERO SFT 的动作 min/max。数据内 state 字段不参与模型。
- 按源转换 manifest 声明采用 20 Hz；源逐帧时间戳未保留，不另声称核实了采集帧率/no-op 处理。

## 初始化与训练

配置 `configs/stage_I.json` 除数据来源元信息外沿用父目录 Stage I：LR=1e-5，warmup=500，weight decay=1e-8，clip=1，role scale=0.01；只训练后验 action flow loss。Qwen vision/merger 与 DINO 冻结。

当前设置：物理 GPU **1、3**，DDP 两个 rank；16000 步；每卡 batch2×每卡累积8×2卡=有效32；每卡2个数据 worker；每个 rank 使用 CPU FP32 master AdamW，同步梯度后更新同一模型。各卡原有进程保持运行，GPU4不使用。

每1000步保存完整模型+optimizer/scheduler/RNG，首步额外保存便于早期恢复；保留最近两个完整 checkpoint 和验证最优 best 模型。每250步进行64个batch的离线 flow 验证，其中8个batch比较关闭角色条件的 loss。离线验证不等于在线 LIBERO-Plus 成功率。当前仅启动 Stage I，不自动启动 Stage II 或在线评估。

## 验证和后台启动

```bash
source env.sh
/home/wangchaoyang/envs/LaWAM/bin/python runtime/verify.py
/home/wangchaoyang/envs/LaWAM/bin/python runtime/verify.py --gpu
bash launch_stage_I.sh
```

验证产物：`verification/data.json`、`verification/gpu.json`。正式训练由 tmux session `lawam_goal_local_plus_stageI` 托管，窗口 train/progress；训练日志 `logs/stage_I.log`，运行目录 `runs/stage_I_20260923`。

```bash
tmux attach -t lawam_goal_local_plus_stageI
```

断开 SSH/VSCode 或 tmux detach 不停止训练。训练状态看 `status.json`，实际更新看 `metrics.jsonl`，仅有 tmux session 不能证明训练仍正常。

## 恢复

先确认旧进程已经结束，只使用存在 `checkpoint_complete.json` 的完整目录。执行 `bash launch_stage_I.sh --resume runs/stage_I_20260923/step_XXXXXX/trainer.pt`，保持双卡、总步数、batch与累积参数一致。DDP checkpoint 保存一份同步模型和optimizer/scheduler，以及两个rank各自RNG、epoch和已消费batch位置；按消费位置重新建立分布式采样器，跳过索引而不解码已经消费的样本。

所有配置、权重、训练和验证产物均留在本工作目录；共享源数据和原 LIBERO checkpoint 不写入。

## 本次验证结果

全量索引、尾部 mask、语言/相机对应、多进程 spawn DataLoader 已通过。训练/验证分别 13630/717 条轨迹、2125407/112629 个时间步。真实 Plus 样本的完整 Stage-I 前反向通过，检查的两路 Resampler、view/horizon、旧 queries、角色投影和 Plan 均有有限非零梯度，冻结视觉前端/DINO 无梯度。单样本 smoke flow loss=2.097999，峰值显存约8.71 GiB；这不是正式训练 loss 或成功率。详细结果见 `verification/summary.json`。

16000 步×有效 batch32约为512000个训练锚点，即当前训练集的0.241个epoch；采样范围覆盖完整训练集，但该预算不意味着遍历完全部样本。

## 双卡执行边界

用户指定切换 GPU1、3 时，旧单卡进程仍在 loading_model、step0，没有完成更新或生成检查点。旧单卡进程已停止，启动参数/状态备份在 `logs/single_gpu_attempt/`。

`runtime/train_ddp.py` 复用相同 GoalLocalPolicy、动作loss与MasterAdamW，新增DDP同步、无交叉DistributedSampler和rank0单次保存。`find_unused_parameters=True` 处理轨迹尾部缺失未来导致的条件分支，前7个累积微步no_sync，第8个微步同步梯度。各rank噪声RNG独立，loss为两卡平均。

全局验证仍是64个batch（128个样本）、8个诊断batch，按完整batch分给两卡，各卡32/4，不将验证预算翻倍。首次更新及每次验证周期检查跨rank抽样参数一致性，结果写入 `ddp_sync_audit.json`。`status_rank0.json/status_rank1.json`分别显示两卡进度。

正式训练前的 `verification/ddp.json` 验证真实NCCL通信、梯度累积中条件分支缺失时的同步、两卡梯度与单卡全局参考一致、更新后参数一致，以及采样无交叉和消费位置恢复。

## 2026-09-24 权重保留策略更新

用户要求所有定期模型权重都保留。每 1000 步保存、每 250 步离线验证保持不变。所有 `model.pt` 和 processor 永久保存在 `runs/stage_I_20260923/retained_checkpoints/step_XXXXXX/`，同盘硬链接无需复制大文件。当前训练不重启，由 tmux retention 窗口每 5 秒归档完整保存的权重；未来启动的 trainer 也在清理前归档。运行目录最近两份完整 checkpoint（含优化器状态）的轮换继续，历史模型权重不会因此丢失；best 仍指向验证 loss 最优模型。`retention_status.json` 记录归档心跳与步骤。

## 训练动态进度与 loss 曲线

`tmux attach -t lawam_goal_local_plus_stageI` 默认选中 progress 窗口，每 2 秒刷新。显示已完成 optimizer updates/16000、微步、最新 action flow loss、MA20/MA100、最近 1000 步的 MA50 线性坐标曲线、全程 log loss 趋势、最近 5 次验证 loss 与最佳验证步、两卡进程状态和心跳、checkpoint 与所有已归档权重。观察器只读训练日志，不控制训练进程。
