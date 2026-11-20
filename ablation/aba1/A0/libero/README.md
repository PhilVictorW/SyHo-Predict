# A0 LIBERO：单组 latent16、仅动作监督、完全移除 DINO

本版本采用用户于 2026-10-19 最新确认的定义。它覆盖旧 guide 中 A0 的双角色定义；A1–A3 和共用代码未修改。旧 A0 入口、配置、说明保存在 `archive/dual_role_a0/`，不能用于启动本版本。

## 模型和初始化

从 config.json 指定的原始 LaWAM LIBERO SFT W0 恢复 Qwen、原 Action8/Flow8 query、动作投影和原 DiT。新增单组 `[16,2048]` latent query，正态初始化 std=0.02。继承的视觉编码器和 merger 冻结，语言网络、旧 query、新 latent、动作头接受唯一 action flow loss。

VLM 输入为当前主视角/腕部图像、任务指令、旧 Action8/Flow8 query 和新 latent16。当前序列保持因果注意力，新 latent 可读取全部有效当前上下文和同组 latent；当前序列不读取追加 latent。位置与历史框架的 sequential Qwen positions 一致。

动作条件是：**当前 VLM 输出中按真实位置删除旧 8+8 query 后的上下文，与新 latent16 输出拼接，再通过继承的 2048→768 投影**。768 是动作 DiT 的接口宽度；没有 DINO 编码、DINO 特征或特征拼接。所有动作交叉注意力 block 读取统一 VLM 条件，交替的动作自注意力保留。没有 LaWM 子目标、Future Resampler、Teacher、蒸馏、Local/Goal 双角色、gate、额外 Plan 或 Stage III。

删除旧 Query 的直接动作条件不等于删除它们对 VLM 的间接影响。当前图像仍由 Qwen 视觉编码器处理。只有未来图像被排除；专家动作序列仍是监督标签。

## 数据、500 optimizer steps 与三卡配置

固定 episode 划分 split_seed=2026，val_ratio=0.05：训练 1614 episodes / 260726 anchors，验证 79 / 12739；启动时强制核验。训练只解码当前两视角图像，监督最多 8×7 个有效动作元素，内部动作表示 50×32，尾部 mask 排除不存在动作。动作归一化继承原 SFT 统计。每样本两组 flow 噪声不增加 epoch 计数。

启动前扫描八张 GPU，选择空闲显存最多且足够的三张，**每张一个训练进程**，共三个 FSDP rank；实际编号见 `config.json` 和 run 的 `launch.json`。每进程 micro-batch=1，每次完整更新执行11个微步，实际有效全局 batch=32：三个rank分别处理11、11、10个样本，第三个rank最后一个微步为零权重占位。CPU 数据加载 worker 不占用 GPU、不参与模型训练通信。本机 GPU 直连传输出现通信屏障阻塞，启动器固定 NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1，通过主机内存通信。FP32 参数/梯度分片，Adam 的 FP32 master 和动量分片放在 CPU，BF16 前向，block activation checkpointing。

当前预算是 **500 次 optimizer updates**，每次32个有效anchors，共16000次样本访问，约0.06137个数据epoch。采样保持完整数据集的固定随机排列，只消费前16000个anchors，第500步即停止，不走完整epoch。三卡第11个微步中的占位样本权重为零，不计入样本数。每个真实样本权重为world_size/实际本次更新样本数，匹配FSDP平均梯度。恢复游标记录实际已消费anchors。

AdamW lr=1e-5，betas=(0.9,0.95)，eps=1e-8，weight_decay=1e-8；梯度范数裁剪1；warmup50次更新，之后cosine衰减至基础lr的5%。总schedule覆盖500次更新，第50步达到基础lr，第500步衰减至基础lr的5%。

## 启动、恢复与停止

所有新代码、运行数据和缓存位于本目录。以下命令从本目录执行；不要启动 archive 中的旧入口。

```bash
# 只读资源/配置检查
bash preflight.sh
# 不加载GPU的预算检查
/home/wangchaoyang/envs/LaWAM/bin/python train.py --output "$PWD/runs/your_run" --dry-run
# 正式500step三卡训练
bash train.sh --output "$PWD/runs/your_run"
# 完整断点恢复，保持同一run、配置、世界大小
bash train.sh --output "$PWD/runs/your_run" --resume "$PWD/runs/your_run/checkpoints/epoch_XXXX_step_XXXXXXXXX"
```

正式训练起点始终为W0+新latent随机初始化，GPU smoke使用独立run，不继承smoke更新。训练step500结束后自动停止优化器更新并写final.json；未完成时不生成final.json。初始及第250、500步验证固定128个anchors，固定每rank噪声种子。验证不改变训练RNG。

每100次更新及第250、500步保存完整checkpoint，保留最近2份和best-flow；step500主结果不被best替换。checkpoint包含每rank FP32参数、Adam状态、RNG、数据游标及训练契约；`model.pt`是供推理使用的BF16完整模型。完整恢复必须读取rank文件，不能仅加载model.pt冒充resume。保存先写.incomplete，完整后原子改名；不覆盖失败快照。磁盘空间不足时明确报错。日志为metrics.jsonl，状态为status.json。

## 无DINO推理和评估

推理只接受当前两视角图像与指令，保持训练架构。随机动作噪声经10步Euler生成有效8×7动作，反归一化后执行8步，再接收新观测。夹爪沿用环境2g−1转换，不重复归一化。

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=0 bash evaluate.sh \
  --checkpoint "$PWD/runs/your_run/checkpoints/epoch_0000_step_000000500" \
  --protocol quick --output "$PWD/evaluation/your_run/quick_40x10"
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=0 bash evaluate.sh \
  --checkpoint "$PWD/runs/your_run/checkpoints/epoch_0000_step_000000500" \
  --protocol full --output "$PWD/evaluation/your_run/full_40x50"
```

实际checkpoint路径以final.json为准。Quick=40任务×10初态，共400episodes；Full=40×50，共2000。两者初态重叠，不能用quick反复选模型后把full称为独立测试。服务严格加载本地新格式，输出逐任务、suite和总体成功率。模拟器和原episode core只读复用，其日志/结果保存在本目录。

`pipeline.py` 用于持久运行：训练成功完成500step后，核对final.json并依次执行quick/full，所有状态写在pipeline_status.json。任何训练或评估错误都会停止pipeline，不继续使用不完整权重。

## 检查与结果解释

旧共用测试中的双角色A0用例已不适用于本版本；本目录提供独立测试。CPU测试覆盖单组注意力、真实旧query位置剔除、动作损失对新旧query的梯度、current-only动作生成、全训练集精确覆盖、尾部权重及恢复游标。三卡真实检查单独验证W0严格加载、前反向/更新、完整保存恢复和无DINO推理。

`checks/`保存检查报告和日志；正式状态以run目录status.json为准。500步训练尚未结束时，不把启动或smoke检查写成训练完成。

与旧guide的A0相比，本版本同时改变query组织、角色/Plan路径和DINO条件，后续与完整方法的结果差异不能全部归因于蒸馏。

### 2026-10-19 三卡启动检查

GPU0/2/7 的两次真实更新（64个训练anchors）、FP32参数/Adam/RNG恢复、无DINO动作输出检查通过。独立单卡导出推理与恢复后的三卡模型最大动作绝对差0.00016224。报告见 `checks/validated_3gpu.json`。训练日志分别记录实际分配显存峰值、当前分配和缓存预留；初始化和保存后释放闲置缓存。检查权重不作为正式训练起点。

### 预算变更与进度监控

用户已将预算从500个完整epoch改为500次参数更新、warmup50步。旧500epoch任务在第3步后取消，未产生完整checkpoint；新任务从原始W0和同一随机种子开始，旧日志不计入新实验。旧预算代码保存于 `archive/500epoch_20261019/`。

只读 tmux 会话为 `a0_libero_500step`，`progress` 窗口展示步数、loss、学习率、更新速度、预计剩余时间和本任务GPU显存；`logs` 窗口查看训练日志。使用 `tmux attach -t a0_libero_500step` 进入，Ctrl-b 后按 d 脱离；不会停止后台训练。三个连续有效参数更新通过后结束人工主动轮询，监控窗口继续每5秒刷新。

后台进度监控每5秒写入run目录的 `monitor_heartbeat.json`，在阶段/步数变化时追加 `monitor_events.jsonl`。训练或pipeline退出、长时间无更新会在监控中显示提示。训练完成后同一窗口继续显示quick/full评估进度；`evaluation`窗口查看全量评估日志。已核对自动流程的审计记录为run目录的 `automation_verified.json`，当前训练进程保持运行。
