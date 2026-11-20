# RoboTwin：先建立 Goal→Local 交互，再进行 Stage III

代码准备日期：2026-10-14。**本次只编写代码、配置、脚本并做 CPU 检查；未启动真实训练、模型服务或模拟器。**

本目录采用用户确认的 **G 路线**：先让 VLM 的 Goal→Local 消息学会有效交互，再冻结兼容 Student，以环境奖励训练倍率 Actor。plan.md 中早期的动作残差倍率路线 R 仅保留为备选分析，本实现不切换到 R。

## 1. 初始化和保留的架构

初始化严格复用 Round10 候选组合，按顺序加载：

1. 原始 LaWAM RoboTwin 发布权重：lawam_robotwin_sft_release/final_model/pytorch_model.pt。
2. robotwin/round4/runs/future_aux/step_000200/adapter.pt。
3. robotwin/round6/runs/control/step_000200/adapter.pt。
4. robotwin/round7/audit/initializers.pt 中的 fixed_queries 和零 scale RelevanceGate。

绝对路径在 [configs/workflow.json](configs/workflow.json)。不会恢复旧 Stage-I1000→Stage-II3000，也不会自动选择 latest 权重。

历史 91.45% 指已完成的 76 个任务模式组合、695/760 episodes；clean 为 330/350=94.29%，randomized 为 365/410=89.02%。这不是完整 50×2×10 的最终成绩。新实验按相同场景、指令、噪声协议配对比较，不直接拿新任务覆盖的结果减 91.45%。

保留原始两组 8-token Action/Flow Query、LaWM 预测未来 latent、Goal/Local 动作条件、4 个 Plan token 和 centered flow。**原动作 residual_gate 固定，Actor 只调节 VLM 第9–16层 Goal→Local 消息。**

| 环节 | 形状/语义 |
|---|---|
| 当前图像 | main [B,3,256,256]，双腕 [B,2,3,256,256] |
| 原始 Qwen 路径 | [B,L,2048]，L 包含原始16个 query |
| 原 LaWM 预测 | [B,256,768]；当前 DINO 同形状 |
| 原动作 base memory | [B,L+512,768] |
| 独立角色路径 | 当前上下文+8 Goal+8 Local，[B,L+16,2048] |
| 第8层 Actor 特征 | Local/Goal/Current 池化并分别无参数归一化，拼接 [B,6144] |
| Actor | FP32 6144→256→2，Beta(alpha,beta)，z∈(0,1)，eta=2z |
| 第9–16层 | Local += eta × scale[layer] × relevance × message；同一动作段共用 eta |
| Goal/Local 动作条件 | 各 [B,8,2048]→[B,8,1024] |
| 分层 DiT | 4 Plan+50 Action，[B,54,1024] |
| Flow / 执行动作 | 内部 [B,50,32]，10次 Euler；输出 [B,36,16]，30Hz 双臂绝对 EEF |

Actor 单次决策后复用于该动作段，不在每个 VLM 深层或每次 Euler 重新采样。eta=0 只关闭新增 Goal→Local 消息；eta=1 使用训练好的原门控强度，不等于恢复交互训练前的全部参数。

## 2. 阶段 A：兼容 Student 交互训练

入口：[prepare_interaction.py](prepare_interaction.py)。

- robotwin_merged 按 episode 划分 train/val，记录实际 episode 和采样索引；训练/验证 episode 互斥。
- 采样当前三视角、未来 Local/Goal 图像和36步16维动作。沿用兼容 Teacher 的未来帧协议：Local t+8 三视角，Goal t+24/40/56/72/88 主视角；跨 episode 的未来目标以 valid mask 排除。动作尾部使用 absolute-action terminal hold。
- 只训练新 q_goal、q_local 和 RelevanceGate，共 135,373,576 个参数；原骨干、LaWM、已有动作模块和兼容后验 Teacher 冻结。
- 默认 **200个优化步，每50步保存/验证；micro-batch=1，累计4次，flow噪声重复2次**。学习率3e-5，20步 warmup，CPU optimizer offload。
- 损失：action + 0.1×keep + ramp×(0.05×KD_local+0.05×KD_goal)。KD 从0渐增至第50步；4/8/12/16层蒸馏。keep 在相同 noisy-action/time 上约束 Round10 参考速度。此为 pilot 初值，未声称已验证最佳。
- 真正优化前检查全模型 step0 与原 Round10 条件等价、query/scale 梯度、冻结参数/缓冲区，以及后验 Teacher 相对基线的动作损失。Teacher screen 失败则停止，不自动继续蒸馏。
- 每50步记录位置/旋转/夹爪误差；连续两次超过配置中的回退阈值便停止，保留最后完整 checkpoint。
- 保存 student_000050.pt 等，以及 optimizer、RNG、采样身份、底座/源码身份和完整性标记；不会重复保存整个骨干。

## 3. 阶段 B：交互验收

交互训练完成不等于可以直接进入奖励训练：

1. sensitivity.py 用固定观测和同一动作噪声检查 eta=0/0.8/1/1.2：必须有非零 gate、Local 和动作可测变化，Goal 不受反向影响，eta=1 与原门控实现等价。
2. make_manifests.py 只生成候选种子；prepare_scenes.py 另行启动模拟器做专家可行性 probe 和两次真实 fresh reset。记录机器人/物体状态、关节目标、随机化信息与首帧图像 hash。候选种子文件不能直接作为 RL 场景。
3. 同一完整验证清单上比较 Round10 baseline 与交互 Student。固定 scene、instruction 和动作噪声协议，统计 clean/randomized 分别及总体配对差异。
4. acceptance.py 核对覆盖完整性、checkpoint 身份和分层 bootstrap 非劣界限。默认至少8任务×2模式，每模式每任务3场景；总体容忍1个百分点，分模式2个百分点。跳过场景会导致覆盖不完整，不能以筛选后的成功率准入。

以上均须实际执行后才有结果。目前 CPU 测试不能替代这些检查。非劣于基线只能支持进入小预算实验，不证明新交互有最终增益。

## 4. 阶段 C：Stage III 奖励训练与最终对照

入口：[train.py](train.py)。准入文件由实测结果生成并重新核对，不接受手填 passed=true。

- 冻结整个交互 Student，只训练 **1,573,634 参数的 FP32 Actor**。
- 默认5轮；每轮4个 scene group，每组4条同初始状态轨迹，最多80条训练轨迹。按任务/模式均衡调度。每轮只做1次 GRPO 更新并保存 actor_0001.pt 等。
- 4条轨迹独立采样 gate 和动作噪声；记录每个实际执行动作段的 feature、z、old_logp、alpha/beta、Actor/Student版本和执行步数。
- 成功回报 1+0.1×(1-steps/Tmax)，正常失败为0；基础设施错误和不完整组不当作失败奖励。组内标准化回报，轨迹等权，段内按轨迹长度平均。
- 更新前校验概率比接近1；使用 clipped GRPO、梯度限制、KL停止与回滚。全组优势为0时跳过 Adam，避免动量造成无依据更新。
- 160000步是训练组的**保守尝试预算**，会为允许的重试预留完整步数，不等于实际有效执行步数；不包含场景 probe 和验证/最终测试，后者单独报告。
- 默认第5轮进行验证，复用当前模型服务，避免再次加载同一大模型。试验超过5轮前要求100个任务模式组合的准入证据。
- finalize.py 只用验证集选择 Actor 和最佳固定 eta，再锁定选择，测试 Round10、交互 Student、最佳固定 eta、Actor 均值四组。默认工作流先完成训练和验证；加 --final-evaluation 才串接最终测试。

默认工作流先取固定的16个任务模式组合，每组合训练2场景、验证3场景、测试3场景。这是小预算 pilot，不能标为全50任务精度。候选种子覆盖100组合，扩大准备范围可用 --cells 100；不自动增加训练轮次。

## 5. 文件与运行方式

所有新输出限制在本目录；历史 robotwin 文件只作为依赖读取。未来运行时固定版本身份，源码/配置/底座改变后需新建 run。

| 文件 | 用途 |
|---|---|
| configs/workflow.json | 权重、数据、损失、预算、超时和验收配置 |
| base_loader.py / policy.py | 精确组合加载；可求梯度的 Goal→Local wrapper |
| gate_actor.py / grpo.py | Beta Actor、直接消息倍率、完整组奖励更新 |
| env_adapter.py / rollout.py | 原 RoboTwin reset/动作接口、完整轨迹记录 |
| server.py / rpc.py / engine.py | 显式启动的模型服务、传输与拥有的子进程管理 |
| acceptance.py / finalize.py | 准入和验证选优后的最终对照 |
| workflow.py / monitor.py | 工作流及只读进度查看 |
| tests/ / reports/ | CPU验证与检查报告 |

**现在可以安全运行的检查命令（不训练、不加载真实权重、不启动模拟器）：**

~~~bash
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/robotwin
bash scripts/check.sh
bash scripts/workflow.sh
~~~

第二条只打印拟执行命令。默认无 GPU 编号；不会沿用历史 GPU0/2/3/5 分配。

**下面仅是未来获得启动指令后的示例，本次没有执行。将 N 换为明确选定的物理 GPU：**

~~~bash
bash scripts/workflow.sh --gpu N --execute
# 如需串接验证选优和最终留出集对照：
bash scripts/workflow.sh --gpu N --execute --final-evaluation
~~~

默认 run 目录 runs/goal_local_pilot；可指定 --run-dir，但必须位于本工作目录内且新建。当前实现是单 GPU、一个冻结模型服务、串行完整 scene group；未加入多卡异步采样。阶段 A 的 optimizer 和 Stage III Actor 更新在 CPU 辅助执行。

可分别启动或续训，以下命令同样只作说明：

~~~bash
bash scripts/interaction.sh --run-dir /绝对路径/stage_III/robotwin/runs/某次实验/interaction --gpu N --execute
bash scripts/interaction.sh --run-dir /绝对路径/stage_III/robotwin/runs/某次实验/interaction --resume /绝对路径/到/student_000050.pt --gpu N --execute
~~~

Stage III 的完整参数可用 scripts/stage3.sh --help 查看。续训需保持同一 student、admission、train/val/test manifest、配置与源码，并加 --resume /绝对路径/到/actor_0001.pt。不能将新源代码直接覆盖到旧身份 checkpoint 继续训练。

总入口不会自动续跑已有 run；分阶段入口支持从完整 checkpoint/完整组记录恢复。失败组不会作为完整缓存复用，已完整提交的组按身份校验复用。没有无限重启循环。

只读监控：

~~~bash
bash scripts/progress.sh --run-dir /绝对路径/stage_III/robotwin/runs/某次实验
~~~

日志包括 interaction/metrics.jsonl、rl/metrics.jsonl、阶段 status.json、每场景 trajectory/result、验证 report 和最终 summary。监控不创建训练进程。

## 6. 超时、检查边界

- 保留用户要求：权重服务就绪后，某 rollout **30分钟无实际任务进度**即终止该工作流自己创建的 worker，跳过并继续其余场景/组。心跳不算进度；实际执行动作、完成 reset/轨迹才更新进度。
- 每条轨迹另有4小时硬上限；加载权重最多30分钟。普通基础设施异常最多重试1次；30分钟卡住的组直接跳过，不重启同一组。只管理自身子进程组，不扫描或终止历史用户进程。
- 模型输出非法、版本冲突或模型服务报错会使本次训练停止，不能被悄悄计作任务失败。
- 原子 checkpoint 和 SHA256 完整性标记、运行目录互斥锁、RNG/optimizer 保存、空间预留用于减少中断损失；不保证操作系统或驱动永不异常。
- [reports/static_check.json](reports/static_check.json) 和 [reports/cpu_tests.json](reports/cpu_tests.json) 是本次检查证据。CPU 测试使用小张量/替身测试实际 wrapper 控制流，包含合成 Actor 更新；没有用真实数据训练模型。
- **尚待未来显式执行**：真实 GPU step0 等价/梯度、显存与吞吐、Teacher screen、RoboTwin reset 可重现性、闭环非劣与 Stage III 收益。因此“代码检查通过”不等于“已经跑通 GPU 全流程”或“精度已提高”。
