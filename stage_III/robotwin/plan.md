# RoboTwin Stage III：基于 Round10 兼容模型的奖励后训练分析与实施计划

核对日期：2026-10-14。工程根目录 ROOT 为 `/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change`；下文源码和权重相对路径均相对于 ROOT。

**状态：用户已选定 G 路线，本目录的代码、配置和脚本已准备；尚未启动真实训练、模型服务或模拟器。** 实际入口、默认预算、续跑说明和检查边界见 [README.md](README.md)。实现只修改本目录，复用历史兼容模型作为依赖。以下保留两条路线的分析，R 不在当前执行流程中。CPU/静态检查报告位于 reports/；真实 GPU 和闭环验收须在显式启动后完成。

## 1. 结论与建议顺序

**可以基于 91.45% 对应的模型开展 RoboTwin 奖励后训练，但不能把参考文档的 Stage III 原样套上去就开始有效训练。** 文档调节的是已训练的 VLM 深层 Goal→Local 消息；Round10 候选这条 RelevanceGate 通路的 8 个 scale 全为零。冻结底座后，任何外部倍率乘零都不改变动作。

| 路线 | 实际优化对象 | 能否直接从当前组合权重出发 | 定位 |
|---|---|---|---|
| **R：动作残差倍率后训练** | 已有 Goal/Local 条件参与形成的 centered action residual | 可以做最小适配；先证明残差对动作和回报有可调影响 | 备选分析；当前不实施 |
| **G：文档定义的 Goal→Local Stage III** | VLM 第9–16层已经学好的 Goal→Local 消息 | 不能直接有效训练 Actor；须先建立可用的兼容学生交互 | **用户选定、当前已编写代码的路线** |

当前执行顺序：**固定 Round10 组合基线 → G 路线兼容 Student 交互训练 → 动作敏感性和配对闭环非劣验收 → 冻结 Student 后进行 Actor-only Stage III → 验证选优与留出集对照。** R 的成功不能替代 G 的研究结论；当前脚本只实现 G，默认仅打印流程，不启动训练。

两条路线均保留原始 LaWM 从当前观测预测未来 latent 的能力。这与“在线不读取真实未来图像”相容，不应为接入 Stage III 再次删除原始动作条件链。

## 2. 91.45% 对应的真实模型和成绩

### 2.1 成绩口径

从 `robotwin/round10/candidate_online/status.json` 按任务与模式去重，仅统计具备完整 summary 的 completed 组合：

| 模式 | 完成任务模式组合 | 成功/episodes | 成功率 |
|---|---:|---:|---:|
| clean | 35 | 330/350 | 94.29% |
| randomized | 41 | 365/410 | 89.02% |
| 合计 | 76 | 695/760 | **91.45%** |

原计划为50任务×2模式×10 episodes=1000 episodes。其余24个组合没有进入上述完整统计，包括异常或未完成组合；重试及部分结果不能重复计数。91.45% 是**已完整完成组合的 episode 加权成功率**，不是完整100组合成绩，也不是两种模式百分比的简单平均。

`candidate_queue.py` 对历史原始 LaWAM 的比较标记 `strict_pair=False`：指令与策略噪声未严格配对。后续提升必须由冻结 Round10 候选和新 Actor 在同一预先固定的验证条件下比较；不能用不同任务覆盖的新结果直接减91.45%。

### 2.2 实际加载顺序

入口是 `robotwin/round10/server.py::Policy(arm='compatible', ...)`，不是旧 Stage-II loader：

1. 原始 SFT 底座：`/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt`。
2. 构建 `robotwin.round6.model.RetainedTeacher`，按顺序叠加 `robotwin/round4/runs/future_aux/step_000200/adapter.pt`、`robotwin/round6/runs/control/step_000200/adapter.pt`。两份 adapter 的顺序和各自参数覆盖范围必须保留，不能假设后一份是完整模型。
3. 构建 `robotwin.round7.model.CurrentQueryStudent`，从 `robotwin/round7/audit/initializers.pt` 加载 `queries.fixed_queries` 和 `gate`。recipe 在 `robotwin/round7/candidate_recipe.json`。
4. 共享冻结的兼容骨干、LaWM与动作模块；在线明确阻断 `encode_future_images`、`future.forward`。

源码断言 `not self.model.gate.scales.any()`；本次 CPU 读取 initializers 也确认其值为 `[0,0,0,0,0,0,0,0]`。

教师侧适配来源链为 Round3 `teacher_terminal_hold36_20260926` 第200步 → Round4 `future_aux` 再200步 → Round6 `control` 再200步。训练数据来自 `robotwin_merged`，更新新增后验角色和动作适配部分，保留原始主要骨干；Round4/6 包含未来动作辅助监督。**这不是一个完成标准两阶段蒸馏的学生。** Round6 类名虽为 RetainedTeacher，选中的是 control 分支，不能由类名推断使用了 retention 分支损失权重。

Round10 使用固定 query 初始化，不是 Round7 学生 pilot 第200步更新后的 query，也不是旧的 Stage-I1000→Stage-II3000 权重。`robotwin/round7/POSTPILOT_DECISION.md` 明确指出那次200步学生训练未证明相对初始化的额外收益。新 Stage III 不加载含糊的 latest/best 或后续轮次权重。

### 2.3 架构现状

| 部分 | Round10 实际状态 |
|---|---|
| 原始 Action / Flow Query | 各8个，保留 |
| 原始 LaWM mapper/decoder | 保留；依据当前输入预测未来 latent |
| 动作 base 条件 | 当前 DINO、预测未来 latent、原始 VLM 全部输出 |
| 新 Goal / Local query | 各8个，固定初始化；经 VLM 后的表示随观测变化 |
| Goal/Local 动作条件、Plan、centered residual | 存在，使用教师侧适配所得参数 |
| VLM Goal→Local RelevanceGate | 模块存在，但零 scale，尚无有效部署消息贡献 |
| 真实未来后验编码 | 在线禁止调用；构造器中仍可能驻留相关对象 |

对象名中有 teacher 不等于在线读取未来；有 Gate 类也不等于已形成有效的跨角色消息学习。

## 3. 与三份参考文档的对应关系

依据：[三阶段总说明](../../reference/stage_I_II_III.md)、[Stage III 实施规格](../../reference/stage_III.md)、[Stage III 解释](../../reference/stage_III_explain.md)。文档以 LIBERO 已训练的 Stage-II Student 为前提，不能把其具体配置原样用于本候选。

| 项目 | 文档默认 | RoboTwin 本次需要 |
|---|---|---|
| 底座 | 已完成交互训练的 Stage-II Student | 原始SFT+Round4/6 adapters+Round7 fixed queries |
| 观测 | 主视角+单腕 | 主视角+左腕+右腕 |
| 控制 | 20Hz、8×7 | **30Hz、36×16、双臂绝对EEF** |
| flow内部buffer | 50×32 | 保留50×32 |
| 动作base memory | DINO+移除旧query的当前输出，L+240 | **当前DINO+LaWM预测latent+原始VLM完整输出，L+512** |
| VLM Goal→Local scale | 继承已训练数值 | 当前为零，直接倍率训练无效 |
| Actor、奖励和GRPO | 新Actor+完整轨迹回报 | 数学方法可复用，环境和倍率位置要适配 |

文档公式是 `Local_new = Local + eta*s[layer]*relevance*message`。当前 `s=0`，冻结底座后任意 eta 都得到同一个 Local；同观测、同动作噪声下不应改变动作分布。有限样本可能让 Actor 数值变化，但这不是有效交互学习。

**不能只将 scale 改为0.01便开始 Actor-only RL。** 冻结的随机 message 不会因为优化 Actor 的 log_prob 就自动学会有用消息。要严格实施文档，须先训练该通路，见路线G。

## 4. 路线R：动作残差备选分析（当前不实施）

### 4.1 改动位置与语义

`robotwin/round5/model.py::CenteredFlow.velocity` 当前计算：

```text
g = tanh(residual_gate) * 3
v = v_ref + g*(v_on-v_ref) - g*(v_off-v_ref)
```

`v_ref` 为 bypass_roles 参考分支；`v_on` 为角色有效的分层分支；`v_off` 为相同分层分支但 role_valid 置零。三者保留现有定义。动作 residual_gate 与零初始化的 VLM RelevanceGate.scales 是不同参数。

建议新增每动作段一个 eta：

```text
g_eta = eta * g
v_eta = v_ref + g_eta*(v_on-v_ref) - g_eta*(v_off-v_ref)
```

保留求值顺序、dtype和reference/off分支；不代数合并成另一种浮点运算顺序。一个 eta 用于整个动作段的10次Euler更新。

| eta | 路线R含义 |
|---|---|
| 1 | 恢复当前Round10残差强度，必须数值验收 |
| 0 | 关闭新增centered动作残差，保留当前包装器的参考分支 |
| 小于/大于1 | 减弱/增强残差 |

R的eta=0不是文档的“仅关闭VLM Goal→Local消息”；也不能未经精度和前向验证就等同原始LaWAM发布模型。

只新增并训练Actor。冻结Qwen、DINO、LaWM、DiT、原query、新query、Plan、Goal/Local投影与attention、动作residual_gate及VLM RelevanceGate。首版只控制一个位置，避免无法归因。

### 4.2 完整前向与维度

B_e为活动环境数，首版保持1；L为包含原16个query的当前上下文长度，随指令变化。

| 步骤 | 张量或操作 |
|---|---|
| 输入 | main `[B_e,3,256,256]`；wrist `[B_e,2,3,256,256]`；instruction |
| 原始Qwen路径 | 保留8 Action+8 Flow Query，输出 `[B_e,L,2048]` |
| 原LaWM | Action-query `[B_e,8,2048]` 经mapper/decoder产生预测未来latent `[B_e,256,768]` |
| 当前DINO | 主图得到 `[B_e,256,768]` |
| 动作base | 两组视觉token+enc_vlm(original)，`[B_e,L+512,768]`，保留原masks |
| 独立角色VLM路径 | C+8 Goal+8 Local，`[B_e,L+16,2048]`；保留mask、RoPE、DeepStack |
| 第8个block后 | 尚未最终norm；G/L各 `[B_e,8,2048]`，Current `[B_e,L,2048]` |
| Actor输入 | G/L槽均值，Current有效位置均值（含旧query、排除padding）；各做无参数LayerNorm，按Local/Goal/Current拼成FP32 `[B_e,6144]` |
| Actor | Linear(6144,256)→SiLU→Linear(256,2)，**1,573,634参数** |
| 输出 | alpha/beta、z/eta各 `[B_e]`；倍率广播 `[B_e,1,1]` |
| 角色VLM继续 | R保留零scale gate；最终norm后G/L各 `[B_e,8,2048]` |
| 动作角色条件 | G/L各投影为 `[B_e,8,1024]`，role_valid全真 |
| DiT | 分层分支4 Plan+50 Action，`[B_e,54,1024]`；bypass参考分支保持原逻辑 |
| flow | 噪声与速度 `[B_e,50,32]`；10次Euler；前36个时间位置有效 |
| 最终动作 | 归一化 `[B_e,36,16]`；原RoboTwin后处理后逐步执行 |

当前代码包含原始条件与角色条件两次Qwen前向，首版不合并。通过隔离wrapper/callback取得角色第8层特征，不为Actor额外重跑VLM。R中eta不改变角色隐藏表示，仅改变centered velocity；G才改变后8层Local。

保留Round10的BF16 Gaussian draws、FP32 Euler状态、autocast和噪声种子协议；Actor RNG与动作RNG分离。保留 replan_steps=36、action_ensemble=False、use_state=False、双臂EEF四元数和夹爪转换。禁止套用LIBERO的7维动作、二值夹爪或8步执行器。

### 4.3 奖励训练前的诊断

1. 同组合权重、观测、噪声、设备与精度，对比旧Round10和新增wrapper的eta=1；先原样透传，再Actor均值路径。记录各速度分支、10步轨迹、最终动作误差。
2. 固定观测测试eta=0/0.8/1/1.2，记录g、v_on-v_off范数和有效动作通道的变化；初期在线只测0.8/1/1.2，扩大范围须先验证。
3. 先测同设备重复误差再定容差。历史GPU5 preflight有单独容差，不能当作新wrapper通过的依据，也不能将跨设备误差当Actor收益。
4. 倍率不改变动作时暂停，检查残差/注入；改变动作却只导致退化也不意味着必有可学习收益。只有控制路径有效才进入采样训练。

## 5. 路线G：严格文档方案的前置交互训练

### 5.1 建立兼容Student

从Round10精确组合状态另建候选，不恢复旧StageII3000。保留原LaWM、旧query、36×16规格及centered flow，动作残差倍率固定为1。

- **零扰动起点：** 继承fixed queries与零scale gate，step0恢复Round10。冻结原骨干和已有动作模块，只开放新query、RelevanceGate relevance/message/scale白名单参数。
- **短程训练：** 在robotwin_merged的episode互斥划分上，使用动作监督，并按证据加入固定兼容后验Teacher的Local/Goal多层蒸馏及Round10输出保持约束。沿用正确未来采样、36步绝对动作末尾保持和16有效通道。
- **初始预算：** 200步pilot，每50步检查；达标后才考虑500/1000步。这是新增“兼容学生交互训练”，不是Stage III，也不是承诺足够收敛。

零scale时message参数起初可能无梯度，scale可先收到梯度，之后message才开始更新。必须检查真实梯度与参数变化，无需在部署时突然打开随机消息。

候选损失为 `L_action + lambda_local*L_KD_local + lambda_goal*L_KD_goal + lambda_keep*L_keep`。保持项使用同输入、同噪声/时间条件的新旧动作或参考轨迹速度，不比较独立噪声生成的动作。权重根据梯度量级与验证结果校准，不机械沿用旧实验。Teacher锁定兼容版本并先核对后验目标质量；真实未来仅离线使用。

保留query-only与query+gate的等预算对照。Round7先前query pilot未证实收益，不能假设此次训练必然改善。

### 5.2 进入Stage III的条件

- s*r*M非零且有限；同状态倍率变化带来Local和动作的可测变化。
- 单向交互不直接改变同状态Current/Goal；没有真实未来泄漏。
- 离线动作误差、尾部压力样本和独立闭环验证无不可接受退化；不以蒸馏loss下降代替在线验收。
- 固定新兼容Student的准确权重、步数、gate值与成绩。它是新起点，**不自动继承91.45%成绩**。

达标后冻结该Student，第9–16层block后以eta乘原s*r*M，严格实现文档Stage III。此时eta=0仅关VLM消息，Goal→Plan仍保留；eta=1对齐该新Student。R/G Actor分别标记 `control_site=residual_strength` / `vlm_goal_local`，禁止交叉加载。

## 6. RoboTwin交互、同初态分组与奖励

### 6.1 环境协议

模型环境为 `/home/wangchaoyang/envs/LaWAM/bin/python`；仿真环境为 `/home/wangchaoyang/envs/RoboTwin4LaWAM/bin/python`，runtime为 `/mnt/hdd/wangchaoyang/results/LaWAM/runtime/RoboTwin`。实施时记录版本、assets、任务及随机化配置。

复用 `robotwin/round10/bridge.py`、`client.py` 的观测/动作/setup约定，新增隔离RL adapter。现有评估队列依次寻找有效seed，不能直接作为GRPO分组器。

`_try_seed_once()` 会调用专家play_once筛查场景，再次setup_demo后交给模型。**专家probe成功不是策略奖励。** 场景预筛标准及拒绝数须记录，对各策略一致；一组是同一场景的多次模型执行，不是4个不同seed。

### 6.2 同一实际初态的4条轨迹

组内共享task、mode、instruction、场景资产与物体/机器人初态、相机/纹理/随机化配置、稳定步骤；gate RNG与flow RNG在各轨迹独立采样。

manifest保存 `group_id/task/mode/instruction/reset_seed/accepted_scene_id/scene_fingerprint/env_config_hash/setup_protocol_version/actor_version`。fingerprint至少覆盖物体身份/位姿、机器人关节及控制状态和随机化参数。首帧可作诊断，不能单靠图像hash证明初态一致。

不能套用LIBERO的set_init_state接口。先验证相同seed和完整setup参数是否重建同场景，特别排除now_ep_num、slot、进程RNG差异。不能复现时，扩展状态导出/恢复或固定场景生成参数，覆盖物理与控制器状态。**组内初态未通过验收前，不进入GRPO。**

4条轨迹可串行，不要求4个模型并驻显存。整轮采样固定behavior Actor，收齐后统一更新；同组内绝不换权重。

### 6.3 单轨迹执行和缓存

```text
恢复本组初态 → 当前三视角和指令 → layer8特征u
→ Actor采样z与eta → 生成36×16动作段
→ 原后处理，逐次take_action，逐步检查eval_success/step_lim
→ 成功或失败立即终止；未终止则取新观测和下一次eta
→ 保存整轨迹终局回报和完整gate trace
```

一次动作段只采样一个eta，10次Euler复用。记录每段起始步和实际执行步数；成功后未执行的动作不计数。T取策略接管后实际take_action_cnt增量，与成功检查一致，不计初始化、专家probe或Euler迭代。

现有bridge会将take_action异常记成episode_done(success=False,error=...)。RL adapter必须区分正常失败、step_lim超时、基础设施异常和策略非法动作，不能一律转成正常零奖励。

### 6.4 奖励与30分钟规则

沿用文档首版：`R = success * [1 + 0.1*(1-T/T_max)]`。T_max读当前任务step_lim；失败为0，无提前失败奖励。不增加reward model，不使用未来标签作为在线Actor输入。

平均reward提高可能只是成功轨迹更快，不保证成功率上升；组内标准化可能放大全成功组的效率差异，单独统计全成功、混合、全失败组的贡献。

遵循用户的**加载完后30分钟无任务进度则跳过**要求：ready后，实际环境步或完成episode等有效进展1800秒未更新，结束本run拥有的job，隔离未完成组，继续后续任务。普通日志/RPC心跳/反复加载不无限重置计时；加载和初始化另设状态及超时。

基础设施错误的不完整组不参与优势计算，按预定有限重试规则处理并报告成本；不能只保留成功轨迹凑组。正常step_lim失败必须保留为零奖励。策略非有限动作触发配置诊断，不能不断丢弃坏样本美化成绩。

## 7. Actor与GRPO-style实现

### 7.1 数学核心和建议配置

- FP32 Actor：alpha/beta=1+softplus(h)+1e-6，初始化均为20；z~Beta，eta=2z，部署用2*alpha/(alpha+beta)。初始eta标准差约0.156，随机Actor不等同eta=1。
- 若随机探索明显退化，另立更高浓度或较窄支持区间配置，并重新采样；不可运行时裁剪后仍使用原密度。
- 每组4条轨迹，`A=(R-mean)/max(population_std,0.01)`；std≤1e-6则整组优势0。
- 同轨迹各决策共享优势；组等权、组内轨迹等权、轨迹内决策平均，不让长轨迹获得更大权重。
- ratio=exp(logp_new(z|u)-old_logp)，clip=0.2；首版每轮只更新1次。最多2次更新须显式配置。
- 仅训练约157万参数Actor；无Critic、无在线action flow loss、无在线Local/Goal蒸馏；不通过模拟器反传。

建议pilot用Adam、lr=3e-5、grad_norm≤1、old→new KL阈值0.01，entropy/reference-KL系数先为0。这些是保守启动建议，不是验证过的最优值。

### 7.2 复用现有实现时的必查问题

`stage_III/libero_plus/gate_actor.py` 已有Actor/reward/advantages/clipped loss可参考。其policy、rollout、loader绑定LIBERO step4000、双视角、20Hz等，不能原样复用。

1. **显式关闭Actor autocast。** features.float()不阻止外层BF16 autocast影响Linear。Actor前向、采样、log_prob、KL和更新全部在autocast(enabled=False)下；缓存实际FP32输入、原始z、alpha/beta、old_logp。CPU/GPU更新差异也要验证。
2. **首次更新前ratio≈1。** 使用同版本Actor、同缓存u/z重算log_prob，超FP32容差则拒绝更新，检查dtype、TF32、设备、预处理与版本。不能事后改写old_logp；Beta.log_prob输入是z，不是eta。
3. **全零优势必须跳过optimizer.step。** 现有参考train_iteration仍调用step，即使日志updated=false；Adam历史动量可能改变参数。新实现需要真正跳步。
4. **更新后重算KL。** 区分初始一致性和真实策略漂移；超阈值停止后续更新，严重异常回滚Actor/optimizer并重新采样。
5. **普通tensor缓存。** 收集可用no_grad；若inference_mode，更新端重建普通FP32 tensor，避免autograd保存inference tensor错误。
6. **完整组与行为版本。** 不完整组整组隔离；只在轮间热更新Actor，全worker确认SHA后开启下轮；不无限复用旧轨迹。
7. **完整冻结审计。** CurrentQueryStudent将teacher放在__dict__，仅遍历student.parameters/state_dict不能覆盖共享底座。显式检查teacher/flow/mapper/decoder、student及buffers；optimizer白名单只有Actor。

每决策缓存features[6144]、z、eta、old_logp、alpha/beta、actor_hash、flow/gate种子、起始步/执行步；每轨迹存success、T、T_max、return、termination_reason和组身份。独立管理RNG；Beta.sample不传不支持的generator参数。

## 8. 划分、预算与对照

### 8.1 在线数据划分

按实际初态建立50任务×2模式的RL-train/val/test互斥manifest。离线robotwin_merged划分不自动形成在线划分。Round10反复查看过的seed属于开发历史，不能再称全新test。原始SFT是否覆盖相同场景可能无法完全证明，报告限定为可追踪的RL划分，不宣称未见任务泛化。

建议每组合准备至少20个train、5个val、10个test有效初态；这是配额建议，生成可用率和成本先审计。不能运行的组合保留缺失原因，不以排除困难任务制造全量成绩。

### 8.2 有界实施顺序

当前代码采用下列有界流程；轮数指完整收集/更新周期，不是旧 Stage-II 步数。扩大预算属于后续实验，当前不自动扩展。

| 阶段 | 已编码的默认预算 | 继续条件 |
|---|---|---|
| P0 固定身份 | 原始权重、两 adapter、initializers、统计与源码快照 | 加载顺序、36×16 与 LaWM 保留一致 |
| G0 交互准备 | query+gate 训练200步，50步保存/验证 | step0等价、实际梯度、Teacher筛查、冻结审计通过 |
| G1 敏感性 | eta=0/0.8/1/1.2，同观测和噪声 | Local/动作有响应，Goal不反向变化，eta=1等价 |
| G2 场景与闭环准入 | 8任务×2模式，train每组合2、val/test各3个真实初态 | 重复reset一致、split互斥、配对非劣与完整覆盖 |
| P3 G路线Stage III | 5轮×4组×4条，最多80条训练轨迹 | 同初态完整组、首次ratio≈1、仅Actor更新 |
| P3 保存与验证 | 每轮保存；第5轮同manifest验证 | 有效优势、KL稳定；异常和零优势明确记录 |
| P4 验证选优/最终对照 | 验证选Actor与最佳固定eta，锁定后测试 | Round10/交互Student/最佳固定eta/Actor均值配对比较 |
| 后续扩展（未自动开启） | 100组合、更长训练和更大test配额 | 全100组合准入、独立资源预算与收益证据 |

初始随机 Actor、训练 Actor 随机部署可用 evaluate.py 的策略入口另做对照；当前 pilot 不能据此直接声称状态依赖控制优于常量 Actor。
每完成一个任务模式的验证就记录配对差异，严重退化及时诊断；不等待全部任务结束才看。一个actor_version的采样轮中不边收集边改模型。

启动P3前按每轮清单预留含有限重试的训练环境步预算；专家probe、验证和测试开销另计。实际吞吐和显存仍需未来GPU运行验证。

### 8.3 如何判定提升

- 主指标：各任务模式成功率、模式内task-macro、clean/randomized等权总指标；另报总successes/episodes、覆盖和基础设施失败率。
- 次指标：成功耗时，以及双方均成功初态上的配对耗时，避免各自成功子集的选择偏差。
- 以初态为配对单位、按任务分层给bootstrap区间；组内4条及同轨迹多次决策不是额外独立场景。
- 对照至少含eta=1、val选出的最佳固定eta、未训练随机Actor、训练后的状态相关Actor。声称状态依赖收益还需同预算常量输入Actor；G另加关闭VLM消息消融。
- 48次pilot仅作筛查。正式部署前预注册非劣界，建议总指标1个百分点、单模式2个百分点，并结合配对区间判断；不能排除退化则记证据不足。这些阈值是建议，不是已通过的检查。
- reward提高但成功率下降不按reward选部署权重；若不胜最佳固定倍率，结论只限倍率校准。
- 未完成组合报告完整计划分母、缺失及共同有效集合结果；不能自动当成功或悄悄删除。

## 9. 本次代码交付

新实现全部位于 stage_III/robotwin；历史 robotwin/round* 与其他 Stage III 实验代码作为依赖读取，不原地改写。完整用法见 [README.md](README.md)。

| 文件 | 已编码职责 |
|---|---|
| base_loader.py / policy.py | 精确Round10组合加载、外部Teacher冻结、角色layer8特征、原LaWM/base条件 |
| prepare_interaction.py | query+gate监督训练、采样身份、KD/keep、离线验证和完整续训状态 |
| gate_actor.py / grpo.py | FP32 Beta Actor、直接消息倍率、完整组优势、概率/KL/零优势检查 |
| env_adapter.py / rollout.py | 实际初态和首帧一致、固定指令、逐步终止、完整组记录 |
| server.py / rpc.py / engine.py | 模型服务、版本绑定、传输、有限重试和拥有的子进程管理 |
| make_manifests.py / prepare_scenes.py | 候选种子与真实reset验证分开、实际状态去重、覆盖配额 |
| sensitivity.py / acceptance.py | 交互敏感性、完整配对验证后的Stage III准入 |
| train.py / evaluate.py / finalize.py | 有界在线训练、同步复用模型验证、验证选优后最终对照 |
| common.py / runtime.py | 原子保存/完整性标记、版本身份、互斥锁、30分钟无进度跳过 |
| workflow.py / scripts/ | 默认仅打印流程，真实执行需明确GPU和--execute |
| monitor.py | 只读进度/近期loss与GRPO指标，不启动任务 |
| configs/workflow.json | 本次G路线统一配置、200步交互和5轮RL pilot |
| tests/ / reports/ | CPU数值/控制流/脚本语法检查及报告 |

本实现没有 residual_flow.py 或 R 路线配置；倍率直接作用于第9–16层的原始 Goal→Local delta，不使用低精度 updated-local 差分重构消息。

`robotwin/configs/stage_I.json` 尚有horizon=8、horizon_sec≈0.2667等历史字段，兼容构造器实际覆盖为36步/1.2秒。必须保存并断言运行时解析配置，不能只复制该JSON作为RL规格。

首版保留构造器内未在线调用的后验对象以减少迁移变量；如以后为省显存裁剪，单独验收等价。不能直接删除名为teacher的整个共享对象，它还承载在线LaWM和动作模块。

## 10. 运行可靠性与交付

### 10.1 资源与进程

本次已写好代码并做CPU检查，不启动/停止已有实验。以后执行前重新检查GPU显存和占用；历史GPU0/3/5分工不代表当前可用。先一模型worker、一仿真slot通过smoke，再扩展。Actor可CPU更新，主要GPU成本仍为底座推理和仿真，不必起大模型DDP。

多GPU并行rollout时，每worker独立端口/cache/run_dir和请求身份，轮间同步Actor版本。只管理本run创建且归属校验通过的PID/进程组，不影响其它训练推理。

本次提供 scripts/progress.sh 只读终端监控，未创建 tmux 会话。后续实际启动时可将其挂入 tmux；显示阶段状态、近期 action/KD/keep 与 GRPO 指标。Stage III不用 action loss 下降代表进展。

### 10.2 防止中断或磁盘不足丢失进度

- 大底座只固定路径/hash，常规保存Actor/optimizer/RNG/config/计数，不每轮重复保存整套模型。
- 临时文件+flush/fsync+原子rename+完成标记；容量不足不覆盖last完整权重。视频抽样保存，trace定额管理并保留恢复必需数据。
- 轮内登记behavior Actor、完整组缓存、消费账本；轮末保存sampler cursor、manifest hashes和验证指标。
- 恢复以已提交轮边界为准；仅复用同一behavior版本未消费的完整组，禁止跨版本混组/重复消费。未完成episode不能仅凭seed假装中途续接。
- 有限重试后继续后续任务，报告异常成本；不用无限重启或强行纳入坏组保持表面进度。

### 10.3 真实实验验收清单（CPU检查不能代替以下运行验收）

- [ ] 锁定原始底座、两adapter、initializers、统计和源码/config版本。
- [ ] R/G控制位置及checkpoint身份明确，不混称研究方法。
- [ ] eta=1复现、36×16输出、LaWM和旧query保留。
- [ ] R有动作敏感性；G先有非零且有效的已训练交互。
- [ ] 当前观测限定、未来阻断、初态一致、逐步终止与奖励正确。
- [ ] 在线train/val/test互斥且可追踪。
- [ ] FP32概率、首次ratio≈1、仅Actor更新、零优势跳步、更新后KL有效。
- [ ] 30分钟无进展跳过、完整组处理、磁盘和恢复测试通过。
- [ ] 基线/固定倍率/Actor同协议对照，覆盖与不确定性完整报告。

最终交付：绑定精确底座的Actor权重、源码和配置、在线manifest、训练/验证曲线、逐任务两模式结果、成本与异常记录、可复现部署入口。**可实施的优化路径存在，但收益未被证明；当前零交互候选不能直接视作已具备文档Stage III前提的完整学生。**

## 11. 证据索引

| 内容 | 本地位置 |
|---|---|
| 三阶段定义与Stage III公式 | reference/stage_I_II_III.md、reference/stage_III.md、reference/stage_III_explain.md |
| 组合加载、未来阻断、精度设置 | robotwin/round10/server.py |
| 91.45%与历史比较口径 | robotwin/round10/candidate_online/status.json、robotwin/round10/candidate_queue.py |
| fixed queries与零gate | robotwin/round7/candidate_recipe.json、robotwin/round7/audit/initializers.pt |
| 两路前向、原LaWM预测latent | robotwin/round7/model.py |
| centered动作残差 | robotwin/round5/model.py |
| 教师适配来源、末尾保持、冻结DiT | robotwin/round3/model.py、robotwin/round4/train.py、robotwin/round6/model.py及各run记录 |
| 学生pilot结论边界 | robotwin/round7/POSTPILOT_DECISION.md |
| 角色VLM和RelevanceGate | goal_local/model.py、goal_local/roles.py |
| setup/probe/逐步成功 | robotwin/round10/bridge.py |
| 重规划、噪声和后处理 | robotwin/round10/client.py、examples/Robotwin/eval_files/model2robotwin_interface.py |
| Actor/GRPO可复用部分与限制 | stage_III/libero_plus/gate_actor.py、policy.py、train.py、group_policy.py |
