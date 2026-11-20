# Goal/Local 两阶段架构：三组消融实验设计

本文依据 [guide.md](guide.md)、[idea_9_20_change.md](/mnt/hdd/wangchaoyang/idea/idea_9_20_change.md) 及当前 `goal_local/model.py`、`goal_local/roles.py` 整理此前讨论。以下均为实验设计，不代表实验已完成。本次只更新文档，不启动实验或修改现有训练、推理。

“三组”指三个实验族，不是只训练三个模型。完整模型 Full 可作为三组共同参照。本文末尾保留原有 aba2 推理消融记录。

## 1. 来源、完整流程与创新假设

### 1.1 谁继承谁

初始参数 W0：

`/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`

架构基线：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2`。

```text
原始 LaWAM LIBERO SFT 参数 W0 + aba2 基线架构
                       ↓
继承当前视觉/语言、两组旧 query、DINO、动作 Flow/DiT
新增 Future Resampler、Goal/Local 动作路径和 Plan tokens
                       ↓
Stage I：当前信息 + 真实未来 → 后验动作 flow loss
                       ↓
             Stage-I 完整 checkpoint T
              ┌────────┴────────┐
       完整冻结 Teacher       初始化独立 Student
       继续接收真实未来        新建 Q_G/Q_L 和 gate
       仅生成蒸馏目标          只接收当前信息
              └───目标──→ 动作 loss + Local/Goal 逐层蒸馏
                                 ↓
                       Stage-II 可部署 Student
                                 ↓
                 当前双视角图像 + 指令 → 动作
```

LIBERO-Plus Stage I 从 W0 重新初始化，不继承此前 Goal/Local 在 LIBERO 上训练的教师或学生。训练数据只读复用：

`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_without_subgoal/libero-plus/data/indexed_rlds`

这里只复用数据，不继承 `LaWAM_without_subgoal` 的结构。此前指定 Plus Stage-I 第 5000 步作为 Stage-II 起点；正式消融需要记录实际 checkpoint 路径及哈希，不能仅用“5000步”标识来源。

### 1.2 保持一致的基线边界

- 当前主/腕视角及语言继续进入 Qwen；Qwen 视觉编码器和 merger、当前主视角 DINO 冻结。
- Qwen 文本嵌入、前 16 层语言骨干及动作模块参与训练。
- 原 8 个 Action Query、8 个 Flow Query 和原 prompt 保留。它们仍输入 VLM；去掉的是动作 base memory 中对应的 16 个输出位置。
- 旧 LaWM/未来视觉重建链路不运行；单独继承 DINO 不等于保留整套旧 LAM。
- 实际 aba2_change 不使用本体状态 `s_t` 或 State MLP，消融不能额外加入状态。
- 动作 horizon 为 8，通常执行 `B×8×7` 动作，部署使用 10 次 flow 更新。

### 1.3 新模块与维度

| 模块 | 完整模型设计 | 主要维度或层数 |
|---|---|---|
| Local 后验 | t+8 的主/腕双视角，经冻结视觉前端及 Resampler | `B×2×N×2048 → B×8×2048` |
| Goal 后验 | 主视角 t+[24,40,56,72,88]，带 horizon embedding | `B×5×N×2048 → B×8×2048` |
| Teacher VLM | `[C;F_G;F_L]`，两角色全程隔离 | 16 层、宽 2048 |
| Student VLM | `[C;Q_G;Q_L]`，只看当前信息 | 两角色各 8 个槽、宽 2048 |
| Student 交互 | 前 8 层独立；后 8 层显式 gated Goal→Local | 原生 attention 始终隔离 |
| 蒸馏 | 两角色独立逐层对齐，cosine + slot Gram | 第 4/8/12/16 层 |
| 动作投影 | Goal/Local 独立投影 | 各 `2048→1024` |
| Goal→Plan | Goal 注入 4 个 Plan tokens | DiT 第 1/5/9/13 层 |
| Local→Action | Local 注入动作 tokens | DiT 全部 16 层 |
| 动作侧联合计算 | Plan 与动作 buffer 共同 self-attention | `B×54×1024`，输出 buffer `B×50×32` |

N 随 processor 图像网格确定，当前指南的主/腕输入各为 `B×3×256×256`。50×32 是动作 buffer，不代表实际执行 50 步、32 维动作，监督和执行使用有效的 8×7 部分。

Stage I 只优化后验 action flow loss。Stage II Teacher 完整冻结、独立前向，不运行教师动作头；Student 复制 Stage-I 公共模块及动作头，新建 query 和 gate：

\[
\mathcal L_I=\mathcal L_{action}^{post},\qquad
\mathcal L_{II}=\mathcal L_{action}^{pri}
+\lambda_L\mathcal L_L^{dist}+\lambda_G\mathcal L_G^{dist}.
\]

Teacher hidden state、KV cache 和后验动作条件不进入 Student 动作前向。部署删除 Teacher/Future Resampler。Stage II 缺失未来只屏蔽相应蒸馏，不关闭学生角色路径。

### 1.4 核心创新的准确表述

**用真实未来学习任务相关的 Local/Goal 后验表示，通过冻结教师逐层蒸馏迁移到 current-only 学生，再用受控跨时间尺度交互和角色化动作路径进行控制。**

对应三个待检验的贡献：

1. 任务驱动后验学习与先验蒸馏：训练时利用未来，部署时不访问未来。
2. 双时间尺度分工：近期结果服务执行，远期有序信息服务任务进程判断。
3. 受控融合与动作条件化：深层单向门控，以及 Goal→Plan / Local→Action。

继承权重、增加 query 或 loss 下降本身不是创新有效性的证据。“Local 执行、Goal 规划”是设计假设，不能仅凭命名认定已学到这些语义；本文也不据此宣称上述机制在文献中首次提出。

## 2. 三组实验的逻辑

| 组别 | 核心问题 | 主对照 | 能检验的贡献 |
|---|---|---|---|
| A：训练范式 | 收益来自 Plus 普通微调，还是未来教师监督？ | 匹配预算 aba2 SFT、同教师初始化但无蒸馏、Full | 区分数据适配与显式蒸馏收益 |
| B：角色分工 | Local/Goal 是否互补？ | Local-only、Goal-only、Full | 双分支系统价值 |
| C：交互与路由 | 两角色如何交互并影响动作？ | 无消息、无相关性门控、自由融合、统一动作注入 | 消息、门控、隔离和动作路由 |

逻辑链：**未来监督是否有用 → 应该传递什么表示 → 如何将表示用于动作。**

三组按问题衔接，不采用同时叠加多项改动的累积比较。每个变体与 Full 比较时，应明确唯一目标因素及无法完全消除的混杂。

## 3. 第一组：两阶段训练与蒸馏的价值

### 3.1 设置

| ID | 变体 | 初始化与训练 |
|---|---|---|
| A0 | W0 直接用 aba2 推理 Plus | 不训练，作为域适配前参考 |
| A1 | aba2 普通 current-only SFT | 从 W0 起，用同一 Plus 数据和动作监督，匹配 Full 总数据预算 |
| A2 | No-Distill | 与 Full 相同 T、相同 Student 结构与新增参数初值；λL=λG=0，仅训练 Stage-II 动作损失 |
| Full | 完整方法 | 相同 T，Stage II 动作损失 + Local/Goal 蒸馏 |

A1 按 Full 两阶段总样本曝光量匹配：若更新数为 K1/K2、有效 batch 为 B1/B2，则对齐 `K1×B1 + K2×B2`。batch 不同不能简单用 K1+K2 步匹配。另报告 GPU 小时；匹配样本预算不等于匹配计算成本。

A2 和 Full 固定相同 Stage-II 采样顺序、batch、学习率计划和训练步数，公共参数与新增 query/gate 初始值一致。A2 可省去无用 Teacher 前向，但报告算力差异。

### 3.2 目的与结论范围

- A1 对 A0：验证普通 Plus 数据适配的收益。
- Full 对 A1：验证完整结构和训练方法的总体价值；差异不能单独归因于蒸馏。
- Full 对 A2：直接检验相同后验初始化和 Student 结构下，Stage-II 蒸馏的增量价值。

**A2 已继承接受真实未来训练的 T，因此不是“完全无未来监督”的模型。** 单独比较 A2 不能证明整个 Stage I 必不可少。

若要进一步拆解 Stage I，可在本组增加“完整 Student 架构从 W0 直接做 current-only 动作训练”的辅助对照，新增模块随机初始化、匹配总样本预算。它检验未来训练路径整体，不替代 A2 对显式蒸馏的比较。

### 3.3 指标与结果解释

主指标是 Student 闭环成功率；记录验证 action loss、两项 distillation loss、训练时间及显存。

- Full 稳定优于 A1/A2：分别支持完整方法价值和蒸馏增益。
- Full≈A2：当前未证明显式蒸馏必要；收益可能主要来自 Stage-I 初始化或动作训练。
- A1≥Full：完整方法尚未超过普通 SFT，应检查教师目标可预测性、损失权重与优化难度。
- 更低蒸馏 loss 但更低成功率：对齐没有转化为控制收益，不能用 loss 代替最终结论。

## 4. 第二组：Local/Goal 双时间尺度与角色分工

### 4.1 设置及重训范围

| ID | Stage I | Stage II / 部署 |
|---|---|---|
| B1 Local-only | 只用 t+8 双视角后验及 Local→Action | 仅 Local query/蒸馏/动作路径有效；无 Goal 消息 |
| B2 Goal-only | 只用远期主视角后验及 Goal→Plan | 仅 Goal query/蒸馏/动作路径有效；无 Local 及跨角色消息 |
| Full | 两类后验独立压缩、Teacher 角色隔离 | 两类 query、蒸馏、门控及角色动作路由 |

B1/B2 均从 W0 重训各自 Stage I，再用各自教师初始化 Stage II。复制完整 T 后临时清零一路，只能作为敏感性诊断，不能代替单分支重训消融。

可保留被删除分支的 8 个占位位置以维持序列位置，但必须同时关闭其可见性、输出、蒸馏、动作注入及 gate 间接通路。占位不能成为可学习或可被读取的额外容量。若保留 4 个 Plan 位置，须注明 Local-only 中它们是无 Goal 条件的计算槽。

### 4.2 能说明什么，不能说明什么

- Full 对 B1：远期分支在近期分支基础上的增量价值。
- Full 对 B2：近期分支在远期分支基础上的增量价值。
- 两者均下降：支持双分支系统互补。

删支也减少有效信息、token 和路径容量，不能单凭下降认定“时间尺度语义分解”已被证明。Local 是双视角单时刻，Goal 是主视角多时刻，还混合了视角/帧数差异。接口长度相同不等于有效容量相同，需报告有效参数与 token 数。

本组可增加容量更接近的控制：**联合未来表示**。保留相同七张未来图像、视角/时间标签、16 个未来槽和 16 个 Student 槽，用联合压缩替代独立 Local/Goal 压缩，仍输出两个 8-slot 接口给原动作路径。尽量匹配参数量并报告残余差异。重点检验后验输入来源隔离与压缩分工，不同时更换动作头；该变体也要重训 Stage I/II。

### 4.3 怎样验证角色能力

报告 4 suite×7 类成功率，并在有可靠行为标注时分析抓取、运输、放置、超时等失败阶段。不能仅由最终 success 自动推断规划或执行失败。

Camera、Robot、Language 等是干扰类别，不天然等同于执行或规划能力。类别变化只是线索；长期能力需按预先定义的任务阶段、长度或实际完成行为判断。

辅助诊断不计为新实验组：

1. 在固定验证样本上分别关闭 Local/Goal 条件，观察 Teacher 后验动作误差。
2. 固定当前输入，替换不匹配的未来，观察动作误差及表示变化。
3. 固定 Goal horizon 标签，只置乱对应帧内容，检查时序敏感性；帧和标签一起重排可能只是集合排列变化。
4. 记录 slot 方差、相似度与有效秩，排查表示退化。

这些是条件利用诊断，不代替重训和学生闭环评估。Teacher 使用真实未来，低后验误差不等于 current-only 控制能力。

## 5. 第三组：跨角色交互与动作路由

本组拆成两个子表，避免同时改变 VLM 和动作头后无法归因。

### 5.1 C-a：Student 交互

固定同一个 Stage-I 教师 T，每个变体重新训练 Stage II，不在已有 Full 学生上临时切开关作为正式结果。

| ID | 修改 | 检验目的 |
|---|---|---|
| C1 No-Message | 禁用第 9–16 层全部 Goal→Local 消息，保留原生隔离 | 跨角色消息是否必要 |
| C2 Ungated-Message | 保留同层、同方向 message attention 和残差 scale，仅令 relevance=1 | 输入相关的 relevance/NULL 抑制是否有效 |
| C3 Free-Fusion | 移除显式 gate adapter，原生 attention 全 16 层允许 Goal↔Local；C 仍不能读角色 | 受控交互整体与自由融合的比较 |
| Full | 前 8 层独立，后 8 层 gated Goal→Local | 共同参照 |

当前 gate 形式为 `Local + scale × relevance × message`。此前“No-Gate”含义不够精确，这里区分“无消息”C1 和“无相关性门控但仍有消息”C2。

C3 同时改变交互层数、方向、参数量与实现方式，只能支持整体融合方案比较，不能单独证明“单向优于双向”或“深层优于浅层”。要作这两项独立结论，应在本组补充匹配交互层数和消息模块的对照。

记录 relevance 分布、NULL 概率和实际注入范数：`||scale×relevance×message|| / (||Local||+ε)`。relevance 高不一定意味着实际贡献大，需结合 scale 和消息范数；可视化本身不能证明因果。

### 5.2 C-b：动作条件化（扩展）

C4 **统一动作注入**：保留 Local/Goal 表示、投影与 Student gate，将角色 memory 合并后统一 cross-attention 注入动作，替代 Goal→Plan / Local→Action 的专门路由。

实施前锁定注入层数、attention 调用次数、Plan 槽保留方式和容量匹配策略。尽量保留 54-token 接口，记录参数和计算差异，不随实验结果修改定义。

动作路由在 Stage I 就参与训练，因此 C4 要从 W0 重训 Stage I/II。对已训练 Full 临时更换路由会带来权重与分布不匹配，只能作为推理干预。

在预算和容量差异受控时，Full 优于 C4 支持角色动作路由的价值，不能仅据此声称 4 个 Plan token 已学到可解释规划。

## 6. 公平训练与评估协议

### 6.1 训练控制

1. 锁定 W0 哈希、代码版本、数据 manifest、episode 级 train/val 划分；同一 episode 的帧不能跨两侧。
2. 固定动作归一化、夹爪语义、图像处理、H=8、未来索引和尾部 mask。indexed 数据夹爪已为 1=开/0=闭，不重复反转。
3. 同阶段固定有效 batch、累计样本量、学习率计划、优化器、冻结策略。GPU 数变化时通过梯度累积保持有效 batch，记录数值及随机性差异。
4. 正式关键比较建议至少 3 个训练 seed；资源有限先做单 seed 筛查，再重复，单 seed 明确标为初步证据。
5. 固定 checkpoint 选择规则：同一步数，或独立验证集选点。不能在最终测试集为每个变体各选最高值。
6. 可沿用教师 5000 步、学生 4000 步作为预先固定预算，但记录实际学习率日程。提前结束 10000 步计划不等于按 4000 步日程重训。

### 6.2 评估及历史部分抽样的适用范围

所有部署模型只输入当前图像和指令。固定模拟器版本、任务 manifest、初始状态、episode seed、动作后处理、控制参数、超时、flow 步数，并尽可能配对动作采样噪声。尤其核对各变体的夹爪映射和推理修复版本一致。

当前历史评估采用“23 类全量 + 5 个低成功率类少测”。该选择依据先前成功率，改变了类别权重；其微平均只代表该固定选择子集，**不能作为 LIBERO-Plus 全量成功率或用来证明全量达到特定精度目标**。

正式消融要求：

- 优先使用全量固定清单；资源有限时，在查看待比较模型结果前，固定独立的 28 类均衡随机抽样清单。
- 每个变体使用相同任务和初始状态，不根据本轮成功率动态删减困难样本或增加容易样本。
- 历史 8380 任务子集可单列为历史选择子集，注明选择依据、每类覆盖率和样本数，只解释该子集上的比较，不替代独立评估。
- 失败 episode 全部保留。基础设施异常单列并按统一规则重试，不静默删除，也不与任务失败混淆。
- 原权重基线与修复后的 Student 使用一致评估处理。不同预处理版本的历史数字不可直接作公平对照。

### 6.3 指标

4×7 每格报告 `成功数/已评估数`、成功率和计划任务数。

- 微平均：`Σ成功数 / Σ已评估数`，反映实际抽样分布。
- 28 类宏平均：各类成功率等权平均。无样本的类标缺测，不填零或静默忽略。
- 按原类别任务数加权的估计：仅在类内抽样有代表性、类别权重已知时使用，明确为估计值。
- 跨训练 seed 均值/标准差；对成功率差用配对任务、按任务或基础任务分簇的 bootstrap 区间，避免把相关变体和 episode 全当独立样本。
- 辅助指标：阶段性完成、执行步数、action loss、两项蒸馏误差、GPU 小时、显存和部署时延。

宏平均缓解类别样本量不同的权重问题，但不能自动修正类内选样偏差。不同类别和任务的效果可能不同，不预设完整模型一定全面胜出。

## 7. 重训范围与建议顺序

| 变体 | 可复用完整教师 T？ | 新训练工作 |
|---|---|---|
| A0 原始权重 | 不需要 | 无 |
| A1 aba2 SFT | 否 | 从 W0 做匹配预算 SFT |
| A2 No-Distill | 是 | Stage II |
| B1/B2 单分支 | 否 | 各自 Stage I + Stage II |
| B 联合未来表示（扩展） | 否 | Stage I + Stage II |
| C1/C2/C3 交互变体 | 是 | 各自 Stage II |
| C4 统一动作路由（扩展） | 否 | Stage I + Stage II |

建议先做正确性检查及 A1/A2，再做 B1/B2，第三组先做 C1/C2，根据资源补 C3/C4 和 B 组容量控制。

固定一个 T、重复 Student seed 的比较，其结论是“条件于该教师”。最终评估整个方法的跨 seed 稳定性，需要重复教师训练，而非只重复学生。

## 8. 不计入正式三组的正确性检查

- Teacher 完整冻结：参数/状态不变，固定输入的监督目标稳定。
- Teacher 角色隔离：只更换 Goal future，Local posterior 不变；反向同理。
- Student 无泄漏：eval 模式固定当前输入、动作噪声及 flow time，只换真实 future，学生表示和动作不变，蒸馏 loss 可以改变。
- 部署一致：同一权重、预处理、噪声下，删除 Teacher 后复现训练代码中 Student 的 eval 输出。
- 梯度有效：新增模块有有效梯度，Teacher/冻结视觉前端无梯度，缺失未来不产生 NaN。
- 条件被利用：检查新增注入是否长期近零，区分“结构连通”和“真正利用”。

这些是实验结论成立的前提，不是精度消融结果。Stage-I 后验动作误差、Stage-II 对齐误差、Student 闭环成功率应分别报告。

## 9. 结果记录模板与结论边界

| 模型 | 教师路径/哈希/seed | 学生步数/seed | 数据预算 | 微平均 | 28类宏平均 | 相对 Full 配对差/区间 | GPU小时 |
|---|---|---|---|---|---|---|---|
| Full | 待填 | 待填 | 待填 | 待测 | 待测 | — | 待测 |
| A1 aba2 SFT | 不适用 | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |
| A2 No-Distill | 同 Full | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |
| B1 Local-only | 专属教师 | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |
| B2 Goal-only | 专属教师 | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |
| C1 No-Message | 同 Full | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |
| C2 Ungated-Message | 同 Full | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |
| C3 Free-Fusion | 同 Full | 待填 | 待填 | 待测 | 待测 | 待测 | 待测 |

每行附 4×7 结果表、任务明细、评估 manifest 哈希及代码版本。C4/联合未来表示作为对应组扩展行，未完成的实验不填写推测数字。

结论围绕观测写，例如“同教师与学生初始化下，加入蒸馏提高 X 个百分点”，而不是“模型恢复了真实未来”。差异不稳定或区间跨零时，报告证据不足。简化模型也可能更好，实验应识别真正有效的模块，而非预设完整模型必胜。

## 附录：早期 aba2 推理消融记录（原文保留）

以下为原文，属于早期 LIBERO 推理干预，不是上述 LIBERO-Plus 重训方案。GPU 限制、路径和精度仅对应原实验，本次未重新核验，不作为当前执行指令。

### LaWAM_aba2: inference without LaWM and action-memory query outputs

Derived from LaWAM_aba1. Training, VLM prompts and forward, checkpoint weights, current DINO features, action normalization, 10 flow steps and 0.4-second action horizon are unchanged.

Enable both `--disable_lawm --remove_query_tokens`. LaWM decoder must not execute; future visual memory is `[B,0,768]`. After the unchanged VLM forward, remove exactly the 8 act and 8 flow placeholder output positions before the action expert, using the two position masks. VLM memory changes from `[B,L,2048]` to `[B,L-16,2048]`; its attention mask is filtered identically. Remaining token values/order and padding validity are preserved. This does not remove queries from VLM input or erase their indirect influence on other tokens.

Original LIBERO SFT checkpoint is read-only via `results/Checkpoints`. New outputs stay in this clone. All 40 LIBERO tasks run for 50 episodes each, simulator seed 0, original initial states and horizon limits, 20 Hz control, 8 executed actions per chunk, bf16. No videos. Persistent servers on physical GPUs 0,1,2,3,6,7 pull whole tasks from a shared queue. GPUs 4 and 5 are forbidden by the runner. Completed episodes are saved after every rollout, allowing resume.

Run: `bash runtime/run_dynamic_ablation.sh`

Results: `results/ablation/inference_without_lawm_and_queries/full_40x50_20260915/`.
The report, suite CSV, task CSV and JSON compare aba2 to aba1 (97.75%) and original LaWAM (98.70%). These are independent evaluation runs, not paired policy-noise draws. This is a test-time intervention with no retraining.
