# 双时间尺度未来学习：消融实验实施方案

**版本：2026-10-18，最终 13 项配置**  
**适用工程：`LaWAM_aba2_change/goal_local`**  
**实验目录：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation`**  
**评估范围：LIBERO、LIBERO-PLUS、RoboTwin**

本文固定本轮讨论最终确认的 A0–A4、B0–B3、C0–C3，不再沿用早期六项 C 组编号。结构事实依据附件 [D1]–[D3]；消融改动依据本轮讨论；为落实实验新增的规则标为“执行约定”。外部资料 [W1]–[W7] 仅用于核对评估协议及通用算法，不代表这些消融已经验证有效。本文未读取服务器源码、权重或实际运行日志，不将配置说明视作已完成实验。

## 1. 完整消融实验表格

### 1.1 A 组：多时间尺度未来知识的互补性

**研究问题：在总角色 Query 数相同的条件下，同时学习近端局部变化和远端趋势，是否优于只学习一种未来？**

| 编号 | 配置 | Local / Goal 槽数 | 未来学习及动作条件 | Student VLM 交互 | 验证目的 |
|---|---|---|---|---|---|
| A0 | w/o Future Supervision | 8 / 8 | 从原始 SFT 权重 W0 出发；保留双角色、双路径动作头，仅用动作 flow loss；不做本方法的未来后验训练和蒸馏 | 保留第 9–16 层 RelevanceGate，由动作损失训练 | 完整显式未来学习方案相对纯动作监督的价值 |
| A1 | Local Only | 16 / 0 | Local-only Teacher + Student；四层 Local 蒸馏；仅 Local→Action | 无 Goal→Local 交互 | 近端未来的独立作用 |
| A2 | Goal Only | 0 / 16 | Goal-only Teacher + Student；四层 Goal 蒸馏；仅 Goal→Plan→Action | 无 Goal→Local 交互 | 远端未来的独立作用 |
| A3 | Local + Goal (Independent) | 8 / 8 | 双尺度 Teacher + Student；四层双角色蒸馏；保留双路径动作引导 | 16 层均无额外 Goal→Local 交互 | 双尺度知识分别使用时的互补价值 |
| A4 | Full Stage II | 8 / 8 | 双尺度 Teacher + Student；四层双角色蒸馏；保留双路径动作引导 | 第 9–16 层训练 RelevanceGate | 在双尺度独立使用基础上，显式交互的额外收益 |

本组不做 Stage III。A1/A2 的 **Teacher Resampler 输出和 Student Query 都是 16 槽**；A3/A4 两个角色各 8 槽。这里控制的是角色 Token 总数，不是保证总参数量和计算量完全一致。

### 1.2 B 组：角色专门化蒸馏和双路径条件引导

**研究问题：为什么需要多层蒸馏，以及 Local→Action、Goal→Plan→Action 两种不同的条件路径？**

| 编号 | 配置 | Stage-II 蒸馏 | 动作条件路径 | VLM 交互 | 验证目的 |
|---|---|---|---|---|---|
| B0 | Full Stage II | Local / Goal 均在第 4、8、12、16 层 | 原双路径 | 保留并训练 | 完整参考模型，等同 A4 |
| B1 | w/o Distillation | 删除两类蒸馏，只保留 Student flow loss | 原双路径 | 保留并训练 | 相同后验初始化下，显式蒸馏是否必要 |
| B2 | Final-Layer Distillation | Local / Goal 仅在第 16 层 | 原双路径 | 保留并训练 | 中间层监督是否优于只对齐最终表示 |
| B3 | Shared Conditioning | 保留第 4、8、12、16 层双角色蒸馏 | Goal / Local 沿 Token 轴拼接；Plan 和 Action 均读取同一份 Future Memory | 保留并训练 | 按角色分配动作条件是否优于共享条件 |

本组均使用 8 个 Local + 8 个 Goal Query，只运行至 Stage II。B1/B2/B3 从同一完整 Stage-I checkpoint 重新初始化 Student；B1 只继承其权重，不再需要 Teacher 目标前向。

### 1.3 C 组：奖励引导的自适应跨尺度交互

**研究问题：在线任务奖励、逐 Local 槽的接收选择，以及双尺度表征作为 Actor 条件，分别有什么作用？**

| 编号 | 配置 | 逐槽 Relevance | GateActor 输入 | 外部倍率及训练 | 验证目的 |
|---|---|---|---|---|---|
| C0 | Original Stage II | 保留原计算，参数冻结 | 不使用 Actor | 固定 eta=1；不做 GRPO | 完整 Stage III 相对原 Stage II 的整体收益 |
| C1 | w/o Selective Relevance | 强制 r=1，绕过 NULL / Relevance 接收选择 | Local + Goal + Current | 独立训练状态相关 Beta GateActor，使用 GRPO | 逐 Local 槽选择接收 Goal 信息的作用 |
| C2 | Current-Only GateActor | 保留原计算，参数冻结 | 仅 Current；Local / Goal 输入块置零 | 独立训练状态相关 Beta GateActor，使用 GRPO | Actor 直接读取双尺度专门化表征的价值 |
| C3 | Full Stage III | 保留原计算，参数冻结 | Local + Goal + Current | 独立训练状态相关 Beta GateActor，使用 GRPO | 完整方法 |

四项绑定同一个完整 Stage-II Student。C1/C2/C3 只训练各自的 Actor；原 VLM、旧新 Query、RelevanceGate、message attention、逐层 scale、DINO、Action DiT 全部冻结。

**复用关系：A4 = B0 = C0。** 在同一 Benchmark、同一基础训练种子、同一 checkpoint 和评估协议下，可以复用权重及基线结果；它们不是三次独立训练或三个独立统计样本。13 个表格条目对应 11 种不同配置形态。

## 2. 所有消融继承的架构和训练规则

### 2.1 基础架构不能改错

以下是附件中的 LIBERO 实现事实，不是对 RoboTwin 配置的假定。[D1, §1–5、§8]

| 项目 | 完整模型中的设置 | 消融时必须守住的边界 |
|---|---|---|
| 当前输入 | 主视角 + 腕部图像 + 指令；`use_state=false` | 不因某项消融额外加入本体状态、历史或其他观测 |
| 旧 Query | 8 个 Action + 8 个 Flow Query | 所有配置保留；不是新 Local / Goal Query |
| 新角色槽 | Local / Goal 各 8 个，hidden=2048 | A1/A2 改为单角色 16 个；动作时间长度不随槽数改变 |
| Local 真值 | t+8 时刻的主视角和腕部图像 | 是同一时刻的两个视角 |
| Goal 真值 | t+[24,40,56,72,88] 的五张主视角 | 保留有序 horizon embedding，不替换为一张终点帧 |
| 视觉链 | Qwen vision、merger 和 DINO 冻结 | Future Resampler 在 Stage I 可训练，不能一起冻结 |
| VLM | 16 层；原生 attention mask 隔离角色 | Current 不读取角色；Goal / Local 不经原生 attention 互读 |
| Student 额外交互 | 第 9–16 个 VLM block 之后，Goal 单向更新 Local | Stage II 已经训练这条通路；Stage III 不是首次建立通路 |
| 动作头 | 4 个 Plan Token + 内部 50 个 Action Token | 保留 4 个 Plan；Local / Goal 条件不是拼进当前 base memory |
| 动作注入 | Goal→Plan 在 DiT 第 1、5、9、13 层；Local→Action 在全部 16 层 | 注入发生于该层原 block 之前；消融不擅改频率或层位 |
| LIBERO 动作 | 20 Hz；有效 chunk=8，动作维度=7；内部为 50×32 | 16 个角色槽不意味着预测 16 个动作步 |
| 动作训练 / 推理 | flow 训练重复噪声采样 B→2B；推理为 10 次 Euler 更新 | flow repeat 不是 GRPO 组大小，也不是推理积分次数 |

令当前上下文长度为 N_C，其中已包含旧 16 个 Query。所有正式配置的角色槽总数均为 16，因此 VLM 序列为：

$$
X\in\mathbb R^{B\times(N_C+16)\times2048}.
$$

送入动作头的 Current 输出按旧 Query 的实际位置删除 16 个 Token，再与当前 DINO 拼接。不要简单截掉末尾 16 个 Token，也不要删除新角色输出。角色单独投影到 1024 维，作为条件 attention 的 K/V。

### 2.2 权重来源和冻结范围

记 W0 为本 Benchmark 的原始 SFT 权重；T_LG 为完整双尺度 Stage-I Teacher；S_LG 为完整 Stage-II Student。

$$
W_0\xrightarrow{\mathrm{Stage\ I}}T_{LG}
\xrightarrow{\mathrm{Stage\ II}}S_{LG}
\xrightarrow{\mathrm{Stage\ III}}(S_{LG},\phi).
$$

Stage I 更新语言骨干、旧 Query、Future Resampler 和动作头。Stage II 冻结独立 Teacher，Student 继承公共参数和动作头，新增角色 Query 与 RelevanceGate，继续用动作及蒸馏损失训练。Stage III 冻结整个 S_LG，只更新 Actor 参数 phi。[D1, §3.1、§5.1；D2, §2.3]

**所有需要重训的消融必须从其规定起点开始。** 不能把 A4 继续微调成 A3、B1、B2 或 B3，再把结果当成从一开始就采用对应方法的实验。A1/A2 也不能从 T_LG 开始删掉一个分支，冒充从未使用另一尺度的单尺度训练。

### 2.3 损失定义统一

完整 Stage I 只有后验动作 flow loss；完整 Stage II 使用：

$$
\mathcal L_{\mathrm{II}}
=\mathcal L_{\mathrm{flow}}+\lambda_L D_L+\lambda_G D_G,
\qquad \lambda_L=\lambda_G=1.
$$

对角色 R∈{L,G}、蒸馏层集合 S：

$$
D_R=\frac{1}{|\mathcal S|}\sum_{\ell\in\mathcal S}D_R^\ell,
\qquad
D_R^\ell=\operatorname{mean}(1-\widehat Z_{R,S}^\ell\cdot\widehat Z_{R,T}^\ell)
+\gamma\operatorname{mean}\!\left[(\widehat Z_{R,S}^\ell\widehat Z_{R,S}^{\ell\top}
-\widehat Z_{R,T}^\ell\widehat Z_{R,T}^{\ell\top})^2\right],
\quad\gamma=1.
$$

帽号表示沿通道归一化；第一项的点积沿通道计算，再对有效样本及槽位平均。Gram 沿槽轴形成 K_R×K_R 矩阵，再按所有元素平均。Teacher 表示停止梯度。8 槽和 16 槽都使用 mean，不能因槽数翻倍把损失改成 sum。[D1, §5.5]

第 4、8 层取 block 输出；第 12 层取 Student gate 后输出；第 16 层取 gate 后再经最终 RMSNorm 的输出。A3 无 gate，取对应 block / 最终 norm 输出；B2 只保留第 16 层且不再除以 4。[D1, §5.4]

**未来缺失规则保持一致。** 不删除 episode 尾部样本、不跨 episode 取帧、不复制最后一帧伪造未来。Stage I 缺某角色未来时关闭该样本对应动作注入；Stage II 仅跳过相应蒸馏，已启用的 Student 角色仍参与动作生成。永久移除的角色与临时未来缺失必须用不同配置语义表示。[D1, §2.3]

## 3. A 组各配置的具体执行

### 3.1 A0：w/o Future Supervision

**验证什么。** 相同部署结构下，本方法新增的真实未来后验训练及蒸馏，是否优于仅靠动作监督学习。主要比较 A4 vs. A0；不能把差异全部归因于某一个蒸馏损失。

**怎么训练。** 从 W0 恢复公共网络及原动作头，直接建立 current-only 策略。新建两组各 8 个角色 Query、4 个 Plan、角色投影、双路径条件 attention 和深层 RelevanceGate；W0 未包含的参数按完整模型对应新增模块的初始化规则处理。保留双角色 native mask、原 Current / DINO 条件和全部动作路径。整个训练过程不加载本方法的未来 Teacher、不运行 Future Resampler、不读取真实未来作为模型条件或监督，只优化：

$$
\mathcal L_{A0}=\mathcal L_{\mathrm{flow}}.
$$

**执行约定。** A0 的离线 anchor 使用量及 optimizer 更新总预算按完整方法所用 Stage-I + Stage-II 预算匹配，使用相同有效 batch。可以按两段预算记录学习率，但两段均为 current-only 动作训练，不称为后验 Stage I。额外明确各段 LR、optimizer 重置规则及新增参数从何时开始更新；不能宣称训练历程或 FLOPs 严格相同。

**推理及检查。** 与 A4 一样输入当前双视角和指令，内部保留全部角色及 gate。检查不存在 Teacher / future 数据依赖；动作损失能更新角色 Query、gate 和动作头。现有 `make_stage_two` 要求后验 checkpoint，因此需要单独构造合法的 prior 初始化路径，不能伪造一个 Stage-I checkpoint 绕过检查。[D1, §5.1]

**结论边界。** 两组 Query 只是保留结构，其 Local / Goal 语义没有真实未来目标保证。W0 的上游训练历史也未在附件中完整审计，所以只能称“未使用本方法新增的显式未来监督”，不能声称模型从未接触过任何未来知识。A0 与 B1 不等价：B1 已继承 T_LG。

### 3.2 A1：Local Only，16 槽

**验证什么。** 把全部 16 个角色槽分配给近端未来，是否足以替代 8 Local + 8 Goal。核心比较 A3 vs. A1。

**Stage I。** 从 W0 独立训练 Local-only Teacher。只读取 t+8 双视角，经冻结 Qwen vision 和可训练 Local Resampler 输出：

$$
F_L\in\mathbb R^{B\times16\times2048},\qquad X_T=[C;F_L].
$$

没有 Goal Resampler、Goal 角色和 Goal→Plan 注入。保留 Current / DINO base 条件、4 个 Plan、完整原 DiT，只通过 Local→Action 在全部 16 层注入未来条件，以后验 flow loss 训练。

**Stage II。** 冻结该 16 槽 Teacher，从其公共权重初始化独立 Student，新增 $Q_L\in\mathbb R^{16\times2048}$，仅计算第 4、8、12、16 层 Local 蒸馏：

$$
\mathcal L_{A1}=\mathcal L_{\mathrm{flow}}+D_L.
$$

Local 动作条件为 B×16×1024。没有 Goal Query、Goal loss 或 Goal→Local gate。推理只使用 Student 当前输入及 Local 条件。

**必须注意。** Teacher 和 Student 同步改为 16 槽，蒸馏 Gram 为 16×16。不能保留一个能影响输出的空 Goal 分支；角色切片、位置编码、mask、validity 和动作条件都须支持 K_G=0。所有残留的“每角色固定 8 槽”断言均需检查。4 个 Plan 仍参与原动作计算，并不因 Goal 缺失而删除。

### 3.3 A2：Goal Only，16 槽

**验证什么。** 把全部 16 个角色槽分配给远端趋势，能否替代双尺度表示。核心比较 A3 vs. A2，辅助比较 A1 vs. A2 的逐任务结果。

**Stage I。** 从 W0 独立训练 Goal-only Teacher。只读取五张有序远端主视角；Goal Resampler 输出 B×16×2048，VLM 输入为 [C;F_G]。保留 4 个 Plan，在 DiT 第 1、5、9、13 层执行 Goal→Plan，再通过原 Plan / Action self-attention 影响动作。所有 Local→Action 注入从训练开始关闭。目标仍为后验 flow loss。

**Stage II。** 从该 Teacher 初始化 Student，新增 16 个 Goal Query。只蒸馏第 4、8、12、16 层 Goal：

$$
\mathcal L_{A2}=\mathcal L_{\mathrm{flow}}+D_G.
$$

Goal 动作条件为 B×16×1024。没有 Local 角色及 RelevanceGate；当前观测、原动作头、动作训练和推理协议保持不变。

**必须注意。** 五个未来时刻不变，16 个 Goal 槽不意味着增加为 16 张未来帧。Goal→Plan 仍在四个指定层注入，不为“补偿”删除 Local 而改成每层注入，否则实验多改变了一个因素。Teacher / Student 的 Goal 槽顺序一致，空 Local 分支不应进入 attention。

### 3.4 A3：Local + Goal (Independent)

**验证什么。** 不建立 VLM 额外跨尺度消息时，分别使用近端与远端知识是否已产生互补收益。A3 vs. A1/A2 是 A 组最直接的双尺度证据。

**怎么训练。** 复用完整 T_LG。从该 checkpoint 重新构造双角色 Student，Query 各 8 个；整个 Stage II 从第一次前向开始就不执行第 9–16 层额外 RelevanceGate。保留角色 native mask、四层蒸馏、Goal→Plan→Action、Local→Action：

$$
\mathcal L_{A3}=\mathcal L_{\mathrm{flow}}+D_L+D_G.
$$

**推理及检查。** 使用这个独立训练的无交互 Student。对同一输入，改变 Goal 角色输入不应通过 VLM 前向改变 Local；动作仍可因 Goal→Plan 通路而改变。Current 不得充当两类角色的隐式信息桥梁。

**必须注意。** 不能拿 A4 在推理时设 eta=0 替代 A3。Independent 只表示 VLM 原生及额外角色通路隔离，不表示两条动作条件互不影响最终动作。隔离通路也不等于角色使用完全独立的骨干参数。

### 3.5 A4：Full Stage II

**验证什么。** 与 A3 比较，检验在双尺度学习和双路径动作指导之外，监督训练的 Goal→Local 交互是否提供额外收益。

**怎么执行。** 使用完整 T_LG 训练两组各 8 槽的 Student，第 9–16 层训练原 RelevanceGate，保留全部四层蒸馏和双路径动作头，损失为 flow + D_L + D_G。评估时无新 Actor、无在线更新，等价于外部倍率 eta=1。

**复用条件。** 已有 A4 可直接用，但必须与新 A3 的数据、初始化来源、训练总步数、LR schedule、checkpoint 选择及 warm-start 规则一致。不一致时须注明历史协议差异；正式控制变量比较应统一训练协议，而非仅统一最终 step 数。

**A 组结论边界。** 单尺度与双尺度已匹配 16 个角色槽，但 Resampler 数量、条件路径、可见 attention 关系仍不同。Local 真值是双视角近帧，Goal 真值是五个远端主视角。因此结果支持“当前近端 / 远端表征设计的互补性”，不能单凭这张表声称已经排除了所有参数、计算或未来输入信息量差异。单尺度保留一项角色蒸馏，双尺度保留两项，每个角色权重均沿用 1；不要为匹配总 loss 而临时将单尺度权重翻倍，也不宣称总蒸馏梯度已被严格匹配。

## 4. B 组各配置的具体执行

### 4.1 B0：Full Stage II

**验证什么。** 作为蒸馏和动作条件路径消融的共同参照，直接对应 A4。

**怎么执行。** 复用同一个完整 Stage-II Student 的权重及评估结果。其监督为四层平均的 Local / Goal 蒸馏，加 Student flow loss；Query 各 8 个，深层 RelevanceGate 和双路径条件均开启。

**必须注意。** 同一训练种子下，B1/B2/B3 应与 B0 使用相同 T_LG，并配对初始化新 Query、gate 和所有公共参数。推荐先保存一份“Stage-II 初始化快照”再分出四个配置，避免构造顺序改变随机数消耗，导致相同 seed 却并非相同初始化。这个快照尚未进行任何 Stage-II 更新。

### 4.2 B1：w/o Distillation

**验证什么。** 已有后验预训练权重的前提下，只用动作损失是否足以让 Student 学会两类角色；直接检验新增表征蒸馏目标的贡献。

**怎么训练。** 从 B0 的同一 Stage-I 来源和同一 Student 初始化状态开始，保留两类 Query、深层 RelevanceGate、全部动作条件。将两类蒸馏权重设为零，并跳过 Teacher 目标前向，只优化：

$$
\mathcal L_{B1}=\mathcal L_{\mathrm{flow}}.
$$

角色 Query 和 gate 仍从动作条件路径接收梯度。训练数据只需当前观测、指令、动作及动作有效 mask；不需要读取未来图像来产生目标。推理方式与 B0 相同。

**必须注意。** 不能从 W0 替代 T_LG 初始化，不能从 B0 训练完的权重开始取消 loss。删除蒸馏只是删除监督项，不是冻结两组 Query 或移除角色分支。与 B0 比较时匹配 optimizer 更新数和原始 anchor 量，不因省掉 Teacher 前向而给 B1 额外训练步数。

**结论边界。** B0 优于 B1 支持 Stage-II 显式蒸馏有用；不能据此声称 B1 没有未来知识，因为它继承的语言骨干及动作头已经经过 Stage I。

### 4.3 B2：Final-Layer Distillation

**验证什么。** 在蒸馏总聚合权重相同的设置下，中间层表征对齐是否比只约束最终输出更有效。

**怎么训练。** 从与 B0 相同的初始化开始，只保留第 16 层的 Local / Goal 蒸馏：

$$
\mathcal L_{B2}=\mathcal L_{\mathrm{flow}}+D_L^{16}+D_G^{16}.
$$

Student 第 9–16 层 RelevanceGate 继续训练；Teacher 仍为完整冻结 T_LG。第 16 层两支都取最终 RMSNorm 后表示，Student 在此之前已完成该层 gate。Cosine、Gram、有效样本 mask、两个角色权重都不变。

**必须注意。** 唯一一层不再除以 4；也不把权重乘以 4 去“补偿”层数。完整模型原来使用四层平均，而非四层求和。只删除中间层的监督项，不删除中间 Transformer 层、关闭 gate 或 detach 中间 hidden。记录蒸馏及 flow 分项和梯度量级，防止归一化实现错误冒充多层收益。

**结论边界。** B0 vs. B2 检验这组指定中间层的价值，不证明第 4/8/12/16 层是所有层组合中的最优选择。

### 4.4 B3：Shared Conditioning

**验证什么。** 已获得双尺度表征后，让 Goal 主要指导 Plan、Local 直接指导 Action，是否优于两个入口同时读取全部未来条件。

**怎么训练。** 复用完整 T_LG，并从与 B0 相同的 Student 初始化开始。VLM、RelevanceGate、四层双角色蒸馏均不变。仅修改 Student 动作头的条件来源：

$$
G_A,L_A\in\mathbb R^{B\times8\times1024},\qquad
F_{\mathrm{shared}}=[G_A;L_A]\in\mathbb R^{B\times16\times1024}.
$$

原来的两个 attention 模块仍保持独立参数，只把各自的 K/V 都换成 F_shared：

$$
\begin{aligned}
P&\leftarrow P+g_i\operatorname{MHA}_{P,i}(P,F_{\mathrm{shared}},F_{\mathrm{shared}}),
&&i\in\{1,5,9,13\},\\
A&\leftarrow A+l_i\operatorname{MHA}_{A,i}(A,F_{\mathrm{shared}},F_{\mathrm{shared}}),
&&i\in\{1,\ldots,16\}.
\end{aligned}
$$

P/A 分别表示 DiT 的 Plan / Action Token，g_i/l_i 沿用原动作侧残差系数。它们不是 VLM 交互中的 s_ell。保留原 Goal / Local 投影、注入层位、执行顺序和全部 base memory。Stage-II loss 仍为 flow + D_L + D_G。

**推理及检查。** 使用按共享条件重新训练的 Student。检查两个动作入口确实收到同一份 16 Token memory；不能仍用角色 mask 把 Plan 限制为只看 Goal、Action 限制为只看 Local。共享的是信息来源，不是把两个 attention 模块强制共享参数。

**必须注意。** 不新增可训练 Fusion MLP，不把拼接误写为通道拼接到 2048 维，不把角色拼进原 768 维 Current / DINO base memory。K/V 从 8 变成 16 会增加注意力计算，必须报告；不能宣称完全等 FLOPs。

**结论边界。** 本表检验“相同分角色 Stage-I 预训练后的 Stage-II 条件组织”。Teacher 及初始化动作头本身已按双路径训练，因此不能扩大为“任意从头训练的共享架构都劣于双路径”。不在此配置中额外重训共享 Teacher，以免破坏 B 组统一初始化前提。

## 5. C 组各配置的具体执行

### 5.1 统一倍率及 Actor 定义

原 Stage-II 在第 9–16 层 block 后计算逐 Local 槽相关性 r_ell、Goal 消息 M_ell 和可学习残差系数 s_ell。Stage III 只在原残差外乘一个倍率：[D2, §1.3、§3]

$$
\widetilde L^\ell
=\bar L^\ell+\eta_k s_\ell r^\ell\odot M^\ell,
\qquad \ell\in\{9,\ldots,16\}.
$$

r_ell 为 B×8×1，M_ell 为 B×8×2048；eta 为每环境、每动作段一个标量，对本次前向的八个深层及所有 Local 槽广播。它不乘到 Goal→Plan 或 Local→Action 的动作侧 scale 上，也不在每次 Euler 更新重新采样。

冻结原模块参数不代表 r、M 或角色 hidden 数值恒定；它们仍随观测和前层 Local 的变化重新计算。s_ell 保留 checkpoint 的实际值和符号，不重置为 0.01，不截断至 [0,1]。

在第 8 个 VLM block 后、最终 RMSNorm 前提取 Local、Goal、Current。Local / Goal 各对 8 槽平均；Current 对有效上下文 Token 掩码平均，包含旧 16 个 Query、排除 padding。池化后各自做无仿射 LayerNorm：

$$
\widehat p_L,\widehat p_G,\widehat p_C\in\mathbb R^{B\times2048},
\qquad
u_k=\operatorname{sg}([\widehat p_L;\widehat p_G;\widehat p_C])
\in\mathbb R^{B\times6144}.
$$

Actor 保持 `Linear(6144,256) → SiLU → Linear(256,2)`。两个输出经 `1 + softplus + 1e-6` 形成 alpha / beta，训练时：

$$
z_k\sim\operatorname{Beta}(\alpha_k,\beta_k),
\qquad \eta_k=2z_k\in(0,2).
$$

沿用附件首版初始化 alpha=beta=20，最后输出层权重为零；确定性均值为 1，随机样本并不恒等于 1。C1/C2/C3 采用配对的同一初始 Actor 权重。Beta 的 mean / log_prob 行为可由官方实现核对；该倍率范围和初始化是本方案设计，不是 GRPO 的必然要求。[D2, §3.3；W3]

正式评估默认统一使用：

$$
\eta_k^{\mathrm{deploy}}=2\frac{\alpha_k}{\alpha_k+\beta_k}.
$$

训练采样、测试均值这一规则在 RL-val 检查后提前固定，不能只给某个消融改用更有利的部署方式。动作头仍保留原 flow 噪声，因此确定性倍率不等于整条策略完全确定。[D2, §10.1]

### 5.2 C0：Original Stage II

**验证什么。** C3 vs. C0 检验在原监督策略上增加在线任务奖励后训练的整体收益。

**怎么执行。** 直接加载 S_LG，保留原 relevance、message、scale 和全部动作路径，固定 eta=1，不加载 Actor，不收集训练轨迹、不更新任何参数。使用与 C3 相同的最终任务、初始状态、动作噪声约定和环境步数上限评估。

**必须注意。** 先验证加入外部倍率代码后的 eta=1 前向能恢复原 Stage II。输入图像、processor、精度和动作噪声必须配对。A4/B0/C0 只有在同一评估清单及模式下才能复用结果。

**结论边界。** C3 多用了在线反馈，不能写成与 C0 完全等训练资源。该比较支持 Stage III 整体增益，不单独证明动态倍率优于一切固定倍率。

### 5.3 C1：w/o Selective Relevance

**验证什么。** 在仍使用在线奖励和状态相关 Actor 的前提下，逐 Local 槽“是否接收 Goal 消息”的机制是否有用。核心比较 C3 vs. C1。

**怎么训练。** 从同一 S_LG 出发，冻结原模型。第 9–16 层绕过原 NULL / relevance 输出，始终强制：

$$
r^\ell\equiv1,
\qquad
\widetilde L^\ell=\bar L^\ell+\eta_k s_\ell M^\ell.
$$

保留原 message cross-attention、s_ell、原生角色 mask 和两条动作路径。Actor 接收完整的 [Local;Goal;Current] 第 8 层特征，按相同 GRPO 规则独立训练。在训练 rollout 和最终推理中都保持 r=1；不能只在测试时切换。

**“没有选择性”的准确含义。** 移除的是逐 Local 槽基于 NULL 的接收抑制。cross-attention 仍为不同 Goal Token 分配权重；Actor 仍可整体缩放消息，s_ell 也仍起作用。所以该项应称“移除逐槽 Relevance 选择”，不能称“完全无选择、均匀使用全部趋势”。[D1, §5.3；W4]

**必须注意。** Stage-II 权重不重训，原 message 已在存在 relevance 的条件下学得。设置 r=1 同时改变消息幅度和 Local 的输入分布；GRPO 只有一个共享标量，未必能恢复原逐槽、逐层抑制。记录各层有效残差相对 Local 的范数、倍率分布及动作变化，不要静默额外归一化消息。

**结论边界。** C3 更好可以支持“保留已训练的逐槽接收机制，在当前后训练框架下有利”。不能单凭结果把全部差异归因于语义选择，也不能证明一个从 Stage II 起就无 relevance 的重训模型必然更差。本配置严格保持最终讨论确定的共享 Stage-II 起点。

### 5.4 C2：Current-Only GateActor

**验证什么。** Actor 直接看到 Local / Goal 专门化表征，是否比只看到 Current 上下文更有利于决定交互强度。核心比较 C3 vs. C2。

**怎么训练。** 从相同 S_LG 和配对 Actor 初始化开始，保留原 relevance、message、scale 及两种未来表征的全部动作作用。只在送入 Actor 时替换输入：

$$
u_k^{C2}=[0_{2048};0_{2048};\widehat p_C]
\in\mathbb R^{B\times6144}.
$$

先按完整模型计算 Current 的掩码平均和分支归一化，再组装零块，不新增整向量归一化。6144→256→2 的 Actor 结构不改；训练 rollout、缓存概率重算、验证和部署均使用同一个屏蔽规则。零块只是 Actor 输入占位，不是把 VLM 中的 Local / Goal hidden 清零。

**必须注意。** Current 随当前图像和指令变化，C2 仍然是状态相关策略，绝不是“固定常量 Actor”。Local / Goal 正常参与深层交互及 Action DiT。缓存中的 features 必须是实际送入 Actor 的 [0;0;Current]；不能采样时屏蔽、计算新概率时却使用完整特征。

**结论边界。** Local / Goal 本来也由当前输入推导，所以 C3 优于 C2 表明专门化表示作为直接门控条件更有效，不是增加了新观测或真实未来。两者网络尺寸相同，但固定零输入会使第一层相应列无法从这些输入获得梯度，不能宣称有效可训练容量完全相同。

### 5.5 C3：Full Stage III

**验证什么。** 完整验证基于在线任务回报优化已有跨尺度交互的方案，同时作为 C0/C1/C2 的共同参照。

**怎么训练。** 冻结 S_LG，保留原逐槽 relevance，Actor 读取完整的归一化 Local / Goal / Current 第 8 层特征。每次动作段采一个 Beta 倍率，调整第 9–16 层消息，运行原动作头并执行。完整轨迹结束后计算回报，仅更新 Actor。没有 Teacher、未来帧输入、flow loss 或蒸馏 loss。

**怎么推理。** 读取当前双视角和指令，前 8 层提取特征，Actor 输出分布并取倍率均值，后 8 层及原动作头完成动作生成。每次新观测重新计算倍率；测试阶段不再计算优势或更新参数。

**必须记录。** 相对 C0 的成功率与配对成功步数变化；相对 C1 的残差幅度及逐槽机制差异；相对 C2 的表现增益及 Actor 特征条件差异。倍率随时间波动并不能自动证明它理解了任务阶段，需将解释限制在实际证据范围。

**C 组结论边界。** 当前四项检验 Stage III 整体收益、逐槽 relevance 和 Actor 表征条件。已删除最佳固定倍率及状态无关 GRPO 对照，因此本表不单独验证“状态自适应严格优于最佳全局倍率”。论文不能保留超出当前对照覆盖范围的结论。

## 6. C1 / C2 / C3 共用的在线训练协议

### 6.1 先分数据，再收轨迹

离线示范的 train / val 划分不能代替在线初始条件划分。建立互斥的 `rl_train`、`rl_val`、`final_test` 清单，记录 task、指令、实际初始状态、环境随机化配置及哈希。同一实际状态换一个 seed 不算不同测试条件。最终测试条件不用于奖励优化、选倍率、选 checkpoint 或调整特征输入。[D2, §4.1–4.2]

沿用已验证的环境 reset、稳定动作、相机和动作后处理。每组轨迹共享同一个实际初始条件；不同轨迹独立采样门控和动作 flow 噪声。组内采集期间 Actor 参数不变，不能一边收同组轨迹一边更新策略。

附件的首版建议是每轮 J=4 组、每组 N_grp=4 条完整轨迹。调试可先 J=1、N_grp=2；这只是流程检查，不能与正式配置结果混报。[D2, §4.1]

### 6.2 一轮训练实际做什么

```text
固定本轮行为 Actor 的参数快照。
按 rl_train 清单取本轮各组初始条件。
对每组，从相同实际初始条件执行 N_grp 条独立轨迹：
    当前观测 → 冻结 VLM 前 8 层 → 按本配置组装 Actor 输入。
    采样 z、计算 eta=2z，缓存实际 features、z 和 old_logp。
    冻结后 8 层执行对应 relevance 模式，再用原动作头生成动作段。
    逐环境步执行并检查成功或终止；未终止则读取新观测继续。
    轨迹结束后记录成功标记、实际执行步数和回报。
在每组内计算相对优势，赋给该轨迹的全部门控决策。
在同一缓存 features、同一已执行 z 上重算新概率，只更新 Actor。
按约定检查 KL，结束少量更新后丢弃本轮训练缓存。
下一轮必须使用更新后的 Actor 重新采样。
```

每个动作段只采一次倍率。成功发生在 chunk 中途时立即停止剩余动作，成功步数统计的是实际环境执行步数 T_i，不是 `8 × chunk 数`，也不是 Euler 次数。用于稳定环境的预备动作按既定口径单列。[D2, §4.3]

### 6.3 奖励、优势及更新目标

沿用附件首版终局回报：

$$
R_i=y_i\left[1+0.1\left(1-\frac{T_i}{T_{\max,i}}\right)\right],
\qquad y_i\in\{0,1\}.
$$

失败回报为零，没有提前失败的效率奖励。T_max 读取本任务的统一评估配置，不能给不同模型不同上限。成功率和执行效率必须分开报告；最大化这个回报不严格等于先最大化成功率再优化步数。[D2, §5.1–5.2]

组 j 内使用总体标准差 s_j：

$$
\widehat A_{j,i}=\begin{cases}
(R_{j,i}-\bar R_j)/\max(s_j,0.01),&s_j>10^{-6},\\
0,&s_j\le10^{-6}.
\end{cases}
$$

一条轨迹的所有门控决策共享该优势。全失败组或回报完全一致的组没有奖励梯度，但采样成本仍计入预算；整轮优势全零时跳过 optimizer step，不能不停补采到出现有利差异而不报告。

在缓存的实际 Actor 输入及 Beta 样本 z 上计算：

$$
\rho_{j,i,k}=\exp\bigl[\log\pi_\phi(z_{j,i,k}\mid u_{j,i,k})-
\log\pi_{\mathrm{old}}(z_{j,i,k}\mid u_{j,i,k})\bigr].
$$

$$
\mathcal L_{\mathrm{III}}
=-\frac1J\sum_j\frac1{N_{\mathrm{grp}}}\sum_i\frac1{K_{j,i}}\sum_k
\min\!\left(\rho_{j,i,k}\widehat A_{j,i},
\operatorname{clip}(\rho_{j,i,k},0.8,1.2)\widehat A_{j,i}\right).
$$

先对轨迹内决策平均，再对轨迹和组平均。直接把全部决策展平后 `.mean()` 会改变长短轨迹的权重。分母使用采样时保存的 old_logp，不重新生成旧概率；调用 `Beta.log_prob(z)`，不是 `log_prob(eta)`。概率及损失采用 FP32，不对仿真器或冻结动作头反向传播。[D2, §6]

这是一种 **GRPO-style 连续门控适配**：借用组内相对优势及 PPO 裁剪目标，没有 Critic；不逐项复刻原 GRPO 的 reference-KL 等设置。这里的组内归一化和轨迹长度平均也不能被称作原始期望回报的精确无偏梯度。[W1–W2；D2, §6.3]

### 6.4 统一超参数及预算口径

| 项目 | 本版沿用值或规则 |
|---|---|
| Actor 初始分布 | Beta(20,20)，三配置配对初始化 |
| 倍率范围 | eta=2z，训练及 Actor 部署均为 (0,2) |
| Actor 优化器 | Adam，lr=1e-4，betas=(0.9,0.999)，weight_decay=0 |
| 每轮采样后更新 | 最多 2 次整批 Actor 更新 |
| 梯度裁剪 | norm≤1.0 |
| 策略裁剪 | epsilon=0.2 |
| KL 监测 | old Beta 相对 new Beta 的 KL；超过 0.01 停止本批后续更新 |
| 其他目标 | 无 flow、蒸馏、Critic、额外熵奖励及 reference-KL loss |
| 验证 / 保存 | 附件建议每 10 轮；正式使用相同验证频率及评估量 |
| 模型选择 | RL-val 成功率为主；无明显退化的候选再比较成功步数 |

以上数值来自 [D2, §8.1] 的**起始建议**，不是已证实最优超参数。发生任何修改时，对 C1/C2/C3 使用相同规则并记录版本，不单独给某项更多调参机会。

**执行约定：主预算按累计环境步数匹配。** 三项使用相同组大小、任务采样分布、预算上限和更新规则。因成功早停及零优势组比例不同，实际轨迹数和 optimizer 更新数可能不同，不能声称这些数量天然全部相等。在完整组边界结束采样，报告必要的预算超出量，不把被预算切断的半条轨迹伪造为失败。验证交互成本单列。

首次更新前 ratio 应接近 1、KL 接近 0；更新后只有 Actor 参数改变。C1 必须持续采用 r=1，C2 必须持续采用屏蔽后的输入。训练结束保存 Actor 时同时绑定基础 Student 哈希、特征处理规则、relevance 模式及 eta 映射，不能只保存一个无配置的 MLP 权重。

## 7. 三个 Benchmark 的评估和公平性

### 7.1 每个 Benchmark 独立锁定训练链

| Benchmark | 本项目需锁定的设置 | 重点结果 |
|---|---|---|
| LIBERO | Spatial、Object、Goal、Long；Long 对应 `libero_10`；当前附件实现为双视角、8×7 有效动作、20 Hz | 四个 suite 成功率、40 任务汇总、Long 逐任务表现 |
| LIBERO-PLUS | 实际使用的 PLUS 训练数据及 checkpoint；官方任务分类文件、扰动配置和逐任务评估次数 | 官方 Total、七类扰动结果；不能用鲁棒性成绩替代长程任务分析 |
| RoboTwin | 本地版本 / commit、具身配置、实际任务集合、采集和测试 task_config、动作表示、视角及控制频率 | 逐任务成功率、指定任务集合平均、所声明测试条件下的表现 |

LIBERO 官方仓库提供固定初始状态的恢复示例，其整体任务体系不止本项目使用的 40 个任务。[W5] LIBERO-PLUS 官方包含 10,030 个任务配置及七类扰动，README 明确将 `num_trials_per_task` 从 50 改为 1。[W6] RoboTwin 2.0 官方文档列出 50 个双臂任务并提供独立环境配置文档；本文不据此假定用户本地已经采用同一版本、全部任务或某个 Easy / Hard 命名。[W7]

**评估次数的执行约定。** 本版 LIBERO 主表采用每任务 50 次、四个 suite 共 2,000 次执行 / 模型 / 训练种子，并固定具体初始状态清单；这是本实验选用协议，不宣称所有实现均使用该次数。LIBERO-PLUS 按官方主协议每任务配置 1 次，完整一轮为 10,030 次；重复完整评估时另记随机种子，不把它改称每实例 50 次。RoboTwin 的次数必须从已选定版本和评测脚本明确填入，附件未提供其本地正式数值。

**训练链不能混用。** 附件明确记录 LIBERO 使用过 Stage-I step 2000 Teacher，LIBERO-PLUS 的 step 5000 Teacher 属于另一套实验；RoboTwin 训练链尚未在这些附件中核实。[D1, 开头及 §6.3] 各 Benchmark 应绑定自己的 W0、Teacher、Student、processor 和动作统计。PLUS 成绩必须明确是否使用 PLUS 数据微调，不能与只用 LIBERO 训练的迁移评测混写。

**未来跨度不能机械复制。** 对每个 Benchmark 记录动作 chunk H、有效动作频率 f，以及未来索引定义。保持本方法近端 H、远端 {3H,5H,7H,9H,11H} 的规则时，应同时记录对应秒数 H/f；“固定索引比例”不等于“跨数据集固定物理时间”。附件只确认 LIBERO 的具体秒数，RoboTwin 的相机、动作通道和时间跨度需依据本地数据核定。同一 Benchmark 的所有配置使用相同规则。

### 7.2 离线训练的控制变量

**统一数据。** 各配置使用同一离线 episode 划分、frame anchor 清单和采样策略；单尺度实验不因某类未来缺失而过滤出更容易的训练子集。共用动作归一化统计、有效通道 mask、语言模板和图像处理规则。

**统一预算。** A1/A2/A3/A4 的 Stage-I / Stage-II 各阶段预算分别对齐，B0–B3 的 Stage-II 预算对齐；Teacher 复用也要记录其上游训练成本。A0 按 §3.1 匹配整体离线动作训练预算。报告可训练参数、有效角色 Token 数、训练耗时及推理耗时，不把“16 槽相同”写成“模型容量完全相同”。

**统一历史交接。** 附件的 LIBERO Student 曾从 step 250 权重继续，重置 optimizer 和 RNG，但延续全局 scheduler 步数；历史 run 也不是简单用 4,000 步重新配置的训练计划。[D1, §6.3] 要严格复用已有 A4，新增对照须按各自 step 250 权重执行相同交接，而不是加载 A4 的 warm250 权重。无法复现该历史协议时，应统一重训参考模型，并明确使用新协议。

**统一模型选择。** 在开跑前锁定各阶段预算、Teacher snapshot 规则、候选 checkpoint 频率和验证指标。附件历史 `best` 依据离线 validation flow loss，不是在线成功率。[D1, §6.2] 不能让某项在训练集择优、某项用最终测试集择优。B1 没有蒸馏项，尤其不能用不可比的“总训练 loss”选择它与 B0 的最佳模型。

### 7.3 成功率、效率和不确定性

LIBERO / RoboTwin 先算逐任务成功率，再对声明的任务集合等权汇总：

$$
\mathrm{SR}_{\mathrm{macro}}
=\frac{100}{M}\sum_{m=1}^{M}\frac1{n_m}\sum_{j=1}^{n_m}y_{m,j}.
$$

LIBERO-PLUS 主表沿用官方 Total 汇总口径，并另报七类扰动；不能把七类未加权平均擅自当作官方 Total，因为各类别任务数未必相同。

成功步数只统计成功执行；没有成功则记 N/A，不写 0。除各自成功轨迹均值外，对相同任务、初始状态和配对动作随机性下**双方都成功**的条件，报告步数差及配对数量。它可以减少成功集合变化带来的偏差，但并不替代成功率指标。[D2, §9.3]

关键配置尽可能配对运行 3 个训练种子，逐种子报告结果及均值、标准差。C 组三个 Actor 可以先共享一个基础 Student 做机制筛查，但这只反映 Actor 随机性；对完整训练稳定性的正式结论还应覆盖不同 Stage-II 基础种子。不能把同一 checkpoint 的重复 rollout 记成不同训练种子。

最终测试使用相同任务 / 初始条件清单、动作段执行长度、Euler 次数、图像和夹爪处理、成功判定及超时上限。动作噪声与 Actor 随机数分开管理，避免增加一次 Actor 采样就意外改变原动作噪声序列。

## 8. 工程落实、验收及结果记录

### 8.1 配置和目录

保留用户指定的层级。以下花括号表示目录集合，不表示必须创建同名花括号目录：

```text
/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/
├── abalation.md
├── aba1/
│   └── {A0,A1,A2,A3,A4}/
│       └── {libero,libero-plus,robotwin}/
├── aba2/
│   └── {B0,B1,B2,B3}/
│       └── {libero,libero-plus,robotwin}/
└── aba3/
    └── {C0,C1,C2,C3}/
        └── {libero,libero-plus,robotwin}/
```

每个 Benchmark 目录下按 seed 保存 `config`、`checkpoint_refs.json`、训练日志、验证指标、`eval_episodes.jsonl` 和 `metrics.json`。A1/A2 分开存 Stage I / II，C1/C2/C3 存独立 Actor 及绑定的基础模型标识。A4/B0/C0 使用权重引用，避免复制大模型造成版本分叉。39 个配置 / Benchmark 目录不意味着 39 次独立完整模型训练。

**以下字段仅为实现语义，不是声称现有 CLI 已支持：**

| 配置项 | 需要表达的内容 |
|---|---|
| 初始化来源 | W0 / T_LG / T_L / T_G / S_LG；路径、哈希、模式和训练来源 |
| 角色数量 | K_L、K_G，及 Teacher Resampler 相同的输出槽数 |
| 蒸馏配置 | 有效层列表、两个角色权重、Cosine / Gram 系数和 mean reduction |
| 动作条件 | local-only / goal-only / role-specialized / shared-memory |
| VLM 交互 | absent / learned-relevance / force-relevance-one |
| Actor 模式 | absent / local-goal-current / current-only；倍率、池化及归一化规则 |
| 训练协议 | batch、累计、步数、LR schedule、warm start、冻结白名单、RL 预算 |
| 数据及评估 | 划分清单哈希、环境版本、初始条件、动作统计、终止规则和随机种子 |

附件定位到的 Stage-I/II 入口是 `goal_local/train.py`，不是保留的旧 `starVLA` 入口；具体开关需在模型、角色、动作头及训练循环中实现。[D1, §10] 同一套公共源码读配置，不复制 13 套略有差异的训练代码。

### 8.2 开训前的最低验收

| 检查对象 | 必须满足的条件 |
|---|---|
| 全部配置 | 旧 Action / Flow Query 仍为 8 / 8，Plan=4，角色槽总数=16；动作长度不随槽数改变 |
| A0 | 不调用未来 Teacher / Resampler，不用未来有效性关闭 Student 角色 |
| A1/A2 | Teacher 与 Student 单角色均为 16 槽；mask、切片、Gram、动作 K/V 支持单角色，空分支不进入 softmax |
| A3 | 从训练开始关闭额外角色通路；Current 不读取角色，Local / Goal 不互读 |
| B1 | 蒸馏为零且无 Teacher 目标前向；Query / gate 仍收到动作梯度 |
| B2 | 仅第 16 层蒸馏，最终 norm 位置正确；唯一一层不除以 4 |
| B3 | 两个动作入口读取同一份 16 Token memory；层位不改，不残留角色专属 K/V mask |
| C0 | 同观测、同动作噪声下 eta=1 恢复原 Stage-II 输出；先定容差再比较 |
| C1 | 每个指定层均 force r=1，训练和测试一致；message、scale 及 Actor 未被误关 |
| C2 | 仅 Actor 的前两块特征为零，Current 有效；VLM 角色与动作条件保持正常 |
| C1/C2/C3 | 更新前 ratio≈1；更新后原模型参数和 buffer 不变，仅 Actor 更新 |
| RL 环境 | 同组恢复相同实际起点；逐步计成功；缓存条数等于门控决策数，执行步数正确 |
| 所有部署 | 不读取真实未来、未来有效性、示范剩余长度或尚未发生的奖励 |

只测 loss 能否下降不够。至少还需检查预测动作是否有限、动作反归一化是否与基线一致、环境 reset 可否复现。基础设施错误与策略失败分开记录，不能反复删除不利轨迹来改善成绩。

### 8.3 实际需要启动的训练

以下按“一个 Benchmark、一个训练种子，且完整 T_LG / S_LG 已可按同协议复用”计算：

| 配置 | 新增训练工作 |
|---|---|
| A0 | 1 条 current-only 动作训练流程，按完整离线预算记录 |
| A1 | 1 个 Local Teacher + 1 个 Local Student |
| A2 | 1 个 Goal Teacher + 1 个 Goal Student |
| A3 | 1 个无交互双尺度 Student |
| B1 / B2 / B3 | 各 1 个对应 Student |
| A4 / B0 / C0 | 无新增训练，共用完整 Stage-II 基线 |
| C1 / C2 / C3 | 各 1 个独立 GRPO Actor |

合计 **9 项离线阶段训练工作 + 3 项在线 Actor 训练工作**。A0 可能分段调度，但这里计作一条动作训练流程。这不是 12 个完整三阶段模型；Teacher 和 Student 的成本必须分开。补充基础训练种子或无法复用历史协议时，额外计入完整参考模型训练。

**执行顺序。** 先锁定基线及数据清单，完成 C0 前向兼容和环境检查；再启动 A1/A2/A3、B1/B2/B3 及 A0；RL 缓存和概率检查通过后启动 C1/C2/C3。各配置仅在验证集选模型，随后用锁定的最终协议统一评估。不要因固定 checkpoint 名称含 `best` 就跳过基线重评。

### 8.4 每个运行最少保存什么

| 记录 | 必需内容 |
|---|---|
| 运行身份 | 配置编号、Benchmark、基础训练 seed、Actor seed、源码 commit |
| 来源 | W0 / Teacher / Student 路径及哈希、processor、动作统计标识 |
| 训练成本 | 原始 anchor 数、optimizer 更新数；RL 环境步数、完整轨迹数、Actor 更新数及耗时 |
| 配置专属检查 | 槽数、有效蒸馏层、条件 memory 长度、relevance 模式、Actor 特征模式 |
| 每次测试 | task、初始状态 / 随机化标识、动作与门控随机性、成功、实际步数、终止原因 |
| 汇总 | 逐任务 / suite / 扰动指标、配对成功步数及样本数、种子结果和不确定性 |
| C 组诊断 | eta 分布、逐层相对残差范数、奖励差异、零优势组比例、ratio / KL / 梯度 |

结果表中的性能数值留空待实测，不填写假设成功率。无成功的步数栏为 N/A；未运行的配置明确写“未运行”，不能用另一项结果代填。共享基线标注 A4/B0/C0 的对应关系。

## 9. 论文中各项比较能够支撑什么

| 关键比较 | 可支持的结论 | 不应扩大成的结论 |
|---|---|---|
| A4 vs. A0 | 本方法完整显式未来学习方案的收益 | 所有增益均来自 Stage-II 蒸馏，或 A0 完全没有未来知识 |
| A3 vs. A1/A2 | 总角色槽数相同时，当前双尺度设计的互补价值 | 所有参数、计算、视角及未来信息量均已严格匹配 |
| A4 vs. A3 | 在完整离线学习中启用深层 Goal→Local 的收益 | Stage III 首次建立了跨尺度交互 |
| B0 vs. B1 | 相同后验初始化下，新增角色蒸馏的价值 | B1 完全未接触真实未来知识 |
| B0 vs. B2 | 指定多层蒸馏相对仅最终层的价值 | 该四层选择对所有模型都最优 |
| B0 vs. B3 | 相同后验预训练下，分角色动作条件的价值 | 任意共享条件架构从头训练都较差 |
| C3 vs. C0 | 在线任务奖励后训练的整体收益 | 与纯离线策略完全等训练资源 |
| C3 vs. C1 | 保留逐槽接收机制在当前后训练框架下的价值 | 已彻底分离语义选择、消息幅度与预训练分布变化 |
| C3 vs. C2 | Local / Goal 专门化特征直接作为 Actor 条件的价值 | 引入了额外真实未来观测，或已经战胜最佳固定倍率 |

主线保持为：**先检验双尺度未来是否互补，再检验这些知识如何学习和用于动作，最后检验在线执行奖励能否改善已有交互的使用。** 对应机制没有观察到稳定增益时，如实报告，不用配置名称替代实验证据。

## 10. 依据和待锁定信息

### 10.1 用户提供的架构依据

| 标记 | 本次实际附件 | 使用范围 |
|---|---|---|
| D1 | `stage_I_II(2).md`；文内核对日期 2026-10-11 | Stage I / II 的数据、维度、损失、冻结范围及历史训练交接 |
| D2 | `stage_III(2).md`；实施参考 v1.0，2026-10-12 | Beta Actor、残差倍率、回报、GRPO-style 更新、在线划分及部署 |
| D3 | `stage_I_II_III(2).md`；整理日期 2026-10-12 | 三阶段总体边界及两份独立文档的合并参考 |

A1/A2 的 16 槽、B3 的共享 memory，以及 C1/C2 的新消融模式，是本轮讨论确认的实验改动，不冒充附件已实现代码。前一版汇总中“A 组 7、8、16 等 Query 设置”为笔误；本版只采用明确的 **16/0、0/16、8/8** 新角色槽分配，旧 Query 仍为 8/8。

### 10.2 外部核对来源

检索日期：2026-10-18。只采用原始论文或官方代码 / 文档；下列来源不提供本项目的实验结果。

**[W1]** Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*，§4.1.1–4.1.2。支持组内相对优势及 outcome-level GRPO 的来源；原方法包含的 reference-KL 与本文适配不完全相同。  
`https://arxiv.org/html/2402.03300v3`

**[W2]** Schulman et al. *Proximal Policy Optimization Algorithms*。支持裁剪策略代理目标的算法来源，不保证任意实现中的策略变化被严格限制。  
`https://arxiv.org/abs/1707.06347`

**[W3]** PyTorch 官方 `torch.distributions.Beta` 实现。支持 Beta 参数、均值及 log_prob 的含义；正式运行还须锁定本地 PyTorch 版本。  
`https://raw.githubusercontent.com/pytorch/pytorch/main/torch/distributions/beta.py`

**[W4]** Vaswani et al. *Attention Is All You Need*，§3.2。支持 cross-attention 对 K/V 的加权读取；去掉外部门控不等于将 attention 权重全部设为一或强制均匀。  
`https://arxiv.org/html/1706.03762v7`

**[W5]** LIBERO 官方仓库，任务体系及固定初始状态恢复示例。  
`https://github.com/Lifelong-Robot-Learning/LIBERO`

**[W6]** LIBERO-PLUS 官方仓库，Benchmark 及 Evaluation 部分。支持 10,030 个任务配置、七类扰动、每任务配置 1 次的官方评估约定。  
`https://github.com/sylvestf/LIBERO-plus`

**[W7]** RoboTwin 2.0 官方文档和项目仓库。支持任务集合及配置需要按具体版本固定；不作为用户本地动作维度或测试次数已经确认的依据。  
`https://robotwin-platform.github.io/doc/`  
`https://github.com/RoboTwin-Platform/RoboTwin`

### 10.3 启动前必须填实的项目

本文不虚构附件未提供的运行值。每个 Benchmark 开始正式消融前，在配置中填实：**W0 / Teacher / Student 的实际路径与哈希、离线和在线划分、各阶段预算及 LR 历史、RoboTwin 动作 / 视角 / 频率、RL 累计环境步数上限、最终测试初始条件及随机种子、checkpoint 选择规则。** 这些是开跑前的配置工作，不是增加新的消融条目。
