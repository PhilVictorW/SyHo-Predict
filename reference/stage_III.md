# GRPO：基于现有 Stage-II Student 的奖励引导跨尺度交互后训练

**版本：2026-10-12 / 实施参考 v1.0**  
**适用对象：`LaWAM_aba2_change/goal_local` 的 LIBERO 实际实现**  
**主要依据：`demision.md`（源文档核对日期 2026-10-11），而非原始 `idea_10_9.md` 中 Stage II 全层隔离的设想。**

> **实施状态。** Stage I、Stage II 的结构和历史配置来自用户提供的 `demision.md`；用户补充已完成两阶段训练及部分推理验证，但本次没有提供这些新验证的完整数值。本文 Stage III 是基于这些事实制定的待实现、待验证方案，不是已经完成的实验记录。本次未读取用户本地源码或 checkpoint，也未运行 LIBERO。附录中的独立 PyTorch 数值核心经过 CPU 合成数据检查，不等于已在用户工程中联调通过。

## 阅读路径及来源约定

全文按“继承什么 → 新增什么 → 一次决策怎样运行 → 如何收集回报 → 如何更新 → 如何接入和验证”组织。首次实现优先执行第 7 节的分步改造，不应先把整个 Stage III 一次性接入。

- **[D, §…]** 表示附件 `demision.md` 的对应章节，是前两个阶段实际实现的依据。
- **[R1]–[R7]** 表示末尾的原始论文、官方文档或官方源码，仅支持相应通用算法/软件事实。
- 标注为“本版设计”“建议值”的内容是本文提出的 Stage III 实施选择，不代表原工程已经存在，也不代表文献已经验证了其在本模型上的有效性。

核心流程：

```text
Stage-II Student checkpoint（已有深层 Goal→Local 交互）
    ↓ 全部冻结，保留原动作采样流程
当前双视角 + 指令 → Student VLM 第 1–8 层
    ↓ Pool(Local, Goal, Current)
新 GateActor → 采样一个倍率 eta
    ↓
Student VLM 第 9–16 层：调节原来已训练的 Goal→Local 残差
    ↓
原 Action DiT：10 次 Euler 更新 → 一段 8×7 动作 → 仿真执行
    ↓ 重复到本条轨迹终止
同任务、同实际初始状态的一组轨迹 → 成功/步数回报
    ↓
组内相对优势 + 裁剪密度比 → 只更新 GateActor
```

## 1. 与当前 Stage I、Stage II 的准确衔接

### 1.1 前两个阶段留下了什么

Stage I 从同一示范 episode 读取当前双视角和指令，读取 `t+8` 的近端双视角以及 `t+[24,40,56,72,88]` 的五张远端主视角。冻结 Qwen vision 分别产生 Local `[B,128,2048]` 和 Goal `[B,320,2048]` memory，经两个可训练 Resampler 各压缩成 `[B,8,2048]`。Teacher 的 16 层原生注意力保持角色隔离，最终通过 Local→Action 和 Goal→Plan→Action 接受动作 flow loss。冻结视觉编码器的输出并不妨碍 Resampler 接收梯度。[D, §2–4]

Stage II 保留独立、冻结的 Stage-I Teacher 作为表征目标。Student 继承公共网络和动作头，新增两组 `[8,2048]` 的可学习 Query，输入观测不含真实未来。Student 在第 9–16 个 VLM block 之后执行显式的 Goal→Local RelevanceGate；原生 attention mask 仍保持角色隔离。第 12 层的 Local 蒸馏取 gate 后表示，第 16 层取 gate 后再经最终 RMSNorm 的表示。因此，深层交互已经接受了动作损失和 Local 蒸馏的训练。[D, §5.1–5.6]

**本版 Stage III 优化的是“已经存在的交互如何使用”，不是“第一次建立 Goal→Local 通路”。** 两类 Teacher 后验分别学习，并不等于当前 Student 的两类先验在整个 VLM 中独立。论文和实现都应保留这一边界。

### 1.2 必须继承的实际配置

| 项目 | 继承内容 | Stage III 是否改变 |
|---|---|---|
| 当前观测 | 主视角、腕部图像，各 `[B,3,256,256]`，加语言指令 | 不改变 |
| 本体状态 | 当前 `use_state=false`，State MLP 不运行 | 不额外加入 |
| 旧 Query | 8 个 Action Query + 8 个 Flow Query，包含在当前上下文长度 `L` 内 | 保留，不删除其 VLM 输入 |
| 新 Query | Goal、Local 各 8 个槽，隐藏维度 2048 | 继承 Stage-II 参数并冻结 |
| VLM 序列 | `[B,L+16,2048]`，16 层 | 不改变 |
| 深层交互 | 第 9–16 层 block 后；单向 Goal→Local | 保留位置和已训练参数 |
| 当前动作 memory | 删除旧 16 个 Query 的输出位置后，与 DINO 拼接为 `[B,L+240,768]` | 不改变 |
| Goal / Local 动作条件 | 各 `[B,8,1024]` | 投影结构不改变 |
| DiT | 4 个 Plan + 50 个 Action，`[B,54,1024]`，16 层 | 不改变 |
| 有效动作 | 20 Hz，8 步，7 维；内部仍为 50×32 | 不改变 |
| 动作生成 | 一次预测中编码条件一次，复用条件做 10 次 Euler 更新 | 不改变 |
| `B_f=2B` | 仅是前两阶段离线 flow loss 的重复噪声/时间采样 | Stage III 不使用 |
| `role_valid` | Student 两类角色条件始终有效 | 不根据未来有效性关闭 |

依据：[D, §1、§3.1、§3.5、§4、§5、§8]。`L=219` 是附件中的示例，而不是模型常数。

附件记录的 LIBERO 第 4000 步 Student 可作为候选起点：

```text
runs/goal_local/stage_II_warm250_20260921/step_004000/model.pt
```

该路径来自历史记录，不意味着它一定是当前最好的在线策略。正式启动应明确指定选中的 Student checkpoint，记录文件 SHA-256、配置和已有在线评估结果。不能因为文件名含 `best` 就当作在线成功率最优：附件的历史 `best` 依据离线 validation flow loss 选择。[D, §6.2–6.3]

### 1.3 原交互公式及其含义

在某个深层 block 输出后，令 Goal 和尚未执行该层交互的 Local 分别为：

\[
G^\ell,\bar L^\ell\in\mathbb R^{B\times8\times2048}.
\]

已有 RelevanceGate 计算：

\[
r^\ell=1-P^\ell(\mathrm{NULL})\in\mathbb R^{B\times8\times1},
\qquad
M^\ell=\operatorname{MHA}_\ell\!\left(\operatorname{LN}(\bar L^\ell),
\operatorname{LN}(G^\ell),\operatorname{LN}(G^\ell)\right)
\in\mathbb R^{B\times8\times2048}.
\]

其原始更新为：

\[
\boxed{\widetilde L^\ell=\bar L^\ell+s_\ell\,r^\ell\odot M^\ell.}
\tag{1}
\]

`r` 来自 Local 对 1 个 NULL key 和 8 个 Goal key 的相关性评分；`M` 来自另一个独立 message attention，二者不是同一次注意力。第 9–16 层共享相关性计算的 LN、投影和 NULL 参数，但拥有各自的 message attention 和残差 `scale`。[D, §5.3]

**`s_ell` 的初始值是 0.01，不代表训练后仍为 0.01；它也不受 `[0,1]` 约束。** Stage III 必须加载其实际数值，不能重新初始化、截断或把负值取绝对值。

## 2. Stage III 的目标和优化边界

### 2.1 要学的是交互使用策略，不是新的世界状态标签

以抽屉任务为例，本阶段让智能体在多次实际执行中比较：当前状态下，较弱或较强地使用 Goal→Local 消息，哪一种更有利于成功完成任务，成功时能否减少执行步数。学习结果是一个随当前观测变化的交互倍率策略。

倍率不是“抽屉应开多少厘米”，增大倍率也不必然让抽屉开得更大。Goal 不是 Local 的真实监督标签。本版没有物体尺寸估计、阶段分类、反事实动力学、额外价值网络或任务进度模型。

可将总体目标写为：

\[
\max_\phi\;J(\phi)
=\mathbb E_{\mathcal T\sim(\pi_\phi,\bar\theta,\mathcal E)}[R(\mathcal T)],
\tag{2}
\]

其中 `phi` 是新门控策略，`bar theta` 是冻结的 Stage-II 整体模型，`E` 是仿真环境。式 (2) 表示优化目标，不保证找到全局最优。第 6 节采用的是组内归一化、裁剪的 GRPO-style 代理目标，不能宣称它就是式 (2) 的无偏精确梯度。

### 2.2 强化学习各要素在本方案中分别是什么

| 概念 | 本方案中的对应对象 |
|---|---|
| 决策输入 | 第 8 层 Local、Goal、当前上下文的汇聚特征 `u_k` |
| RL 动作 | 连续隐变量 `z_k`，映射成交互倍率 `eta_k=2z_k` |
| 门控策略 | `pi_phi(z_k | u_k)`，由新 MLP 输出的 Beta 分布 |
| 实际控制动作 | 冻结 Student + DiT 在该倍率下生成的 8×7 动作段 |
| 一条轨迹 | 从一个实际初始状态开始，反复生成并执行动作段，直到成功、失败或超时 |
| 回报 | 整条轨迹是否成功，以及成功所需的实际环境步数 |
| 优势 | 该轨迹回报相对同组轨迹回报的高低 |
| 优化器更新对象 | 仅 GateActor 的参数 `phi` |

`u_k` 是观测表征，不是仿真器的完整状态；首版不假定它严格满足马尔可夫性。在部分可观测场景下，轨迹级策略梯度仍可针对该观测条件策略优化，不能据此宣称模型拥有完整状态信息。

### 2.3 固定模块和新增模块

本版执行：

\[
\bar\theta=\operatorname{Freeze}(\theta_{\mathrm{StageII}}),
\qquad\text{唯一可训练参数为 }\phi.
\]

| 模块 | Stage III 状态 |
|---|---|
| Qwen vision、merger、DeepStack merger | 继承并冻结，维持 eval |
| 词嵌入、16 层语言骨干、最终 RMSNorm | 冻结，正常前向 |
| 旧 Action / Flow Query，新 Goal / Local Query | 参数冻结，隐藏表示正常随输入更新 |
| 原 RelevanceGate 的 LN、投影、NULL key | 保留并冻结 |
| 原 8 组 message MHA、`s_ell` | 保留训练值并冻结 |
| DINO、VLM/base 投影、Goal/Local 投影 | 冻结 |
| Plan、两路动作注入 attention/scale、动作编解码、DiT | 冻结 |
| Teacher、Future Resampler | 不加载到 Stage-III 运行模型中 |
| 新 GateActor MLP | 由 GRPO-style 损失更新 |

冻结的是参数，不是 `r`、`M` 或 Local 隐藏表示的数值。对同一观测改变倍率，前一层 Local 会变化，后续层计算出的 `r` 和 `M` 也可以变化。每次观测更新后，整个条件链都重新计算。

## 3. 新门控的定义及一次完整前向

### 3.1 在已有残差外增加一个倍率

本版只将式 (1) 改成：

\[
\boxed{
\widetilde L^\ell
=\bar L^\ell+\eta_k\,s_\ell\,r^\ell\odot M^\ell,
\quad\ell\in\{9,\ldots,16\}.
}
\tag{3}
\]

`k` 是第几次动作段决策；一个动作段采样一个标量，对该次 VLM 前向的 8 个深层和各 Local 槽广播。它不是 8 个动作时间步各一个倍率，也不是每次 ODE 更新各一个倍率。

| 倍率 | 准确含义 |
|---|---|
| `eta=1` | 在相同观测、动作噪声和数值设置下，恢复当前 Stage-II 的全部已有交互 |
| `eta=0` | 关闭 VLM 内已有 Goal→Local 残差；Goal→Plan→Action 仍存在 |
| `0<eta<1` | 缩小已有残差的幅度 |
| `1<eta<2` | 放大已有残差的幅度，保留原 `s_ell` 的符号 |

关闭交互不能恢复“重新训练过的全层隔离 Stage II”。它只是当前已训练 Student 的路径消融；两种模型不能视作等价实验。

本版选择 `eta in (0,2)`，是为了让 `1` 位于区间内部，允许围绕已有策略双向探索。这个范围是待验证的工程选择，不是理论最优范围。若固定倍率诊断显示放大会造成明显退化，应先减小探索方差或另立较窄支持区间的配置，再重新采样训练；不得执行时临时裁剪倍率而继续假装其服从原分布。

### 3.2 第 8 层特征如何形成门控输入

当前双视角和指令沿原 processor、embedding、DeepStack 路径形成 `C`。保留旧 16 个 Query，再追加新 Goal / Local Query：

\[
X^0=[C;Q_G;Q_L]\in\mathbb R^{B_e\times(L+16)\times2048}.
\]

`B_e` 表示当前一次前向的活动环境数，与旧训练 micro-batch `B`、flow repeat 的 `B_f`、GRPO 组大小均不同。

运行原第 1–8 层，取 **第 8 个 block 输出，尚未经过最终 `lm.norm`**：

\[
L^8,G^8\in\mathbb R^{B_e\times8\times2048},
\qquad H_C^8\in\mathbb R^{B_e\times L\times2048}.
\]

Local、Goal 分别在 8 个槽上平均；Current 在全部有效 C token 上掩码平均。Current 的有效位置包括旧 16 个 Query，排除 padding，不要求删除特殊 token；各次采样与更新使用完全一致的规则。

\[
p_L=\frac18\sum_{j=1}^8L_j^8,
\quad p_G=\frac18\sum_{j=1}^8G_j^8,
\quad p_C=\frac{\sum_j a_{C,j}H_{C,j}^8}{\sum_j a_{C,j}},
\qquad p_\cdot\in\mathbb R^{B_e\times2048}.
\]

**本版新增的输入处理：** 仅在送入 GateActor 的池化向量上分别使用无可学习仿射参数的 `F.layer_norm`，再拼接，减少三组向量尺度差异；它不修改任何 VLM 隐藏表示，也不是替换 Qwen 的归一化层。

\[
\boxed{
u_k=\operatorname{sg}\left([
\operatorname{LN}_0(p_L);
\operatorname{LN}_0(p_G);
\operatorname{LN}_0(p_C)]\right)
\in\mathbb R^{B_e\times6144}.
}
\tag{4}
\]

`sg` 表示停止梯度。缓存的就是 Actor 实际收到的这份 FP32 `u_k`，不是另一份归一化前的向量。不得把未来有效性、剩余示范动作数、完整仿真状态或稍后才出现的奖励加入 Actor 输入。

### 3.3 Beta 策略、初始化和密度

GateActor 的首版结构为：

\[
6144\xrightarrow{\mathrm{Linear}}256
\xrightarrow{\mathrm{SiLU}}256
\xrightarrow{\mathrm{Linear}}2.
\]

若最后两个输出为 `h_alpha,h_beta`，定义：

\[
\alpha_k=1+\operatorname{softplus}(h_{\alpha,k})+\varepsilon_\pi,
\quad
\beta_k=1+\operatorname{softplus}(h_{\beta,k})+\varepsilon_\pi,
\quad\varepsilon_\pi=10^{-6}.
\]

\[
\boxed{z_k\sim\operatorname{Beta}(\alpha_k,\beta_k),\qquad\eta_k=2z_k.}
\tag{5}
\]

`alpha,beta` 是分布参数，不是动作或奖励；`z` 是训练时采样的 RL 动作。Beta 用于有界连续策略有既有研究依据 [R3]，PyTorch 官方实现提供其采样、均值和 `log_prob` [R4]。本版限制两参数大于 1，是额外的保守参数化选择，不是 GRPO 的要求，也不允许任意双峰门控分布。

**初始化使用 `alpha=beta=20`。** 最后一层权重初始化为 0，两个 bias 取相应 softplus 逆变换，使所有初始输入都得到上述分布。此时：

\[
\mathbb E[\eta_k]=1,
\qquad\operatorname{Std}(\eta_k)=\frac1{\sqrt{41}}\approx0.1562.
\]

这比前期讨论中的 `Beta(5,5)` 或 `Beta(10,10)` 探索更保守，是本文建议的首版起点。它仍有随机性；**均值等于 1 不代表每次随机采样等于 1，也不保证随机策略与 Stage II 行为等价。** 精确基线检查必须直接 override `eta=1`，而非随机采样。

Beta 隐变量的对数密度为：

\[
\log p_\phi(z\mid u)
=(\alpha-1)\log z+(\beta-1)\log(1-z)
-\log\operatorname B(\alpha,\beta).
\tag{6}
\]

因 `eta=2z`，`log p_eta(eta|u)=log p_z(eta/2|u)-log 2`。新旧策略使用同一映射时，常数项在密度比中抵消。因此本版统一存储 `z` 并调用 `Beta.log_prob(z)`，不把 `eta` 传给 Beta。

所有分布参数、采样、对数密度和策略损失使用 FP32，置于 autocast 之外。不要把采样结果转换为 bf16 后再计算密度；倍率送入冻结残差时按原计算 dtype 使用即可，缓存仍保留原 FP32 `z`。`sample()` 返回的结果作为固定数据，不能通过仿真器反向传播。不要硬截断 `z` 到 0/1 后沿用原密度；检测到非法端点或非有限值应报错诊断，而不是静默修补。[R4]

### 3.4 第 9–16 层怎样继续运行

每层先运行原 VLM block，原生角色 attention mask 不变。当前层的 `bar L^ell` 已经包含此前各层交互产生的影响，再用当前层 **block 后** 的 Local / Goal 计算原 `r^ell` 和 `M^ell`，最后执行式 (3)。不要改成原始 Idea 中“用上一层输入生成复用 attention 消息”的另一种实现。

维度过程：

| 运算 | 形状 |
|---|---|
| 本层 block 输出 | `[B_e,L+16,2048]` |
| 切出 Goal / Local | 各 `[B_e,8,2048]` |
| 原 relevance 投影 | 各 `[B_e,8,256]` |
| NULL + Goal keys | `[B_e,9,256]` |
| relevance logits / softmax | `[B_e,8,9]` |
| `r=1-P(NULL)` | `[B_e,8,1]` |
| 原 message MHA | `[B_e,8,2048]` |
| `eta[:,None,None]` | `[B_e,1,1]` |
| `eta*s*r*message` | `[B_e,8,2048]` |
| 更新 Local 并拼回完整序列 | `[B_e,L+16,2048]` |

第 16 层 gate 之后，仍按原代码运行最终 `lm.norm`，再拆出 Current、Goal、Local。对同一固定观测，倍率不应经前向改变 Goal 或 Current：它们没有读取 Local 的路径；但倍率改变动作后，下一个环境观测通常不同，因此后续时刻的 Goal 和 Current 当然可以不同。

### 3.5 动作分支保持原实现

最终 Current `[B_e,L,2048]` 按旧 Query 的位置 mask 删除 16 个对应输出，得到 `[B_e,L-16,2048]`，不是默认裁掉最后 16 个 token。经 `enc_vlm` 投影后，与当前主视角 DINO 的 256 个 token 拼接：

\[
C_{base}\in\mathbb R^{B_e\times(L+240)\times768}.
\]

Goal / Local 分别投影为 `[B_e,8,1024]`。VLM 门控不直接乘到 DiT 的 `goal_scale`、`local_scale` 或 Goal→Plan 条件上。

动作采样使用原 `[B_e,50,32]` 高斯噪声和前 8 个有效时间位置，执行 10 次 Euler 更新。每次 velocity 前向内部仍是 `[B_e,54,1024]`，16 个 DiT block 的真实排布如下。[D, §4.5、§8]

| DiT 层 | 角色注入 | 原 block 注意力 |
|---|---|---|
| 1、5、9、13 | Goal→Plan，Local→Action | `[Plan;Action]` 读取当前 VLM memory |
| 2、6、10、14 | Local→Action | Plan / Action 联合 self-attention |
| 3、7、11、15 | Local→Action | `[Plan;Action]` 读取当前 DINO memory |
| 4、8、12、16 | Local→Action | Plan / Action 联合 self-attention |

所以不能写成“每个 DiT 层都有一次 Plan / Action 联合自注意力”。角色注入在对应原 block 之前，保留附件中的计算顺序。

\[
x_{j+1}=\left(x_j+\frac1{10}v_{\bar\theta}(x_j,j/10;C_{base},G_A,L_A)\right)
\odot m_{token},\quad j=0,\ldots,9.
\tag{7}
\]

式 (7) 为简写；实际仍按原 `bucket=floor(1000*j/10)`、原时间编码和动作编解码执行。最终取前 8 个时间位置、前 7 个通道；前 6 维按原统计反归一化，第 7 维按原 `>0.5` 二值化。仿真侧额外的图像翻转、夹爪映射等保持已验证的部署约定，不能根据本文自行改动。

**一个动作段只运行一次 VLM/DINO、采样一次倍率；该段 10 次 velocity 计算复用这些条件。** 新观测到来后再做下一次完整决策。

### 3.6 数值例子：`B_e=2,L=219`

| 位置 | Stage III 张量 |
|---|---|
| 当前双视角 | 各 `[2,3,256,256]` |
| 当前 C / 加新角色后的序列 | `[2,219,2048]` / `[2,235,2048]` |
| 第 8 层 Goal / Local | 各 `[2,8,2048]` |
| Pool 后拼接 `u` | `[2,6144]` |
| Actor 两个分布参数 | 合计 `[2,2]`；alpha、beta 各 `[2]` |
| `z` / `eta` | 各 `[2]`；广播时 `[2,1,1]` |
| 原 relevance / message | `[2,8,1]` / `[2,8,2048]` |
| 最终序列 | `[2,235,2048]` |
| 删除旧 Query 的 C | `[2,203,2048]` |
| DINO + Current base | `[2,459,768]` |
| Goal / Local DiT 条件 | 各 `[2,8,1024]` |
| DiT hidden / velocity | `[2,54,1024]` / `[2,50,32]` |
| 最终动作 | `[2,8,7]` |

不存在 Stage-II flow repeat 的 `[4,54,1024]`，除非本次真的同时运行 4 个环境。GRPO 一组 4 条轨迹也不要求同时使用 batch=4，可以串行收集。

## 4. 在线轨迹如何收集

### 4.1 一组比较必须从同一个实际初始条件出发

设每轮采样包含 `J` 组，每组 `N_grp` 条完整轨迹。首版建议 `J=4,N_grp=4`；调试时可先用一组、两条。每组共享任务、指令、实际仿真初始状态和环境配置，但在之后的动作段中独立采样门控和动作噪声。

LIBERO 官方封装提供 `set_init_state`、`get_sim_state`、`check_success`，官方评估代码使用固定初始状态控制实验随机性。[R5] 但这些通用事实不等于用户当前部署脚本已经按本文要求工作。接入前应核对本地版本以及已有推理代码。

建议的分组记录：

```text
group_id
  task_id / instruction
  bddl_file / environment_config_hash
  initial_state_id / initial_state_hash
  reset_seed / settling_protocol
  behavior_policy_version
  rollout_0 ... rollout_(N_grp-1)
```

“同一个 seed”不必然等于“同一个实际初始状态”。应按原已验证的 reset 流程重置环境、恢复同一初始状态、完成相同的稳定步骤，并检查恢复后的状态或首帧一致性。仅保存 MuJoCo 物理状态未必保存控制器或包装器内部计数；优先从一次干净 reset 开始复现既有流程，不假设 `get_sim_state()` 是所有软件状态的完整快照。

不要将官方另一份脚本中的稳定步数强行替换到当前工程。本文没有收到当前 rollout 配置，因此稳定步数、每类任务 `T_max`、相机字段、图像方向、夹爪到环境动作的映射都应从用户已验证的推理程序显式导入。

### 4.2 先建立 RL 专用的训练、验证、测试划分

前两阶段的 episode 级离线划分并不会自动形成在线 RL 的初始状态划分。Stage III 必须建立单独的 manifest：

| 集合 | 允许用途 |
|---|---|
| RL-train | 在线采样并更新 Actor；固定倍率的训练诊断 |
| RL-val | 选择探索范围、学习率、checkpoint 和部署方式，不参与梯度更新 |
| Final-test | 最终一次或预先约定的正式评估，不用来选倍率或调参 |

同一实际初始状态不能因换了随机 seed 就被当成不同集合。保存状态标识、哈希和生成来源，进行训练、验证、测试之间的去重；有条件时还应检查与已有示范初始状态的对应关系，并如实说明先前 SFT 数据的覆盖范围。

标准同任务后训练允许 RL 使用训练任务，然后在该任务的保留初始条件评估；这不等于“未见任务泛化”。若目标是未见任务泛化，还需要任务级互斥划分。若直接从官方评估初始状态中划出一部分用于训练，必须说明评测协议已经改变，不能仍把该集合整体当作未参与训练的官方测试集。

没有可追溯的在线划分时，应先完成 manifest 和 baseline 重评，而不是默认“改 seed 即可避免泄漏”。

### 4.3 一条轨迹中的执行顺序

对第 `i` 条轨迹，固定行为策略快照 `phi_old`，从本组初始条件出发。每次动作段决策 `k`：

```text
1. 读取当前主/腕图像、指令。
2. 冻结 Student 前 8 层，得到并缓存 Actor 的实际输入 u_k。
3. 行为 Actor 计算 alpha_old、beta_old，采样 z_k，令 eta_k=2*z_k。
4. 缓存 z_k、old_logp、alpha_old、beta_old、决策起始环境步数。
5. 冻结 Student 后 8 层，用 eta_k 调节原有交互残差。
6. 构造原三类动作条件，原 DiT 执行 10 次 Euler 更新。
7. 得到 8×7 动作，按已验证的后处理逐个环境步执行。
8. 每个环境步检查成功、失败、终止和最大步数；终止时立即停下。
9. 未终止则读取新的观测，进入下一次动作段决策。
```

执行动作段不是必须把 8 步全部执行完。例如本段第 3 个环境步已经成功，剩下 5 步不再执行；本段仍只有一次门控决策，但 `T_i` 只增加 3。不能只在每个 chunk 末尾检查成功，否则成功步数可能多计并改变奖励。

一般情况下：

\[
T_i\le8K_i,
\]

其中 `K_i` 是实际执行过动作的门控决策次数。不要把 `T_i` 直接写成 `8*K_i`。任务开始前的固定稳定动作按当前统一协议单独记录，本版的效率计数从策略接管后开始；所有比较模型使用相同口径。

`done`、`truncated`、成功条件不可混用。Gym/Gymnasium 或自定义 wrapper 的返回格式可能不同，应在适配层明确转换为：

```text
success / terminal_failure / timeout / infrastructure_error
```

优先核对任务成功谓词的实际语义，而非默认任何 `done=True` 都代表成功。时间预算耗尽但未成功，本版记任务失败且奖励为 0，不进行 value bootstrap。渲染器崩溃、加载失败等基础设施错误单独记录，不能伪造为任务成功；按预先规则废弃并补采整个不完整组。若错误由策略诱发，例如非有限动作，不应通过不断删除坏样本制造高成功率；应暂停该配置、诊断并报告。

### 4.4 随机性怎样管理

需要区分四类随机性：实际初始条件、Beta 门控采样、动作 flow 噪声、底层仿真/渲染的残余非确定性。

本版训练默认：每组共享实际初始条件；各条轨迹独立使用门控随机流和动作噪声随机流。门控随机流不应因为动作噪声采样次数变化而被无意扰动，反之亦然。每轮更换训练初始条件或相应种子，避免 Actor 只适应固定噪声。

固定倍率诊断和配对评估可以为不同方法使用相同的动作噪声序列，以减少比较方差，但应独立报告采用该设置，且不能将一条有利噪声序列当作全部评估。训练首版不依赖“同组共享所有动作噪声”这一额外设计。

PyTorch 的 `Beta.sample()` 不接受 `generator=` 参数。[R4] 工程上可通过独立采样进程或妥善管理的 RNG 上下文保存/恢复门控随机流；不要在代码中添加一个该方法并不支持的参数。需要复现时，保存每次采样种子或 RNG 状态，而不是仅保存一次全局 seed。

### 4.5 每次必须保存哪些数据

| 字段 | 单条轨迹中的形状/类型 | 用途 |
|---|---|---|
| `features` | `[K_i,6144]`，FP32 | 旧策略当时实际收到的 `u` |
| `z` | `[K_i]`，FP32 | 未裁剪的 Beta 隐变量 |
| `eta` | `[K_i]`，FP32 | 可由 `2*z` 重建，用于诊断 |
| `old_logp` | `[K_i]`，FP32 | 采样时 `log Beta_old(z)` |
| `old_alpha,old_beta` | 各 `[K_i]`，FP32 | 密度核对、旧新策略 KL 诊断 |
| `chunk_start_step,executed_steps` | 各 `[K_i]` | 精确环境步数和执行边界 |
| `success,total_steps,return` | 标量 | 轨迹回报和评价 |
| `termination_reason` | 枚举 | 区分失败、超时、基础设施错误 |
| `task/group/reset/policy_version` | 标识字段 | 保证正确组内比较和 on-policy 更新 |
| `action_noise_seed` | 每动作段标识 | 复现动作采样 |
| `mean_r,scale,delta_norm` | 按层统计，可选 | 分析原交互是否实际起作用 |

Actor 更新时不需要把全程图像、完整 VLM activation 或动作 DiT 计算图放在显存中。视频和完整环境状态用于抽样诊断，不能进入 Actor 的训练输入。缓存大小与门控决策数有关，不与 10 次 ODE 更新数相乘。

同一轮收集期间行为策略保持不变。不能在一组收集到第 2 条轨迹时先更新网络，再收集第 3、4 条；也不能更新 `phi_old` 后重算旧概率。已执行轨迹只允许参与该轮少量近策略更新，不建无限复用的离线 replay buffer。

## 5. 任务回报及组内相对优势

### 5.1 采用成功奖励加成功效率奖励

本版不新建 reward model，不读取未来图像作为奖励标签。第 `i` 条轨迹的回报：

\[
\boxed{
R_i=y_i\left[1+\lambda_{eff}\left(1-\frac{T_i}{T_{max,i}}\right)\right],
\quad y_i\in\{0,1\},\quad0\le\lambda_{eff}<1.
}
\tag{8}
\]

首版建议 `lambda_eff=0.1`，不是已有实验选出的最佳值。`T_max` 读取相应任务的既有评测配置，不在本文凭空指定所有任务统一的数值。

成功获得基础分 1，成功且更快可额外加分；失败回报为 0，即使提前结束也无效率奖励。在线 rollout 阶段没有示范动作作为正确答案，不计算 flow loss 或 Teacher 蒸馏。

以下仅为数学示例，设 `T_max=300`：

| 轨迹 | 成功 | 实际执行步数 | 回报 |
|---|---:|---:|---:|
| 1 | 是 | 120 | 1.06 |
| 2 | 是 | 180 | 1.04 |
| 3 | 否 | 300 | 0 |
| 4 | 是 | 150 | 1.05 |

一次轨迹记录的是成功指示 `y_i`，成功率是对多条评估轨迹求平均得到的统计量，二者不能混称。

### 5.2 两个不能省略的奖励边界

**第一，单条成功轨迹得分高于失败轨迹，不等于优化期望奖励就严格优先最大化成功率。** 一个稍低成功率但更快的策略仍可能获得更高平均回报。因此 checkpoint 选择和最终结果必须分别报告成功率与成功步数；不能只按训练 reward 选择模型，也不能用“效率权重小于 1”证明成功率不会下降。

**第二，组内标准化会削弱效率权重的绝对尺度含义。** 在一组全部成功、只有执行步数不同的情况下，奖励相对差异整体与 `lambda_eff` 成比例，除以标准差后这一比例可能相消。因此 `lambda_eff=0.1` 不意味着这类组的 Actor 梯度必然只有成功/失败组的十分之一。

本版用下一小节的标准差下限抑制很小的组内差异被过度放大，并单独记录全成功组、混合组、全失败组的更新占比。若后续希望更严格地限制效率目标，需明确新增独立目标权重或约束，而不是把它写成现有式 (8) 已经保证的性质。

### 5.3 组内优势计算

对同一个任务、同一个初始条件的第 `j` 组，计算：

\[
\bar R_j=\frac1{N_{grp}}\sum_iR_{j,i},
\qquad
s_j=\sqrt{\frac1{N_{grp}}\sum_i(R_{j,i}-\bar R_j)^2}.
\]

GRPO 的组内相对优势思想来自同输入下的候选回报比较，不需要另训 Critic。[R1] 本文把完整轨迹作为候选，并明确采用以下带方差保护的适配：

\[
\boxed{
\widehat{\mathcal A}_{j,i}=
\begin{cases}
\displaystyle\operatorname{sg}\!\left(\frac{R_{j,i}-\bar R_j}
{\max(s_j,\sigma_{floor})}\right),&s_j>\sigma_{skip},\\[6pt]
0,&s_j\le\sigma_{skip}.
\end{cases}
}
\tag{9}
\]

首版建议 `sigma_floor=0.01`、`sigma_skip=1e-6`。`std` 使用总体标准差，即 `correction=0`，避免不同实现对小组标准差的定义不一致。这个分母下限是本文新增的稳定化选择，不是原始 GRPO 必须采用的公式。

一条轨迹的所有门控决策使用同一优势：

\[
\widehat{\mathcal A}_{j,i,k}=\widehat{\mathcal A}_{j,i}.
\]

在上面的示例中，组均值为 `0.7875`，标准差约 `0.45472`，优势约为：

\[
[0.5993,\;0.5553,\;-1.7318,\;0.5773].
\]

直观含义：这组里三条成功轨迹相对更好，失败轨迹相对更差；更快的成功轨迹得到略高优势。这不表示失败轨迹里的每一次门控都错误，也不表示成功轨迹里的每一次门控都最优。首版没有逐决策或逐层的独立因果归因。

全失败组以及全成功且步数相同的组优势为 0；保留其完成率和采样成本记录，但没有奖励驱动梯度。整轮全为零优势时跳过 optimizer step，不要制造随机正负优势。不能不停补采直到出现“有差异的组”却不报告筛选规则，否则会改变训练采样分布。

## 6. GRPO-style 更新究竟怎样发生

### 6.1 为什么只计算门控密度就能学习

一次决策先采样 `z_k`，冻结的模型和动作噪声再决定控制动作，环境产生后续观测及最终回报。对固定的观测历史而言，只有门控分布含可训练参数 `phi`；冻结动作生成器和环境的条件转移不含 `phi`。

由此，策略的得分函数项来自：

\[
\sum_k\nabla_\phi\log p_\phi(z_k\mid u_k).
\]

这说明可以通过已采样门控的 log density 学习，不必计算整段 flow 动作的概率密度，也不必对仿真器或 10 次 Euler 更新求导。实际训练仍采用下述近策略裁剪代理目标，而不是计算整条轨迹密度的乘积。[R1、R2]

虽然后续观测随早前门控而变化，更新时仍把本轮收集到的 `u_k`、`z_k` 和奖励作为固定数据，重新计算当前 Actor 在**同一个缓存状态、同一个已执行采样值**上的密度。这与“用新 Actor 重新走一条轨迹来代替旧数据”不同。

### 6.2 计算新旧策略的密度比

\[
\boxed{
\rho_{j,i,k}(\phi)
=\exp\left[
\log p_\phi(z_{j,i,k}\mid u_{j,i,k})
-\log p_{\phi_{old}}(z_{j,i,k}\mid u_{j,i,k})
\right].
}
\tag{10}
\]

分母必须使用采样时保存的 `old_logp`，且停止梯度。`rho>1` 表示当前网络增加了当时该采样值附近的密度，`rho<1` 表示降低了密度。连续密度不是“精确浮点数发生的离散概率”，密度大于 1 本身不构成错误。

**第一次更新前 `phi=phi_old`，所有 `rho` 应接近 1。** 不满足时先查缓存特征、Actor mode、dtype、采样值和概率版本，不能直接开始 RL。

### 6.3 裁剪目标及轨迹长度加权

本版每组相同 `N_grp`，完整采样 `J` 组。使用：

\[
\boxed{
\mathcal L_{III}(\phi)
=-\frac1J\sum_{j=1}^{J}\frac1{N_{grp}}\sum_{i=1}^{N_{grp}}
\frac1{K_{j,i}}\sum_{k=1}^{K_{j,i}}
\min\!\left[
\rho_{j,i,k}\widehat{\mathcal A}_{j,i},
\operatorname{clip}(\rho_{j,i,k},1-\epsilon_c,1+\epsilon_c)
\widehat{\mathcal A}_{j,i}
\right].
}
\tag{11}
\]

建议 `epsilon_c=0.2`。优势为正时，目标鼓励增加所采样门控的密度；优势为负时，鼓励降低其密度。裁剪去掉了过度改变密度比带来的部分收益，但**不是对策略变化的硬性保证**。[R2、R6]

先对一条轨迹中的决策平均，再对轨迹和组平均，避免较长轨迹仅因为记录更多门控就获得更大权重。展平所有决策后，令总轨迹数 `M=J*N_grp`，轨迹 `i` 的每个决策权重为 `1/(M*K_i)`。不能展平后直接 `.mean()`，那会无意改成更偏向长轨迹的目标。

保留零优势完整组时，它们在式 (11) 中贡献 0，但仍计入 `J`。不要为有优势的组临时重算更小分母却不说明目标变了。基础设施错误造成的不完整组，应在形成这一批有效完整组之前按预定规则补齐。

这里采用的轨迹长度平均、组内归一化和裁剪均是实际代理目标的一部分，不能宣称其等同于不加权 episodic return 的精确无偏梯度。它也不提供每个门控的独立价值估计，本文不另设 Critic 或 GAE。

### 6.4 稳定性监测

在缓存状态上，用已保存的旧 Beta 参数计算：

\[
D_{old,new}=\operatorname{Avg}_{trajectory}
\operatorname{Avg}_{decision}
D_{KL}\!\left(\operatorname{Beta}(\alpha_{old},\beta_{old})
\;\|\;\operatorname{Beta}(\alpha_\phi,\beta_\phi)\right).
\tag{12}
\]

PyTorch 提供 Beta–Beta 的 KL 实现 [R7]。本版建议 `target_kl=0.01`：某次更新后超过阈值，则停止继续使用这批数据做更多更新。KL 在此是监测和早停依据，不是损失中的固定参考策略 KL 惩罚。[R6]

这只测旧策略访问状态上的条件分布变化，不保证整个环境轨迹或动作分布变化同样小。默认首版每轮最多 2 次整批 Actor 更新，不设置额外 Critic、熵奖励或 reference-KL loss。监测探索方差及熵；需要加入正则时单独立配置，不静默改变首版目标。

遇到非有限 loss、概率或梯度，停止本次更新，恢复最近完整的稳定 checkpoint 后诊断；不要将 log-ratio 任意硬裁剪以掩盖网络错误。额外的硬回滚阈值可作为后续工程保护，但不是式 (11) 自带的功能。

### 6.5 一次完整的采样—更新循环

```text
加载并冻结选定的 Stage-II Student；构造 GateActor 与独立 optimizer。

每轮 iteration：
    固定 behavior_actor = 当前 GateActor 的参数快照。
    从 RL-train manifest 选择 J 个组条件。

    对每个组条件：
        用同一个 behavior_actor 收集 N_grp 条完整轨迹。
        相同任务和实际初始状态；门控/动作噪声按规则独立。
        保存每次决策 features、z、old_logp、旧 alpha/beta。
        逐环境步统计是否成功与 T_i，计算 R_i。

    在每个组内部计算 advantage，不跨组归一化。
    把轨迹的 advantage 复制给该轨迹的全部决策。
    形成展平缓存，保留 trajectory_id、lengths、group_id。

    全部 advantage=0：只记日志，跳过优化，进入下一轮采样。
    否则最多执行 2 次更新：
        在缓存 features、缓存 z 上重算当前 Actor log_prob。
        按式 (11) 算 loss，按轨迹加权，而非按所有决策直接平均。
        optimizer.zero_grad → backward → clip_grad_norm_(1.0) → step。
        重算旧新 KL；超过 target_kl 则停止本轮后续更新。

    释放本轮训练缓存，下一轮用更新后的策略重新采样。
    按计划运行 RL-val，不在验证时更新任何参数。
    保存 Actor、optimizer、基础 checkpoint 标识、配置、种子及计数。
```

只有新 Actor 的 MLP 需要保留 autograd 图。收集时原 Student/DINO/DiT 均 `eval` 和 `no_grad`，Actor 采样也可在 `no_grad` 下完成。更新时 Actor 必须在启用梯度的独立代码块重新计算，不能因为外层 `@torch.no_grad()` 导致它也不学习。

若用 `torch.inference_mode()` 收集，缓存要转换成更新阶段可供 Linear 反向保存的普通 Tensor；首版用 `no_grad()` 加 `detach().clone()` 更易避免 inference tensor 的 autograd 限制。Actor 不加 dropout，收集和更新之间没有随机网络层差异。

不要把基础模型参数放进 optimizer。尤其是原 `message attention` 位于门控采样之后，当前仅依靠 `log p_phi(z|u)` 的目标不会自动训练它；即使取消它的 `requires_grad=False` 也不构成正确的训练方案。

**loss 接近 0 不等于没有梯度。** 在初始 `rho=1` 时，同组优势均值接近 0，数值 loss 也可能接近 0，但各采样值对应的 log-prob 梯度不同，Actor 仍可得到非零更新；应检查梯度和密度变化，而不是只看 loss 的绝对值。

## 7. 工程接入及分步实施

### 7.1 建议的代码改动范围

以下文件名和函数名是**建议新增或适配的工程组织**，不是已确认存在的 API。实际源代码未随附件提供，因此不能把此处示意直接视作可运行的完整工程补丁。

| 位置 | 需要做的修改 | 必须保持的内容 |
|---|---|---|
| `goal_local/roles.py` 的原 RelevanceGate | 增加可选外部 `rl_multiplier`，乘在原交互残差上 | 原 LN、NULL、relevance、message、scale、层顺序 |
| `goal_local/model.py` 的 Student VLM 遍历 | 第 8 层输出后获取 `u` 并调用 Actor；把倍率传给后 8 层 | 原 embedding、DeepStack、mask、position IDs、最终 norm |
| `goal_local/action.py` | 原采样逻辑不变；为可复现性按需暴露动作噪声输入/随机流 | 10 次 Euler、动作归一化、mask、DiT 原层序 |
| 新 `goal_local/rl/gate_actor.py` | Actor、特征池化、采样、确定性均值部署 | 不加载 Teacher，不访问未来 |
| 新 `goal_local/rl/rollout.py` | 环境分组、逐环境步执行、缓存完整轨迹 | 与已有推理完全相同的图像和动作处理 |
| 新 `goal_local/rl/grpo.py` | 奖励、优势、裁剪目标、KL 监测 | 旧概率不可变，保持轨迹加权 |
| 新 `goal_local/rl/train.py` | 收集与更新循环、验证、checkpoint、恢复 | 不改写原 Stage-I/II 入口的默认行为 |
| 新 `goal_local/rl/evaluate.py` | 固定倍率、Actor 均值/随机策略的统一评估 | 相同测试条件和预算，不在测试时训练 |

Stage I、II 的真实入口仍是 `goal_local/train.py`，不是旧 `starVLA` 入口。[D, §1、§6] 不应为接 Stage III 顺便重写前两阶段、替换 processor 或重置旧 Query。

### 7.2 第一步：先验证已训练交互是否具有可调空间

加载选定的 Stage-II Student，冻结全部参数。对训练诊断/验证划分执行固定倍率 `0、0.5、1、1.5、2`；数值先检查，再做在线任务测试。全程没有 RL。

检查三个层次：

| 层次 | 记录内容 | 解释 |
|---|---|---|
| 原残差 | 每层 `s_ell`、`r` 分布、`||s*r*M||/(||bar L||+eps)` | 当前交互是否近似失效，是否存在过大残差 |
| 固定观测前向 | 配对相同噪声，比较不同倍率的 Local 和最终动作差异 | 倍率能否改变控制输出 |
| 在线执行 | 任务成功率、成功步数、失败模式 | 改变行为是否提供任务收益空间 |

倍率改变但动作基本不变时，先检查残差 scale、相关性、消息范数和下游条件注入，不应指望仅训练 Actor 能修复一个冻结且近似无效的消息通路。第 12/16 层的 gate 后蒸馏可能影响交互的强弱，这是需要检查的可能原因，不是已证明发生的事实。

### 7.3 第二步：实现最小倍率改动并保证兼容

原残差与新残差的概念差异：

```python
# 保留原代码的具体 dtype、运算次序和 delta 计算。
delta = scale * relevance * message
if rl_multiplier is not None:
    # 实際倍率 shape 为 [B]；不要把 batch 维广播错到槽维。
    eta = rl_multiplier.reshape(-1, 1, 1).to(delta)
    delta = eta * delta
local_new = local + delta
```

`None` 保持旧逻辑；`1.0` 必须恢复旧前向；`0.0` 仅移除这条残差。集成时保留原始求值次序并用数值测试确认，不要为了形式统一重写原 attention。固定倍率作为诊断模式，不进入 GRPO 的连续概率比训练。

### 7.4 第三步：增加 Actor，但先不更新

在现有 VLM loop 的第 8 层输出处增加回调最直接：该位置调用 `pool_gate_features` 和行为 Actor，取得一个倍率，再继续原第 9 层。无需从头再跑一遍前 8 层，也不应对后 8 层沿用一个无视倍率变化的旧 Local KV cache。

可采用如下建议的功能契约；名字只是示意：

```text
encode_until_layer8(current_images, instruction)
    -> full_hidden8, current_mask, role_positions,
       old_query_positions, additive_mask, position_ids, original_aux_data

actor.collect(pool_gate_features(...))
    -> z, eta, old_logp, old_alpha, old_beta

encode_remaining_layers(prefix_state, eta)
    -> normalized_current_hidden, normalized_goal, normalized_local

sample_original_action(conditions, action_noise)
    -> action_chunk[8,7]
```

`prefix_state` 必须保留原实现恢复后半段所需的 mask、位置和有效性信息，不只是保存 Local/Goal。当前图像 DeepStack 已按原逻辑注入第 1–3 层，不能在 resume 后重复注入。所有环境在第 8 层决策时使用的 `u` 与缓存 `u` 应逐元素一致。

先评估确定性 `eta_mean=1` 是否恢复旧策略，再测随机初始化 Actor 的成功率和残差分布。随机初始化即使均值正确也可能导致行为退化，发现后应先调小探索幅度，不能把退化隐瞒为“仍等价 Stage II”。

### 7.5 第四步：先跑通完整轨迹缓存，再接优化器

先用 `J=1,N_grp=2` 做流程验证。完成轨迹后核对：缓存长度等于门控决策数；逐段执行步数之和等于总步数；成功中断没有多执行动作；`z` 未被覆盖成 `eta`；所有旧参数/概率来自同一个行为策略版本。

确认新旧 Actor 尚未更新时密度比为 1，再允许执行第一次梯度更新。设置 Actor 参数集白名单，检查其他模块没有梯度，基础 checkpoint 中参数和浮动 buffer 没有变化。

此阶段才将第 5、6 节的计算连接起来。不要调用旧的离线 `compute_loss()` 代替在线回报，也不要用验证 flow loss 模拟任务成功回报。

### 7.6 第五步：正式后训练与验证

按训练 manifest 采集多组完整轨迹、执行少量 Actor 更新。逐轮记录采样成本和无信息组比例，定期在固定 RL-val 条件上比较当前 Stage III 与 `eta=1` 的 Stage-II 基线。

只在通过前向兼容和小规模采样检查后扩大任务覆盖。若某类任务长期全失败，没有组内差异，应检查基线能力、合法探索范围和奖励是否正确，而不是让零信号训练空转。扩大组大小可以增加比较样本，但并不保证产生成功或解决消息无效的问题。

若之后决定解冻原 message attention、重新训练 query 或扩展到第 1–8 层，需要另立版本和新的优化路径。这些不是当前“仅训练倍率 Actor”的隐含操作。

## 8. 配置、记录和恢复

### 8.1 建议起始配置

下面是配置模板，不是现有工程已支持的字段。`null` 项必须从用户的已验证 rollout 配置和本次 manifest 明确填入；训练器应缺项即报错，不能自行猜测。

```yaml
method: reward_guided_cross_horizon_gate_grpo
status: proposed_stage_III_v1
base:
  student_checkpoint: runs/goal_local/stage_II_warm250_20260921/step_004000/model.pt
  checkpoint_sha256: null        # 启动时计算并写入；路径仅为附件中的候选
  freeze_all: true
  use_state: false
  preserve_old_queries: [8, 8]
  query_count_per_role: 8
  vlm_layers: 16
  interaction_layers_one_based: [9, 10, 11, 12, 13, 14, 15, 16]
  action_chunk: 8
  action_hz: 20
  flow_sampling_steps: 10
actor:
  observation_layer: 8
  pool_current_include_old_queries: true
  pool_current_mask_padding: true
  pooled_branch_norm: layer_norm_without_affine
  input_dim: 6144
  hidden_dim: 256
  output_dim: 2
  activation: silu
  granularity: one_scalar_per_chunk
  distribution: beta
  concentration_transform: one_plus_softplus_plus_eps
  parameter_eps: 1.0e-6
  initial_alpha: 20.0
  initial_beta: 20.0
  multiplier_min: 0.0
  multiplier_max: 2.0
  dtype: float32
rollout:
  train_manifest: null
  val_manifest: null
  final_test_manifest: null
  groups_per_iteration: 4
  trajectories_per_group: 4
  envs_in_parallel: 1           # 并行度不决定组大小，按资源和原环境支持调整
  matched_initial_state: true
  independent_gate_rng: true
  independent_action_rng: true
  reuse_actions_or_future_targets: false
  task_max_steps_config: null
  settling_protocol_config: null
  observation_and_action_adapter_config: null
reward:
  success_bonus: 1.0
  efficiency_weight: 0.1
  failure_reward: 0.0
  count_unit: executed_environment_steps_after_policy_takeover
  gamma: 1.0                    # 本版终局回报，不另做时间折扣或 GAE
advantage:
  group_population_std: true
  std_skip: 1.0e-6
  std_floor: 0.01
  assign_trajectory_advantage_to_all_decisions: true
  keep_zero_advantage_groups_in_reduction: true
grpo:
  learning_rate: 1.0e-4         # 建议起点；可在验证划分比较 3e-5
  optimizer: Adam
  betas: [0.9, 0.999]
  weight_decay: 0.0
  max_actor_steps_per_collection: 2
  full_cached_batch_update: true
  clip_epsilon: 0.2
  max_gradient_norm: 1.0
  target_kl_old_new: 0.01
  entropy_coefficient: 0.0
  reference_kl_coefficient: 0.0
  critic: false
  flow_loss_coefficient: 0.0
  distillation_loss_coefficient: 0.0
evaluation:
  deployment_multiplier: beta_mean
  also_evaluate_stochastic_actor: true
  compare_against_stage_II_eta_one: true
  selection_primary: task_macro_success_rate
  selection_secondary: success_steps_with_selection_bias_checks
  eval_every_collections: 10
budget:
  max_environment_steps: null
  max_collections: null
logging:
  save_every_collections: 10
  record_policy_and_environment_seeds: true
  record_base_parameter_checksum: true
```

这些学习率、组大小、探索浓度和检查频率均为启动建议，没有可据以保证效果的实验依据。正式实验需要预先确定交互预算；本文不虚构一个“训练多少轮必然收敛”的数值。

### 8.2 每轮应记录的指标

| 类别 | 指标 |
|---|---|
| 实际成本 | 累计环境步数、完整轨迹数、Actor 更新数、墙钟耗时 |
| 任务结果 | 逐任务及总体成功率、原始回报、成功执行步数、超时比例 |
| 组内信号 | 奖励标准差、全失败组/全成功组/混合组比例、零优势组比例 |
| 门控行为 | `eta` 均值、标准差、分位数；alpha/beta；各层原 `r` 与残差比例 |
| 策略更新 | loss、梯度 norm、ratio 统计、clip fraction、旧新 KL、早停次数 |
| 不变量 | 基础模型参数 hash、旧策略版本、倍率/概率对应检查 |
| 异常 | 非有限样本、环境错误、初始状态复现失败、超预算终止 |

只绘制 `eta` 随时间变化不能证明它对应可解释的任务阶段。应结合控制条件、行为视频以及相同状态下的对照，区分相关性与因果作用。

### 8.3 Checkpoint 和恢复边界

建议 Stage-III checkpoint 保存：

```text
actor_state_dict
actor_config / feature_preprocessing_version / eta_mapping
optimizer_state_dict
iteration / actor_update_count / total_env_steps / completed_episodes
base_student_checkpoint_path / base_student_sha256
base_config / processor_and_action_statistics_identifiers
train_val_test_manifest_hashes
rollout_adapter_config / simulator_version / relevant_repository_commits
reward_config / advantage_config / objective_config
gate_rng_state / action_rng_state / sampling_rng_state
best_validation_metrics / deployment_mode / validation_seeds
```

基础 Student 可以单独保存，但 Actor 必须绑定准确的基础模型哈希。只保存 Actor 而丢失 action statistics、processor、倍率映射或基础 checkpoint 标识，不构成完整可复现部署。

首版只在“一轮完整采样和更新结束后”保存可恢复训练 checkpoint，不要求从仿真 episode 中间恢复。要做轮中恢复，必须同时保存行为策略版本、未消费缓存、环境/控制器状态和 RNG，不能仅保存 Adam 状态后接着跑。

## 9. 必须通过的测试和实验对照

### 9.1 代码级验收清单

| 编号 | 测试 | 通过条件 |
|---|---|---|
| T01 | 原 Stage-II 与 `eta=1` | 同观测、同噪声、同精度下前向一致；先记录 fp32 误差，再比较实际精度容差 |
| T02 | `eta=0` | 原 Goal→Local 残差为零；Goal→Plan 不被误关 |
| T03 | 同观测改变倍率 | Current/Goal 不受该单向通路直接影响；Local/动作是否变化被如实记录 |
| T04 | 序列维度 | 旧 Query 数、`L+16`、`L-16`、`L+240`、54、50×32 均保持 |
| T05 | Actor 初始化 | 确定性均值为 1；随机采样有非零方差；不混称两者等价 |
| T06 | 密度一致 | 第一次优化前新旧 ratio≈1，KL≈0；统一在缓存 `z` 上计算 |
| T07 | 冻结边界 | Actor 更新后基础参数无梯度、无数值变化，原模型始终 eval |
| T08 | 奖励与终止 | 失败奖励=0；同任务成功更早奖励更高；逐步中断准确计数 |
| T09 | 组内优势 | 不跨任务/初始条件混组；零方差组不制造梯度 |
| T10 | 轨迹长度 | 展平后每条轨迹总权重相同，不让长轨迹多占权重 |
| T11 | 更新方向 | 合成数据上正/负优势确实改变对应密度，梯度有限且非零 |
| T12 | 非法数据 | Beta 端点、NaN、旧策略版本不一致会报错，不静默裁剪 |
| T13 | 环境分组 | 同组起始状态可复现；后续状态允许分叉；不同 RNG 用途可追踪 |
| T14 | 采样次数 | 每个动作段只采一次倍率；不乘上 10 次 ODE 或旧 flow repeat |
| T15 | 无未来访问 | Student/Actor 不读取真实未来、`future_valid`、剩余示范长度或奖励输入 |
| T16 | 恢复训练 | 恢复后的 Actor、映射、基础权重和配置一致；不会错误续用过期行为概率 |

bitwise 一致性受硬件与 attention 内核影响时，应使用预先固定的数值容差并报告，而不是随意放宽到忽略实质差异。`eta=1` 比较失败时先修复模型改动，不能继续用后续成功率变化解释效果。

### 9.2 核心对照实验

| 组别 | 做法 | 能支持的判断 |
|---|---|---|
| A | 原 Stage II，`eta=1` | Stage III 的主要基线 |
| B | 同一个 Stage-II checkpoint，`eta=0` | 现有 VLM 交互路径的消融，不是独立训练的全隔离模型 |
| C | 验证集选定的最佳固定倍率 | 改善是否只需要一个全局常数 |
| D | 初始化后不训练的随机 Beta Actor | 改善是否只是随机扰动带来 |
| E | 输入为常量的可训练 Beta Actor，使用相同 RL 预算 | 改善是否只来自全局分布/倍率校准，而非状态相关调节 |
| F | 本版状态相关 Actor，奖励后训练 | 检验任务反馈指导自适应交互的价值 |

固定倍率必须在 RL-val 选择，不能看 Final-test 成绩再挑最优。E 和 F 应尽量采用相同交互预算；否则无法区分“状态输入有效”与“用了更多环境交互”。所有组使用相同冻结 Student、数据/环境划分、动作生成步数、图像与动作处理。

若状态相关 Actor 与最佳固定倍率或常量 Actor 表现相当，只能支持交互校准有用，不能据此声称已经验证“按任务状态选择性交互”的优势。若最终倍率基本恒定，也应如实报告，不人为要求它必须随阶段切换。

### 9.3 成功率与执行效率怎样报告

主指标采用逐任务成功率，再按声明的任务/suite 权重汇总。报告评估 episode 数、训练随机种子数、均值及不确定性；相同初始状态的多次运行不是新的独立任务，应按任务或初始条件分组进行配对统计。不能用全部 episode 混在一起的单一均值掩盖任务覆盖不均。

成功步数只在成功轨迹上有意义；无成功的任务记为 N/A，而不是 0。还应比较双方都成功的配对条件上的步数差，防止“只剩容易成功的任务，因此平均更快”的选择偏差。原始回报作为补充，不能替代成功率。

选择 checkpoint 时，先依据 RL-val 的成功率和其统计波动筛选；没有明显退化的候选之间再比较效率。这个选择规则是一项实验保护，不是算法保证。冻结参数并不保证行为或成功率不退化。

应报告额外 RL 环境交互量。Stage III 使用在线反馈，不能把它与纯离线训练的基线无条件称为完全同等训练资源比较。

## 10. 部署方式、后续扩展及论文边界

### 10.1 完成后训练后的部署

加载固定 Stage-II Student 和训练后的 Actor。每次当前观测到来后运行前 8 层，计算 Beta 参数，默认使用：

\[
\boxed{\eta_k^{deploy}=2\frac{\alpha_k}{\alpha_k+\beta_k}.}
\tag{13}
\]

然后运行后 8 层、原动作头和环境执行。部署不采集同组轨迹、不计算优势、不更新参数、不访问 Teacher 或真实未来。

均值部署是确定性选择，不是奖励最大化解。必须在 RL-val 同时检查随机 Actor 和均值 Actor，然后预先固定 Final-test 的部署方式。即使倍率确定，原动作头仍有 flow 噪声；“确定性门控”不等于整条策略完全确定。

### 10.2 逐层倍率仅作为后续版本

首版共享标量已验证后，可将 `eta_k` 扩展为第 9–16 层的 8 个倍率：

\[
\widetilde L^\ell=\bar L^\ell+\eta_k^\ell s_\ell r^\ell\odot M^\ell.
\]

Actor 输出 16 个数，组织成 8 对 Beta 参数；采样 `z in R^[B_e,8]`，映射成每层倍率。条件独立参数化下，**同一次门控决策的联合 log density 是 8 个分量之和**：

\[
\log p_\phi(\boldsymbol z\mid u)=\sum_{\ell=9}^{16}\log p_\phi(z^\ell\mid u).
\]

先得到该联合密度比再放入裁剪目标，不能未经说明地对 8 层各自裁剪后平均，冒充相同目标。KL 也应按分量合计后统计。该扩展增加探索和归因难度，不能仅凭一条轨迹终局回报宣称识别了每层的独立价值。

Beta 连续倍率可以近似压低某层消息，但不会给出严格的 0，也不会自动省掉该层计算。要做真正离散层选择、跳算或扩展到第 1–8 层，需要额外机制和训练验证，不能把它们写入当前首版已具备的能力。

### 10.3 最终方法叙述应与当前实现一致

三个阶段可以表述为：

> Stage I 以真实近端与远端未来训练动作相关教师后验；Stage II 通过多层蒸馏和动作监督学习学生先验及基础策略，并训练深层单向交互；Stage III 冻结已有模型，通过实际任务回报优化该交互的状态相关使用强度。

保留“未来知识的角色学习”和“任务奖励引导的跨尺度协同”两个研究点，但不能写成 Stage II 全层无交互、Stage III 首次建立通路，也不能把组内回报最高的某次样本称为已找到全局最优监督强度。

本版不声称学会任意候选动作的反事实后果、显式任务阶段、精确几何可行性或逐层独立价值。更合理的研究结论需要由第 9 节对照和行为分析支持：**在既有双尺度未来表征与交互通路上，任务反馈是否有助于学习更有效的使用策略。**

## 11. 原始依据与外部参考

**[D] 用户附件。** `demision.md`，标题为《LaWAM_aba2_change：Stage I / Stage II 完整训练流程与维度变化（LIBERO）》，文内核对日期 2026-10-11。本文继承其结构、维度、冻结范围、采样规则、损失和历史 checkpoint 记录；未独立复核其提到的本地源码或权重。原文件 SHA-256：

```text
fa1cb7039f5b8741cfea313301af90b04504e8981706865b113a37556300d49a
```

**[R1] GRPO 原始来源。** Zhihong Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. 2024，§4.1，尤其 outcome-level group-relative advantages。原始方法包含参考策略 KL 项等设置；本文不是原算法的逐项复刻，而是无 Critic 的连续门控适配。

`https://arxiv.org/html/2402.03300v3`

**[R2] 裁剪策略优化来源。** John Schulman et al. *Proximal Policy Optimization Algorithms*. 2017。

`https://arxiv.org/abs/1707.06347`

**[R3] 有界 Beta 连续策略来源。** Po-Wei Chou, Daniel Maturana, Sebastian Scherer. *Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution*. ICML 2017，PMLR 70:834–843。该工作不验证本文的未来表征门控。

`https://proceedings.mlr.press/v70/chou17a.html`

**[R4] PyTorch 官方 Beta 实现。** 用于核对参数含义、均值、采样和 `log_prob` 行为；运行工程仍应固定本地 PyTorch 版本。

`https://raw.githubusercontent.com/pytorch/pytorch/main/torch/distributions/beta.py`

**[R5] LIBERO 官方环境及评估代码。** 用于核对恢复初始状态、成功检查和固定初始状态评估；不据此替换用户既有部署约定。

`https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py`  
`https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/lifelong/metric.py`

**[R6] OpenAI Spinning Up，PPO 官方说明。** 用于核对 clipping 不是硬约束，以及通过 KL 监测提前停止后续更新的做法。

`https://spinningup.openai.com/en/latest/algorithms/ppo.html`

**[R7] PyTorch 官方 KL 实现。** 用于核对 Beta–Beta KL 的可用性。本文监测方向是 `old || new`，不混同为固定 reference-KL loss。

`https://raw.githubusercontent.com/pytorch/pytorch/main/torch/distributions/kl.py`

外部资料查询日期：2026-10-12。上述文献仅支持通用技术来源，不构成本方案在 LIBERO 上已有效、已收敛或具有首创性的证明。正式实验记录中应补充工程和依赖版本的 commit/version 标识，而不是把网页 `main/master` 当成不可变版本。

## 附录 A：可独立检查的 PyTorch 数值核心

以下代码实现 Actor、特征池化、回报、组内优势和带轨迹权重的裁剪损失。它**不包含用户工程的 Student 适配、LIBERO 环境构造、rollout 或完整训练器**，因此不是复制后即可运行全部 Stage III 的脚本。需按第 7 节接入现有工程。运行条件为 Python 3.10+、PyTorch；Actor 与分布运算置于 FP32、autocast 之外。

将代码保存为 `gate_core.py`，可用于先验证数值部分。

```python
"""Independent Stage-III gate/GRPO numerical reference, not a LIBERO trainer.
Requires Python 3.10+ and PyTorch. Run in FP32 outside autocast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Beta, kl_divergence


@torch.no_grad()
def pool_gate_features(local: Tensor, goal: Tensor, current: Tensor,
                       current_mask: Tensor) -> Tensor:
    """Read the un-final-normalized output of VLM block 8; return [B,6144]."""
    if local.ndim != 3 or local.shape[1:] != (8, 2048):
        raise ValueError('local must be [B,8,2048]')
    if goal.shape != local.shape or current.ndim != 3:
        raise ValueError('goal/current shape mismatch')
    if current.shape[0] != local.shape[0] or current.shape[-1] != 2048:
        raise ValueError('current must be [B,L,2048]')
    if current_mask.shape != current.shape[:2]:
        raise ValueError('current_mask must be [B,L]')
    mask = current_mask.to(torch.bool).unsqueeze(-1)
    count = mask.sum(dim=1)
    if (count == 0).any():
        raise ValueError('current context cannot be all padding')
    # New actor-only, parameter-free normalization; does NOT change VLM hidden.
    l_pool = local.float().mean(dim=1)
    g_pool = goal.float().mean(dim=1)
    c_pool = torch.where(mask, current.float(), 0.0).sum(dim=1) / count
    parts = [F.layer_norm(x, (2048,)) for x in (l_pool, g_pool, c_pool)]
    u = torch.cat(parts, dim=-1).detach()
    if not torch.isfinite(u).all():
        raise FloatingPointError('non-finite gate features')
    return u


class GateActor(nn.Module):
    """One Beta sample per action chunk; eta=2*z. All parameters belong to phi."""
    def __init__(self, hidden_dim: int = 256, initial_concentration: float = 20.0,
                 parameter_eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim <= 0 or initial_concentration <= 1.0 + parameter_eps:
            raise ValueError('invalid actor initialization')
        self.parameter_eps = float(parameter_eps)
        self.trunk = nn.Sequential(nn.Linear(6144, hidden_dim), nn.SiLU())
        self.head = nn.Linear(hidden_dim, 2)
        # 1 + softplus(bias) + eps == initial_concentration.
        v = initial_concentration - 1.0 - parameter_eps
        bias = v + math.log(-math.expm1(-v))
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, bias)

    def distribution(self, u: Tensor) -> Beta:
        if u.ndim != 2 or u.shape[-1] != 6144:
            raise ValueError('u must be [number_of_decisions,6144]')
        h = self.head(self.trunk(u.detach().float()))
        ab = 1.0 + F.softplus(h) + self.parameter_eps
        if not torch.isfinite(ab).all():
            raise FloatingPointError('non-finite Beta parameters')
        return Beta(ab[:, 0], ab[:, 1], validate_args=True)

    @torch.no_grad()
    def collect(self, u: Tensor) -> dict[str, Tensor]:
        dist = self.distribution(u)
        z = dist.sample()  # sample(), not a reparameterized simulator path.
        if not ((z > 0.0) & (z < 1.0)).all():
            raise FloatingPointError('Beta sample reached an endpoint; do not clamp')
        logp = dist.log_prob(z)
        if not torch.isfinite(logp).all():
            raise FloatingPointError('non-finite old log probability')
        return {
            'z': z.detach(), 'eta': (2.0 * z).detach(),
            'old_logp': logp.detach(),
            'old_alpha': dist.concentration1.detach(),
            'old_beta': dist.concentration0.detach(),
        }

    @torch.no_grad()
    def deterministic_eta(self, u: Tensor) -> Tensor:
        return 2.0 * self.distribution(u).mean  # [B]; NOT reward-optimal by definition.


@torch.no_grad()
def group_advantages(rewards: Tensor, std_skip: float = 1e-6,
                     std_floor: float = 1e-2) -> tuple[Tensor, Tensor, Tensor]:
    """rewards [J,N_grp]; invalid groups must be excluded BEFORE this function.
    Zero-variance complete groups stay in the batch with zero advantages.
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError('need [J,N_grp] with N_grp>=2')
    if not 0.0 <= std_skip < std_floor:
        raise ValueError('require 0 <= std_skip < std_floor')
    r = rewards.detach().float()
    if not torch.isfinite(r).all():
        raise FloatingPointError('non-finite reward')
    centered = r - r.mean(dim=1, keepdim=True)
    std = centered.square().mean(dim=1).sqrt()  # population std, correction=0.
    active = std > std_skip
    adv = centered / std.clamp_min(std_floor).unsqueeze(1)
    adv = torch.where(active.unsqueeze(1), adv, torch.zeros_like(adv))
    return adv, std, active


@dataclass
class FlatGateBatch:
    features: Tensor       # [D,6144], cached normalized features used by old actor.
    z: Tensor              # [D], original Beta latent, never eta.
    old_logp: Tensor        # [D], old Beta(z) log density.
    old_alpha: Tensor      # [D]
    old_beta: Tensor       # [D]
    advantage: Tensor      # [D], the trajectory advantage repeated for each decision.
    trajectory_id: Tensor  # [D], contiguous IDs 0..M-1.
    lengths: Tensor        # [M], positive number of decisions K_i.

    def validate(self) -> None:
        d = self.features.shape[0]
        if self.features.shape != (d, 6144) or d == 0:
            raise ValueError('invalid feature shape')
        for value in (self.z, self.old_logp, self.old_alpha,
                      self.old_beta, self.advantage, self.trajectory_id):
            if value.shape != (d,):
                raise ValueError('all per-decision fields must be [D]')
        m = self.lengths.numel()
        if self.lengths.ndim != 1 or m == 0 or (self.lengths <= 0).any():
            raise ValueError('lengths must be a nonempty positive [M] tensor')
        if self.trajectory_id.dtype != torch.long:
            raise ValueError('trajectory_id must be int64')
        if (self.trajectory_id < 0).any() or (self.trajectory_id >= m).any():
            raise ValueError('out-of-range trajectory id')
        counts = torch.bincount(self.trajectory_id, minlength=m)
        if not torch.equal(counts, self.lengths.to(counts)):
            raise ValueError('a trajectory was truncated or counted incorrectly')
        if not ((self.z > 0) & (self.z < 1)).all():
            raise ValueError('z must be in (0,1), not eta or clipped samples')
        for x in (self.features, self.z, self.old_logp,
                  self.old_alpha, self.old_beta, self.advantage):
            if not torch.isfinite(x).all():
                raise FloatingPointError('non-finite cached data')
        if (self.old_alpha <= 0).any() or (self.old_beta <= 0).any():
            raise ValueError('old Beta parameters must be positive')
        # Require a single trajectory-level advantage, not arbitrary token rewards.
        for tid in range(m):
            a = self.advantage[self.trajectory_id == tid]
            if not torch.allclose(a, a[:1].expand_as(a), atol=1e-6, rtol=1e-6):
                raise ValueError('inconsistent advantage within a trajectory')

    def weights(self) -> Tensor:
        # Equal groups (fixed N_grp), equal trajectories, then equal decisions within each.
        return 1.0 / (self.lengths.numel() * self.lengths[self.trajectory_id].float())


def clipped_gate_loss(actor: GateActor, batch: FlatGateBatch,
                      clip_eps: float = 0.2) -> tuple[Tensor, dict[str, float]]:
    if not 0.0 < clip_eps < 1.0:
        raise ValueError('clip_eps must be in (0,1)')
    batch.validate()
    dist = actor.distribution(batch.features.detach())
    new_logp = dist.log_prob(batch.z.detach())
    log_ratio = new_logp - batch.old_logp.detach()
    if not torch.isfinite(log_ratio).all() or (log_ratio.abs() > 60).any():
        raise FloatingPointError('policy/data mismatch or unstable step; do not clamp ratios')
    ratio = log_ratio.exp()
    adv = batch.advantage.detach()
    objective = torch.minimum(ratio * adv,
                              ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv)
    w = batch.weights()
    loss = -(w * objective).sum()
    with torch.no_grad():
        old = Beta(batch.old_alpha.detach(), batch.old_beta.detach(), validate_args=True)
        kl = kl_divergence(old, dist)  # old || current, at recorded states only.
        metrics = {
            'loss': float(loss.detach()),
            'kl_old_new': float((w * kl).sum()),
            'clip_fraction': float((w * ((ratio - 1).abs() > clip_eps)).sum()),
            'mean_eta': float((w * (2.0 * dist.mean)).sum()),
            'beta_entropy': float((w * dist.entropy()).sum()),
            'ratio_max_error': float((ratio - 1.0).abs().max()),
        }
    if not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('non-finite loss/diagnostics')
    return loss, metrics


def success_efficiency_reward(success: bool, steps: int, max_steps: int,
                              efficiency_weight: float = 0.1) -> float:
    if max_steps <= 0 or not 0 <= steps <= max_steps:
        raise ValueError('invalid executed-step count')
    if not 0.0 <= efficiency_weight < 1.0:
        raise ValueError('efficiency_weight must be in [0,1)')
    return float(success) * (1.0 + efficiency_weight * (1.0 - steps / max_steps))
```

### 附录 A.1：Actor 更新的最小调用方式

下面的 `batch` 必须是同一行为策略快照下收集的完整轨迹缓存，轨迹优势已按组算好。参数更新只针对 `actor.parameters()`。先验收旧新概率一致，再执行更新；示例中没有加入 reference-KL loss。

```python
actor = actor.float()
optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4, weight_decay=0.0)

# 每轮 collect 完成后；不要每轮重新创建上述 optimizer。
if batch.advantage.abs().max().item() > 0:
    _, initial_metrics = clipped_gate_loss(actor, batch)
    if initial_metrics['ratio_max_error'] > 1e-4:
        raise RuntimeError('fresh buffer and current actor are not the same behavior version')

    for _ in range(2):
        loss, metrics = clipped_gate_loss(actor, batch)
        if metrics['kl_old_new'] > 0.01:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            _, metrics_after = clipped_gate_loss(actor, batch)
        if metrics_after['kl_old_new'] > 0.01:
            break
# 完成本轮后重新 collect；不长期重放该 buffer。
```

`FlatGateBatch` 的所有 Tensor 需处于相同设备；`trajectory_id` 为 int64。该参考代码按完整缓存更新，避免随机抽取门控决策导致轨迹权重错误。大规模训练可按小块累加 `1/(M*K_i)` 权重的梯度，但必须保持式 (11) 的求和关系。

### 附录 A.2：本次已完成的数值检查

本次在 CPU、PyTorch `2.10.0+cpu` 下执行了合成数据检查：

| 检查 | 结果 |
|---|---|
| 第 8 层池化特征尺寸 | `[2,6144]` |
| Actor 参数量 | 1,573,634 |
| 初始 alpha / beta | 两者均为 20，确定性均值倍率为 1 |
| 初始倍率理论标准差 | 约 0.156174 |
| 全零回报组 | 优势全为 0 |
| 不同长度轨迹 | 每条轨迹总权重相同 |
| 更新前密度比 | 最大误差为 0；旧新 KL 为 0 |
| 正负优势的反向传播 | Actor 得到有限、非零梯度 |
| 优化后密度 | 与旧策略出现变化 |
| 奖励 | 成功 120/300 步、效率权重 0.1 得 1.06；失败得 0 |
| 残差倍率 | 合成张量下 `eta=1` 恢复旧残差，`eta=0` 移除残差，包括负 scale 情况 |
| 非法 Beta 端点 | 明确报错，不静默裁剪 |

上述“残差检查”只验证合成张量公式，不代替完整 Student 的前向等价测试；“密度变化”不是任务成功率提升。真实环境集成、动作输出敏感性、训练收敛、全部消融和正式性能数据均尚待在用户工程中验证。

## 附录 B：实施交付完成条件

Stage III 可以进入正式实验前，至少应具备：正确绑定的 Stage-II checkpoint；可重复的 `eta=1` 基线；经检查的交互敏感性；互斥且可追溯的在线 manifest；逐环境步的成功计数；完整门控缓存；旧新概率一致性；只更新 Actor 的参数检查；在线验证和恢复机制。

最终产物应包含 Stage-III Actor 权重、基础 Student 标识、实验配置、RL 训练成本、验证记录和正式对照结果。本文给出的是完成这些工作所需的实施规格，不以文档完成代替实验完成。
