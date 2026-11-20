**GRPO.md 中的 Stage III，可以理解为：前两阶段已经训练出“会预测未来、会利用未来信息生成动作”的模型，第三阶段不重学这些能力，而是通过实际任务结果，学习如何更合适地使用已有的 Goal→Local 交互。** 它调整的是已有交互的使用强度，不是第一次建立交互通路。GRPO

为了把强化学习讲清楚，下面始终围绕两个过程展开：

**执行时：门控倍率怎样改变 Local 表征，进而改变动作？**

**训练时：任务成功或失败，怎样变成更新门控网络的梯度？**

## 一、先接上 Stage I、II：第三阶段从什么基础上开始？

### Stage I 留下的是“有真实未来可看的教师”

你的 Stage I 从示范轨迹中读取近端双视角 \(o_{t+H}\) 和远端有序观测 \(\{o_{t+nH}\}\)，分别编码、压缩为两组未来 Token：

\[
F_L,F_G\in\mathbb R^{B\times8\times2048}.
\]

它们与当前上下文一起进入 Teacher VLM，产生 Local、Goal 两类后验，再沿两条路径指导动作：

\[
\boxed{
\mathrm{Local}\rightarrow\mathrm{Action},
\qquad
\mathrm{Goal}\rightarrow\mathrm{Plan}\rightarrow\mathrm{Action}.
}
\]

训练目标是动作 Flow Matching：

\[
\mathcal L_{\mathrm I}=\mathcal L_{\mathrm{flow}}^T.
\]

也就是在有效动作位置上，比较预测速度 \(\hat v_T\) 与目标速度 \(A^{pad}-\epsilon\) 的误差。这样，两个 Resampler 和 Teacher VLM 学到的是**对动作生成有用的未来信息**，而不是仅仅把未来图像压缩成特征。GRPO demision

### Stage II 留下的是“只看当前信息，也能预测并使用未来知识的学生”

Student 不再读取真实未来，而是加入两组可学习 Query：

\[
[C;Q_G;Q_L]
\xrightarrow{\mathrm{Student\ VLM}}
[H_C;Z_G;Z_L].
\]

它同时接受动作监督和两类未来表征蒸馏：

\[
\boxed{
\mathcal L_{\mathrm{II}}
=
\mathcal L_{\mathrm{flow}}^S
+\lambda_L\mathcal L_L^{dist}
+\lambda_G\mathcal L_G^{dist}.
}
\]

其中，蒸馏让 Student 学习教师的未来表征，动作损失让 Student 学会使用这些表征。你实际的 Stage II 还在第 9–16 层训练了 Goal→Local 交互；第 12、16 层的 Local 蒸馏发生在交互之后。demision

所以，**Stage II 已经存在随输入变化的相关性门控。Stage III 的新增之处不是“第一次实现自适应交互”，而是“用真实任务回报训练一个额外的交互调节策略”。**

两阶段的区别可以这样理解：

> Stage II 根据“表征是否接近教师、动作是否接近示范”训练交互；Stage III 根据“实际任务是否完成、完成得是否高效”调整交互的使用强度。GRPO GRPO

---

## 二、Stage III 究竟改了哪个公式？

你实际 Stage II 的深层交互可以写成：

\[
\boxed{
\widetilde L^\ell
=
\bar L^\ell+
s_\ell r^\ell\odot M^\ell.
}
\]

这里，\(\ell\) 是 VLM 层号。

| 符号 | 通俗理解 |
|---|---|
| \(\bar L^\ell\) | 当前层尚未接收 Goal 消息的 Local 表征 |
| \(M^\ell\) | 从 Goal 中提取出来、准备传给 Local 的信息 |
| \(r^\ell\) | 每个 Local 槽接收 Goal 信息的程度 |
| \(s_\ell\) | Stage II 已经学到的该层残差系数 |
| \(\widetilde L^\ell\) | 接收 Goal 信息之后的 Local 表征 |

其中，相关性为：

\[
r^\ell=1-P^\ell(\mathrm{NULL}).
\]

可以把 NULL 理解为“不接收 Goal 信息”这个选项。某个 Local 槽给 NULL 的权重越大，\(r^\ell\) 就越小。而真正传递什么内容，由另一套 message attention 计算：

\[
M^\ell
=
\operatorname{MHA}_{\ell}
\bigl(
\operatorname{LN}(\bar L^\ell),
\operatorname{LN}(G^\ell),
\operatorname{LN}(G^\ell)
\bigr).
\]

**\(r^\ell\) 决定接收程度，\(M^\ell\) 决定接收内容，两者不是同一个注意力计算。** GRPO

Stage III 保留这些已经训练好的模块，只在原残差外增加一个倍率：

\[
\boxed{
\widetilde L^\ell
=
\bar L^\ell+
\underbrace{\eta_k}_{\text{新增的奖励驱动倍率}}
\underbrace{s_\ell r^\ell\odot M^\ell}_{\text{Stage II 已学到的交互}},
\qquad \ell=9,\ldots,16.
}
\]

\(k\) 表示第几次动作段决策。文档首版为**每个动作段产生一个倍率，统一用于本次 VLM 前向的第 9–16 层**，不是逐层各产生一个倍率。GRPO

这个倍率的含义很直接：

| 倍率 | 对已有交互的影响 |
|---|---|
| \(\eta_k=1\) | 保留原 Stage II 的交互 |
| \(\eta_k=0.7\) | 将原交互残差缩小到 70% |
| \(\eta_k=1.3\) | 将原交互残差放大到 130% |
| \(\eta_k=0\) | 关闭 VLM 内的 Goal→Local 残差 |

**\(\eta_k=0\) 不会关闭 Goal→Plan→Action，Goal 仍可通过动作头影响动作。** 此外，训练时采用的连续倍率范围是 \((0,2)\)，精确的 0 是额外设置的消融模式。GRPO

这里也能回答一个常见疑惑：

> VLM、原门控和 Action DiT 全部冻结了，动作为什么还能变化？

因为**冻结的是参数，不是隐藏表征**。倍率改变之后，Local 接收到的消息发生变化，后续 Local 表征及动作条件也会变化，最终动作就可能不同。并且后续层的 \(r^\ell\)、\(M^\ell\) 会根据已经变化的 Local 重新计算，并非整段网络始终使用一份固定消息。GRPO

---

## 三、跟着一次动作段决策，看完整前向流程

下面设 \(B_e\) 为同时进行前向计算的环境数量。它与 Stage II 的训练 batch、\(B_f=2B\) 的 flow 重复采样，以及 GRPO 的轨迹组大小都不是同一个概念。GRPO

### 第一步：照常运行 Student 的前 8 层

输入仍然是当前主视角、腕部图像和任务指令。保留旧 8 个 Action Query、8 个 Flow Query，再追加新 Goal、Local Query：

\[
X^0=[C;Q_G;Q_L]
\in\mathbb R^{B_e\times(L+16)\times2048}.
\]

前 8 层运行完后，取出：

\[
\begin{aligned}
L^8,G^8&\in\mathbb R^{B_e\times8\times2048},\\
H_C^8&\in\mathbb R^{B_e\times L\times2048}.
\end{aligned}
\]

**到这一步，计算与你现有 Stage II 相同。** 门控网络读取的是第 8 个 block 的输出，此时还没有经过 VLM 最后的 RMSNorm。GRPO

### 第二步：把三组表征汇总，交给新增的小网络

Local 的 8 个 Token 求平均，Goal 的 8 个 Token 求平均，Current 对有效 Token 求平均，得到三个 2048 维向量：

\[
p_L,p_G,p_C\in\mathbb R^{B_e\times2048}.
\]

文档对这三个池化向量分别做无可学习参数的归一化，然后拼接：

\[
\boxed{
u_k=
\operatorname{sg}
\left[
\operatorname{LN}_0(p_L);
\operatorname{LN}_0(p_G);
\operatorname{LN}_0(p_C)
\right]
\in\mathbb R^{B_e\times6144}.
}
\]

这里 \(2048\times3=6144\)。\(\operatorname{sg}\) 表示停止梯度，后面的强化学习不通过它修改 Student。Current 的池化保留旧 Query 的有效位置，但排除 padding。GRPO

可以将 \(u_k\) 理解为：

> 门控网络在决定交互强度之前，先同时查看“当前上下文、近端预测、远端预测”的汇总信息。

它没有读取真实未来、任务最终奖励或仿真器的完整状态。

### 第三步：门控网络不是直接给唯一答案，而是先给出一个采样分布

新增网络叫作 **GateActor**，也就是“决定门控倍率的策略网络”。结构为：

\[
u_k
\xrightarrow{\mathrm{Linear}\ 6144\to256}
\xrightarrow{\mathrm{SiLU}}
\xrightarrow{\mathrm{Linear}\ 256\to2}
(h_{\alpha,k},h_{\beta,k}).
\]

两个输出经过变换，形成 Beta 分布的参数：

\[
\alpha_k=1+\operatorname{softplus}(h_{\alpha,k})+\varepsilon_\pi,
\qquad
\beta_k=1+\operatorname{softplus}(h_{\beta,k})+\varepsilon_\pi.
\]

然后：

\[
\boxed{
z_k\sim\operatorname{Beta}(\alpha_k,\beta_k),
\qquad
\eta_k=2z_k.
}
\]

因为 \(z_k\in(0,1)\)，所以倍率 \(\eta_k\in(0,2)\)。这些结构、变换和范围都是 GRPO.md 的首版设计。GRPO

**为什么训练时要采样，而不是直接输出一个固定倍率？**

因为需要尝试不同交互程度，并通过执行结果判断哪些选择更值得保留。Beta 是用于有界连续策略的一种已有参数化方法；它本身不意味着你的门控一定有效，效果仍需在当前模型上验证。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/chou17a.html)

文档将初始分布设为：

\[
\alpha=\beta=20.
\]

此时：

\[
\mathbb E[\eta]=1,
\qquad
\operatorname{Std}(\eta)\approx0.156.
\]

也就是**先围绕原 Stage II 的倍率 1 做相对保守的随机探索**，而不是从一开始就大幅改变现有策略。需要区分：平均倍率为 1，不代表每次采样都是 1；验证与 Stage II 完全一致时，必须直接固定 \(\eta=1\)。GRPO

### 第四步：带着这个倍率，继续运行后 8 层

第 9–16 层仍然先运行原 VLM block，再在原有 RelevanceGate 的位置执行：

\[
\widetilde L^\ell
=
\bar L^\ell+
\eta_k s_\ell r^\ell\odot M^\ell.
\]

维度不变：

\[
r^\ell:\ B_e\times8\times1,
\qquad
M^\ell:\ B_e\times8\times2048.
\]

倍率从 \([B_e]\) 变成 \([B_e,1,1]\)，广播到 Local 的各个槽和通道。第 16 层交互之后，再执行原来的最终 RMSNorm。GRPO

**本次前向中，倍率直接改变的是 Local，不直接改变 Goal 或 Current。** 但动作执行后环境不同，下一次观测下三组表征都可能不同。

### 第五步：仍然由原 Action DiT 生成动作

最终两类表征投影为：

\[
G_A,L_A\in\mathbb R^{B_e\times8\times1024}.
\]

当前 VLM 输出删除旧 16 个 Query 的对应位置，经投影后与 DINO 拼接，得到：

\[
C_{\mathrm{base}}
\in\mathbb R^{B_e\times(L+240)\times768}.
\]

原有两条动作条件路径保持不变：

\[
G_A\rightarrow\mathrm{Plan}\rightarrow\mathrm{Action},
\qquad
L_A\rightarrow\mathrm{Action}.
\]

**新倍率只调节 VLM 内的交互残差，不直接乘到 Action DiT 的 Goal 或 Local 条件上。** GRPO

Action DiT 从噪声出发，执行原来的 10 次 Euler 更新：

\[
x_{j+1}
=
\left[
x_j+\frac1{10}
v_{\bar\theta}
\left(x_j,j/10;C_{\mathrm{base}},G_A,L_A\right)
\right]\odot m_{\mathrm{token}}.
\]

每次速度预测仍然使用：

\[
[B_e,54,1024]
\longrightarrow
[B_e,50,32].
\]

最终取有效部分并进行原来的后处理，得到：

\[
A_k\in\mathbb R^{B_e\times8\times7}.
\]

**Stage III 仍然使用 Flow 模型生成动作，但不再计算 Flow Matching 损失来训练它。** 一次动作段只编码一次 VLM/DINO、采样一次门控，随后 10 次速度计算复用这些条件。GRPO

正常执行 8 个环境步后，读取新观测，再重复以上过程。若本段第 3 步已经完成任务，就立即停止，不必执行剩余 5 步。GRPO

---

## 四、执行结果怎样变成“交互价值”？

前面只说明了倍率怎样影响动作，还没有发生强化学习更新。

**真正的学习，需要先执行完整任务，再比较执行结果。**

### 先从同一个初始条件出发，执行一组完整轨迹

文档建议每组采集 4 条轨迹。同一组使用相同任务、指令和实际仿真初始状态，但各条轨迹独立采样门控与动作噪声。它们后续经历的状态可以不同。GRPO GRPO

一条轨迹不是只使用一个倍率，而是包含多次动作段决策：

\[
u_1\rightarrow\eta_1\rightarrow A_1
\rightarrow u_2\rightarrow\eta_2\rightarrow A_2
\rightarrow\cdots
\rightarrow\text{任务结束}.
\]

例如，第 1 次决策可能是接近抽屉，第 2 次是拉开抽屉，后续再执行放置和关闭。这只是帮助理解的任务示意，模型并没有显式获得这些阶段标签。

每次决策至少要保存：

\[
\boxed{
u_{i,k},\quad z_{i,k},\quad
\log p_{\phi_{\mathrm{old}}}(z_{i,k}\mid u_{i,k}).
}
\]

分别表示“当时看到什么特征”“当时采样了什么”“当时的策略对该采样值赋予多少对数密度”。这些数据随后用于更新 Actor。整个采集批次中，行为策略 \(\phi_{\mathrm{old}}\) 必须保持不变。GRPO

### 用任务成功和实际执行步数计算奖励

GRPO.md 的回报为：

\[
\boxed{
R_i
=
y_i
\left[
1+\lambda_{\mathrm{eff}}
\left(1-\frac{T_i}{T_{\max,i}}\right)
\right].
}
\]

其中，\(y_i=1\) 表示成功，失败则为 0；\(T_i\) 是实际执行的环境步数，\(T_{\max,i}\) 是对应任务的步数上限。文档建议先取 \(\lambda_{\mathrm{eff}}=0.1\)。GRPO

通俗来说：

> 完成任务获得基础分；成功且更快，额外加一点分；失败不给效率分。

沿用文档中的**数学示例，而非实验结果**，设 \(T_{\max}=300\)：

| 轨迹 | 结果 | 实际步数 | 回报 |
|---|---|---:|---:|
| 1 | 成功 | 120 | 1.06 |
| 2 | 成功 | 180 | 1.04 |
| 3 | 失败 | 300 | 0 |
| 4 | 成功 | 150 | 1.05 |

这里的 120、180 是环境真正执行的动作步数，和 DiT 的 10 次积分没有关系。一次轨迹只有“成功或失败”的指示，多条评估轨迹汇总后才得到成功率。GRPO

### 再比较：这条轨迹比同组其他轨迹好多少？

GRPO 的核心是**组内相对比较**。原始 GRPO 使用同一输入下的一组候选回报构造相对优势，无需另外训练一个 Critic，即预测回报的价值网络。你的文档将“候选答案”替换为“同一初始条件下的完整执行轨迹”。[arXiv](https://arxiv.org/html/2402.03300v3)

对于第 \(j\) 组，先计算平均回报 \(\bar R_j\) 和回报标准差 \(s_j\)，再计算：

\[
\boxed{
\widehat{\mathcal A}_{j,i}
=
\frac{R_{j,i}-\bar R_j}
{\max(s_j,0.01)}.
}
\]

这是文档在组内标准差大于 \(10^{-6}\) 时使用的公式；标准差过小时，整组优势直接设为 0。分母下限是本方案额外加入的数值保护。GRPO

\(\widehat{\mathcal A}\) 叫作**优势**，含义是：

> 这条轨迹的表现，相对于同组平均表现，好多少或差多少。

上述四条轨迹的优势约为：

\[
[0.5993,\ 0.5553,\ -1.7318,\ 0.5773].
\]

三条成功轨迹获得正优势，失败轨迹获得负优势；成功且更快的轨迹优势稍高。GRPO

文档首版将整条轨迹的优势赋给该轨迹中的所有门控决策：

\[
\widehat{\mathcal A}_{j,i,k}
=
\widehat{\mathcal A}_{j,i}.
\]

**这并不证明成功轨迹里的每次门控都正确，也不证明失败轨迹里的每次门控都错误。** 它只是用整条任务的结果给沿途门控提供一个共同的学习方向，暂时无法精确区分每次门控的独立贡献。GRPO

---

## 五、最关键的一步：奖励到底怎样更新 GateActor？

这一部分可以抓住一句话：

**不是把高奖励当成一个“正确倍率标签”，而是提高高优势轨迹中已有门控选择的概率密度，降低低优势轨迹中相应选择的概率密度。**

例如，在某个状态下采样了 \(\eta=0.7\)，最终表现较好，学习方向可以是今后更倾向于选择类似的较小倍率。**奖励高不等于把倍率调大。** GRPO

### 先在旧数据上，重新计算当前 Actor 的密度

轨迹采集结束后，不必重新运行整段 VLM，也不必再次执行环境。

将缓存的同一个 \(u_{i,k}\) 输入当前 GateActor，计算它对**当时已经采样、已经执行过的同一个 \(z_{i,k}\)** 的密度：

\[
p_\phi(z_{i,k}\mid u_{i,k}).
\]

与采样时保存的旧密度比较：

\[
\boxed{
\rho_{j,i,k}(\phi)
=
\exp\left[
\log p_\phi(z_{j,i,k}\mid u_{j,i,k})
-
\log p_{\phi_{\mathrm{old}}}(z_{j,i,k}\mid u_{j,i,k})
\right].
}
\]

\(\rho>1\) 表示当前 Actor 更倾向于当时的选择，\(\rho<1\) 表示倾向降低。这里比较的是连续分布的**密度**，不是某个精确小数的离散发生概率。GRPO

需要特别区分：**更新时不重新采样一个新的 \(z\) 来替换旧 \(z\)**。新的采样没有参与刚才那条轨迹，不能直接配上那条轨迹的奖励。

另外，文档存储和计算的是 \(z\) 的 Beta 密度，不是把 \(\eta=2z\) 直接传给 Beta 分布。两种变量相差的固定变换因子会在新旧密度比中抵消。GRPO

### 用优势决定方向，用裁剪抑制过大的单次变化

文档的完整损失是：

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{III}}(\phi)
=
-\frac1J\sum_{j=1}^{J}
\frac1{N_{\mathrm{grp}}}\sum_{i=1}^{N_{\mathrm{grp}}}
\frac1{K_{j,i}}\sum_{k=1}^{K_{j,i}}
\min\Big[
&\rho_{j,i,k}\widehat{\mathcal A}_{j,i},\\
&\operatorname{clip}(\rho_{j,i,k},1-\epsilon_c,1+\epsilon_c)
\widehat{\mathcal A}_{j,i}
\Big].
\end{aligned}
}
\]

其中，\(J\) 是组数，\(N_{\mathrm{grp}}\) 是每组轨迹数，\(K_{j,i}\) 是该轨迹的门控决策次数；文档建议 \(\epsilon_c=0.2\)。GRPO

这个公式可以拆成三个作用理解。

**第一，优势决定鼓励还是抑制。** 正优势鼓励增加当时所采样门控的密度，负优势鼓励降低它。

**第二，裁剪让继续大幅改变密度不再带来同样的目标收益。** 这里裁剪的是新旧密度比，不是把门控倍率限制在 \(0.8\) 到 \(1.2\)。PPO 的裁剪也不是“策略绝不会变化过大”的硬性保证，因此文档还监测新旧策略的 KL 差异。[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

**第三，先按轨迹内决策平均，再按轨迹和组平均。** 例如，一条轨迹有 20 次门控，另一条有 40 次，后者不会仅因为更长就获得两倍权重。因此不能把所有门控记录直接展平后无差别求平均。GRPO

### 为什么不对仿真器求导，也能训练？

因为 GateActor 输出的 Beta 分布参数是神经网络计算得到的，所以：

\[
\log p_\phi(z\mid u)
\]

能够对 Actor 参数 \(\phi\) 求梯度。

执行结果产生的奖励、优势、已采样倍率以及缓存特征，在更新时都视为固定数据。它们通过上述损失告诉 Actor：**当前应该增加还是减少这些选择的倾向。**

最后执行：

\[
\phi
\leftarrow
\phi-\xi\nabla_\phi\mathcal L_{\mathrm{III}},
\]

其中 \(\xi\) 是学习率。

因此，实际的梯度路径是：

\[
\boxed{
\mathcal L_{\mathrm{III}}
\rightarrow
\log p_\phi(z\mid u)
\rightarrow
\mathrm{GateActor}.
}
\]

**梯度不需要穿过动作执行、仿真器或 10 次 Euler 积分；也不更新旧 RelevanceGate、Student VLM 和 Action DiT。** 冻结模型负责让门控选择产生行为后果，Actor 的密度负责提供可训练的梯度。GRPO GRPO

还有一个容易误判的现象：首次更新前，新旧 Actor 相同，\(\rho\approx1\)，组内优势平均又接近 0，所以数值 loss 可能接近 0。**这不等于没有梯度**，因为不同状态和采样值对应的密度梯度并不相同。GRPO

---

## 六、把整轮训练和最终部署连起来

现在可以把 Stage III 分为两个交替发生的环节。

**环节一：执行任务，收集经验。** 固定当前 GateActor 的行为策略快照，从训练划分选择若干初始条件；每个条件执行一组完整轨迹，保存门控记录、成功结果和实际步数。

**环节二：根据这批经验更新 Actor。** 在每个组内计算优势，在缓存的特征和采样值上计算裁剪损失，只更新 GateActor。文档建议每轮最多做 2 次整批更新；旧新 Beta 分布的平均 KL 超过 0.01 时，停止继续使用该批数据更新。随后重新采集轨迹，而不是长期反复使用旧数据。这里的 KL 是监测及早停依据，不是额外加入损失的固定参考策略惩罚项。GRPO

整个循环就是：

```text
固定当前门控策略
    ↓
从相同初始条件执行一组完整任务
    ↓
根据成功与实际步数计算回报
    ↓
在每组内部计算相对优势
    ↓
用缓存的特征、门控采样值和旧密度更新 GateActor
    ↓
用更新后的 GateActor 重新执行任务
```

这里有三个数字不能混淆：

| 数字 | 在文档中的含义 |
|---|---|
| 8 个环境步 | 一段动作的正常执行长度；提前终止时可能不足 8 步 |
| 10 次 Euler 更新 | 生成这段动作时，动作头内部的计算次数 |
| 每组 4 条轨迹 | 为比较任务回报而重复执行的完整任务数量 |

前两阶段的 \(B_f=2B\) 只是离线 Flow Matching 的重复噪声采样，不参与这个在线 GRPO 流程。GRPO GRPO

### 训练完成后，不再执行组内比较

部署时，每次取得当前观测，运行前 8 层并得到 \(\alpha_k,\beta_k\)。文档默认用 Beta 分布的均值确定倍率：

\[
\boxed{
\eta_k^{deploy}
=
2\frac{\alpha_k}{\alpha_k+\beta_k}.
}
\]

随后运行后 8 层和原动作头，执行动作。**不需要先试四次任务再选一次，也不计算奖励或更新参数。** 均值部署是否优于随机采样，需要单独验证；它不是天然的最优倍率。GRPO

### 最后，怎样理解这一阶段能够学到什么？

它试图学到的是：

> **面对当前观测和两类未来预测，这一次应当更多还是更少地使用已有的 Goal→Local 消息，才能提高后续任务的执行回报。**

这里的“价值判断”没有对应一个额外的价值网络，而是通过**组内轨迹比较、优势计算和门控策略更新**体现出来。

但文档也保留了几个重要边界：全失败或同回报组没有有效优势；整条轨迹奖励不能精确评价每一次门控；回报提高也不自动意味着成功率提高，所以需要分别报告成功率和成功步数。还应与原 Stage II、最佳固定倍率以及常量输入的 Actor 比较，才能判断收益是否真正来自状态相关的调节。GRPO GRPO

**三个阶段最终连成的逻辑是：Stage I 学习真实未来中的动作相关知识；Stage II 让学生仅凭当前信息预测、使用这些知识并形成初步交互；Stage III 根据实际执行回报，优化已有跨尺度交互的使用强度。** GRPO