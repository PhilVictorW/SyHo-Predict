# 从 aba2 到 Goal/Local LaWAM：逐版改造方案

> 本文是待逐版讨论、实施的方案，不代表代码已完成改造。本次只写文档，不修改模型、不启动训练或评测。
> 工作目录：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2`。
> 基线：已完成评测的 aba2，1945/2000，LIBERO 成功率 97.25%。

## 1. 本次采用的最终架构

采用 `idea_9_9_11.01_new.md` 的核心机制：**Goal/Local prior queries、同一次 forward 中的 privileged posterior、逐层过程蒸馏、V-JEPA 终点监督、Goal→Plan / Local→Action 分层条件注入**。

采用 `change(4).md` 中与当前 LaWAM 兼容的工程形式：**Qwen 本身承担 World Reasoner，保留 16 层、2048 维；动作隐藏维度保持 1024，交叉注意力 memory 保持 768 维**。本方案不额外创建原 Idea 中独立的 8 层、768 维 World Reasoner。

本次用户要求覆盖旧文档的条款：

| 项目 | 本方案 |
|---|---|
| 起点 | V0 为 aba2；不恢复 LaWM 未来条件 |
| 当前观测 | 当前主视角＋当前腕部相机，沿用原预处理 |
| Local future | 仅主视角，`t_local=t+H`；默认 `H=floor(0.4×20)=8` |
| Goal future | 仅主视角，成功示范 episode 的最后一帧；长度为 T 时 `t_goal=T−1` |
| 未来视觉编码 | 建议沿用 `change(4).md` 的共享 Qwen Vision＋Spatial Merger；不沿用原 Idea 的未来双视角 DINO |
| 原 Q_act / Q_flow | 各 8 个，保留在 VLM 内，保留原位置、顺序、checkpoint 初始化；不直接进入动作 memory |
| 原两段 prompt | 全程保留原文，不执行旧计划中的 prompt 替换 |
| 新 Q_G / Q_L | 各 8 个，随机初始化，不能用 Q_act 或 VLMToLAM 权重替代 |
| 状态输入 | 保持 `use_state=false`；不同时加入 proprioception |
| 动作时间尺度 | 0.4 秒、20 Hz、8 个有效动作；内部 buffer 50×32 不变 |
| 当前 DINO | 前期保留；后期单独验证移除，不与新条件注入一起删除 |

未来主视角使用 Qwen、V-JEPA 也只使用主视角，是本版的明确设计选择。若后续决定改用 DINO future encoder，另立版本，不混入同一轮比较。

**V0 的准确边界：** 当前 aba2 仍会计算 `Q_act hidden→Q-Former→[B,1,32]`，但结果不参与动作推理；LaWM decoder 被跳过。V0 原样存档，在新模型的 V2 中再移除这段无效计算。保留 Q_act 本身不等于保留 Q-Former。

## 2. 逐版实施顺序

每次只实施当次讨论的版本，验证后再进入下一版；不一次改完。原 `LaWAM` 入口和 aba2 消融评测入口保留，新建 `GoalLocalLaWAM` 入口与版本配置。

| 版本 | 具体改动 | 验收重点 |
|---|---|---|
| **V0：冻结基线** | 保存现有 aba2 代码、checkpoint 路径、配置和 97.25% 评测记录；不改计算图 | 随时可复现当前 aba2 |
| **V1：只做数据** | 增加当前观测、Local 主视角、Goal 主视角、action chunk、索引与有效性；模型和 prompt 不变 | 三帧同 episode；Goal 为最后一帧；Local 精确偏移 8 步；无越界伪目标 |
| **V2：建立新模型的纯动作路径** | 新入口继承 aba2 动作条件，去掉 Q-Former、LAM teacher、LaWM decoder 的执行与依赖；当前 DINO 独立加载；保留原两组 query | 不训练时，新入口输出与 V0 在固定随机数下数值一致；checkpoint 保留模块全部正确加载 |
| **V3：新 query 与未来编码** | 未来主视角走共享、冻结的 Qwen Vision；先固定池化为 8 tokens；新增随机 Q_G/Q_L。用独立调试 forward，暂不影响 aba2 动作路径 | `F_G/F_L` 与 `Q_G/Q_L` 都是 `[B,8,2048]`；冻结旧模型时动作输出不变 |
| **V4：过程蒸馏** | 增加 prior/posterior 的中间层对齐，先用分开的 forward 排查；监督第 4、8、12 层；暂不打开 Goal→Local | prior 只看当前输入；两种 horizon 可区分；新 query 获得梯度 |
| **V5：终点监督** | 加入冻结 V-JEPA 主视角 transition target；将固定 future pooling 替换为可训练 R_G/R_L；增加 posterior endpoint loss | target 固定；R_G/R_L 能通过 posterior loss 学习；不存在无梯度的可训练 teacher |
| **V6：统一 packed Qwen** | 当前序列、Q_G/Q_L、F_G/F_L 一次进入同一个 16 层 Qwen；前 4 层独立，第 5–16 层 Goal→Local | packed prior 与去掉 future 的 prior-only 一致；原 query 不读取新 query 或 future |
| **V7：逐个接入动作条件** | 7A 仅新增 Goal 条件；7B 再新增 Local 条件；保留当前 DINO、原动作骨干和基础条件 | 比较 V6/7A/7B；新条件关闭时恢复原动作分支；从此训练和推理均使用预测 prior |
| **V8：单独移除当前 DINO** | 动作基础条件统一为剩余 current VLM memory，比较有/无 DINO；无 DINO 验证通过后移除其依赖 | 避免将 DINO 删除造成的变化混入 Goal/Local 收益；若退化，保留有 DINO 版本作对照 |
| **V9：最终分层 Action DiT** | 新增 4 个 Plan queries；Goal 在逻辑块 1/4/7 更新 Plan，Local 每块更新 Action；Plan/Action 联合自注意力 | 直接路由符合设计，8 个 Cross/Self 主干逻辑块保留；与 V8 作单独比较 |

V7 是普通条件注入的中间对照，允许 Goal 暂时直接影响 Action；V9 才采用最终 Goal→Plan、Local→Action 的严格直接路由。不能把 V7 当成最终结构。

V2 起新模型训练应与它的部署条件一致，基础 loss 仅为 `L_flow`，不再保留旧 latent-action distill/perceptual loss。**V0 的历史两阶段训练不变，但新架构训练不再机械沿用旧两阶段计算图。** V2 若进行纯 `L_flow` 继续训练，要单独记录为“aba2 继续微调对照”，不能仍称其为原始 97.25% 基线。

## 3. V1 的数据与 V3 的模型输入要分开

V1 dataloader 输出：

```text
current_main, current_wrist
local_main, goal_main
instruction, action_chunk, action_mask, action_hz, embodiment_id
episode_id, t, t_local, t_goal, local_valid, goal_valid
```

- `Q_G/Q_L` 是模型的可学习参数，不是 dataset 中每次随机生成的数据；只在模型初始化时随机一次。
- `F_G/F_L` 是未来图像经视觉编码与 resampler 得到的特征，到 V3 才计算；V1 只准备图像和索引。
- 原两段 prompt 由 adapter 按原逻辑构造，action chunk 只送动作训练损失，不进入 Qwen 上下文。
- Goal 仅对已确认成功的示范启用；不能把失败 episode 的最后一帧当成功目标。
- `t_local>=T` 时标为 invalid，不能夹到最后一帧冒充真实 Local。无效目标的 posterior token、辅助 loss 均屏蔽；有效 action loss 仍可保留。
- 不采集未来 wrist；当前 wrist 保留。缓存按 episode、时间索引、编码器版本和预处理建立键，避免 current/future 错配。

两段原始 prompt 保留为：

```text
Your task is {instruction}.

From main-view observations, infer the robot arm's motion.
```

```text
Then use observations and the inferred motion to produce the robot policy actions.
```

## 4. 保留原 query 时，Qwen 的输入和 mask 如何安排

设原 aba2 完整 Qwen 序列为 `C_old`，长度为 `L0`。它包括图像、两段 prompt、原 Q_act/Q_flow、chat template 等，旧序列的顺序不重排：

```text
C_old = [当前 main ; 原 motion prompt ; Q_act×8 ; 当前 wrist ; 原 policy prompt ; Q_flow×8 ; 原模板 token]
```

这是语义示意，实际沿用 tokenizer 模板和原位置索引，不手写替代模板。

在完整旧序列末尾增加新分组：

```text
训练：[C_old ; Q_G×8 ; Q_L×8 ; F_G×8 ; F_L×8]
推理：[C_old ; Q_G×8 ; Q_L×8]
```

| 谁读取 | 第 1–4 层允许读取 | 第 5–16 层允许读取 |
|---|---|---|
| C_old（含 Q_act/Q_flow） | C_old 内原有 causal 可见位置 | 同左 |
| Q_G | C_old、Q_G | 同左 |
| Q_L | C_old、Q_L | C_old、Q_G、Q_L |
| F_G | C_old、F_G | 同左 |
| F_L | C_old、F_L | C_old、F_G、F_L |

表中表示分组可见性；padding、invalid future 另行屏蔽。旧上下文的因果规则保持不变；新 query/future 组内部采用明确的组内可见规则，不能依赖默认 causal mask 代替整张表。

- Prior 与 posterior 不互相读取；C_old 也不读新分组，阻断 future→current→prior 的间接泄漏。
- 新 Q_G/Q_L 可以读取 C_old 中的原 query；原 query 自身仍不直接进入动作 memory。
- V4 分开 prior/posterior forward 只能解决未来隔离；Goal/Local 要独立，还需固定分组 mask 或分别运行两种 horizon，不能默认它们互不可见。
- Qwen 自定义 mask 必须检查实际 attention backend 是否采用；位置编码在 packed/prior-only 中保持一致，保留当前图像的 grid 与多模态注入逻辑。
- 组装动作条件使用显式位置 mask，不用“取最后 8/16 个 token”猜测，也不把完整 packed hidden 直接给动作头。

最终 current memory 只取 `C_old` 去掉 Q_act/Q_flow 的位置：

```text
C_final: [B,L0−16,2048]
C_mem = 原 flow.enc_vlm(C_final): [B,L0−16,768]
```

原 query 对其后 token 的间接影响仍然保留，这是 aba2 的既有语义；本方案不把“删掉直接 query memory”解释为“抹去所有 query 信息”。

## 5. 权重继承与训练安排

首次初始化明确使用文件：

```text
/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt
```

不能误用当前默认训练 YAML 中的 pretrain checkpoint。后续版本优先继承上一版已验证的新 checkpoint；共有模块的初始来源仍是上述 LIBERO SFT 权重，不在每一版把已学到的模块重置回原版。

| 模块 | 初始化与更新建议 |
|---|---|
| Qwen Vision、Merger、保留的 16 层语言模型、text embedding | 从原 LIBERO SFT 加载；前期冻结；学习新 world 分支时逐步解冻语言层 |
| 原 Q_act / Q_flow | 原权重、各 8×2048、原位置；本轮默认冻结参数，不替换为 Q_G/Q_L |
| Action Encoder/Decoder、time embedding、原 flow.enc_vlm、兼容 DiT 参数 | 继承原权重；前期冻结验证，接入新条件后继续微调 |
| 当前 DINO | V2 独立加载与原版相同的视觉权重并冻结，直到 V8 单独移除 |
| Q_G/Q_L、R_G/R_L、endpoint/Goal/Local projection、Plan queries、新增分支/gates | 新初始化；分别检查有效梯度 |
| V-JEPA | 使用独立的预训练权重，全程冻结，仅训练期提供 target |

继承权重不代表以后永久冻结。旧 query 默认冻结是本版减少变量的建议；解冻另做实验。新模型删除旧 LAM 组件后，加载器需明确列出允许缺失/忽略的参数和 DiT 映射，禁止宽松加载时静默丢失保留模块。

建议训练顺序为：先让新 world 分支学会对齐目标，再联合微调语言模型和动作头。不要同时改变图像分辨率、动作 horizon、归一化、状态输入和噪声路径。

## 6. 最终训练流程与维度（V9 无 DINO 主方案）

以下是本方案完成后的目标，尚未实现。`B` 为 batch size，`N_C=L0−16`；文本和模板长度可变，因此不固定 L0。

### 6.1 当前与未来的视觉输入

| 数据流 | 维度变化 |
|---|---|
| 当前 main | `[B,3,256,256] → Qwen Vision [B,256,1024] → Merger [B,64,2048]` |
| 当前 wrist | 同上；两视角合计 `[B,128,2048]`，按原模板插入 C_old |
| Local main | `[B,3,256,256] → 共享 Qwen Vision＋Merger [B,64,2048] → R_L [B,8,2048]` |
| Goal main | `[B,3,256,256] → 共享 Qwen Vision＋Merger [B,64,2048] → R_G [B,8,2048]` |
| 新 prior queries | Q_G、Q_L 参数各 `[8,2048]`，按 batch 扩展为 `[B,8,2048]` |

表中 64/view 对应默认 256 分辨率，实际按 processor 的 `image_grid_thw` 检查，不在通用代码硬编码。Future 只走 Vision＋Merger，不走一次完整语言 forward，也不混入当前图像序列。主方案冻结共享 Vision＋Merger，future 特征 detach；R_G/R_L 保持可训练。

### 6.2 单次 packed Qwen 与两种监督

```text
[C_old ; Q_G ; Q_L ; F_G ; F_L]
[B,L0+32,2048]
    → 16 层 Qwen＋逐层分组 mask
    → [B,L0+32,2048]

第 4、8、12 层：S_G/S_L 与 P_G/P_L，各 [B,8,2048]
第 16 层：prior Z_G/Z_L，各 [B,8,2048]
          posterior P_G_final/P_L_final，各 [B,8,2048]
```

过程监督：`L_proc = Σ_l w_l [D(S_G^l,sg(P_G^l)) + D(S_L^l,sg(P_L^l))]`，按各 horizon 的有效样本归一化。默认 token-wise cosine；Idea 的 query 关系矩阵项可作为可配置附加项，单独验证。

V-JEPA 只读取主视角的两个时间端点：

```text
[current main, local main] 或 [current main, goal main]
→ 按 V-JEPA 预处理得到 [B,3,2,384,384]
→ 冻结 V-JEPA video encoder
→ 预期 [B,576,768]
→ 固定空间分区 pooling（例如 24×24 → 2×4，按固定顺序展平）
→ T_L / T_G: [B,8,768]
```

本地参考权重为 `weights/vjepa2.1-vit-base-384`，配置是 hidden=768、patch=16、tubelet=2。576 是两帧输入下的目标 token 数；实施时需验证实际 video 接口和输出，避免默认 processor 自动采样成配置中的 64 帧。这里使用编码器表征，不用 predictor 的 1664 维 teacher 输出。双帧只是本方案选取的时间端点监督，不宣称包含完整中间轨迹。

先使用固定 pooling，不能使用随机初始化、无有效训练目标的 Q-Former 充当固定标签生成器。若改用 V-JEPA resampler，需先独立校准并冻结。

```text
g_end: 2048 → 768
Z_G/Z_L → [B,8,768]，对齐 sg(T_G/T_L) → L_end
P_G_final/P_L_final → 同一 g_end → [B,8,768]，对齐 sg(T_G/T_L) → L_post
```

endpoint 默认 SmoothL1；Idea 中的 cosine＋relation 可另配对照。`L_proc` 对 posterior stop-gradient，`L_post` 不截断 posterior→R_G/R_L 的梯度；冻结 Qwen 参数时也不能对整个 posterior forward 使用 no_grad，否则 resampler 学不到。只有 future 视觉编码器和 V-JEPA target 使用 no_grad。

### 6.3 动作条件、层级 DiT 与 Flow loss

```text
C_final [B,N_C,2048] → 继承的 flow.enc_vlm → C_mem [B,N_C,768]
Z_G [B,8,2048] → 新 Proj_G → G [B,8,768]
Z_L [B,8,2048] → 新 Proj_L → L [B,8,768]

M_G = [C_mem ; G]: [B,N_C+8,768]
M_L = [C_mem ; L]: [B,N_C+8,768]
```

动作 memory 不含 Q_act/Q_flow 输出、posterior F_G/F_L、latent action 或 LaWM future。V8 之前额外保留 `[B,256,768]` 的当前 DINO；上式是 V8 验证通过后的无 DINO 版本。

```text
真实有效动作 [B,8,7]
→ 原归一化、时间/维度 padding与mask → A [B,50,32]
→ 采样 ε、流时间 τ：x_τ=(1−τ)ε+τA
→ Action Encoder：U [B,50,1024]
→ 新 Plan Queries：P [B,4,1024]
```

原 16 个 Cross/Self 主干层整理为 8 个逻辑块，每块执行：

1. 在逻辑块 `{1,4,7}`：Plan 对 `M_G` 做 cross-attention；其他块不直接读取 Goal memory。
2. 每个逻辑块：Action 对 `M_L` 做 cross-attention。
3. 联合 self-attention：`[P;U] [B,54,1024] → [B,54,1024]`，使 Plan 与 Action 交互；动作 padding 始终屏蔽。

尽量复用原 Action cross/self-attention、FFN、time conditioning 和输出层权重，新增 Goal→Plan 的 3 个 cross 分支。因此“保留 16 层主干”不等于参数量与算量完全不变。联合注意力中 4 个 Plan token 有效，50 个动作槽仅当前有效动作参与。

新分支使用残差门控。**仅把新 token 乘以零再 concat，不保证保持原输出**，因为它仍改变 attention softmax 分母；V7 应对独立 attention 输出门控，或在关闭时真正 mask 掉新增 K/V。Plan 引入也要单独验证，不能声称零初始化 Plan/gate 就自动等价于原 self-attention。

```text
取最终 U [B,50,1024]
→ 原 Action Decoder → 预测速度 [B,50,32]
→ 与 A−ε 计算有效位置上的 L_flow

L_total = L_flow + λ_proc L_proc + λ_end L_end + λ_post L_post
```

可选 query variance/relationship regularization 独立配置，不与首轮结构变化捆绑。旧 `L_distill/L_perceptual` 不再计算。延续 `repeated_diffusion_steps=2` 时，仅 Flow 分支把 B 扩为 2B、独立采样噪声和 τ；Qwen/world supervision 不为此重复 forward。

## 7. 最终推理流程与维度

```text
当前主视角＋当前腕部图像＋任务指令
→ 原两段 prompt、原 Q_act×8、原 Q_flow×8 构成 C_old
→ 追加 Q_G×8、Q_L×8
→ Qwen 输入/输出 [B,L0+16,2048]
→ 取 C_final [B,L0−16,2048]、Z_G/Z_L 各 [B,8,2048]
→ 投影，构造 M_G/M_L 各 [B,L0−8,768]

随机动作噪声 [B,50,32]
→ 每个 ODE 步：Action Encoder [B,50,1024]
                初始化本次速度评估的 Plan [B,4,1024]
                8 个分层逻辑块，内部联合序列 [B,54,1024]
                Action Decoder → 速度 [B,50,32]
→ 10 次 Euler 更新 x ← x + Δτ·v
→ 取 8 个有效动作 [B,8,32]
→ 取 LIBERO 7 维、反归一化/夹爪处理 [B,8,7]
→ 执行 action chunk，再读取当前观测循环
```

Qwen 和条件 memory 每个 action chunk 只计算一次，在 10 个 ODE 步中复用。Plan 在每次速度网络调用内从可学习初始 query 展开，不额外引入跨 ODE 步或跨控制周期的隐藏状态。Goal“稀疏”指动作逻辑块中的注入频率，不代表降低环境观测或重规划频率。

最终部署保留：Qwen（含原 Q_act/Q_flow 和新 Q_G/Q_L）、current/Goal/Local 投影、Plan queries、分层 Action DiT 与原动作 I/O、时间编码。

最终部署不读取：未来图像、F_G/F_L、posterior、R_G/R_L、V-JEPA、LAM teacher、LaWM decoder、VLMToLAMQFormer；无 DINO 方案验收后也不加载 DINO。训练专用模块应可从推理图/导出权重中剥离。

## 8. 文件边界和每版检查

建议新增，避免改变原 aba2 入口行为：

```text
starVLA/model/framework/goal_local_lawam_framework.py
starVLA/model/framework/goal_local/       # 数据契约、future encoder、mask、loss、运行组件
starVLA/model/framework/vlas/goal_local_lawam.py
starVLA/model/framework/vlas/hierarchical_action_dit.py
starVLA/dataloader/goal_local_train_collator.py
starVLA/config/training/train_goal_local_libero_v*.yaml
```

复用原 dataloader 的 action normalization 与采样规则，只增加按 episode 精确读取主视角时间索引的 helper。每版配置和新 checkpoint 独立保存。

`results/Checkpoints` 是指向原权重库的软链接；新的 run root 使用本目录下独立的 `results/goal_local/`，不覆盖原 LIBERO 权重或已完成的 aba2 评测。

每版至少检查：数据对齐、保留权重加载清单、各 loss 的梯度去向、动作与条件维度。涉及 mask 的版本必须固定 current、置换 future，验证 prior/current/旧 queries/固定噪声下的动作不变；深层扰动 Goal 可影响 Local，反向不应影响。最终增加 Goal→Plan、Local→Action 路由与无训练期组件的部署检查。

关键版本记录 LIBERO 成功率、每个 action chunk 的延迟、显存和参数量。当前主目标是 LIBERO；不自动扩大到 RoboTwin，也不自动启动下一版。固定噪声用于计算图一致性检查，完整成功率比较沿用 40 任务×50 回合协议，报告独立运行的随机波动。

参考文档：
- `/mnt/hdd/wangchaoyang/idea/idea_9_9_11.01_new.md`：方法核心。
- `/mnt/hdd/wangchaoyang/idea/change(4).md`：Qwen/LaWAM 工程适配。
- 优先级：本轮用户明确要求 > 本方案中的新约定 > 旧文档。
