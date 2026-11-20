# aba2 → 两阶段 Goal/Local 特权未来蒸馏：架构与实施计划

> 依据：`/mnt/hdd/wangchaoyang/idea/idea_9_20_change.md`，按用户本轮要求适配 aba2。
> 实施目录：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change`。
> 实施状态（2026-09-20）：已按本计划新增 `goal_local/` 独立训练与推理实现；详细代码入口、维度和验证状态见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。原文以下保留设计与验收依据；验证用短步更新不等于正式训练或 LIBERO 成功率评测。

## 1. 最终要得到什么模型

**先训练使用真实未来的后验教师，再冻结完整教师，训练只使用当前观测的学生。部署仅保留学生。**

- Local：当前 action chunk 结束时的结果，来自 `o_(t+H)` 的主视角＋腕部视角。
- Goal：较远时间范围内的任务进程，来自主视角的 `o_(t+3H), o_(t+5H), o_(t+7H), o_(t+9H), o_(t+11H)`。
- 教师在 Qwen 内始终隔离 Goal/Local；学生仅在第 9–16 层通过显式 relevance gate 将 Goal 信息注入 Local。
- 动作头保留当前 VLM＋当前 DINO 条件，另外以 Goal→Plan、Local→Action 两条独立路径注入新增表征。

**这里有新的两阶段训练，但不需要重跑旧 LaWAM 的 Stage I。** 新 Stage I 是“后验教师的动作监督训练”，新 Stage II 是“冻结教师指导下的学生蒸馏＋动作训练”；两者都不同于原 LaWAM 的 LAM/LaWM 两阶段。旧 LIBERO SFT 权重是新 Stage I 的初始化，不能直接充当已经学会新 Goal/Local 职责的教师。

### 1.1 从 aba2 保留、移除、增加什么

| 部分 | 本轮处理 |
|---|---|
| Qwen Vision＋Merger | 保留原 SFT 权重；新两阶段均冻结 |
| Qwen 前 16 层语言模型，2048 维 | 保留结构与初始权重；新 Stage I 和 Student 中训练 |
| 原 Q_act、Q_flow | 各 8 个、原位置、原权重初始化；继续参与 Qwen；原 16 个输出位置始终不直接送动作头 |
| 原两段 prompt、当前 main/wrist | 原样保留，不换成新的 world prompt |
| 当前主视角 DINO | 保留并冻结，仍输出 `[B,256,768]`；本轮不删除 |
| 原 Flow 数学路径、动作归一化、时间编码、动作编解码器 | 保留；动作 buffer 50×32，有效 chunk 8×7，部署 10 次积分 |
| 旧 latent-action Q-Former、LAM teacher、LaWM decoder | 从新模型执行图和加载依赖中删除；不计算 `[B,1,32]` latent action |
| 旧 latent-action distill/perceptual loss | 删除，不混入新 Stage I/II |
| 本体状态 | 不作为模型输入，不建 State MLP/state token，不提供启用选项；底层若暂需占位字段，也不允许形成计算路径 |
| 新增训练教师部分 | Local/Goal Future Resampler、view/horizon embedding、教师角色隔离 mask |
| 新增学生部分 | Q_G/Q_L 各 8 个、独立 Student、深层 relevance head＋Goal→Local adapter、逐层蒸馏 |
| 新增动作部分 | 4 个 Plan queries、Goal/Local 的 2048→1024 投影、角色 cross-attention |

“保留旧 query”指保留其数量、位置、初始化和 current-only 职责，不等于永久冻结参数。按新 idea 的训练日程，Stage I 和 Student 中旧 query 可训练；Stage-II Teacher 的两组旧 query 副本必须冻结。

### 1.2 新文档覆盖旧 change.md 的地方

不再使用 episode 最后一帧 Goal；不再使用单次 `[C;Q_G;Q_L;F_G;F_L]` 师生 packed forward；不使用 V-JEPA endpoint/anchor loss、固定 pooling anchor、posterior endpoint loss。本轮 Qwen future encoder 为主方案，V-JEPA 仅保留为后续独立 encoder 消融，不进入当前实施链。

Goal/Local 的动作投影为 **2048→1024**；768 维只属于保留的 current base memory。Goal 注入使用 **DiT 物理层 1/5/9/13**，不能沿用旧方案的逻辑块 1/4/7。

## 2. 第一步：构造正确的训练数据

数据源只读：

```text
/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero_merged_no_noops_20hz
```

已核对元信息：LeRobot v3、1693 个 episode、273465 帧、40 个任务；两个相机均 256×256、20 Hz；动作 7 维。数据中虽然有 `observation.state`，本模型不使用。抽查起始数据的 frame_index 和 timestamp 为逐帧递增、0.05 秒间隔；实施 V1 时还需做跨 episode 的完整一致性检查及真实视频解码抽查。

### 2.1 时间与样本契约

```text
H = floor(horizon_sec × action_hz) = floor(0.4 × 20) = 8
Local 索引：t+8
Goal 索引：[t+24, t+40, t+56, t+72, t+88]
```

时间约为 0.4 秒和 `[1.2,2.0,2.8,3.6,4.4]` 秒，均按转换后数据集的控制步解释，不据此推断被删除 no-op 前的原始时间。

设 episode 有 T 条观测、合法索引为 `0…T−1`：

```text
m_L = (t+8 < T)
m_G[j] = (t+n_j×8 < T), n=[3,5,7,9,11]
u_L = m_L
u_G = any(m_G)
```

按 batch 组织的 raw 样本：

| 字段 | 形状/含义 |
|---|---|
| current_main / current_wrist | 各 `[B,3,256,256]` |
| local_main / local_wrist | 各 `[B,3,256,256]`，同一个 t+8 |
| goal_main | `[B,5,3,256,256]`，保留上述 horizon 顺序 |
| local_valid / goal_frame_valid | `[B]` / `[B,5]` |
| action_chunk | `[B,8,7]`，尾部不足部分单独标记有效性 |
| actions / actions_mask | 经原归一化和维度 packing 后，各 `[B,50,32]` |
| instruction | 任务文本，由原 adapter 构造两段 prompt |
| episode_id、t、future_indices、action_hz、embodiment_id | 时间对齐、诊断与既有动作接口所需字段 |

不包含未来 action token 输入、不包含最后一帧额外 Goal、不包含任何未来 wrist Goal。Student 的 Q_G/Q_L 是 Stage-II 模型参数，不是 dataloader 每个样本重新随机生成的字段。

### 2.2 尾部 mask 必须贯穿三个位置

1. **编码前与 Resampler：** 无效帧不参加真实未来编码或聚合。Goal 全无效时跳过该样本 Goal Resampler，用有限确定占位填满 8 个位置，避免全掩码 softmax；Local 同理。不能用末帧重复冒充真实未来。
2. **Stage I 动作：** 缺失 Goal/Local 时，关闭整条对应角色注入残差，包括 projection bias 的贡献；保留当前条件和仍有效的动作监督。
3. **Stage II：** 缺失角色只关闭相应教师蒸馏项；Student 的 Q_G/Q_L、gate 和动作条件照常启用。部署不知道未来有效性，Student 前向不得读取 m_L/m_G 决定是否工作。

五维 `m_G` 是帧 mask，不是八个 Goal latent slot 的 mask。先屏蔽无效帧，再按 `u_G` 决定整体 Goal 蒸馏是否有效；不存在五帧与八个 slot 的一一对应。

基于 episode 长度统计，若全部帧都作为候选 anchor，各未来位置的存在比例为：t+8 约 95.05%；t+24/40/56/72/88 约 85.14%/75.24%/65.33%/55.43%/45.61%。这只是索引层面的统计，不代表已完成视频有效性验证。不能为了避免 mask，默认丢弃所有缺少五帧远期目标的样本。

### 2.3 原 dataloader 的两个具体适配点

原 `gr00t_lerobot/datasets.py::get_video()` 会对 wrist 执行 `delta[:1]`，并对越界 step 使用 `np.clip`。新数据路径新增按 **episode 内绝对 step** 取帧的 helper，不改变原 aba2 的 helper 行为。

LeRobot v3 的多个 episode 可共用一个 video 文件；必须使用 episode metadata 的 `dataset_from_index/to_index` 与各相机 `from_timestamp`，结合样本 timestamp 解码，不能把 episode step 当整段视频的绝对帧号。main/wrist 使用同一 episode step，各自应用正确视频偏移。

动作尾部也要保留原始索引有效性，不能仅根据 padding 后数组长度推断有效动作。新路径对不存在的动作置有限 padding 并设 mask=false。旧 Flow 会断言动作有效数等于完整 0.4 秒窗口；新路径应按真实 episode 剩余动作数与频率窗口的交集逐样本校验，不能照搬该总数断言或直接删除所有检查。归一化、动作含义及流时间网格不变，尾部 mask 修正单独记入实验配置，继续训练的 baseline 对照也使用相同规则。

## 3. 最终当前上下文：适配两组旧 query、无 state

令 L 为 aba2 原完整 Qwen 序列长度，包含图像、文本、模板、Q_act×8、Q_flow×8；令 `N_C=L−16`。L 随指令和 padding 变化，不固定成某个数字。

原序列沿用实际 tokenizer 模板，不重排已有位置：

```text
C = [current main ; 原 motion prompt ; Q_act×8 ; current wrist ; 原 policy prompt ; Q_flow×8 ; 原模板 token]
C: [B,L,2048]
```

保留两段原始 prompt：

```text
Your task is {instruction}.

From main-view observations, infer the robot arm's motion.
```

```text
Then use observations and the inferred motion to produce the robot policy actions.
```

默认 256×256 图像：

```text
每个当前视角：[B,3,256,256]
→ Qwen Vision [B,256,1024]
→ Spatial Merger [B,64,2048]
双视角合计 128 个 visual tokens，按原模板分布于 C 中。
```

两个新角色都可以读取 C（含旧 query），但 C 自己只保留原 causal 可见性，不读取新角色。旧 query 对其他 current token 的间接影响保留；排除的是它们对动作头的直接输出条件。

## 4. 真实未来编码：Local 双视角，Goal 有序五帧主视角

### 4.1 Local

```text
local main + local wrist
→ 冻结的 Qwen Vision＋Merger，分别编码
→ 两组 [B,64,2048]，加可区分的 view identity
→ F_L_raw [B,128,2048]
→ 可训练 R_L（内部 8 个压缩 query）
→ F_L^0 [B,8,2048]
```

### 4.2 Goal：首版建议采用逐帧 Qwen＋时序 Resampler

新 idea 允许 Qwen 视觉/视频编码。为了避免无效帧在视频 temporal patch 中污染有效帧，首版实施建议显式选择 `framewise_qwen_temporal_resampler`：有效远期图像分别走冻结 Qwen 视觉编码，再由带 horizon identity 的 Goal Resampler 聚合。它仍是有序多帧 Goal，不是无序图像均值。

```text
goal main [B,5,3,256,256]
→ 只编码有效图像，回填固定位置
→ [B,5,64,2048]
→ 每帧加入对应 E_(3H/5H/7H/9H/11H)
→ 展平 [B,320,2048]，帧 mask 广播为 token mask
→ 可训练 Temporal R_G（内部 8 个压缩 query）
→ F_G^0 [B,8,2048]
```

64/view、128、320 是默认分辨率下该首版实现的维度；通用代码读取实际 grid。horizon identity 不能因有效帧压缩而从 3H/5H/…变成简单的第 1/2/…帧。

若后续选用真正的 Qwen video processor，改为 `[B,N_G,2048]→[B,8,2048]`，N_G 由 temporal patch、帧补齐和 grid 决定，不能继续假定 320。必须保留真实时间间隔、处理奇数有效帧以及 processor 重复帧的来源映射；作为显式 encoder 实现对照记录，不在同一实验中悄悄切换。

R_G/R_L 内部压缩 query 在 Stage I 学习并随 Teacher 冻结，和 Stage II 新增的 Student Q_G/Q_L 是不同参数。

## 5. 新 Stage I：训练任务驱动的后验教师

### 5.1 一次教师 VLM 前向

```text
当前双视角＋两段 prompt＋旧 Q_act/Q_flow → C^I [B,L,2048]
Local 真实未来 → R_L → F_L^0 [B,8,2048]
Goal 真实未来 → R_G → F_G^0 [B,8,2048]
                         ↓
             [C^I ; F_G^0 ; F_L^0]
                 [B,L+16,2048]
                         ↓ 16 层 Qwen，角色隔离
 H_C^I [B,L,2048]；P_G/P_L 各 [B,8,2048]
                         ↓
              后验条件化 Action DiT
                         ↓
            预测速度 [B,50,32] → L_I
```

Stage I 不输入 Student Q_G/Q_L，不包含 relevance adapter，不计算逐层蒸馏。

全部 16 层的教师 mask：

| 读取者 | C | Goal 角色 | Local 角色 |
|---|---|---|---|
| C | 原 causal＋padding | 禁止 | 禁止 |
| Goal | 允许 | 组内双向 | 禁止 |
| Local | 允许 | 禁止 | 组内双向 |

该 mask 同样用于 Stage-II 冻结 Teacher；教师两角色始终不互读。动作侧的 Plan/Action 融合不会反向改写同一次已生成的 VLM hidden。

### 5.2 损失与梯度

```text
L_I = masked_flow_loss(v_post, A_pad−ε)
梯度：Action DiT → Goal/Local 投影 → P_G/P_L → Qwen → R_G/R_L
```

训练 Qwen 语言骨干、文本嵌入、旧两组 query、Future Resampler、view/horizon embedding、当前条件投影、Plan 与动作模块。冻结 Qwen Vision＋Merger、DINO。不得沿用原 YAML 的旧 LAM loss 或旧 freeze 规则来隐式决定新阶段。

新增角色注入的残差 scale 用可配置的小非零初始化，避免 Stage I 初期两条 future 路径都被固定零门控截断。具体数值经小样本梯度与稳定性检查确定，不预设最优值。

## 6. 两阶段共用的动作模块

### 6.1 current base 与角色条件分开

Stage I 使用教师后验，Stage II/部署使用学生先验，但都先过滤旧 query：

```text
H_C [B,L,2048]
→ 删除 Q_act×8、Q_flow×8 的输出位置并同步过滤 mask
→ [B,L−16,2048]
→ 继承的 flow.enc_vlm：2048→768
→ C_Q [B,L−16,768]

current main → 冻结 DINO → D_t [B,256,768]
C_base = [D_t ; C_Q]：[B,L+240,768]
```

不能照抄参考文档的 `[B,L+256,768]`：其上下文包含的旧 query 在 aba2 中不直接送动作头。

角色投影独立于 current base：

```text
Stage I：U_G=P_G，U_L=P_L
Stage II/部署：U_G=Z_G，U_L=Z_L

U_G [B,8,2048] → Proj_G → G_A [B,8,1024]
U_L [B,8,2048] → Proj_L → L_A [B,8,1024]
```

**G_A/L_A 不与 768 维 C_base 直接 concat**，它们由新增 1024 维 K/V 的角色 cross-attention 读取。旧 base attention 仍使用 768 维 K/V。

### 6.2 带噪动作和 Plan

```text
有效 action chunk [B,8,7]
→ 原归一化、padding、真实有效性 mask
→ A_pad/M_A [B,50,32]

X_τ=(1−τ)ε+τA_pad，ε~N(0,I)
→ 原 Action Encoder → U [B,50,1024]
新增 Plan 参数 [4,1024] → P [B,4,1024]
联合序列 [P;U]：[B,54,1024]
```

参考 idea 将 τ 写为 Uniform，但 aba2 实际使用 `τ=(1−Beta(1.5,1.0))×0.999`。本适配版保留 aba2 采样，避免同时改动 Flow 训练分布；Uniform 仅能另作消融。流插值公式、速度目标、物理时间编码和部署积分保持原样。

### 6.3 物理层编号与 block 顺序的适配

aba2 的 16 个物理层是 Cross/Self 交替，不是 16 个各自都含完整 Cross＋Self 的逻辑块。建议把角色残差放在每个物理层原 block 之前，然后执行原 block 的 attention/FFN/time conditioning：

| 物理层 k（从 1 开始） | 原有操作 | 新增操作 |
|---|---|---|
| 1、5、9、13 | 读取剩余 VLM 的 base cross-attention | Goal→Plan、Local→Action |
| 3、7、11、15 | 读取当前 DINO 的 base cross-attention | Local→Action |
| 2、4、6、8、10、12、14、16 | self-attention | Local→Action；原 self 改为 Plan/Action 联合 self |

原 base cross-attention 的流选择与投影保留，序列扩展为 `[P;U]`，沿用相同 current 条件给新增 Plan 位置；Plan/Action 融合发生在原有的 8 个 self 层。**不额外给全部 16 层都增加一次 self-attention，也不把 Goal 稀疏层改成旧方案的 1/4/7。** 这是针对 aba2 的具体 block 解释，实施前后需用 hook 核验。

新增角色残差：

```text
k∈{1,5,9,13}：P ← P + ρ_G γ_G^k CrossAttn(P, G_A, G_A)
每个 k=1…16：U ← U + ρ_L γ_L^k CrossAttn(U, L_A, L_A)
原 self 层：[P;U] [B,54,1024] → [B,54,1024]
```

Stage I：`ρ_G=u_G, ρ_L=u_L`；Stage II/部署：二者恒为 1。ρ 是数据有效性开关，γ 是残差 scale，不新增动作侧 relevance gate。

Plan 的 4 个位置有效；动作位置按真实时间 mask。Plan 不具有真实动作时间，不能错用 50 个动作的 physical-time embedding；为其补齐独立的中性位置输入，沿用同一次 flow timestep 条件。最后只取 U：

```text
[B,54,1024] → 丢弃 Plan 输出 → [B,50,1024]
→ 原 Action Decoder → v_hat [B,50,32]
```

保留 16 层主干不意味着参数或计算量不变：新增 4 个 Goal cross 分支、16 个 Local cross 分支和 Plan token。新增模块关闭时的 baseline 回归应真实 bypass Plan/角色分支；不能仅把新 token 置零就声称输出严格不变。

## 7. 新 Stage II：完整冻结 Teacher，独立训练 Student

### 7.1 交接与两次独立 forward

完成 Stage I 后保存完整 snapshot，从它建立：

- Teacher：未来 encoder、Resampler、view/horizon embedding、Qwen 语言骨干、文本嵌入、两组旧 query、相关 norm/state 全冻结；eval、no_grad、不加入 optimizer、不做 EMA。
- Student：独立复制 Stage-I 当前编码模块、Qwen、旧 query、动作条件投影、Plan 和动作头；新增随机 Q_G/Q_L 和 relevance adapter。可训练参数不得与 Teacher 共享存储。

```text
Teacher：当前＋真实未来
→ [C^T ; F_G ; F_L] [B,L+16,2048]
→ 冻结的 16 层 Qwen
→ 第 4/8/12/16 层 P_G^ell、P_L^ell，各 [B,8,2048]
→ 仅作为蒸馏 target，不运行 Teacher Action DiT

Student：当前信息
→ [C^S ; Q_G ; Q_L] [B,L+16,2048]
→ 可训练 16 层 Qwen＋深层 adapter
→ 第 4/8/12/16 层 Q_G^ell、Q_L^ell，各 [B,8,2048]
→ 最终 Z_G/Z_L 与学生 H_C^S
→ 学生动作头 → v_pri [B,50,32]
```

Stage II 有两次独立 VLM forward，均为 L+16，不是一次 L+32。Teacher 无反向激活图，但仍占模型内存与前向时间。可以共享真正冻结的视觉前端和同预处理的 current-only 视觉特征，不能复用学生生成的可训练 context tensor、教师/学生的融合 hidden 或 KV cache。

### 7.2 Student-only Goal→Local relevance gate

Student 原生 mask 全部 16 层均与教师的三分区隔离结构相同，Goal/Local 不直接通过 self-attention 互读。第 1–8 层无跨角色模块；第 9–16 层在本层 Qwen block 完成后执行：

```text
Q_G_base、Q_L_base：各 [B,8,2048]
→ LN＋relevance projection → R_G、R_L：各 [B,8,256]
NULL key [1,256]＋R_G → K_rel [B,9,256]
A_rel = softmax(R_L K_rel^T / sqrt(256))：[B,8,9]
g = 1−A_rel[:,:,NULL]：[B,8,1]

message = CrossAttn(Q_L_base, Q_G_base, Q_G_base)：[B,8,2048]
Q_L = Q_L_base + η_GL^ell × g × message
Q_G = Q_G_base
```

只有 Local 更新，更新结果送下一层。第 9–16 层共享 relevance head；adapter/残差 scale 的参数映射显式记录，先沿用逐层残差接口。η 使用可配置小残差初始化。NULL 允许选择不注入 Goal 信息，g 不是人工标注的相关性标签或校准概率。

Local 蒸馏和学生动作 loss 训练 gate；不增加 gate label、反事实双前向 loss。Local loss 可以沿该路径更新 Goal 参数，不代表开放 Local→Goal 的前向信息流。

### 7.3 蒸馏及总损失

监督层为 `{4,8,12,16}`，各层默认权重 1/4。对应角色按 slot 对齐：

```text
S/T：[B,8,2048]
归一化后 token-wise cosine distance
＋ α_rel × Gram distance（每个样本的关系矩阵 [8,8]，除以 8²）
```

Teacher 和 Student 取相同的 hidden-state 接口。深层 Student Local 使用 adapter 更新后的结果；最后一层 final norm 的前后位置必须匹配，不能盲目读取默认 hidden_states 列表。

```text
L_local_dist = 四层距离，按 m_L 的有效样本归一化
L_goal_dist  = 四层距离，按 any(m_G) 的有效样本归一化
L_II = L_action_pri + λ_L L_local_dist + λ_G L_goal_dist
```

无有效目标的角色直接返回有限零损失，不先计算 NaN 再乘零。λ_L/λ_G、α_rel 是需要验证的超参数，不预设最优值。

Stage II 的学生动作头只使用学生自己的 current base、Z_G、Z_L；无 posterior 替换、无教师 hidden 注入、无未来条件混合。只反向更新 Student 和其动作模块。

若保留 aba2 的 `repeated_diffusion_steps=2`，两阶段动作分支均可将条件和动作扩为 2B，独立采样噪声/流时间；Teacher 目标与 Qwen 表征每个原 batch 计算一次，蒸馏仍按 B 归一化。

## 8. 最终部署流程与维度

```text
current main/wrist：[B,3,256,256] 各一张，任务指令
→ 原两段 prompt＋原 Q_act×8＋原 Q_flow×8
→ C^S [B,L,2048]
→ 追加 Student Q_G×8、Q_L×8
→ [B,L+16,2048]
→ Student Qwen 16 层（1–8 隔离；9–16 隔离＋门控 adapter）
→ H_C^S [B,L,2048]、Z_G/Z_L 各 [B,8,2048]

H_C^S 删除旧 16 个 query 输出 → [B,L−16,2048] → [B,L−16,768]
current main → 冻结 DINO [B,256,768]
→ C_base [B,L+240,768]
Z_G/Z_L 分别投影 → G_A/L_A 各 [B,8,1024]

噪声 x [B,50,32]
→ 每个 ODE 步：Action Encoder [B,50,1024]
               Plan [B,4,1024]
               16 个物理 DiT 层，内部 [B,54,1024]
               取动作 hidden [B,50,1024]
               Action Decoder → 速度 [B,50,32]
               Euler 更新 x
→ 10 步后 [B,50,32]
→ 取前 8 个有效时间位置 [B,8,32]
→ 取 LIBERO 7 维、原反归一化/夹爪处理 [B,8,7]
→ 执行 action chunk，获取新观测，循环
```

Student Qwen、relevance adapter、当前 DINO 和条件投影每个 action chunk 计算一次，在 10 个 ODE 步中复用。Plan 在每次速度网络调用内重新从可学习参数展开，不引入跨 ODE 步或跨 chunk 的额外 recurrent state。

部署保留一份 Student、当前 Qwen 视觉前端、当前 DINO、旧两组 query、新两组 prior query、gate/adapter、动作模块。删除 Teacher、future Resampler、真实未来输入和蒸馏。无本体状态、旧 LAM/LaWM/Q-Former，也不需要 V-JEPA。

## 9. 权重来源与冻结表

新 Stage I 的共有模块从以下文件初始化：

```text
/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt
```

不能使用 aba2 默认 YAML 中的 pretrain 初始化路径代替它。权重继承需逐模块显式报告 loaded/missing/unexpected/shape mismatch；被删除的 LAM 路径和新参数列入白名单，保留模块不允许静默漏载。

| 模块 | 新 Stage I | Stage-II Teacher | Stage-II Student / 部署 |
|---|---|---|---|
| Qwen Vision＋Merger | 原 SFT 初始化，冻结 | 冻结 | 冻结 |
| 文本 embedding、16 层语言骨干、旧 Q_act/Q_flow | 原 SFT 初始化，训练 | Stage-I 副本完整冻结 | Stage-I 独立副本训练，部署保留 |
| DINO | 提取与 aba2 相同的视觉权重，独立冻结加载 | 目标链不需要 | 冻结保留 |
| Future R_G/R_L、view/horizon embedding | 新初始化并训练 | Stage-I 副本冻结 | 不存在 |
| 新 Q_G/Q_L、relevance head/adapter | 不存在 | 不存在 | 随机初始化并训练，部署保留 |
| 原 current 投影、动作编解码、time、兼容 DiT 参数 | 继承 SFT 并训练 | 不执行 | 复制 Stage I 并继续训练 |
| Goal/Local 角色投影、Plan、角色 cross 分支 | 新初始化并训练 | 不执行 | 复制 Stage I 并继续训练 |

独立的 Student 部署 checkpoint 必须包含所需视觉/动作参数及 processor、位置规则、动作统计信息，加载时不依赖 Teacher 文件或未来数据字段。

## 10. 从 aba2 一步步实施：先数据，再教师，再学生

“版本”表示工程增量；Stage I/II 表示训练阶段，二者不要混用。每版验证通过后再推进，不自动越级开始大规模训练。

| 工程版本 | 修改内容 | 进入下一步前的验收 |
|---|---|---|
| **V0：记录基线** | 只读记录 aba2 代码/配置/权重/97.25% 结果；以后所有新增文件位于 aba2_change | 原 aba2 不变；运行与 checkpoint 输出路径隔离 |
| **V1：数据契约** | 新 dataset/collator/helper；读取 current、Local 双视角、Goal 五帧主视角；生成三类 mask 和索引 | 随机与边界 episode 可视化；不跨 episode、不夹取伪 future；wrist 时间正确；缺失目标安全 |
| **V2：可训练的 aba2 条件基线** | 在新入口去除旧 Q-Former/LAM teacher/LaWM 依赖；当前 DINO 单独加载；训练和推理都只用 aba2 条件；不输入 state | 固定输入/噪声下、不更新参数时与 aba2 数值一致；共有权重加载完整；训练前向已无旧 loss |
| **V3：未来编码与教师 Qwen** | R_L/R_G、view/horizon identity；教师三分区 mask；读取各层后验 hidden；暂不启动正式训练 | 输出尺寸正确；Goal/Local 隔离；无效未来不污染有效目标；当前输出在冻结测试中不受未来改变 |
| **V4：完整后验动作路径** | Plan、角色投影、Goal 稀疏注入、Local 全物理层注入；适配 base mask、physical-time 与动作 padding | 角色路由 hook、masked loss、小批次有效梯度与无 NaN；Stage I 不含 Student query/adapter |
| **V5：训练新 Stage I** | 从原 LIBERO SFT 权重训练完整后验教师；按 episode 划分固定验证集，记录目标有效率 | 验证动作误差、Resampler 梯度、正常/缺失/置乱 future 对照；不能只看总 loss 下降 |
| **V6：冻结交接** | 保存完整 Stage-I snapshot；构造冻结 Teacher 与独立 Student；复制动作头，不执行 Teacher 动作头 | Teacher 参数/状态冻结；无可训练共享存储；同输入在 Student 更新前后教师 target 稳定 |
| **V7：Student 与蒸馏** | 新 Q_G/Q_L；先验证无 adapter 的蒸馏链，再接第 9–16 层 relevance adapter；加入 current-only 动作 loss | 层/slot/final norm 对位；未来置换不改变 Student 输出；单向 gate 拓扑和梯度正常 |
| **V8：训练新 Stage II** | 固定完整 Teacher，联合训练 Student 蒸馏和动作；记录角色 loss、gate、资源、闭环小评测 | 教师保持固定；Student 与部署前向一致；future validity 不控制 Student 注入 |
| **V9：导出与完整对比** | 导出不依赖 Teacher 的 Student；LIBERO 40×50，输出 suite/task/overall 对比 | 完整性、错误数、成功率、chunk latency、显存；保留 aba2 和匹配预算继续微调对照 |

V2 的纯动作继续训练不是本轮直接启动的工作；若作为控制组运行，其数据、尾部 mask、训练预算须与新模型可比。aba2 的现有 97.25% 是未经新增训练的消融成绩，不能把它单独当作排除“额外训练收益”的全部证据。

第一轮固定验证划分按 episode（并尽量覆盖各任务），不让同 episode 的相邻帧落入训练和验证两侧。不要无条件继承旧配置 `train_split_all=true` 来报告独立验证误差。

## 11. 建议新增文件与实验记录

后续获准实施时，在 aba2_change 内建立独立代码副本和入口，原 aba2 只读：

```text
starVLA/model/framework/goal_local_distill_framework.py
starVLA/model/framework/goal_local_distill/
    types.py / config_builder.py
    future_encoder.py / future_resampler.py
    masked_qwen.py / relevance_adapter.py
    teacher_student.py / losses.py / checkpoint.py
starVLA/model/framework/vlas/role_conditioned_flow.py
starVLA/dataloader/goal_local_distill_collator.py
starVLA/dataloader/goal_local_episode_reader.py
starVLA/config/training/goal_local_stage1_libero.yaml
starVLA/config/training/goal_local_stage2_libero.yaml
runtime/goal_local/                     # 数据诊断、阶段验收、后续评测
results/goal_local/                    # 所有新 checkpoint、日志、报告
```

保留旧 `LaWAM` registry，新模型单独注册。两阶段由明确的 stage 配置及独立模型构建函数控制，不能通过“传空 future”猜测当前阶段。

源数据和原 CKPT 均只读引用；不要把新 run_root 指向继承的 `results/Checkpoints` 软链接而写回原权重库。新 snapshot 记录原初始化来源、数据划分、horizon、mask、encoder 模式、Qwen 层数、hidden 接口、两组旧 query 的过滤位置、DiT 注入物理层、噪声采样和动作统计。

## 12. 必须验证的关键点

- **数据：** 索引/视频 timestamp 对齐；Local future wrist 确实为 t+8；Goal 有序；末尾全无效/部分无效有限且不串 episode；动作 mask 与真实可用动作一致。
- **教师隔离：** 固定 current 和 Local，只换 Goal future，P_L 不变；反向同理。C 不读取未来。
- **目标链：** 完整冻结 Teacher 的参数和状态；不只 detach 一个会随 Student 更新的上下文 tensor。
- **学生无泄漏：** 固定 current、噪声、流时间，改变真实 future 或其有效性，Student Z_G/Z_L/动作不变，蒸馏 loss 可以改变。
- **门控与层接口：** 9–16 层 adapter 更新之后取 Local 蒸馏值；Goal 不读 Local；NULL、relevance 分布、γ/η 及梯度有记录。
- **动作路由：** 旧 Q_act/Q_flow 输出始终被过滤；Goal 仅直接注入 Plan，Local 仅直接注入 Action；base stream 的 VLM/DINO 层选择仍为原规则。
- **梯度：** Stage I 两个 Resampler 都能从动作 loss 得到梯度；Stage II Teacher 无更新、Student query/adapter/动作模块可更新；新增路径未被旧 detach 开关截断。
- **未来是否被利用：** 对教师分别屏蔽 Local/Goal、置换 future、固定 horizon 标签打乱 Goal 帧内容，检查动作误差与表示变化。共享当前条件很强，不能仅凭动作 loss 下降宣称学会长期规划。
- **部署：** 无 Teacher、future、Resampler 文件也能加载 Student；同 current 和同噪声时与 Stage-II Student 输出一致；报告真实训练开销与单 Student 部署开销。

本方案的第一项实际实施应为 **V1 数据契约与采样验证**，随后才进入新模型清理和教师路径。新 idea 的两阶段学习与模块增删按以上顺序推进，不把旧 change.md 的单次 packed/V-JEPA anchor/移除 DINO 方案混入本轮。
