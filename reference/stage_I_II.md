# LaWAM_aba2_change：Stage I / Stage II 完整训练流程与维度变化（LIBERO）

核对日期：2026-10-11。本文结合 [summary.md](summary.md)、`LaWAM_aba2_change/goal_local` 实际代码、LIBERO 历史运行配置，以及 Stage-I step-2000 checkpoint 的模型配置/参数形状整理。**实际入口是 `goal_local/train.py`，不是目录内保留的旧 `starVLA` 训练入口。**

**Stage I 用“当前观测 + 真实未来”学习后验 Goal/Local 表示，以动作 flow loss 训练；Stage II 冻结这个后验 Teacher，让只接收当前观测的 Student 同时学习动作预测和逐层 Goal/Local 蒸馏。部署时只保留 Student。**

本文的 LIBERO 指 `libero_merged_no_noops_20hz` 实验，Stage-II Teacher 来自 Stage I 第 **2000** 步。LIBERO-Plus 第 5000 步 Teacher 属于另一套实验，不能混用。

## 1. 统一符号和关键维度

| 符号 | LIBERO 中的值 | 含义 |
|---|---:|---|
| `B` | 历史训练为 2 | 一个 micro-batch 的样本数 |
| `B_f` | `2B`，历史训练为 4 | 动作 flow 分支重复采样两次后的 batch 大小 |
| `H` | 8 | 动作 chunk 长度，20 Hz 下覆盖 0.4 秒控制窗口 |
| `T_buf` | 50 | 动作头内部固定时间轴长度 |
| `D_action` | 32 | 动作头内部通道数；LIBERO 只监督前 7 维 |
| `N` | 64 | 每幅 256×256 图像的 Qwen merged visual token 数 |
| `L` | 随指令及 padding 变化 | 当前上下文 C 的长度，包含旧 8 Action + 8 Flow Query，不含新角色槽 |
| `S` | `L+16` | 加入新 Goal/Local 槽后的语言序列长度 |
| `D_vlm` | 2048 | Qwen 语言 hidden / Goal / Local 通道数 |
| `D_qwen_visual` | 1024 | Qwen 视觉主干内部通道数，merger 后才变为 2048 |
| `D_base` | 768 | DINO 和动作头 base memory 的通道数 |
| `D_dit` | 1024 | DiT、Plan token、投影后 Goal/Local 通道数 |
| `K_G / K_L` | 8 / 8 | Goal / Local 各自槽数 |
| `K_plan` | 4 | DiT 的 Plan token 数 |
| `M_base` | `256+(L−16)=L+240` | 动作头 base memory 长度 |

旧 `act_query[8,2048]`、旧 `flow_action_query[8,2048]`、新 Goal 的 8 槽、新 Local 的 8 槽是不同参数/表示。动作的 8 个时间步又是另一条轴。

```text
Stage I：
当前双视角 + 指令 + 旧 16 个 query ─────────────────→ C
真实 Local 双视角 → 冻结 Qwen vision → Local Resampler → F_L
真实 Goal 五帧   → 冻结 Qwen vision → Goal Resampler  → F_G
                    [C; F_G; F_L] → 16 层角色隔离 VLM
                                      ↓
        当前 VLM memory + 当前主视角 DINO + Goal/Local hidden
                                      ↓
                         分层条件化 DiT → 动作 flow loss

Stage II：
冻结 Teacher：当前信息 + 真实未来 → 第 4/8/12/16 层目标表示
                                              ↓ 仅作为 loss target
可训练 Student：当前信息 + Q_G/Q_L → 16 层 VLM → Student 动作 flow loss
                                   ↓
                       Local 蒸馏 loss + Goal 蒸馏 loss
```

## 2. 数据采样、归一化与有效性规则

对应源码：[goal_local/data.py](LaWAM_aba2_change/goal_local/data.py)。数据位于 `/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero_merged_no_noops_20hz`。

### 2.1 一个样本的组成

在长度为 E 的 episode 内选 anchor t，只从同一个 episode 取：

| 数据 | 时间位置 | 相对当前时间 | 视角 |
|---|---|---:|---|
| 当前观测 | t | 0 | 主视角 + 腕部 |
| 动作标签 | t…t+7，截断至 episode 末尾 | 8 个控制步 | 7 维动作 |
| Local future | t+8 | 0.4 秒 | 主视角 + 腕部 |
| Goal future | t+[24,40,56,72,88] | [1.2,2.0,2.8,3.6,4.4] 秒 | 五张主视角 |

Local 是**一个未来时刻的两个视角**；Goal 是**五个有序未来时刻的主视角**。Goal 用 horizon embedding 标识时间，不是把五帧作为视频联合编码。

图像解码为 RGB uint8；DataLoader 堆叠得到：

| batch 字段 | 形状 | 作用 |
|---|---|---|
| `current_main/current_wrist` | 各 `[B,3,256,256]` | 当前双视角 |
| `local_main/local_wrist` | 各 `[B,3,256,256]` | 真实 Local future |
| `goal_main` | `[B,5,3,256,256]` | 真实 Goal future |
| `instruction` | B 个字符串 | 文本指令 |
| `local_valid` | `[B]` | t+8<E |
| `goal_frame_valid` | `[B,5]` | 五个 Goal 索引分别是否小于 E |
| `action_chunk` | `[B,8,7]` | 原始动作窗口，尾部补零 |
| `actions/actions_mask` | 各 `[B,50,32]` | 动作头标签 / 逐时间、逐通道监督 mask |
| `remaining_actions` | `[B]` | E−t |
| `action_hz` | `[B]`，值 20 | 时间网格和动作有效长度 |
| `embodiment_id` | `[B]`，值 25 | 选择 LIBERO 动作编解码参数组 |
| `future_indices` | `[B,6]` | Local 和五个 Goal 的索引 |

Stage-II batch 仍包含未来，以供 Teacher 生成目标；**Student 的 encode() 不读取未来图像或未来有效性。**系统外层才用未来 mask 决定是否计算对应蒸馏。

### 2.2 动作标签如何从 8×7 变成 50×32

使用原 LIBERO SFT `dataset_statistics.json` 中 `franka.action.min/max`：

```text
a_norm[d] = 2 × (a[d]−min[d]) / (max[d]−min[d]) − 1，d=0,…,5
a_norm[6] = float(a[6] > 0.5)
r = min(8,E−t)
actions[:r,:7] = normalize(raw[:r,:7])
actions_mask[:r,:7] = True
其余标签补 0，监督 mask=False
```

代码不额外 clipping 前 6 维；夹爪阈值化为 0/1，不映射成 −1/+1。完整样本实际只监督 `8×7=56` 个标量，50×32 是继承动作头的内部容量。

### 2.3 episode 尾部如何处理

尾部样本不丢弃，不跨 episode 取未来，不复制最后一帧伪造有效未来。缺失图像在 batch 中是零占位，但只编码有效帧。全无效角色的 Resampler 直接输出 `[8,2048]` 的零表示，避免全 masked softmax。

零表示进入 VLM 后仍可能产生非零 hidden，因此还需以下规则：

| 情况 | Stage I | Stage II |
|---|---|---|
| Local 缺失 | 关闭 Local→Action 注入 | 对该样本跳过 Local 蒸馏；Student Local→Action 仍开启 |
| 五个 Goal 全缺失 | 关闭 Goal→Plan 注入 | 对该样本跳过 Goal 蒸馏；Student Goal→Plan 仍开启 |
| 部分 Goal 有效 | Resampler 只聚合有效 horizon | 整组 8 个 Goal 槽参与蒸馏 |
| 动作剩 r<8 步 | 只监督 r×7 | 同样只监督 r×7 |

例如 E=100、t=94：剩 6 步动作，未来全无效。Stage I 两路未来注入关闭，仍训练 6 步动作；Stage II 该样本两项蒸馏为零，Student 仍使用自己的 Goal/Local 训练动作。

## 3. Stage I：后验模型的初始化与 VLM 前向

对应源码：[model.py](LaWAM_aba2_change/goal_local/model.py)、[roles.py](LaWAM_aba2_change/goal_local/roles.py)。

### 3.1 初始化与训练范围

原始权重 W0：`/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`。

`GoalLocalPolicy.from_baseline()` 从 W0 的 `policy_backend.*` 恢复 Qwen、DINO、旧 query、原动作头。Qwen/DINO 参数均来自这份 SFT 权重，qwen_path/dino_path 提供配置及 processor。

| 模块 | 来源 | Stage I |
|---|---|---|
| Qwen vision 24 层、merger、DeepStack merger | W0 | 冻结，eval + no_grad |
| Qwen 词嵌入、前 16 层语言骨干、最终 RMSNorm | W0 | 训练 |
| DINOv3 ViT-B/16 | W0 的 lam.vision_encoder.model.* | 冻结，eval + no_grad |
| 旧 Action / Flow Query，各 8×2048 | W0 | 训练 |
| 原 Flow 动作编解码、VLM 投影、时间编码、DiT | W0 | 参与前向的可训练参数更新 |
| 两个 Future Resampler、view/horizon embedding | 新初始化 | 训练 |
| Goal/Local 投影、Plan、新 attention、残差 scale | 新初始化 | 训练 |

新 query、view/horizon、Plan 参数按 std=0.02 随机初始化；新增动作残差 scale 初值 0.01。Stage I 没有 Student q_goal/q_local 或 RelevanceGate。旧 LaWM/LAM 编解码、未来重建链和 State MLP 均不运行，`use_state=false`。

基础 Qwen 配置的语言层数是 28，但本实现和实际 checkpoint 使用 **16 层**。

### 3.2 一幅图像经过 Qwen vision 的完整维度

当前和有效未来共用冻结 Qwen vision。当前 processor 保持 256×256，像素按 mean/std=0.5/0.5 归一化。

| 步骤 | 单图形状 | 说明 |
|---|---|---|
| RGB 输入 | `[3,256,256]` | uint8 |
| temporal patch 补齐 | `[2,3,256,256]` | 静态图复制以适配 temporal_patch_size=2，不是额外真实帧 |
| patchify | `[256,1536]` | 256=16×16；1536=3×2×16×16 |
| image_grid_thw | `[1,16,16]` | temporal grid=1 |
| Conv3D patch embedding | `[256,1024]` | kernel/stride=(2,16,16) |
| 24 层视觉 Transformer | `[256,1024]` 不变 | 16 heads×64；MLP 1024→4096→1024 |
| 2×2 spatial merge | `[64,4096]` | 拼接 4 个 1024 维 patch |
| merger MLP | `[64,4096]→[64,4096]→[64,2048]` | 进入语言 hidden 空间 |

所以当前固定配置 N=64，双视角合计 128 个图像 token。一般公式为 `N=grid_t×grid_h×grid_w/2²`，64 不是任意分辨率的通用常数。

当前 2B 幅图像产生 `pixel_values:[512B,1536]`、`image_grid_thw:[2B,3]`、merged features `[128B,2048]`，后者填回各样本图像 token 位置。

视觉 block 索引 [5,11,17]（第 6/12/18 层）还产生 3 组 `[128B,2048]` DeepStack features，分别加到语言第 1/2/3 层输出的**当前图像位置**。未来分支仅使用最终 pooler_output，不直接向语言层注入未来 DeepStack。

### 3.3 当前上下文 C

沿用 [vlm_adapter.py](LaWAM_aba2_change/starVLA/model/framework/latent_world/vlm_adapter.py) 的消息顺序：

```text
chat 起始标记
当前主视角（64 image tokens）
Your task is {instruction}.
From main-view observations, infer the robot arm's motion.
旧 Action Query 占位符 × 8
当前腕部视角（64 image tokens）
Then use observations and the inferred motion to produce the robot policy actions.
旧 Flow Query 占位符 × 8
chat 结束/assistant 起始标记及 padding
```

实际首段 prompt 在 task 句与 motion 句之间有两个换行。processor 输出 input_ids/attention_mask `[B,L]`；词嵌入矩阵 `[151936,2048]` 产生 `[B,L,2048]`，图像位置替换为视觉特征，16 个 `<ACT_PH>` 位置替换为旧 query。

L 包括图像、文本、旧 query、特殊 token、padding。这里执行 hidden-state 前向，不生成动作文本、不计算语言交叉熵。

### 3.4 真实未来 → 两个 Resampler

全有效时每样本另编码 7 幅未来图像，processor pixel_values `[1792B,1536]`，merged features 共 `[448B,2048]`。缺失时只编码有效图片，再按原视角/时间位置 scatter。

| 分支 | 冻结视觉输出 | 身份 embedding | 展平后的 memory / mask | 输出 |
|---|---|---|---|---|
| Local | `[B,2,64,2048]` | view `[2,2048]` | `[B,128,2048]` / `[B,128]` | F_L `[B,8,2048]` |
| Goal | `[B,5,64,2048]` | horizon `[5,2048]` | `[B,320,2048]` / `[B,320]` | F_G `[B,8,2048]` |

两个 Resampler 参数独立，各有 query `[8,2048]`，每个执行一次 cross-attention 和残差 FFN：

```text
Q0：[8,2048] → broadcast [B_valid,8,2048]
Q=LN(Q0)，K=V=LN(memory)
8-head attention，每 head 256 维
Local 理论 score：[B_valid,8_heads,8_slots,128]
Goal 理论 score： [B_valid,8_heads,8_slots,320]
x = Q0 + Attention(Q,K,V)                    [B_valid,8,2048]
FFN：LN → Linear → GELU → Linear             2048→4096→2048
输出 = LN(x + FFN(x))                        [B_valid,8,2048]
scatter 回完整 batch                        [B,8,2048]
```

B_valid 是该角色至少有一个有效未来的样本数。Local 两视角共用 local_valid；Goal 将逐帧有效性展开到各帧 64 个 token。全无效行直接返回零。

### 3.5 拼接、角色隔离及 16 层语言模型

```text
[C;F_G;F_L]：[B,L,2048] + [B,8,2048] + [B,8,2048]
           → [B,L+16,2048]
```

16 层 native attention 规则一致，行=query，列=key/value：

| Query / Key-Value | C | Goal | Local |
|---|---|---|---|
| C | causal，仅当前位置及之前的有效 C | 禁止 | 禁止 |
| Goal | 全部有效 C | 同组 8 槽双向可见 | 禁止 |
| Local | 全部有效 C | 禁止 | 同组 8 槽双向可见 |

加性 mask `[B,1,S,S]`，允许处 0、禁止处 −∞；C padding 不能作为有效 key。C 不吸收未来，不能成为 Goal↔Local 的隐式桥梁。

位置张量 `[3,B,S]`：C 使用顺序位置 0…L−1，三条位置轴相同；角色追加 L…L+15。这是 current_embeddings() 的实际逻辑，不能依据旧注释写成图像网格原生 MRoPE。Teacher/Student 位置约定相同。

| 每层 Qwen 文本运算 | 维度 |
|---|---|
| 输入 / RMSNorm | `[B,S,2048]` |
| Q projection | 2048→2048，reshape `[B,16,S,128]` |
| K/V projection | 各 2048→1024，reshape `[B,8,S,128]` |
| GQA | 8 组 KV 对应 16 个 Q heads；逻辑 score `[B,16,S,S]` |
| attention 输出及 residual | `[B,S,2048]` |
| SwiGLU MLP | gate/up 各 2048→6144，相乘后 6144→2048 |
| block 输出 | `[B,S,2048]`，16 层形状不变 |

高效 attention 不一定显式存储整张 score 矩阵；本文 score 形状指数学轴尺寸。

第 16 层后最终 RMSNorm，再拆分：

```text
完整 hidden           [B,L+16,2048]
当前 hidden           [B,L,2048] → 删除旧 16 个 query 位置 → [B,L−16,2048]
当前有效性 mask       [B,L] → [B,L−16]
Goal hidden Z_G       [B,8,2048]
Local hidden Z_L      [B,8,2048]
```

删除的是旧 query 对应位置，不一定是 C 最后 16 个连续位置。旧 query 已经参与 VLM，仍能被新角色读取；新 Goal/Local 走独立动作条件分支。

## 4. 两阶段共用的动作分支：条件构造、Flow Matching 和 DiT

对应源码：[goal_local/action.py](LaWAM_aba2_change/goal_local/action.py)、[flowmatching_expert.py](LaWAM_aba2_change/starVLA/model/framework/vlas/flowmatching_expert.py)、[cross_attention_dit.py](LaWAM_aba2_change/starVLA/model/framework/vlas/cross_attention_dit.py)。Stage I 使用后验 Z_G/Z_L；Stage II 使用 Student 自己产生的 Z_G/Z_L。动作计算结构相同。

### 4.1 当前主视角的 DINO 分支

DINO 只接收当前主视角，不接收腕部或未来：

```text
current_main uint8                       [B,3,256,256]
除以 255，按 ImageNet mean/std 归一化    [B,3,256,256]
16×16 patch embedding                   [B,256,768]
加 1 个 CLS + 4 个 register tokens       [B,261,768]
DINOv3 ViT-B/16，12 层                   [B,261,768]
取 hidden_states[-2]                    [B,261,768]
去掉前 5 个 CLS/register token          [B,256,768]
无可学习参数的 F.layer_norm             [B,256,768]
```

mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`。`hidden_states[-2]` 是倒数第二个返回的 hidden state，对这里 12 层 DINO 对应第 11 层输出，不是直接取最终 pooler。每层 12 heads×64，MLP 768→3072→768。

虽然 DINO 配置名义 image_size=224，当前代码直接传 256×256，不调用将图片缩到 224 的 processor，所以这里是 256 个 patch，不是 196 个。整个分支冻结、no_grad。

### 4.2 三类动作条件及其 mask

```text
当前 VLM 输出（已删除旧 query）      [B,L−16,2048]
enc_vlm：Linear(2048,768)+LayerNorm  [B,L−16,768]
与当前 DINO 沿 token 轴拼接          [B,256+(L−16),768]
base memory                         [B,L+240,768]

Z_G → goal_proj：Linear(2048,1024)   G_dit [B,8,1024]
Z_L → local_proj：Linear(2048,1024)  L_dit [B,8,1024]
```

base memory 按 `[DINO; current_VLM_without_old_queries]` 排列。Goal/Local 不拼到这条 768 维 memory，而是经独立的 1024 维 cross-attention 注入。

| 条件/mask | 形状 | 语义 |
|---|---|---|
| base | `[B,M_base,768]` | 当前 DINO + 当前 VLM |
| image_mask | `[B,M_base]` | 仅前 256 个 DINO 位置有效 |
| vlm_mask | `[B,M_base]` | 仅后面的有效当前 VLM 位置可见，屏蔽其 padding |
| G_dit / L_dit | 各 `[B,8,1024]` | 两路角色条件 |
| role_valid | `[B,2]` | 列顺序为 Goal、Local |

Stage I：`role_valid=[goal_frame_valid.any(-1),local_valid]`。Stage-II Student：role_valid 恒为 True；不会因训练样本缺未来而关闭 Student 的角色条件。

### 4.3 动作 flow loss 的噪声与时间采样

训练时不是执行 10 次积分，而是对随机 flow 时间训练速度预测。先把已算好的条件、动作标签及 mask 沿 batch 轴 repeat_interleave 两次：

```text
原动作/监督 mask：[B,50,32] → [B_f,50,32]，B_f=2B
base：[B,M_base,768] → [B_f,M_base,768]
Goal/Local：各 [B,8,1024] → [B_f,8,1024]
其余 mask、hz、embodiment_id 同步重复
```

每份独立采噪声与 flow 时间：

```text
ε ~ Normal(0,I)                 [B_f,50,32]
u ~ Beta(1.5,1.0)               [B_f]
τ = (1−u) × 0.999               [B_f] → [B_f,1,1]
m_token = actions_mask.any(-1)  [B_f,50]
x_τ = ((1−τ)ε + τa) × m_token   [B_f,50,32]
velocity target = a−ε           [B_f,50,32]
bucket = clamp(floor(1000τ),0,999)  [B_f]
```

τ 是从噪声到动作的 flow 插值时间，同一样本的所有时间 token 共用 τ。不要与动作发生的自然时间 `j/20` 混淆。

**两种 mask 的用途不同：**输入噪声只按 m_token 清零无效时间位置；有效时间的第 8…32 个通道仍可能带噪声。最终 loss 才用 `[B_f,50,32]` 的 actions_mask 屏蔽无效动作通道。因此不能说训练内部只有 7 个噪声通道。

`repeated_diffusion_steps=2` 表示重复噪声/时间样本，不是两个优化器 step，也不是训练时做两次 Euler 积分。Qwen/DINO/Resampler 不因此重新前向两次。

### 4.4 动作编码、Plan 前缀和两种时间嵌入

动作编码器是按 embodiment 选择参数的两层 MLP，LIBERO 使用 id=25：

```text
x_τ                                    [B_f,50,32]
CategorySpecificLinear W1 + swish       [B_f,50,1024]
CategorySpecificLinear W2               [B_f,50,1024]
乘 m_token                             [B_f,50,1024]
4 个可学习 Plan token broadcast        [B_f,4,1024]
[Plan; Action]                         [B_f,54,1024]
```

W1 参数 `[32_embodiments,32,1024]`，W2 参数 `[32_embodiments,1024,1024]`。这里两个 32 分别表示 embodiment 数量和动作通道数，不是同一含义。

Plan 参数本身为 `[4,1024]`，是 Stage I 起学习并被 Student 继承的参数，不是 VLM Goal hidden 直接裁剪成 4 个 token。

自然时间位置与 flow 时间条件分开编码：

| 项目 | 计算 | 形状 |
|---|---|---|
| 自然时间 grid | j/hz，j=0…49 | `[B_f,50]` |
| 频率有效长度 | floor(0.4×20)=8，再受 episode 剩余长度限制 | token mask `[B_f,50]` |
| ContinuousTimeEncoder | sin/cos，自然时间→1024 维，invalid 时间位置清零 | `[B_f,50,1024]` |
| Plan 位置前缀 | 4 个零位置向量 | 拼接后 `[B_f,54,1024]` |
| flow bucket 编码 | bucket→256 维正弦投影→MLP 256→1024→1024 | temb `[B_f,1024]` |
| DiT AdaLN | SiLU(temb)→Linear 1024→2048→scale/shift | 各 `[B_f,1024]`，广播到 token 轴 |

自然时间 embedding 在每个 block 的 attention 前加到归一化 hidden；flow temb 则控制 AdaLN。完整 chunk 有 4 个有效 Plan + 8 个有效动作 token，但物理张量始终是 `[B_f,54,1024]`。

### 4.5 16 层 DiT 究竟每层做什么

`HierarchicalFlow.velocity()` 自己遍历原 DiT 的 blocks。下面层号从 1 开始，代码 i 从 0 开始。**新增角色注入发生在该层原 block 运算之前。**

| DiT 物理层 | Goal→Plan | Local→Action | 原 block 的 attention |
|---|---|---|---|
| 1、5、9、13 | 是 | 是 | [Plan;Action] cross-attend 当前 VLM memory |
| 2、6、10、14 | 否 | 是 | [Plan;Action] 联合 self-attention |
| 3、7、11、15 | 否 | 是 | [Plan;Action] cross-attend DINO memory |
| 4、8、12、16 | 否 | 是 | [Plan;Action] 联合 self-attention |

每层具体先执行：

```text
若 i ∈ {0,4,8,12}：
  Plan ← Plan + goal_scale[i] × valid_G × MHA(Plan,G_dit,G_dit)
所有 i：
  Action ← Action + local_scale[i] × valid_L × MHA(Action,L_dit,L_dit)
随后拼接回 [Plan;Action]，进入原 block
```

| attention | Q / K,V 输入形状 | heads / 理论 score |
|---|---|---|
| Goal→Plan | `[B_f,4,1024]` / `[B_f,8,1024]` | 16 heads×64，`[B_f,16,4,8]` |
| Local→Action | `[B_f,50,1024]` / `[B_f,8,1024]` | 16 heads×64，`[B_f,16,50,8]` |
| 原 cross-attention | `[B_f,54,1024]` / `[B_f,M_base,768]` | Q/K/V 投影到 16 heads×64；score `[B_f,16,54,M_base]` |
| 原 self-attention | Q=K=V 为 `[B_f,54,1024]` | 16 heads×64，score `[B_f,16,54,54]` |

原 cross-attention 接收整条 base memory，通过 vlm_mask/image_mask 决定该层读取哪一段；不是把 memory 物理改成 1024 通道。K/V 投影在 attention 内部完成 768→1024。

原 block 为 AdaLN/位置注入→attention+residual→LayerNorm→FFN+residual。FFN 为 1024→4096→1024（GELU approximate）；每层输出仍 `[B_f,54,1024]`。self-attention 屏蔽无效动作 key，最终速度再次按 token mask 清零；无效位置的中间计算不保证始终为零。

Goal 首先更新 Plan，随后通过 Plan/Action self-attention 影响动作；Local 每层直接更新 Action。**“VLM 中 Goal/Local 隔离”不意味着“动作头中的 Plan/Action 隔离”。**

### 4.6 速度输出与 Stage-I loss

```text
16 层 DiT 输出                        [B_f,54,1024]
temb → SiLU → Linear(1024,2048)
拆成输出 shift/scale                  各 [B_f,1024]
norm_out(hidden) × (1+scale) + shift   [B_f,54,1024]
proj_out_2：1024→1024                 [B_f,54,1024]
去掉前 4 个 Plan token                [B_f,50,1024]
动作 decoder：1024→1024→32，ReLU      [B_f,50,32]
乘有效时间 mask                       v_pred [B_f,50,32]
```

decoder 同样按 embodiment=25 选参数。loss 在 FP32 中计算：

```text
L_flow = sum(actions_mask × (v_pred − (a−ε))²)
         / max(sum(actions_mask),1)
L_StageI = L_posterior_flow
```

完整 B=2、B_f=4 时，分母为 `4×8×7=224`；episode 尾部按实际有效动作数减少。没有像素重建、latent 重建、KL、语言 CE 或 Stage-I 蒸馏项。

梯度由 action loss 回传到动作头、语言骨干、旧 query，以及 Future Resampler/view/horizon；冻结 Qwen vision 和 DINO 不更新。Resampler 虽然接收 no_grad 视觉输出，其自己的 attention/FFN/query 仍可学习。Stage I 完成后保存的是后验模型，包括 future.* 和完整动作头，供 Stage II 交接。

## 5. Stage II：独立冻结 Teacher 与 current-only Student

对应源码：`model.py::make_stage_two/DistillationSystem`，以及 `roles.py::RelevanceGate/role_distillation`。

### 5.1 从 Stage I 到两条独立分支

读取已经训练的 Stage-I checkpoint，要求 mode=posterior 且 source_step>0，然后：

1. 构造 Student，深拷贝 Stage-I Qwen 和 DINO。
2. 将 Stage-I 除 future.* 外的公共参数加载给 Student，包括旧 query、完整分层动作头、Goal/Local 投影、Plan、两路动作 attention/scale。
3. Student 新增 `q_goal:[8,2048]`、`q_local:[8,2048]` 和 RelevanceGate；这些不是从某个样本的 Teacher hidden 拷贝得到的。
4. 原后验模型成为 Teacher：保留 Qwen、旧 query、Future Resamplers、view/horizon，全部 requires_grad=False 且 eval；其 `flow=None`，移除动作头。
5. Teacher 和 Student 不共享参数存储。优化器只包含 requires_grad=True 的参数，并检查 Teacher 在 Student 更新后没有梯度/参数版本变化。

| 模块 | Stage-II Teacher | Stage-II Student |
|---|---|---|
| Qwen vision / merger | 冻结 | 冻结副本 |
| Qwen 词嵌入和 16 层语言骨干 | 冻结 | 继承 Stage I 后继续训练 |
| 原 8 Action + 8 Flow Query | 冻结 | 继承后继续训练 |
| Future Resampler / view / horizon | 冻结，继续处理真实未来 | 不存在 |
| 新 Q_G / Q_L | 不存在 | 新初始化，训练 |
| 深层 RelevanceGate | 不存在 | 新初始化，训练 |
| 分层动作 Flow/DiT | 被移除，不预测动作 | 继承 Stage I，继续训练 |
| DINO | 对象仍可保留，但 Teacher.encode() 不调用 | 冻结，用于 Student 动作条件 |

Student 没有重新随机初始化整个动作头。Teacher 也不是 Student 当前参数的 EMA，而是固定的 Stage-I 模型。

### 5.2 一个 micro-batch 的两次 VLM 前向

```text
with no_grad():
    Teacher：当前双视角/指令 + 真实 Local/Goal future
           → future resamplers
           → [C_T;F_G;F_L]                 [B,L+16,2048]
           → 16 层隔离 VLM
           → 第 4/8/12/16 层 G_T、L_T     每项 [B,8,2048]

Student：当前双视角/指令
         q_goal/q_local broadcast         各 [8,2048]→[B,8,2048]
         → [C_S;Q_G;Q_L]                  [B,L+16,2048]
         → 16 层 VLM，第 9–16 层加单向 gate
         → 第 4/8/12/16 层 G_S、L_S       每项 [B,8,2048]
         → 最后一层 Student 表示 + Student 当前 DINO
         → 第 4 节动作分支，B→B_f=2B
         → Student action flow loss
```

两支各自计算当前图像与 VLM 表示；Teacher 不给 Student 传 hidden、KV cache、Resampler 输出或动作条件。Teacher 表示仅在损失的 target 端使用。随着 Student 更新，C_S 与冻结 Teacher 的 C_T 也可以不同。

Student 训练时接收真实动作 a 作为 flow matching 标签/构造带噪动作的依据；“current-only”指**观测条件**不包含未来，不表示训练不使用动作标签。

### 5.3 第 9–16 层的 Goal→Local relevance gate

Student 的 native attention mask 始终沿用第 3.5 节的角色隔离规则。区别在于：第 1–8 层没有额外角色通信；第 9–16 层在每个语言 block 之后，显式用 Goal 单向更新 Local。

令该层 Goal/Local 为 G、L，各 `[B,8,2048]`：

| 步骤 | 计算 | 维度 |
|---|---|---|
| 归一化 | g=LN(G)，l=LN(L) | 各 `[B,8,2048]` |
| relevance 投影 | q_rel=W_L l，k_goal=W_G g | 各 `[B,8,256]` |
| 拼 NULL key | `[k_null;k_goal]` | `[B,9,256]`，NULL 参数 `[1,256]` |
| relevance logits | q_rel × keysᵀ / √256 | `[B,8,9]` |
| softmax | 对最后 9 个候选做 softmax，FP32 | `[B,8,9]` |
| 每个 Local 槽的相关性 | r=1−P(NULL) | `[B,8,1]` |
| Goal message | MHA(Q=l,K=g,V=g) | `[B,8,2048]` |
| 门控残差 | L_new=L+scale[layer]×r×message | `[B,8,2048]` |

message attention 为 8 heads×256，理论 score `[B,8_heads,8_local,8_goal]`。relevance 的 `[B,8,9]` 表示**每个 Local 槽对 1 个 NULL + 8 个 Goal 槽的评分**，不是新增了第 9 个 Goal 槽。

relevance 产生“是否接收 Goal 信息”的逐 Local 槽标量；具体接收什么内容由另一个 message MHA 决定，不能把这两套权重混成一次 attention。

第 9–16 层共用 relevance 的 LN/投影/NULL 参数，但各有独立 message attention 和残差 scale，共 8 组。scale 初值 0.01，可训练；r 在 0…1，scale 本身不被限制到 0…1。

更新后拼回 `[C;G;L_new]`，总体仍 `[B,L+16,2048]`。C、Goal 不被这条 gate 更新，Local→Goal 仍禁止。Student Goal 没有 detach，所以动作/Local loss 可通过 gate 回传到 Student Goal 分支。

### 5.4 蒸馏取哪些层，取哪个位置

| 语言层 | Teacher 目标 | Student 表示 | 是否最终 norm 后 |
|---|---|---|---|
| 4 | block 输出的 Goal/Local | block 输出的 Goal/Local | 否 |
| 8 | block 输出的 Goal/Local | block 输出的 Goal/Local | 否 |
| 12 | block 输出的 Goal/Local | 第 12 层 gate 后的 Goal/Local | 否 |
| 16 | 最终 Goal/Local | 第 16 层 gate 后再最终 norm | 是，两支都用 lm.norm 后表示 |

每层每个角色均 `[B,8,2048]`。4 层×2 角色，共 8 组对齐。Teacher 目标停止梯度；不蒸馏整段 C，也不蒸馏 Teacher 动作。

### 5.5 Cosine + slot Gram 的完整公式与维度

Local 选择 `local_valid=True` 的样本；Goal 选择 `goal_frame_valid.any(-1)=True` 的样本。对某层、某角色保留下来的 B_v 个样本：

```text
S,T                             [B_v,8,2048]
S_hat = normalize(S.float(),dim=-1)
T_hat = normalize(T.detach().float(),dim=-1)

cosine_per_slot = sum(S_hat*T_hat,dim=-1)       [B_v,8]
L_cos = mean(1−cosine_per_slot)                 scalar

Gram_S = S_hat @ S_hat.transpose(-1,-2)         [B_v,8,8]
Gram_T = T_hat @ T_hat.transpose(-1,-2)         [B_v,8,8]
L_gram = mean((Gram_S−Gram_T)²)                 scalar
L_role_at_layer = L_cos + α L_gram              scalar
```

Cosine 对齐同一槽位的方向；Gram 对齐 8 个槽之间的相似性结构。Gram 是 `8×8`，不是 `2048×2048`；没有额外槽位排列匹配。如果一个 micro-batch 该角色全无效，则返回可微的 0。

每种角色在四层上取平均：

```text
L_local = (L_local_4 + L_local_8 + L_local_12 + L_local_16) / 4
L_goal  = (L_goal_4  + L_goal_8  + L_goal_12  + L_goal_16) / 4

L_StageII = L_student_flow + λ_local L_local + λ_goal L_goal
λ_local=1，λ_goal=1，α=gram_alpha=1
```

蒸馏 batch 始终是 B；只有 action flow loss 内部扩为 B_f=2B。不是先把四层表示沿 token 轴拼成 32 个槽再算一个 Gram。

### 5.6 Stage II 反向更新路径

L_student_flow 更新 Student 动作头，并经角色/当前条件回到 Student 语言骨干、旧 query、新 Q_G/Q_L、gate；蒸馏直接更新对应 Student 角色表征路径。第 12/16 层的 Local 蒸馏也可以经 gate 影响 Student Goal。

Teacher 的 Qwen、Resampler、所有参数完全不更新；Student Qwen vision 和 DINO 也冻结。Stage II 没有 Teacher action loss、未来图像重建 loss、像素 loss 或额外状态监督。

## 6. 两阶段实际训练循环与 LIBERO 历史交接

对应源码：[train.py](LaWAM_aba2_change/goal_local/train.py)、[optim.py](LaWAM_aba2_change/goal_local/optim.py)。

### 6.1 数据划分和优化配置

该 LIBERO 历史审计记录 1693 个 episode、273465 个 frame、40 类任务。按任务分组、seed=2026、val_ratio=0.05，在 episode 层面划分；训练/验证 episode 不重叠。

| 项目 | 实际设置 |
|---|---|
| 训练集 | 1614 episodes / 260726 frames |
| 验证集 | 79 episodes / 12739 frames |
| 训练采样 | 训练 frame anchor 均匀 shuffle，不因远期未来缺失过滤样本 |
| 每次 micro-batch | 2 |
| 梯度累计 | 16 个 micro-batch |
| 每次 optimizer update 的 anchor 数 | 2×16=32 |
| flow 重复 | 每个 anchor 两份噪声/时间采样，即每 update 64 份 flow 训练实例；仍只有 32 个原始 anchor |
| 学习率 | 基础值 1e-5 |
| 优化器 | AdamW，FP32 master weights，betas=(0.9,0.95)，eps=1e-8 |
| weight decay | 1e-8 |
| 梯度裁剪 | 全局 norm 上限 1.0 |
| optimizer device | 历史设置为 CPU offload |
| 前向精度 | CUDA bf16 autocast；loss/蒸馏关键运算用 FP32 |
| 激活重计算 | checkpoint_blocks=true，训练时作用于语言/DiT blocks |
| warmup | 阶段配置为 500 步；实际公式取 min(cfg.warmup_steps, max(1,total_steps//10)) |
| 后续 LR | warmup 后 cosine 衰减至基础 LR 的 5% |
| validation | 固定选取 64 batches×2=128 个 anchor；按任务循环抽样，允许自然缺失未来 |
| 验证间隔 | 250 optimizer steps；训练前先验证一次 |

这里引用的是 `configs/goal_local/stage_I.json`、`stage_II.json` 与实际 run 配置；`default.json` 中 warmup_steps=100，不代表上述历史实验用了 100 步 warmup。

### 6.2 一次 optimizer step 的完整顺序

```text
model.train()，但冻结 vision/DINO 仍为 eval；Stage-II Teacher 永远 eval
清空梯度
重复 16 个 micro-batch：
    读取 B=2 的当前/未来/动作/有效性 mask
    Stage I：后验 encode → action flow loss
    Stage II：Teacher no_grad encode → Student encode
              → Student action flow + Local/Goal distillation
    loss / 16
    backward，累计到可训练参数
对所有可训练参数执行 clip_grad_norm_(max_norm=1.0)
复制梯度到 FP32 master weights
AdamW 更新 FP32 master weights，并拷回模型参数
scheduler.step()
记录 train loss、各分项、grad norm、LR、future 有效比例
Stage II 额外检查 Teacher 仍冻结、无梯度且参数未变
到验证点：eval + no_grad，固定样本和随机种子计算 validation
到保存点：保存模型、processor、trainer 状态和完成标记
```

验证并不是在线 LIBERO rollout：它评估的是离线 flow/蒸馏 loss。`evaluate()` 返回 flow 指标，**best checkpoint 按 validation flow 选择**，不是按 Stage-II 总 loss，更不是按在线成功率。

Stage I 的少量诊断 batch 还会将两路 role_valid 置零，比较有/无角色注入的 flow loss；这只是验证诊断，不是额外训练 loss。

### 6.3 checkpoint 保存和两个阶段的交接

`model.pt` 保存 mode、step、配置、Qwen/DINO config、归一化统计、模型 state_dict 等，旁边保存 processor。周期性完整 checkpoint 另有 trainer.pt（FP32 master/Adam 状态、scheduler、随机数状态等）以及 checkpoint_complete.json；best 一般是模型权重保存，不等价于完整 trainer 恢复包。

LIBERO 历史链路是：

```text
原始 LIBERO SFT W0
    ↓ Stage I posterior 训练
runs/goal_local/stage_I_20260921/step_002000
    ↓ 固定保存为 Teacher snapshot
runs/goal_local/teachers/stage_I_step_002000
    ↓ 新建独立 Teacher / Student
runs/goal_local/stage_II_teacher2000_20260921
    ↓ 取其 best，保存的学生 step=250
runs/goal_local/stage_II_warm250_20260921
    ↓ 从全局 step=250 继续，Teacher 仍固定为 Stage-I step=2000
step_004000/model.pt：部署用 Student
```

Stage I 的 arguments 原计划 10000 步、每 1000 步保存；这里选取第 2000 步教师，**不表示 Stage I 在所有后续历史中只运行了 2000 步**。Stage-II warm250 运行原计划也为 10000 步，每 250 步保存/验证；其 status 记录按要求在完整第 4000 步 checkpoint 后停止，保留原始 LR schedule。

warm250 的 warm_start.json 明确记录：从 step=250 的 Student 权重继续，**重置 optimizer 和 RNG，但 scheduler 延续全局步数 250**。这不是从零重新训练 4000 步，也不是恢复了全部 Adam 历史状态。Teacher 来源保持一致。

重要路径（相对 LaWAM_aba2_change）：

| 用途 | 路径 |
|---|---|
| LIBERO 固定 Teacher | `runs/goal_local/teachers/stage_I_step_002000/model.pt` |
| Teacher 来源记录 | `runs/goal_local/teachers/stage_I_step_002000/source.json` |
| Student 教师配置 | `runs/goal_local/stage_II_warm250_20260921/teacher_source.json` |
| Student warm start 记录 | `runs/goal_local/stage_II_warm250_20260921/warm_start.json` |
| Student 第 4000 步 | `runs/goal_local/stage_II_warm250_20260921/step_004000/model.pt` |

## 7. 一个完整数值例子：B=2、L=219、N=64

为确认序列长度，本次在 CPU 上用 Stage-I step-2000 保存的 processor，对 256×256 占位 RGB 图像和数据集中的两条真实指令运行了 apply_chat_template：

```text
1. put the white mug on the left plate and put the yellow and white mug on the right plate
2. put the white mug on the plate and put the chocolate pudding to the right of the plate
```

得到未 padding 长度分别 219、218；batch padding 后 L=219。每样本 128 个 image token、16 个旧 query。图像像素内容不影响这里的 token 数；这次验证的是 processor/形状，不是一次完整神经网络训练前向。

假设这两个样本的未来均有效、动作均有 8 步，则：

| 位置 | Stage I / Stage-II Teacher | Stage-II Student |
|---|---|---|
| 当前主/腕图像 | 各 `[2,3,256,256]` | 相同 |
| 当前 pixel_values | `[1024,1536]` | 相同，独立计算 |
| 当前 image_grid_thw | `[4,3]`，每行为 [1,16,16] | 相同 |
| 当前 merged visual features | `[256,2048]` | 相同 |
| input_ids / current mask | `[2,219]` | 相同 |
| 当前 C embeddings | `[2,219,2048]` | `[2,219,2048]` |
| Local 视觉特征 | `[2,2,64,2048]` | 不存在 |
| Goal 视觉特征 | `[2,5,64,2048]` | 不存在 |
| Local/Goal 展平 memory | `[2,128,2048]` / `[2,320,2048]` | 不存在 |
| 新角色输入 | F_G/F_L 各 `[2,8,2048]` | Q_G/Q_L broadcast 后各 `[2,8,2048]` |
| 拼接后 VLM 序列 | `[2,235,2048]` | 相同 |
| VLM 加性 mask | `[2,1,235,235]` | 相同 |
| VLM position_ids | `[3,2,235]` | 相同 |
| 每层 Q / K,V（分头后） | `[2,16,235,128]` / 各 `[2,8,235,128]` | 相同 |
| 第 4/8/12/16 层 G/L | 每项 `[2,8,2048]` | 每项相同形状；深层 Local 经 gate |
| gate relevance logits | 不存在 | `[2,8,9]`，relevance `[2,8,1]` |
| 去除旧 query 后的 C | `[2,203,2048]` | 相同 |

以下动作分支在 Stage I 和 Stage-II Student 中执行，**Stage-II Teacher 不执行**：

| 动作分支位置 | B=2 时的形状 |
|---|---|
| 当前主视角 DINO tokens | `[2,256,768]` |
| 投影后的当前 C | `[2,203,768]` |
| base memory | `[2,459,768]` |
| image_mask / vlm_mask | 各 `[2,459]` |
| Goal/Local 投影 | 各 `[2,8,1024]` |
| 数据标签与 mask | 各 `[2,50,32]`，有效区域每样本 8×7 |
| repeat=2 后标签/噪声/带噪动作 | 各 `[4,50,32]` |
| repeat 后 base | `[4,459,768]` |
| repeat 后 Goal/Local | 各 `[4,8,1024]` |
| 动作 embedding | `[4,50,1024]` |
| Plan 前缀 | `[4,4,1024]` |
| 16 层 DiT hidden | `[4,54,1024]` |
| flow 时间条件 temb | `[4,1024]` |
| 输出 velocity | `[4,50,32]` |
| flow loss | 标量，完整样本监督分母 224 |
| Stage-II 每层/角色 Gram | `[2,8,8]`，这里未来全有效 |
| Local/Goal 蒸馏 loss | 各一个标量 |

第二条指令少一个有效文本 token，因此其当前输出物理长度仍 203，但有效长度为 202；base 物理长度仍 459，其中有效当前条件位置总数为 458。padding 由 mask 处理。

**L=219 只是这组指令的具体例子，不是模型常数。**替换指令或 batch 后，应按 L+16、L−16、L+240 重新计算相应序列长度，2048/768/1024 等通道数不变。

## 8. 训练完成后的 Student 部署维度

对应源码：`GoalLocalPolicy.predict_action()` 和 `HierarchicalFlow.sample()`。输入仅为当前主视角、当前腕部图像和语言指令；没有 Teacher、真实未来、Resampler、蒸馏 loss 或本体状态 s_t。

```text
当前双视角与指令 → C + 内部旧 query + 内部 Q_G/Q_L
                 → 16 层 Student VLM + 第 9–16 层 gate
                 → current memory / Goal / Local
当前主视角 → 冻结 DINO
                 → 动作条件（形状与训练相同，但不做 B→2B 重复）
初始高斯噪声 x0                          [B,50,32]
20 Hz / 0.4 秒 mask：仅前 8 个时间有效
10 次 Euler 更新：
  bucket_k = floor((k/10)×1000)，k=0,…,9
  x_(k+1) = (x_k + velocity(x_k,bucket_k)/10) × token_valid
最终取前 8 时间、前 7 通道               [B,8,7]
前 6 维反归一化；第 7 维按 >0.5 二值化  [B,8,7]
```

每次 velocity 都用 4 Plan + 50 Action 的 `[B,54,1024]` DiT hidden；Qwen/DINO 的条件在一次 predict_action 中只编码一次，再被 10 次 flow 更新复用。训练随机 τ 的速度回归和部署 10 次积分是两个不同过程。

正常训练样本受 episode 尾部 mask 限制；在线推理没有示范 episode 剩余长度输入，固定预测 8 步动作 chunk。仿真执行端的图像约定、夹爪映射等属于部署接口，不应反过来改写上述训练归一化定义。

## 9. 与 summary.md 对照时应保留的精确边界

| 简略说法 | 本实现中的精确含义 |
|---|---|
| 图像 B×3×256×256 | 主/腕各一份；一幅经 Qwen 为 64×2048，当前 DINO 仅主视角为 256×768 |
| VLM 输入 L+16 | L 已经含旧 16 query；额外 +16 是新 Goal/Local，不是再次加入旧 query |
| 删除旧 query | 删除送入动作头 base memory 的对应输出位置；旧 query 仍进入 VLM |
| Teacher 完整冻结 | Stage-II 目标生成链完整冻结，但实现移除 Teacher 动作头，且不调用 Teacher DINO |
| Goal/Local 相互隔离 | native VLM mask 隔离；Student 第 9–16 层额外有单向 Goal→Local gate |
| 8×7 动作 | 有效控制标签/最终输出；内部标签、噪声、速度为 50×32 |
| 4 Plan + 50 Action | DiT 物理序列长 54；常规样本只有 4+8 个有效 token |
| 两次 flow repeat | B→2B 的噪声/时间样本，不增加独立数据 anchor，不重复 VLM |
| 缺失未来 | 影响 Teacher 聚合、Stage-I 角色注入和 Stage-II 蒸馏 mask，不关闭 Student 角色 |
| 两阶段训练 | Stage I 后验 action flow；Stage II Student action flow + 两路逐层蒸馏 |

## 10. 源码、配置和验证来源

下列链接均指向本地已核对文件，便于沿着数据→模型→loss→训练循环回查。

| 内容 | 主要来源 |
|---|---|
| 项目总结和架构继承 | [summary.md](summary.md)、[guide.md](LaWAM_aba2_change/guide.md) |
| 数据、未来索引、动作归一化和 mask | [goal_local/data.py](LaWAM_aba2_change/goal_local/data.py) |
| 初始化、图像编码、语言前向、两阶段交接 | [goal_local/model.py](LaWAM_aba2_change/goal_local/model.py) |
| Resampler、角色 mask、gate、蒸馏 | [goal_local/roles.py](LaWAM_aba2_change/goal_local/roles.py) |
| 分层动作条件、DiT 遍历、flow loss / sampling | [goal_local/action.py](LaWAM_aba2_change/goal_local/action.py) |
| 动作编解码、连续时间、噪声和 τ 采样 | [flowmatching_expert.py](LaWAM_aba2_change/starVLA/model/framework/vlas/flowmatching_expert.py) |
| 原 DiT block / AdaLN / FFN | [cross_attention_dit.py](LaWAM_aba2_change/starVLA/model/framework/vlas/cross_attention_dit.py) |
| 当前双视角 prompt 顺序 | [vlm_adapter.py](LaWAM_aba2_change/starVLA/model/framework/latent_world/vlm_adapter.py) |
| 训练循环、验证、checkpoint、warm start | [goal_local/train.py](LaWAM_aba2_change/goal_local/train.py) |
| FP32 master AdamW | [goal_local/optim.py](LaWAM_aba2_change/goal_local/optim.py) |
| Stage-I / II 配置 | [stage_I.json](LaWAM_aba2_change/configs/goal_local/stage_I.json)、[stage_II.json](LaWAM_aba2_change/configs/goal_local/stage_II.json) |
| 数据规模与划分 | [dataset_audit.json](LaWAM_aba2_change/runs/goal_local/stage_I_20260921/dataset_audit.json) |
| Stage-I 实际训练参数 | [arguments.json](LaWAM_aba2_change/runs/goal_local/stage_I_20260921/arguments.json) |
| Stage-II 实际训练参数 | [arguments.json](LaWAM_aba2_change/runs/goal_local/stage_II_warm250_20260921/arguments.json) |
| 固定 Teacher 来源 | [source.json](LaWAM_aba2_change/runs/goal_local/teachers/stage_I_step_002000/source.json)、[teacher_source.json](LaWAM_aba2_change/runs/goal_local/stage_II_warm250_20260921/teacher_source.json) |
| 权重热启动 / 第 4000 步停止 | [warm_start.json](LaWAM_aba2_change/runs/goal_local/stage_II_warm250_20260921/warm_start.json)、[status.json](LaWAM_aba2_change/runs/goal_local/stage_II_warm250_20260921/status.json) |
| Qwen 基础配置 / 图像 processor | [config.json](/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/qwen3_weights/config.json)、[preprocessor_config.json](/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/qwen3_weights/preprocessor_config.json) |
| DINO 配置 | [config.json](/mnt/hdd/wangchaoyang/CKPT/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m/config.json) |

本次还通过 CPU mmap 只读 Stage-I step-2000 model.pt，确认语言层数 16、embedding `[151936,2048]`、Q/K/V 投影、视觉 merger、动作编解码和 DiT FFN 参数形状；并运行保存的 processor 验证 N=64、示例 L=219 及两组旧 query 数量。其余维度按前向代码逐步推导，未重跑训练或在线评估，也未修改模型/配置。
