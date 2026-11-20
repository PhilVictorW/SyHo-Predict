# LIBERO A 组消融实验执行方案（A0–A4）

编写日期：2026-10-19。实验根目录：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1`。

**本次交付仅为方案文档，不修改训练、模型、数据或推理代码，不启动实验。用户已明确：A0 先训练 500 个完整 epoch，再推理；不是 500 个 optimizer step。** A1–A4 的 Stage I / II epoch 数尚未指定，本文明确其预算对齐规则，不将建议值当成已确认参数。

依据：[消融定义 abalation.md](../../reference/abalation.md)（2026-10-18 版，重点为 §1.1、§2、§3）、[Stage I / II 架构与历史审计](../../reference/stage_I_II.md)、[完整三阶段 idea](../../reference/stage_I_II_III.md)、[Stage III idea](../../reference/stage_III.md)，并结合现有 `goal_local` 训练、数据、模型和评估入口核对。下文区分原定义、历史事实和面向新实验的执行约定。

## 1. 对当前理解的判断

你的 A0–A3 理解总体正确，需要把以下边界落实到实现和实验记录。

| 项目 | 判断及需要精确化的部分 |
|---|---|
| A0 | 正确。从原始 LIBERO SFT 权重 W0 直接建立带两组角色 Query 的 current-only Student，只训练动作 flow loss；不经过本方法的 Stage I，不继承完整 Teacher，也不蒸馏。保留两组各 8 槽、深层 RelevanceGate、4 个 Plan 和双路径动作条件，新增模块从第一次更新起接受动作梯度。 |
| A1 | 正确，必须从 W0 独立训练 Local-only Teacher，再教自己的 Local-only Student。Local 真值是同一个 `t+H=t+8` 时刻的主视角和腕部图像。Teacher Resampler 和 Student Query 都用 16 槽，通道仍为 2048。 |
| A2 | 正确，必须从 W0 独立训练 Goal-only Teacher。远端趋势不是任意一个 `t+nH`，而是固定 `n=[3,5,7,9,11]` 的五张有序主视角图像；保留 horizon embedding。Teacher / Student 同样都用 16 槽。 |
| “单一 Query 尺寸两倍” | 指单个角色的**槽数从 8 增至 16**，即 `[16,2048]`；不是 hidden 从 2048 改成 4096，也不是把完整双角色的 16 槽再翻倍成 32，更不是动作长度改为 16。 |
| A3 | 正确，复用与 A4 相同的完整 Stage-I Teacher；从 Stage II 第一次前向开始就禁用 VLM 的额外 Goal→Local 交互，同时保留原生角色隔离 mask、双角色蒸馏和双路径动作条件。不能只在训练完成后的 A4 上关 gate 做推理来代替 A3。 |
| A4 | 是完整 **Stage II Student**，作为本组参照。A 组全部不运行 Stage III；三轮强化学习后的权重不属于 A4。 |

“16 层角色无交互”只针对 VLM 中的直接信息通路。A3 的 Goal 和 Local 仍可以分别经动作条件路径影响同一动作；两组角色共享骨干参数也不违反该定义。

A1/A2 与 A3/A4 匹配的是新角色 Token 总数。未来图像数量、Resampler 数量、动作条件路径、参数量、FLOPs 和蒸馏总梯度不完全相等，不能将“槽数一致”写成“排除了所有模型容量和信息量差异”。

## 2. A0–A4 的固定定义

| 编号 | Teacher 训练来源 | Local / Goal 槽数（Teacher 与 Student 一致） | Student 训练目标 | Student VLM 显式交互 | 动作条件 |
|---|---|---:|---|---|---|
| A0 | 无 Teacher；Student 直接从 W0 开始 | 8 / 8（仅 Student） | flow | 第 9–16 层 RelevanceGate | Local→Action；Goal→Plan→Action |
| A1 | W0 → Local-only Stage I | 16 / 0 | flow + Local 四层蒸馏 | 无 | 仅 Local→Action |
| A2 | W0 → Goal-only Stage I | 0 / 16 | flow + Goal 四层蒸馏 | 无 | 仅 Goal→Plan→Action |
| A3 | W0 → 完整双尺度 Stage I，与 A4 共用 | 8 / 8 | flow + Local 四层蒸馏 + Goal 四层蒸馏 | 全程无额外交互 | 保留双路径 |
| A4 | 同 A3 | 8 / 8 | flow + Local 四层蒸馏 + Goal 四层蒸馏 | 第 9–16 层 RelevanceGate | 保留双路径 |

主要比较关系：

- **A3 vs. A1 / A2**：没有额外 VLM 跨角色交互时，两种未来表征分别用于动作是否有互补价值。
- **A4 vs. A3**：完整双尺度学习基础上，监督训练的 Goal→Local 交互是否带来增益。
- **A4 vs. A0**：本方法完整显式未来学习（Stage I 后验训练 + Stage II 蒸馏）的整体价值，不能仅归因于蒸馏 loss。
- **A1 vs. A2**：近端变化与远端趋势在不同任务中的独立作用，结合逐任务结果解释。

A0 不等于 B1。B1 已继承完整 Stage-I 后验训练权重，而 A0 从 W0 直接开始。

## 3. 文件、权重和历史参照

以下 `P`、`A` 是文档中的路径缩写，不是现有程序参数：

```text
P = /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
A = P/ablation/aba1
```

### 3.1 本组输入资源

| 资源 | 实际路径 / 说明 |
|---|---|
| W0 原始 SFT 权重文件 | `/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt` |
| 原始动作统计 | `/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/dataset_statistics.json` |
| 训练数据 | `/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero_merged_no_noops_20hz` |
| Qwen 配置 / processor | `/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/qwen3_weights` |
| DINO 配置目录 | `/mnt/hdd/wangchaoyang/CKPT/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m` |
| 现有训练入口 | [goal_local/train.py](../../goal_local/train.py) |
| 现有阶段配置 | [stage_I.json](../../configs/goal_local/stage_I.json)、[stage_II.json](../../configs/goal_local/stage_II.json) |
| 现有推理实现 | [predict.py](../../goal_local/predict.py)、[eval_server.py](../../goal_local/eval_server.py)、[eval_task.py](../../goal_local/eval_task.py)、[eval_full.py](../../goal_local/eval_full.py) |
| 历史 LIBERO 运行环境源码 | `/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/LIBERO` |
| 历史 LIBERO 配置目录 | `/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/libero_config` |

Qwen 和 DINO 的模型参数应从 W0 恢复；上面的独立目录用于配置 / processor，不能在加载 W0 后又用另一份视觉权重覆盖。

### 3.2 历史完整流程中可参考的权重

```text
W0
  → P/runs/goal_local/stage_I_20260921/step_002000/model.pt
  → P/runs/goal_local/teachers/stage_I_step_002000/model.pt（固定 Teacher 副本）
  → P/runs/goal_local/stage_II_teacher2000_20260921
  → P/runs/goal_local/stage_II_warm250_20260921/step_004000/model.pt
```

历史 Student 使用 Stage-I step 2000 Teacher。Student 曾从自己的 step 250 权重 warm-start：重置 optimizer 和 RNG，scheduler 按全局 step 延续，最终使用 step 4000。原阶段 schedule 按 10000 步配置，不能将其等同于“从零连续训练、总 schedule 为 4000 步”。

已核对的历史 Stage-II 推理参照：

| 历史评估 | 结果文件（相对 P） | 成功数 / 总数 | 总成功率 | Spatial / Object / Goal / Long |
|---|---|---:|---:|---|
| Quick：40 tasks × 10 trials | `results/goal_local/libero_40x10/step_004000/summary.json` | 394 / 400 | **98.5%** | 100% / 100% / 97% / 97% |
| Full：40 tasks × 50 trials | `results/goal_local/libero_40x50/step_004000/summary.json` | 1959 / 2000 | **97.95%** | 99.4% / 99.2% / 97.8% / 95.4% |

这回答了“98.5% quick inference”的来源：它是上述历史 Stage-II step 4000 Student 的 400 次评估，不能代替 2000 次 full 评估，也不能直接记为新 A4 的结果。

只有初始化来源、数据划分、样本预算、LR、warm-start 规则、checkpoint 选择及推理协议均匹配时，才能复用历史 A4 做正式对照。若按新的 epoch 方案重训，应重训 A4，并让 A3/A4 共享新 Teacher；历史 step 2000 Teacher 不能自动成为新的等预算 Teacher。

## 4. 全组固定的输入、结构和采样

### 4.1 当前输入与动作规格

| 项目 | 固定值 |
|---|---|
| 当前输入 | 当前主视角 + 腕部 RGB 图像，256×256，以及同一任务指令 |
| 本体状态 | `use_state=false`，不额外输入状态或历史 |
| 旧 Query | 原 8 个 Action Query + 8 个 Flow Query，五组全部保留 |
| 新角色 Query | 总计 16 槽，hidden=2048；与旧 Query 完全不同 |
| VLM | 原 16 层语言网络；当前输入的 prompt / processor 保持一致 |
| 视觉冻结 | Qwen vision / merger 与 DINO 冻结；Stage I Future Resampler 可训练 |
| 动作头 | 4 个 Plan Token；内部 50 个 Action Token、32 个动作通道 |
| LIBERO 有效动作 | 20 Hz，H=8，有效维度 7，即每次预测 0.4 秒动作窗口 |
| 动作训练 | flow 噪声重复采样 B→2B；该操作不增加数据 epoch 计数 |
| 动作推理 | 10 次 Euler 更新，执行最多 8 个动作步后再观测 |

Current 上下文 C 已含旧 16 个 Query。令其长度为 `N_C`，VLM 总序列长度为 `N_C+16`：

```text
A0 / A3 / A4 Student： [C; Q_G(8); Q_L(8)]
A1 Teacher / Student： [C; F_L(16)] / [C; Q_L(16)]
A2 Teacher / Student： [C; F_G(16)] / [C; Q_G(16)]
完整双尺度 Teacher：  [C; F_G(8); F_L(8)]
```

角色切片必须按配置计算：Goal 是 `[N_C, N_C+K_G)`，Local 是 `[N_C+K_G, N_C+K_G+K_L)`。位置编码、padding 和有效 mask 也须支持 `K_G=0` 或 `K_L=0`，不能保留硬编码的“末尾 8 槽”逻辑。

原生 VLM attention 的规则不变：C 仅按原规则读取 C；Goal 读取有效 C 和本组 Goal；Local 读取有效 C 和本组 Local；角色内可互读，Goal / Local 不经原生 attention 互读，C 不能读取角色。这同时防止 C 成为 A3 的隐式跨角色信息桥梁。

动作 base memory 保留当前 VLM 和当前 DINO 条件。需要删除的是 C 中**旧 Query 的真实位置**，不是简单截掉序列末尾 16 个 Token。角色输出单独投影 2048→1024，作为条件 attention 的 K/V，不拼入当前 base memory。

### 4.2 未来真值采样

| 分组 | Stage I 和 Stage II Teacher 所需未来图像 | Resampler 输出 |
|---|---|---|
| A0 | 不读取任何本方法的未来图像，不运行 Future Resampler | 无 |
| A1 | `t+8` 的主视角 + 腕部图像，共两个视角 | Local：`[B,16,2048]` |
| A2 | `t+[24,40,56,72,88]` 的五张主视角，保持时间顺序 | Goal：`[B,16,2048]` |
| A3 / A4 | 上述 Local 两视角 + Goal 五帧 | 各 `[B,8,2048]` |

固定分辨率下，每幅未来图像经冻结 Qwen vision 得到 64 个 merged tokens。Local memory 为 `[B,128,2048]`，Goal memory 为 `[B,320,2048]`。A1 保留 Local view embedding；A2 保留五个 horizon embedding。16 个输出槽不改变未来输入帧数。

单尺度意味着数据解码、Teacher、蒸馏和动作条件四处都只启用对应未来角色。A2 不读取未来腕部图像，但**当前腕部图像仍保留**；A1 同样保留完整当前双视角。不能只将一项蒸馏权重设为零，却继续让另一种真实未来参与 Teacher 或动作生成。

继续沿用数据文件的 episode 边界、帧索引与视频时间偏移映射。不得把 episode 内索引错误地当成整段视频绝对索引。

### 4.3 尾部样本与有效性

所有组使用相同训练 anchors；不因某组未来缺失而删除 episode 尾部样本，不跨 episode 取未来，也不复制最后一帧伪造有效未来。

| 情况 | Stage I Teacher | Stage II Student / 蒸馏 |
|---|---|---|
| 某个已启用角色的未来全部缺失 | 对该样本关闭该角色的动作注入；避免全 masked attention 产生 NaN | 仅跳过该角色蒸馏；Student 的该角色仍参与动作预测 |
| Goal 只有部分 horizon 有效 | Resampler 只聚合有效帧 | 只要至少一个 Goal horizon 有效，就对该样本的整组 Goal 槽蒸馏 |
| 动作标签剩余不足 8 步 | 只监督真实存在的动作 | 同左 |
| 某个角色在实验中永久不存在 | 不构造或不执行对应模块 | 不计算该角色、loss、gate 和动作条件 |

“永久移除角色”与“某样本临时缺失未来”必须使用不同的配置语义。不能用 batch 的 `goal_valid=false` 冒充 A1 的结构配置。

动作标签继续沿用原始统计：前六维按 min/max 归一化，第七维按阈值生成 0/1 夹爪标签；只监督前 `min(8,E−t)×7` 个有效元素，其余 50×32 缓冲区补零且 mask 关闭。训练标签的 `remaining_actions` 不能成为部署时额外输入。A0 同样用动作有效 mask，但不依赖未来有效性。

## 5. 分组训练与权重继承

### 5.1 A0：W0 → 纯动作 Student → 500 epoch → 推理

1. 从 W0 恢复公共 Qwen 语言/视觉参数、DINO、旧 Query 和原动作头。
2. 直接构造 current-only Student。新建 `Q_L=[8,2048]`、`Q_G=[8,2048]`、4 个 Plan、角色投影、Goal→Plan / Local→Action 条件 attention、逐层残差 scale，以及第 9–16 层 RelevanceGate。
3. 新模块沿用完整模型对应初始化规则；例如 Query / Plan 的 std=0.02、相关新增 scale 初值 0.01。没有 W0 来源的权重应进入明确的新增参数白名单，不能用宽松加载掩盖公共参数漏载。
4. 只读取当前观测、指令、动作和动作 mask，不建立未来 Teacher，不运行 Future Resampler，不加载任何本方法的 Stage-I / Stage-II 权重。
5. 优化 `L_A0=L_flow`。语言骨干、旧 Query、新角色 Query、gate 和参与动作前向的可训练动作模块均从首次更新开始训练；视觉与 DINO 保持冻结。
6. 按第 7 节定义完整训练 500 epoch。主结果使用 epoch 500 的最终权重，先 quick 再 full 推理。离线 best-flow 权重可附加报告，但不能替代用户指定的 epoch 500 结果。

现有 `make_stage_two` 要求已训练的 posterior Teacher；后续实现必须新增合法的 W0→prior 初始化路径，不能伪造 Stage-I checkpoint 绕过校验，也不能先构造完整 Teacher 再偷偷继承其训练参数。

A0 保留的 Local / Goal 名称仅表示结构位置，真实未来语义不由动作 loss 保证。其含义是“不使用本方法新增的显式未来后验训练和蒸馏”，不对 W0 的全部上游训练历史作额外断言。

### 5.2 A1：Local-only Teacher → Local-only Student

**Stage I**

- 从同一 W0 独立初始化，不从完整双尺度 Teacher 删分支。
- 只编码 `t+8` 两视角，训练 16 槽 Local Resampler 和对应后验策略。
- 保留原 Current / DINO base memory、4 个 Plan 和原 DiT；Local→Action 在全部 16 个 DiT block **之前**注入。
- 不存在 Goal Resampler、Goal horizon embedding、Goal 角色条件或 Goal→Local gate；没有 Goal→Plan 注入。
- 损失只有后验动作 flow loss。阶段结束，固定本组 Teacher checkpoint，记录 epoch、step、源 W0 和文件哈希。

**Stage II**

- 从本组 Local-only Teacher 复制公共网络、旧 Query、Plan 和本组动作头参数到独立 Student；不把 Teacher Resampler query 直接当成 Student query。
- 新增 16 槽 Local Student Query，按统一规则初始化；冻结独立 Teacher。
- Teacher 只提供第 4/8/12/16 层 Local 目标，Student 只看当前输入。
- 优化 `L_A1=L_flow+D_L`，Local 权重为 1；推理只部署该 Student。
- Local 动作条件为 `[B,16,1024]`，不额外添加 Local→Plan 路径。4 个 Plan 仍保留在原动作计算中。

### 5.3 A2：Goal-only Teacher → Goal-only Student

**Stage I**

- 从 W0 独立初始化，仅用五个指定远端主视角，经 16 槽 Goal Resampler 学后验动作策略。
- 保留有序 horizon embedding、Current / DINO、4 个 Plan、原 DiT；不保留 Local Resampler、Local view embedding 或 Local→Action 注入。
- Goal→Plan 仅在 DiT 第 1、5、9、13 层的原 block 之前发生，再由原 Plan / Action self-attention 影响动作。
- 只优化后验 flow loss；固定本组 Teacher checkpoint。

**Stage II**

- 继承本组 Teacher 的公共网络和本组动作头，新建 16 槽 Goal Student Query；Teacher 独立冻结。
- 优化 `L_A2=L_flow+D_G`，只做第 4/8/12/16 层 Goal 蒸馏。
- Goal 条件为 `[B,16,1024]`。没有 Local Query、Local loss 和 RelevanceGate。
- 推理沿用 Goal→Plan→Action，不增加 Goal→Action 直连，不把四层 Goal 注入频率增加为每层。

### 5.4 A3 与 A4：共用完整 Teacher，配对初始化 Student

**共同 Stage I**

从 W0 按完整 idea 训练 Local 8 槽 + Goal 8 槽的后验 Teacher，使用完整未来采样、原生角色隔离 mask、双路径动作条件，只有后验 flow loss。Stage I 没有 Student Query 和 Student RelevanceGate。

**配对初始化约定**

从同一固定 Teacher 生成一份尚未进行 Stage-II 更新的公共初始化快照，保存公共权重、两组 Query 和相关 RNG 信息。A3/A4 使用完全相同的公共参数、角色 Query、动作模块初值。A4 再按既定规则启用自己的 gate；A3 从第一次前向开始无 gate。

仅设置相同 seed 不足以保证配对初始化，因为模块构造顺序和 gate 的随机初始化会改变 RNG 消耗。应对公共参数逐项比较，不能从已训练 A4 的权重构造 A3。

**A3 Stage II**

- 双角色均为 8 槽，Teacher 冻结，loss 为 `flow+D_L+D_G`。
- 第 1–16 层都没有额外 Goal→Local 更新，原生 attention mask 继续隔离角色。
- 建议不实例化 gate；如为兼容 checkpoint 保留其参数，必须证明它不参与前向、不进入有效 optimizer 参数集合，并单独报告有效参数量。
- 推理沿用训练时的无交互结构。

**A4 Stage II**

- 双角色均为 8 槽，loss 同 A3。
- 在 VLM 第 9–16 层 block 后使用原 RelevanceGate，Goal 单向更新 Local。
- gate、message attention、逐层 scale 与 Student 其他可训练模块共同训练。
- 推理直接使用普通 Stage-II 策略，不加载 Stage-III GateActor，不进行在线更新；概念上对应外部倍率 eta=1。

A3/A4 的 Teacher、训练 anchors 顺序、Student 公共初始化、预算、LR、验证 anchors 与推理协议均需配对固定。

## 6. 蒸馏、冻结和动作路径验收规则

### 6.1 统一的蒸馏定义

对存在的角色 R，四层集合为 `S={4,8,12,16}`：

```text
D_R = mean_over_4_layers(D_R_layer)

D_R_layer =
  mean(1 - dot(normalize(Z_student), normalize(Z_teacher)))
  + mean((Gram(normalize(Z_student)) - Gram(normalize(Z_teacher)))²)

L_A1 = flow + D_L
L_A2 = flow + D_G
L_A3 = L_A4 = flow + D_L + D_G
```

通道归一化和损失按原实现使用适当浮点精度，Teacher 目标 detach。Gram 沿槽轴形成 `K_R×K_R` 矩阵；单尺度是 16×16，双尺度各为 8×8。所有槽和 Gram 元素按 mean 聚合，四层也取 mean，`lambda_local=lambda_goal=gram_alpha=1`。

单尺度仅保留一项角色蒸馏，不将其权重乘 2，不用 sum 补偿槽数。无有效目标时返回有限、可参与训练汇总的零损失，不能产生 NaN。统计有效样本数量，避免损失均值口径随缺失比例改变而不被察觉。

取层位置：

| 层位 | A1 / A2 / A3 | A4 |
|---|---|---|
| 4、8 | 对应 block 输出 | 对应 block 输出 |
| 12 | 对应 block 输出 | 该层 gate 后的角色输出 |
| 16 | 最后 block 后经最终 RMSNorm | 最后 block、gate、最终 RMSNorm 后 |

Teacher 相应第 16 层也取最终 norm 后的目标。不能比较 Student norm 前与 Teacher norm 后表示。

### 6.2 冻结和复制

Stage I 更新语言骨干、旧 Query、启用的 Future Resampler 和动作模块，冻结 Qwen 视觉 / merger 与 DINO。Stage II Teacher 必须是独立模型，`eval + no_grad + requires_grad=false`；Student 复制可继承参数后，Teacher 可移除不再使用的动作计算，以减少开销，但不能影响已复制的 Student。

Stage II 只计算 Student 动作 flow；Teacher 提供表征目标，不额外跑一遍 Teacher 动作 loss。验证 Teacher 参数和相关 buffer 不随 Student 更新变化，Teacher / Student 不共享会被 optimizer 修改的参数存储。

### 6.3 动作条件保持原位置

- Goal→Plan：DiT 第 1、5、9、13 层，各在原 block 前。
- Local→Action：全部 16 层，各在原 block 前。
- A1 只保留第二条，A2 只保留第一条，A0/A3/A4 两条均保留。
- 不以“删除某角色”为由删除 Plan、改变原 base memory、自注意力、动作编码器、Euler 步数或执行 chunk。
- 不以原实现的 `bypass_roles` 代替单尺度开关；该路径会影响 Plan 等行为，不是 A1/A2 的定义。

## 7. 500 epoch 的严格定义与公平预算

### 7.1 固定数据划分

沿用历史任务分组、episode 级划分方案，首次固定 `split_seed=2026`、`val_ratio=0.05`，保存实际 episode 清单和 anchor manifest。

| 划分 | episode 数 | anchor / frame 数 |
|---|---:|---:|
| Train | 1614 | 260726 |
| Validation | 79 | 12739 |
| Total | 1693 | 273465 |

以上数量来自现有数据审计；正式启动时应根据固定 manifest 重算并校验。数据变更时不能继续沿用旧的 epoch/step 数而不更新记录。多训练种子实验应固定这份划分，不把模型 seed 改动同时变成数据划分改动。

**1 epoch = 完整遍历 260726 个训练 anchors 一次。** 每轮重新 shuffle，但每个 anchor 在该轮恰好出现一次；不按 episode 数、视频数、rollout 数或 flow 噪声重复次数计 epoch。

### 7.2 A0 的 500 epoch 计数

按历史单进程配置作为新实验建议：micro-batch=2，gradient accumulation=16，通常有效 batch=32；`drop_last=false`，每个 epoch 结束时 flush 剩余梯度。

```text
每 epoch micro-batches = 260726 / 2 = 130363
130363 = 8147 × 16 + 11

每 epoch：
  8147 次完整累积更新，每次 32 个 anchors
  1 次尾部更新，11 个 micro-batches，共 22 个 anchors
  总计 8148 次 optimizer 更新

500 epoch：
  原始 anchors 使用量 = 260726 × 500 = 130363000
  optimizer 更新量  = 8148 × 500   = 4074000
  flow B→2B 后的噪声实例数约为 260726000，不计成额外 epoch
```

尾部 11 个 micro-batches 的 loss 应按本次实际累积量归一化，不能仍无条件除以 16。若出现不同大小 micro-batch，还需按样本数正确加权。结束 epoch 时清空梯度，记录实际样本计数。

这套方案不丢样本、不为凑 batch 重复尾部样本。若改为 DDP、改变 batch 或改为跨 epoch 连续累积，必须重新推导更新次数并记录 sampler 的补齐/重复规则；不能直接复用 4074000。

现有训练脚本按 `--steps` 控制、没有上述 epoch 完成与尾部 flush 语义。**设置 `--steps 500` 绝不是 500 epoch；机械地设置 `--steps 4074000` 也不能替代 epoch/sampler 逻辑改造。**

### 7.3 A1–A4 的预算如何与 A0 对齐

500 epoch 是用户指定的 A0 长训练预算。历史完整方法的 Teacher 来源为 2000 step、Student 为 4000 step；按有效 batch=32，其所选权重对应的两阶段名义训练量合计约 192000 anchors。A0 的 130363000 anchors 约为其 **679 倍**。直接把两者并列，不能声称离线训练预算一致。

后续正式实验可选以下协议；目前只定义选项，不擅自确认 A1–A4 的 epoch 数：

| 协议 | A0 | A1 / A2 / A3 / A4 | 用途 |
|---|---|---|---|
| 以用户指定 500 epoch 为主预算 | 500 epoch | Stage I 为 E_I、Stage II 为 E_II，且 **E_I+E_II=500**；三种 Teacher 用相同 E_I，四种 Student 用相同 E_II | 预算一致的主要 A 组对照 |
| 另建历史量级的预算匹配实验 | 单独训练与两阶段合计相同的 anchors / 更新数，例如约 6000 steps | 各自匹配 Stage I 2000 + Stage II 4000 steps，并明确 LR 与 warm-start 规则 | 较低预算的独立 A 组对照；A0-500epoch 另列长训练结果 |

若采用第一种，可把 `E_I=167, E_II=333` 作为接近历史 1:2 阶段比例的**示例**，但它不是用户已指定参数，也不是经过验证的最佳分配。阶段比例应在正式启动前写入实验计划，不能看最终成功率后反复选择。

第二种也不意味着可以直接取 A0 长 schedule 的第 6000 步与短 schedule 方法等价比较：学习率轨迹不同，必须单独定义匹配实验。历史 A4 若包含 warm250，其他对应 Student 也需要从各自未混用的权重复现同样规则；更清晰的推荐是整组按统一的新协议连续训练，历史结果只作参照。

正式报告至少列出：各阶段 anchors 数、optimizer 更新数、effective batch、总 GPU 时间 / 峰值显存、可训练参数量及推理延迟。A3/A4 可共享一次 Teacher 训练，但每个方法的算法预算都包含这段 Stage I；不能因为计算复用就把某项的 Teacher 成本记为零。

### 7.4 建议的优化与保存协议

以下为继承现有配置并适配 epoch 的执行建议；最终数值应写入各 run 的配置。

| 项目 | 建议 |
|---|---|
| 优化器 | 沿用 MasterAdamW，FP32 master 参数；betas=(0.9,0.95)，eps=1e-8 |
| 基础学习率 / weight decay | 1e-5 / 1e-8 |
| 梯度裁剪 | 1.0；非有限梯度明确记录并处理 |
| 精度 | 沿用 BF16 autocast，归一化 / loss 的精度遵循原实现 |
| A0 optimizer | 从 W0 初始化后连续训练 500 epoch，重启应完整 resume，不在中途任意清空 Adam 状态 |
| 阶段切换 | A1–A4 从 Stage I 转 Stage II 时新建 Student optimizer，四组使用相同规则 |
| warmup | 建议每个训练阶段 500 **optimizer updates**；不是 500 epoch |
| LR schedule | warmup 后 cosine 到基础 LR 的 5%；按该阶段实际总更新数生成 |
| A0 schedule | 总更新数为上述 4074000；不能把 10000-step cosine 曲线未经修改延伸数百万步 |
| 离线验证 | 初始、每个 epoch 结束、阶段结束；同一固定 128 个跨任务验证 anchors，固定 flow 噪声/时间随机协议 |
| checkpoint | 每个 epoch 结束保存可恢复状态，保留最近两份、best-flow 和固定的阶段终点；可另设按墙钟时间的恢复快照 |
| 日志 | 每次 optimizer 更新记 step、epoch、已见 anchors、LR、flow / 蒸馏分项、梯度范数和耗时 |

验证若扩展到全部 12739 anchors，应全组同步更改并固定，不能某组 128、另一组全量。按固定验证 flow 选择的 best 可以作为附加分析；主表统一使用预先规定预算终点，A0 必须包含 epoch 500。Teacher 也按事先固定规则选择，建议以阶段终点为主，不用在线评估成绩挑 Teacher。

500 epoch 可能过拟合，文档不预设其成功率。不要为了提前得到高分而停止训练并将早期权重改名为 epoch 500。长期运行需先测实际更新耗时与磁盘容量，再估算总时间；不能在没有实测吞吐时承诺完成时长。为避免海量大权重堆积，不沿用“数百万步每 250 步永久保存”的策略。

## 8. LIBERO 推理执行协议

### 8.1 推理对象

所有 A 组部署权重都是各自的 current-only Student。推理包应包含模型配置、权重、processor、动作统计和来源元数据；服务启动时核对实验编号、角色槽数、gate 配置、epoch / step 和 checkpoint 哈希。

推理输入只包含当前主视角、当前腕部和指令。禁止依赖 Teacher、未来图像、future validity 或示范剩余长度。A0/A3/A4 的内部结构区别必须由 checkpoint 配置决定，不能统一按完整 A4 强行加载。

A0 使用 epoch 500 最终权重；A1/A2 使用各自 Stage-II 终点；A3/A4 使用共享 Teacher 后分别训练的 Stage-II 终点。

### 8.2 Quick 与 Full 的统一设置

| 项目 | 统一协议 |
|---|---|
| Suite | libero_spatial、libero_object、libero_goal、libero_10（Long），各 10 个任务 |
| Quick | 40 tasks × 10 trials = 400，使用各任务固定初始状态索引 0–9 |
| Full | 40 tasks × 50 trials = 2000，使用各任务固定初始状态索引 0–49 |
| 顺序 | A0 完成 500 epoch 后先 quick 验证运行，再 full；其他组也按相同协议 |
| 环境 seed | 沿用历史 seed=0，并保存实际初始状态标识 / reset 协议 |
| 稳定步骤 | 沿用 10 个 settling 步；与策略动作步数分开记录 |
| 最大策略动作步数 | Spatial 250；Object 300；Goal 320；Long 550 |
| 动作生成 | 每次 10 次 Euler 更新，输出有效 8×7 动作 |
| 动作执行 | 最多执行 8 步后重新观测；成功、终止或达到时限时即刻停止，不强行执行完剩余 chunk |
| 输入处理 | 与历史一致的图像方向、RGB、缩放、processor、指令和环境包装器 |
| 视频 | 可沿用每任务首个成功视频策略；没有视频不能视作任务未运行，失败同样计入分母 |

沿用历史可复现的分段动作噪声规则时，suite_index 固定依次为 Spatial / Object / Goal / Long 的 0/1/2/3：

```text
action_noise_seed =
  (suite_index + 1) × 10000000
  + task_id × 10000
  + episode_index × 100
  + chunk_index
```

所有组使用相同规则；保存每个 rollout 的实际标识，不能用“启动时设了相同 seed”替代实际初始状态一致性。模型初始化种子与环境 / 动作噪声种子分开记录。

动作反归一化沿用原 SFT 统计与已验证推理实现：前六维还原并按原服务的合法范围处理，夹爪从模型标签空间转换到环境动作空间（原实现的 2g−1 规则）。不要训练阶段和推理阶段各做一次重复归一化 / 反归一化。

Quick 的 0–9 是 Full 0–49 的子集。若只用于检查运行、不据此选模型，可以沿用此协议；若用 quick 成功率调参或挑 checkpoint，必须单独建立不重叠的开发 / 最终评估 manifest，并如实调整分母，不能再声称原 full 2000 是独立、未参与选择的测试结果。

### 8.3 指标与结果判定

每组至少报告四个 suite 的成功率、40 个任务逐任务成功率、总成功数 / 总 rollout 数和总成功率。每任务等额 trials 时，任务平均成功率与总成功率一致；如有缺失，不应悄悄改变任务权重。

基础设施错误和真实策略失败分开记录。因基础设施问题重跑必须保持同一 episode 标识、种子和状态，并保留错误记录；不能只保留成功重跑结果。没有完成全部任务的结果标记为 incomplete。

成功执行步数可以补充报告，但不能只比较各模型自己的成功子集来宣称速度收益；配对比较应列出“两模型都成功”的共同 episode 数及步数差。无成功样本时写 N/A。

先用训练 seed=2026 跑通五组；正式统计建议至少 3 个模型训练种子，固定数据划分和评估 manifest，报告跨训练种子的均值与标准差。重复同一模型的 rollout 不能冒充多个训练种子。

## 9. 后续代码修改的具体位置与要求

**本节是后续开发清单，本次没有实施。** 当前代码不能仅改几个配置值就完整运行 A0–A3；也没有已验证的 `--epochs`、`--local-only` 等开箱即用参数。本文不提供会被误认为已经可执行的新 CLI 命令。

建议在同一套公开结构配置中表达五组，避免复制五套模型实现后逐渐漂移。新增配置至少要能表达：实验编号、训练模式、K_L / K_G、是否读取各类未来、是否启用额外交互、蒸馏角色 / 层位、epoch 预算、固定数据划分与 checkpoint 来源。字段名称在实现时确定，不能把下表当作现有接口。

| 文件 / 模块 | 已有约束与后续需支持的内容 |
|---|---|
| [goal_local/model.py](../../goal_local/model.py) | 当前配置校验固定 slots=8、默认双角色 posterior。需要分别表达 Local / Goal 槽数、合法 W0→prior A0 初始化、单尺度 Teacher / Student、可禁用 gate、动态角色切片，以及按角色计算蒸馏。 |
| [goal_local/roles.py](../../goal_local/roles.py) | 原 mask / Resampler 假定两组等长 8 槽。需要支持 16/0、0/16、8/8；保留正确的 native 隔离、位置规则、空角色处理与未来有效帧聚合。 |
| [goal_local/data.py](../../goal_local/data.py) | 目前取样逻辑面向完整未来。需要按实验只解码所需未来；A0 无未来字段依赖；仍保留全部相同 anchors、尾部样本、帧时间映射及动作 mask。 |
| [goal_local/action.py](../../goal_local/action.py) | 原条件接口面向 Goal / Local 两路。需支持单路缺省且不对空 K/V 执行 attention，保持 Plan / base memory / 注入位置不变；不能用 bypass_roles 直接替代。 |
| [goal_local/train.py](../../goal_local/train.py) | 现有 stage 选项为 I/II，按 steps 运行；需支持 A0 action-only、真实 epoch / sampler 状态、尾部梯度 flush、按实际累积量归一化、动态未来统计和完整恢复。 |
| [goal_local/optim.py](../../goal_local/optim.py) | 沿用数值方案，检查长 schedule、FP32 master 与 optimizer 恢复是否完整，不为单个消融更换优化器。 |
| [goal_local/predict.py](../../goal_local/predict.py) | 根据保存的角色 / gate 配置构造正确 Student；推理输入不依赖任何真实未来。 |
| [goal_local/eval_common.py](../../goal_local/eval_common.py) | 当前常量绑定历史训练目录、student_eval_snapshots、libero_40x10 和缓存位置。需参数化实验 checkpoint / 输出目录，不能把新结果写进历史 step_004000 结果。 |
| [goal_local/eval_server.py](../../goal_local/eval_server.py) | 当前加载受 snapshot 根目录、完成标记及模型缓存约束。需明确允许新的实验快照目录、验证完整性；切换角色配置时正确重建模型，不能复用不兼容缓存或忽略 shape 错误。 |
| [eval_task.py](../../goal_local/eval_task.py)、[eval_full.py](../../goal_local/eval_full.py)、[eval_queue.py](../../goal_local/eval_queue.py) | 同步透传实验目录、模型来源和 quick / full 协议，按各自输出归档、正确汇总分母。 |

配置检查应拒绝非法组合，例如 A1 启用 Goal 蒸馏、A2 启用 Local 注入、A3 开启 gate、A0 加载 Teacher 或未来监督。旧完整 A4 的配置与权重应仍可按原语义加载，以便做固定输入 / 固定噪声回归对照。

## 10. 目录、checkpoint 和记录规范

以下是**后续建议布局**；本次仅写入本文件，不创建占位结果、不复制权重：

```text
ablation/aba1/
  guide.md
  protocol/
    experiment_plan.json       # 分组定义、阶段预算、LR、种子
    offline_split.json         # 固定 episode 划分及数据版本
    validation_anchors.json
    eval_protocol.json         # suites、初态、噪声、时限、指标
  A0/libero/seed_2026/<run_id>/
    config.json
    checkpoint_refs.json
    action_only/
      checkpoints/
      logs/
    evaluation/quick_40x10/
    evaluation/full_40x50/
  A1/libero/seed_2026/<run_id>/
    stage_I/
    stage_II/
    evaluation/
  A2/libero/seed_2026/<run_id>/
    stage_I/
    stage_II/
    evaluation/
  A3/libero/seed_2026/<run_id>/
    teacher_source.json        # 指向与 A4 相同的完整 Teacher
    stage_II/
    evaluation/
  A4/libero/seed_2026/<run_id>/
    stage_I/
    stage_II/
    evaluation/
```

已有目录或其他文档如存在，保持其内容，不覆盖已有实验。大型权重可以存至指定存储目录，但实验目录必须保留准确路径、哈希和完整来源；不要只记“best”或“latest”。

每个 run 必须保存：

1. 配置、代码版本 / 修改摘要、依赖版本、数据 manifest 哈希、processor 和动作统计来源。
2. W0 的精确文件来源；Teacher 的实验编号、阶段、epoch、step、哈希；Student 初始化复制规则及新增参数清单。
3. 公共参数加载检查、训练 / 冻结参数列表、有效参数量、角色 shape、训练 / 验证 anchors 数量。
4. loss 分项、各角色有效样本数、LR、梯度范数、epoch、global step、阶段内 step、实际已见 anchors、耗时和资源统计。
5. 可恢复 checkpoint：模型、FP32 master 参数、optimizer、scheduler、epoch / batch 游标、sampler / RNG 状态；完成写入后才生成完成标记。
6. 每个评估 episode 的任务、初态、seed、成功/失败、动作步数、错误状态、checkpoint 哈希及协议版本。

模型权重恢复不等于完整训练恢复。若只有权重而没有 optimizer / RNG / sampler，必须标记 warm-start，并统计可能重复消费的 anchors；不能称为严格连续的 500 epoch。

## 11. 正式长训练前的验收清单

后续完成代码修改后，先用真实 batch 验证完整一次 optimizer 更新、保存/加载和少量 rollout，再进入正式预算。只看到 loss 有限还不足以验收。

- [ ] A0 公共参数来自 W0，新增参数加载缺失项完全符合白名单；没有 Stage-I / II 参数混入，无 Teacher 或未来字段依赖。
- [ ] 五组保留旧 8+8 Query、总计 16 个新角色槽、4 个 Plan、有效 8×7 动作；hidden、DiT、Euler 步数不变。
- [ ] A1/A2 的 Teacher Resampler 和 Student Query 都是 16 槽；单尺度数据解码只读对应未来，不运行已删除角色。
- [ ] 全有效、部分 Goal 有效、未来全缺失、动作不足 8 步四种 batch 均无越界 / NaN，mask 行为符合第 4.3 节。
- [ ] Stage I 启用的 Resampler 确实有梯度；Stage II 新角色 Query 有梯度；A0 的 Query、gate 和动作条件可从 flow 获得梯度。
- [ ] A3 的 Goal 表示受扰动时，VLM Local 表示不通过前向通路改变；最终动作可以因 Goal→Plan 改变。
- [ ] A3/A4 的公共 Stage-II 初值逐项相同，Teacher 同源；A4 gate 只在规定层生效。
- [ ] Teacher 参数和 buffer 在 Student 更新后保持不变，模型间不存在可训练参数共享。
- [ ] 蒸馏形状、四层均值、cosine / Gram 均值、final norm 和有效样本归一化正确。
- [ ] 1 epoch 的 sampler 日志恰好覆盖固定训练 anchors 一次；尾部 11 个 micro-batches 正确更新；500 epoch 总数可由日志审计。
- [ ] 中断恢复后 optimizer、LR、RNG、采样游标与未中断基准一致；不能重复一个 epoch 后不记账。
- [ ] 各组保存/加载后，在固定输入和动作噪声下保持对应行为；旧完整 A4 的兼容性回归记录精度与误差容限。
- [ ] 去掉 batch 中全部未来相关字段，部署 Student 仍可独立生成动作；无 Stage-III Actor 或在线更新路径。
- [ ] 推理服务回报正确实验配置 / checkpoint 哈希，切换 A1/A2/A3 时没有沿用 A4 缓存。
- [ ] 评估初始状态、settling、chunk 中止、步数上限、分母和错误重跑规则均正确。

这里的检查属于后续开发任务，本次未执行模型训练测试或 rollout。

## 12. 后续执行顺序与结果表

建议按以下依赖顺序推进：

1. 固定数据 / 推理 manifest、A0-500epoch 协议，并在启动 A1–A4 前确定 E_I / E_II 和 LR 轨迹；记录选用第 7.3 节的哪种预算协议。
2. 完成第 9 节所列代码改造并通过第 11 节验收。
3. A0 从 W0 连续训练满 500 epoch，保存最终权重，依次完成 quick / full 推理。
4. 从 W0 分别训练 A1 Local-only、A2 Goal-only、A3/A4 共用的完整 Teacher，共三种 Teacher；固定各自阶段终点。
5. A1/A2 各用自己的 Teacher 初始化 Student；A3/A4 从共享 Teacher 配对初始化，按相同 Stage-II 预算训练。
6. 所有 Student 以相同协议推理、完整汇总；再按预定方案扩展训练种子，保留每次运行的来源。
7. 论文或报告将新的预算匹配实验、A0 长训练补充实验、历史结果分开标注。A4 与后续 B0/C0 仅在 checkpoint 和协议相同时复用，复用结果不是独立样本。

尚未运行的新实验表：

| 组别 | Teacher 预算 / 来源 | Student 预算 | Quick 400 | Full 2000 | 状态 |
|---|---|---|---|---|---|
| A0 | 无；W0 直接初始化 | **500 epoch** | — | — | 仅方案，未运行 |
| A1 | 独立 Local-only，E_I 待锁定 | E_II 待锁定 | — | — | 仅方案，未运行 |
| A2 | 独立 Goal-only，E_I 待锁定 | E_II 待锁定 | — | — | 仅方案，未运行 |
| A3 | 与 A4 共用完整 Teacher | E_II 待锁定，无 gate | — | — | 仅方案，未运行 |
| A4 | 新协议完整 Teacher；或明确注明历史协议 | E_II 待锁定，有 gate | — | — | 仅方案，未运行 |

结果解释应与实验实际控制的变量一致：A3 相对 A1/A2 的优势支持当前近端 / 远端未来设计的互补性；A4 相对 A3 的优势支持显式 Goal→Local 交互；A4 相对 A0 的优势支持完整未来学习流程。若实际结果不呈现这些关系，应如实报告逐任务差异和不确定性，不预设 A4 必然最高，也不把历史 98.5% 填入尚未运行的新实验。
