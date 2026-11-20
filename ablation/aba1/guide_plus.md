# LIBERO-Plus A 组消融执行方案：A0–A4

编写日期：2026-10-19。状态：**设计与后续实现规格；本次只写文档，不改代码、不启动训练或推理。**

依据：[reference/abalation.md](../../reference/abalation.md) §1.1、§2、§3、§7、§8，结合实际 Plus 数据、训练入口、checkpoint 和评测协议。另参考 [Stage I/II](../../reference/stage_I_II.md) 与[三阶段说明](../../reference/stage_I_II_III.md)。文中的新增配置字段是后续实现语义，不能当成现有 CLI 已支持的参数。

## 1. 对你的理解的判断与比较目标

你的 A0–A3 理解整体准确，需要明确以下边界。

| 组别 | 初始化与训练 | Local / Goal 槽数 | 动作条件 | Student VLM 额外交互 |
|---|---|---:|---|---|
| A0 | 原始 SFT W0 → 直接 current-only 动作训练，无本方法 Teacher／后验训练／蒸馏 | 8 / 8 | 保留双路径 | 保留第 9–16 层 RelevanceGate，由动作损失训练 |
| A1 | W0 → Local-only Teacher → Local-only Student | 16 / 0 | 仅 Local→Action | 无 |
| A2 | W0 → Goal-only Teacher → Goal-only Student | 0 / 16 | 仅 Goal→Plan→Action | 无 |
| A3 | 完整双尺度 Teacher → 从训练开始无额外交互的双尺度 Student | 8 / 8 | 双路径 | 从第一次前向开始关闭 |
| A4 | 完整双尺度 Teacher → 完整 Stage-II Student | 8 / 8 | 双路径 | 第 9–16 层训练 RelevanceGate |

**A 组全部不做 Stage III，不加载 GateActor。A4 是完整 Stage II，不是经过 RL 的模型。**

必须补充的理解：

- A0 保留两组 query、4 个 Plan、双角色投影、双路径动作条件和深层 gate。它只取消本方法的显式未来学习，不取消部署结构。不能从训练好的 Teacher／Student 出发再将蒸馏 loss 置零；那不是 A0。
- A1/A2 翻倍的是 **token 槽数 8→16**，Teacher Resampler 输出和 Student Query 都同步改为 `[16,2048]`；hidden 维度不翻倍，动作长度仍为 8。
- A1 的真实未来是 `t+H` 的主／腕双视角。A2 是固定五个有序远端主视角 `{t+3H,t+5H,t+7H,t+9H,t+11H}`，不是任意一个 `t+nH`，也不是单张终点图。
- A3 可复用完整 Stage-I Teacher，因为完整 Teacher 本来就没有 Student 的额外 RelevanceGate；必须重新训练无交互 Student。不能用 A4 推理时 `eta=0` 替代。
- 单尺度组仍保留原 DiT 和 4 个 Plan。A1 无 Goal 时也不删除 Plan；A2 不把 Goal 注入频率改成每层以补偿 Local 缺失。

主要比较：**A4 vs A0** 检验完整显式未来学习方案；**A3 vs A1/A2** 是双尺度互补性的直接证据；**A4 vs A3** 隔离监督训练的跨尺度交互收益。A1 vs A2 补充近／远端设计的逐任务差异。

16 个角色 token 相同不代表总参数、FLOPs、未来信息量完全相同：单／双 Resampler、条件路径、未来图像数量、attention 可见性仍有差异。结论应限定为“当前近端／远端表征设计的互补性”，同时报告这些成本。

## 2. 实际路径、数据与可复用基线

### 2.1 路径

```text
R = /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
A = R/ablation/aba1
W0 = /mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt
STATS = /mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/dataset_statistics.json
DATA = /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_without_subgoal/libero-plus/data/indexed_rlds
RAW_DATA = /mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero-plus/sft/rlds_mix
BENCH = /mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero-plus/inference/LIBERO-plus
T_LG = R/libero-plus/runs/stage_I_20260923/retained_checkpoints/step_005000
S_LG_INIT = R/libero-plus/runs/stage_II_20260925/step_000000
S_LG = R/libero-plus/runs/stage_II_20260925/step_004000
FULL_EVAL = R/libero-plus/evaluation/stage_II_20260925/full_manifest.json
```

W0 使用实际大写 `Checkpoints`。本次已核实三个模型目录的 `model.pt` 均存在。`S_LG_INIT` 是尚未更新的学生初始化快照，可用于 A3/A4 配对；它不是已经训练的 A4。

复用的是 `LaWAM_without_subgoal` 中的数据，**不继承该工程的模型或 state token**。DATA 来自 RAW_DATA，保留原始 JPEG 字节。使用完整 1024 分片 manifest，不能误读 `active_window.json` 将训练缩为一个数据窗口。历史数据索引位于 `R/libero-plus/cache/index`，后续只读复用或另建缓存，不能覆盖历史索引。

### 2.2 默认采用历史 Plus 划分，固定终点选模

| 项目 | 设置 |
|---|---|
| 全部数据 | 14347 episodes，2238036 帧，1024 分片 |
| 划分 | 按指令分层、episode ID 划分；split seed=2026；验证比例 5% |
| 训练 | 13630 episodes，2125407 个 frame anchors |
| 验证 | 717 episodes，112629 个 frame anchors |
| 采样 | 同一训练 anchor 集、相同 sampler 与 shuffle 规则；同阶段组间配对顺序 |
| 当前输入 | 主／腕 RGB 256×256＋语言指令，`use_state=false` |
| 动作 | STATS 归一化，有效 8×7，内部 padded 为 50×32 |
| 频率 | 按转换 manifest 声明为 20 Hz；无逐帧时间戳，不声称重新核实采集频率 |

历史审计发现：717 条验证 episode 均有同指令、同完整动作序列的训练版本，图像扰动可能不同。因此离线验证只用于拟合／数值诊断，**不能称为独立原始轨迹泛化验证**。

本轮默认沿用该划分以隔离架构变量，主模型固定为预算终点。若改为按原始轨迹及其扰动版本分组的新划分，另建协议版本，重训对应单尺度和双尺度 Teacher 及 A0–A4，不能把旧 A4 混入新协议。真实来源 ID 为主、动作序列哈希为辅助进行去重；不是简单重新随机划分。

### 2.3 A4 历史成绩仅作明确标注的参考

- 10030 任务全量：**8683/10030＝86.5703%**。
- 根据旧 step500 类别成绩非均匀抽样的 8380 子集：**7510/8380＝89.6181%**。

本方案主表采用 10030 全量。89.62% 不等于官方全量 Total，不能与新组全量成绩直接比较。Stage III 第一轮 89.27% 不属于 A4。

## 3. 预算、学习率和 checkpoint 选择

### 3.1 先明确“1000 epoch”的单位

原始请求写的是 **1000 epoch**。本轮已提出单位澄清；截至文档写入尚未收到确认，**未自动将它改成 1000 optimizer steps**。下列建议预实验与正式预算不是对原始单位的替换。

按当前训练 anchors 和有效 batch=32：

```text
1 nominal epoch ≈ 2,125,407 / 32 ≈ 66,419 次 optimizer 更新
1000 完整 epoch ≈ 66,418,969 次更新（约 6642 万次）
1000 optimizer steps = 32,000 anchors ≈ 0.0151 nominal epoch
9000 optimizer steps = 288,000 anchors ≈ 0.1355 nominal epoch
```

实际 epoch 更新数还受分布式 sampler、drop_last、跨 epoch 累积影响，应记录实测 anchor 数。flow repeat=2 是同一 anchor 的噪声／时间重复，不把数据 epoch 翻倍。

**建议的快速预实验**为 `A0_pilot_step1000`：W0 初始化后更新 1000 次，保存完整恢复状态，再执行推理，结果单列。它用于检查流程，不替代正式等预算 A0。

若确认要训练 **1000 个完整 epoch**，则命名 `A0_long1000epochs`，作为独立长训练扩展，不能与现有 5000＋4000 更新的 A4 冒充等预算对照。它需要单独锁定长训练 LR 曲线、保存频率和实际样本数，不沿用下面的 9000-step 调度。本文件没有启动这项训练。

### 3.2 建议的正式 A 组预算：与现有 Plus 继承链匹配

| 组 | 第一段／Stage I | 第二段／Stage II | 总 anchor 预算 | 正式权重 |
|---|---|---|---:|---|
| A0 | action-only 5000 更新 | action-only 4000 更新 | 288000 | 累计 step9000 |
| A1 | Local Teacher 5000 | Local Student 4000 | 288000 | Student step4000 |
| A2 | Goal Teacher 5000 | Goal Student 4000 | 288000 | Student step4000 |
| A3 | 双尺度 Teacher 5000，可复用 | 无交互 Student 4000 | 288000，含上游 Teacher 成本 | Student step4000 |
| A4 | 相同双尺度 Teacher 5000 | 完整 Student 4000，可条件复用 | 288000，含上游 Teacher 成本 | Student step4000 |

A0 分两段只为匹配更新和调度，**不是后验 Stage I 或蒸馏 Stage II**。所有新增 query、Plan、条件投影／attention、gate 从第一个动作训练 step 开始更新，段间保留模型参数。

历史 Teacher step5000 后额外运行到约5217的更新不进入继承权重；Stage II 磁盘故障后重做的未保存计算不增加最终有效 step。预算匹配继承链的有效更新，额外试运行、重算及实际 GPU 小时另报，不隐去训练成本。

### 3.3 必须保留历史学习率曲线，而非只匹配最后的 step 数

| 项目 | 第一段／Stage I | 第二段／Stage II |
|---|---|---|
| 基础 LR | 1e-5 | 1e-5 |
| scheduler 总长 | **16000 updates** | **10000 updates** |
| warmup | 500 | 500 |
| 实际停止 | 5000 | 4000 |
| 调度 | 原 warmup＋cosine，最低倍率0.05 | 同左 |
| optimizer | AdamW、FP32 master、betas=(0.9,0.95)、eps=1e-8 | 同左 |
| weight decay / clip | 1e-8 / 1.0 | 同左 |
| 有效 batch / flow repeat | 32 / 2 | 32 / 2 |

执行规则：

1. 新入口分开表达 `schedule_total_steps` 与 `stop_after_updates`。现有 `--steps` 同时控制调度和循环，不能直接改成5000／4000／1000并声称复现历史 LR。还应对比真实 LR trace、首步调用顺序、梯度裁剪位置和 FP32 master 更新。
2. A1–A4 的 Stage II 新建 optimizer、Stage-II RNG 和 sampler，不继承 Teacher 优化器。故障恢复则完整继承 optimizer／scheduler／RNG／采样游标，二者不同。
3. A0 在累计step5000处保留全部模型权重，按第二段规则重建 optimizer、scheduler、RNG／sampler；不加载 Teacher，不重置角色参数。两段均仅 action loss，不能宣称训练历程或 FLOPs 与完整方法完全相同。
4. 若采用step1000预实验并续训正式A0，从一开始就用第一段16000-step调度，评测后从完整恢复状态续训。不能将短调度预实验直接接入并冒充相同协议。
5. Plus 不采用普通 LIBERO 的 `warm250` 特殊交接。不要复制普通 LIBERO 的step250优化器重置，也无须重演磁盘故障。
6. 可沿用 Stage I：2 ranks×batch2×accumulation8；Stage II：4 ranks×batch1×accumulation8。同阶段组间固定布局与采样规则，物理GPU后续安排。多个rank共享同一GPU时不能直接用重复设备NCCL，参考原CPU Gloo梯度平均实现。

Teacher每1000步保存，固定取5000；每250步诊断flow。Student每500步保存／诊断，固定取4000。A0至少保留step0、预实验1000、阶段边界5000、累计9000，并同时记录阶段步数和累计步数。

最终checkpoint在开跑前指定，不按最终benchmark成功率择优，也不以单／双尺度不可比的总loss选模。第一轮训练seed=2026；扩展2027、2028时固定split manifest、只改变训练随机性。现有代码将seed用于数据划分，后续必须拆开 `train_seed` 和 `split_seed`。每个训练seed需要对应Teacher／Student；A3/A4在同一seed内共享Teacher和公共初始化。

## 4. 共同结构、采样、损失和冻结规则

### 4.1 结构不变量

- 保留原8个Action Query和8个Flow Query的VLM输入；它们与新增16个角色token不同。
- 语言模型16层、hidden2048；VLM序列为 `[C; roles]`，每组都只追加16个角色token。
- 原生mask：C内causal，C不读角色；角色读有效C及同角色槽；双角色原生attention不互读。A0/A4另在第9–16层block后执行Goal→Local门控；其余组无该通路。
- 送入动作头前按旧query的实际位置移除其16个输出，不简单截去末尾16个token，也不删新角色。Current与DINO组成原768维base memory。
- 角色单独投影为1024维K/V，不拼入base memory。保留4个Plan、内部50个Action位置和原16层DiT；有效动作8×7。
- Goal存在时只在DiT第1/5/9/13层原block之前注入Plan；Local存在时在全部16层原block之前注入Action。其余原self-/cross-attention不变。

### 4.2 未来采样与尾部

`H=8`，Local=`t+8`双视角，Goal=`t+[24,40,56,72,88]`五张有序主视角。按声明的20Hz分别对应0.4秒和[1.2,2.0,2.8,3.6,4.4]秒。

所有组使用相同anchors，不能因某角色未来缺失筛掉尾部样本。未来不跨episode，不复制末帧伪造有效帧。Stage I只聚合有效未来；某角色全部缺失时输出有限占位并关闭该样本对应的Teacher动作注入。Stage II只跳过缺失角色的蒸馏，**Student启用的角色仍参与动作预测**，不能用未来mask控制Student。

“某角色永久移除”通过结构配置表达；“当前样本未来缺失”通过mask表达。单尺度删除的是未来分支，当前主／腕双视角始终保留。

### 4.3 损失

所有Teacher的Stage I仅用后验action flow loss。Student与A0的目标：

```text
A0: L_flow
A1: L_flow + D_Local
A2: L_flow + D_Goal
A3: L_flow + D_Local + D_Goal
A4: L_flow + D_Local + D_Goal
```

角色蒸馏层固定为4、8、12、16，四层平均。每层先沿hidden维L2归一化，再计算：

```text
D_R_layer = mean(1 - cosine(student_slot, teacher_slot))
          + mean((Gram(student_normalized) - Gram(teacher_normalized))^2)
D_R = mean over the four selected layers
lambda_Local = lambda_Goal = gram_alpha = 1
```

Teacher目标detach；只对该角色未来有效样本计算。8槽Gram为8×8，16槽为16×16，均按元素mean，不改sum，不为单尺度自动加倍权重。没有有效样本时返回可微有限零。

第4/8层取block输出；有gate的第12层取gate后；第16层取最后block／gate后再经最终RMSNorm。无gate的A1/A2/A3取相应block输出，最终norm约定相同。不同组loss项数不同，不宣称总蒸馏梯度严格匹配。

### 4.4 权重与冻结

W0的保留Qwen、DINO、旧query和原flow参数严格加载；不恢复已移除的LaWM重建链或State MLP。Qwen vision／merger和DINO全程冻结、eval；语言骨干、旧query及参与前向的动作模块更新。

Stage I的对应Resampler和view／horizon embedding可训练。Stage II Teacher完全冻结、eval、no_grad，不入optimizer，不与Student共享可训练存储；先复制公共模块与动作头给Student，再移除Teacher不再使用的动作头。

新query／Plan等按现有对应模块规则初始化，角色／gate残差尺度沿用0.01；投影和attention按其实际模块初始化规则记录，不把所有新参数统称为同一种分布。16槽Teacher不能通过复制8槽双尺度Teacher输出得到。

结构相同的新增公共模块应采用模块级RNG或初始化快照配对；“同seed”不足以保证不同构造顺序下权重一致。A3/A4尤其要逐张量核对共同初始参数。

## 5. 分组执行：从训练到推理

### 5.1 A0：W0直接动作训练

```text
W0公共网络＋原动作头
 + Goal Query[8,2048]、Local Query[8,2048]
 + Plan[4,1024]、双角色投影／双路径attention、深层RelevanceGate
 → 当前双视角＋指令 → 角色先验 → Action DiT → 仅L_flow
```

无Future Resampler、Teacher或蒸馏前向。Dataset只读当前图像、指令、动作和标签有效性；示范剩余长度仅用于训练mask，不进入策略条件。直接W0→prior需要合法新构造路径，不能伪造Stage-I checkpoint调用现有 `make_stage_two()`。

按第3节选择已明确单位的预实验；建议的正式A0训练累计9000更新，两段均action-only。评测加载其自身模型和processor／STATS，保留两组query、gate和双路径，不因没有Teacher而删除部署模块。

A0的Local／Goal名称只表示结构位置，没有本方法显式未来监督保证其时间语义；W0上游知识仍存在。结论是“未使用本方法新增未来监督”，不是“从未接触任何未来知识”。

### 5.2 A1：Local-only Teacher → Local-only Student

Stage I从W0独立构造，只读取`t+8`主／腕双视角：

```text
未来双视角 → 冻结Qwen vision → view embedding＋Local Resampler
          → F_L[B,16,2048]
[C; F_L] → 隔离VLM → Local projection[B,16,1024]
         → 全16层Local→Action → L_flow
```

没有Goal Resampler、Goal角色、Goal horizon embedding、Goal→Plan注入和RelevanceGate。保留当前双视角、DINO、旧query、4个Plan和原DiT。训练到Teacher5000。

Stage II冻结自己的T_L，复制公共权重和保留动作分支，新增 `Q_L[16,2048]`，只蒸馏Local四层，loss=`L_flow+D_Local`。训练到Student4000后部署，单角色仍为16槽。不能保留会通过bias、attention或残差影响输出的空Goal分支。

### 5.3 A2：Goal-only Teacher → Goal-only Student

Stage I从W0独立构造，只读取五个固定有序远端主视角：

```text
五帧 → 逐帧冻结Qwen vision → horizon embedding＋Goal Resampler
     → F_G[B,16,2048]
[C; F_G] → 隔离VLM → Goal projection[B,16,1024]
         → 第1/5/9/13层Goal→Plan → 原self-attention → Action → L_flow
```

没有Local Resampler、Local角色、Local→Action或RelevanceGate；当前腕部观测仍保留。Goal不增加到16帧，不改成终点单帧，不提高注入频率。训练到Teacher5000。

Stage II冻结自己的T_G，继承公共权重及保留动作分支，新增 `Q_G[16,2048]`，只蒸馏Goal四层，loss=`L_flow+D_Goal`。Student4000后部署，只通过Goal→Plan→Action注入角色条件。

### 5.4 A3：双尺度独立Student

Stage I复用T_LG5000。Stage II从该Teacher重新初始化Goal8＋Local8的Student，保留双路径与四层双角色蒸馏，**从step0起不执行额外RelevanceGate**，训练到4000。

与历史A4配对时，可读取S_LG_INIT中的公共参数和未训练query，核对teacher_step=5000及step=0，显式移除／跳过gate并记录预期键差异；也可从T_LG重建后复制相同未训练query。禁止从A4 step4000开始微调。

不能仅将可训练gate初始scale置零，因为后续它仍可建立交互。部署用真正从头训练的无交互Student。固定C和Local初始槽、扰动Goal时，Local各层VLM表示应不变；反向亦然。最终动作可以因Goal→Plan改变，Independent不表示两条动作路径或共享骨干参数互不影响。

### 5.5 A4：完整Stage II及复用验收

保留Goal8＋Local8、深层RelevanceGate、双路径和四层双角色蒸馏。可引用历史S_LG4000，但正式复用需要：

1. 数据、split、Teacher、STATS、processor、采样、有效batch与新组匹配。
2. A3/A4共同初始化、LR trace、optimizer、精度和预算一致；不只是最终step相同。
3. 改造后完整配置加载原A4，同观测／同噪声／同dtype的hidden与动作回归通过，容差预先规定。
4. 用第6节统一推理程序重评，或证明历史结果全部协议字段、模型和代码hash一致。

不满足时，从T_LG和配对step0快照重新训练A4，历史成绩单列；若改了Teacher协议或split，连Teacher一起重训。部署直接走监督Student路径，无Actor。“等价eta=1”不意味着必须借用Stage III包装器。

## 6. 统一推理与结果统计

### 6.1 全量任务与固定参数

主清单为FULL_EVAL：Spatial2402、Object2518、Goal2591、Long2519，共10030任务配置，每项1个episode。固定suite顺序 `libero_spatial, libero_object, libero_goal, libero_10`。

| 项目 | 固定协议 |
|---|---|
| 模型 | A0直接策略／A1–A4各自Student；无Teacher、无Actor |
| 输入 | 当前主／腕RGB256×256＋指令，无state／真实未来／未来mask／示范剩余长度 |
| 初始条件 | 原reset流程、初始状态index0、环境seed0、稳定等待动作10步 |
| chunk / Euler | 8 / 10 |
| 最大策略步数 | 本地adapter：Spatial250、Object300、Goal320、Long550；稳定等待单独计数 |
| 动作噪声 | `(suite_index+1)*10000000 + task_id*10000 + episode_idx*100 + chunk_index`，suite_index从0开始 |
| 精度 | 沿用已验证Plus Student的实际BF16／FP32路径和BF16初始噪声；五组完全一致并记录dtype |
| 后处理 | 相同图像resize／相机方向、反归一化、前六维限幅和夹爪二值化 |
| 视频 | 默认关闭，调试视频单独记录 |

按七类Camera／Robot／Language／Light／Background／Noise／Layout及4×7矩阵汇总。Total是全部成功数/10030，不以七类未加权均值替代。8380只能另作固定历史子集对照，不能与全量混写，不能按当前成绩调整配额或提前结束。

### 6.2 Plus夹爪和新checkpoint适配

统一 `libero_plus_open_to_legacy_close_v1`：训练数据和模型输出1=开、0=闭；客户端交给旧core的字段转换为 `1-g`，再由core映射到仿真器。只翻转一次，不重复反归一化，不原地修改缓存动作。A0虽从原SFT出发，但在Plus数据上训练后仍使用这套显式语义。

当前 `student_action_protocol.py` 将路径限定在旧 `runs/stage_II_20260925`，并只接受Stage-II标记。它不能直接加载A0和新的消融目录。后续按动作metadata和受控ablation路径扩展兼容，不伪造阶段、不覆盖旧run、不全局删除检查。

每个环境步检查成功和终止，成功后不把剩余chunk强行执行完。基础设施错误保存并按预定次数重试；失败不删除，不无限重试直到成功。存在未解决错误时标为incomplete，分别报告完成数、失败数、错误数和完整分母，不能将完成项成功率冒充完整Total。

### 6.3 执行顺序与选模

先对各组做相同预先固定的小型联通检查，核对reset、图像、有限动作和物理夹爪，再对预先指定checkpoint完成全部10030项。联通结果与正式结果隔离。

A0若先评step1000，标记预实验，正式step9000在预算中预先锁定，不依据预实验测试成败改类别、调主表超参数。需要在线调参时另建开发任务／初始状态集合，避免与最终测试相同的实际条件；换seed不等于独立。已有历史benchmark结果已被观察，报告时保留这个事实。

结果至少包括Total、四suite、七扰动、4×7矩阵、逐任务成败／执行步数、错误与重试、初始状态hash、模型和协议hash、随机种子。成功步数只统计成功轨迹，无成功记N/A；另报双方都成功条件下的配对步数差及样本数。

尽量扩展至少3个配对训练seed，报告各seed、均值和标准差。同一checkpoint重复rollout不能记为多个训练seed。训练／冻结参数量、有效token、有效更新、anchor量、上游Teacher成本、GPU小时、显存和推理耗时同时报告。

## 7. 后续代码实现范围（本次未修改）

**当前代码有大量Goal8＋Local8硬编码，不能只改 `slots=16` 或将某项loss置零来完成A1/A2。**

| 位置 | 后续需要支持的能力 |
|---|---|
| `goal_local/model.py` 构造／validate／save／load | 显式goal_slots、local_slots；合法8/8、0/16、16/0；A0的W0→prior；checkpoint携带结构和来源 |
| 同文件language／encode | 去除固定`length+8`切片；按实际role span构造位置、mask、层表示；显式gate开关 |
| `goal_local/roles.py` | 可配置单／双角色mask与Resampler槽数；C不读角色；无角色分支不执行 |
| `goal_local/action.py` | 单尺度只执行存在的条件投影和attention；空K/V不能进softmax再乘0；Plan和原DiT不变 |
| `make_stage_two`／`DistillationSystem` | 从对应T_L/T_G/T_LG初始化，只蒸馏存在角色，保留严格继承／冻结检查 |
| `libero-plus/runtime/indexed_future.py` | 按角色解码未来，A0不读未来；共用anchors，不筛尾部；缓存只读复用或隔离 |
| `runtime/train_ddp.py` | 单尺度Teacher／A0路径；硬编码future.goal.query等梯度与同步审计按配置生成 |
| `runtime/train_stage_two.py` | 支持对应Teacher与角色布局、独立split seed、schedule总长与停止预算；不绕过来源验证 |
| checkpoint／resume | 模型、FP32 optimizer、scheduler、各rank RNG、sampler游标一致提交；故障续训和阶段重置分开 |
| 推理server／client | 按metadata加载A0–A4结构，支持新路径及A0阶段，保留Plus动作协议验证 |

用同一套公共代码的显式配置表达变体，避免复制五套逐渐分叉的训练器。新能力必须回归完整A4。下表是设计schema，不是已实现配置文件：

| 组 | goal_slots | local_slots | teacher_future | distill_roles | student_interaction | action_conditioning |
|---|---:|---:|---|---|---|---|
| A0 | 8 | 8 | none | none | learned_relevance | role_specialized |
| A1 | 0 | 16 | local | local | absent | local_only |
| A2 | 16 | 0 | goal | goal | absent | goal_only |
| A3 | 8 | 8 | local_goal | local,goal | absent | role_specialized |
| A4 | 8 | 8 | local_goal | local,goal | learned_relevance | role_specialized |

全部 `actor_mode=absent`。具体启动CLI在实现完成后依据实际parser补充；本稿不提供会误启动旧实验的占位shell命令。

## 8. 开训前和部署前验收

| 检查 | 必须满足 |
|---|---|
| 权重来源 | W0／Teacher／未训练Student／数据／STATS可追溯；单尺度没有继承双尺度已训练权重 |
| 形状 | 总角色16；Teacher／Student单角色均16；hidden2048，动作条件1024，Plan4，动作8×7 |
| A0无未来 | 去掉batch未来字段仍可训练；future读取计数0；无Teacher；动作loss对query／gate／启用动作分支有有限梯度 |
| A1/A2单尺度 | 另一尺度未来不读取／不编码／不进入Teacher；不存在的角色不参与投影、attention或loss |
| 隔离 | Teacher原生角色隔离；A3从step0无额外交互；扰动角色测试排除经C泄漏 |
| 冻结 | 视觉和DINO不更新；Teacher无梯度且不入optimizer；Student启用模块梯度有限，不强求未参与前向参数有梯度 |
| 蒸馏 | 正确层位、最终norm、detach、有效mask、四层mean、Gram尺寸和角色权重 |
| 尾部 | 不跨episode、不复制末帧伪造未来、不删anchors；缺失输入有限；Student角色不因Teacher mask关闭 |
| 动作路径 | Goal注入1/5/9/13、Local注入全16层，均在block前；不新增补偿路径，不删除Plan |
| 初始化公平性 | A3/A4公共权重和query逐张量相同；A0没有Teacher污染 |
| A4回归 | 同图像／指令／噪声／dtype下，新旧完整配置hidden和动作满足预先规定容差 |
| 存取续训 | 保存加载动作一致；完整resume保持LR、采样和更新一致；单尺度不被默认8/8覆盖 |
| Plus部署 | 夹爪物理开合正确，1-g一次；前六维和缓存正确；新路径检查是扩展而非移除 |
| 环境统计 | reset可复现，逐步判成功，任务ID／名称完整，Total和分类能从原始记录重算 |

现有 `tests/goal_local` 10项测试在前一轮回溯中通过，但它们不能替代新增单尺度和A0无未来的行为测试。后续针对真实行为补充验收，不只断言配置字段存在。

## 9. 目录、执行顺序与交付记录

现有A0–A4各自已有 `libero-plus` 子目录。建议后续结构：

```text
R/ablation/aba1/
├── guide_plus.md
└── A{0,1,2,3,4}/libero-plus/protocol_plus_v1/seed_2026/
    ├── config.json
    ├── checkpoint_refs.json
    ├── manifests/          # 数据split、anchors、开发／正式评测清单和hash
    ├── audits/             # 继承、冻结、梯度、数值与部署检查
    ├── stage_I/            # A1/A2教师；A3/A4可保存只读Teacher引用
    ├── stage_II/           # A1–A4学生
    ├── action_only/        # A0，记录阶段步数和累计步数
    ├── evaluation/
    │   ├── pilot_step1000/ # 采用建议的A0预实验时
    │   └── full10030_final/
    │       ├── protocol.json
    │       ├── episodes/
    │       ├── errors/
    │       ├── results.json
    │       └── matrix.csv
    └── summary.json
```

这是规划，不表示本次已经创建上述产物。历史Teacher和Student只读引用并保存SHA-256；避免复制大模型造成版本分叉。新输出、端口、锁、队列、日志、optimizer恢复目录均隔离，不覆盖原训练、评测或Stage III。

执行顺序：

1. 明确A0预算单位，锁定正式更新预算、数据划分、10030任务清单和协议版本。
2. 完成第7节实现和第8节验收；先回归A4，再验证A0／单尺度／A3。
3. A0按确认单位先训练再推理；若为step1000预实验，保留完整续训状态，正式主对照继续至累计9000。
4. A1、A2分别W0→各自Teacher5000→各自Student4000。A3从T_LG及配对未训练Student训练4000；A4满足条件则复用，否则配对重训。
5. 对五组预先指定正式权重完整评测10030项，重算Total和分类，保留逐任务成败。
6. 扩展配对训练seed，报告差值、不确定性与成本。更改split／LR／评测口径时另立版本和匹配基线。

先做A0只是执行顺序，不授予额外测试集调参机会。并行与GPU安排留到后续实际执行，不改变预算。

每个run至少记录：变体、协议、stage、train/split seed、槽数、未来偏移、交互模式、动作路径、蒸馏层／权重、来源路径和hash、代码版本／文件hash、processor／STATS、batch、LR trace、schedule总长、停止步数、实际anchors、完整resume状态、环境版本、动作语义、任务和初始状态hash。

模型及声明的恢复状态完整提交后才写 `checkpoint_complete.json`。纯推理快照与完整续训快照区分。磁盘按模型、FP32 master、Adam状态和保留份数估算，不只计算单个约5GB模型；临时盘恢复状态不能当作永久归档，重要恢复点需持久保存。

结果表模板（主表Local/Goal顺序与第1节一致）：

| 组 | seed | 有效训练预算 | Local/Goal槽数 | 交互 | 成功数/10030 | Total | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 待填 | action-only累计9000 | 8/8 | 有 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| A1 | 待填 | Teacher5000+Student4000 | 16/0 | 无 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| A2 | 待填 | Teacher5000+Student4000 | 0/16 | 无 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| A3 | 待填 | Teacher5000+Student4000 | 8/8 | 无 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| A4 | 待填 | Teacher5000+Student4000 | 8/8 | 有 | 待统一验收／评测 | 待填 | 待填 | 待填 | 待填 | 待填 |

历史A4的86.5703%保留在第2.3节，不自动填入新协议结果表。预实验、8380历史子集和长训练扩展各自单列。另表报告七扰动、参数／计算／训练与推理成本。

## 10. 参考与可复查的本地证据

- [A组定义和公平性要求](../../reference/abalation.md)。
- [Plus Stage-I说明](../../libero-plus/README.md)、[Stage-II历史记录](../../libero-plus/STAGE_II_20260925.md)。
- [实际Stage-I配置](../../libero-plus/configs/stage_I.json)、[Stage-I运行参数](../../libero-plus/runs/stage_I_20260923/arguments.json)、[Stage-II运行参数](../../libero-plus/runs/stage_II_20260925/arguments.json)。
- [教师来源](../../libero-plus/runs/stage_II_20260925/teacher_source.json)、[step0初始化标记](../../libero-plus/runs/stage_II_20260925/step_000000/checkpoint_complete.json)。
- [数据审计](../../libero-plus/runs/stage_II_20260925/dataset_audit.json)、[数据重复与夹爪诊断](../../libero-plus/audits/diagnosis_20260925/REPORT.md)。
- [model.py](../../goal_local/model.py)、[roles.py](../../goal_local/roles.py)、[action.py](../../goal_local/action.py)。
- [Plus Stage-I实际入口](../../libero-plus/runtime/train_ddp.py)、[Stage-II实际入口](../../libero-plus/runtime/train_stage_two.py)，不要误用普通LIBERO的历史启动脚本。
- [Plus动作适配](../../libero-plus/runtime/student_action_protocol.py)、[全量清单](../../libero-plus/evaluation/stage_II_20260925/full_manifest.json)。
- [A4历史全量结果](../../libero-plus/evaluation/stage_II_20260925/step_004000_gripper_v1/results.json)、[89.62%历史子集结果](../../libero-plus/evaluation/stage_II_steps2500_4000_parallel_20261008/step_004000/step_004000_step500_stratified_gripper_v1/results.json)。

本轮交付仅为本Markdown方案，没有新增代码／训练配置、改动数据或权重，也没有启动消融实验。
