# LaWAM_aba2 → Goal/Local 两阶段架构：实现与权重继承指南

更新：2026-09-23。本文以本目录 `goal_local/` 的实际代码和已运行实验的初始化记录为准。

## 1. 三种来源与继承关系

- **架构基线**：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2`。本目录由该基线独立复制；新模型入口是 `goal_local`，不是保留的历史 `starVLA` 训练入口。
- **设计指导**：`/mnt/hdd/wangchaoyang/idea/idea_9_20_change.md`。采用“两阶段、角色隔离、真实未来后验教师、current-only 学生、分层动作条件化”方案。
- **初始化权重**：`/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`。这是 aba2 评估使用的原始 LIBERO SFT 权重，不是本项目后来训练的教师或学生。

初始化还读取同目录上一级的 `dataset_statistics.json`、`CKPT/LaWAM/results/Checkpoints/qwen3_weights` 的配置/tokenizer/processor，以及 `CKPT/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m` 的模型配置。Qwen 和 DINO 的实际参数从上述 LIBERO checkpoint 提取，不重新下载、不从 DiT4DiT 加载。

```text
aba2 的架构 + 原始 LaWAM LIBERO SFT 参数
                    ↓ 严格加载可复用部分，初始化新增模块
Goal/Local 后验模型（Stage I）
                    ↓ 示范动作 flow loss 训练
新的 Stage-I checkpoint
          ┌─────────┴──────────┐
完整冻结的后验 Teacher       复制公共参数和动作头得到 Student
继续接收真实未来             新增 Q_G/Q_L 和深层 Goal→Local gate
          └────目标表征───────→ 动作 loss + Local/Goal 蒸馏
                               ↓
                       可部署的 Stage-II Student
```

“教师”是 Stage-I 模型在后续 Stage II 中的角色。Stage I 本身由真实示范动作监督，不存在另一个蒸馏教师。

## 2. 从旧 checkpoint 继承了什么

加载入口：`goal_local/model.py::GoalLocalPolicy.from_baseline`。旧参数键前缀为 `policy_backend.`。

| 模块 | checkpoint 来源/加载规则 | Stage I | Stage-II Student |
|---|---|---|---|
| Qwen3-VL 当前图像视觉编码器及 merger | `vlm.model.visual.*`，随 Qwen 严格加载 | 冻结 | 冻结副本 |
| Qwen 文本嵌入、前 16 层语言骨干、最终 norm | `vlm.model.*` 中对应参数；保留原 checkpoint 词表大小 | 训练 | 从 Stage I 复制后训练 |
| 当前主视角 DINO | `lam.vision_encoder.model.*`，严格加载；仅复用此视觉子模块 | 冻结 | 冻结副本 |
| 原 Action Query，8×2048 | `act_query` 原值复制 | 训练 | 从 Stage I 复制后训练 |
| 原 Flow Query，8×2048 | `flow_action_query` 原值复制 | 训练 | 从 Stage I 复制后训练 |
| 原动作 Flow/DiT 模块 | `flow.*`；原动作编码/解码、时间编码、VLM 条件投影、DiT block 等 | 训练 | 从 Stage I 复制后训练 |
| 归一化统计 | 原 LIBERO `dataset_statistics.json` | 固定数据约定 | 随 checkpoint 保存 |
| processor、tokenizer、两段 prompt | 原配置和 `vlm_adapter.py`；恢复 `<ACT_PH>` 特殊 token | 沿用 | 沿用 |

已有 LIBERO 初始化审计 `runs/goal_local/stage_I_20260921/initialization.json` 记录：Qwen **493** 个 tensor、DINO **211** 个 tensor、Flow **245** 个 tensor，另复制两组旧 query。这里是 tensor 数，不是参数元素量。

Qwen/DINO 使用 `strict=True`。Flow 的 `strict=False` 仅用于容纳白名单内的新参数；发现任何其他 missing/unexpected key 会报错，不能静默漏载旧动作头。

### 两组旧 query 和 prompt 的真实位置

两组旧 query 仍按原模板放进当前 Qwen 上下文 C，仍参与推理，不被新 Local/Goal token 替换。旧 query 的最终 16 个输出位置通过 `remove_old_queries` 从送往动作头的当前条件中移除，但新角色可读取这些 current-only query 的中间表示。

保留两段文本模板：

```text
Your task is {instruction}.

From main-view observations, infer the robot arm's motion.
```

```text
Then use observations and the inferred motion to produce the robot policy actions.
```

移除的旧链路包括旧 latent-action Q-Former/LAM encoder、旧 decoder/LaWM 及 `vlm_to_lam` 路径；不加载、不运行旧未来重建或 latent/perceptual loss。DINO 虽来自旧 `lam` 命名空间，仍被单独保留，不能据键名前缀误判为保留整套 LAM。

## 3. 新增模块及初始化

| 新模块 | 实现 | 初始参数 | 出现阶段 |
|---|---|---|---|
| Local/Goal 两个 Future Resampler | `roles.py::FutureResamplers/Resampler` | 各 8 个压缩 query，Normal(0,0.02)；Attention/Linear/Norm 用 PyTorch 默认初始化 | Stage I 训练；Stage-II Teacher 冻结；Student 无此模块 |
| Local view embedding、Goal horizon embedding | `future.view` 2×2048；`future.horizon` 5×2048 | Normal(0,0.02) | 同上 |
| Goal/Local 动作条件投影 | `flow.goal_proj/local_proj`，2048→1024 | 新 Linear 默认初始化 | Stage I 起训练；Stage II 继承 |
| 4 个 Plan tokens | `flow.plan`，4×1024 | Normal(0,0.02) | Stage I 起训练；Stage II 继承 |
| Goal→Plan cross-attention | DiT 物理层 1/5/9/13 | 新 Attention 默认初始化 | Stage I 起训练；Stage II 继承 |
| Local→Action cross-attention | DiT 全部 16 层 | 新 Attention 默认初始化 | Stage I 起训练；Stage II 继承 |
| 两路残差 scale | `flow.goal_scale/local_scale` | 0.01，小但非零 | Stage I 起训练；Stage II 继承 |
| Student Q_G、Q_L | 各 8×2048 | Normal(0,0.02)；不拿单个样本的教师表示初始化 | 仅 Stage II 新建 |
| Student relevance gate | 2048→256 投影、NULL key、8 个深层 message attention、scale | 默认层初始化；NULL Normal(0,0.02)，scale=0.01 | 仅 Stage II 新建 |

角色 attention mask、未来有效性 mask、动作 mask、蒸馏公式是计算规则，不含需要从 checkpoint 继承的训练参数。

## 4. 分步骤构建 Stage I

1. **加载基线。**恢复上表的 Qwen、DINO、两组旧 query 和 Flow 参数，冻结 Qwen 视觉前端/merger 及 DINO。新增模块从各自初始化起训练；不要把整个动作 DiT 随机重建后忽略旧权重。
2. **构造样本。**同一 episode 内取当前主/腕视角及语言；动作窗口 H=8；Local 为 t+8 双视角，Goal 为主视角 t+[24,40,56,72,88]。不跨 episode，尾部不强制丢弃。
3. **真实未来编码。**有效未来帧逐张经过冻结 Qwen vision；Local 加视角标记，Goal 加 horizon 标记，各自经可训练 Resampler 得到 8×2048。无效帧不参与编码/聚合；全无效角色返回有限零占位。
4. **拼接角色序列。**Qwen 输入 `[C;F_G;F_L]`。C 包含当前双视角、指令和两组旧 query。Stage I 没有 Student Q_G/Q_L，也没有 Student gate。
5. **保持角色隔离。**16 层中，C 只读 causal C；Goal 只读 C+Goal；Local 只读 C+Local。Goal/Local 在 VLM 内不互读，不能经 C 间接交换真实未来。
6. **分层条件化动作。**当前 Qwen 输出去掉旧 query 的 16 个位置，经旧投影变为 768 维，与当前主视角 DINO 的 256×768 拼成 base memory。Goal/Local 各投影至 8×1024。Goal 注入 Plan，Local 注入 Action；原 DiT self-attention 让 Plan/Action 联合计算。
7. **监督更新。**只计算 masked posterior action flow loss；梯度经过动作头、语言骨干和两个 Future Resampler。冻结视觉前端没有梯度，不新增像素重建、anchor 或 Stage-I 蒸馏损失。

### 维度约定

令 C 长度为 L（含旧 16 个 query，不含 state token）。

| 位置 | batch 维度 |
|---|---|
| 当前主/腕视角 | 各 B×3×256×256 |
| Local 视觉 token | B×2×N×2048 → B×8×2048 |
| Goal 视觉 token | B×5×N×2048 → B×8×2048 |
| Qwen 输入 | B×(L+16)×2048 |
| 当前动作条件 | B×(L−16)×2048 → B×(L−16)×768 |
| DINO / base memory | B×256×768 / B×(L+240)×768 |
| Goal/Local 动作条件 | 各 B×8×1024 |
| Plan + 动作 buffer | B×4×1024 + B×50×1024 = B×54×1024 |
| velocity / 有效动作监督 | B×50×32 / 通常 B×8×7 |

N 由 processor 实际图像网格决定，当前 256 分辨率配置通常为 64，不在适配器中硬编码。尾部只有 min(8,剩余步数)×7 有效动作；不以重复末尾动作补成真实标签。

## 5. Stage II 与部署：谁继承谁

`make_stage_two` 从**已经训练的新 Stage-I checkpoint**构建两个独立参数副本：

- Teacher 保留后验 Qwen、Future Resamplers、view/horizon 及其上下文参数，完整冻结并 eval。移除 Teacher 的 Flow 动作头，只产生中间层监督。
- Student 复制 Stage-I 公共模块及完整新动作头，去掉 `future.*`，新建 Q_G/Q_L 和 gate。不与 Teacher 共享可更新的 parameter storage。
- 两支各自前向；Student 只看当前图像和指令。其 native mask 始终保持角色隔离，仅第 9–16 层显式 gate 允许 Goal→Local。
- 在 Qwen 第 4/8/12/16 层分别对齐 Local 和 Goal，loss 为 cosine + slot Gram；4/8/12 用 block 输出，16 用最终 norm 输出。
- 总损失为 Student action flow + λLocal·Local distillation + λGoal·Goal distillation。默认两个 λ 和 Gram α 均为 1。
- Stage II 缺失未来仅屏蔽相应蒸馏，不关闭学生角色或其动作注入。

部署仅加载 Student，使用当前双视角+语言及内部可学习 query，10 次 flow/Euler 更新输出 8×7 动作。没有真实未来、Teacher、Future Resampler 或蒸馏 loss。Stage-I 后验模型不是 current-only 部署策略，不能直接拿它替代 Stage-II 学生在线控制。

## 6. 与 idea 的一致点及明确适配

- 一致：两阶段、独立冻结 Teacher、8+8 角色槽、teacher 角色隔离、student 深层单向 gate、Goal→Plan / Local→Action、分层蒸馏和学生动作监督。
- **不使用 s_t / State MLP**：idea 提到了本体状态，本次 aba2 基线适配继续禁用；`validate_config` 拒绝 `flow.use_state=true`。
- **保留两组旧 query**：idea 主要写原 8 个 Action Query，实际 aba2 还包含 8 个 Flow Query，二者均保留，但不直接作为最终动作 base memory 位置。
- **Goal 逐帧 Qwen 编码+有序 horizon embedding**：没有另加 V-JEPA，也没有把无效帧拼进视频视觉编码。
- **位置编号按 aba2 实际调用**：原调用传 `inputs_embeds`、不缓存；新实现保持顺序位置行为，不宣称额外切换为图像网格 MRoPE。师生角色位置规则一致。
- 角色注入 scale 初值 0.01，不是全零；这是实现选择，不是从旧权重继承。
- 结构和梯度连通不等于已经证明 Local/Goal 学到理想语义；需未来条件消融和最终学生闭环评估。不能用 Stage-I loss 代替成功率。

## 7. 已有 LIBERO 实验的证据

- `runs/goal_local/stage_I_20260921/initialization.json`：真实初始化来源及 tensor 迁移记录。
- 同目录 `arguments.json`：原始 Stage I 命令的总步数是 **10000**，不依据口头记忆写成 16000；实际训练/保存进度应另外读取日志与 checkpoint。
- `runs/goal_local/stage_II_warm250_20260921/teacher_source.json`：Stage-II 固定蒸馏教师来源。
- `IMPLEMENTATION.md`、`plan.md`、`tests/goal_local/`、`artifacts/goal_local/`：实现说明和结构验证。

## 8. 将训练数据替换为 LIBERO-Plus

工作目录：`LaWAM_aba2_change/libero-plus`。**重新从第 1 节原始 LIBERO SFT 初始化**；不从旧 Goal/Local LIBERO 教师、4000 步学生或 `LaWAM_without_subgoal` 的模型继续训练。

只读复用 `LaWAM_without_subgoal/libero-plus/data/indexed_rlds`：1024 分片、14347 episodes、2238036 steps、40 种指令文本。该目录是 `arrays.npz + images.bin` 自定义索引格式，不是 LeRobot；JPEG 字节原样保留，已经存在 episode 边界、语言和 7 维动作。

按以下顺序接入：

1. 使用完整 `manifest.json`，不使用遗留的 50 分片 `active_window.json`。校验 source fingerprint、完成标记、索引范围及数值。
2. `libero-plus/runtime/indexed_future.py` 将小型动作/图像偏移索引转为只读 mmap（约 130 MiB），JPEG 留在源目录，不复制或重新编码。使用 `(shard, source_episode)` 映射为唯一 episode ID。
3. 按指令分层、episode 级 95%/5% train/val 划分；在训练帧上均匀随机采样当前锚点，保留自然尾部无未来样本。未来索引和动作 mask 与原 `FutureDataset` 一致。
4. indexed 数据夹爪已转换为 **1=开、0=闭**；不得再次执行 RLDS 原始 -1/+1 的反转。前六维沿用原 LIBERO checkpoint 的 min/max，保持继承动作头的坐标尺度。
5. 源 manifest 声明 20 Hz，转换脚本未保留时间戳。本实验明确采用该 20 Hz 控制约定及 H=8，不能声称独立重建了源采集时间或证明 no-op 处理完全等同原数据。
6. 单卡入口 `libero-plus/runtime/train.py` 只替换 Dataset 后端；用户后来指定 GPU1、3 双卡，正式入口改为 `runtime/train_ddp.py`，新增 DDP 训练编排。模型、loss、FP32 master AdamW、冻结策略及模型 checkpoint 格式继续使用 `goal_local`。这不是采用 without_subgoal 的 state-token 架构。
7. 先检查全量索引、语言与图像解码、尾部 mask、train/val 不交叉、spawn DataLoader，再检查真实 Plus batch 的完整模型前反向与新增模块梯度。
8. 从原 SFT 干净初始化正式 Stage I，日志与检查点写入 Plus 工作目录。后续如训练 Stage II，必须使用这次 Plus 教师；现有 trainer 会从教师继承 config，因此不能只改旧 Stage-II JSON 而仍加载旧 LIBERO 教师。

当前实验启动参数与运行状态见 `libero-plus/README.md`、`libero-plus/configs/stage_I.json`、对应 run 的 `arguments.json/status.json`。当前按用户要求使用 GPU1、3 双卡 DDP，每卡 batch2、累积8，有效 batch32；CPU FP32 master AdamW，保留最近两个完整 checkpoint 和 best 模型；临时保存新 checkpoint 时额外需要一个 checkpoint 的空间。每次验证是离线 flow loss，不是在线 LIBERO-Plus 成功率。

原单卡训练器的恢复会开启新 iterator，不承诺逐样本精确续跑。Plus 双卡入口使用独立 DDP trainer 格式，保存每个 rank 的 RNG、epoch 和已消费 batch 位置；恢复时保持 world size/batch/累积/总步数一致，按索引跳过已消费批次。模型文件仍能由 GoalLocalPolicy.load 加载，用于后续 Stage II。
