# Goal/Local 两阶段实现

本目录从 `LaWAM_aba2` 独立复制。新模型入口为 **`goal_local`**；保留的旧 `starVLA` 训练脚本仍是历史 LaWAM 入口，不用于本次新架构。

## 已实现的训练与部署路径

- **新 Stage I**：原 LIBERO SFT 初始化保留模块；真实未来 → 冻结 Qwen Vision → Local/Goal Resampler → 角色隔离的 Qwen 16 层 → 分层动作头。只计算动作 flow loss。
- **新 Stage II**：从新 Stage-I 后验检查点建立完整冻结 Teacher；复制公共模块与动作头为独立 Student，新增随机 Q_G/Q_L 和第 9–16 层门控。损失为学生动作 flow + 第 4/8/12/16 层 Local/Goal cosine＋Gram 蒸馏。
- **部署**：只加载 Student；当前双视角和原两段 prompt → Qwen → 角色条件动作头 → 10 步 Euler → `[B,8,7]` 动作。没有 Teacher、真实未来、Future Resampler、旧 latent-action Q-Former、LAM teacher 或 LaWM decoder。

旧 action/flow query 各 8 个仍在原模板位置，可训练；其 16 个最终输出位置和相应 mask 一起从当前动作条件中移除。没有本体状态条件。

## 维度

L 为含原 16 个 query 的当前 Qwen 序列长度；图像默认为 256×256。

| 位置 | 维度 |
|---|---|
| 当前 main / wrist | 各 `[B,3,256,256]` |
| Local t+8 的 main＋wrist | Qwen 编码后 `[B,2,64,2048]` → R_L → `[B,8,2048]` |
| Goal t+[24,40,56,72,88] 的 main | `[B,5,64,2048]`＋horizon identity → R_G → `[B,8,2048]` |
| Teacher 或 Student 的 Qwen 输入 | `[C;G;L] = [B,L+16,2048]` |
| 当前动作条件 | 去旧 16 个 query 后 `[B,L−16,2048]` → `[B,L−16,768]` |
| 当前 DINO | 倒数第二层＋LayerNorm，`[B,256,768]` |
| 当前 base memory | `[B,L+240,768]` |
| 两路角色动作条件 | 各 `[B,8,2048]` → 独立投影 → `[B,8,1024]` |
| Plan＋动作 buffer | `[B,4,1024]`＋`[B,50,1024]` = `[B,54,1024]` |
| 预测速度 | 丢弃 Plan 后 `[B,50,32]`；有效监督通常 `[B,8,7]` |
| 最终动作 | `[B,8,7]`，沿用原 min/max 与 binary gripper 归一化体系 |

Goal→Plan 只在 DiT 物理层 **1/5/9/13**；Local→Action 在全部 **16** 层；角色残差先于对应原层。原 8 个 cross 层与 8 个 self 层保持交替，self 在 Plan＋Action 上联合计算。Plan 每次速度预测重新初始化，不跨 ODE 步累积。

Teacher 内 Goal/Local 全层互不可见，当前 C 不读取任何新增角色。Student 使用相同原生注意力 mask，只允许第 9–16 层显式 gate 建立 Goal→Local 通路。Relevance 为 256 维，NULL 槽与 8 个 Goal 槽归一化，`g=1−p(NULL)`。

## 数据与数值细节

- 使用原数据只读采样，无需复制或重编码视频。按 episode metadata 的相机时间偏移读取 LeRobot v3 视频。
- 无效未来不解码、不编码；全无效角色不调用 Resampler。Stage I 关闭缺失角色的整条注入残差；Stage II 只关闭该角色蒸馏，Student 动作分支一直启用。
- 动作尾部按真实剩余帧数与 8 步窗口求交集，50×32 buffer 中只有有效的 7 维动作参加损失。
- 按任务分层做 episode 级 train/validation 划分；默认 5% validation，不复用原 `train_split_all`。
- 保留原 `(1−Beta(1.5,1.0))*0.999` 流时间分布和重复噪声训练 2 次。
- 新增 residual scale 初值 0.01；训练学习率默认 1e−5、蒸馏权重 1。这些是起始配置，不是已经验证最优的超参数。
- **实际代码核对后的补充**：aba2 在 Qwen 前向传 `inputs_embeds` 而不传 `input_ids`，且不使用缓存，实际是顺序位置编号。新代码保持这一行为；没有额外切换成基于图像网格的 MRoPE。Teacher/Student 的新增角色位置严格相同。
- 中间层 4/8/12 取 block 后输出；16 层取最终 norm 后输出，两分支一致。深层 Local 取 gate 更新后的输出。

## 文件

- `goal_local/data.py`：未来采样、真实尾部 mask、统计归一化、episode 划分。
- `goal_local/roles.py`：隔离 mask、两路 Resampler、深层 gate、蒸馏损失。
- `goal_local/action.py`：继承原 Flow 的权重兼容动作头；新增 Plan 与两路角色注入。
- `goal_local/model.py`：原权重严格加载、两阶段模型、检查点保存/加载、Student 推理。
- `goal_local/train.py`：两阶段单卡训练，支持梯度累积、验证、优化器保存和恢复。
- `goal_local/smoke.py`、`tests/goal_local/`：真实尺寸及结构/信息隔离验证。
- `configs/goal_local/default.json`：明确记录本次实验结构与数据契约。

## 运行

在本目录使用 `/home/wangchaoyang/envs/LaWAM/bin/python`，依赖当前环境 `transformers==5.2.0` 的 Qwen3-VL 接口。

```bash
# 索引审核＋边界帧实际解码
python -m goal_local.audit_data --decode

# 结构测试
python -m unittest discover -s tests/goal_local -v

# 新 Stage I；步数为示例，需观察验证集与未来条件利用情况后选择教师
python -m goal_local.train --stage I --output runs/goal_local/stage_I \
  --device cuda:2 --steps 1000 --batch-size 1 --accumulation 8

# 新 Stage II 必须加载新训练的后验 Teacher
python -m goal_local.train --stage II \
  --stage-one runs/goal_local/stage_I/step_001000 \
  --output runs/goal_local/stage_II --device cuda:2 \
  --steps 1000 --batch-size 1 --accumulation 8

# 当前双视角图片必须是 256×256 RGB；输出反归一化的 8×7 动作，不生成视频
python -m goal_local.predict \
  --checkpoint runs/goal_local/stage_II/step_001000 \
  --main current_main.png --wrist current_wrist.png \
  --instruction 'put the cup on the plate' --device cuda:2 \
  --output artifacts/goal_local/actions.json
```

检查点包含权重、Qwen/DINO 配置、processor、原归一化统计和结构配置；加载 Student 不再打开原 SFT 或 Teacher 文件。训练输出路径限制在本目录，禁止写入继承的共享 `results/Checkpoints` 符号链接。

`--resume` 恢复模型、优化器、scheduler 与 Torch RNG；数据 shuffle 会从新 iterator 开始，**不是逐样本完全复现的断点续跑**。当前入口是单卡＋梯度累积，尚未接入多卡 DDP。

## 验证状态

验证产物写入 `artifacts/goal_local/`。正式训练与完整 LIBERO 成功率评测不等同于前反向连通性测试；本次生成的 `smoke` 检查点只用于测试，不能充当正式 Stage-I 教师或性能结论。实际执行结果见 `artifacts/goal_local/validation.md`。
