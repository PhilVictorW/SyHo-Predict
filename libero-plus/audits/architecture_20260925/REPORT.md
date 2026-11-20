# LIBERO-Plus 架构迁移审计（2026-09-25）

> 后续深入排查更新：已确认当前 LIBERO-Plus 训练标签与在线执行端的夹爪语义相反。请优先阅读 `../diagnosis_20260925/REPORT.md` 及对应证据。本报告的 Plan 数值测试仍有效，但不是对本次退化主因的证明。下文已修正实际部署精度的判断。

## 范围与操作状态

核查链路为原 LIBERO SFT → LaWAM_aba2 的去 LaWM/去旧 query 输出条件版本 → Goal/Local Stage I → 教师第 5000 步初始化的 Stage II → 学生第 500 步部署。

按用户最新要求，已停止学生第 500 步 LIBERO-Plus 在线评估及其调度器，删除 `libero-plus/evaluation/stage_II_20260925/step_000500` 下结果、对应评估日志及汇总中的第 500 步记录。保留模型权重、评估任务清单和暂停状态记录；队列保持暂停，避免立即重新生成结果。没有停止训练或原 SFT baseline 的评估。2026-09-25 12:36 左右读取的训练状态为第 840 步、`gradient_sync_optimizer` 阶段，四个训练进程仍存活。

本次未修改生产模型、训练、推理代码或运行配置。诊断仅使用 CPU，禁用 CUDA，主要数值实验限制为两个 PyTorch CPU 线程。新增文件仅为本目录诊断脚本、数值结果和本报告。没有重新启动在线推理。

## 结论

未发现教师→学生的公共权重漏加载、错加载，或当前视觉/文本主干在 FP32 下发生明显的计算图迁移错误。但发现一个可复现的行为保持风险、一个推理精度策略偏移和一个夹爪阈值边界差异。蒸馏目标还存在明确的幅值盲区。它们值得优先排查，尚不能据此宣称已证明在线成功率下降的唯一或主要原因。

### 1. 高优先级：Plan tokens 插入不是行为保持的初始化

代码：`goal_local/action.py:42` 将 4 个 Plan tokens 无条件加入动作序列，并将其 attention validity 设为真；`:55`、`:57` 的小尺度参数只控制 Goal/Local cross-attention 注入，不能控制 Plan→Action 的原有 self-attention 路径。

因此，即使 `goal_scale=local_scale=0`，动作仍会关注这 4 个新 token，attention 的归一化与中间状态均会变化。把角色注入关掉不等于恢复原 aba2 模型；真正的基线回归需要同时去掉 Plan tokens。

使用原 LIBERO SFT 的真实动作头权重、固定合成条件、同一个噪声和时间步，在 CPU 上得到：

| 比较 | 有效动作速度输出相对 L2 差异 |
| --- | ---: |
| 新实现完整绕过角色及 Plan vs 原 DiT | 0 |
| 新增角色与 Plan 的初始化 vs 原 DiT | 17.566% |
| 角色注入尺度置零但保留 Plan vs 原 DiT | 17.586% |

这证明原动作计算循环可精确复现，而插入 Plan 本身会改变继承策略。它不代表在线成功率下降 17.6%，也不是训练后第 500 步的动作差异测量。该拓扑与 idea 的设计方向一致，问题在于缺少保证迁移初期行为不被破坏的机制，而不是 Plan tokens 本身必然不可用。

### 2. 修正：不能由 autocast 声明推断实际 baseline 为 FP32

原实现 `LaWAM_aba2/starVLA/model/framework/vlas/lawam.py:842,877` 写有 FP32 autocast，但进一步核查实际服务启动参数为 `--use_bf16`，`deployment/model_server/server_policy.py:45` 会先将整个模型转为 BF16；Flow 的 `_compute_dtype()` 又取参数 dtype。因此不能据此称实际原服务的 Flow 为 FP32。新实现 `goal_local/model.py:237` 的 BF16 autocast 覆盖整个采样，二者具体混合精度路径仍不同，但上轮将实际 baseline 概括为 FP32 的判断不成立。新评估端提供 BF16 初始噪声。

固定条件 CPU 探针中，动作速度 FP32/BF16 的相对 L2 差异为 0.318%。这不是 CUDA 闭环结论，不能据此认定其解释严重退化；但比较模型效果时应先统一该推理协议，避免混入另一项变量。不能只修改 autocast 而忽略 Flow 参数、条件和采样状态的实际 dtype。

### 3. 已确认：夹爪在恰好 0.5 时的处理不同

原 `examples/LIBERO/eval_files/model2libero_interface.py:413`：`>=0.5` 输出 1；新 `goal_local/data.py:37`：`>0.5` 输出 1。

对 `[0.49, 0.5, 0.51]`，分别输出 `[0,1,1]` 与 `[0,0,1]`。这是一处确定的部署协议差异。尚未测量真实轨迹中恰好 0.5 的频率，不能把它当作主要原因；BF16 有限精度下也不能假定该边界永远不会出现。

### 4. 蒸馏 loss 下降不保证角色幅值或动作保持一致

`goal_local/roles.py:90` 对师生特征分别 L2 normalize 后计算 cosine 与 Gram loss。令学生特征为教师的 3 倍，数值实验中蒸馏 loss 仅 `1.12e-8`，原始特征相对误差却为 2。这是该损失定义本身的尺度不敏感性。

在一条固定真实验证样本上，教师 5000 步与学生 500 步的结果为：

| 层 | Goal 学生/教师 RMS | Local 学生/教师 RMS | Goal cosine | Local cosine |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.162 | 0.073 | 0.675 | 0.699 |
| 8 | 0.550 | 0.158 | 0.876 | 0.876 |
| 12 | 1.005 | 0.374 | 0.913 | 0.898 |
| 16（最终 norm 后） | 1.012 | 0.982 | 0.889 | 0.953 |

中间层幅值确有差距，但不能遗漏反向证据：直接提供给动作头的最终层幅值已经接近教师。最终层仍存在方向误差，Goal/Local 原始相对 L2 分别约 0.473/0.307。以上只是一条样本，不是整个验证集统计，也不能证明幅值是当前动作退化主因。

教师的角色来自未来视觉，学生新建的 query 仅依靠当前图像与文本估计角色；教师训练出的动作头依赖真实未来条件，而学生初始化不能立即提供同分布条件。这是 Stage I→II 的固有分布变化。低教师 action loss 或持续下降的学生 cosine/Gram loss，并不能单独验证部署策略已经恢复。当前蒸馏监督为角色特征，没有直接要求师生在同噪声、同时间步下的动作速度输出相等。

## 已排除或暂未发现异常的部分

- 教师 5000 步→学生 0 步：1038 个应继承张量逐个完全相同，没有 missing/mismatch；学生新增的 44 个 state keys 全属于 `q_goal`、`q_local`、`gate.*`。未来视觉 resampler 不存在于学生中，符合部署设计。
- 师生配置、动作归一化统计一致。原 SFT→教师 5000→学生 500 的冻结 Qwen visual 315 个、DINO 211 个张量均完全一致。
- 原动作头 245 个张量成功继承，87 个新增角色/Plan 张量独立初始化，无 unexpected keys。没有证据表明保存的权重来自错误模型。
- 两项目的 `flowmatching_expert.py`、`cross_attention_dit.py`、`vlm_adapter.py`、`batch_utils.py` 文件内容完全一致。
- 保留原 Action/Flow query 的 VLM 输入，但从传给动作头的当前 VLM 输出中移除这 16 个位置，符合用户指定的 aba2 继承方式。新增 Local/Goal tokens 走独立条件注入。
- 当前路径对角色/未来的 attention 隔离、教师 Goal/Local 隔离、学生深层 Goal→Local gate、Goal 第 1/5/9/13 层注入、Local 全 16 层注入均通过现有结构测试；学生编码不读取未来图像。
- 真实样本下，含 3 路 DeepStack 特征的手动当前路径与原生 Qwen 当前路径：CPU FP32 相对 L2 为 `1.16e-6`、最大绝对误差 `3.81e-5`。BF16 同比较为约 1.486%；FP32 回归支持主干计算图基本一致，BF16 差异不能直接当作 attention mask 错误。
- 小模型 gradient checkpoint 开关对前向 loss 和梯度的差异均为 0。已有 `tests/goal_local` 共 10 个测试全部通过；它们不是 GPU 闭环测试。
- 当前两路 RGB 图像均按 256×256 输入；DINO 当前主视角的归一化、层选择和 register token 移除路径与 baseline 对齐。8×7 有效动作与 50×32 padded 形状的 mask 未发现新错位。
- 不使用本体状态是用户此前指定的架构选择，不将其列为意外漏接输入。教师使用未来信息、学生部署不使用未来信息，也是既定设计。

## 建议处理顺序（尚未实施）

1. 先统一原基线和新模型的 Flow FP32 推理策略及夹爪阈值，做同图像、同指令、同噪声的动作回归；保留所有现有权重作对照。
2. 对 Plan→Action 路径建立可真正关闭的控制或渐进启用方案，并验证初始化时完整恢复原动作头输出。仅调小 Goal/Local cross-attention scale 不足以达到该目标。此项涉及架构/训练变更，不应直接套用后假定现有 checkpoint 可保持表现。
3. 扩展至固定真实验证集，分别测量师生最终角色方向/幅值、同噪声动作速度差异以及完整采样动作差异。必要时再评估幅值约束、动作级蒸馏或未来条件弱化等训练改动，而不是仅凭三个 loss 总量决定方案。
4. 上述离线对照完成后，再使用固定任务子集比较闭环结果；本次按要求保持在线评估停止。

尚未完成“哪个因素贡献多少成功率变化”的因果归因；这需要受控对照。CPU 实验不替代 CUDA 数值与在线动作评估，也不能承诺只修复一处即可恢复原精度。

## 可复查证据

- `audit_cpu.py` / `audit_cpu.json`：逐张量继承、冻结视觉、精度探针、Plan 插入、梯度回归。
- `real_features_cpu.py` / `real_features_cpu.json`：单条真实验证样本的角色统计与 BF16 当前路径比较。
- `current_fp32_cpu.py` / `current_fp32_cpu.json`：真实样本当前路径 FP32 对照。
- `contracts.log`：10 项现有测试结果。
- `gripper_boundary.json`：阈值边界数值证据。
- `../../evaluation/stage_II_20260925/paused_for_architecture_audit.json`：本次停止及删除范围记录。
