# A0 LIBERO-Plus：单组 latent16，保留 DINO

本目录按用户 2026-10-19 最新要求执行。它已经不同于 guide_plus.md 原先的双角色 A0：使用单组 latent16，不使用 Goal/Local 分工、Plan、门控或分层注入。最新更正明确保留 DINO。A1–A3 与共享实现未修改。旧入口及配置保存在 archive_dual_role_v1/。

## 模型

- W0：`/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`。
- 保留并严格加载 Qwen（16 层语言骨干）、旧 Action Query 8、旧 Flow Query 8、DINO 和原 Flow/DiT 参数。
- 新初始化 `latent_query[16,2048]`。当前双视角、指令、旧 query 为上下文 C；新 latent 读取 C 与同组 latent；C 不读取新增 latent。
- 仅新 latent 的最终 VLM 输出进入 `enc_vlm: 2048→768`。旧 query 输出和当前上下文输出都不直接送入动作头。
- 与当前主视角 DINO 特征沿 token 轴拼接为动作条件。DiT 原有交替 attention 保留：第 1/5/9/13 层读取 latent，第 3/7/11/15 层读取 DINO，偶数层自注意力。
- 无 LaWM 重建/子目标链、Teacher、Future Resampler、蒸馏、Actor、额外 Plan 或门控。仅优化 action flow loss。
- Qwen vision/merger、DINO 冻结；语言骨干、新旧 query、动作头更新。未使用的 CFG embedding 不进入优化器。
- 动作仍为 20 Hz、chunk8、有效 8×7，内部 padded50×32；推理使用 10 步 flow 积分。

## 数据与训练预算

实际数据：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_without_subgoal/libero-plus/data/indexed_rlds`。
沿用 seed2026 的按 episode 划分：训练 13,630 episodes / 2,125,407 anchors；验证 717 episodes / 112,629 anchors。
仅解码当前双视角，不读取未来图像；动作标签取 t 起最多 8 步，保留 episode 尾部 mask。

用户已明确将预算更正为 **1000 optimizer steps**。一次 step 是梯度累积完成后的一次参数更新。
GPU **0、7**，每卡一个 FSDP 训练 rank；每卡 micro batch=8，累积=2，全局 batch=32。
共训练 32,000 anchors，约 0.01506 nominal epoch。在第 1000 步保存最终权重后停止，并自动执行全量推理。
训练集随机排列采样，支持精确游标恢复；每个 anchor 的 action flow loss 按有效维度求均值。
FP32 参数与 Adam 状态分片，BF16 前向，activation checkpoint；非最后一个 microstep 使用 no_sync。
按 guide_plus.md 的 step1000 预实验建议，保留第一段 16000-step 学习率调度，实际在1000步停止；LR=1e-5，warmup500。

## 启动和恢复

```bash
bash preflight.sh
bash train.sh --dry-run
bash train.sh --execute --smoke --smoke-updates 4 --no-after-eval
bash run_1000steps.sh
```

正式 run：`runs/single_latent16_dino_step1000_2gpu07_20261019/`。
短程验证与正式训练隔离，正式训练从原始 LaWAM W0 重新初始化；验证更新不计入1000步。

```bash
bash train.sh --execute --resume /absolute/path/to/run/checkpoints/epoch_... --after-eval
```

每250步及终点保存完整检查点，保留最近两份。保存每rank FP32参数、Adam状态、RNG、epoch/cursor/step、BF16部署权重、processor、数据与代码hash。
`latest.json`用于恢复，`final.json`只在完成1000步后生成；`status.json`与`metrics.jsonl`用于监控。
旧的1000epoch配置和实现已归档至 `archive_1000epochs_before_step_budget_20261019/`。

## 推理

输入仅当前主／腕 RGB 和指令；保留相同的 latent+DINO 条件。沿用经过验证的 Plus 夹爪转换：模型 1=open，客户端只做一次 1-g 后交给旧 rollout core。
正式全量清单 10,030 个任务配置，各自固定初始状态 index0，一次 episode；输出总体、suite、category 成功率。
评估仍使用 GPU 0、7，各卡一个模型服务，配套独立仿真客户端。

```bash
./evaluate.sh --dry-run
./evaluate.sh --execute --gpus 0,7 --scope full
```

未完成全量评测时，主 success_rate 留空，不用已完成子集冒充最终精度。

## 当前后台运行

`tmux` 会话：`a0_plus_step1000_gpu07`。流程状态：`workflow_step1000_gpu07.json`。
流程先等待本次双卡短程验证，再执行导出推理一致性检查，全部通过后从W0开始正式1000步训练，结束后全量评测。
正式训练日志：`logs/train_step1000_2gpu07_20261019.log`。短程日志：`logs/smoke_2gpu07_step1000_20261019.log`。
短程验证失败时停止后续流程；每个阶段的失败会记录到流程状态文件。
