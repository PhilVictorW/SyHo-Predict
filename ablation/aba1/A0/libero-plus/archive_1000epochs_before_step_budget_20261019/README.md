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

## 数据和真实 epoch

实际数据：`/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_without_subgoal/libero-plus/data/indexed_rlds`。
沿用 seed2026 的按 episode 划分：训练 13,630 episodes / 2,125,407 anchors；验证 717 episodes / 112,629 anchors。
仅解码当前双视角，不读取未来图像。动作标签使用从 t 开始的最多 8 步示教动作；尾部有效动作 mask 保留。

正式目标为 **1000 个完整 epoch**，不是 1000 updates。
GPU 0、1、3、7，每卡一个 FSDP rank；micro batch=4，累积=2，完整 update 的全局 batch=32。
每 epoch 66,419 次 optimizer update，合计 66,419,000 次、2,125,407,000 个真实 anchors。
每 epoch 使用一个完整随机排列，所有真实 anchors 恰好出现一次；末尾 31 anchors 按实际数量归一化，dummy 的 loss 权重为零。
FP32 参数与 Adam 状态分片，BF16 前向，非重入 activation checkpoint。LR=1e-5，warmup500 updates，然后按整个正式预算 cosine 到 5%。

## 启动和恢复

```bash
./preflight.sh
./train.sh --dry-run
./train.sh --execute --smoke --no-after-eval
./train.sh --execute --after-eval
```

正式 run：`runs/single_latent16_dino_1000epochs_4gpu0137_20261019/`。
启动器成功完成 1000 epoch 后，自动加载 final.json 指向的最终权重进行全量 LIBERO-Plus 推理。不能把 smoke 权重作为正式最终模型。

```bash
./train.sh --execute --resume /absolute/path/to/run/checkpoints/epoch_... --after-eval
```

检查点保存每 rank FP32 参数、Adam 状态、RNG、epoch/cursor/step、模型导出、processor、来源和代码 hash。默认每 1000 更新和每 epoch 末保存，仅保留最近两份完整 checkpoint；最终模型按预算锁定，不以测试集择优。
`latest.json` 用于恢复，`status.json` 与 `metrics.jsonl` 用于监控。smoke 默认 12 次更新不计入正式训练。

## 推理

输入仅当前主／腕 RGB 和指令；保留相同的 latent+DINO 条件。沿用经过验证的 Plus 夹爪转换：模型 1=open，客户端只做一次 1-g 后交给旧 rollout core。
正式全量清单 10,030 个任务配置，各自固定初始状态 index0，一次 episode；输出总体、suite、category 成功率。
评估仍使用 GPU 0、1、3、7，各卡一个模型服务，配套独立仿真客户端。

```bash
./evaluate.sh --dry-run
./evaluate.sh --execute --gpus 0,1,3,7 --scope full
```

未完成全量评测时，主 success_rate 留空，不用已完成子集冒充最终精度。

四卡执行每卡 micro batch 4，累积 2 次再更新；非最后一次 microstep 使用 FSDP no_sync，减少梯度同步。每个 anchor 先按其有效动作维度求均值，再按真实 anchor 数归一化，保证尾部与 padding 不改变样本权重。
