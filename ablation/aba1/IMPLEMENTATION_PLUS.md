# LIBERO-Plus A0–A3 代码交付记录

日期：2026-10-19。状态：**实现和离线检查完成，停在正式训练／实验推理启动之前。**

## 已落地

| 组 | 入口目录 | 实现 |
|---|---|---|
| A0 | [A0/libero-plus](A0/libero-plus/README.md) | W0→双query当前策略；gate保留；action-only两段训练；无Teacher、未来读取或蒸馏 |
| A1 | [A1/libero-plus](A1/libero-plus/README.md) | 从W0训练Local-only Teacher16；对应Student16；仅Local蒸馏及Local→Action |
| A2 | [A2/libero-plus](A2/libero-plus/README.md) | 从W0训练Goal-only Teacher16；对应Student16；仅Goal蒸馏及Goal→Plan |
| A3 | [A3/libero-plus](A3/libero-plus/README.md) | 完整Teacher5000＋配对Student0公共权重/query；从构造开始无gate；重新训练Student |

各目录包含配置、Python及shell训练／评测／预检入口、组别README与预检报告。公共实现位于[plus_common](plus_common/README.md)，没有改动原goal_local、原Plus运行、Stage III、普通LIBERO实验或其权重。

A1/A2不存在的角色不会进入数据解码、Resampler、VLM角色切片、动作投影／attention或蒸馏。单尺度16槽、双尺度8＋8槽；旧query、Plan4、16层DiT、8步动作不变。A3公共初始化通过逐张量检查，只移除gate。所有组无Actor、无RL。

## 已完成的验证

| 检查 | 结果与证据 |
|---|---|
| CPU行为回归 | 23项通过，见[日志](plus_common/audits/unit_tests_23.log)；包括真实微型Qwen层、真实DiT、角色梯度／隔离、未来有效性不泄漏、模型存取、LR trace、sampler恢复、Gloo两进程FP32平均／恢复、夹爪转换及完整分母 |
| checkpoint资产完整性 | 新增1项通过，见[日志](plus_common/audits/checkpoint_integrity_test.log)；processor损坏会拒绝，旧纯模型快照不能冒充完整恢复 |
| 实际CLI | 7个训练阶段计划＋8个评测计划，共15项dry-run通过，见[记录](plus_common/audits/cli_dryruns.json)；未创建正式实验目录 |
| 真实权重 | [来源审计](plus_common/audits/real_sources.json)：Teacher step5000与Student step0的1,038个公共张量完全相同；记录W0/Teacher/Student0完整SHA256；核验query为8×2048 |
| 真实数据 | 四组同一train=2,125,407 anchors、13,630 episodes；val=112,629 anchors、717 episodes；按组实际解码首部／尾部样本，未越episode或读取被删除角色 |
| 正式评测清单 | 10030项，suite计数2402／2518／2591／2519，task与分类清单hash已记录 |
| 仿真依赖及初始状态 | [导入审计](plus_common/audits/simulator_import.json)：指定Plus包、历史rollout core、4个suite任务数、初始状态index0及250／300／320／550步上限均通过；未创建环境、未执行动作 |

CPU测试中一次hook计数把activation checkpoint反向重算计入前向，已修正测试范围后通过。仿真导入检查发现历史启动脚本依赖额外ImageMagick sysroot及LIBERO_CONFIG_PATH，已显式接入新环境配置；还固定了历史LaWAM_aba2下rollout core的导入路径，避免同名examples包加载到另一份实现。没有安装依赖或更改系统库。

## 运行约定

- 默认运行训练／评测入口只显示计划；正式启动必须显式传 --execute 和GPU编号。各组README列出了确切命令。
- A0正式5000＋4000次更新；可选1000-update pilot单列。**未把“1000 epoch”自动替换为1000 steps，也未启用literal1000epoch长训练。**
- 第一段LR总长16000，第二段10000；停止预算单独配置。I/action1为2个NCCL ranks，II/action2为4个Gloo ranks，均有效batch32。
- A0第二段从本组action1/step_005000接续模型、重建优化器与采样器；最终action2/step_004000的cumulative_step=9000。
- A1/A2第二段只能继承各自新Teacher5000；现在这些文件尚未生成，dry-run显示pending是正常依赖，正式入口不会借用完整Teacher跳过Stage I。
- checkpoint通过临时目录完成模型、processor、动作语义、FP32 optimizer、scheduler、各rank RNG及采样游标后原子提交；新状态提交后才清理旧恢复状态。
- 保留最近两份模型与A0要求的step0/1000模型，每阶段仅最新状态支持完整恢复。所有来源、配置、代码、依赖版本、数据／processor hash均记录并在恢复时核对。
- 评测固定28项smoke与10030项full；动作协议一次1-g、chunk8、Euler10、BF16初始噪声；错误有限重试，未完成时总成功率留空，单列已完成子集统计。
- 当前磁盘检查约174 GiB可用；正式启动入口要求至少90 GiB，保存前另检查实际需求。四组的完整恢复状态不应未经容量安排同时常驻。

## 验证边界与停止位置

本次**没有**运行全尺寸GPU前向／反向、模型服务、环境reset/step或真实rollout，没有开始A0–A3训练，没有产生新的消融精度。

GPU显存／吞吐、物理夹爪联通及真实A4端到端回归需要后续正式运行时检查；微型回归不能替代这些测量。历史86.57%／89.62%未填入新结果表。

当前正停在各组README中带 --execute 的启动命令之前。主设计仍保留于[guide_plus.md](guide_plus.md)，其“本次只写文档”描述属于前一轮文档任务；本次新增实现与实际验证以本交付记录为准。
