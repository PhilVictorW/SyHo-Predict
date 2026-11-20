# LIBERO-Plus A0–A3：实现及启动边界

本次交付代码并完成离线检查，**没有启动正式训练、模型服务或仿真 rollout**。设计依据：[guide_plus.md](../guide_plus.md)、[abalation.md](../../../reference/abalation.md)。脚本默认只显示计划，显式添加 --execute 才启动。

## 实现范围

四组目录各有 config.json、train.py/.sh、evaluate.py/.sh、preflight.py/.sh、README.md 和 preflight_report.json。公共代码集中在 plus_common，避免复制四套训练器。

| 组 | 模型／来源 | 正式终点 |
|---|---|---|
| A0 | W0直接建立Goal8＋Local8 Student，保留gate，只读当前帧，仅flow loss | action1 5000＋action2 4000，累计9000 |
| A1 | 独立Local16 Teacher，仅t+8主／腕未来；Local16 Student单角色蒸馏 | Teacher5000 → Student4000 |
| A2 | 独立Goal16 Teacher，仅t+[24,40,56,72,88]主视角未来；Goal16 Student | Teacher5000 → Student4000 |
| A3 | 完整Teacher5000＋历史未训练Student0公共参数/query，结构上删除gate | 新Student4000 |

A1/A2不构造另一角色的Resampler、query、projection或attention。全部保留旧Action8＋Flow8、4个Plan及16层DiT。Goal只在DiT第1/5/9/13层注入Plan，Local在全部16层注入Action。角色总数16、hidden2048，动作仍为8步。A0不加载Teacher，全部配置不接入Stage III/Actor。

model.py、action.py、roles.py为隔离实现；起始结构取自同目录已有的通用角色模块，其原文件hash记录在scaffold_sources.json。没有使用普通LIBERO训练器或修改它。基础Qwen/DiT、动作归一化、FP32 optimizer及只读数据索引沿用原项目；运行时记录实际导入源码hash。

## 训练协议

- 完整1024分片indexed RLDS，train=2,125,407 anchors／13,630 episodes，val=112,629 anchors／717 episodes；split_seed=2026与train_seed分开，不使用active_window。
- 历史val存在同轨迹扰动版本，仅作拟合／数值诊断。正式模型固定预算终点，不按benchmark成绩择优。
- I/action1：2个NCCL ranks × batch2 × accumulation8，16000-step LR曲线，停止5000。
- II/action2：4个CPU Gloo ranks，两ranks共享每张GPU，batch1 × accumulation8，10000-step曲线，停止4000。
- 有效batch32，LR1e-5，warmup500，AdamW betas=(0.9,0.95)、eps1e-8、weight_decay1e-8、clip1.0、flow repeat2。
- 第一段保留模型梯度裁剪→FP32 master更新→LambdaLR.step顺序；第二段保留CPU FP32全局平均→裁剪→rank0更新→广播。日志lr_used明确记录本次实际LR，不混淆旧日志的next LR。
- 默认不共享frozen encoder、不offload saved tensors，与历史Stage-II参数一致；可显式开启相应CLI开关。视觉与DINO冻结，Teacher全冻结，训练参数不共享存储。
- 新公共模块使用语义名称派生RNG，避免分支构造顺序扰动公共初始化；A3公共参数和query直接逐张量核对历史step0。
- A0 action2继承本组action1/step_005000的模型，重建optimizer、LR、sampler与RNG；这不是蒸馏。--resume则恢复完整训练状态。
- Teacher每1000步保存／250步诊断；Student每500步保存／诊断；均保存step0。
- 保留最近两个模型、A0 action1的step0和step1000，以及每阶段最新一份完整恢复状态。新checkpoint完整提交后才清理旧恢复状态。保留的旧模型标记resumable=false，续训应使用该阶段latest.json中的完整checkpoint。

**单位没有被偷偷转换。** 按guide实现A0正式5000＋4000次optimizer更新，以及显式可选的 --stop-after-updates 1000 pilot。pilot不是1000 epochs。literal 1000 epochs的独立长训练方案没有启用，不能用本命令声称已完成它。

## 使用入口

进入对应A*/libero-plus：

~~~bash
bash preflight.sh --report preflight_report.json
bash train.sh --stage I --dry-run
bash evaluate.sh --scope full --dry-run
~~~

A0用action1/action2，A3仅用II；A1/A2用I、II。尚未训练出的上游checkpoint在dry-run中标记pending，正式启动会拒绝缺失或不匹配的来源。

**真正启动**需增加 --execute --gpus 两个不同物理GPU编号；完整顺序见各组README。不会自动选卡或顺带启动评测。训练／评测使用独立锁，服务采用动态空闲端口，只终止自己创建的子进程。

--stop-after-updates可以提前停止；--resume读取同阶段完整checkpoint，允许pilot1000继续至原定5000，不改变LR总长。恢复检查配置、来源SHA256、model/optimizer hash、源码、依赖版本、数据与processor资产、每rank RNG及sampler游标。

输出目录：

~~~text
A*/libero-plus/protocol_plus_v1/seed_2026/
  I/step_005000/           # A1/A2 Teacher
  II/step_004000/          # A1/A2/A3 Student
  action1/step_005000/     # A0第一段
  action2/step_004000/     # A0第二段，累计step9000
  evaluation/...
~~~

A0最终目录名称是第二段step_004000，marker与日志cumulative_step=9000；不可误读为只训练4000步。

## 评测

默认加载本组预先规定的最终Student，受控路径、format、stage、完整标记、hash及Plus动作语义均校验。没有伪装历史Stage-II路径或移除旧保护。

~~~bash
# 这些是后续启动示例，本次未执行。GPU编号需按当时资源选择。
bash evaluate.sh --scope smoke --execute --gpus 0
bash evaluate.sh --scope full --execute --gpus 0
~~~

smoke固定为每suite×类别第一项，共28项，不依成绩抽样。full锁定最终checkpoint。A0 pilot1000需显式用 --scope pilot --checkpoint .../action1/step_001000，单独记录。

正式全量10030任务，每项episode0，初始状态index0，env seed0，稳定等待10步，policy上限250/300/320/550；当前双视角256、chunk8、Euler10、BF16初始噪声，seed公式沿用历史。原core逐动作步判断终止。模型夹爪1=open只转换一次1-g后交给旧core；不重复反归一化、不原地修改缓存动作；默认不录像。

基础设施错误最多3次尝试，任务失败不会重试至成功。未完成时总success_rate=null，另列completed_success_rate；不把部分分母当作全量成绩。输出逐episode、初始状态hash、动作步数、四suite、七类别、4×7矩阵、模型与协议hash。

## 验证及边界

见audits/、各组preflight_report.json和[交付记录](../IMPLEMENTATION_PLUS.md)。CPU测试包含微型真实Qwen block、真实DiT、梯度、保存加载、Gloo双进程同步恢复和协议检查；真实权重仅做只读元数据／张量／hash核对，真实数据只做解码采样。

**未做全尺寸GPU前向／反向、模型服务启动或环境rollout。** 不声称已测得GPU显存、吞吐、物理夹爪联通或实验精度。微型完整配置回归不能替代真实A4端到端评测；86.57%/89.62%不能填进本轮未运行结果。

检查时磁盘可用约174 GiB。FP32 master和Adam状态大于模型文件，启动至少需90 GiB空闲，保存前也按实际参数估算空间。四组不能据此认为可以同时保留全部恢复状态；实际启动时需安排磁盘与归档。A0/A1/A2第二段等待各自上游训练产物属于正常依赖。

**最后尚未执行的步骤：用户安排资源后，使用带 --execute 的训练／评测命令启动。**
