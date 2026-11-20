# A0 LIBERO-Plus

W0直接初始化Goal8＋Local8当前观测策略，保留gate与双路径，只优化action flow loss。没有Teacher、未来图像解码或蒸馏。正式预算为5000＋4000次更新；可选1000-update pilot独立标记，不代表1000 epochs。

公共实现、协议、保存清理与验证边界见[plus_common/README.md](../../plus_common/README.md)。设计见[guide_plus.md](../../guide_plus.md)。本次未运行正式训练或推理。

## 预检

进入本目录后：

~~~bash
bash preflight.sh --report preflight_report.json
bash evaluate.sh --scope full --dry-run
~~~

不带--execute时，训练和评测入口均仅显示计划。两段的有效batch均32，LR总长分别16000／10000，停止分别5000／4000，不能用停止步数缩短LR曲线。

## 训练顺序

~~~bash
# 只显示计划，不启动
bash train.sh --stage action1 --dry-run
bash train.sh --stage action2 --dry-run

# 下列带execute的命令供后续启动；本次未执行，GPU编号按当时资源替换。
bash train.sh --stage action1 --execute --gpus 0,1
bash train.sh --stage action2 --execute --gpus 0,1
~~~

第二段自动且只允许继承本目录protocol_plus_v1/seed_2026/action1/step_005000；保留模型，重建optimizer和采样器，不加载任何Teacher。最终权重为action2/step_004000，cumulative_step=9000。

如后续选择先做1000次更新的pilot：

~~~bash
bash train.sh --stage action1 --stop-after-updates 1000 --execute --gpus 0,1
bash evaluate.sh --scope pilot --checkpoint protocol_plus_v1/seed_2026/action1/step_001000 --execute --gpus 0
bash train.sh --stage action1 --resume protocol_plus_v1/seed_2026/action1/step_001000 --execute --gpus 0,1
bash train.sh --stage action2 --execute --gpus 0,1
~~~

pilot沿用16000-step曲线，续训到第一段5000时不会重置模型／optimizer；开始action2时才按方案重建优化器。pilot1000之后继续训练会清理旧optimizer，因此进一步恢复应读取该阶段latest.json。

## 完整恢复

在同阶段添加--resume及完整checkpoint目录；从latest.json读取最新路径。恢复所有rank RNG、sampler游标、FP32 master/Adam状态与scheduler。旧模型可能仅能推理，resumable=false时不可当作完整恢复状态。

## 推理

~~~bash
# 固定28项联通检查；以下真正启动示例本次未执行。
bash evaluate.sh --scope smoke --execute --gpus 0
# 完整10030项，固定本组最终checkpoint
bash evaluate.sh --scope full --execute --gpus 0
~~~

结果落在本组protocol_plus_v1/seed_2026/evaluation，输出逐episode、错误、results.json与matrix.csv。不使用8380历史子集；无Actor。尚未实测，不能填入89.62%等历史分数。

I/action1需要两张不同GPU；II/action2在同两张GPU上启动4个Gloo ranks。GPU0,1只是命令示例，没有自动占卡。检查时磁盘空闲约174 GiB，完整checkpoint包含FP32恢复状态，实际启动前按容量安排；入口会检查空间。
