# A3 LIBERO-Plus

完整Teacher固定为libero-plus/runs/stage_I_20260923/retained_checkpoints/step_005000；Student公共权重和query逐张量匹配libero-plus/runs/stage_II_20260925/step_000000。两组各8槽，gate模块从构造时移除，保留双路径及四层双角色蒸馏。不会从训练后的A4继续微调。

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
bash train.sh --stage II --dry-run
# 后续正式启动示例，本次未执行
bash train.sh --stage II --execute --gpus 0,1
~~~

不需要重训完整Stage I，本入口拒绝--stage I。来源step必须分别为5000／0，公共Teacher参数必须逐张量相同；只允许跳过gate键。第二段训练4000次更新，最终权重为protocol_plus_v1/seed_2026/II/step_004000。

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
