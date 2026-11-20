# 第五轮：表示与动作注入参数交叉替换

固定等权组及其初始权重，32条相同轨迹、一个噪声；只读、不更新优化器。
representation为future resampler；injection包括Goal/Local投影、注意力与逐层scale。原始骨干、Plan及总门控固定，辅助头不参与动作输出。

|组合|位置cm|旋转°|夹爪MAE|
|---|---:|---:|---:|
|rep_initial__inj_initial__full|0.281697|0.732469|0.0042361|
|rep_initial__inj_initial__wrong|0.281852|0.730696|0.0042134|
|rep_initial__inj_trained__full|0.299371|0.728978|0.0046783|
|rep_initial__inj_trained__wrong|0.298714|0.730424|0.0046564|
|rep_trained__inj_initial__full|0.292385|0.724597|0.0045251|
|rep_trained__inj_initial__wrong|0.290226|0.720409|0.0045138|
|rep_trained__inj_trained__full|0.294791|0.731063|0.0045963|
|rep_trained__inj_trained__wrong|0.294651|0.733287|0.0046159|
|base|0.302204|0.732968|0.0047284|

正收益表示参数更新降低误差；区间为按轨迹bootstrap的探索性区间，未校正多重比较。

representation_update_with_initial_injection: -0.028588, 95% [-0.03734205036242212, -0.020147028148509893]

injection_update_with_initial_representation: -0.049081, 95% [-0.061163390851531226, -0.03783760665947516]

injection_update_with_trained_representation: -0.010614, 95% [-0.015820935381568618, -0.006028277536075829]

representation_update_with_trained_injection: 0.009879, 95% [0.0036671816533040747, 0.0160645452795984]

最终模型与base/wrong逐样本重现原评估（容差1e-6）。交叉组合可能改变模块间配合，只能诊断这组权重，不能证明某个模块在所有训练配置下均有害。
