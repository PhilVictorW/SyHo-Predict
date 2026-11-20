# 第0步预检修复

第一次预检在反向传播时触发CheckpointError，尚无优化器更新、训练metrics或权重保存。原日志保留在failed_preflight_equal。

Plan-off分支在no_grad中执行仍可训练的Goal/Local模块；关闭该分支的autocast权重缓存，避免其无梯度转换权重被随后有梯度的分支复用。保留梯度检查点和元数据一致性检查，不屏蔽错误。重新执行输出等价性、冻结参数与有限梯度预检。

## 修复验证结果

修复后完整反向传播预检通过；候选输出差异0、关闭角色与原始基线差异0，冻结参数无梯度。详见runs/equal/preflight.json。首次失败发生在任何优化器更新之前。
