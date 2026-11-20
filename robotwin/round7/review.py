"""Summarize the completed, preregistered student initialization audit."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent

def main():
    d=json.loads((R/'audit/results.json').read_text());assert d['completed'] and len(d['rows'])==128
    assert d['future_access_guard_passed']==64 and d['current_base_max_abs']==0 and d['zero_gate_equivalence_max_abs']==0
    assert d['reproduction_max_abs']<1e-6 and d['frozen_audit']['passed']
    m=d['manifest'];assert m['primary_candidate']=='fixed_queries'
    assert not (set(m['cohorts']['fresh']['episodes'])&set(m['cohorts']['fresh']['excluded_episodes']))
    s=d['summary']['fresh'];passed=s['comparisons']['fixed_queries']['exploratory_screen_passed']
    keys=['position_cm','rotation_deg','gripper_mae'];strata={}
    for cohort in ['diagnostic','fresh']:
        cr=[r for r in d['rows'] if r['cohort']==cohort];strata[cohort]={}
        for name,predicate in [('all_future_valid',lambda r:r['all_future_valid']),('missing_future',lambda r:not r['all_future_valid']),('remaining_at_most_36',lambda r:r['remaining']<=36)]:
            rows=[r for r in cr if predicate(r)]
            strata[cohort][name]=dict(episodes=len({r['episode'] for r in rows}),pairs=len(rows),
                means={v:{k:float(np.mean([r['metrics'][v][k] for r in rows])) for k in keys} for v in ['teacher','base','fixed_queries','mean_queries','zero_queries']} if rows else None)
    (R/'audit/tail_review.json').write_text(json.dumps(strata,indent=2)+'\n')
    pilot_status=json.loads((R/'query_pilot/status.json').read_text()) if (R/'query_pilot/status.json').exists() else {}
    recipe=dict(status='eligible_for_bounded_student_pilot' if passed else 'primary_initialization_screen_failed',
        primary_candidate='fixed_queries',teacher_checkpoint=m['teacher'],source_checkpoint=m['source'],initializers=str(R/'audit/initializers.pt'),
        student_implementation=str(R/'model.py'),student_inputs=m['inference_inputs'],role_valid='always_true',relevance_gate_scale=0.,
        original_LaWM_and_queries_retained=True,stage_II_3000_resumed=False,new_student_training_started=pilot_status.get('step',0)>0,student_pilot_status=str(R/'query_pilot/status.json'),online_evaluation_paused=True,
        offline_only=True,fresh_screen=s['comparisons']['fixed_queries'],requires_terminal_stress_test_before_long_training=True)
    (R/'candidate_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    lines=['# 第七轮完成后的判断','',
        '128/128组只读验证完成：旧32条轨迹×2噪声、新32条轨迹×2噪声。新轨迹排除了此前144条轨迹。训练集校准32条轨迹不与验证集重叠。','',
        '当前观测条件与零门控语言路径逐元素一致；旧教师结果复现误差<1e-6；64个样本的未来路径阻断检查及全部冻结参数检查通过。学生输入只有当前图像与指令，角色有效性恒为真。','',
        '|新轨迹配置|位置cm|姿态°|夹爪MAE|','|---|---:|---:|---:|']
    for k,v in s['means'].items():lines.append(f"|{k}|{v['position_cm']:.6f}|{v['rotation_deg']:.6f}|{v['gripper_mae']:.7f}|")
    lines+=['',f'预先指定主候选fixed_queries的探索性筛查通过：{passed}。',
        f"相对教师的标准化误差增量95%区间：{s['comparisons']['fixed_queries']['bootstrap_95']}。",'']
    if passed:
        lines+=['已有证据支持把第六轮兼容教师的动作改善迁移到不读取真实未来的学生初始化。该结论限于当前离线样本，不能直接宣告闭环成功率恢复，也不证明后验蒸馏有额外收益。',
            '下一步应从此初始化做有限步数、动作目标主导的学生试验；先限制可训练参数并记录动作保持，再决定是否逐步引入角色蒸馏。避免立刻重启旧Stage II3000或进行10000步长训。']
    else:
        lines+=['主候选未通过预定筛查，暂不启动学生训练。优先查看未来缺失/轨迹末段分层，定位移除未来掩码后是否退化。即使次要候选均值更好，也须独立确认后再采用。']
    lines+=['','## 末段覆盖','', '|样本集|子集|轨迹数|配对数|','|---|---|---:|---:|']
    for c,sub in strata.items():
        for k,v in sub.items():lines.append(f"|{c}|{k}|{v['episodes']}|{v['pairs']}|")
    lines+=['','自然抽样中的末段样本有限，长训前仍需专门末段压力测试；分层仅作诊断，未进行多重比较校正。',
        '兼容候选保留原始LaWM和旧query，并非纯aba2消融。此次没有优化器更新；旧Stage II和在线评测继续暂停。']
    if pilot_status.get('step',0)>0:
        lines+=['',f"后续小规模学生试验已启动：{pilot_status.get('step')}/200步，阶段{pilot_status.get('phase')}。只训练两组query（32768参数），每50步保存；action loss为唯一优化目标，Local/Goal KD只作诊断。详见QUERY_PILOT.md。上述零更新结论仅指初始化验证。"]
    (R/'DECISION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(recipe,indent=2))
if __name__=='__main__':main()
