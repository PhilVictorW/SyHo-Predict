import json,time
from pathlib import Path
from dashboard import read,rows
R=Path(__file__).resolve().parent

def generate():
    text=['# 第六轮：采样轨迹上的动作保持约束','',
          '同一份累计400步教师初始化，各追加200步，固定Plan0/Role3并冻结原始骨干。这里是教师训练，不是Stage II学生训练。','',
          '|组别|追加步数|阶段|最新action loss|辅助loss|辅助系数|','|---|---:|---|---:|---:|---:|']
    record={}
    for v in ['control','retained']:
        p=R/'runs'/v;s=read(p/'status.json');data=rows(p/'metrics.jsonl');tr=[x for x in data if x.get('split')=='train']
        q=tr[-1] if tr else {};a=read(R/'assessment'/v/'status.json');d=read(R/'assessment'/v/'results.json')
        text.append(f"|{v}|{s.get('step',0)}|{s.get('phase','queued')}|{q.get('flow','')}|{q.get('auxiliary','')}|{q.get('lambda_aux','')}|")
        record[v]=dict(status=s,assessment=a,summary=d.get('summary'))
    text+=['','## 新轨迹配对评估','', '|组别|条件|位置cm|姿态°|夹爪MAE|','|---|---|---:|---:|---:|']
    for v,d in record.items():
        if d['summary']:
            for cond,m in d['summary']['mean'].items():text.append(f"|{v}|{cond}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
    text+=['','两组辅助loss均等权并采用同一自适应梯度上限规则；实际辅助系数随训练状态变化。设置上的对照为动作保持系数0或10。两组冻结DiT均保持eval。检查原始基线与冻结参考策略的动作指标。','两组共用验证轨迹和噪声种子。判断辅助监督是否改善真实未来的动作收益，同时检查相对原始基线是否退化；不以辅助loss下降本身宣告成功。',
           '没有自动启动学生训练或在线评测。原始SFT可能见过这些示范数据，因此离线结果不等于闭环成功率。']
    if all(x['summary'] for x in record.values()):
        import numpy as np
        ds={v:read(R/'assessment'/v/'results.json') for v in record}
        assert [(x['episode'],x['noise_seed']) for x in ds['control']['rows']]==[(x['episode'],x['noise_seed']) for x in ds['retained']['rows']]
        keys=['position_cm','rotation_deg','gripper_mae']
        def array(v,c):return np.array([[np.mean([x['metrics'][c][k] for x in ds[v]['rows'] if x['sample']==i]) for k in keys] for i in range(32)])
        base=array('control','base').mean(0);delta=((array('control','full')-array('retained','full'))/np.maximum(base,1e-10)).mean(1)
        draws=np.random.default_rng(4026).integers(0,32,(4000,32));ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        record['retained_vs_control']=dict(normalized_physical_benefit=float(delta.mean()),bootstrap_95=ci,positive_favors_retained=True)
        text+=['',f'动作保持组相对对照组的标准化物理误差收益：{float(delta.mean()):.6f}，轨迹bootstrap 95%区间：{ci}。']
    text += ['', '## 动作保持损失', '', '|组别|最新保持loss|系数|', '|---|---:|---:|']
    for v in ['control','retained']:
        tr=[x for x in rows(R/'runs'/v/'metrics.jsonl') if x.get('split')=='train']
        if tr:text.append(f"|{v}|{tr[-1].get('retention')}|{tr[-1].get('lambda_keep')}|")
    text += ['', '## 训练分项趋势（连续50步均值）', '', '|组别|区间|action loss|位置辅助loss|旋转辅助loss|夹爪辅助loss|', '|---|---|---:|---:|---:|---:|']
    for v in ['control','retained']:
        tr=[x for x in rows(R/'runs'/v/'metrics.jsonl') if x.get('split')=='train']
        for start in range(0,len(tr),50):
            batch=tr[start:start+50]
            if not batch:continue
            avg=lambda key:sum(x['components'][key] for x in batch)/len(batch)
            flow=sum(x['flow'] for x in batch)/len(batch)
            text.append(f"|{v}|{batch[0]['step']}-{batch[-1]['step']}|{flow:.7f}|{avg('aux_position'):.6f}|{avg('aux_rotation'):.6f}|{avg('aux_gripper'):.6f}|")
    for v in ['control','retained']:
        d=read(R/'assessment'/v/'results.json')
        if not d.get('completed'):continue
        import numpy as np
        ep=sorted({x['sample'] for x in d['rows'] if x.get('auxiliary')})
        if not ep:continue
        draws=np.random.default_rng(5026).integers(0,len(ep),(4000,len(ep)))
        component_summary={}
        for key in ['aux_position','aux_rotation','aux_gripper']:
            delta=np.array([np.mean([x['auxiliary']['wrong_future'][key]-x['auxiliary']['full'][key] for x in d['rows'] if x['sample']==i]) for i in ep])
            ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
            component_summary[key]=dict(future_benefit_mean=float(delta.mean()),bootstrap_95=ci)
        record[v]['auxiliary_component_benefit']=component_summary
        text += ['', f'{v}：辅助预测的正确未来收益（未做多重比较校正）：', '', '|分项|均值收益|95%区间|', '|---|---:|---|']
        for key,x in component_summary.items():text.append(f"|{key}|{x['future_benefit_mean']:.6f}|{x['bootstrap_95']}|")
    (R/'REPORT.tmp').write_text('\n'.join(text)+'\n');(R/'REPORT.tmp').replace(R/'REPORT.md')
    (R/'summary.tmp').write_text(json.dumps(record,indent=2)+'\n');(R/'summary.tmp').replace(R/'summary.json')
    return all(x['assessment'].get('phase')=='completed' for v,x in record.items() if v in ['control','retained'])

if __name__=='__main__':
    while True:
        if generate():break
        time.sleep(30)
