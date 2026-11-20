import json,time
from pathlib import Path
from dashboard import read,rows
R=Path(__file__).resolve().parent

def generate():
    text=['# 第四轮：未来动作辅助监督对照','',
          '同一个修正版教师200步初始化，各追加200步，原始骨干冻结。这里是教师训练，不是Stage II学生训练。','',
          '|组别|追加步数|阶段|最新action loss|辅助loss|辅助系数|','|---|---:|---|---:|---:|---:|']
    record={}
    for v in ['action_only','future_aux']:
        p=R/'runs'/v;s=read(p/'status.json');data=rows(p/'metrics.jsonl');tr=[x for x in data if x.get('split')=='train']
        q=tr[-1] if tr else {};a=read(R/'assessment'/v/'status.json');d=read(R/'assessment'/v/'results.json')
        text.append(f"|{v}|{s.get('step',0)}|{s.get('phase','queued')}|{q.get('flow','')}|{q.get('auxiliary','')}|{q.get('lambda_aux','')}|")
        record[v]=dict(status=s,assessment=a,summary=d.get('summary'))
    text+=['','## 新轨迹配对评估','', '|组别|条件|位置cm|姿态°|夹爪MAE|','|---|---|---:|---:|---:|']
    for v,d in record.items():
        if d['summary']:
            for cond,m in d['summary']['mean'].items():text.append(f"|{v}|{cond}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
    text+=['','两组共用验证轨迹和噪声种子。判断辅助监督是否改善真实未来的动作收益，同时检查相对原始基线是否退化；不以辅助loss下降本身宣告成功。',
           '没有自动启动学生训练或在线评测。原始SFT可能见过这些示范数据，因此离线结果不等于闭环成功率。']
    if all(x['summary'] for x in record.values()):
        import numpy as np
        ds={v:read(R/'assessment'/v/'results.json') for v in record}
        assert [(x['episode'],x['noise_seed']) for x in ds['action_only']['rows']]==[(x['episode'],x['noise_seed']) for x in ds['future_aux']['rows']]
        keys=['position_cm','rotation_deg','gripper_mae']
        def array(v,c):return np.array([[np.mean([x['metrics'][c][k] for x in ds[v]['rows'] if x['sample']==i]) for k in keys] for i in range(32)])
        base=array('action_only','base').mean(0);delta=((array('action_only','full')-array('future_aux','full'))/np.maximum(base,1e-10)).mean(1)
        draws=np.random.default_rng(4026).integers(0,32,(4000,32));ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        record['aux_vs_control']=dict(normalized_physical_benefit=float(delta.mean()),bootstrap_95=ci,positive_favors_auxiliary=True)
        text+=['',f'辅助组相对action-only组的标准化物理误差收益：{float(delta.mean()):.6f}，轨迹bootstrap 95%区间：{ci}。']
    (R/'REPORT.tmp').write_text('\n'.join(text)+'\n');(R/'REPORT.tmp').replace(R/'REPORT.md')
    (R/'summary.tmp').write_text(json.dumps(record,indent=2)+'\n');(R/'summary.tmp').replace(R/'summary.json')
    return all(x['assessment'].get('phase')=='completed' for v,x in record.items() if v in ['action_only','future_aux'])

if __name__=='__main__':
    while True:
        if generate():break
        time.sleep(30)
