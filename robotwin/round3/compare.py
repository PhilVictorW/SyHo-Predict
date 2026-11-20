"""Read-only comparison against round2; invoked by the progress reporter."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'round2/runs/teacher_compat36_20260925'
NEW=ROOT/'round3/runs/teacher_terminal_hold36_20260926'

def read_rows(path):
    rows=[]
    if path.exists():
        for l in path.read_text().splitlines():
            try:rows.append(json.loads(l))
            except json.JSONDecodeError:pass
    return rows

def generate():
    newstatus=json.loads((NEW/'status.json').read_text()) if (NEW/'status.json').exists() else {'phase':'pending','step':0}
    diagnosis=json.loads((ROOT/'round3/diagnostics/tail_control.json').read_text())
    idx=json.loads((OLD/'samples.json').read_text())['train_indices']
    episodes=json.loads((ROOT/'cache/index.json').read_text())['episodes']
    valid=set(np.random.default_rng(2026).permutation(len(episodes))[:round(len(episodes)*.05)].tolist())
    train=[e for e in episodes if e['episode_index'] not in valid];offsets=np.cumsum([0]+[e['length'] for e in train])
    remaining=[]
    for i in idx:
        j=int(np.searchsorted(offsets,i,side='right')-1);remaining.append(int(train[j]['length']-(i-offsets[j])))
    association={}
    for name,path in [('old',OLD),('fixed',NEW)]:
        rows=[r for r in read_rows(path/'metrics.jsonl') if r.get('split')=='train']
        association[name]={}
        for has_tail in [False,True]:
            losses=[row['flow'] for row in rows if any(n<36 for n in remaining[4*(row['step']-1):4*row['step']])==has_tail]
            association[name]['tail_batch' if has_tail else 'interior_batch']=dict(count=len(losses),mean=mean(losses) if losses else None)
    paired=[]
    for p in sorted(NEW.glob('validation_*.json')):
        op=OLD/p.name
        if op.exists():
            a=json.loads(op.read_text());b=json.loads(p.read_text());paired.append(dict(step=b['step'],old=a['mean']['full'],fixed=b['mean']['full'],base=b['mean']['base']))
    payload=dict(new_status=newstatus,tail_samples=sum(x<36 for x in remaining),total_samples=len(remaining),
                 train_loss_association=association,matched_velocity_summary=diagnosis.get('summary'),paired_validation=paired)
    (ROOT/'round3/comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    text=['# 轨迹尾部处理修正：受控对照', '',f"新实验状态：{newstatus['phase']}，步数{newstatus['step']}。",'',
          '已定位的不一致：原始绝对EEF动作在尾部重复最后一帧，保持36个有效token；旧自定义采样器缩短有效token，改变自注意力输入。',
          '修正后训练保持36个token和末值补齐；物理指标仍只计真实采集的动作帧。', '',
          f"上一轮800个anchor中有{sum(x<36 for x in remaining)}个不足36帧。原始/修正实验按相同采样序列比较。",'',
          '## 单样本配对诊断', '',str(diagnosis.get('summary','运行中')), '',
          '同图像、条件、噪声、时间；两种设置的误差都只计算原本存在的动作帧。没有用补齐帧稀释诊断误差。', '',
          '## 完整36步预测的物理误差（只统计真实帧）', '',
          '|步数|设置|位置cm|姿态°|夹爪MAE|','|---|---|---:|---:|---:|']
    for row in paired:
        for name in ['old','fixed','base']:
            m=row[name];text.append(f"|{row['step']}|{name}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
    text+=['','只有修正后的真实未来稳定优于关闭/错配未来，才有依据继续学生蒸馏。当前不宣称在线成功率恢复。',
           'GPU2、5现有Stage II及旧源码均未修改，旧自动在线评测继续暂停。', '',
           '本修正只处理已确认的数据/注意力窗口问题，不证明Goal/Local设计本身已经有效。']
    (ROOT/'round3/COMPARISON.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':generate()
