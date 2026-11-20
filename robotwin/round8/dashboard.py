import json,time
from pathlib import Path
R=Path(__file__).resolve().parent/'paired_pilot'
def read(name):
    try:return json.loads((R/name).read_text())
    except (OSError,json.JSONDecodeError):return {}
def rows():
    result=[]
    try:
        for line in (R/'metrics.jsonl').read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    except OSError:pass
    return result
while True:
    s=read('status.json');data=rows();tr=[x for x in data if x.get('split')=='train'];va=[x for x in data if x.get('split')=='validation']
    step=s.get('step',0)
    print('\033[2J\033[H',end='')
    print('RoboTwin round8 | physical GPU0 | paired query-only vs query+RelevanceGate')
    print(f"[{('#'*int(40*step/200)).ljust(40,'.')}] {step}/200 steps PER ARM | {s.get('phase','queued')}")
    print('Active arm:',s.get('arm','-'),'| micro:',s.get('micro',0),'/4 | fixed validation:',s.get('validation_sample',0),'/16')
    if s.get('updated_at'):print('Heartbeat:',round(time.time()-s['updated_at'],1),'s ago')
    if s.get('error'):print('ERROR:',s['error'])
    print('Preflight:',read('preflight.json').get('passed','pending'),'| Initial reproduction:',read('initial_reproduction.json').get('passed','pending'),'| First real gate updates:',read('first_updates.json').get('passed','pending'))
    if tr:
        last=tr[-1];print('\nCurrent action loss:',last['action_loss'])
        print('Local/Goal KD (monitor only, weight0):',last['kd_diagnostic'])
        print('Gate scales:',[round(x,7) for x in last['gate_scales']])
        print('\nAction loss history: successive 20-step means')
        for start in range(0,len(tr),20):
            part=tr[start:start+20]
            means={a:sum(x['action_loss'][a] for x in part)/len(part) for a in ['query_only','query_gate']}
            print(f"{part[0]['step']:3}-{part[-1]['step']:3} | query {means['query_only']:.8f} | query+gate {means['query_gate']:.8f}")
    if va:
        print('\nFixed16 validation: step / arm / position_cm / rotation_deg / gripper_MAE')
        for x in va:
            for a,m in x['mean'].items():print(f"{x['step']:3} {a:11} {m['position_cm']:.6f} {m['rotation_deg']:.6f} {m['gripper_mae']:.7f}")
    saves=sorted(p.name.split('_')[1] for p in R.glob('pair_*_complete.json'));print('\nComplete checkpoint pairs:',', '.join(saves) or 'none yet')
    print('Same initialization/data/noise/LR. Backbone+actor frozen. Both arms stop on sustained >10% regression.')
    print('Old StageII3000 and online rollout remain paused. Ctrl-b d detaches.',flush=True)
    time.sleep(5)
