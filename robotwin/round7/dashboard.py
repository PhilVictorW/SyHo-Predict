import json,time
from pathlib import Path
R=Path(__file__).resolve().parent/'audit'
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,json.JSONDecodeError):return {}
while True:
    s=read(R/'status.json');n=s.get('completed',0);total=s.get('total',128)
    print('\033[2J\033[H',end='')
    print('RoboTwin round7 | current-only student initialization | GPU0 | ZERO OPTIMIZER UPDATES')
    print(f"[{('#'*int(40*n/total)).ljust(40,'.')}] {n}/{total} {s.get('phase','queued')}")
    print('cohort:',s.get('cohort','-'),'sample:',s.get('sample','-'),'train calibration:',s.get('calibration_completed',0),'/32')
    if s.get('updated_at'):print('heartbeat age:',round(time.time()-s['updated_at'],1),'s')
    if s.get('error'):print('ERROR:',s['error'])
    d=read(R/'results.json') if s.get('phase')=='completed' else read(R/'partial.json')
    rows=d.get('rows',[])
    for c in ['diagnostic','fresh']:
        rr=[r for r in rows if r['cohort']==c]
        if not rr:continue
        print('\n'+c+f' | {len(rr)}/64 | incomplete means are not conclusions')
        for k in rr[0]['metrics']:
            a=[sum(x['metrics'][k][m] for x in rr)/len(rr) for m in ['position_cm','rotation_deg','gripper_mae']]
            print(f'{k:16s} position {a[0]:.6f} cm | rotation {a[1]:.6f} deg | gripper {a[2]:.7f}')
        if d.get('completed'):
            for k,v in d['summary'][c]['comparisons'].items():print(k,'screen:',v['exploratory_screen_passed'],'excess CI:',v['bootstrap_95'])
    if rows:print('\nreproduction max:',d.get('reproduction_max_abs'),'| future guards passed:',d.get('future_access_guard_passed'))
    print('\nOnly current images/instruction; no future masks or remaining length in student input.')
    print('Old Stage II3000 and online rollout remain paused. Ctrl-b d detaches.',flush=True)
    time.sleep(5)
