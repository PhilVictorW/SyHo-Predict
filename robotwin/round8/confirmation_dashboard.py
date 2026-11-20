import json,time
from pathlib import Path
R=Path(__file__).resolve().parent/'confirmation'
def read(name):
    try:return json.loads((R/name).read_text())
    except (OSError,json.JSONDecodeError):return {}
while True:
    s=read('status.json');result=read('results.json') or read('partial.json');done=s.get('completed',0)
    print('\033[2J\033[H',end='');print('RoboTwin round8 | GPU0 | independent OFFLINE confirmation + same-checkpoint gate ablation')
    print(f"[{('#'*int(40*done/128)).ljust(40,'.')}] {done}/128 noise pairs | {s.get('phase','queued')} | {s.get('cohort','-')} anchor {s.get('anchor',0)}/32")
    if s.get('updated_at'):print('Heartbeat:',round(time.time()-s['updated_at'],1),'s ago')
    if s.get('error'):print('ERROR:',s['error'])
    print('Preflight reproduction:',read('preflight.json').get('passed','pending'))
    print('Primary: gate200 vs query-only200. gate_off: same trained gate checkpoint, only scales set to zero.')
    for c in ['fresh','tail']:
        rows=[r for r in result.get('rows',[]) if r['cohort']==c]
        if not rows:continue
        episodes=sorted({r['episode'] for r in rows});print(f'\n{c}: {len(rows)} pairs / {len(episodes)} episodes (partial until completed)')
        print('     variant   position_cm   rotation_deg  gripper_MAE')
        for v in ['initial','query_only','query_gate','gate_off','base']:
            means=[]
            for k in ['position_cm','rotation_deg','gripper_mae']:
                em=[sum(r['metrics'][v][k] for r in rows if r['episode']==e)/sum(r['episode']==e for r in rows) for e in episodes]
                means.append(sum(em)/len(em))
            print(f'{v:>12} {means[0]:13.6f} {means[1]:14.6f} {means[2]:12.7f}')
    if result.get('completed'):
        print('\nPrimary improvement supported:',result['primary_improvement_supported'],'| non-regression:',result['primary_nonregression_passed'])
        print('Primary normalized excess CI:',result['summary']['fresh']['comparisons']['query_gate_vs_query_only']['bootstrap_95'])
    print('\nNo optimizer updates. Old StageII3000 and online rollout remain paused. Ctrl-b d detaches.',flush=True)
    time.sleep(5)
