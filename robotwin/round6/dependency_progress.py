import json,time
from pathlib import Path
R=Path(__file__).resolve().parent/'dependency_audit'
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,json.JSONDecodeError):return {}
while True:
    s=read(R/'status.json');n=s.get('completed',0);total=s.get('total',128)
    print('\033[2J\033[H',end='')
    print('Round6 | future-value dependency audit | GPU0 | NO TRAINING / NO ONLINE ROLLOUT')
    print(f"[{('#'*int(40*n/total)).ljust(40,'.')}] {n}/{total} | {s.get('phase','queued')}")
    print('checkpoint:',s.get('checkpoint','-'),'sample:',s.get('sample','-'),'prepared:',s.get('prepared','-'))
    if s.get('updated_at'):print('last heartbeat:',round(time.time()-s['updated_at'],1),'seconds ago')
    if s.get('error'):print('ERROR:',s['error'])
    d=read(R/'results.json') if s.get('phase')=='completed' else read(R/'partial.json')
    rows=d.get('rows',[])
    if rows:
        print('\nIntermediate means are descriptive; conclusions wait for all 32 episodes / 2 noises.')
        for ck in ['round6_control','round5_equal']:
            subset=[r for r in rows if r['checkpoint']==ck]
            if not subset:continue
            print('\n'+ck)
            for k in subset[0]['metrics']:
                means=[sum(r['metrics'][k][m] for r in subset)/len(subset) for m in ['position_cm','rotation_deg','gripper_mae']]
                print(f'{k:20s} position {means[0]:.6f}cm | rotation {means[1]:.6f}deg | gripper {means[2]:.7f}')
        print('published-result reproduction max difference:',d.get('reproduction_max_abs'))
    print('\nOld Stage II3000 and online evaluation remain paused. Ctrl-b d to detach.',flush=True)
    time.sleep(5)
