import json,time
from pathlib import Path
R=Path(__file__).resolve().parent/'query_pilot'
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,json.JSONDecodeError):return {}
def rows():
    out=[]
    try:
        for line in (R/'metrics.jsonl').read_text().splitlines():
            try:out.append(json.loads(line))
            except json.JSONDecodeError:pass
    except OSError:pass
    return out
while True:
    s=read(R/'status.json');data=rows();tr=[x for x in data if x.get('split')=='train'];val=[x for x in data if x.get('split')=='validation'];step=s.get('step',0)
    print('\033[2J\033[H',end='')
    print('RoboTwin round7 | GPU0 | query-only student pilot | 32,768 trainable parameters')
    print(f"[{('#'*int(40*step/200)).ljust(40,'.')}] {step}/200 | {s.get('phase','queued')} | micro {s.get('micro',0)}/4")
    if s.get('updated_at'):print('heartbeat:',round(time.time()-s['updated_at'],1),'s ago')
    if s.get('error'):print('ERROR:',s['error'])
    if tr:
        x=tr[-1];print(f"action loss {x['action_loss']:.8f} | grad {x['grad_norm']:.4g} | lr {x['lr']:.3g}")
        kd=x['kd_diagnostic'];print('KD diagnostics (weight=0): local',kd['local'],'goal',kd['goal'],'| valid',kd['local_valid'],kd['goal_valid'])
        print('\nAction loss means (successive 20-step blocks):')
        for start in range(0,len(tr),20):
            part=tr[start:start+20];mean=sum(y['action_loss'] for y in part)/len(part)
            print(f"{part[0]['step']:3}-{part[-1]['step']:3}: {mean:.8f}")
    if val:
        print('\nFixed validation (every 50 steps):')
        for x in val:
            m=x['mean'];print(f"{x['step']:3}: pos {m['position_cm']:.6f}cm | rot {m['rotation_deg']:.6f}deg | grip {m['gripper_mae']:.7f} | drift {x['initial_action_drift_mse']:.7g}")
    checkpoints=sorted(p.parent.name for p in R.glob('step_*/checkpoint_complete.json'));print('\nComplete checkpoints:',', '.join(checkpoints) or 'none yet')
    print('Actor/backbone/gate frozen; saves every50; >10% regression twice stops this pilot.')
    print('Old Stage II3000 and online rollout remain paused. Ctrl-b d detaches.',flush=True)
    time.sleep(5)
