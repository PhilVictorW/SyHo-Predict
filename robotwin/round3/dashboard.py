import json
import time
from pathlib import Path
RUN=Path(__file__).resolve().parent/'runs/teacher_terminal_hold36_20260926'
OLD=Path(__file__).resolve().parents[1]/'runs/stage_II_20260925/status.json'
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def show():
    s=read(RUN/'status.json');step=s.get('step',0);total=s.get('total_steps',200)
    print('\033[2J\033[H',end='')
    print('RoboTwin round 3 terminal-hold teacher | GPU0 | no online evaluation')
    print(f"[{('#'*int(40*step/total)).ljust(40,'.')}] {step}/{total} {s.get('phase','waiting')}")
    print('Updated',time.strftime('%F %T',time.localtime(s.get('updated_at',time.time()))),'PID',s.get('pid'),
          'micro',s.get('micro',0),'validation',s.get('validation_sample',0),'/',s.get('validation_total',0))
    if s.get('error'):print('ERROR:',s['error'])
    rows=[]
    if (RUN/'metrics.jsonl').exists():
        for line in (RUN/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    train=[r for r in rows if r['split']=='train']
    if train:
        r=train[-1];print(f"action loss {r['flow']:.7f} | gate {r['gate']:.7f} | lr {r['lr']:.2g} | step {r['seconds']:.1f}s")
        print('Recent action loss:', ' '.join(f"{r['step']}:{r['flow']:.5f}" for r in train[-10:]))
    print('\nFixed 16-sample validation, full 36-step actions:')
    print(' step   condition       position/cm  rotation/deg   gripper/MAE')
    for row in [r for r in rows if r['split']=='validation']:
        for variant,m in row['mean'].items():
            print(f"{row['step']:5}   {variant:14}  {m['position_cm']:11.5f}  {m['rotation_deg']:12.5f}  {m['gripper_mae']:12.6f}")
    old=read(OLD)
    print(f"\nExisting GPU2,5 Stage II: {old.get('step','?')}/{old.get('total_steps','?')} {old.get('phase','?')}")
    print('Online queue paused. Ctrl-b d detaches. Refresh: 5 seconds.',flush=True)
if __name__=='__main__':
    while True:
        show();time.sleep(5)
