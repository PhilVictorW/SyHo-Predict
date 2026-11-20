import json,time
from pathlib import Path
R=Path(__file__).resolve().parent

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def rows(p):
    out=[]
    if p.exists():
        for line in p.read_text().splitlines():
            try:out.append(json.loads(line))
            except json.JSONDecodeError:pass
    return out

def render():
    print('\033[2J\033[H',end='')
    print('RoboTwin round6 | GPU0 | fixed Plan0/Role3 | source cumulative teacher400')
    print('Each arm: +200 teacher updates; anchor retention comparison; no Stage II / online rollout.')
    report={'arms':{}}
    for v in ['control','retained']:
        run=R/'runs'/v;s=read(run/'status.json');a=read(R/'assessment'/v/'status.json')
        data=rows(run/'metrics.jsonl');train=[x for x in data if x.get('split')=='train'];val=[x for x in data if x.get('split')=='validation']
        step=s.get('step',0)
        print(f"\n{v:12} [{('#'*int(25*step/200)).ljust(25,'.')}] {step}/200 {s.get('phase','queued')}")
        if train:
            q=train[-1];print(f"flow {q['flow']:.7f} | aux {q['auxiliary']:.6f} | lambda {q['lambda_aux']:.3g} | gate {q['gate']:.5f}")
        if train and 'components' in train[-1]:
            print('raw components:', train[-1]['components'])
            print('component weights:', train[-1]['component_weights'])
        if train:
            print('retention loss:', train[-1].get('retention'), '| coefficient:', train[-1].get('lambda_keep'))
        if val:
            q=val[-1]['mean']['full'];print(f"val position {q['position_cm']:.5f} cm | rotation {q['rotation_deg']:.5f} deg | gripper {q['gripper_mae']:.6f}")
        if a:print('fresh assessment:',a.get('phase'),a.get('completed_pairs',0),'/64')
        if s.get('error'):print('ERROR',s['error'])
        result=read(R/'assessment'/v/'results.json')
        report['arms'][v]=dict(status=s,assessment=a,last_train=train[-1] if train else None,last_validation=val[-1] if val else None,
                              fresh_summary=result.get('summary'))
    old=read(R.parent/'runs/stage_II_20260925/status.json')
    print(f"\nOld GPU2,5 Stage II: {old.get('phase')} at {old.get('step')}; recovery checkpoint 3000 retained.")
    print('Ctrl-b d detaches; refresh 5 seconds.',flush=True)
    return report

if __name__=='__main__':
    while True:
        render();time.sleep(5)
