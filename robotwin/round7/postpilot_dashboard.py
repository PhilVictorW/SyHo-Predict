"""Live offline confirmation progress; no GPU imports."""
import json
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent/'postpilot_eval'


def read(name):
    try: return json.loads((OUT/name).read_text())
    except (OSError, json.JSONDecodeError): return {}


while True:
    s = read('status.json'); done = s.get('completed', 0); total = s.get('total', 128)
    result = read('results.json') or read('partial.json')
    print('\033[2J\033[H', end='')
    print('RoboTwin round7 | GPU0 | post-pilot OFFLINE confirmation (no optimizer updates)')
    print(f"[{('#'*int(40*done/max(1,total))).ljust(40,'.')}] {done}/{total} paired noise samples | {s.get('phase','queued')}")
    print('Cohort:', s.get('cohort', '-'), '| anchor:', s.get('anchor', '-'), '/32')
    if s.get('updated_at'): print('Heartbeat:', round(time.time()-s['updated_at'], 1), 's ago')
    if s.get('error'): print('ERROR:', s['error'])
    print('\nPrimary: step200; intermediate checkpoints exploratory. Lower errors are better.')
    print('Fresh: 32 new episodes x 2 noises. Tail: 8 new episodes x 4 remaining lengths x 2 noises.')
    for cohort in ['fresh', 'tail']:
        rows = [r for r in result.get('rows', []) if r['cohort'] == cohort]
        if not rows: continue
        episodes = sorted({r['episode'] for r in rows})
        print(f'\n{cohort}: {len(rows)} pairs, {len(episodes)} episodes (partial until completed)')
        print('       checkpoint  position_cm  rotation_deg  gripper_MAE')
        for variant in rows[0]['metrics']:
            metrics = []
            for key in ['position_cm', 'rotation_deg', 'gripper_mae']:
                means = [sum(r['metrics'][variant][key] for r in rows if r['episode'] == e)/sum(r['episode'] == e for r in rows) for e in episodes]
                metrics.append(sum(means)/len(means))
            print(f'{variant:>17} {metrics[0]:12.6f} {metrics[1]:13.6f} {metrics[2]:12.7f}')
    if result.get('completed'):
        print('\nStep200 fresh screen:', result['primary_fresh_passed'], '| tail subgroup screens:', result['primary_tail_passed'])
    print('\nOffline errors are NOT closed-loop task success rates. Old Stage II/online rollout remain paused.')
    print('Ctrl-b d detaches. Other progress windows: Ctrl-b n / p.', flush=True)
    time.sleep(5)
