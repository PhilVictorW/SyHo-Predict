import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'crossover_audit'


def read(name):
    try:
        return json.loads((ROOT/name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


while True:
    status = read('status.json')
    done, total = status.get('completed', 0), status.get('total', 128)
    width = min(40, int(40 * done / max(total, 1)))
    print('\033[2J\033[H', end='')
    print('RoboTwin future representation / action injection crossover | GPU0 | read-only')
    print('Old Stage II remains stopped. Online evaluations remain paused.')
    print(f"\n{status.get('phase', 'waiting')}  [{'#'*width}{'.'*(40-width)}] {done}/{total}")
    print(f"Status age: {time.time()-status.get('updated_at', time.time()):.0f}s")
    if status.get('error'):
        print('ERROR:', status['error'])
    report = read('results.json') or read('partial.json')
    print('Completed episode/seed pairs:', len(report.get('rows', [])))
    if report.get('mean'):
        print('\nMode                   Position cm    Rotation deg    Gripper MAE')
        for mode, m in report['mean'].items():
            print(f"{mode:23s} {m['position_cm']:11.6f} {m['rotation_deg']:15.6f} {m['gripper_mae']:14.7f}")
        print('\nCorrect-future benefit CI (exploratory):')
        for mode, summary in report.get('summary', {}).items():
            print(mode, summary.get('future_benefit', {}).get('bootstrap_95'))
    print('\nCtrl-b d: detach. Refresh: 5 seconds.', flush=True)
    time.sleep(5)
