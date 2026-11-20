"""Read-only dedicated tmux dashboard for the parallel step2000 evaluation."""
import argparse
import json
from pathlib import Path
import time
from student_eval_common import WORK,SUITES,CATEGORIES,GPUS

CONTROL=WORK/'evaluation/stage_II_step2000_parallel_20260926'
FOLDER=CONTROL/'step_002000_step500_stratified_gripper_v1'
CACHE={}

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (ValueError,OSError):return default

def render():
    q=read(CONTROL/'status.json',{});r=read(FOLDER/'results.json',{})
    n=r.get('episodes',0);expected=8380;ok=r.get('successes',0)
    rate=f'{100*ok/n:.2f}%' if n else '--';filled=int(36*n/expected)
    lines=['LIBERO-Plus | STUDENT STEP 2000 | GPU 6,6,7,7 | refresh 3s',
           time.strftime('%Y-%m-%d %H:%M:%S')+' | '+q.get('phase','starting'),
           'STRATIFIED SUBSET: identical 8380 tasks to step1000; 23 full cells + 5 reduced cells',
           '['+'#'*filled+'-'*(36-filled)+f'] {n}/{expected} ({100*n/expected:.2f}%)',
           f'Observed success: {ok}/{n} = {rate}',
           'Cells: completed/quota, then observed success rate',
           f'{"Suite":9}'+''.join(f'{c:^12}' for c in CATEGORIES),'-'*93]
    for suite in SUITES:
        counts=f'{suite.replace("libero_", ""):9}';rates=' '*9
        for category in CATEGORIES:
            cell=r.get('cells',{}).get(suite,{}).get(category,{})
            amount=f"{cell.get('episodes',0)}/{cell.get('expected','--')}"
            value=cell.get('success_rate');pct=f'{100*value:.1f}%' if value is not None else '--'
            counts+=f'{amount:^12}';rates+=f'{pct:^12}'
        lines.extend([counts,rates])
    for path in (FOLDER/'episodes').glob('*.json'):
        if path.name not in CACHE:
            row=read(path)
            if row is not None:CACHE[path.name]=row
    lines+=['','Worker GPU / Port | alive | phase | successful / completed']
    for worker,gpu in enumerate(GPUS):
        status=read(FOLDER/f'worker{worker}.json',{});pid=status.get('pid',0)
        rows=[row for row in CACHE.values() if row.get('worker')==worker]
        success=sum(bool(row['success']) for row in rows)
        pct=f'{100*success/len(rows):.2f}%' if rows else '--'
        alive=pid>0 and Path(f'/proc/{pid}').exists()
        lines.append(f'{worker:>2} GPU{gpu} / {24900+worker} | {str(alive):5} | {status.get("phase","loading"):12} | {success}/{len(rows)} {pct}')
    if q.get('error'):lines.append('ERROR: '+q['error'])
    lines.extend(['','Tmux: lawam_plus_step2000_eval',f'Results: {FOLDER}'])
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        print(('' if args.once else '\033[H\033[2J')+render(),flush=True)
        if args.once:break
        time.sleep(3)

if __name__=='__main__':main()
