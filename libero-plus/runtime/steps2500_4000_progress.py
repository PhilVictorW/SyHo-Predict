"""Combined detached-tmux dashboard for four single-GPU stratified evaluations."""
import argparse
import json
from pathlib import Path
import time
from student_eval_common import WORK

CONTROL=WORK/'evaluation/stage_II_steps2500_4000_parallel_20261008'
JOBS=[(2500,0,24930),(3000,2,24931),(3500,6,24932),(4000,7,24933)]

def read(p,d=None):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return d

def render():
    lines=['LIBERO-Plus | STRATIFIED SUBSET | 2500/3000/3500/4000',time.strftime('%Y-%m-%d %H:%M:%S')+' | refresh 3s','Each row: one checkpoint, one evaluator process, one physical GPU; 8380 tasks (23 full cells + 5 reduced cells)','']
    for step,gpu,port in JOBS:
        c=CONTROL/f'step_{step:06d}';s=read(c/'status.json',{});f=c/f'step_{step:06d}_step500_stratified_gripper_v1';r=read(f/'results.json',{})
        n=r.get('episodes',0);N=r.get('expected',8380);ok=r.get('successes',0);pct=(100*n/N if N else 0);rate=(100*ok/n if n else None);rate_text=(f'{rate:.2f}%' if rate is not None else '--');width=30;filled=int(width*n/max(N,1))
        pid=read(f/'server0.json',{}).get('pid','--');alive=Path('/proc/'+str(pid)).exists() if pid!='--' else False
        lines.append(f'Step {step:4} | GPU{gpu} | port {port} | {s.get("phase","starting")} | server {pid} alive={alive}')
        lines.append('['+'#'*filled+'-'*(width-filled)+f'] {n}/{N} ({pct:.2f}%) | success {ok}/{n} = {rate_text}'.format(n=n,N=N,pct=pct,ok=ok,rate_text=rate_text))
        if r.get('complete'):lines.append(f'FINAL: {ok}/{n} = {rate:.4f}%')
        lines.append('')
    lines+=['All four evaluations are detached under tmux: lawam_plus_steps2500_4000_eval',f'Results: {CONTROL}']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        print(('' if a.once else '\033[H\033[2J')+render(),flush=True)
        if a.once:break
        time.sleep(3)
if __name__=='__main__':main()
