"""Single-checkpoint 4-suite x 7-category detached-tmux dashboard."""
import argparse
import json
from pathlib import Path
import time
from student_eval_common import WORK, SUITES, CATEGORIES

CONTROL=WORK/'evaluation/stage_II_steps2500_4000_parallel_20261008'
GPUS={2500:0,3000:2,3500:6,4000:7}
PORTS={2500:24930,3000:24931,3500:24932,4000:24933}

def read(p,d=None):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return d

def render(step):
    gpu=GPUS[step];port=PORTS[step];base=CONTROL/f'step_{step:06d}';folder=base/f'step_{step:06d}_step500_stratified_gripper_v1';r=read(folder/'results.json',{})
    n=r.get('episodes',0);N=r.get('expected',8380);ok=r.get('successes',0);rate=100*ok/n if n else None
    server=read(folder/'server0.json',{});worker=read(folder/'worker0.json',{});spid=server.get('pid');wpid=worker.get('pid')
    alive_s=bool(spid and Path(f'/proc/{spid}').exists());alive_w=bool(wpid and Path(f'/proc/{wpid}').exists())
    filled=int(40*n/max(N,1))
    lines=[f'LIBERO-Plus | STEP {step} | GPU{gpu} | port {port} | refresh 3s',time.strftime('%Y-%m-%d %H:%M:%S')+' | '+read(base/'status.json',{}).get('phase','starting'),f'[{"#"*filled}{"-"*(40-filled)}] {n}/{N} ({100*n/N:.2f}%) | overall {ok}/{n} = {f"{rate:.2f}%" if rate is not None else "--"}',f'Server PID {spid} alive={alive_s} | Worker PID {wpid} alive={alive_w}', 'Each cell: completed/quota | observed success rate',f'{"Suite":12}'+''.join(f'{c:^14}' for c in CATEGORIES),'-'*110]
    for suite in SUITES:
        counts=f'{suite.replace("libero_",""):12}';rates=' '*12
        for cat in CATEGORIES:
            cell=r.get('cells',{}).get(suite,{}).get(cat,{})
            counts+=f'{str(cell.get("episodes",0))+"/"+str(cell.get("expected","--")):^14}'
            v=cell.get('success_rate');rates+=f'{(f"{100*v:.1f}%" if v is not None else "--"):^14}'
        lines.extend([counts,rates])
    lines+=['','Results: '+str(folder),'Tmux session: lawam_plus_steps2500_4000_eval']
    return '\n'.join(lines)

def main():
    a=argparse.ArgumentParser();a.add_argument('--step',type=int,required=True);a.add_argument('--once',action='store_true');args=a.parse_args();assert args.step in GPUS
    while True:
        print(('' if args.once else '\033[H\033[2J')+render(args.step),flush=True)
        if args.once:break
        time.sleep(3)
if __name__=='__main__':main()
