"""Read-only live progress for both models and every requested physical GPU."""
import argparse
import sys
import time
from common import *


def bar(done,total,width=24):
    filled=int(width*done/total) if total else 0
    return '['+'#'*filled+'-'*(width-filled)+']'


def render(variant=None):
    controller=read(RUN/'controller.json',{})
    launched=read(RUN/'launch.json',{})
    started=launched.get('started',time.time())
    elapsed=max(1,time.time()-started)
    lines=[f'LIBERO-PRO | plus vs libero | {time.strftime("%Y-%m-%d %H:%M:%S")}',
           'Official released Object / Position / Semantic / Task: 160 tasks x 50 initial states = 8000 episodes/model',
           f'Elapsed {elapsed/3600:.2f} h | controller PID {controller.get("pid","starting")} | heartbeat age {time.time()-controller.get("updated",time.time()):.0f} s',
           'GPU 4 excluded. Running success rate is provisional; infrastructure errors are displayed separately.','']
    for tag in [variant] if variant else ['plus','libero']:
        data=read(RUN/tag/'summary.json',{})
        n=data.get('completed',0);k=data.get('successes',0);total=data.get('total',8000)
        rate=f'{100*k/n:.2f}%' if n else '--'
        lines.extend([f'{tag.upper():6s} {bar(n,total,32)} {n:5d}/{total} ({100*n/total:6.2f}%)  Success {k}/{n} = {rate}  {data.get("phase","starting")}',
                      ' GPU   parallel  loading  progress                     completed/assigned   successes   success rate  errors  free VRAM'])
        by_gpu={x['gpu']:x for x in data.get('gpus',[])}
        for gpu in GPUS[tag]:
            row=by_gpu.get(gpu,{})
            count=row.get('completed',0);passed=row.get('successes',0);assigned=row.get('total',0)
            sr=f'{100*passed/count:6.2f}%' if count else '     --'
            free=controller.get('gpu_free_mib',{}).get(str(gpu))
            free_text=f'{free/1024:.1f} GiB' if free is not None else '--'
            lines.append(f' {gpu:3d}   {row.get("workers",0):8d} {row.get("loading",0):8d}  {bar(count,assigned)} {count:6d}/{assigned:<6d}   {passed:6d}    {sr:>9s}  {row.get("errors",0):6d}  {free_text}')
        cats=[]
        for row in data.get('categories',[]):
            count=row['completed'];passed=row['successes'];sr=f'{100*passed/count:.2f}%' if count else '--'
            cats.append(f'{row["category"]}: {count}/{row["total"]}, SR {sr}')
        lines.extend(['  '+' | '.join(cats),''])
    lines.extend(['Each parallel slot = 1 student inference process + 1 simulator process.',
                  'GPU assignments may rebalance as queues drain. Completed episodes are never rerun by scheduling.',
                  f'Results: {RUN}', 'Detach tmux: Ctrl-b, then d. Detaching does not stop nohup evaluation.'])
    return '\n'.join(lines)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['plus','libero']);parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    while True:
        if not args.once:sys.stdout.write('\033[H\033[2J')
        print(render(args.variant),flush=True)
        if args.once:break
        time.sleep(3)
