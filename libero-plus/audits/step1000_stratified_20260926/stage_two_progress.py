"""Read-only tmux progress for student training and the fixed 28-cell eval queue."""
import argparse
import json
import os
from pathlib import Path
import time
from student_eval_common import TRAIN, ROOT, SUITES, CATEGORIES, GPUS

_EPISODE_CACHE={}

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def metrics():
    out=[]
    try:
        for line in (TRAIN/'metrics.jsonl').read_text().splitlines():
            try:out.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    return out

def bar(n,total,width=40):
    count=int(width*n/max(total,1));return '['+'#'*count+'-'*(width-count)+']'

def spark(values,width=65):
    if not values:return '--'
    values=values[-1000:]
    bins=[values[int(i*len(values)/min(width,len(values))):int((i+1)*len(values)/min(width,len(values)))] for i in range(min(width,len(values)))]
    means=[sum(x)/len(x) for x in bins];lo,hi=min(means),max(means)
    return ''.join('▁▂▃▄▅▆▇█'[min(7,int(7*(x-lo)/max(hi-lo,1e-12)))] for x in means)

def training():
    s=read(TRAIN/'status.json',{});data=metrics();rows=[r for r in data if r.get('split')=='train'];val=[r for r in data if r.get('split')=='validation']
    n=s.get('step',0);total=s.get('total_steps',10000)
    gpu_map=','.join(str(read(TRAIN/f'status_rank{rank}.json',{}).get('physical_gpu','?')) for rank in range(4))
    lines=[f'LIBERO-Plus Stage II | fixed teacher 5000 | GPU {gpu_map} | effective batch 32',
        time.strftime('%Y-%m-%d %H:%M:%S')+' | refresh 3s',f'{bar(n,total)} {n}/{total} ({100*n/total:.2f}%) | {s.get("phase")} micro {s.get("micro",0)}/8',
        'Loss                 Latest       Mean(last 20)  Trend (high -> low)']
    if rows:
        for k in ('flow','local_distill','goal_distill','loss'):
            series=[r[k] for r in rows];lines.append(f'{k:18} {series[-1]:.6f}     {sum(series[-20:])/len(series[-20:]):.6f}   {spark(series,45)}')
        speed=sum(r['update_seconds'] for r in rows[-20:])/len(rows[-20:]);hours=(total-n)*speed/3600
        lines.append(f'Last20 speed {speed:.1f}s/update | estimated remaining {hours:.1f} h (variable) | LR {rows[-1]["lr"]:.3g}')
    if val:
        r=val[-1];lines.append(f'Offline validation @{r["step"]}: action {r["flow"]:.5f}, local {r["local_distill"]:.5f}, goal {r["goal_distill"]:.5f}')
    lines+=['','Rank  GPU     PID       Alive   Step/micro   Phase                     Status age']
    for rank in range(4):
        r=read(TRAIN/f'status_rank{rank}.json',{});pid=r.get('pid',0);alive=Path(f'/proc/{pid}').exists()
        lines.append(f'{rank:4} {r.get("physical_gpu","?"):>4} {pid:9} {str(alive):>7} {r.get("step",0):5}/{r.get("micro",0)}       {r.get("phase","--"):25} {time.time()-r.get("updated_at",time.time()):.0f}s')
        if r.get('error'):lines.append('ERROR: '+r['error'])
    checkpoints=sorted(int(p.parent.name.split('_')[1]) for p in TRAIN.glob('step_*/checkpoint_complete.json') if p.parent.name!='step_000000')
    lines+=['','Saved student weights (all retained): '+(', '.join(map(str,checkpoints)) or 'waiting for step 500'),
        'Save / online evaluation: every 500 steps; full LIBERO-Plus 10030 tasks x 1 episode.',
        'Teacher frozen, current-only student; action + local/goal distillation losses.',
        'Sync audit: '+str(read(TRAIN/'sync_audit.json',{})),
        f'Weights: {TRAIN}', 'Full optimizer resume: /tmp/lawam_plus_stageII_optimizer_20260925/latest.json']
    q=read(ROOT/'status.json',{});lines.append(f'Evaluation queue: {q.get("phase","starting")} | step {q.get("step","--")}')
    return '\n'.join(lines)

def evaluation():
    q=read(ROOT/'status.json',{});lines=['LIBERO-Plus student | FULL 10030 tasks x 1 episode | GPU 6,6,7,7',
        time.strftime('%Y-%m-%d %H:%M:%S')+' | refresh 3s',f'Queue: {q.get("phase","starting")} | checkpoint {q.get("step","--")}']
    if q.get('error'):lines.append('ERROR: '+q['error'])
    folder=Path(q['folder']) if q.get('folder') else None
    if folder is None:
        completed=sorted(ROOT.glob('step_*/results.json'))
        if completed:folder=completed[-1].parent
    r=read(folder/'results.json',{}) if folder else {}
    if r:
        rate=f'{100*r["success_rate"]:.2f}%' if r.get('success_rate') is not None else '--'
        lines.append(f'{bar(r["episodes"],r["expected"])} {r["episodes"]}/{r["expected"]} | success {r["successes"]}/{r["episodes"]} = {rate}')
        if r['smoke']:lines.append('SMOKE ONLY: initialized student, 4 tasks; not a trained-checkpoint accuracy result.')
        lines+=['Cells: completed/expected | success rate among completed episodes',f'{"Suite":9}'+''.join(f'{c:^16}' for c in CATEGORIES),'-'*121]
        for s in SUITES:
            line=f'{s.replace("libero_", ""):9}'
            for c in CATEGORIES:
                cell=r['cells'][s][c];rate=f'{100*cell["success_rate"]:.1f}%' if cell['episodes'] else '--'
                line+=f'{str(cell["episodes"])+"/"+str(cell["expected"])+" "+rate:^16}'
            lines.append(line)
        lines+=['','Worker / GPU | phase | successes/completed (worker totals)']
        episode_paths=list((folder/'episodes').glob('*.json'))
        for p in episode_paths:
            if str(p) not in _EPISODE_CACHE:_EPISODE_CACHE[str(p)]=read(p,{})
        episodes=[_EPISODE_CACHE[str(p)] for p in episode_paths]
        for i,gpu in enumerate(GPUS):
            s=read(folder/f'worker{i}.json',{});owned=[x for x in episodes if x.get('worker')==i];ok=sum(bool(x['success']) for x in owned)
            task=s.get('task',{});suffix=f'{task.get("suite","")} {task.get("category","")} task {task.get("task_id","")}' if task else ''
            lines.append(f'{i} / GPU{gpu}: {s.get("phase","loading"):12} {ok}/{len(owned)}  {suffix}')
    lines+=['','Completed checkpoint evaluations:']
    for p in sorted(ROOT.glob('step_*/results.json')):
        x=read(p,{})
        if x.get('complete'):lines.append(f'  step {x["step"]:5}: {x["successes"]}/{x["episodes"]} = {100*x["success_rate"]:.2f}%')
    saved=[int(p.parent.name.split('_')[1]) for p in TRAIN.glob('step_*/checkpoint_complete.json') if p.parent.name!='step_000000']
    done={x['step'] for p in ROOT.glob('step_*/results.json') if (x:=read(p,{})).get('complete')}
    lines+=['Pending saved checkpoints: '+(', '.join(map(str,sorted(set(saved)-done))) or 'none; waiting for training'),
        'Baseline evaluation has separate processes/ports and continues independently.',f'Results: {ROOT}']
    return '\n'.join(lines)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','eval'],required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        text=training() if args.mode=='train' else evaluation()
        print(('' if args.once else '\033[H\033[2J')+text,flush=True)
        if args.once:break
        time.sleep(3)
