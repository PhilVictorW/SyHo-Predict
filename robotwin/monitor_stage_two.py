"""Live Stage II loss dashboard plus complete loss CSV/PNG/SVG history."""
import argparse,csv,json,os,shutil,time
from pathlib import Path

def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def metrics(path):
    rows=[]
    try:
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    return rows

def spark(values,width):
    if not values:return ''
    import math
    ys=[math.log10(max(x,1e-10)) for x in values]
    if len(ys)>width:ys=[sum(ys[i*len(ys)//width:(i+1)*len(ys)//width])/len(ys[i*len(ys)//width:(i+1)*len(ys)//width]) for i in range(width)]
    lo,hi=min(ys),max(ys);chars=' .:-=+*#%@'
    return ''.join(chars[min(9,int(9*(v-lo)/max(hi-lo,1e-12)))] for v in ys)

def plot(run,rows):
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    keys=['flow','local_distill','goal_distill','loss']
    with (run/'stage_II_loss_history.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['step','split',*keys,'grad_norm','lr','time'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for key,ax in zip(keys,axes.flat):
        for split,color in [('train','steelblue'),('validation','darkorange')]:
            part=[r for r in rows if r['split']==split]
            if part:ax.plot([r['step'] for r in part],[max(r[key],1e-10) for r in part],label=split,color=color,linewidth=.7 if split=='train' else 1.8,alpha=.7)
        ax.set(title=key,xlabel='Student optimizer step',ylabel='loss');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(run/'stage_II_loss_curve.png',dpi=140);fig.savefig(run/'stage_II_loss_curve.svg');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();last=0
    while True:
        status=read(a.run/'status.json',{});rows=metrics(a.run/'metrics.jsonl');train=[x for x in rows if x['split']=='train'];val=[x for x in rows if x['split']=='validation']
        e=read(a.run/'quick_eval/status.json',{});width=max(25,min(90,shutil.get_terminal_size((110,35)).columns-4))
        step=status.get('step',0);total=status.get('total_steps',10000)
        lines=['RoboTwin Stage II | train GPU 2,5 | frozen teacher Stage I / 1000',
            f"[{int(width*step/total)*'=':<{width}}] {step}/{total} ({100*step/total:.2f}%)",
            f"Phase: {status.get('phase','starting')} | micro {status.get('micro',0)}/16 | skipped updates {status.get('skipped_updates',0)}",
            f"ETA: {status.get('eta_seconds',0)/3600:.2f} h | {time.strftime('%F %T')}",
            'Loss: latest train / mean of last 50 / latest fixed validation']
        for key,label in [('flow','Action'),('local_distill','Local distill'),('goal_distill','Goal distill'),('loss','Total')]:
            latest=train[-1][key] if train else float('nan');avg=sum(x[key] for x in train[-50:])/len(train[-50:]) if train else float('nan');v=val[-1][key] if val else float('nan')
            lines.extend([f'{label:15s} {latest:.7f} / {avg:.7f} / {v:.7f}',spark([x[key] for x in train],width)])
        progress=e.get('progress',{})
        lines.extend(['',f"Evaluation GPU 0,5: {e.get('phase','waiting')} | step {e.get('step','-')} | {e.get('mode','-')}",
            f"Episodes {progress.get('episodes',0)} | successes {progress.get('successes',0)} | skipped tasks {progress.get('skipped_tasks',0)}",
            'Each checkpoint: 50 tasks x 2 modes x 10 episodes; skip after 40 min without episode progress.',
            'Complete four-loss curves: stage_II_loss_curve.png/.svg; data: stage_II_loss_history.csv'])
        if status.get('error'):lines.append('ERROR: '+status['error'])
        print('\033[2J\033[H'+'\n'.join(lines),flush=True)
        if time.time()-last>60:
            try:plot(a.run,rows)
            except Exception as exc:print('Plot error:',exc,flush=True)
            last=time.time()
        time.sleep(5)

if __name__=='__main__':main()
