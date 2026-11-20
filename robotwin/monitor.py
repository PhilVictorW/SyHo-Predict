"""Refresh training metrics, standalone loss chart and an attachable dashboard."""
import argparse,csv,json,os,time
from pathlib import Path

def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def snapshot(run,plot=False):
    rows=[]
    try:
        for line in (run/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    train=[r for r in rows if r.get('split')=='train']
    val=[r for r in rows if r.get('split')=='validation']
    s=read(run/'status.json',{})
    gpu_label=','.join(str(x) for x in s.get('gpu_ids',[5]))
    data=dict(status=s,train_updates=len(train),latest_train=train[-1] if train else None,
        first_validation=val[0] if val else None,latest_validation=val[-1] if val else None)
    if train:
        w=min(50,len(train));first=sum(r['flow'] for r in train[:w])/w;last=sum(r['flow'] for r in train[-w:])/w
        data.update(window=w,first_window_flow=first,last_window_flow=last,relative_change=(last-first)/first if first else None)
    temp=run/'loss_summary.tmp';temp.write_text(json.dumps(data,indent=2));temp.replace(run/'loss_summary.json')
    if plot and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        if train:
            x=[r['step'] for r in train];y=[r['flow'] for r in train]
            ax.plot(x,y,alpha=.3,lw=.6,label='Train flow loss')
            smooth=[sum(y[max(0,i-49):i+1])/len(y[max(0,i-49):i+1]) for i in range(len(y))]
            ax.plot(x,smooth,lw=1.6,label='Train rolling mean (up to 50 updates)')
        if val:ax.plot([r['step'] for r in val],[r['flow'] for r in val],'o-',label='Fixed validation flow loss')
        ax.set(xlabel='Optimizer step',ylabel='Masked action flow MSE',title='RoboTwin Goal/Local Stage I')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(run/'loss_curve.png',dpi=150);fig.savefig(run/'loss_curve.svg');plt.close(fig)
        fields=['step','split','flow','loss','grad_norm','lr','update_seconds','time']
        with (run/'loss_history.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        lines=['# RoboTwin Goal/Local Stage I','',f"Status: {s.get('phase','starting')}; step {s.get('step',0)}/4000.",'',
            f"GPUs {gpu_label}; effective batch {s.get('effective_batch',32)}.",
            'RoboTwin SFT initialization; 30 Hz, 8 actions, three cameras, 16 EEF dimensions.',
            'All 500-step model weights retained; latest two checkpoints retain optimizer states.','',
            'Loss is offline posterior action flow MSE, not online success rate.','',
            '| Step | Fixed validation flow loss |','|---:|---:|']
        lines += [f"| {r['step']} | {r['flow']:.6f} |" for r in val]
        if train:lines += ['',f"Latest training loss: {train[-1]['flow']:.6f}",f"Latest rolling mean ({data['window']} updates): {data['last_window_flow']:.6f}"]
        lines += ['','![Loss](loss_curve.png)','']
        (run/'REPORT.md').write_text('\n'.join(lines))
    return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--once',action='store_true');args=p.parse_args()
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True);last_plot=0
    while True:
        now=time.time();plot=args.once or now-last_plot>=60
        d=snapshot(run,plot);s=d['status']
        if plot:last_plot=now
        print('\033[2J\033[H',end='')
        print('ROBOTWIN STAGE I | GPU',','.join(str(x) for x in s.get('gpu_ids',[5])),'|',time.strftime('%F %T'))
        print('Run:',run.name,'| PID:',s.get('pid'))
        print('Phase:',s.get('phase','starting'),'| Step:',s.get('step',0),'/',s.get('total_steps',4000),'| Micro:',s.get('micro',0),'/',s.get('accumulation_per_gpu',32))
        for key in ('latest_train','first_validation','latest_validation'):
            row=d[key]
            if row:print(key, 'step',row['step'],'flow',f"{row['flow']:.6f}")
        if 'eta_seconds' in s:print('ETA hours:',round(s['eta_seconds']/3600,2))
        if s.get('error'):print('ERROR:',s['error'])
        print('Metrics:',run/'loss_history.csv');print('Plot:',run/'loss_curve.png',flush=True)
        if args.once:return
        time.sleep(5)

if __name__=='__main__':main()
