"""Terminal-only Stage I progress and action flow loss trends; never controls training."""
import argparse,json,math,os,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'runs/stage_I_20260923'
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {} if default is None else default

def metrics():
    rows=[]
    try:
        for line in (ROOT/'metrics.jsonl').read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if math.isfinite(float(r.get('flow',float('nan')))):rows.append(r)
    except OSError:pass
    return rows

def mean(a):return sum(a)/len(a) if a else None

def fmt(x):return '--' if x is None else f'{x:.5f}'

def duration(s):
    s=max(0,int(s));return f'{s//86400}d {s//3600%24:02d}h {s//60%60:02d}m'

def proc(pid):
    try:
        s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
        return 'EXITED' if s=='Z' else f'alive({s})'
    except OSError:return 'EXITED'

def spark(values,width=65):
    if not values:return '--'
    chars='▁▂▃▄▅▆▇█';width=min(width,len(values));points=[]
    for i in range(width):
        group=values[i*len(values)//width:(i+1)*len(values)//width];points.append(math.log10(max(mean(group),1e-10)))
    lo,hi=min(points),max(points)
    return ''.join(chars[round(7*(v-lo)/(hi-lo)) if hi>lo else 3] for v in points)

def graph(train,width=66,height=6):
    smoothed=[];window=[]
    for r in train:
        window.append(float(r['flow']))
        if len(window)>50:window.pop(0)
        smoothed.append((int(r['step']),mean(window)))
    data=smoothed[-1000:]
    if not data:return ['Waiting for completed optimizer steps...']
    width=min(width,len(data));ys=[mean([v for _,v in data[i*len(data)//width:(i+1)*len(data)//width]]) for i in range(width)]
    lo,hi=min(ys),max(ys);span=max(hi-lo,1e-6);grid=[[' ']*width for _ in range(height)]
    previous=None
    for x,y in enumerate(ys):
        row=height-1-round((height-1)*(y-lo)/span)
        if previous is not None:
            for yy in range(min(previous,row),max(previous,row)+1):grid[yy][x]='.'
        grid[row][x]='*';previous=row
    out=[f'Train action loss: MA50, latest {len(data)} updates (linear y-axis)']
    for j,line in enumerate(grid):out.append(f'{hi-j*span/(height-1):.5f} |'+''.join(line))
    out.extend(['        +'+('-'*width),f"         step {data[0][0]}"+(' '*max(1,width-len(str(data[0][0]))-len(str(data[-1][0]))-11))+f"step {data[-1][0]}"])
    return out

def screen():
    now=time.time();s=read(ROOT/'status.json');rows=metrics();train=[r for r in rows if r.get('split')=='train'];val=[r for r in rows if r.get('split')=='validation']
    done=int(train[-1]['step']) if train else 0;total=int(s.get('total_steps',16000));loss=[float(r['flow']) for r in train]
    recent=mean(loss[-100:]);previous=mean(loss[-200:-100]);first=mean(loss[:100]);best=min(val,key=lambda r:r['flow']) if val else None
    delta=100*(recent/previous-1) if recent is not None and previous else None
    speed=mean([r['update_seconds'] for r in train[-20:] if 'update_seconds' in r]);width=max(30,min(76,shutil.get_terminal_size((100,30)).columns-12))
    out=['LIBERO-Plus | Goal/Local Stage I teacher | GPU 1,3 DDP',time.strftime('%Y-%m-%d %H:%M:%S')+' | refresh 2s | action loss = flow loss',f"[{'#'*int(40*done/max(total,1)):-<40}] {done}/{total} ({100*done/max(total,1):.2f}%) | {s.get('phase','starting')} micro {s.get('micro',0)}/8",f'Action loss: latest {fmt(loss[-1] if loss else None)} | MA20 {fmt(mean(loss[-20:]))} | MA100 {fmt(recent)}',f'MA100 vs previous 100: {delta:+.2f}%' if delta is not None else 'MA100 comparison: waiting for 200 updates']
    if val:out.append(f"Validation: {val[-1]['flow']:.5f} @ {val[-1]['step']} | best {best['flow']:.5f} @ {best['step']}")
    if speed:out.append(f'Last20 speed: {speed:.1f}s/update | estimated remaining {duration(speed*(total-done))} (variable)')
    out.extend(graph(train,width))
    out.append('Full history (log loss; high -> low): '+spark(loss,max(20,width-30)))
    out.append('First100 mean '+fmt(first)+' -> latest100 mean '+fmt(recent))
    if val:out.append('Validation: '+' | '.join(f"{r['step']}:{r['flow']:.4f}" for r in val[-5:]))
    out.append('Rank/GPU   PID       Process     Phase             Step/micro    Heartbeat age')
    for rank,gpu in ((0,1),(1,3)):
        v=read(ROOT/f'status_rank{rank}.json');pid=v.get('pid');age=now-v.get('updated_at',now)
        out.append(f" {rank} / {gpu}   {str(pid):9s} {proc(pid):11s} {v.get('phase','--'):17s} {v.get('step',0):5d}/{v.get('micro',0)}        {age:.0f}s")
    out.append('Checkpoint: '+Path(s.get('checkpoint','--')).name+' | save 1000 / validation 250 updates')
    retention=read(ROOT/'retention_status.json');saved=retention.get('saved',[])
    out.append('Archived model weights: '+(', '.join(str(int(x.split('_')[1])) for x in saved) or '--')+f" | retention heartbeat {now-retention.get('updated',now):.0f}s")
    if retention.get('error'):out.append('RETENTION ERROR: '+retention['error'])
    if now-s.get('updated_at',now)>180:out.append('NOTICE: training status has not refreshed for >180s; check train window.')
    out.append('Validation is offline action loss, not online task success rate.')
    return '\n'.join(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        print(('' if a.once else '\033[2J\033[H')+screen(),flush=True)
        if a.once:break
        time.sleep(2)
