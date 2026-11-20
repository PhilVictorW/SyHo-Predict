"""Read-only combined status for released four categories and local Environment."""
import sqlite3
from common import *
ORIGINAL_RUN=BASE/'runs/20261105_233840_fixed2'
LABELS={'object':'Object','swap':'Position','lan':'Semantic','task':'Task','env':'Environment (local)'}
def queue_summary(run,variant):
    path=run/variant/'queue.sqlite'
    if not path.exists(): return dict(total=0,completed=0,successes=0,counts={},gpus=[],categories=[])
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=30) as db:
        db.execute('BEGIN')
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone():
            return dict(total=0,completed=0,successes=0,counts={},gpus=[],categories=[])
        counts=dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        success=db.execute("SELECT COALESCE(SUM(success),0) FROM jobs WHERE state='done'").fetchone()[0]
        g=db.execute('SELECT gpu,state,COUNT(*),COALESCE(SUM(success),0) FROM jobs GROUP BY gpu,state').fetchall()
        cats=db.execute("SELECT json_extract(payload,'$.category'),COUNT(*),SUM(state='done'),COALESCE(SUM(CASE WHEN state='done' THEN success ELSE 0 END),0) FROM jobs GROUP BY json_extract(payload,'$.category')").fetchall()
    return dict(total=sum(counts.values()),completed=counts.get('done',0),successes=success,counts=counts,
                gpus=[dict(gpu=gpu,total=sum(r[2] for r in g if r[0]==gpu),
                  completed=sum(r[2] for r in g if r[0]==gpu and r[1]=='done'),
                  pending=sum(r[2] for r in g if r[0]==gpu and r[1]=='pending'),
                  running=sum(r[2] for r in g if r[0]==gpu and r[1]=='running')) for gpu in GPUS[variant]],
                categories=[dict(category=c,label=LABELS[c],total=t,completed=n,successes=k) for c,t,n,k in cats])
def get_status():
    out=dict(updated=time.time(),gpu_assignment=GPUS,inference_processes_per_gpu=2,
             environment_run=str(RUN),original_run=str(ORIGINAL_RUN),
             protocol='Official four released categories + locally generated Environment extension',variants={})
    for variant in GPUS:
        old=queue_summary(ORIGINAL_RUN,variant);env=queue_summary(RUN,variant)
        n=old['completed']+env['completed'];k=old['successes']+env['successes']
        counts={key:old['counts'].get(key,0)+env['counts'].get(key,0) for key in ('done','pending','running','failed')}
        out['variants'][variant]=dict(total=10000,prepared=old['total']+env['total'],completed=n,successes=k,
            success_rate=k/n if n else None,counts=counts,original=old,environment=env,
            categories=old['categories']+env['categories'])
    return out

def bar(done,total,width=24):
    filled=max(0,min(width,int(width*done/total))) if total else 0
    return '['+'#'*filled+'.'*(width-filled)+']'

def render():
    data=get_status();old=read(ORIGINAL_RUN/'controller.json',{});new=read(RUN/'controller.json',{})
    slots=old.get('slots',[])+new.get('slots',[])
    free=dict(old.get('gpu_free_mib',{}));free.update(new.get('gpu_free_mib',{}))
    heartbeat=lambda row: f'{max(0,time.time()-row["updated"]):.0f}s' if row.get('updated') else '--'
    lines=[f'LIBERO-PRO | five dimensions | {time.strftime("%Y-%m-%d %H:%M:%S")}',
        'LIBERO: GPU 0,1,2,3 | PLUS: GPU 5,6,7 | max 2 inference replicas/GPU | transfer to PLUS paused',
        f'Controllers: base PID {old.get("pid","--")} {old.get("phase","--")} ({heartbeat(old)}) | env PID {new.get("pid","--")} {new.get("phase","--")} ({heartbeat(new)})','']
    order=('object','swap','lan','task','env')
    for variant,d in data['variants'].items():
        n=d['completed'];k=d['successes'];sr=f'{k/n:.2%}' if n else '--'
        original=d['original'];env=d['environment']
        lines.append(f'{variant.upper():6s} {bar(n,10000,32)} {n:5d}/10000 {n/100:6.2f}% | success {k}/{n} = {sr}')
        en=env['completed']
        lines.append(f'  Base {original["completed"]:4d}/8000 | Environment {bar(en,2000,20)} {en:4d}/2000 {en/20:6.2f}% | infrastructure errors {d["counts"]["failed"]}')
        cats={c['category']:c for c in d['categories']}
        lines.append('  '+' | '.join(f'{LABELS[c]} {cats[c]["completed"]}/{cats[c]["total"]}' for c in order if c in cats))
        for gpu in GPUS[variant]:
            relevant=[s for s in slots if s['variant']==variant and s['gpu']==gpu and s['phase'] not in ('completed','stopped')]
            active=[s for s in relevant if s['phase']=='running']
            loading=[s for s in relevant if s['phase']=='loading']
            original_gpu=next((g for g in original['gpus'] if g['gpu']==gpu),{})
            env_gpu=next((g for g in env['gpus'] if g['gpu']==gpu),{})
            stage='base' if original_gpu.get('pending',0)+original_gpu.get('running',0) else 'env'
            g=original_gpu if stage=='base' else env_gpu
            completed=g.get('completed',0);total=g.get('total',0)
            details=' | '.join(f'w{s["rank"]} {s.get("progress",{}).get("category","?")} step {s.get("progress",{}).get("action_step",0)}' for s in active)
            if not details:
                if loading:details='loading checkpoint'
                elif total and completed==total:details='completed'
                else:details=next((s.get('reason') or s['phase'] for s in relevant if s['phase']=='waiting_memory'),'waiting')
            lines.append(f'  GPU{gpu} {stage:4s} {bar(completed,total,14)} {completed:4d}/{total:<4d} | run {len(active)}/2 load {len(loading)} | free {free.get(str(gpu),0)/1024:5.1f} GiB | {details}')
        lines.append('')
    lines.extend(['Environment (local): 40 tasks x 50 validated states; identical states and policy noise for both weights.',
        'Success rates use completed episodes. Each replica has a model server and a simulator helper.',
        'Refresh: 3 seconds | Detach: Ctrl-b then d | Detaching keeps evaluation running.',
        f'Results: {RUN}'])
    return '\n'.join(lines)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    if a.json: print(json.dumps(get_status(),indent=2))
    elif a.once: print(render(),flush=True)
    else:
        last_frame=None
        while True:
            try:
                frame=render()
                last_frame=frame
            except (sqlite3.Error,OSError,ValueError) as exc:
                warning=f'STATUS READ ERROR at {time.strftime("%H:%M:%S")}: {exc}. Retrying in 3s; displayed data may be stale.'
                frame=('\n'.join(last_frame.splitlines()[:-1])+'\n'+warning) if last_frame else warning
            print('\033[H\033[2J'+frame,flush=True)
            time.sleep(3)
