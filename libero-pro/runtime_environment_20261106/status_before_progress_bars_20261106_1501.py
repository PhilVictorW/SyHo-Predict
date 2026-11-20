"""Read-only combined status for released four categories and local Environment."""
import sqlite3
from common import *
ORIGINAL_RUN=BASE/'runs/20261105_233840_fixed2'
LABELS={'object':'Object','swap':'Position','lan':'Semantic','task':'Task','env':'Environment (local)'}
def queue_summary(run,variant):
    path=run/variant/'queue.sqlite'
    if not path.exists(): return dict(total=0,completed=0,successes=0,counts={},gpus=[],categories=[])
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=30) as db:
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

def render():
    data=get_status();old=read(ORIGINAL_RUN/'controller.json',{});new=read(RUN/'controller.json',{})
    slots=old.get('slots',[])+new.get('slots',[])
    free=dict(old.get('gpu_free_mib',{}));free.update(new.get('gpu_free_mib',{}))
    lines=[f'LIBERO-PRO five dimensions | {time.strftime("%Y-%m-%d %H:%M:%S")}',
        'Each weight: released 8000 episodes + locally generated Environment 2000 = 10000.',
        'LIBERO: GPU 0,1,2,3 | PLUS: GPU 5,6,7 | maximum 2 inference replicas/GPU | GPU4 excluded.',
        'Automatic transfer of GPUs 0-3 to PLUS is disabled.',
        f'Environment controller PID {new.get("pid","not launched")} | {new.get("phase","preparing")} | heartbeat {time.time()-new.get("updated",time.time()):.0f}s','']
    for variant,d in data['variants'].items():
        n=d['completed'];k=d['successes'];sr=f'{k/n:.2%}' if n else '--'
        lines.append(f'{variant.upper()}: {n}/10000 ({n/100:.2f}%) | success {k}/{n} = {sr}')
        lines.append(f'  Original four categories: {d["original"]["completed"]}/8000 | Environment: {d["environment"]["completed"]}/2000 | errors {d["counts"]["failed"]}')
        for gpu in GPUS[variant]:
            relevant=[s for s in slots if s['variant']==variant and s['gpu']==gpu and s['phase'] not in ('completed','stopped')]
            active=[s for s in relevant if s['phase']=='running']
            loading=[s for s in relevant if s['phase']=='loading']
            progress=' | '.join(f'w{s["rank"]} {s["progress"].get("category","?")} job {s["progress"].get("job_id","?")} step {s["progress"].get("action_step",0)}' for s in active if 'progress' in s)
            reason=next((s.get('reason',s['phase']) for s in relevant if s['phase']=='waiting_memory'),'')
            lines.append(f'  GPU{gpu}: running {len(active)}/2 loading {len(loading)} free {free.get(str(gpu),0)/1024:.1f} GiB | {progress or reason or "completed"}')
        lines.append('  '+' | '.join(f'{c["label"]} {c["completed"]}/{c["total"]}' for c in d['categories']))
        lines.append('')
    lines.extend(['Environment uses identical validated states and policy noise seeds for both weights.',
        'Each inference replica consists of a model server and a simulator helper.',
        f'Environment results: {RUN}','Detach: Ctrl-b then d. Closing tmux does not stop evaluation.'])
    return '\n'.join(lines)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    if a.json: print(json.dumps(get_status(),indent=2))
    else:
        while True:
            if not a.once:print('\033[H\033[2J',end='')
            print(render(),flush=True)
            if a.once:break
            time.sleep(3)
