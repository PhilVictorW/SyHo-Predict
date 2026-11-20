"""Read-only aggregation and per-task watchdog for the final 100x50 evaluation."""
import json, os, time
from pathlib import Path

ROOT=Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change')
WORK=ROOT/'stage_III/robotwin'
HERE=Path(__file__).resolve().parent
STALL_SECONDS=2400

def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return {} if default is None else default

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def task_data(job, name):
    paths=list(Path(job).glob('*/*/tasks/*/'+name))
    return read(paths[0]) if paths else {}

def episodes(job):
    data=task_data(job,'summary.json') or task_data(job,'episodes.partial.json')
    rows=data.get('episodes',[])
    # The bridge persists outcomes immediately after each completed episode.
    return sorted({int(x['episode_id']):x for x in rows}.values(),key=lambda x:x['episode_id'])

def last_progress(job, started):
    event=read(Path(job)/'activity.json')
    summary=task_data(job,'progress.json')
    return max(started,float(event.get('last_progress_at',started)),float(summary.get('last_episode_at') or started))

def timed_out(job, started, now=None):
    return (time.time() if now is None else now)-last_progress(job,started)>=STALL_SECONDS

def aggregate(run, jobs):
    modes={m:dict(successes=0,episodes=0,episode_errors=0,target=2500,completed_cells=0,skipped_cells=0)
           for m in ('demo_clean','demo_randomized')}
    results=[]
    for job in jobs:
        folder=Path(run)/'jobs'/job['key'];eps=episodes(folder)
        end=read(folder/'result.json');row=modes[job['mode']]
        row['episodes']+=len(eps);row['successes']+=sum(bool(x['success']) for x in eps)
        row['episode_errors']+=sum(bool(x.get('error')) for x in eps)
        row['completed_cells']+=int(end.get('status')=='completed')
        row['skipped_cells']+=int(end.get('status','').startswith('skipped'))
        if end: results.append(end)
    for row in modes.values():
        row['success_rate']=row['successes']/row['episodes'] if row['episodes'] else None
    n=sum(v['episodes'] for v in modes.values());success=sum(v['successes'] for v in modes.values())
    return dict(modes=modes,episodes=n,target=5000,successes=success,
                success_rate=success/n if n else None,
                completed_cells=sum(v['completed_cells'] for v in modes.values()),
                skipped_cells=sum(v['skipped_cells'] for v in modes.values()),
                finished_cells=len(results),results=results)
