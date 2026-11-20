"""Stop only this student run after step4000 model and full recovery commit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import zipfile

WORK=Path(__file__).resolve().parents[1]
RUN=WORK/'runs/stage_II_20260925'
OPT=Path('/tmp/lawam_plus_stageII_optimizer_20260925')
TARGET=4000

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def atomic(path,data):
    tmp=path.with_suffix('.stopwatch.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def ready(run=RUN,opt=OPT):
    cp=run/f'step_{TARGET:06d}';marker=read(cp/'checkpoint_complete.json')
    resume=read(cp/'resume.json');latest=read(opt/'latest.json');state=opt/f'trainer_{TARGET:06d}.pt'
    if marker.get('step')!=TARGET or marker.get('mode')!='student':return False
    if latest.get('step')!=TARGET or latest.get('checkpoint')!=str(cp):return False
    if latest.get('optimizer')!=str(state) or resume.get('optimizer')!=str(state):return False
    if resume.get('latest_manifest')!=str(opt/'latest.json'):return False
    return all(f.is_file() and zipfile.is_zipfile(f) for f in [cp/'model.pt',state])

def identity(pid):
    try:
        p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=pid,parent=int(fields[1]),group=int(fields[2]),started=fields[19],
                    argv=(p/'cmdline').read_bytes().decode().split('\0'),cwd=str((p/'cwd').resolve()))
    except (OSError,ValueError):return None

def training_processes():
    rows=[]
    for rank in range(4):
        status=read(RUN/f'status_rank{rank}.json');proc=identity(status.get('pid',0))
        if proc is None:continue
        if proc['cwd']!=str(WORK) or 'runtime/train_stage_two.py' not in proc['argv']:
            raise RuntimeError('Refuse to signal unexpected process')
        argv=proc['argv'];output=argv[argv.index('--output')+1]
        if (WORK/output).resolve()!=RUN:raise RuntimeError('Unexpected training output')
        rows.append(proc)
    parents={r['parent'] for r in rows}
    if len(parents)>1:raise RuntimeError('Ranks do not share a launcher')
    launcher=identity(next(iter(parents))) if parents else None
    if launcher and ('torch.distributed.run' not in launcher['argv'] or launcher['cwd']!=str(WORK)):
        raise RuntimeError('Unexpected launcher')
    return rows,launcher

def same(proc):
    now=identity(proc['pid']);return now is not None and now['started']==proc['started']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        ranks,launcher=training_processes();print(json.dumps(dict(target=TARGET,ready=ready(),ranks=[r['pid'] for r in ranks],launcher=launcher['pid'] if launcher else None)));return
    lock=(RUN/'stop_at_4000.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    heartbeat=0
    while not ready():
        if time.monotonic()-heartbeat>=10:
            s=read(RUN/'status.json');atomic(RUN/'stop_watcher.json',dict(pid=os.getpid(),target=TARGET,phase='waiting_for_complete_checkpoint_and_optimizer',training_step=s.get('step'),updated=time.time()));heartbeat=time.monotonic()
        time.sleep(.5)
    ranks,launcher=training_processes()
    if launcher:os.kill(launcher['pid'],signal.SIGTERM)
    deadline=time.monotonic()+30
    while any(same(r) for r in ranks) and time.monotonic()<deadline:time.sleep(.25)
    for r in ranks:
        if same(r):os.kill(r['pid'],signal.SIGTERM)
    deadline=time.monotonic()+10
    while any(same(r) for r in ranks) and time.monotonic()<deadline:time.sleep(.25)
    for r in ranks:
        if same(r):os.kill(r['pid'],signal.SIGKILL)
    deadline=time.monotonic()+10
    while any(same(r) for r in ranks) and time.monotonic()<deadline:time.sleep(.25)
    if any(same(r) for r in ranks):raise RuntimeError('Training processes did not exit')
    record=dict(target_step=TARGET,phase='stopped_after_complete_checkpoint',checkpoint=str(RUN/f'step_{TARGET:06d}'),optimizer=str(OPT/f'trainer_{TARGET:06d}.pt'),updated=time.time(),launcher=launcher['pid'] if launcher else None,ranks=[r['pid'] for r in ranks],evaluation_untouched=True)
    atomic(RUN/'stopped_at_004000.json',record);atomic(RUN/'stop_watcher.json',dict(record,pid=os.getpid()))
    for path in [RUN/'status.json',*RUN.glob('status_rank*.json')]:
        state=read(path);state.update(step=TARGET,micro=0,phase='stopped_at_requested_step',stop_at_step=TARGET,updated_at=time.time());state.pop('error',None);atomic(path,state)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
