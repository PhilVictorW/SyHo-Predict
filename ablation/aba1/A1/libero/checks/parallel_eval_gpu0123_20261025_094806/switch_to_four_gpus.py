from pathlib import Path
import json,os,signal,time,subprocess,sys,fcntl,shlex
W=Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1/A1/libero')
sys.path[:0]=[str(W),str(W.parents[3])]
import parallel_evaluate as pe
from ablation.aba1.libero_common.config import atomic_json
record=pe.watch.read(W/'nohup/current.json');run=Path(record['run']);watchdir=W/'auto_eval'/run.name
state=pe.watch.read(watchdir/'status.json');out=Path(state['output'])
backup=Path(Path('/tmp/a1_parallel_eval_backup').read_text())
pid=record['processes']['auto_eval']['pid']
def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except OSError:return False
def identity(pid,token):
    p=Path('/proc')/str(pid)
    if not alive(pid):return
    if token not in p.joinpath('cmdline').read_bytes() or p.joinpath('cwd').resolve()!=W:
        raise RuntimeError('Process identity mismatch: '+str(pid))
identity(pid,b'watch_full_eval.py')
assert state['watcher_pid']==pid
old_eval=state['evaluation_pid'];old_server=pe.watch.read(out/'server_ready.json')['pid']
atomic_json(backup/'pre_switch_status.json',state)
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+60
while any(alive(p) for p in [pid,old_eval,old_server]):
    if time.monotonic()>deadline:raise RuntimeError('Old evaluator still alive; not starting conflicting workers')
    time.sleep(.5)
# Prove the original writer released both ownership locks.
with (watchdir/'watch.lock').open('a') as a,(out/'.evaluation.lock').open('a') as b:
    fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=pe.pending_tasks(out)
    preserved={f'{suite}__task{task:02d}':pe.episode_rows(pe.task_path(out,suite,task),task,50)
               for suite in pe.SUITES for task in range(10)}
    atomic_json(backup/'preserved_episodes.json',preserved)
    atomic_json(backup/'validation.json',dict(tests_passed=8,old_processes_exited=[pid,old_eval,old_server],
        preserved_episodes=sum(len(v) for v in preserved.values()),remaining_tasks=len(pending),
        protocol_sha256=pe.file_sha256(out/'protocol.json'),time=time.time()))
control=W/'nohup'/run.name;python='/home/wangchaoyang/envs/LaWAM/bin/python'
cmd=['nohup',python,'-B','-u',str(W/'parallel_evaluate.py'),'--run',str(run),'--gpus','0','1','2','3','--load-gap','10']
log=control/'parallel_eval.log'
with log.open('ab',buffering=0) as stream:
    child=subprocess.Popen(cmd,cwd=W,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record['processes']['auto_eval']=dict(pid=child.pid,command=cmd,log=str(log))
old_progress=record['processes']['progress']['pid'];identity(old_progress,b'monitor.py')
if alive(old_progress):os.kill(old_progress,signal.SIGTERM)
progress=['nohup',python,'-B','-u',str(W/'eval_progress.py'),'--run',str(run),'--interval','30']
progress_log=control/'eval_progress.log'
with progress_log.open('ab',buffering=0) as stream:
    process=subprocess.Popen(progress,cwd=W,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record['processes']['progress']=dict(pid=process.pid,command=progress,log=str(progress_log))
record.update(evaluation_physical_gpus=[0,1,2,3],evaluation_parallel_started_at=time.time(),
    evaluation_output=str(out),evaluation_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
atomic_json(W/'nohup/current.json',record);atomic_json(control/'parallel_launch_record.json',record)
atomic_json(backup/'launch_record.json',record)
session='a1_libero_gpu012'
for index,command in [(0,[python,'-B','-u',str(W/'eval_progress.py'),'--run',str(run),'--interval','5']),
                       (2,['tail','-n','40','-F',str(log)])]:
    current=subprocess.check_output(['tmux','display-message','-p','-t',f'{session}:{index}','#{pane_start_command}'],text=True)
    if str(W) not in current:raise RuntimeError('Foreign tmux window')
    subprocess.run(['tmux','respawn-window','-k','-t',f'{session}:{index}','-c',str(W),shlex.join(command)],check=True)
alias='a1_libero_eval_gpu0123'
if subprocess.run(['tmux','has-session','-t',alias],capture_output=True).returncode:
    subprocess.run(['tmux','new-session','-d','-s',alias,'-t',session],check=True)
subprocess.run(['tmux','select-window','-t',f'{alias}:0'],check=True)
print(json.dumps(dict(manager_pid=child.pid,progress_pid=process.pid,gpus=[0,1,2,3],
    preserved_episodes=sum(len(v) for v in preserved.values()),remaining_tasks=len(pending),
    log=str(log),tmux=alias),indent=2))
