"""Starts and cleans up ONLY this diagnostic's children; live queues untouched."""
import os,sys,json,time,subprocess,socket,signal
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];WORK=ROOT/'libero-plus'
sys.path[:0]=[str(ROOT),str(WORK/'runtime')]
from student_eval_queue import environment
from student_eval_common import PY,SIMPY,atomic
from goal_local.eval_client import rpc
checkpoint=WORK/'runs/stage_II_20260925/step_000500'
manifest=json.loads((WORK/'evaluation/stage_II_20260925/full_manifest.json').read_text())['tasks']
selection=[('libero_spatial','Camera'),('libero_spatial','Robot'),('libero_object','Language'),('libero_object','Light'),
           ('libero_goal','Background'),('libero_goal','Noise'),('libero_10','Layout'),('libero_10','Camera')]
rng=np.random.default_rng(20260925);tasks=[]
for suite,cat in selection:
    options=[t for t in manifest if t['suite']==suite and t['category']==cat]
    tasks.append(options[int(rng.integers(len(options)))])
atomic(HERE/'paired_protocol.json',dict(checkpoint=str(checkpoint),tasks=tasks,seed=0,selection_seed=20260925,
    variants=['unchanged','flipped'],only_difference='raw_action.open_gripper = 1 - raw_action.open_gripper',
    physical_gpu=6,automatic_queue_restarted=False,training_modified=False))
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
env=environment(6);env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
sysroot=ROOT.parent/'LaWAM_without_subgoal/libero-plus/runtime/sysroot/usr'
env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MAGICK_HOME=str(sysroot),
    LD_LIBRARY_PATH=str(sysroot/'lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH',''),
    LIBERO_HOME='/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero-plus/inference/LIBERO-plus',
    LIBERO_CONFIG_PATH=str(ROOT.parent/'LaWAM_aba2/libero-plus/config/libero'))
children=[]
def start(cmd,log):
    with (HERE/log).open('a') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p);return p
def summarize():
    rows=[json.loads(p.read_text()) for p in (HERE/'paired_episodes').glob('*.json')]
    stats={}
    for v in ('unchanged','flipped'):
        r=[x for x in rows if x['variant']==v];stats[v]=dict(completed=len(r),successes=sum(bool(x['success']) for x in r),errors=sum(bool(x.get('error')) for x in r))
    atomic(HERE/'paired_summary.json',dict(stats=stats,expected_per_variant=8,results=rows,updated=time.time()))
    return stats
try:
    server=start([PY,str(WORK/'runtime/student_server.py'),'--port',str(port),'--ready',str(HERE/'paired_server.json')],'paired_server.log')
    url=f'http://127.0.0.1:{port}';deadline=time.monotonic()+600
    while True:
        if server.poll() is not None:raise RuntimeError('Diagnostic server exited')
        try:rpc(url,'/health',timeout=2);break
        except Exception:
            if time.monotonic()>deadline:raise TimeoutError('Diagnostic startup')
            time.sleep(2)
    info=rpc(url,'/load',dict(checkpoint=str(checkpoint)),timeout=1200);assert info['step']==500
    workers=[start([SIMPY,str(HERE/'paired_rollout.py'),'--worker',str(i),'--url',url],f'paired_worker{i}.log') for i in range(2)]
    atomic(HERE/'paired_processes.json',dict(controller=os.getpid(),server=server.pid,workers=[w.pid for w in workers],gpu=6,port=port))
    previous=None
    while any(p.poll() is None for p in workers):
        state=summarize()
        if state!=previous:print(json.dumps(state),flush=True);previous=state
        if server.poll() is not None:raise RuntimeError('Diagnostic server died')
        time.sleep(5)
    assert all(p.returncode==0 for p in workers),[p.returncode for p in workers]
    print('FINISHED',json.dumps(summarize()),flush=True)
finally:
    for p in children:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
