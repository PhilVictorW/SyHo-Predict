"""Durable, resumable evaluation queue. Never signals or modifies training."""
import argparse,csv,fcntl,hashlib,io,json,os,queue,signal,subprocess,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .eval_common import *
from .eval_client import rpc

def task_dir(step,suite,task):return OUTPUT/f'step_{step:06d}'/'tasks'/f'{suite}__task{task:02d}'
def episodes(path,task):
 result={}
 try:
  with (path/'episodes.checkpoint.jsonl').open() as f:
   for line in f:
    try:d=json.loads(line)
    except ValueError:continue # An unfinished final write is picked up on the next refresh.
    if int(d['task_id'])!=task or not 0<=int(d['episode_idx'])<10:raise ValueError('Unexpected task/episode in results')
    if d.get('error'):continue
    idx=int(d['episode_idx'])
    if idx in result and result[idx]!=d:raise ValueError('Conflicting duplicate episode')
    result[idx]=d
 except FileNotFoundError:pass
 return result

def aggregate(step):
 suites={};tasks=[]
 for suite in SUITES:
  n=k=0
  for task in range(10):
   data=episodes(task_dir(step,suite,task),task);count=len(data);success=sum(bool(x['success']) for x in data.values())
   tasks.append(dict(suite=suite,task_id=task,episodes=count,successes=success,success_rate=success/count if count else None));n+=count;k+=success
  suites[suite]=dict(episodes=n,successes=k,success_rate=k/n if n else None)
 n=sum(v['episodes'] for v in suites.values());k=sum(v['successes'] for v in suites.values())
 return dict(step=step,status='complete' if n==400 else 'partial' if n else 'queued',episodes_completed=n,expected_episodes=400,successes=k,success_rate=k/n if n else None,suites=suites,tasks=tasks,updated_at=time.time())

def refresh():
 steps=sorted(int(x.parent.name.split('_')[-1]) for x in SNAPSHOTS.glob('step_*/eval_snapshot_complete.json'))
 latest=max(steps,default=2000);rows=[]
 for step in range(2000,latest+1,250):
  if step not in steps:row=dict(step=step,status='missing_checkpoint',episodes_completed=0,expected_episodes=400,successes=None,success_rate=None,suites={})
  else:
   row=aggregate(step);atomic_json(OUTPUT/f'step_{step:06d}'/'summary.json',row)
  rows.append(row)
 atomic_json(OUTPUT/'checkpoint_results.json',dict(updated_at=time.time(),checkpoints=rows))
 fields=['step','status','episodes_completed','expected_episodes','successes','success_rate']+list(SUITES)
 stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
 for row in rows:
  writer.writerow({**{k:row[k] for k in fields if k not in SUITES},**{s:row['suites'].get(s,{}).get('success_rate') for s in SUITES}})
 tmp=OUTPUT/'checkpoint_results.csv.tmp';tmp.write_text(stream.getvalue());tmp.replace(OUTPUT/'checkpoint_results.csv')
 return rows

class Worker:
 def __init__(self,gpu,worker_id,stop_event):
  self.worker_id=worker_id;self.stop_event=stop_event;self.gpu=gpu;self.server=None;self.child=None;self.url=None;self.step=None;self.state=dict(worker=self.worker_id,gpu=gpu,phase='idle');self.log=None
  self.env=os.environ.copy();self.env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu))
 def start(self):
  if self.stop_event.is_set():raise RuntimeError('Queue stopping')
  self.stop()
  available=int(subprocess.check_output(['nvidia-smi',f'--id={self.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if available<16000:raise RuntimeError(f'GPU {self.gpu} has insufficient free memory: {available} MiB')
  ready=SSD/f'worker_{self.worker_id}_gpu{self.gpu}_ready.json';ready.unlink(missing_ok=True)
  self.log=(ROOT/f'runtime/goal_local/evaluation/server_{self.worker_id}_gpu{self.gpu}.log').open('a')
  self.server=subprocess.Popen([PYTHON,'-m','goal_local.eval_server','--ready',str(ready)],cwd=ROOT,env=self.env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+900
  while time.monotonic()<deadline:
   if self.stop_event.is_set():raise RuntimeError('Queue stopping')
   info=read_json(ready,{})
   if info.get('pid')==self.server.pid:
    self.url=f"http://127.0.0.1:{info['port']}";rpc(self.url,'/health');return
   if self.server.poll() is not None:raise RuntimeError(f'GPU {self.gpu} server exited: {self.server.returncode}')
   time.sleep(2)
  raise TimeoutError('Inference server startup timeout')
 def load(self,step):
  self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='loading',step=step)
  if self.server is None or self.server.poll() is not None:self.start()
  p=SNAPSHOTS/f'step_{step:06d}';info=rpc(self.url,'/load',dict(checkpoint=str(p)))
  if info['step']!=step or info['uses_state'] or info['uses_future']:raise RuntimeError('Unexpected policy loaded')
  self.step=step
 def run(self,step,jobs):
  failures=[]
  try:self.load(step)
  except Exception as e:
   self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='server_error',step=step,error=repr(e));traceback.print_exc();return [('server',repr(e))]
  while not self.stop_event.is_set():
   try:suite,task=jobs.get_nowait()
   except queue.Empty:break
   out=task_dir(step,suite,task);out.mkdir(parents=True,exist_ok=True)
   for attempt in range(1,4):
    if self.stop_event.is_set():break
    self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='evaluating',step=step,suite=suite,task_id=task,attempt=attempt)
    with (out/'run.log').open('a') as log:
     log.write(f'\n--- attempt {attempt}, GPU {self.gpu}, time {time.time()}, worker {self.worker_id} ---\n');log.flush()
     cmd=[SIM_PYTHON,'-m','goal_local.eval_task','--checkpoint',str(SNAPSHOTS/f'step_{step:06d}'),'--url',self.url,'--suite',suite,'--task-id',str(task),'--episodes','10','--output',str(out)]
     self.child=subprocess.Popen(cmd,cwd=ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:code=self.child.wait(timeout=14400)
     except subprocess.TimeoutExpired:self.end(self.child);code=-9
     self.child=None
    if code==0 and len(episodes(out,task))==10:break
    print(json.dumps(dict(event='task_retry',step=step,suite=suite,task=task,gpu=self.gpu,attempt=attempt,exit_code=code)),flush=True)
    if attempt==3:failures.append((suite,task,code))
    elif self.server.poll() is not None:
     try:self.load(step)
     except Exception as e:failures.append((suite,task,repr(e)));break
   jobs.task_done()
  self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='idle',step=step,failures=failures);return failures
 @staticmethod
 def end(process):
  if process is not None and process.poll() is None:
   # Only subprocesses created by this worker are ever signalled.
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:return
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 def stop(self):
  self.end(self.child);self.end(self.server);self.child=self.server=None
  if self.log:self.log.close();self.log=None

def protocol():
 def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 relative=Path('examples/LIBERO/eval_files/libero_eval_core.py')
 actual=digest(ROOT/relative);baseline=digest(BASELINE/relative)
 if actual!=baseline:raise RuntimeError('LIBERO rollout core differs from baseline')
 return dict(training_directory=str(TRAIN),snapshots=str(SNAPSHOTS),libero_home=os.environ['LIBERO_HOME'],libero_config_path=os.environ['LIBERO_CONFIG_PATH'],baseline_reference=str(BASELINE/'results/ablation/inference_without_lawm_and_queries/full_40x50_20260915'),rollout_sha256=actual,suites=list(SUITES),task_ids=list(range(10)),episodes_per_task=10,initial_state_indices=list(range(10)),simulator_seed=0,num_steps_wait=10,image_size=256,action_chunk=8,flow_sampling_steps=10,max_steps=dict(zip(SUITES,[250,300,320,550])),uses_state=False,uses_future=False,teacher_loaded=False,noise_seed='(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index',gripper_conversion='baseline core: 2*g-1',score='successful episodes / 400 only when all tasks complete',missing_checkpoint='NA, never treated as 0%')

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='6,6');p.add_argument('--through',type=int,default=10000);p.add_argument('--hold-after',type=int);a=p.parse_args();gpus=[int(v) for v in a.gpus.split(',')]
 if not gpus or any(g in (0,4) for g in gpus):raise ValueError('GPU 0 is reserved for training; GPU 4 is excluded from LIBERO evaluation')
 SSD.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
 lock=(SSD/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stop_event=threading.Event()
 manifest=protocol();manifest.update(evaluation_gpus=gpus,worker_ids=[f'w{i+1}' for i in range(len(gpus))],checkpoint_order='strict oldest incomplete first')
 atomic_json(OUTPUT/'protocol.json',manifest);workers=[Worker(g,f'w{i+1}',stop_event) for i,g in enumerate(gpus)]
 def stop_signal(*_):
  stop_event.set()
  for w in workers:w.stop()
  raise SystemExit(0)
 signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
 try:
  with ThreadPoolExecutor(max_workers=len(workers)) as pool:
   while True:
    rows=refresh();todo=[r['step'] for r in rows if r['status'] in ('queued','partial') and r['step']<=a.through and (SNAPSHOTS/f"step_{r['step']:06d}"/'model.pt').is_symlink()]
    if not todo:
     atomic_json(OUTPUT/'queue_status.json',dict(pid=os.getpid(),time=time.time(),phase='waiting_for_checkpoint',workers=[w.state for w in workers]))
     if any(r['step']==a.through and r['status']=='complete' for r in rows) and not any(r['status'] in ('queued','partial') for r in rows):return
     time.sleep(15);continue
    step=min(todo);jobs=queue.Queue()
    for suite in SUITES:
     for task in range(10):
      if len(episodes(task_dir(step,suite,task),task))<10:jobs.put((suite,task))
    print(json.dumps(dict(event='checkpoint_start',step=step,tasks=jobs.qsize(),gpus=gpus)),flush=True)
    futures=[pool.submit(w.run,step,jobs) for w in workers]
    while not all(f.done() for f in futures):
     refresh();atomic_json(OUTPUT/'queue_status.json',dict(pid=os.getpid(),time=time.time(),phase='evaluating',step=step,workers=[w.state for w in workers]));time.sleep(15)
    failures=[f.result() for f in futures];row=aggregate(step);refresh()
    print(json.dumps(dict(event='checkpoint_end',step=step,status=row['status'],episodes=row['episodes_completed'],successes=row['successes'],failures=failures)),flush=True)
    if row['status']!='complete':
     atomic_json(OUTPUT/'queue_status.json',dict(pid=os.getpid(),time=time.time(),phase='retrying_incomplete_checkpoint',step=step,workers=[w.state for w in workers]))
     stop_event.wait(30)
    elif step==a.hold_after:
     release=OUTPUT/f'continue_after_{step:06d}.json'
     while read_json(release,{}).get('pid')!=os.getpid():
      atomic_json(OUTPUT/'queue_status.json',dict(pid=os.getpid(),time=time.time(),phase='awaiting_result_report',step=step,release_file=str(release),workers=[w.state for w in workers]))
      stop_event.wait(3)
      if stop_event.is_set():return
 finally:
  stop_event.set()
  for w in workers:w.stop()
if __name__=='__main__':main()
