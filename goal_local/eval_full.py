"""Four task workers on GPU 6: student step 4000, LIBERO 40 x 50."""
import argparse,csv,fcntl,io,json,os,queue,signal,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .eval_common import ROOT,SSD,SNAPSHOTS,SUITES,SIM_PYTHON,atomic_json,read_json,model_metadata
from .eval_queue import Worker,protocol

FULL=ROOT/'results/goal_local/libero_40x50/step_004000'
_CACHE={}
def task_path(suite,task):return FULL/'tasks'/f'{suite}__task{task:02d}'
def read_episodes(path,task):
 file=path/'episodes.checkpoint.jsonl'
 try:stat=file.stat()
 except FileNotFoundError:return {}
 signature=(stat.st_mtime_ns,stat.st_size)
 if file in _CACHE and _CACHE[file][0]==signature:return _CACHE[file][1]
 rows={}
 for line in file.read_text().splitlines():
  try:d=json.loads(line)
  except ValueError:continue
  if d['task_id']!=task or not 0<=d['episode_idx']<50:raise ValueError(f'Unexpected episode: {file}')
  if d.get('error'):continue
  if d['episode_idx'] in rows and rows[d['episode_idx']]!=d:raise ValueError(f'Conflicting duplicate: {file}')
  rows[d['episode_idx']]=d
 _CACHE[file]=(signature,rows)
 return rows
def task_result(suite,task):
 path=task_path(suite,task);rows=read_episodes(path,task)
 n=len(rows);success=sum(bool(x['success']) for x in rows.values())
 videos=[p for p in path.glob('*_success.mp4') if p.stat().st_size>0]
 if len(videos)>1:raise ValueError(f'Multiple success videos: {path}')
 video_ok=bool(videos) or (n==50 and success==0)
 return dict(suite=suite,task_id=task,episodes=n,successes=success,success_rate=success/n if n else None,
  complete=n==50 and video_ok,video=str(videos[0]) if videos else None,
  video_status='saved' if videos else 'no_success' if n==50 and success==0 else 'pending',
  task_description=next(iter(rows.values()),{}).get('task_description'))
def summarize():
 tasks=[task_result(s,t) for s in SUITES for t in range(10)]
 n=sum(t['episodes'] for t in tasks);k=sum(t['successes'] for t in tasks)
 suites={}
 for suite in SUITES:
  group=[t for t in tasks if t['suite']==suite];count=sum(t['episodes'] for t in group);passed=sum(t['successes'] for t in group)
  suites[suite]=dict(episodes=count,expected_episodes=500,successes=passed,success_rate=passed/count if count else None)
 completed=sum(t['complete'] for t in tasks)
 result=dict(step=4000,status='complete' if completed==40 else 'partial' if n else 'queued',episodes_completed=n,
  expected_episodes=2000,successes=k,success_rate=k/n if n else None,tasks_completed=completed,expected_tasks=40,
  success_videos=sum(bool(t['video']) for t in tasks),suites=suites,tasks=tasks,updated_at=time.time())
 atomic_json(FULL/'summary.json',result)
 stream=io.StringIO();fields=['suite','task_id','episodes','successes','success_rate','complete','video_status','video','task_description']
 writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(tasks)
 tmp=FULL/'task_results.csv.tmp';tmp.write_text(stream.getvalue());tmp.replace(FULL/'task_results.csv')
 atomic_json(FULL/'videos.json',[dict(suite=t['suite'],task_id=t['task_id'],path=t['video']) for t in tasks if t['video']])
 return result

class FullWorker(Worker):
 def run_full(self,jobs):
  failures=[]
  try:self.load(4000)
  except Exception as e:
   self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='server_error',error=repr(e));traceback.print_exc();return [repr(e)]
  while not self.stop_event.is_set():
   try:suite,task=jobs.get_nowait()
   except queue.Empty:break
   out=task_path(suite,task);out.mkdir(parents=True,exist_ok=True)
   for attempt in range(1,4):
    if self.stop_event.is_set():break
    self.state=dict(worker=self.worker_id,gpu=6,phase='evaluating',step=4000,suite=suite,task_id=task,attempt=attempt)
    atomic_json(out/'assignment.json',dict(worker=self.worker_id,gpu=6,attempt=attempt,time=time.time()))
    with (out/'run.log').open('a') as log:
     log.write(f'\n--- worker {self.worker_id}, GPU 6, attempt {attempt}, time {time.time()} ---\n');log.flush()
     command=[SIM_PYTHON,'-m','goal_local.eval_task','--checkpoint',str(SNAPSHOTS/'step_004000'),
      '--url',self.url,'--suite',suite,'--task-id',str(task),'--episodes','50','--output',str(out),'--first-success-video']
     self.child=subprocess.Popen(command,cwd=ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:code=self.child.wait(timeout=14400)
     except subprocess.TimeoutExpired:self.end(self.child);code=-9
     self.child=None
    if self.stop_event.is_set():break
    result=task_result(suite,task)
    if code==0 and result['complete']:
     print(json.dumps(dict(event='task_complete',worker=self.worker_id,**result)),flush=True);break
    print(json.dumps(dict(event='task_retry',worker=self.worker_id,suite=suite,task_id=task,attempt=attempt,exit_code=code)),flush=True)
    if attempt==3:failures.append(dict(suite=suite,task_id=task,exit_code=code))
    elif self.server is None or self.server.poll() is not None:
     try:self.load(4000)
     except Exception as e:failures.append(dict(suite=suite,task_id=task,error=repr(e)));break
   jobs.task_done()
  self.state=dict(worker=self.worker_id,gpu=6,phase='idle',step=4000,failures=failures)
  return failures

def status(phase,workers,**extra):
 atomic_json(FULL/'queue_status.json',dict(pid=os.getpid(),time=time.time(),phase=phase,step=4000,gpu=6,
  workers=[{**w.state,'server_pid':w.server.pid if w.server else None,'task_pid':w.child.pid if w.child else None} for w in workers],**extra))

def run():
 FULL.mkdir(parents=True,exist_ok=True)
 lock=(SSD/'full_4000_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 meta=model_metadata(SNAPSHOTS/'step_004000/model.pt')
 if meta['mode']!='student' or meta['step']!=4000:raise RuntimeError('Wrong student checkpoint')
 manifest=protocol();manifest.update(step=4000,checkpoint=str(SNAPSHOTS/'step_004000'),episodes_per_task=50,
  initial_state_indices=list(range(50)),expected_episodes=2000,evaluation_gpus=[6,6,6,6],workers=4,
  scheduling='Shared task queue: one task per worker, 50 episodes sequentially, then claim next task',
  save_videos='Only first successful episode per task; no failure videos; stop recording after first success',
  score='Successful episodes / 2000 after all 40 tasks finish',started_at=time.time())
 atomic_json(FULL/'protocol.json',manifest)
 stop=threading.Event();workers=[FullWorker(6,f'full4000_w{i+1}',stop) for i in range(4)]
 def shutdown(*_):
  stop.set()
  for w in workers:w.stop()
  status('stopped',workers)
  raise SystemExit(0)
 signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
 phase='starting'
 try:
  status(phase,workers)
  with ThreadPoolExecutor(max_workers=4) as pool:
   while not stop.is_set():
    summary=summarize()
    if summary['status']=='complete':phase='completed';break
    jobs=queue.Queue()
    for task in summary['tasks']:
     if not task['complete']:jobs.put((task['suite'],task['task_id']))
    print(json.dumps(dict(event='round_start',tasks=jobs.qsize(),completed_episodes=summary['episodes_completed'],gpu=6,workers=4)),flush=True)
    futures=[pool.submit(w.run_full,jobs) for w in workers]
    while not all(f.done() for f in futures):
     status('evaluating',workers);summarize();stop.wait(10)
    failures=[]
    for f in futures:
     try:failures.extend(f.result())
     except Exception as e:traceback.print_exc();failures.append(repr(e))
    summary=summarize()
    if summary['status']=='complete':phase='completed';break
    status('retrying_incomplete_tasks',workers,failures=failures);stop.wait(30)
 finally:
  stop.set()
  for w in workers:
   w.stop();w.state.update(phase='stopped')
  status(phase if phase=='completed' else 'stopped',workers)
  summary=summarize()
  print(json.dumps(dict(event='queue_end',status=summary['status'],episodes=summary['episodes_completed'],successes=summary['successes'],videos=summary['success_videos'])),flush=True)

def dashboard(once=False):
 while True:
  row=read_json(FULL/'summary.json',{});state=read_json(FULL/'queue_status.json',{})
  n=row.get('episodes_completed',0);k=row.get('successes',0);filled=int(24*n/2000)
  lines=[f'LIBERO FULL | student 4000 | GPU 6 x4 | {time.strftime("%m-%d %H:%M:%S")}',
   f'Queue: {state.get("phase","starting")} | heartbeat age {time.time()-state.get("time",time.time()):.0f}s',
   '['+'#'*filled+'-'*(24-filled)+f'] {n}/2000 ({100*n/2000:.1f}%)',
   f'Success {k}/{n} = {100*k/n if n else 0:.2f}% | tasks {row.get("tasks_completed",0)}/40 | videos {row.get("success_videos",0)}/40']
  tasks={(t['suite'],t['task_id']):t for t in row.get('tasks',[])}
  for worker in state.get('workers',[]):
   lines+=['',f'{worker["worker"]} | GPU 6 | {worker.get("phase","starting")}']
   key=(worker.get('suite'),worker.get('task_id'));t=tasks.get(key,{})
   if t:
    count=t['episodes'];passed=t['successes']
    lines +=[f'  {key[0]} task {key[1]:02d}: {count}/50 | success {passed}/{count} ({100*passed/count if count else 0:.1f}%)']
  lines+=['','SUITE              EPISODES    SUCCESS RATE']
  for suite in SUITES:
   s=row.get('suites',{}).get(suite,{});count=s.get('episodes',0);passed=s.get('successes',0)
   lines.append(f'{suite:18s} {count:3d}/500     {passed:3d}/{count:<3d} {100*passed/count if count else 0:.2f}%')
  lines+=['Final score requires 2000 episodes. Detach: Ctrl-b then d']
  if not once:sys.stdout.write('\033[H\033[2J')
  print('\n'.join(lines),flush=True)
  if once:return
  time.sleep(3)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dashboard',action='store_true');parser.add_argument('--once',action='store_true');args=parser.parse_args()
 dashboard(args.once) if args.dashboard else run()

