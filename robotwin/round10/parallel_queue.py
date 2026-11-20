"""Round10 paired full benchmark with original GPU0 and compatible GPU3 pipelined."""
import json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'robotwin/round10';OUT=R/'online';SIM='/mnt/hdd/wangchaoyang/results/LaWAM/runtime/RoboTwin';PY='/home/wangchaoyang/envs/LaWAM/bin/python';SIMPY='/home/wangchaoyang/envs/RoboTwin4LaWAM/bin/python'
from examples.Robotwin.eval_files.robotwin_eval_common import DEFAULT_TASKS
CKPT='/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt'
TASKS=list(DEFAULT_TASKS);assert len(TASKS)==50;MODES=['demo_clean','demo_randomized'];PAIRS=[(t,m) for t in TASKS for m in MODES];GPUS={'original':'0','compatible':'3'};PORTS={'original':26690,'compatible':26691};TARGET_EP=10;STALL=1800

def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def listening(port):
 try:
  with socket.create_connection(('127.0.0.1',port),timeout=1):return True
 except OSError:return False
def stop(p):
 if not p:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=20)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait()
def summary_for(job):
 ps=list((job/'tasks').glob('*/summary.json'))
 return read(ps[0]) if ps else {}
def progress_for(job):
 # bridge.py nests the run directory below the queue job directory. Search
 # recursively so the queue observes the progress file it actually writes.
 ps=list(job.rglob('progress.json'))
 return read(ps[0]) if ps else {}
def activity_mtime(job):
 """Return the latest bridge/request/progress heartbeat for a running job."""
 paths=list(job.rglob('bridge.log'))+list(job.rglob('requests.jsonl'))+list(job.rglob('progress.json'))
 mt=[]
 for p in paths:
  try: mt.append(p.stat().st_mtime)
  except FileNotFoundError: pass
 return max(mt, default=0.0)
def base_env():
 e=os.environ.copy()
 for k in list(e):
  if k.startswith('ROBOTWIN_') or k in ('OUTPUT_ROOT','RESUME_RUN_DIR','GPU_IDS'):e.pop(k)
 e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',ROBOTWIN_PATH=SIM,ROBOTWIN_NUM_SLOTS='1',ROBOTWIN_TEST_NUM=str(TARGET_EP),ROBOTWIN_SAVE_VIDEO='0',ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN='1',ROBOTWIN_SHARED_WARP_CACHE_DIR=str(OUT/'cache/warp'))
 (OUT/'cache/warp').mkdir(parents=True,exist_ok=True);return e
def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'parallel.lock').open('w')
 try: import fcntl;fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except (ImportError,BlockingIOError):raise RuntimeError('parallel queue already running')
 assert not (OUT/'parallel_manifest.json').exists(),'refusing overwrite'
 jobs=[]
 for i,(t,m) in enumerate(PAIRS):
  for a in ('original','compatible'):jobs.append(dict(index=i,task=t,mode=m,arm=a,key=f'{t}__{m}__{a}'))
 manifest=dict(started_at=time.time(),gpu_map=GPUS,ports=PORTS,tasks=TASKS,modes=MODES,total_episodes=len(PAIRS)*TARGET_EP*2,episodes_per_job=TARGET_EP,stall_timeout_sec=STALL,model_order='original GPU0 and compatible GPU3 pipelined; compatible reuses original accepted seeds',jobs=jobs)
 write(OUT/'parallel_manifest.json',manifest)
 env=base_env();servers={};active={};results=[];pairs=[];orig_next=0;comp_next=0;orig_proc=None;comp_proc=None;orig_meta=None;comp_meta=None
 def state(phase,**kw):
  n=sum(len(x.get('summary',{}).get('episodes',[])) for x in results)
  for meta in (orig_meta,comp_meta):
   if meta:n+=int(progress_for(meta['job']).get('completed_episodes',0))
  write(OUT/'status.json',dict(phase=phase,updated_at=time.time(),target_episodes=len(PAIRS)*TARGET_EP*2,results=results,pairs=pairs,completed_jobs=len(results),active={a:(dict(task=m['task'],mode=m['mode'],arm=a,key=m['key'],gpu=GPUS[a],progress=progress_for(m['job']),no_episode_progress_seconds=time.time()-m['last']) if m else None) for a,m in [('original',orig_meta),('compatible',comp_meta)]},**kw))
 def launch_server(arm):
  ready=OUT/f'{arm}_ready.json';ready.unlink(missing_ok=True);ee=env.copy();ee['CUDA_VISIBLE_DEVICES']=GPUS[arm]
  with (OUT/f'{arm}_server.log').open('a') as log:p=subprocess.Popen([PY,'-m','robotwin.round10.server','--arm',arm,'--ckpt_path',CKPT,'--port',str(PORTS[arm]),'--ready',str(ready)],cwd=ROOT,env=ee,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  t=time.time();servers[arm]=p
  while True:
   if p.poll() is not None:raise RuntimeError(f'{arm} server failed rc={p.returncode}')
   if ready.exists() and listening(PORTS[arm]):return
   if time.time()-t>1800:raise TimeoutError(f'{arm} server startup timeout')
   state('loading_and_checking_policy',server=arm,server_pid=p.pid,elapsed=time.time()-t);time.sleep(5)
 def launch_job(pair_i,arm,seeds=None):
  t,m=PAIRS[pair_i];key=f'{t}__{m}__{arm}';job=OUT/'jobs'/key;job.mkdir(parents=True,exist_ok=True);ee=env.copy();ee.update(CUDA_VISIBLE_DEVICES=GPUS[arm],PAIR_TASK=t,PAIR_MODE=m,PAIR_ARM=arm,ROBOTWIN_EVAL_ROOT=str(job),ROBOTWIN_RUN_TAG='round10_parallel',PYTHONPATH=f'{SIM}:{ROOT}:{ROOT}/examples/Robotwin/eval_files')
  if seeds:ee['ROBOTWIN_FIXED_SEEDS']=','.join(map(str,seeds))
  cmd=[SIMPY,str(R/'bridge.py'),'--config',str(ROOT/'examples/Robotwin/eval_files/deploy_policy.yml'),'--overrides','--task_name',t,'--task_config',m,'--ckpt_setting','round10_parallel','--seed','0','--instruction_type','seen','--policy_name','model2robotwin_interface','--host','127.0.0.1','--port',str(PORTS[arm]),'--policy_ckpt_path',CKPT,'--replan_steps','36','--action_ensemble','0','--action_ensemble_alpha','0']
  log=(job/'bridge.log').open('a');p=subprocess.Popen(cmd,cwd=SIM,env=ee,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);return dict(proc=p,pair_i=pair_i,arm=arm,task=t,mode=m,key=key,job=job,last=time.time(),last_count=0,log=log)
 def finalize(meta):
  if not meta:return None
  p=meta['proc'];rc=p.poll();stop(p);meta['log'].close();sm=summary_for(meta['job']);pr=progress_for(meta['job']);eps=sm.get('episodes',[]);timed=meta.get('timedout',False);complete=rc==0 and len(eps)==TARGET_EP
  row=dict(task=meta['task'],mode=meta['mode'],arm=meta['arm'],key=meta['key'],gpu=GPUS[meta['arm']],status='completed' if complete else 'skipped_timeout' if timed else 'incomplete_error',exit_code=rc,summary=sm,partial_progress=pr,elapsed=time.time()-meta['started']);write(meta['job']/'result.json',row);results.append(row);return row
 try:
  launch_server('original');launch_server('compatible')
  # Fill pipeline: original pair i runs while compatible pair i-1 runs; each arm has one active process.
  orig_proc=launch_job(0,'original');orig_proc['started']=time.time();orig_meta=orig_proc;orig_next=1
  while orig_meta or comp_meta or orig_next<len(PAIRS) or comp_next<len(PAIRS):
   for name in ('original','compatible'):
    meta=orig_meta if name=='original' else comp_meta
    if not meta:continue
    pr=progress_for(meta['job']);count=int(pr.get('completed_episodes',0))
    if count>meta['last_count']:meta['last_count']=count;meta['last']=time.time()
    # A long episode can legitimately take more than 30 minutes. The user
    # timeout is for no task activity, so bridge/request heartbeats reset it.
    activity=activity_mtime(meta['job'])
    if activity>meta.get('activity_mtime',0.0):
     meta['activity_mtime']=activity;meta['last']=time.time()
    if time.time()-meta['last']>=STALL:
     meta['timedout']=True;stop(meta['proc']);meta['proc'].wait();
     if name=='original':orig_meta=None
     else:comp_meta=None
     finalize(meta);continue
    if meta['proc'].poll() is not None:
     row=finalize(meta)
     if name=='original':
      seeds=[int(e['seed']) for e in row['summary'].get('episodes',[])] if row else []
      if row and row['status']=='completed' and len(seeds)==TARGET_EP:
       if comp_meta is None:comp_meta=launch_job(meta['pair_i'],'compatible',seeds);comp_meta['started']=time.time()
      else: raise RuntimeError(f'original job failed: {meta["key"]}')
      orig_meta=None
     else:
      if row and row['status']=='completed':
       peer=next((x for x in results[:-1] if x['task']==row['task'] and x['mode']==row['mode'] and x['arm']=='original'),None)
       if peer:
        a={(e['seed'],e.get('instruction')) for e in peer['summary'].get('episodes',[])};b={(e['seed'],e.get('instruction')) for e in row['summary'].get('episodes',[])}
        if a!=b:raise RuntimeError(f'pair mismatch {row["key"]}')
        pairs.append(dict(task=row['task'],mode=row['mode'],matched=True,original=peer['summary']['successes'],compatible=row['summary']['successes'],episodes=TARGET_EP))
      comp_meta=None
   if orig_meta is None and orig_next<len(PAIRS):orig_meta=launch_job(orig_next,'original');orig_meta['started']=time.time();orig_next+=1
   if comp_meta is None and comp_next<len(PAIRS):
    # Only start compatible for a completed original peer; select next pair whose original result exists.
    ready=[i for i in range(comp_next,orig_next) if not any(x['task']==PAIRS[i][0] and x['mode']==PAIRS[i][1] and x['arm']=='compatible' for x in results)]
    if ready:
     i=ready[0];peer=next((x for x in results if x['task']==PAIRS[i][0] and x['mode']==PAIRS[i][1] and x['arm']=='original'),None)
     if peer and peer['status']=='completed':
      seeds=[int(e['seed']) for e in peer['summary']['episodes']];comp_meta=launch_job(i,'compatible',seeds);comp_meta['started']=time.time();comp_next=max(comp_next,i+1)
   state('evaluating',orig_next=orig_next,comp_next=comp_next)
   time.sleep(5)
  state('completed',completed_jobs=len(results))
 except BaseException as e:
  state('failed',error=repr(e));traceback.print_exc();raise
 finally:
  for m in (orig_meta,comp_meta):
   if m:stop(m['proc'])
  for p in servers.values():stop(p)
if __name__=='__main__':main()
