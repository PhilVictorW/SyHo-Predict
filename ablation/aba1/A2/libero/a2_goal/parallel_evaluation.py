"""A2 full LIBERO evaluation: six independent model servers on physical GPU 6."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

from .config import WORK, ROOT
from .launcher import completed_stage, evaluation_complete
from ablation.aba1.libero_common.config import atomic_json, digest_json, file_sha256, load_config
from ablation.aba1.libero_common.evaluation import episode_rows, summarize, stop_child
from goal_local.eval_client import rpc
from goal_local.eval_common import SUITES, SIM_PYTHON, BASELINE, model_metadata

SETTINGS = WORK/'evaluation_parallel.json'

def read(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {}

def load_settings():
    settings=read(SETTINGS)
    if (type(settings.get('workers')) is not int or not 1<=settings['workers']<=32
        or type(settings.get('physical_gpu')) is not int or settings['physical_gpu']<0
        or not isinstance(settings.get('load_gap_seconds'),(int,float)) or settings['load_gap_seconds']<0):
        raise ValueError('Invalid parallel evaluation settings')
    return settings

def task_path(output,suite,task):
    return Path(output)/'tasks'/f'{suite}__task{task:02d}'

def pending_tasks(output):
    return [(suite,task) for suite in SUITES for task in range(10)
            if len(episode_rows(task_path(output,suite,task),task,50))<50]

def live_rows(path):
    result={}
    try:lines=(Path(path)/'episodes.checkpoint.jsonl').read_text().splitlines()
    except FileNotFoundError:return result
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:row=json.loads(line)
        except ValueError:
            if i==len(lines)-1:continue
            raise
        if not row.get('error'):result[row['episode_idx']]=row
    return result

def live_progress(output):
    episodes=successes=completed=0
    for suite in SUITES:
        for task in range(10):
            rows=live_rows(task_path(output,suite,task))
            episodes+=len(rows);successes+=sum(bool(x['success']) for x in rows.values());completed+=len(rows)==50
    return dict(episodes_completed=episodes,expected_episodes=2000,tasks_completed=completed,
                successes=successes,success_rate=successes/episodes if episodes else None)

def env_for(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        LIBERO_HOME=str(BASELINE/'runtime/LIBERO'),LIBERO_CONFIG_PATH=str(BASELINE/'runtime/libero_config'),
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PYTHONPATH']=os.pathsep.join([str(WORK),str(ROOT),env['LIBERO_HOME']])
    return env

class ServerPool:
    def __init__(self,workers,gpu,checkpoint,output,checksum,stop,gap,sim_python=SIM_PYTHON,first_success_video=False):
        self.gpu=gpu;self.sim_python=sim_python;self.first_success_video=first_success_video
        self.checkpoint=checkpoint;self.output=output;self.checksum=checksum;self.stop_event=stop
        self.gap=gap;self.load_lock=threading.Lock();self.last_loaded=0
        self.entries={slot:dict(server=None,url=None,phase='queued',health=None) for slot in range(workers)}

    def ensure(self,slot):
        entry=self.entries[slot]
        server=entry['server']
        if server is not None and server.poll() is None and entry['url']:
            return entry['url']
        entry['phase']='waiting_for_load'
        with self.load_lock:
            delay=max(0,self.last_loaded+self.gap-time.time())
            if self.stop_event.wait(delay):raise RuntimeError('Evaluation stopping')
            if self.stop_event.is_set():raise RuntimeError('Evaluation stopping')
            stop_child(server)
            folder=self.output/'.parallel_workers'/f'slot{slot}';folder.mkdir(parents=True,exist_ok=True)
            ready=folder/'server_ready.json'
            command=[sys.executable,'-B','-m','ablation.aba1.libero_common.server',
                     '--variant','A2','--device','cuda:0','--ready',str(ready)]
            with (folder/'server.log').open('a') as log:
                server=subprocess.Popen(command,cwd=ROOT,env=env_for(self.gpu),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry.update(server=server,url=None,phase='loading_model',load_started_at=time.time(),health=None)
            if self.stop_event.is_set():
                stop_child(server);raise RuntimeError('Evaluation stopping')
            deadline=time.monotonic()+900
            while True:
                if self.stop_event.is_set():raise RuntimeError('Evaluation stopping')
                if server.poll() is not None:raise RuntimeError(f'Worker {slot} on GPU {self.gpu} inference server exited {server.returncode}')
                info=read(ready)
                if info.get('pid')==server.pid and info.get('port'):break
                if time.monotonic()>deadline:raise TimeoutError(f'Worker {slot} on GPU {self.gpu} server startup')
                self.stop_event.wait(1)
            url=f"http://127.0.0.1:{info['port']}"
            loaded=rpc(url,'/load',dict(checkpoint=str(self.checkpoint)))
            expected=dict(variant='A2',step=2000,model_sha256=self.checksum,physical_gpu=str(self.gpu),
                local_slots=0,goal_slots=16,interaction=False,teacher_loaded=False,uses_future=False,uses_state=False)
            if any(loaded.get(k)!=v for k,v in expected.items()):
                raise ValueError(f'Worker {slot} on GPU {self.gpu} loaded a different model: {loaded}')
            if self.stop_event.is_set():raise RuntimeError('Evaluation stopping')
            self.last_loaded=time.time()
            entry.update(url=url,phase='ready',load_finished_at=self.last_loaded,health=loaded)
            atomic_json(folder/'loaded_policy.json',dict(loaded,load_started_at=entry['load_started_at'],
                load_finished_at=self.last_loaded))
            print(json.dumps(dict(event='model_loaded',slot=slot,physical_gpu=self.gpu,pid=server.pid,time=time.time())),flush=True)
            return url

    def reset(self,slot):
        entry=self.entries[slot];stop_child(entry['server'])
        entry.update(server=None,url=None,phase='retry_pending',health=None)

    def stop(self):
        for entry in self.entries.values():stop_child(entry['server'])

class Worker:
    def __init__(self,slot,pool,stop):
        self.slot=slot;self.pool=pool;self.stop_event=stop;self.child=None
        self.state=dict(slot=slot,phase='queued',tasks_finished=0)

    def run(self,jobs):
        failures=[];finished=0
        try:
            while not self.stop_event.is_set():
                try:suite,task=jobs.get_nowait()
                except queue.Empty:break
                out=task_path(self.pool.output,suite,task);out.mkdir(parents=True,exist_ok=True)
                complete=False;last_error=None
                try:
                    for attempt in range(1,4):
                        if self.stop_event.is_set():break
                        self.state=dict(slot=self.slot,phase='preparing',suite=suite,task_id=task,
                                        attempt=attempt,tasks_finished=finished)
                        try:
                            existing=len(episode_rows(out,task,50))
                            if existing==50:complete=True;break
                            url=self.pool.ensure(self.slot)
                            if self.stop_event.is_set():break
                            cmd=[self.pool.sim_python,'-B','-m','goal_local.eval_task','--checkpoint',str(self.pool.checkpoint),
                                '--url',url,'--suite',suite,'--task-id',str(task),'--episodes','50','--output',str(out)]
                            if self.pool.first_success_video:cmd.append('--first-success-video')
                            with (out/'run.log').open('a') as log:
                                log.write(f'\nParallel A2 worker {self.slot} on GPU {self.pool.gpu}, attempt {attempt}, preserved {existing} episodes, time {time.time()}\n')
                                log.flush()
                                self.child=subprocess.Popen(cmd,cwd=ROOT,env=env_for(self.pool.gpu),
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            self.state=dict(self.state,phase='evaluating',task_pid=self.child.pid,
                                task_started_at=time.time(),episodes_at_start=existing)
                            if self.stop_event.is_set():stop_child(self.child)
                            deadline=time.monotonic()+14400
                            while self.child.poll() is None:
                                if self.stop_event.is_set() or self.pool.entries[self.slot]['server'].poll() is not None:
                                    stop_child(self.child);break
                                if time.monotonic()>deadline:
                                    stop_child(self.child);raise TimeoutError('Task timeout')
                                self.stop_event.wait(1)
                            code=self.child.returncode;self.child=None
                            complete=code==0 and len(episode_rows(out,task,50))==50
                            if not complete:raise RuntimeError(f'Incomplete task {suite}/{task}: exit {code}')
                            break
                        except Exception as error:
                            last_error=repr(error);self.state=dict(self.state,phase='retry_pending',error=last_error)
                            stop_child(self.child);self.child=None;self.pool.reset(self.slot)
                            print(json.dumps(dict(event='task_retry',slot=self.slot,suite=suite,task=task,attempt=attempt,error=last_error)),flush=True)
                            if not self.stop_event.is_set() and attempt<3:self.stop_event.wait(10)
                    if complete:finished+=1
                    elif not self.stop_event.is_set():failures.append(dict(suite=suite,task_id=task,error=last_error))
                finally:jobs.task_done()
            self.state=dict(slot=self.slot,phase='idle',tasks_finished=finished,failures=failures)
            return failures
        finally:stop_child(self.child)

    def snapshot(self):
        state=dict(self.state);entry=self.pool.entries[self.slot];server=entry['server']
        state.update(physical_gpu=self.pool.gpu,server_pid=server.pid if server else None,server_alive=server is not None and server.poll() is None)
        if state['phase'] in ('queued','preparing'):state['phase']=entry['phase']
        if entry['url'] and state['server_alive']:
            try:entry['health']=rpc(entry['url'],'/health',timeout=2)
            except Exception:pass
        if entry.get('health'):state['calls']=entry['health'].get('calls',0)
        for key in ('load_started_at','load_finished_at'):
            if key in entry:state[key]=entry[key]
        return state

def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--protocol',choices=['quick','full'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--sim-python',default=SIM_PYTHON)
    parser.add_argument('--first-success-video',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.protocol=='quick':
        from ablation.aba1.libero_common.evaluation import main as sequential_main
        return sequential_main('A2',WORK/'config.json',argv)
    settings=load_settings();checkpoint=Path(args.checkpoint).resolve();output=Path(args.output).resolve()
    if not checkpoint.is_relative_to(WORK/'runs') or checkpoint.name!='step_002000':
        parser.error('Full evaluation requires this A2 run final step2000 checkpoint')
    run=checkpoint.parents[2]
    if checkpoint!=run/'stage_II/checkpoints/step_002000':parser.error('Expected Stage II checkpoint')
    if not output.is_relative_to(WORK/'evaluation') or output==WORK/'evaluation':parser.error('Invalid evaluation output')
    if args.device!='cuda:0':parser.error('Use logical cuda:0; all six servers bind to the configured physical GPU')
    cfg=load_config(WORK/'config.json','A2')
    protocol=dict(variant='A2',checkpoint=str(checkpoint),protocol='full',
        episodes_per_task=50,expected_episodes=2000,suites=list(SUITES),task_ids=list(range(10)),
        initial_state_indices=list(range(50)),simulator_seed=0,num_steps_wait=10,
        action_chunk=8,flow_steps=10,max_steps=dict(zip(SUITES,[250,300,320,550])),
        uses_state=False,uses_future=False,teacher_loaded=False,
        noise_seed='(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index',
        first_success_video=args.first_success_video,
        rollout_sha256=file_sha256(ROOT/'examples/LIBERO/eval_files/libero_eval_core.py'))
    execution=dict(workers=settings['workers'],physical_gpus=[settings['physical_gpu']]*settings['workers'],
        load_gap_seconds=settings['load_gap_seconds'],sequential_model_loading=True,
        one_model_server_per_worker=True,one_simulator_per_worker=True,
        task_assignment='shared_queue_each_task_once',resume_completed_episodes=True)
    if args.dry_run:
        print(json.dumps(dict(protocol=protocol,execution=execution,checkpoint_exists=(checkpoint/'model.pt').is_file(),starts_processes=False),indent=2));return
    if not Path(args.sim_python).is_file():raise FileNotFoundError(args.sim_python)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=completed_stage(run,'II',cfg,2000)
        checksum=final['model_sha256'];meta=model_metadata(checkpoint/'model.pt')
        protocol.update(checkpoint_sha256=checksum,config_sha256=digest_json(meta['config']),step=2000,extra=meta['extra'])
        manifest=output/'protocol.json'
        if manifest.exists() and read(manifest)!=protocol:raise ValueError('Output belongs to another checkpoint/protocol')
        if not manifest.exists() and (output/'tasks').exists():raise ValueError('Unowned existing task results')
        atomic_json(manifest,protocol)
        atomic_json(output/'parallel_plan.json',dict(execution,checkpoint=str(checkpoint),model_sha256=checksum,
            expected_episodes=2000,episodes_per_task=50,total_tasks=40,pid=os.getpid(),time=time.time()))
        stop=threading.Event();pool=None;workers=[];executor=None
        def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
        def publish(phase,**extra):
            snapshots=[worker.snapshot() for worker in workers]
            for item in snapshots:
                if item.get('suite') is not None:
                    item['episodes_completed']=len(live_rows(task_path(output,item['suite'],item['task_id'])))
            state=dict(phase=phase,run=str(run),output=str(output),evaluation_pid=os.getpid(),
                physical_gpus=execution['physical_gpus'],worker_count=settings['workers'],checkpoint_step=2000,
                workers=snapshots,**live_progress(output),**extra,updated_at=time.time())
            atomic_json(output/'status.json',state);return state
        try:
            summary=summarize(output,50,checksum)
            if summary['status']=='complete':evaluation_complete(output,checksum);publish('completed');return
            jobs=queue.Queue()
            for task in pending_tasks(output):jobs.put(task)
            pool=ServerPool(settings['workers'],settings['physical_gpu'],checkpoint,output,checksum,
                stop,settings['load_gap_seconds'],args.sim_python,args.first_success_video)
            workers=[Worker(slot,pool,stop) for slot in range(settings['workers'])]
            executor=ThreadPoolExecutor(max_workers=settings['workers'])
            futures=[executor.submit(worker.run,jobs) for worker in workers]
            last_print=0
            while not all(f.done() for f in futures):
                state=publish('evaluating_parallel')
                if time.time()-last_print>=30:
                    print(json.dumps(state,ensure_ascii=False),flush=True);last_print=time.time()
                time.sleep(5)
            failures=[error for future in futures for error in future.result()]
            summary=summarize(output,50,checksum)
            if failures:raise RuntimeError(f'Incomplete evaluation: {failures}')
            evaluation_complete(output,checksum)
            publish('completed');print(json.dumps(summary,ensure_ascii=False),flush=True)
        except BaseException:
            publish('interrupted_or_failed',error=traceback.format_exc());raise
        finally:
            stop.set()
            for worker in workers:stop_child(worker.child)
            if pool:pool.stop()
            if executor:executor.shutdown(wait=True,cancel_futures=True)
            if pool:pool.stop()
            for sig,handler in previous.items():signal.signal(sig,handler)

if __name__=='__main__':main()
