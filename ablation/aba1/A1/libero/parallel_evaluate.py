#!/usr/bin/env python3
"""A1-only four-GPU continuation of the existing LIBERO full evaluation."""
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

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[3]
sys.path[:0]=[str(WORK),str(ROOT)]
import watch_full_eval as watch
from ablation.aba1.libero_common.config import atomic_json, digest_json, file_sha256
from ablation.aba1.libero_common.evaluation import episode_rows, summarize, stop_child
from goal_local.eval_client import rpc
from goal_local.eval_common import SUITES, SIM_PYTHON, BASELINE, model_metadata

def task_path(output,suite,task):
    return Path(output)/'tasks'/f'{suite}__task{task:02d}'

def pending_tasks(output):
    return [(suite,task) for suite in SUITES for task in range(10)
            if len(episode_rows(task_path(output,suite,task),task,50))<50]

def validate_protocol(output,checkpoint,checksum):
    meta=model_metadata(checkpoint/'model.pt')
    expected=dict(variant='A1',checkpoint=str(checkpoint),protocol='full',
        episodes_per_task=50,expected_episodes=2000,suites=list(SUITES),task_ids=list(range(10)),
        initial_state_indices=list(range(50)),simulator_seed=0,num_steps_wait=10,
        action_chunk=8,flow_steps=10,max_steps=dict(zip(SUITES,[250,300,320,550])),
        uses_state=False,uses_future=False,teacher_loaded=False,
        noise_seed='(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index',
        first_success_video=False,rollout_sha256=file_sha256(ROOT/'examples/LIBERO/eval_files/libero_eval_core.py'),
        checkpoint_sha256=checksum,config_sha256=digest_json(meta['config']),step=2000,extra=meta['extra'])
    if watch.read(output/'protocol.json')!=expected:
        raise ValueError('Existing evaluation checkpoint/protocol differs; preserve results')
    return expected

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
    def __init__(self,gpus,checkpoint,output,checksum,stop,gap):
        self.checkpoint=checkpoint;self.output=output;self.checksum=checksum;self.stop_event=stop
        self.gap=gap;self.load_lock=threading.Lock();self.last_loaded=0
        self.entries={gpu:dict(server=None,url=None,phase='queued',health=None) for gpu in gpus}

    def ensure(self,gpu):
        entry=self.entries[gpu]
        server=entry['server']
        if server is not None and server.poll() is None and entry['url']:
            return entry['url']
        entry['phase']='waiting_for_load'
        with self.load_lock:
            delay=max(0,self.last_loaded+self.gap-time.time())
            if self.stop_event.wait(delay):raise RuntimeError('Evaluation stopping')
            if self.stop_event.is_set():raise RuntimeError('Evaluation stopping')
            stop_child(server)
            folder=self.output/'.parallel_workers'/f'gpu{gpu}';folder.mkdir(parents=True,exist_ok=True)
            ready=folder/'server_ready.json'
            command=[sys.executable,'-B','-m','ablation.aba1.libero_common.server',
                     '--variant','A1','--device','cuda:0','--ready',str(ready)]
            with (folder/'server.log').open('a') as log:
                server=subprocess.Popen(command,cwd=ROOT,env=env_for(gpu),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry.update(server=server,url=None,phase='loading_model',load_started_at=time.time(),health=None)
            if self.stop_event.is_set():
                stop_child(server);raise RuntimeError('Evaluation stopping')
            deadline=time.monotonic()+900
            while True:
                if self.stop_event.is_set():raise RuntimeError('Evaluation stopping')
                if server.poll() is not None:raise RuntimeError(f'GPU {gpu} inference server exited {server.returncode}')
                info=watch.read(ready)
                if info.get('pid')==server.pid and info.get('port'):break
                if time.monotonic()>deadline:raise TimeoutError(f'GPU {gpu} server startup')
                self.stop_event.wait(1)
            url=f"http://127.0.0.1:{info['port']}"
            loaded=rpc(url,'/load',dict(checkpoint=str(self.checkpoint)))
            expected=dict(variant='A1',step=2000,model_sha256=self.checksum,physical_gpu=str(gpu),
                local_slots=16,goal_slots=0,interaction=False,teacher_loaded=False,uses_future=False,uses_state=False)
            if any(loaded.get(k)!=v for k,v in expected.items()):
                raise ValueError(f'GPU {gpu} loaded a different model: {loaded}')
            if self.stop_event.is_set():raise RuntimeError('Evaluation stopping')
            self.last_loaded=time.time()
            entry.update(url=url,phase='ready',load_finished_at=self.last_loaded,health=loaded)
            atomic_json(folder/'loaded_policy.json',dict(loaded,load_started_at=entry['load_started_at'],
                load_finished_at=self.last_loaded))
            print(json.dumps(dict(event='model_loaded',gpu=gpu,pid=server.pid,time=time.time())),flush=True)
            return url

    def reset(self,gpu):
        entry=self.entries[gpu];stop_child(entry['server'])
        entry.update(server=None,url=None,phase='retry_pending',health=None)

    def stop(self):
        for entry in self.entries.values():stop_child(entry['server'])

class Worker:
    def __init__(self,gpu,pool,stop):
        self.gpu=gpu;self.pool=pool;self.stop_event=stop;self.child=None
        self.state=dict(gpu=gpu,phase='queued',tasks_finished=0)

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
                        self.state=dict(gpu=self.gpu,phase='preparing',suite=suite,task_id=task,
                                        attempt=attempt,tasks_finished=finished)
                        try:
                            existing=len(episode_rows(out,task,50))
                            if existing==50:complete=True;break
                            url=self.pool.ensure(self.gpu)
                            if self.stop_event.is_set():break
                            cmd=[SIM_PYTHON,'-B','-m','goal_local.eval_task','--checkpoint',str(self.pool.checkpoint),
                                '--url',url,'--suite',suite,'--task-id',str(task),'--episodes','50','--output',str(out)]
                            with (out/'run.log').open('a') as log:
                                log.write(f'\nParallel A1 GPU {self.gpu}, attempt {attempt}, preserved {existing} episodes, time {time.time()}\n')
                                log.flush()
                                self.child=subprocess.Popen(cmd,cwd=ROOT,env=env_for(self.gpu),
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            self.state=dict(self.state,phase='evaluating',task_pid=self.child.pid,
                                task_started_at=time.time(),episodes_at_start=existing)
                            if self.stop_event.is_set():stop_child(self.child)
                            deadline=time.monotonic()+14400
                            while self.child.poll() is None:
                                if self.stop_event.is_set() or self.pool.entries[self.gpu]['server'].poll() is not None:
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
                            stop_child(self.child);self.child=None;self.pool.reset(self.gpu)
                            print(json.dumps(dict(event='task_retry',gpu=self.gpu,suite=suite,task=task,attempt=attempt,error=last_error)),flush=True)
                            if not self.stop_event.is_set() and attempt<3:self.stop_event.wait(10)
                    if complete:finished+=1
                    elif not self.stop_event.is_set():failures.append(dict(suite=suite,task_id=task,error=last_error))
                finally:jobs.task_done()
            self.state=dict(gpu=self.gpu,phase='idle',tasks_finished=finished,failures=failures)
            return failures
        finally:stop_child(self.child)

    def snapshot(self):
        state=dict(self.state);entry=self.pool.entries[self.gpu];server=entry['server']
        state.update(server_pid=server.pid if server else None,server_alive=server is not None and server.poll() is None)
        if state['phase'] in ('queued','preparing'):state['phase']=entry['phase']
        if entry['url'] and state['server_alive']:
            try:entry['health']=rpc(entry['url'],'/health',timeout=2)
            except Exception:pass
        if entry.get('health'):state['calls']=entry['health'].get('calls',0)
        for key in ('load_started_at','load_finished_at'):
            if key in entry:state[key]=entry[key]
        return state

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3]);parser.add_argument('--load-gap',type=float,default=10)
    args=parser.parse_args();run=args.run.resolve();gpus=args.gpus
    if not run.is_relative_to(WORK/'runs') or run==WORK/'runs':parser.error('Expected A1 run')
    if len(set(gpus))!=len(gpus) or not gpus or any(g<0 for g in gpus) or args.load_gap<0:parser.error('Invalid GPUs/gap')
    checkpoint=watch.ready_checkpoint(run)
    if checkpoint is None:raise ValueError('Training has not completed')
    output=WORK/'evaluation'/(run.name+'_full40x50_step2000');watch_dir=WORK/'auto_eval'/run.name
    watch_dir.mkdir(exist_ok=True,parents=True)
    with (watch_dir/'watch.lock').open('a') as watch_lock,(output/'.evaluation.lock').open('a') as eval_lock:
        fcntl.flock(watch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(eval_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=threading.Event();pool=None;workers=[];executor=None
        def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
        def publish(phase,**extra):
            progress=watch.progress(output);progress.pop('current_task',None)
            snapshots=[worker.snapshot() for worker in workers]
            for item in snapshots:
                if item.get('suite') is not None:
                    item['episodes_completed']=len(watch_rows(task_path(output,item['suite'],item['task_id'])))
            state=dict(phase=phase,watcher_pid=os.getpid(),evaluation_pid=os.getpid(),run=str(run),output=str(output),
                physical_gpus=gpus,checkpoint_step=2000,workers=snapshots,**progress,**extra,updated_at=time.time())
            atomic_json(watch_dir/'status.json',state)
            atomic_json(output/'status.json',dict(state,phase='completed' if phase=='completed' else 'evaluating_parallel' if phase=='evaluating' else phase))
            atomic_json(watch_dir/'parallel_status.json',state)
            return state
        try:
            publish('verifying_step2000')
            checksum=watch.verify_checkpoint(run,checkpoint,watch.read(WORK/'config.json'))
            validate_protocol(output,checkpoint,checksum)
            pending=pending_tasks(output);before=watch.progress(output)
            manifest=dict(run=str(run),checkpoint=str(checkpoint),model_sha256=checksum,physical_gpus=gpus,
                one_model_server_per_gpu=True,one_task_worker_per_gpu=True,load_gap_seconds=args.load_gap,
                tasks=40,episodes_per_task=50,expected_episodes=2000,preserved_episodes=before['episodes_completed'],
                pending_tasks=pending,started_at=time.time(),pid=os.getpid(),protocol_unchanged=True)
            atomic_json(watch_dir/'parallel_plan.json',manifest)
            atomic_json(watch_dir/'verified_checkpoint.json',dict(path=str(checkpoint),step=2000,sha256=checksum))
            jobs=queue.Queue()
            for task in pending:jobs.put(task)
            pool=ServerPool(gpus,checkpoint,output,checksum,stop,args.load_gap)
            workers=[Worker(gpu,pool,stop) for gpu in gpus]
            executor=ThreadPoolExecutor(max_workers=len(gpus))
            futures=[executor.submit(worker.run,jobs) for worker in workers]
            last_print=0
            while not all(f.done() for f in futures):
                state=publish('evaluating')
                if time.time()-last_print>=30:
                    print(json.dumps(state,ensure_ascii=False),flush=True);last_print=time.time()
                time.sleep(5)
            failures=[x for future in futures for x in future.result()]
            summary=summarize(output,50,checksum)
            if failures or not watch.complete(output,checksum):raise RuntimeError(f'Incomplete evaluation: {failures}')
            publish('completed');print(json.dumps(summary,ensure_ascii=False),flush=True)
        except BaseException:
            publish('watcher_stopped_or_failed',error=traceback.format_exc())
            raise
        finally:
            stop.set()
            for worker in workers:stop_child(worker.child)
            if pool:pool.stop()
            if executor:executor.shutdown(wait=True,cancel_futures=True)
            if pool:pool.stop()

def watch_rows(path):
    """Read live append-only progress; exact validation occurs before/after each task."""
    result={}
    try:
        for line in (Path(path)/'episodes.checkpoint.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if not row.get('error'):result[row['episode_idx']]=row
    except OSError:pass
    return result

if __name__=='__main__':main()
