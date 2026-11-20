#!/usr/bin/env python3
"""Parallel, resumable A0 LIBERO evaluation with four workers on GPU 1."""
import argparse
import fcntl
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ABA, ROOT, WORK, atomic_json, digest_json, file_sha256, load_config
from .evaluation import SUITES, episode_rows, summarize
from goal_local.eval_common import BASELINE, SIM_PYTHON, model_metadata
from goal_local.eval_client import rpc


class ServerPool:
    """One model server per physical GPU; many rollout workers may share it."""
    def __init__(self, gpus, checkpoint, output, stop_event):
        self.gpus=sorted(set(gpus))
        self.checkpoint=Path(checkpoint).resolve()
        self.output=Path(output).resolve()
        self.stop_event=stop_event
        self.entries={gpu:dict(server=None,url=None,log=None,lock=threading.Lock()) for gpu in self.gpus}
        self.expected_sha=file_sha256(self.checkpoint/'model.pt')

    def env_for(self,gpu):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                   LIBERO_HOME=str(BASELINE/'runtime/LIBERO'),
                   LIBERO_CONFIG_PATH=str(BASELINE/'runtime/libero_config'),
                   MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                   TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH']=os.pathsep.join([str(WORK),str(ROOT),env['LIBERO_HOME'],env.get('PYTHONPATH','')])
        return env

    @staticmethod
    def stop_process(process):
        if process is None or process.poll() is not None:return
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()

    def ensure(self,gpu):
        entry=self.entries[gpu]
        with entry['lock']:
            server=entry['server']
            if server is not None and server.poll() is None and entry['url']:
                return entry['url']
            self.stop_process(server)
            if entry['log']:entry['log'].close()
            worker_dir=self.output/'.parallel_workers'/f'gpu{gpu}'
            worker_dir.mkdir(parents=True,exist_ok=True)
            ready=worker_dir/'server_ready.json';ready.unlink(missing_ok=True)
            log=(worker_dir/'server.log').open('a')
            env=self.env_for(gpu)
            command=[sys.executable,'-m','a0_single_latent.server','--variant','A0',
                     '--device','cuda:0','--ready',str(ready)]
            server=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry.update(server=server,url=None,log=log)
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                if self.stop_event.is_set():raise RuntimeError('evaluation stopping')
                if server.poll() is not None:raise RuntimeError(f'GPU {gpu} server exited: {server.returncode}')
                try:info=json.loads(ready.read_text())
                except (OSError,ValueError):info={}
                if info.get('pid')==server.pid and info.get('port'):
                    url=f"http://127.0.0.1:{info['port']}"
                    rpc(url,'/health')
                    loaded=rpc(url,'/load',dict(checkpoint=str(self.checkpoint)))
                    if loaded.get('step')!=500 or loaded.get('model_sha256')!=self.expected_sha:
                        raise RuntimeError(f'GPU {gpu} loaded an unexpected checkpoint')
                    if loaded.get('uses_dino') or loaded.get('uses_future') or loaded.get('uses_state'):
                        raise RuntimeError(f'GPU {gpu} loaded an unexpected A0 policy')
                    entry['url']=url
                    return url
                time.sleep(1)
            raise TimeoutError(f'GPU {gpu} server startup timeout')

    def stop(self):
        for entry in self.entries.values():
            self.stop_process(entry['server'])
            if entry['log']:entry['log'].close()
            entry.update(server=None,url=None,log=None)


class Worker:
    def __init__(self,gpu,worker_id,checkpoint,output,count,stop_event,pool):
        self.gpu=int(gpu);self.worker_id=worker_id;self.checkpoint=Path(checkpoint).resolve()
        self.output=Path(output).resolve();self.count=count;self.stop_event=stop_event;self.pool=pool
        self.child=None;self.url=None;self.state=dict(worker=worker_id,gpu=self.gpu,phase='starting')
        self.env=pool.env_for(self.gpu)

    @staticmethod
    def stop_process(process):
        if process is None or process.poll() is not None:return
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()

    def task_output(self,suite,task):
        return self.output/'tasks'/f'{suite}__task{task:02d}'

    def run(self,jobs):
        failures=[]
        while not self.stop_event.is_set():
            try:suite,task=jobs.get_nowait()
            except queue.Empty:break
            out=self.task_output(suite,task);out.mkdir(parents=True,exist_ok=True);complete=False
            try:
                for attempt in range(1,4):
                    if self.stop_event.is_set():break
                    existing=len(episode_rows(out,task,self.count))
                    if existing==self.count:complete=True;break
                    self.url=self.pool.ensure(self.gpu)
                    self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='evaluating',suite=suite,task_id=task,attempt=attempt)
                    with (out/'run.log').open('a') as log:
                        log.write(f'\n--- GPU {self.gpu}, worker {self.worker_id}, attempt {attempt}, time {time.time()} ---\n');log.flush()
                        command=[SIM_PYTHON,'-m','goal_local.eval_task','--checkpoint',str(self.checkpoint),
                                 '--url',self.url,'--suite',suite,'--task-id',str(task),
                                 '--episodes',str(self.count),'--output',str(out)]
                        self.child=subprocess.Popen(command,cwd=WORK,env=self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:code=self.child.wait(timeout=14400)
                        except subprocess.TimeoutExpired:self.stop_process(self.child);code=-9
                        self.child=None
                    complete=(code==0 and len(episode_rows(out,task,self.count))==self.count)
                    if complete:break
                    self.url=None
                if not complete:failures.append((suite,task,'incomplete'))
            except Exception as error:
                failures.append((suite,task,repr(error)))
            finally:
                jobs.task_done()
        self.state=dict(worker=self.worker_id,gpu=self.gpu,phase='idle',failures=failures)
        return failures

    def stop(self):
        self.stop_process(self.child);self.child=None


def protocol(variant,checkpoint,kind,count,config,gpu_list):
    path=Path(checkpoint).resolve();marker=json.loads((path/'checkpoint_complete.json').read_text())
    meta=model_metadata(path/'model.pt')
    if meta['format']!='a0_single_latent_no_dino_v1' or meta['mode']!='student' or meta['step']!=500:
        raise ValueError('Expected the completed A0 student checkpoint')
    return dict(variant=variant,checkpoint=str(path),protocol=kind,episodes_per_task=count,
        expected_episodes=40*count,suites=list(SUITES),task_ids=list(range(10)),
        initial_state_indices=list(range(count)),simulator_seed=0,num_steps_wait=10,
        action_chunk=8,flow_steps=10,max_steps=dict(zip(SUITES,[250,300,320,550])),
        uses_state=False,uses_future=False,teacher_loaded=False,uses_dino=False,latent_slots=16,
        noise_seed='(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index',
        first_success_video=False,rollout_sha256=file_sha256(ROOT/'examples/LIBERO/eval_files/libero_eval_core.py'),
        checkpoint_sha256=marker['model_sha256'],config_sha256=digest_json(meta['config']),
        step=meta['step'],extra=meta.get('extra',{}),parallel_gpus=gpu_list,shared_model_server=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--protocol',choices=['quick','full'],required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--gpus',default='1,1,1,1');parser.add_argument('--variant',default='A0',choices=['A0','A1','A2','A3'])
    args=parser.parse_args();cfg=load_config(WORK/'config.json',args.variant)
    checkpoint=Path(args.checkpoint).resolve();output=Path(args.output).resolve()
    if not checkpoint.is_relative_to((ABA/args.variant/'libero/runs').resolve()):raise ValueError('Checkpoint must belong to this ablation run directory')
    if not (checkpoint/'checkpoint_complete.json').is_file():raise ValueError('Incomplete checkpoint')
    count=10 if args.protocol=='quick' else 50;output.mkdir(parents=True,exist_ok=True)
    gpu_list=[int(x) for x in args.gpus.split(',')]
    if gpu_list!=[1,1,1,1]:raise ValueError('A0 recovery requires four workers on GPU 1')
    manifest=protocol(args.variant,checkpoint,args.protocol,count,cfg,gpu_list);old=output/'protocol.json'
    if old.exists():
        prior=json.loads(old.read_text())
        for key in ('variant','checkpoint','protocol','checkpoint_sha256','step','uses_dino','uses_future','uses_state'):
            if prior.get(key)!=manifest.get(key):raise ValueError(f'Existing output belongs to another {key}')
        prior.update(parallel_gpus=gpu_list,shared_model_server=True);atomic_json(old,prior)
    else:atomic_json(old,manifest)
    lock=(output/'.evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop_event=threading.Event();jobs=queue.Queue()
    for suite in SUITES:
        for task in range(10):
            out=output/'tasks'/f'{suite}__task{task:02d}'
            if len(episode_rows(out,task,count))<count:jobs.put((suite,task))
    atomic_json(output/'status.json',dict(phase='evaluating_parallel',protocol=args.protocol,
        workers=[dict(worker=f'w{i+1}',gpu=1) for i in range(4)],tasks_remaining=jobs.qsize(),time=time.time()))
    if jobs.qsize()==0:
        summary=summarize(output,count,manifest['checkpoint_sha256'])
        if summary['status']!='complete':raise RuntimeError(f'Existing evaluation is incomplete: {summary["episodes_completed"]}/{summary["expected_episodes"]}')
        atomic_json(output/'status.json',dict(phase='completed',protocol=args.protocol,workers=[dict(worker=f'w{i+1}',gpu=1) for i in range(4)],summary=summary,time=time.time()));print(json.dumps(summary,indent=2));return
    pool=ServerPool(gpu_list,checkpoint,output,stop_event)
    workers=[Worker(1,f'w{i+1}',checkpoint,output,count,stop_event,pool) for i in range(4)]
    def stop_signal(*_):
        stop_event.set()
        for worker in workers:worker.stop()
        pool.stop();raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(worker.run,jobs) for worker in workers]
            while not all(f.done() for f in futures):
                atomic_json(output/'status.json',dict(phase='evaluating_parallel',protocol=args.protocol,workers=[worker.state for worker in workers],tasks_remaining=jobs.qsize(),time=time.time()));time.sleep(5)
            failures=[f.result() for f in futures]
        flat=[item for group in failures for item in group];summary=summarize(output,count,manifest['checkpoint_sha256'])
        if flat or summary['status']!='complete':
            atomic_json(output/'status.json',dict(phase='interrupted_or_failed',protocol=args.protocol,failures=flat,summary=summary,time=time.time()))
            raise RuntimeError(f'parallel {args.protocol} incomplete: {summary["episodes_completed"]}/{summary["expected_episodes"]}')
        atomic_json(output/'status.json',dict(phase='completed',protocol=args.protocol,workers=[dict(worker=f'w{i+1}',gpu=1) for i in range(4)],summary=summary,time=time.time()));print(json.dumps(summary,indent=2))
    finally:
        stop_event.set()
        for worker in workers:worker.stop()
        pool.stop()

if __name__=='__main__':main()
