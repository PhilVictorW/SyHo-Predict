"""Persistent GPU policy servers pulling whole LIBERO tasks from a shared queue."""
from __future__ import annotations
import argparse, datetime, fcntl, json, os, queue, signal, socket, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ['libero_10', 'libero_object', 'libero_goal', 'libero_spatial']


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def read_rows(path):
    rows={}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                continue
            key=(int(row['task_id']),int(row['episode_idx']))
            if key in rows and rows[key]!=row:
                raise RuntimeError(f'Conflicting duplicate record {path}: {key}')
            rows[key]=row
    return list(rows.values())


def task_complete(path, task_id):
    rows=read_rows(path/'episodes.checkpoint.jsonl')
    keys={(int(r['task_id']),int(r['episode_idx'])) for r in rows}
    return keys=={(task_id,e) for e in range(50)} and not any(r.get('error') for r in rows)


class Runner:
    def __init__(self,run,gpus):
        self.run=run
        self.gpus=gpus
        self.lock=threading.RLock()
        self.stop=threading.Event()
        self.processes={}
        self.jobs=queue.Queue()
        self.state={'mode':'dynamic_task_queue','updated_at':time.time(),'gpus':{},'tasks':{},'status':'running'}
        prior_path=run/'queue/state.json'
        prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
        jobs=[]
        for task_id in range(10):
            for suite in SUITES:
                key=f'{suite}__task{task_id:02d}'
                path=run/'tasks'/key
                path.mkdir(parents=True, exist_ok=True)
                rows=read_rows(path/'episodes.checkpoint.jsonl')
                done=task_complete(path,task_id)
                task={'suite':suite,'task_id':task_id,'status':'done' if done else 'pending',
                      'gpu':prior.get('tasks',{}).get(key,{}).get('gpu'),'attempts':0,
                      'inherited_episodes':len(rows),'output':str(path)}
                self.state['tasks'][key]=task
                if not done: jobs.append((0 if rows else 1,task_id,SUITES.index(suite),key))
        for _,_,_,key in sorted(jobs):self.jobs.put(key)
        for gpu in gpus:
            self.state['gpus'][str(gpu)]={'status':'starting','task':None,'port':22940+gpu,'server_pid':None,'eval_pid':None}
        self.save()

    def save(self):
        with self.lock:
            self.state['updated_at']=time.time()
            atomic_json(self.run/'queue/state.json',self.state)

    def launch(self,args,env,log):
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with self.lock:self.processes[proc.pid]=proc
        return proc

    def terminate(self,proc):
        if proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
        with self.lock:self.processes.pop(proc.pid,None)

    def shutdown(self):
        self.stop.set()
        with self.lock:processes=list(self.processes.values())
        for proc in processes:self.terminate(proc)

    def server(self,gpu,env):
        info=self.state['gpus'][str(gpu)]
        port=info['port']
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',port))
            except OSError:raise RuntimeError(f'Port {port} already in use; refusing to connect to an unrelated server')
        path=self.run/'queue'/f'gpu{gpu}'
        path.mkdir(parents=True,exist_ok=True)
        log=(path/'server.log').open('a',buffering=1)
        proc=self.launch([os.environ['STAR_VLA_PYTHON'],'deployment/model_server/server_policy.py',
            '--ckpt_path',os.environ['CKPT_PATH'],'--port',str(port),'--use_bf16','--disable_lawm','--remove_query_tokens'],env,log)
        log.close()
        with self.lock:
            info.update(status='loading',server_pid=proc.pid)
            self.save()
        deadline=time.monotonic()+1200
        while not self.stop.is_set() and time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError(f'GPU {gpu} policy server exited: {proc.returncode}')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):pass
                from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
                client=WebsocketClientPolicy('127.0.0.1',port)
                try:meta=client.get_server_metadata()
                finally:client.close()
                if not meta.get('inference_remove_query_tokens'):
                    raise RuntimeError('Policy server does not confirm query ablation')
                if not meta.get('inference_disable_lawm'):
                    raise RuntimeError('Policy server does not confirm LaWM ablation')
                if Path(meta['ckpt_path']).resolve()!=Path(os.environ['CKPT_PATH']).resolve():
                    raise RuntimeError('Policy checkpoint mismatch')
                with self.lock:info['status']='idle';self.save()
                return proc
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                self.stop.wait(1)
        raise RuntimeError(f'GPU {gpu} server startup timed out or stopped')

    def worker(self,gpu):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),GPU_ID=str(gpu),EVAL_GPU_ID=str(gpu))
        info=self.state['gpus'][str(gpu)]
        server=None
        try:
            server=self.server(gpu,env)
            while not self.stop.is_set():
                try:key=self.jobs.get_nowait()
                except queue.Empty:break
                task=self.state['tasks'][key]
                path=Path(task['output'])
                with self.lock:
                    task.update(status='running',gpu=gpu,started_at=time.time(),attempts=task['attempts']+1)
                    info.update(status='running',task=key)
                    self.save()
                args=[os.environ['LIBERO_PYTHON'],'examples/LIBERO/eval_files/eval_libero.py',
                    '--args.pretrained-path',os.environ['CKPT_PATH'],'--args.host','127.0.0.1',
                    '--args.port',str(info['port']),'--args.task-suite-name',task['suite'],
                    '--args.task-id',str(task['task_id']),'--args.num-trials-per-task','50',
                    '--args.num-workers','1','--args.seed','0','--args.video-out-path',str(path),
                    '--args.resume-incomplete',
                    '--args.worker-result-timeout-sec','1200']
                with (path/'eval.log').open('a',buffering=1) as log:
                    log.write(f'\nQUEUE ATTEMPT {task["attempts"]}, GPU {gpu}, {datetime.datetime.now().isoformat()}\n')
                    proc=self.launch(args,env,log)
                    with self.lock:info['eval_pid']=proc.pid;self.save()
                    deadline=time.monotonic()+7200
                    while proc.poll() is None and not self.stop.is_set() and time.monotonic()<deadline:
                        if server.poll() is not None:break
                        self.stop.wait(2)
                    if proc.poll() is None:self.terminate(proc)
                    rc=proc.wait()
                with self.lock:self.processes.pop(proc.pid,None)
                done=rc==0 and task_complete(path,task['task_id'])
                with self.lock:
                    task.update(status='done' if done else 'interrupted' if self.stop.is_set() else 'retry' if task['attempts']<3 else 'failed',
                        exit_code=rc,finished_at=time.time())
                    info.update(task=None,eval_pid=None,status='idle')
                    atomic_json(path/'queue_result.json',task)
                    self.save()
                print(f'GPU {gpu} {key}: {task["status"]}, exit={rc}',flush=True)
                if not done and not self.stop.is_set() and task['attempts']<3:self.jobs.put(key)
                self.jobs.task_done()
                if server.poll() is not None:
                    self.terminate(server)
                    server=self.server(gpu,env)
        except Exception as exc:
            with self.lock:
                info.update(status='error',error=repr(exc))
                current=info.get('task')
                if current and self.state['tasks'][current]['status']=='running':
                    self.state['tasks'][current].update(status='pending',error=repr(exc))
                    self.jobs.put(current)
                self.save()
            print(f'GPU {gpu} worker error: {exc!r}',flush=True)
        finally:
            if server is not None:self.terminate(server)
            with self.lock:
                if info['status']!='error':info['status']='stopped' if self.stop.is_set() else 'finished'
                self.save()

    def run_all(self):
        threads=[threading.Thread(target=self.worker,args=(gpu,),name=f'gpu{gpu}') for gpu in self.gpus]
        for thread in threads:thread.start()
        last_summary=0
        while any(thread.is_alive() for thread in threads):
            self.save()
            if time.monotonic()-last_summary>=20:
                subprocess.run([sys.executable,str(ROOT/'runtime/summarize_ablation.py'),str(self.run)],stdout=subprocess.DEVNULL,check=True)
                last_summary=time.monotonic()
            self.stop.wait(3) if not self.stop.is_set() else time.sleep(1)
        for thread in threads:thread.join()
        self.shutdown()
        complete=all(t['status']=='done' for t in self.state['tasks'].values())
        with self.lock:self.state['status']='complete' if complete else 'incomplete';self.save()
        subprocess.run([sys.executable,str(ROOT/'runtime/summarize_ablation.py'),str(self.run)],check=True)
        atomic_json(self.run/'queue/completion.json',{'complete':complete,'finished_at':time.time()})
        return 0 if complete else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3,6,7])
    args=parser.parse_args()
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or any(g not in [0,1,2,3,6,7] for g in args.gpus):
        raise ValueError('aba2 GPUs must be unique and drawn from 0,1,2,3,6,7; GPUs 4 and 5 are excluded.')
    (args.run_dir/'queue').mkdir(parents=True, exist_ok=True)
    lock_file=(args.run_dir/'queue/runner.lock').open('w')
    fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=Runner(args.run_dir,args.gpus)
    signal.signal(signal.SIGTERM,lambda *_:runner.shutdown())
    signal.signal(signal.SIGINT,lambda *_:runner.shutdown())
    try:return runner.run_all()
    finally:runner.shutdown()

if __name__=='__main__':
    sys.exit(main())
