"""Independent four-worker evaluation queue. Never touches baseline processes."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time

from student_eval_common import *
sys.path.insert(0,str(WORK.parent))
from goal_local.eval_client import rpc
from student_action_protocol import ensure_checkpoint_protocol

def environment(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
        PYTHONPATH=':'.join(map(str,[WORK/'runtime',BASE,WORK.parent,BENCH,BASE/'libero-plus/runtime'])))
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
    return env

def launch(cmd,gpu,log):
    with Path(log).open('a') as stream:return subprocess.Popen(cmd,env=environment(gpu),cwd=WORK,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)

def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def evaluate(step,checkpoint,*,spec=None,status_root=None,ports=None,gpus=None):
    action_protocol=ensure_checkpoint_protocol(checkpoint)
    spec=spec or evaluation_spec(step)
    folder=spec['folder'];folder.mkdir(parents=True,exist_ok=True)
    status_root=status_root or ROOT
    worker_ports=ports or PORTS
    worker_gpus=gpus or GPUS
    assert len(worker_ports)==len(worker_gpus)>=1 and len(set(worker_ports))==len(worker_ports)
    manifest=json.loads(spec['manifest'].read_text())
    tasks=manifest['tasks']
    assert len(tasks)==len({(t['suite'],t['task_id']) for t in tasks})
    if spec['scope']=='full_libero_plus':
        assert len(tasks)==10030
    else:
        full=json.loads((ROOT/'full_manifest.json').read_text())
        lookup={(t['suite'],t['task_id']):t for t in full['tasks']}
        assert step in (500,1000,1500,2000,2500,3000,3500,4000) and manifest['sampling']['basis_step']==500
        assert len(tasks)==manifest['sampling']['selected_task_count']
        assert all(lookup[(t['suite'],t['task_id'])]==t for t in tasks)
        assert {(t['suite'],t['category']) for t in tasks}=={(s,c) for s in SUITES for c in CATEGORIES}
    protocol=dict(step=step,smoke=False,scope=spec['scope'],checkpoint=str(checkpoint.resolve()),tasks=tasks,seed=0,episodes_per_task=1,
        action_chunk=8,flow_steps=10,uses_state=False,uses_future=False,save_videos=False,task_manifest=str(spec['manifest']),
        classification_sha256=manifest['source_sha256'],action_protocol=action_protocol)
    if spec['scope']!='full_libero_plus':protocol['sampling']=manifest['sampling']
    if (folder/'protocol.json').exists():assert json.loads((folder/'protocol.json').read_text())==protocol
    else:atomic(folder/'protocol.json',protocol)
    (folder/'episodes').mkdir(exist_ok=True)
    result=collect(folder)
    if result['complete']:return result
    atomic(checkpoint/'eval_snapshot_complete.json',dict(step=step,mode='student',checkpoint_complete=True))
    db=sqlite3.connect(folder/'queue.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY,payload TEXT,state TEXT,worker INTEGER,attempts INTEGER DEFAULT 0,error TEXT)')
    for i,task in enumerate(tasks):
        done=(folder/'episodes'/f"{task['suite']}_{task['task_id']:05d}.json").exists()
        db.execute('INSERT OR IGNORE INTO tasks (id,payload,state) VALUES (?,?,?)',(i,json.dumps(task),'done' if done else 'pending'))
        if done:db.execute("UPDATE tasks SET state='done' WHERE id=?",(i,))
    db.execute("UPDATE tasks SET state='pending' WHERE state='running'");db.commit();db.close()

    def worker(i):
        gpu=worker_gpus[i];port=worker_ports[i];url=f'http://127.0.0.1:{port}';server=None;client=None
        try:
            # Never adopt/kill an existing listener; these ports belong only to this queue.
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',port))
            server=launch([PY,str(WORK/'runtime/student_server.py'),'--port',str(port),'--ready',str(folder/f'server{i}.json')],gpu,folder/f'server{i}.log')
            deadline=time.monotonic()+1200
            while True:
                if server.poll() is not None:raise RuntimeError(f'Student server {i} exited')
                try:rpc(url,'/health',timeout=2);break
                except Exception:
                    if time.monotonic()>deadline:raise TimeoutError('Student server startup')
                    time.sleep(2)
            info=rpc(url,'/load',dict(checkpoint=str(checkpoint)),timeout=1800)
            assert info['step']==step and not info['uses_future'] and not info['teacher_loaded']
            for attempt in range(3):
                client=launch([SIMPY,str(WORK/'runtime/student_rollout.py'),'--folder',str(folder),'--worker',str(i),'--url',url],gpu,folder/f'worker{i}.log')
                last=time.monotonic();previous=0
                while client.poll() is None:
                    if server.poll() is not None:raise RuntimeError(f'Student server {i} died')
                    count=len(list((folder/'episodes').glob('*.json')))
                    if count>previous:previous=count;last=time.monotonic()
                    if time.monotonic()-last>3600:terminate(client);break
                    time.sleep(3)
                db=sqlite3.connect(folder/'queue.sqlite',timeout=60)
                db.execute("UPDATE tasks SET state='pending' WHERE state='running' AND worker=?",(i,));db.commit();db.close()
                if client.returncode==0:break
                if attempt==2:raise RuntimeError(f'Worker {i} failed repeatedly, see log')
        finally:
            if client is not None:terminate(client)
            if server is not None:terminate(server)

    atomic(status_root/'status.json',dict(phase='evaluating',step=step,smoke=False,scope=spec['scope'],expected=len(tasks),folder=str(folder),updated=time.time()))
    with ThreadPoolExecutor(max_workers=len(worker_ports)) as pool:
        jobs=[pool.submit(worker,i) for i in range(len(worker_ports))]
        while not all(j.done() for j in jobs):collect(folder);time.sleep(5)
        for job in jobs:job.result()
    result=collect(folder)
    if not result['complete']:raise RuntimeError(f'Incomplete evaluation: {result["episodes"]}/{result["expected"]}; see errors/')
    return result

def main():
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(ROOT/'queue_owner.json',dict(pid=os.getpid(),started=time.time(),physical_gpus=GPUS,ports=PORTS))
    while True:
        candidates=[]
        for marker in TRAIN.glob('step_*/checkpoint_complete.json'):
            step=int(marker.parent.name.split('_')[1])
            if 500<=step<=10000 and step%500==0:candidates.append((step,marker.parent))
        outstanding=[]
        for step,checkpoint in sorted(candidates):
            folder=eval_folder(step)
            if (folder/'results.json').exists():
                previous=json.loads((folder/'results.json').read_text())
                if previous.get('action_protocol')!=ACTION_PROTOCOL:raise ValueError('Incompatible completed evaluation protocol')
                if previous['complete']:continue
            outstanding.append((step,checkpoint))
        if not outstanding:
            atomic(ROOT/'status.json',dict(phase='waiting_for_checkpoint',scope='full_libero_plus',expected=10030,updated=time.time(),completed_steps=[s for s,p in candidates],interval=500))
            if len(candidates)==20:break
            time.sleep(10);continue
        step,checkpoint=outstanding[0]
        try:
            evaluate(step,checkpoint)
            summaries=[]
            for path in sorted(ROOT.glob('step_*/results.json')):summaries.append(json.loads(path.read_text()))
            atomic(ROOT/'summary.json',summaries)
        except BaseException as exc:
            atomic(ROOT/'status.json',dict(phase='failed',step=step,error=repr(exc),updated=time.time()))
            raise
    atomic(ROOT/'status.json',dict(phase='completed',updated=time.time(),last_step=10000))

if __name__=='__main__':main()
