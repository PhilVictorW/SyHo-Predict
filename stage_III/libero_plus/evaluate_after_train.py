"""Wait for committed round-one completion, then run 8380 isolated evaluations."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error

from .eval_common import (ROOT, STAGE3, SUITES, CATEGORIES, GPUS, PORTS, atomic, read, digest,
                          connect, initialize_queue, result_summary, requeue_errors, training_complete)


def log(message):
    print(f'[Stage III衔接][{time.strftime("%m-%d %H:%M:%S")}] {message}',flush=True)


def process_args(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    except OSError:
        return []


def terminate_owned(pid, required):
    args=process_args(pid)
    if not args or not all(str(x) in args for x in required):return
    log(f'关闭本流程所属进程 PID={pid}')
    os.kill(pid,signal.SIGTERM)
    for _ in range(15):
        if not process_args(pid):return
        time.sleep(1)
    if all(str(x) in process_args(pid) for x in required):os.kill(pid,signal.SIGKILL)


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:raise RuntimeError(f'process {proc.pid} cannot exit after SIGKILL')


def prepare(folder, train_run):
    if not folder.is_relative_to((STAGE3/'evaluation').resolve()):
        raise ValueError('evaluation output must stay inside isolated Stage III evaluation directory')
    source=ROOT/'libero-plus/evaluation/stage_II_steps2500_4000_parallel_20261008/manifest_same_as_steps1000_1500_2000.json'
    manifest=json.loads(source.read_text());tasks=manifest['tasks']
    if (len(tasks)!=8380 or len({(t['suite'],t['task_id']) for t in tasks})!=8380
            or {(t['suite'],t['category']) for t in tasks}!={(s,c) for s in SUITES for c in CATEGORIES}):
        raise ValueError('evaluation manifest must cover the original 8380 tasks and all 28 cells')
    if read(train_run/'manifest.json')['tasks'] != tasks:
        raise ValueError('training and baseline evaluation task lists differ')
    baseline=read(STAGE3/'baseline_snapshot.json')
    protocol=dict(format='stage3_round1_full8380_eval_v1',train_run=str(train_run),tasks=tasks,
        base_student=baseline['base_student'],base_student_sha256=baseline['base_student_model_sha256'],
        teacher_source=baseline['teacher_source'],actor_checkpoint=str(folder/'model/actor_round1_update0262.pt'),
        actor_iteration=262,gate_mode='beta_mean',gpus=GPUS,ports=PORTS,seed=0,initial_state_index=0,
        episodes_per_task=1,action_chunk=8,num_steps_wait=10,flow_steps=10,save_videos=False,
        noise_seed_rule='(suite_index+1)*10000000 + task_id*10000 + episode_idx*100 + chunk_index',
        action_protocol='libero_plus_open_to_legacy_close_v1',manifest_source=str(source),
        manifest_sha256=digest(source),sampling=manifest['sampling'],
        evaluation_scope='same_8380_training_subset',max_attempts=3,stall_timeout=180,import_timeout=600,hard_timeout=3600,
        baseline_result=baseline['baseline_result'],baseline_result_sha256=baseline['baseline_result_sha256'],
        simulator_core=str(ROOT.parent/'LaWAM_aba2/examples/LIBERO/eval_files/libero_eval_core.py'))
    folder.mkdir(parents=True,exist_ok=True)
    for sub in ('model','servers','workers','episodes','errors'): (folder/sub).mkdir(exist_ok=True)
    existing=read(folder/'protocol.json')
    if existing and existing != protocol:raise ValueError('refusing to change an existing evaluation protocol')
    atomic(folder/'protocol.json',protocol)
    if not (folder/'manifest.json').exists():shutil.copy2(source,folder/'manifest.json')
    if digest(folder/'manifest.json') != protocol['manifest_sha256']:raise ValueError('evaluation manifest changed')
    return protocol


def wait_for_training(folder, train_run):
    last_report=0
    while True:
        status=read(train_run/'status.json');pid=status.get('trainer_pid')
        alive=bool(pid and process_args(pid) and 'stage_III.libero_plus.full_train' in process_args(pid))
        atomic(folder/'status.json',dict(phase='waiting_training',pid=os.getpid(),train_run=str(train_run),
            training_phase=status.get('phase'),training_batch=status.get('batch'),training_processed=status.get('groups_processed'),
            trainer_alive=alive,updated_at=time.time()))
        if training_complete(status) and not alive:
            with (train_run/'trainer.lock').open('r') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(10);continue
                if not training_complete(read(train_run/'status.json')):continue
            log('训练已完整提交 262 批且主进程已退出，开始校验最终权重。')
            # Cleanup only matching original training service PIDs after completion.
            for i in range(8):
                ready=train_run/'servers'/f'server_{i}_ready.json';record=read(ready)
                if record.get('pid'):
                    terminate_owned(record['pid'],['stage_III.libero_plus.server','--ready',str(ready)])
            return status
        if time.time()-last_report>=60:
            log(f'等待训练完成：{status.get("groups_processed","?")}/8380 组，'
                f'批次 {status.get("batch","?")}/262，阶段 {status.get("phase","?")}，主进程存活={alive}')
            if not alive and not training_complete(status):
                log('训练未满足完整结束条件；继续监控，不会使用中途权重启动评估。')
            last_report=time.time()
        time.sleep(10)


def snapshot_model(folder, train_run, protocol):
    import torch
    from .gate_actor import GateActor
    from .train import BASELINE, STUDENT, verify_baseline
    from .checkpoint_cache import prepare_student_cache
    source=train_run/'checkpoints/actor_update_0262.pt'
    current=train_run/'checkpoints/actor_current.pt'
    source_hash=digest(source)
    if digest(current)!=source_hash:raise ValueError('final archive and current actor mismatch')
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if (payload.get('format')!='stage3_gate_actor_v1' or payload.get('iteration')!=262
            or payload['config']['base_student']!=str(STUDENT)
            or payload['config']['selected_groups']!=8380):raise ValueError('wrong final actor identity')
    actor=GateActor();actor.load_state_dict(payload['actor_state_dict'],strict=True)
    if not all(torch.isfinite(p).all() for p in actor.parameters()):raise ValueError('non-finite final actor')
    dest=Path(protocol['actor_checkpoint'])
    if dest.exists():
        if digest(dest)!=source_hash:raise ValueError('refusing to replace existing evaluation actor')
    else:
        tmp=dest.with_suffix('.tmp');shutil.copy2(source,tmp)
        if digest(tmp)!=source_hash:raise ValueError('actor copy hash mismatch')
        tmp.replace(dest)
    cache=prepare_student_cache(STUDENT,BASELINE['base_student_model_sha256'])
    os.environ['STAGE3_STUDENT_CACHE']=str(cache)
    verify_baseline('评估启动前',model_cache=cache)
    bundle=dict(actor_checkpoint=str(dest),actor_sha256=source_hash,actor_source=str(source),iteration=262,
        base_student=str(STUDENT),base_student_sha256=BASELINE['base_student_model_sha256'],
        student_load_cache=str(cache),teacher_source=BASELINE['teacher_source'],gate_mode='beta_mean',
        baseline_result_sha256=BASELINE['baseline_result_sha256'],verified_at=time.time())
    atomic(folder/'model_bundle.json',bundle)
    log('最终 GateActor 快照、冻结 Student 和原 89.62% 结果校验通过；进入推理准备。')
    return cache


def start_server(folder, protocol, slot):
    from .train import MODEL_PY, runtime_env, rpc
    port=PORTS[slot];gpu=GPUS[slot];ready=folder/'servers'/f'server_{slot}_ready.json'
    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    ready.unlink(missing_ok=True)
    env=runtime_env(gpu);env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cmd=[str(MODEL_PY),'-u','-m','stage_III.libero_plus.eval_server','--device','cuda:0','--port',str(port),
         '--ready',str(ready),'--checkpoint',protocol['base_student'],'--actor-checkpoint',protocol['actor_checkpoint']]
    stream=(folder/'servers'/f'server_{slot}.log').open('a',buffering=1)
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(folder/'servers'/f'owner_{slot}.json',dict(pid=proc.pid,ready=str(ready),gpu=gpu,port=port))
    url=f'http://127.0.0.1:{port}';last_report=0
    try:
        while True:
            if proc.poll() is not None:raise RuntimeError(f'evaluation server {slot} exited, see {stream.name}')
            phase='等待端口'
            try:
                info=rpc(url,'/health',timeout=10)
                if info.get('load_state')=='failed':raise RuntimeError(info.get('load_error'))
                if (info.get('load_state')=='ready' and info.get('checkpoint')==protocol['base_student']
                        and info.get('actor_checkpoint')==protocol['actor_checkpoint']
                        and info.get('gate_mode')=='beta_mean' and str(info.get('physical_gpu'))==str(gpu)):
                    log(f'评估路{slot} GPU{gpu}模型就绪，GateActor=最终262批，部署模式=Beta均值。')
                    return proc,url,stream
                if info.get('load_state','idle')=='idle':
                    info=rpc(url,'/start_load',dict(checkpoint=protocol['base_student'],actor_checkpoint=protocol['actor_checkpoint']),timeout=15)
                phase=info.get('load_phase','模型加载中')
            except (urllib.error.URLError,TimeoutError,ConnectionError):phase='加载连接等待'
            if time.time()-last_report>=30:
                log(f'评估路{slot} GPU{gpu}：{phase}');last_report=time.time()
            time.sleep(2)
    except BaseException:
        stop(proc);stream.close();raise


def mark_interrupted(folder, slot, state, reason):
    from .eval_common import finish
    db=connect(folder)
    try:
        for row in db.execute("SELECT * FROM tasks WHERE state='running' AND worker=?",(slot,)).fetchall():
            job=dict(row);record=dict(id=job['id'],worker=slot,attempt=job['attempts'],error=reason,time=time.time())
            atomic(folder/'errors'/f'{job["id"]:05d}_attempt{job["attempts"]}.json',record)
            finish(db,job,slot,state,record,reason)
            log(f'评估路{slot} 样本{job["id"]} {state}：已记录，继续下一项。')
    finally:db.close()


def run_slot(folder, protocol, slot, servers, stop_event):
    from .train import SIMPY,runtime_env
    env=runtime_env(GPUS[slot]);env['CUDA_VISIBLE_DEVICES']=''
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    worker_log=(folder/'workers'/f'worker_{slot}.log').open('a',buffering=1)
    startup_failures=0
    try:
        while not stop_event.is_set():
            db=connect(folder)
            pending=db.execute("SELECT COUNT(*) FROM tasks WHERE state='pending'").fetchone()[0];db.close()
            if not pending:return
            proc=subprocess.Popen([str(SIMPY),'-u','-m','stage_III.libero_plus.eval_worker',
                '--folder',str(folder),'--worker',str(slot),'--url',servers[slot][1]],cwd=ROOT,env=env,
                stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(folder/'workers'/f'owner_{slot}.json',dict(pid=proc.pid,slot=slot,gpu=GPUS[slot],started=time.time()))
            launched=time.time();interrupted=False;server_bad=False
            try:
                while proc.poll() is None:
                    if stop_event.is_set():stop(proc);return
                    if servers[slot][0].poll() is not None:
                        stop(proc);mark_interrupted(folder,slot,'error','模型服务退出，将重启服务')
                        server_bad=True;interrupted=True;break
                    db=connect(folder)
                    job=db.execute("SELECT id,started FROM tasks WHERE state='running' AND worker=?",(slot,)).fetchone();db.close()
                    heartbeat=read(folder/'workers'/f'worker_{slot}.json')
                    active=bool(job and heartbeat.get('pid')==proc.pid and heartbeat.get('job')==job['id'])
                    latest=float(heartbeat.get('updated_at',launched)) if active else (job['started'] if job else launched)
                    budget=float(heartbeat.get('timeout_seconds',600)) if active else 600
                    elapsed=time.time()-(job['started'] if job else launched)
                    if time.time()-latest>budget or elapsed>3600:
                        phase=heartbeat.get('phase','启动') if active else '启动'
                        reason=f'{phase}无进展{time.time()-latest:.0f}秒/上限{budget}，本项耗时{elapsed:.0f}秒'
                        stop(proc);mark_interrupted(folder,slot,'timeout',reason)
                        interrupted=True;server_bad=phase=='模型推理';break
                    time.sleep(2)
                if not interrupted and proc.returncode not in (0,None):
                    mark_interrupted(folder,slot,'error',f'采样进程退出 code={proc.returncode}')
                    startup_failures+=1
                    if startup_failures>=3:raise RuntimeError(f'worker {slot} repeatedly failed; see {worker_log.name}')
                else:startup_failures=0
                if server_bad:
                    old=servers[slot];stop(old[0]);old[2].close()
                    servers[slot]=start_server(folder,protocol,slot)
            finally:
                if proc.poll() is None:stop(proc)
    finally:worker_log.close()


def cleanup_previous(folder):
    for i in range(8):
        owner=read(folder/'workers'/f'owner_{i}.json')
        if owner.get('pid'):terminate_owned(owner['pid'],['stage_III.libero_plus.eval_worker','--folder',str(folder)])
        owner=read(folder/'servers'/f'owner_{i}.json')
        if owner.get('pid'):terminate_owned(owner['pid'],['stage_III.libero_plus.eval_server','--ready',owner['ready']])
    db=connect(folder)
    db.execute("UPDATE tasks SET state='pending',worker=NULL WHERE state='running'");db.close()


def write_results(folder,protocol):
    summary=result_summary(folder,protocol);atomic(folder/'results.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train-run',required=True,type=Path)
    parser.add_argument('--folder',required=True,type=Path);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();folder=args.folder.resolve();train_run=args.train_run.resolve()
    if not train_run.is_relative_to((STAGE3/'runs').resolve()):raise ValueError('wrong training run location')
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=prepare(folder,train_run)
    if args.prepare_only:
        atomic(folder/'status.json',dict(phase='prepared',train_run=str(train_run),updated_at=time.time()))
        write_results(folder,protocol);log('衔接协议及 4×7 进度表已准备完毕；没有启动 GPU 进程。');return
    servers=[None]*8;stop_event=threading.Event()
    try:
        wait_for_training(folder,train_run)
        atomic(folder/'status.json',dict(phase='verifying',pid=os.getpid(),updated_at=time.time()))
        cache=snapshot_model(folder,train_run,protocol)
        initialize_queue(folder,protocol['tasks']);cleanup_previous(folder)
        current=write_results(folder,protocol)
        if current['complete']:
            atomic(folder/'status.json',dict(phase='completed',pid=os.getpid(),updated_at=time.time()));return
        atomic(folder/'status.json',dict(phase='starting_servers',pid=os.getpid(),updated_at=time.time()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs={pool.submit(start_server,folder,protocol,i):i for i in range(8)}
            errors=[]
            for job in concurrent.futures.as_completed(jobs):
                try:servers[jobs[job]]=job.result()
                except Exception as exc:errors.append(exc)
            if errors:raise errors[0]
        log('八路评估服务全部就绪：GPU1/3/6/7 各两路，开始8380项推理。')
        for wave in range(1,4):
            db=connect(folder);pending=db.execute("SELECT COUNT(*) FROM tasks WHERE state='pending'").fetchone()[0];db.close()
            if not pending:
                pending=requeue_errors(folder,3)
                if not pending:break
            log(f'第{wave}次评估遍历开始，待运行{pending}项；超时记录后继续，其余路独立推进。')
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(run_slot,folder,protocol,i,servers,stop_event) for i in range(8)]
                last=0
                while not all(f.done() for f in futures):
                    failed=[f for f in futures if f.done() and f.exception() is not None]
                    if failed:
                        stop_event.set();raise failed[0].exception()
                    summary=write_results(folder,protocol)
                    atomic(folder/'status.json',dict(phase='evaluating',pid=os.getpid(),wave=wave,
                        started_at=read(folder/'evaluation_start.json').get('time',time.time()),updated_at=time.time()))
                    if not (folder/'evaluation_start.json').exists():atomic(folder/'evaluation_start.json',dict(time=time.time()))
                    if time.time()-last>=60:
                        log(f'推理进度 {summary["processed"]}/8380；正常完成{summary["done"]}，'
                            f'成功{summary["successes"]}，超时{summary["timeout"]}，错误{summary["error"]}')
                        last=time.time()
                    time.sleep(5)
                for f in futures:f.result()
            if not requeue_errors(folder,3):break
        summary=write_results(folder,protocol)
        with (folder/'matrix.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['suite','category','expected','done','successes','timeout','error','success_rate'])
            for suite in SUITES:
                for category in CATEGORIES:
                    cell=summary['cells'][suite][category]
                    writer.writerow([suite,category,*[cell[k] for k in ('expected','done','successes','timeout','error')],
                                     cell['successes']/cell['done'] if cell['done'] else None])
        from .train import verify_baseline
        verify_baseline('评估结束',model_cache=cache)
        if digest(protocol['actor_checkpoint'])!=read(folder/'model_bundle.json')['actor_sha256']:
            raise ValueError('actor snapshot changed during evaluation')
        phase='completed' if summary['complete'] else 'completed_with_errors'
        atomic(folder/'status.json',dict(phase=phase,pid=os.getpid(),completed_at=time.time(),updated_at=time.time()))
        log(f'推理结束：正常完成{summary["done"]}/8380，成功{summary["successes"]}，'
            f'超时/错误{summary["timeout"]+summary["error"]}；完整覆盖={summary["complete"]}。')
    except BaseException as exc:
        stop_event.set();atomic(folder/'status.json',dict(phase='failed',pid=os.getpid(),error=repr(exc),updated_at=time.time()))
        log(f'衔接/评估异常已记录：{exc!r}');raise
    finally:
        for item in servers:
            if item:
                stop(item[0]);item[2].close()
        lock.close()


if __name__=='__main__':main()
