"""Fixed full10030/smoke evaluator. Without --execute it only prints the plan."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import sys,threading,base64
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import signal
import sqlite3
import subprocess
import time
from a0_single_latent.config import *
from ablation.aba1.plus_common.preflight import check_manifest
from a0_single_latent.deployment import validate_checkpoint
from ablation.aba1.plus_common.environment import subprocess_environment
from goal_local.eval_client import rpc

BASE=ROOT.parent/"LaWAM_aba2"

def query_gpus():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=10)
    return [dict(gpu=int(g),free_mib=int(m)) for g,m in
            (line.split(',') for line in result.stdout.splitlines() if line.strip())]

def choose_placement(snapshot, count=4, maximum=3, per_worker_gib=8, reserve_gib=4):
    if count!=4 or not 1<=maximum<=3:raise ValueError('Exactly four GPUs; at most three workers per GPU')
    selected=sorted(snapshot,key=lambda x:(-x['free_mib'],x['gpu']))[:count]
    if len(selected)!=count:raise ValueError('Four physical GPUs are required')
    selected=[dict(x,workers=max(1,min(maximum,int((x['free_mib']/1024-reserve_gib)//per_worker_gib)))) for x in selected]
    workers=[str(x['gpu']) for slot in range(maximum) for x in selected if slot<x['workers']]
    return dict(selected=selected,worker_gpus=workers,max_workers_per_gpu=maximum,
        estimated_worker_gib=per_worker_gib,reserve_gib=reserve_gib,
        selection_rule='largest free memory after training exits',time=time.time())



def validate_final_budget(meta,cfg):
    extra=meta['extra'];target=cfg['stop_after_updates']
    batch=cfg['world_size']*cfg['micro_batch']*cfg['accumulation']
    if (extra.get('smoke') or not extra.get('budget_complete') or
        meta['step']!=target or extra.get('completed_updates')!=target or
        extra.get('stop_after_updates')!=target or extra.get('seen_anchors')!=target*batch):
        raise ValueError('Full evaluation requires the final 1000-update model')

def summarize(protocol,rows,states=None):
    wanted={(r["suite"],r["task_id"]):r for r in protocol["tasks"]};seen=set()
    for r in rows:
        key=(r["suite"],r["task_id"])
        if key in seen or key not in wanted or r.get("error"):raise ValueError("Duplicate/unexpected/error episode")
        if (r["checkpoint"]!=protocol["checkpoint"] or r["episode_idx"]!=0 or
            r["model_sha256"]!=protocol["model_sha256"] or r["protocol_sha256"]!=protocol["protocol_sha256"] or
            r["action_protocol_id"]!=ACTION_PROTOCOL["id"] or r["category"]!=wanted[key]["category"]):
            raise ValueError("Episode provenance mismatch")
        seen.add(key)
    def aggregate(tasks,selected):
        n=len(selected);success=sum(bool(r["success"]) for r in selected)
        steps=[r["num_actions"] for r in selected if r["success"]]
        return dict(completed=n,expected=len(tasks),successes=success,complete=n==len(tasks),
            success_rate=success/len(tasks) if n==len(tasks) and tasks else None,
            completed_success_rate=success/n if n else None,
            mean_success_actions=sum(steps)/len(steps) if steps else None)
    tasks=protocol["tasks"]
    result=aggregate(tasks,rows)
    result.update(scope=protocol["scope"],checkpoint=protocol["checkpoint"],states=states or {},
        model_sha256=protocol["model_sha256"],protocol_sha256=protocol["protocol_sha256"],
        infrastructure_errors=(states or {}).get("failed",0),updated_at=time.time())
    result["suites"]={s:aggregate([t for t in tasks if t["suite"]==s],[r for r in rows if r["suite"]==s]) for s in SUITES}
    result["categories"]={c:aggregate([t for t in tasks if t["category"]==c],[r for r in rows if r["category"]==c]) for c in CATEGORIES}
    result["cells"]={s:{c:aggregate([t for t in tasks if t["suite"]==s and t["category"]==c],
        [r for r in rows if r["suite"]==s and r["category"]==c]) for c in CATEGORIES} for s in SUITES}
    return result

def collect(folder):
    folder=Path(folder);protocol=json.loads((folder/"protocol.json").read_text())
    rows=[json.loads(p.read_text()) for p in sorted((folder/"episodes").glob("*.json"))]
    states={}
    if (folder/"queue.sqlite").exists():
        with sqlite3.connect(folder/"queue.sqlite",timeout=60) as db:
            states=dict(db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
    result=summarize(protocol,rows,states);atomic_json(folder/"results.json",result)
    with (folder/"matrix.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(["suite","category","completed","expected","successes","success_rate","mean_success_actions"])
        for s in SUITES:
            for c in CATEGORIES:
                x=result["cells"][s][c];writer.writerow([s,c,*[x[k] for k in
                    ("completed","expected","successes","success_rate","mean_success_actions")]])
    return result

def terminate(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(WORK/"config.json"));p.add_argument("--checkpoint")
    p.add_argument("--scope",choices=["full","smoke"],default="full")
    p.add_argument("--gpus",help="One policy server/simulator per comma-separated physical GPU")
    p.add_argument("--execute",action="store_true");p.add_argument("--dry-run",action="store_true")
    p.add_argument("--auto-gpus",type=int,default=4)
    p.add_argument("--max-workers-per-gpu",type=int,default=3)
    p.add_argument("--memory-per-worker-gib",type=float,default=8.)
    p.add_argument("--reserve-gib",type=float,default=4.)
    args=p.parse_args(argv);cfg=load_config(args.config)
    if args.gpus:p.error("This evaluator selects four GPUs automatically")
    if args.auto_gpus!=4 or not 1<=args.max_workers_per_gpu<=3:p.error("Need four GPUs and 1..3 workers per GPU")
    if args.memory_per_worker_gib<6 or args.reserve_gib<2:p.error("Insufficient configured memory allowance")
    if args.execute and args.dry_run:p.error("--execute conflicts with --dry-run")
    manifest_info=check_manifest(cfg)
    final_file=run_root(cfg)/'final.json'
    final=Path(json.loads(final_file.read_text())['path']) if final_file.exists() else run_root(cfg)/'final_checkpoint_pending'
    checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else final
    tasks=json.loads(Path(cfg["full_eval_manifest"]).read_text())["tasks"]
    if args.scope=="smoke":
        tasks=[next(t for t in tasks if t["suite"]==s and t["category"]==c) for s in SUITES for c in CATEGORIES]
    folder=checkpoint.parent.parent/"evaluation"/f"{args.scope}_{checkpoint.name}" if checkpoint.parent.name=="checkpoints" else run_root(cfg)/"evaluation"/args.scope
    ready=(checkpoint/"model.pt").exists()
    if ready:validate_checkpoint(checkpoint,cfg)
    summary=dict(checkpoint=str(checkpoint),checkpoint_available=ready,scope=args.scope,tasks=len(tasks),
        output=str(folder),action_protocol=ACTION_PROTOCOL,experimental_inference_started=False)
    if not args.execute:print(json.dumps(summary,indent=2));return
    if not ready:raise FileNotFoundError(checkpoint)
    meta,marker=validate_checkpoint(checkpoint,cfg,verify_hash=True)
    if args.scope=="full":validate_final_budget(meta,cfg)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/".evaluation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        placement=choose_placement(query_gpus(),args.auto_gpus,args.max_workers_per_gpu,args.memory_per_worker_gib,args.reserve_gib)
        gpus=placement["worker_gpus"]
        atomic_json(folder/"placement.json",placement)
        atomic_json(folder/f"placement_{time.time_ns()}.json",placement)
        print(json.dumps(dict(phase="adaptive_gpu_selection",**placement)),flush=True)
        protocol=dict(protocol=PROTOCOL,scope=args.scope,checkpoint=str(checkpoint),
            model_sha256=marker["files"]["model.pt"]["sha256"],step=meta["step"],
            cumulative_step=marker["cumulative_step"],tasks=tasks,
            manifest_sha256=manifest_info["sha256"],benchmark=cfg["benchmark"],max_attempts=3,
            sim_sysroot=cfg["sim_sysroot"],libero_config_path=cfg["libero_config_path"],
            libero_config_sha256=file_sha256(Path(cfg["libero_config_path"])/"config.yaml"),
            env_seed=0,initial_state_index=0,episodes_per_task=1,num_steps_wait=10,max_policy_steps=MAX_STEPS,
            action_chunk=8,flow_steps=10,uses_state=False,uses_future=False,uses_dino=True,latent_slots=16,actor=False,save_videos=False,
            noise_seed_rule="(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index",
            inference_autocast="bfloat16",initial_noise_dtype="bfloat16",action_protocol=ACTION_PROTOCOL,
            code=code_identity(),adaptive_evaluator_sha256=file_sha256(Path(__file__)),eval_core_sha256=file_sha256(BASE/"examples/LIBERO/eval_files/libero_eval_core.py"),
            package_versions=runtime_environment())
        protocol["protocol_sha256"]=digest_json(protocol)
        protocol_path=folder/"protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
            raise ValueError("Resume evaluation protocol changed")
        atomic_json(protocol_path,protocol)
        for name in ("episodes","errors"): (folder/name).mkdir(exist_ok=True)
        with sqlite3.connect(folder/"queue.sqlite",timeout=60) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY,payload TEXT,state TEXT,worker INTEGER,attempts INTEGER DEFAULT 0,error TEXT)")
            for i,task in enumerate(tasks):
                done=(folder/"episodes"/f"{task['suite']}_{task['task_id']:05d}.json").exists()
                db.execute("INSERT OR IGNORE INTO tasks (id,payload,state) VALUES (?,?,?)",
                    (i,json.dumps(task),"done" if done else "pending"))
                if done:db.execute("UPDATE tasks SET state='done' WHERE id=?",(i,))
            db.execute("UPDATE tasks SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END WHERE state='running'")
        startup_lock=threading.Lock()
        def pending():
            with sqlite3.connect(folder/"queue.sqlite",timeout=60) as db:
                return db.execute("SELECT COUNT(*) FROM tasks WHERE state='pending'").fetchone()[0]
        def worker(index,gpu):
            env=subprocess_environment(cfg,gpu,folder/"cache"/f"worker{index}")
            env["PYTHONPATH"]=str(WORK)+os.pathsep+env["PYTHONPATH"]
            env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
            def state(phase,**extra):
                atomic_json(folder/f"scheduler_worker{index}.json",
                    dict(worker=index,gpu=int(gpu),phase=phase,time=time.time(),**extra))
            last_error=None
            for generation in range(3):
                processes=[];client=None;server=None
                def launch(cmd,name):
                    with (folder/name).open("a") as stream:
                        proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=stream,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    processes.append(proc);return proc
                try:
                    # Only one new model is constructed at a time. Existing rollouts continue.
                    while True:
                        if not pending():state("no_pending_tasks");return
                        state("waiting_for_memory_and_load_slot",generation=generation)
                        startup_lock.acquire()
                        try:
                            free=next(x["free_mib"] for x in query_gpus() if x["gpu"]==int(gpu))
                        except BaseException:
                            startup_lock.release();raise
                        if free>=(args.memory_per_worker_gib+args.reserve_gib)*1024:break
                        startup_lock.release();time.sleep(10)
                    try:
                        state("loading_model",generation=generation,free_before_mib=free)
                        ready_file=folder/f"server{index}.json"
                        server=launch([cfg["model_python"],"-B","-m","a0_single_latent.server",
                            "--config",str(Path(args.config).resolve()),"--ready",str(ready_file),"--port","0"],f"server{index}.log")
                        deadline=time.monotonic()+1800
                        while True:
                            if server.poll() is not None:raise RuntimeError(f"Server {index} exited; inspect log")
                            info=json.loads(ready_file.read_text()) if ready_file.exists() else {}
                            if info.get("pid")==server.pid and info.get("port"):break
                            if time.monotonic()>deadline:raise TimeoutError("Server startup")
                            time.sleep(2)
                        url=f"http://127.0.0.1:{info['port']}"
                        info=rpc(url,"/load",dict(checkpoint=str(checkpoint)),timeout=1800)
                        if info["step"]!=meta["step"] or info["teacher_loaded"] or info["uses_future"] or info["uses_state"]:
                            raise ValueError("Wrong deployment policy")
                        # Exercise vision and DiT allocations before admitting the next model.
                        rgb=base64.b64encode(bytes(256*256*3)).decode()
                        warmup=rpc(url,"/infer",dict(checkpoint=str(checkpoint),main=rgb,wrist=rgb,
                            instruction="pick up the object and place it in the target container",noise_seed=0),timeout=1800)
                        if warmup.get("shape")!=[8,7]:raise ValueError("Warmup action shape mismatch")
                        client=launch([cfg["simulator_python"],"-B","-m","a0_single_latent.rollout",
                            "--folder",str(folder),"--worker",str(index),"--url",url],f"worker{index}.log")
                        state("running",generation=generation,server_pid=server.pid,client_pid=client.pid,url=url)
                    finally:startup_lock.release()
                    last=time.monotonic();mtime=0
                    while client.poll() is None:
                        if server.poll() is not None:raise RuntimeError("Policy server died")
                        status_file=folder/f"worker{index}.json"
                        stamp=status_file.stat().st_mtime_ns if status_file.exists() else 0
                        if stamp!=mtime:mtime=stamp;last=time.monotonic()
                        if time.monotonic()-last>3600:raise TimeoutError("Simulator heartbeat stopped")
                        time.sleep(3)
                    if client.returncode!=0:raise RuntimeError(f"Simulator exited {client.returncode}")
                    state("finished");return
                except Exception as error:
                    last_error=error
                    state("retrying",generation=generation,error=repr(error))
                    atomic_json(folder/"errors"/f"scheduler_{index}_{generation}_{time.time_ns()}.json",
                        dict(worker=index,gpu=gpu,generation=generation,error=repr(error),time=time.time()))
                finally:
                    for proc in reversed(processes):terminate(proc)
                    with sqlite3.connect(folder/"queue.sqlite",timeout=60) as db:
                        db.execute("UPDATE tasks SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END,error=COALESCE(error,?) WHERE state='running' AND worker=?",
                            (repr(last_error),index))
                time.sleep(5)
            state("failed",error=repr(last_error))
            raise RuntimeError(f"Worker {index} exhausted startup/restart attempts: {last_error}")
        try:
            if not collect(folder)["complete"]:
                with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                    jobs=[pool.submit(worker,i,gpu) for i,gpu in enumerate(gpus)]
                    failures=[]
                    for job in as_completed(jobs):
                        try:job.result()
                        except Exception as error:failures.append(repr(error))
                    atomic_json(folder/"worker_failures.json",failures)
        finally:result=collect(folder)
        if not result["complete"]:raise RuntimeError(f"Incomplete: {result['completed']}/{result['expected']}; errors retained")
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
