"""Fixed full10030/smoke evaluator. Without --execute it only prints the plan."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
from .config import *
from ablation.aba1.plus_common.preflight import check_manifest
from .deployment import validate_checkpoint
from ablation.aba1.plus_common.environment import subprocess_environment
from goal_local.eval_client import rpc

BASE=ROOT.parent/"LaWAM_aba2"

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
    args=p.parse_args(argv);cfg=load_config(args.config)
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
    if not args.gpus:p.error("--execute requires explicit --gpus")
    gpus=args.gpus.split(",")
    if not all(g.isdigit() for g in gpus):p.error("GPU IDs must be numeric")
    meta,marker=validate_checkpoint(checkpoint,cfg,verify_hash=True)
    if args.scope=="full":validate_final_budget(meta,cfg)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/".evaluation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
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
            code=code_identity(),eval_core_sha256=file_sha256(BASE/"examples/LIBERO/eval_files/libero_eval_core.py"),
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
        def worker(index,gpu):
            env=subprocess_environment(cfg,gpu,folder/"cache"/f"worker{index}")
            env["PYTHONPATH"]=str(WORK)+os.pathsep+env["PYTHONPATH"]
            processes=[]
            def launch(cmd,name):
                with (folder/name).open("a") as stream:
                    proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(proc);return proc
            ready_file=folder/f"server{index}.json"
            try:
                server=launch([cfg["model_python"],"-B","-m","a0_single_latent.server",
                    "--config",str(Path(args.config).resolve()),"--ready",str(ready_file),"--port","0"],f"server{index}.log")
                deadline=time.monotonic()+1200
                while True:
                    if server.poll() is not None:raise RuntimeError(f"Server {index} exited; inspect log")
                    info=json.loads(ready_file.read_text()) if ready_file.exists() else {}
                    if info.get("pid")==server.pid and info.get("port"):break
                    if time.monotonic()>deadline:raise TimeoutError("Server startup")
                    time.sleep(2)
                url=f"http://127.0.0.1:{info['port']}"
                info=rpc(url,"/load",dict(checkpoint=str(checkpoint)),timeout=1800)
                if info["step"]!=meta["step"] or info["teacher_loaded"] or info["uses_future"] or info["uses_state"]:raise ValueError("Wrong deployment policy")
                for restart in range(3):
                    client=launch([cfg["simulator_python"],"-B","-m","a0_single_latent.rollout",
                        "--folder",str(folder),"--worker",str(index),"--url",url],f"worker{index}.log")
                    last=time.monotonic();mtime=0
                    while client.poll() is None:
                        if server.poll() is not None:raise RuntimeError("Policy server died")
                        status_file=folder/f"worker{index}.json"
                        stamp=status_file.stat().st_mtime_ns if status_file.exists() else 0
                        if stamp!=mtime:mtime=stamp;last=time.monotonic()
                        if time.monotonic()-last>3600:terminate(client);break
                        time.sleep(3)
                    if client.returncode!=0:
                        atomic_json(folder/"errors"/f"worker{index}_restart{restart}_{time.time_ns()}.json",
                            dict(worker=index,exit_code=client.returncode,kind="simulator_process_exit",restart=restart))
                    with sqlite3.connect(folder/"queue.sqlite",timeout=60) as db:
                        db.execute("UPDATE tasks SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END, error=COALESCE(error,?) WHERE state='running' AND worker=?",
                            (f"worker exit {client.returncode}",index))
                    if client.returncode==0:break
                    if restart==2:raise RuntimeError("Simulator repeatedly exited")
            finally:
                for proc in reversed(processes):terminate(proc)
        try:
            if not collect(folder)["complete"]:
                with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                    jobs=[pool.submit(worker,i,gpu) for i,gpu in enumerate(gpus)]
                    for job in jobs:job.result()
        finally:result=collect(folder)
        if not result["complete"]:raise RuntimeError(f"Incomplete: {result['completed']}/{result['expected']}; errors retained")
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
