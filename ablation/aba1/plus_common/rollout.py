"""Persistent simulation worker; only launched by explicit evaluation --execute."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback
from .config import ACTION_PROTOCOL,SUITES,MAX_STEPS,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--folder",required=True);p.add_argument("--worker",type=int,required=True)
    p.add_argument("--url",required=True);args=p.parse_args()
    folder=Path(args.folder);protocol=json.loads((folder/"protocol.json").read_text())
    from sim_compat import enable
    enable()
    import libero.libero as libero
    import numpy as np
    if not Path(libero.__file__).resolve().is_relative_to(Path(protocol["benchmark"])):
        raise ValueError("Wrong LIBERO environment imported")
    from .environment import load_legacy_eval_core
    core=load_legacy_eval_core()
    from .deployment import PlusStudentClient
    core._configure_logging();adapter=core.get_benchmark_adapter("libero_plus")
    if {s:adapter.get_max_steps(s) for s in SUITES}!=MAX_STEPS:raise ValueError("Environment horizon changed")
    benchmark=core._get_benchmark_dict();suites={};np.random.seed(0)
    client=PlusStudentClient(protocol["checkpoint"],args.url,ACTION_PROTOCOL)
    db=sqlite3.connect(folder/"queue.sqlite",timeout=60,isolation_level=None)
    def status(**values):
        atomic_json(folder/f"worker{args.worker}.json",dict(worker=args.worker,pid=os.getpid(),time=time.time(),**values))
    try:
        while True:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT id,payload FROM tasks WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:db.execute("COMMIT");break
            job,payload=row;task=json.loads(payload)
            db.execute("UPDATE tasks SET state='running',worker=?,attempts=attempts+1 WHERE id=?",(args.worker,job))
            db.execute("COMMIT");status(phase="episode",task=task)
            count=db.execute("SELECT attempts FROM tasks WHERE id=?",(job,)).fetchone()[0]
            try:
                suite_name=task["suite"];tid=task["task_id"]
                if suite_name not in suites:suites[suite_name]=benchmark[suite_name]()
                suite=suites[suite_name];actual=suite.get_task(tid)
                if actual.name!=task["name"]:raise ValueError("Task ID/name mismatch")
                states=suite.get_task_init_states(tid);index=core._task_init_state_index(states,0)
                if index!=0:raise ValueError("Expected initial state index0")
                state=states[index]
                raw=np.ascontiguousarray(state)
                state_hash=hashlib.sha256(str(raw.dtype).encode()+str(raw.shape).encode()+raw.tobytes()).hexdigest()
                client.configure_episode(SUITES.index(suite_name),tid,0)
                options=core.EvalArgs(pretrained_path=protocol["checkpoint"],benchmark_variant="libero_plus",
                    task_suite_name=suite_name,task_id=tid,num_trials_per_task=1,num_workers=1,seed=0,
                    num_steps_wait=10,save_videos=False,video_out_path=str(folder),eval_action_chunk_len=8)
                core._configure_mujoco_backend(options,adapter)
                result=core._run_single_episode(args=options,adapter=adapter,task=actual,task_id=tid,
                    episode_idx=0,init_state=state,client_model=client,run_output_dir=folder,env_seed=0)
                if result.error:raise RuntimeError(result.error)
                data=dataclasses.asdict(result)
                data.update(suite=suite_name,category=task["category"],task_name=task["name"],
                    checkpoint=protocol["checkpoint"],model_sha256=protocol["model_sha256"],
                    action_protocol_id=ACTION_PROTOCOL["id"],protocol_sha256=protocol["protocol_sha256"],
                    initial_state_index=0,initial_state_sha256=state_hash,env_seed=0,
                    noise_seed=(SUITES.index(suite_name)+1)*10000000+tid*10000,
                    termination_reason="success" if result.success else "horizon",
                    worker=args.worker,attempt=count,completed_at=time.time())
                atomic_json(folder/"episodes"/f"{suite_name}_{tid:05d}.json",data)
                db.execute("UPDATE tasks SET state='done',error=NULL WHERE id=?",(job,))
                status(phase="completed_episode",task=task,success=result.success)
            except Exception as exc:
                traceback.print_exc()
                atomic_json(folder/"errors"/f"{job:05d}_{count}.json",
                    dict(task=task,error=repr(exc),attempt=count,worker=args.worker,time=time.time()))
                db.execute("UPDATE tasks SET state=?,error=? WHERE id=?",
                    ("failed" if count>=protocol["max_attempts"] else "pending",repr(exc),job))
        status(phase="finished")
    finally:client.close();db.close()

if __name__=="__main__":main()
