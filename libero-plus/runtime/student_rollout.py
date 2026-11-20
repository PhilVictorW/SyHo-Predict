"""Persistent simulator worker claiming one selected task at a time from SQLite."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

from student_eval_common import SUITES, BENCH, ACTION_PROTOCOL, atomic

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);parser.add_argument('--worker',type=int,required=True);parser.add_argument('--url',required=True)
    args=parser.parse_args();folder=Path(args.folder);protocol=json.loads((folder/'protocol.json').read_text())
    from sim_compat import enable
    enable()
    import libero.libero as libero
    assert Path(libero.__file__).resolve().is_relative_to(BENCH)
    import numpy as np
    np.random.seed(0)
    from examples.LIBERO.eval_files import libero_eval_core as core
    from student_action_protocol import PlusStudentClient
    core._configure_logging()
    adapter=core.get_benchmark_adapter('libero_plus')
    benchmark=core._get_benchmark_dict();suites={}
    client=PlusStudentClient(protocol['checkpoint'],args.url,action_protocol=protocol['action_protocol'])
    db=sqlite3.connect(folder/'queue.sqlite',timeout=60,isolation_level=None)
    status_path=folder/f'worker{args.worker}.json'
    def status(**extra):atomic(status_path,dict(worker=args.worker,pid=os.getpid(),updated=time.time(),**extra))
    status(phase='ready')
    try:
        while True:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("SELECT id,payload FROM tasks WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                db.execute('COMMIT');break
            job,payload=row;task=json.loads(payload)
            db.execute("UPDATE tasks SET state='running',worker=?,attempts=attempts+1 WHERE id=?",(args.worker,job));db.execute('COMMIT')
            status(phase='evaluating',task=task)
            try:
                suite_name=task['suite'];tid=task['task_id']
                if suite_name not in suites:suites[suite_name]=benchmark[suite_name]()
                suite=suites[suite_name];actual=suite.get_task(tid)
                # The official manifest is 1-based, the runtime suite is 0-based.
                if getattr(actual,'name',None)!=task['name']:
                    raise ValueError(f'Task manifest/name mismatch: {actual} versus {task}')
                initial=suite.get_task_init_states(tid)
                client.configure_episode(SUITES.index(suite_name),tid,0)
                eval_args=core.EvalArgs(pretrained_path=protocol['checkpoint'],benchmark_variant='libero_plus',task_suite_name=suite_name,
                    task_id=tid,num_trials_per_task=1,num_workers=1,seed=0,num_steps_wait=10,save_videos=False,
                    video_out_path=str(folder),eval_action_chunk_len=8)
                core._configure_mujoco_backend(eval_args,adapter)
                result=core._run_single_episode(args=eval_args,adapter=adapter,task=actual,task_id=tid,episode_idx=0,
                    init_state=initial[core._task_init_state_index(initial,0)],client_model=client,run_output_dir=folder,env_seed=0)
                data=dataclasses.asdict(result)
                if result.error:
                    raise RuntimeError(result.error)
                data.update(suite=suite_name,category=task['category'],checkpoint=protocol['checkpoint'],worker=args.worker,
                    action_protocol_id=ACTION_PROTOCOL['id'],completed_at=time.time())
                atomic(folder/'episodes'/f'{suite_name}_{tid:05d}.json',data)
                db.execute("UPDATE tasks SET state='done' WHERE id=?",(job,))
                print(json.dumps(dict(event='episode',suite=suite_name,task_id=tid,success=result.success)),flush=True)
            except BaseException as exc:
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
                traceback.print_exc()
                count=db.execute('SELECT attempts FROM tasks WHERE id=?',(job,)).fetchone()[0]
                atomic(folder/'errors'/f'{job:04d}_{count}.json',dict(task=task,error=repr(exc),worker=args.worker,time=time.time()))
                db.execute("UPDATE tasks SET state=?,error=? WHERE id=?",('failed' if count>=3 else 'pending',repr(exc),job))
        status(phase='finished')
    finally:
        client.close();db.close()

if __name__=='__main__':main()
