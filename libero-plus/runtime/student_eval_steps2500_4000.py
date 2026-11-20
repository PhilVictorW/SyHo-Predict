"""Four independent stratified evaluations: one checkpoint per physical GPU."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import time
from student_eval_common import WORK, TRAIN, atomic
from student_eval_queue import evaluate

CONTROL=WORK/'evaluation/stage_II_steps2500_4000_parallel_20261008'
MANIFEST=CONTROL/'manifest_same_as_steps1000_1500_2000.json'
# One checkpoint and one evaluator process on each requested physical GPU.
JOBS=[(2500,0,24930),(3000,2,24931),(3500,6,24932),(4000,7,24933)]

def one(step,gpu,port):
    checkpoint=TRAIN/f'step_{step:06d}'
    marker=json.loads((checkpoint/'checkpoint_complete.json').read_text())
    assert marker.get('step')==step and marker.get('mode')=='student'
    data=json.loads(MANIFEST.read_text())
    assert len(data['tasks'])==8380 and data['sampling']['basis_step']==500
    control=CONTROL/f'step_{step:06d}'
    folder=control/f'step_{step:06d}_step500_stratified_gripper_v1'
    control.mkdir(parents=True,exist_ok=True)
    atomic(control/'status.json',dict(step=step,gpu=gpu,port=port,scope=data['scope'],expected=8380,phase='starting',folder=str(folder),updated=time.time()))
    result=evaluate(step,checkpoint,spec=dict(folder=folder,manifest=MANIFEST,scope=data['scope']),status_root=control,ports=[port],gpus=[gpu])
    atomic(control/'summary.json',result)
    atomic(control/'status.json',dict(step=step,gpu=gpu,port=port,scope=data['scope'],expected=8380,phase='completed',folder=str(folder),episodes=result['episodes'],successes=result['successes'],success_rate=result['success_rate'],updated=time.time()))
    return result

def main():
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(CONTROL/'queue_owner.json',dict(pid=os.getpid(),started=time.time(),jobs=[dict(step=s,gpu=g,port=p) for s,g,p in JOBS],detached_tmux='lawam_plus_steps2500_4000_eval'))
    atomic(CONTROL/'status.json',dict(phase='starting',jobs=[dict(step=s,gpu=g,port=p) for s,g,p in JOBS],updated=time.time()))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one,*job) for job in JOBS]
            results=[]
            for job,f in zip(JOBS,futures):
                results.append(f.result())
        atomic(CONTROL/'summary.json',results)
        atomic(CONTROL/'status.json',dict(phase='completed',jobs=[dict(step=s,gpu=g,port=p) for s,g,p in JOBS],updated=time.time()))
    except BaseException as exc:
        atomic(CONTROL/'status.json',dict(phase='failed',error=repr(exc),updated=time.time()));raise

if __name__=='__main__':main()
