"""Isolated step500 evaluation alongside the existing student queue."""
import fcntl
import json
import os
from pathlib import Path
import time

from student_eval_common import WORK,TRAIN,SUITES,CATEGORIES,atomic
from student_eval_queue import evaluate

CONTROL=WORK/'evaluation/stage_II_step500_parallel_20260926'
FOLDER=CONTROL/'step_000500_step500_stratified_gripper_v1'
PORTS=[24920,24921,24922,24923]
GPUS=[6,6,6,6]

def main():
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=TRAIN/'step_000500'
    marker=json.loads((checkpoint/'checkpoint_complete.json').read_text())
    assert marker['step']==500 and marker['mode']=='student'
    manifest=CONTROL/'manifest_same_as_step1000.json'
    m=json.loads(manifest.read_text())
    reference=WORK/'evaluation/stage_II_20260925/step_001000_step500_stratified_gripper_v1/protocol.json'
    original=json.loads(reference.read_text())
    assert m['tasks']==original['tasks'] and m['sampling']==original['sampling']
    assert len(m['tasks'])==8380 and len({(t['suite'],t['category']) for t in m['tasks']})==28
    atomic(CONTROL/'queue_owner.json',dict(pid=os.getpid(),started=time.time(),step=500,physical_gpus=GPUS,ports=PORTS,checkpoint=str(checkpoint),reference_protocol=str(reference)))
    base=dict(step=500,folder=str(FOLDER),scope=m['scope'],expected=8380)
    atomic(CONTROL/'status.json',dict(base,phase='starting',updated=time.time()))
    try:
        result=evaluate(500,checkpoint,spec=dict(folder=FOLDER,manifest=manifest,scope=m['scope']),status_root=CONTROL,ports=PORTS,gpus=GPUS)
        atomic(CONTROL/'summary.json',[result])
        atomic(CONTROL/'status.json',dict(base,phase='completed',episodes=result['episodes'],successes=result['successes'],success_rate=result['success_rate'],updated=time.time()))
    except BaseException as exc:
        atomic(CONTROL/'status.json',dict(base,phase='failed',error=repr(exc),updated=time.time()));raise

if __name__=='__main__':main()
