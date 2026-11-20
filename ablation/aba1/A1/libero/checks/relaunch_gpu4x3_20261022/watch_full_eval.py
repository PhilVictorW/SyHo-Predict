#!/usr/bin/env python3
"""Independent watcher: completed A1 student step2000 -> LIBERO full 40x50.

Reads training state only. Does not signal training, edit its configuration,
or occupy CUDA before both stages and the training launcher have completed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[3]
sys.path.insert(0,str(ROOT))
from ablation.aba1.libero_common.config import atomic_json, file_sha256, FORMAT
from goal_local.eval_common import SUITES, SIM_PYTHON, model_metadata
OWNER = 'aba1.A1.libero.step_training'
PYTHON = '/home/wangchaoyang/envs/LaWAM/bin/python'


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return {}


def snapshot(run):
    pipeline=read(run/'pipeline.json')
    first=read(run/'stage_I/status.json');second=read(run/'stage_II/status.json')
    return dict(pipeline_phase=pipeline.get('phase','not_started'),stage=pipeline.get('stage'),
        stage_I_step=first.get('step',0),stage_II_step=second.get('step',0),
        world_size=pipeline.get('world_size'),stage_I_phase=first.get('phase'),stage_II_phase=second.get('phase'))


def ready_checkpoint(run):
    """Fail closed on partial, wrong-step, or mismatched completed artifacts."""
    run=Path(run).resolve()
    pipeline=read(run/'pipeline.json')
    if pipeline.get('phase') != 'completed': return None
    expected=run/'stage_II/checkpoints/step_002000'
    for stage,step in [('I',1000),('II',2000)]:
        folder=run/('stage_'+stage)
        state=read(folder/'status.json');final=read(folder/'final.json')
        checkpoint=folder/'checkpoints'/f'step_{step:06d}'
        marker=read(checkpoint/'checkpoint_complete.json')
        mode='posterior' if stage=='I' else 'student'
        if state.get('phase')!='completed' or state.get('step')!=step:
            raise ValueError(f'Completed pipeline has incomplete Stage {stage}')
        if Path(final.get('path','')).resolve()!=checkpoint or final.get('step')!=step:
            raise ValueError(f'Stage {stage} final checkpoint mismatch')
        if (marker.get('owner'),marker.get('stage'),marker.get('step'),marker.get('mode'))!=(OWNER,stage,step,mode):
            raise ValueError(f'Stage {stage} completion marker mismatch')
        if not marker.get('model_sha256') or marker['model_sha256']!=final.get('model_sha256'):
            raise ValueError(f'Stage {stage} checkpoint digest mismatch')
        if not (checkpoint/'model.pt').is_file():
            raise ValueError(f'Missing Stage {stage} model file')
    if Path(pipeline.get('student_checkpoint','')).resolve()!=expected:
        raise ValueError('Pipeline final student is not step2000 from this run')
    return expected


def verify_checkpoint(run, checkpoint, cfg):
    meta=model_metadata(checkpoint/'model.pt')
    if (meta['format'],meta['mode'],meta['step'])!=(FORMAT,'student',2000) or meta['config']!=cfg:
        raise ValueError('Expected this A1 Local16 student at step2000')
    if (cfg['ablation'],cfg['local_slots'],cfg['goal_slots'],cfg['interaction'])!=('A1',16,0,False):
        raise ValueError('A1 Local-only topology mismatch')
    teacher=meta['extra'].get('teacher_source') or {}
    expected_teacher=run/'stage_I/checkpoints/step_001000'
    if teacher.get('step')!=1000 or Path(teacher.get('path','')).resolve()!=expected_teacher:
        raise ValueError('Student teacher provenance mismatch')
    if teacher.get('model_sha256')!=read(expected_teacher/'checkpoint_complete.json').get('model_sha256'):
        raise ValueError('Student teacher digest mismatch')
    checksum=file_sha256(checkpoint/'model.pt')
    if checksum!=read(checkpoint/'checkpoint_complete.json')['model_sha256']:
        raise ValueError('Student file checksum mismatch')
    return checksum


def command(checkpoint, output):
    return [PYTHON,'-B',str(WORK/'evaluate.py'),'--checkpoint',str(checkpoint),'--protocol','full',
            '--output',str(output),'--device','cuda:0','--sim-python',SIM_PYTHON]


def progress(output):
    episodes=successes=tasks=0
    for suite in SUITES:
        for task in range(10):
            file=output/'tasks'/f'{suite}__task{task:02d}'/'episodes.checkpoint.jsonl'
            valid={}
            if file.exists():
                for line in file.read_text().splitlines():
                    try: row=json.loads(line)
                    except ValueError: continue  # A concurrent append may be unfinished.
                    if not row.get('error') and row.get('task_id')==task and 0<=row.get('episode_idx',-1)<50:
                        valid[row['episode_idx']]=row
            episodes+=len(valid);successes+=sum(bool(x['success']) for x in valid.values());tasks+=len(valid)==50
    return dict(episodes_completed=episodes,expected_episodes=2000,tasks_completed=tasks,total_tasks=40,
                successes=successes,success_rate=successes/episodes if episodes else None,
                current_task=read(output/'status.json'))


def complete(output, checksum):
    result=read(output/'summary.json')
    tasks=result.get('tasks',[])
    expected={(suite,task) for suite in SUITES for task in range(10)}
    return (result.get('status')=='complete' and result.get('episodes_completed')==2000
        and result.get('expected_episodes')==2000 and result.get('checkpoint_sha256')==checksum
        and len(tasks)==40 and {(x['suite'],x['task_id']) for x in tasks}==expected
        and all(x.get('complete') and x.get('episodes')==50 for x in tasks))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    plan=read(WORK/'run_plan.json');cfg=read(WORK/'config.json')
    run=(WORK/'runs'/plan['run_name']).resolve()
    output=WORK/'evaluation'/(run.name+'_full40x50_step2000')
    watch=WORK/'auto_eval'/run.name
    checkpoint=run/'stage_II/checkpoints/step_002000'
    manifest=dict(run=str(run),checkpoint=str(checkpoint),output=str(output),physical_gpu=5,
        protocol='full',tasks=40,episodes_per_task=50,expected_episodes=2000,poll_seconds=5,
        max_evaluation_attempts=3,training_files_modified=False,command=command(checkpoint,output))
    if args.once or not args.execute:
        print(json.dumps(dict(manifest,snapshot=snapshot(run),ready=ready_checkpoint(run) is not None,
                             starts_evaluation=False),indent=2));return
    watch.mkdir(parents=True,exist_ok=True)
    lock=(watch/'watch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=read(watch/'plan.json')
    if existing and existing!=manifest: raise ValueError('Watcher output belongs to another plan')
    atomic_json(watch/'plan.json',manifest)
    previous=None;last_print=0;child=None
    def update(phase,**details):
        nonlocal previous,last_print
        state=dict(phase=phase,watcher_pid=os.getpid(),run=str(run),output=str(output),**details)
        atomic_json(watch/'status.json',dict(state,updated_at=time.time()))
        if state!=previous or time.time()-last_print>=60:
            line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+json.dumps(state,ensure_ascii=False)
            print(line,flush=True)
            with (watch/'watcher.log').open('a') as f: f.write(line+'\n')
            previous=state;last_print=time.time()
    def interrupted(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupted)
    try:
        while True:
            ready=ready_checkpoint(run)
            if ready is not None: break
            current=snapshot(run)
            phase='training_failed_waiting_for_recovery' if current['pipeline_phase']=='failed' else 'waiting_for_training'
            update(phase,**current)
            time.sleep(5)
        update('verifying_step2000',**snapshot(run))
        checksum=verify_checkpoint(run,checkpoint,cfg)
        atomic_json(watch/'verified_checkpoint.json',dict(path=str(checkpoint),step=2000,sha256=checksum))
        if complete(output,checksum):
            update('completed',**progress(output));return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',CUDA_DEVICE_ORDER='PCI_BUS_ID',MUJOCO_EGL_DEVICE_ID='5',
            MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        for attempt in range(1,4):
            with (watch/'evaluation.log').open('a') as log:
                child=subprocess.Popen(command(checkpoint,output),cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    update('evaluating',attempt=attempt,evaluation_pid=child.pid,checkpoint_step=2000,**progress(output))
                    time.sleep(5)
                code=child.returncode;child=None
            if code==0 and complete(output,checksum):
                update('completed',attempt=attempt,**progress(output));return
            update('evaluation_retry_pending' if attempt<3 else 'evaluation_failed',attempt=attempt,exit_code=code,
                   log=str(watch/'evaluation.log'),**progress(output))
            if attempt<3: time.sleep(60)
        raise RuntimeError('Full evaluation failed after three resumable attempts; inspect evaluation.log')
    except BaseException:
        update('watcher_stopped_or_failed',error=traceback.format_exc())
        raise
    finally:
        # This child is the evaluator only. Training PIDs are never signaled.
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        lock.close()


if __name__=='__main__':main()
