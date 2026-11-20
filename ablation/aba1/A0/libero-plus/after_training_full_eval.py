"""Verify step1000, remove step750, then run the full adaptive evaluation in tmux."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(WORK.parents[3])]
from a0_single_latent.config import load_config,atomic_json
from a0_single_latent.deployment import validate_checkpoint
from a0_single_latent.evaluation import validate_final_budget

RUN=WORK/'runs/single_latent16_dino_step1000_2gpu07_20261019'
STATE=WORK/'workflow_full_eval.json'
AUDIT=RUN/'final_checkpoint_handoff'
LOG=WORK/'logs/adaptive_full_eval.log'

def process_running(pid):
    try:
        text=Path('/proc',str(pid),'stat').read_text()
        return text[text.rfind(')')+2:].split()[0] not in ('Z','X')
    except OSError:return False

def state(phase,**extra):
    row=dict(phase=phase,pid=os.getpid(),time=time.time(),run=str(RUN),
        target_step=1000,auto_gpu_count=4,max_workers_per_gpu=3,log=str(LOG),**extra)
    atomic_json(STATE,row);print(json.dumps(row),flush=True)

def digest(path):
    h=hashlib.sha256();done=0;reported=0;size=path.stat().st_size
    state('verifying_step1000',file=path.name,verified_bytes=0,file_bytes=size)
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):
            h.update(chunk);done+=len(chunk)
            if done-reported>=1024**3:
                state('verifying_step1000',file=path.name,verified_bytes=done,file_bytes=size);reported=done
    return h.hexdigest()

def verify_final(final,cfg):
    checkpoint=Path(final['path']).resolve()
    expected=RUN/'checkpoints/epoch_0000_step_000001000'
    if checkpoint!=expected or final['step']!=1000:raise ValueError('Unexpected final checkpoint')
    meta,marker=validate_checkpoint(checkpoint,cfg)
    validate_final_budget(meta,cfg)
    if marker['world_size']!=2 or marker['cursor']!=32000:raise ValueError('Final checkpoint counters mismatch')
    if set(marker['files'])!={'rank_0.pt','rank_1.pt','model.pt'}:raise ValueError('Incomplete recovery files')
    verified={}
    for name in ('model.pt','rank_0.pt','rank_1.pt'):
        path=checkpoint/name;info=marker['files'][name]
        if not path.is_file():raise FileNotFoundError(path)
        if 'bytes' in info and path.stat().st_size!=info['bytes']:raise ValueError('Checkpoint size mismatch: '+name)
        sha=digest(path)
        if sha!=info['sha256']:raise ValueError('Checkpoint checksum mismatch: '+name)
        verified[name]=dict(bytes=path.stat().st_size,sha256=sha)
    if verified['model.pt']['sha256']!=final['model_sha256']:raise ValueError('Final model pointer hash mismatch')
    proof=dict(complete=True,step=1000,checkpoint=str(checkpoint),files=verified,time=time.time())
    atomic_json(AUDIT/'step1000_verified.json',proof)
    return checkpoint

def delete_old(checkpoint):
    proof=json.loads((AUDIT/'step1000_verified.json').read_text())
    if not proof['complete'] or Path(proof['checkpoint'])!=checkpoint:raise ValueError('Missing final verification')
    old=RUN/'checkpoints/epoch_0000_step_000000750'
    if not old.exists():return
    if old.is_symlink() or old.resolve().parent!=checkpoint.parent:raise ValueError('Unsafe old checkpoint path')
    marker=json.loads((old/'checkpoint_complete.json').read_text())
    if marker['step']!=750 or not marker['complete']:raise ValueError('Wrong checkpoint to delete')
    latest=json.loads((RUN/'latest.json').read_text())
    if Path(latest['path']).resolve()!=checkpoint or latest['step']!=1000:raise ValueError('Latest is not step1000')
    record=dict(path=str(old),authorized_by_user=True,verified_final=str(checkpoint),old_marker=marker,time=time.time())
    atomic_json(AUDIT/'delete_step750.json',record)
    state('deleting_step750',final_checkpoint=str(checkpoint))
    shutil.rmtree(old)
    atomic_json(AUDIT/'delete_step750.json',dict(record,deleted=True,completed_at=time.time()))

def main():
    AUDIT.mkdir(exist_ok=True)
    with (AUDIT/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=load_config(WORK/'config.json')
        handoff=json.loads((WORK/'after_training_handoff.json').read_text())
        state('waiting_for_step1000')
        while not (RUN/'final.json').exists():
            status=json.loads((RUN/'status.json').read_text())
            if status.get('phase')=='failed':raise RuntimeError('Training failed: '+str(status.get('error')))
            if not any(process_running(pid) for pid in handoff['rank_pids']):
                raise RuntimeError('Training stopped without a final checkpoint')
            time.sleep(10)
        state('waiting_for_training_exit')
        while any(process_running(pid) for pid in
                  [handoff['training_launcher_pid'],handoff['torchrun_pid'],*handoff['rank_pids']]):
            time.sleep(3)
        # Retire the paused old controller only after its training children exit.
        old=handoff['old_controller_pid']
        if process_running(old):
            cmd=Path('/proc',str(old),'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(WORK/'resume_step750_gpu02_staggered.py') not in cmd:raise ValueError('Controller PID changed')
            os.kill(old,signal.SIGKILL)
        final=json.loads((RUN/'final.json').read_text())
        checkpoint=verify_final(final,cfg)
        delete_old(checkpoint)
        state('starting_adaptive_full_evaluation',final_checkpoint=str(checkpoint))
        with LOG.open('a') as stream:
            subprocess.run([cfg['model_python'],'-B','-u',str(WORK/'adaptive_full_eval.py'),
                '--config',str(WORK/'config.json'),'--checkpoint',str(checkpoint),'--scope','full',
                '--auto-gpus','4','--max-workers-per-gpu','3','--execute'],
                cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,check=True)
        state('completed',final_checkpoint=str(checkpoint))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('failed',error=repr(error))
        raise
