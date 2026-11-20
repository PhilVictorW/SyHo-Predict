#!/usr/bin/env python3
"""Durable train -> final step500 quick -> full pipeline, scoped to A0/libero."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
from a0_single_latent.config import atomic_json,load_config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--resume');args=parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):raise ValueError('Invalid run id')
    cfg=load_config(WORK/'config.json')
    run=WORK/'runs'/args.run_id;run.mkdir(parents=True,exist_ok=True)
    with (run/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        def stage(name,cmd,log,environment):
            atomic_json(run/'pipeline_status.json',dict(phase=name,pid=os.getpid(),time=time.time(),command=cmd))
            with log.open('a') as output:
                result=subprocess.run(cmd,cwd=WORK,env=environment,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:
                if name=='training':
                    state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
                    state.update(phase='failed',returncode=result.returncode,time=time.time(),log=str(log))
                    atomic_json(run/'status.json',state)
                raise RuntimeError(f'{name} failed: exit {result.returncode}; see {log}')
        try:
            if not (run/'final.json').is_file():
                cmd=['bash','train.sh','--output',str(run)]
                if args.resume:cmd+=['--resume',args.resume]
                stage('training',cmd,run/'training.log',env)
            final=json.loads((run/'final.json').read_text())
            if final['step']!=cfg['total_steps'] or final['seen_anchors']!=cfg['total_steps']*cfg['global_batch']:raise ValueError('Final training budget mismatch')
            evaluation=WORK/'evaluation'/args.run_id;evaluation.mkdir(parents=True,exist_ok=True)
            env.update(CUDA_VISIBLE_DEVICES=str(cfg['physical_gpus'][0]), MUJOCO_EGL_DEVICE_ID=str(cfg['physical_gpus'][0]))
            for protocol,suffix in [('quick','quick_40x10'),('full','full_40x50')]:
                stage(protocol,['bash','evaluate_parallel.sh','--checkpoint',final['path'],'--protocol',protocol,'--output',str(evaluation/suffix),'--gpus','1,1,1,1'],evaluation/(protocol+'.log'),env)
            atomic_json(run/'pipeline_status.json',dict(phase='completed',time=time.time(),final=final))
        except BaseException as error:
            atomic_json(run/'pipeline_status.json',dict(phase='failed',time=time.time(),error=repr(error)));raise
if __name__=='__main__':main()
