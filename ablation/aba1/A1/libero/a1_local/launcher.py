"""Launch a fixed single-node job on the configured physical GPUs."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from .config import WORK, ROOT, load_plan, run_path, resource_report, atomic_json, source_manifest, physical_gpus
from .checkpoints import verify_teacher, verify_student, seed_student_checkpoint


def commands(plan, output, world=None):
    world = plan['world_size'] if world is None else world
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    return {stage: [plan['model_python'], '-B', '-m', 'torch.distributed.run',
        '--rdzv-backend=static', '--node-rank=0', '--master-addr=127.0.0.1', f'--master-port={port}',
        '--nnodes=1', f'--nproc-per-node={world}', '--max-restarts=0', '--log-dir', str(output/'logs'/('stage_'+stage+f'_ranks{world}')),
        '--tee', '3', '-m', 'a1_local.training', '--stage', stage, '--experiment-root', str(output)] for stage in ['I','II']}


def is_cuda_oom(output, stage):
    texts = [p.read_text(errors='replace').lower() for p in (output/('stage_'+stage)).glob('failure_rank_*.json')]
    return any(token in text for text in texts for token in ('out of memory', 'outofmemoryerror', 'cublas_status_alloc_failed'))



def validate_start(output, cfg, start_stage):
    """Allow Stage II in a fresh run seeded with a verified completed teacher."""
    if (output/'pipeline.json').exists() or (output/'stage_II').exists():
        raise FileExistsError('Preserve existing run; start in a fresh run directory.')
    if start_stage == 'I':
        if (output/'stage_I').exists():
            raise FileExistsError('Stage I already exists; do not overwrite it.')
        return None
    folder = output/'stage_I'
    checkpoint = (folder/'checkpoints/step_001000').resolve()
    status = json.loads((folder/'status.json').read_text())
    final = json.loads((folder/'final.json').read_text())
    marker = json.loads((checkpoint/'checkpoint_complete.json').read_text())
    if status.get('phase') != 'completed' or status.get('step') != 1000:
        raise ValueError('Stage II requires completed Stage I step1000')
    if Path(final.get('path', '')).resolve() != checkpoint:
        raise ValueError('Stage I final path does not match this run')
    if marker.get('mode') != 'posterior' or any(final.get(k) != marker.get(k) for k in ('owner','stage','step','mode','model_sha256')):
        raise ValueError('Stage I final record and completion marker differ')
    return verify_teacher(checkpoint, cfg)


def main():
    p = argparse.ArgumentParser(__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    p.add_argument('--output')
    p.add_argument('--start-stage', choices=['I','II'], default='I')
    p.add_argument('--resume-student', type=Path)
    args = p.parse_args()
    if args.resume_student and args.start_stage != 'II':
        p.error('--resume-student requires --start-stage II')
    plan, cfg = load_plan()
    plan.pop('resume_student', None)
    output = run_path(plan, args.output)
    stages = ['I','II'] if args.start_stage == 'I' else ['II']
    report = dict(start_stage=args.start_stage, stages=stages, plan=plan, config=cfg, output=str(output), resources=resource_report(plan),
        commands=commands(plan,output), starts_training=args.execute, resource_thresholds_enforced=False,
        oom_fallback=plan['oom_fallback_world_sizes'])
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    queue_lock = (WORK/'queue.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    teacher = validate_start(output, cfg, args.start_stage)
    if teacher is not None:
        report['imported_teacher'] = teacher
    if args.resume_student:
        source = verify_student(args.resume_student, cfg, teacher, plan['stage_II'])
        plan['resume_student'] = seed_student_checkpoint(source, output/'stage_II')
        atomic_json(output/'stage_II/resume_source.json', plan['resume_student'])
        atomic_json(output/'stage_II/status.json', dict(phase='loading', stage='II', step=source['step'],
            target_steps=plan['stage_II']['steps'], updated_at=time.time()))
    atomic_json(output/'launch.json', report)
    atomic_json(output/'run_plan.json', plan)
    atomic_json(output/'source_manifest.json', source_manifest())
    devices = physical_gpus(plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, devices)), CUDA_DEVICE_ORDER='PCI_BUS_ID', A1_PIPELINE_RUN=str(output),
        GLOO_SOCKET_IFNAME='lo',
        PYTHONPATH=str(WORK)+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS=str(plan['cpu_threads_per_rank']), MKL_NUM_THREADS=str(plan['cpu_threads_per_rank']),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    stage, world = args.start_stage, plan['world_size']
    try:
        for stage in stages:
            while True:
                command = report['commands'][stage] if world == plan['world_size'] else commands(plan,output,world)[stage]
                state=dict(phase='training',stage=stage,world_size=world,physical_gpu=plan['physical_gpu'],
                    physical_gpus=devices,command=command,run=str(output),time=time.time())
                atomic_json(WORK/'queue_status.json',state)
                atomic_json(output/'pipeline.json',state)
                result=subprocess.run(command,cwd=WORK,env=env)
                if result.returncode == 0:
                    break
                if world-1 not in plan['oom_fallback_world_sizes'] or not is_cuda_oom(output,stage):
                    raise RuntimeError(f'Stage {stage}, ranks={world}, failed (exit {result.returncode}); inspect rank logs')
                # Failed attempts never count toward the successful stage's step budget.
                archive=output/'failed_attempts'/f'stage_{stage}_ranks{world}_{int(time.time())}'
                archive.parent.mkdir(parents=True,exist_ok=True)
                (output/('stage_'+stage)).rename(archive)
                world-=1
                print(json.dumps(dict(event='cuda_oom_reduce_ranks',stage=stage,new_world_size=world,
                    restart_stage_from_initialization=True,failed_attempt=str(archive))),flush=True)
            final=json.loads((output/('stage_'+stage)/'final.json').read_text())
            state=json.loads((output/('stage_'+stage)/'status.json').read_text())
            if state['phase'] != 'completed' or final['step'] != plan['stage_'+stage]['steps']:
                raise RuntimeError('Stage completion contract failed')
        state=dict(phase='completed',student_checkpoint=final['path'],world_size=world,physical_gpu=plan['physical_gpu'],physical_gpus=devices,automatic_evaluation=False,time=time.time())
        atomic_json(output/'pipeline.json',state)
        atomic_json(WORK/'queue_status.json',state)
    except BaseException:
        state=dict(phase='failed',stage=stage,world_size=world,physical_gpu=plan['physical_gpu'],physical_gpus=devices,error=traceback.format_exc(),time=time.time())
        atomic_json(output/'pipeline.json',state)
        atomic_json(WORK/'queue_status.json',state)
        raise
    finally:
        lock.close();queue_lock.close()
