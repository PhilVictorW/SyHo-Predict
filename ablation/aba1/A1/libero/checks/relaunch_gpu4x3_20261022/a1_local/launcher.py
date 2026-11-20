"""Start on GPU 5 immediately; reduce ranks after an actual CUDA OOM."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from .config import WORK, ROOT, load_plan, run_path, resource_report, atomic_json, source_manifest


def commands(plan, output, world=4):
    return {stage: [plan['model_python'], '-B', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', f'--nproc-per-node={world}', '--max-restarts=0', '--log-dir', str(output/'logs'/('stage_'+stage+f'_ranks{world}')),
        '--tee', '3', '-m', 'a1_local.training', '--stage', stage, '--experiment-root', str(output)] for stage in ['I','II']}


def is_cuda_oom(output, stage):
    texts = [p.read_text(errors='replace').lower() for p in (output/('stage_'+stage)).glob('failure_rank_*.json')]
    return any(token in text for text in texts for token in ('out of memory', 'outofmemoryerror', 'cublas_status_alloc_failed'))


def main():
    p = argparse.ArgumentParser(__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    plan, cfg = load_plan()
    output = run_path(plan, args.output)
    report = dict(plan=plan, config=cfg, output=str(output), resources=resource_report(plan),
        commands=commands(plan,output), starts_training=args.execute, resource_thresholds_enforced=False,
        oom_fallback=[4,3,2,1])
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    queue_lock = (WORK/'queue.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'pipeline.json').exists() or any((output/('stage_'+s)).exists() for s in ['I','II']):
        raise FileExistsError('Preserve existing run; weight-only exports do not restore Adam state.')
    atomic_json(output/'launch.json', report)
    atomic_json(output/'source_manifest.json', source_manifest())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', CUDA_DEVICE_ORDER='PCI_BUS_ID', A1_PIPELINE_RUN=str(output),
        PYTHONPATH=str(WORK)+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS=str(plan['cpu_threads_per_rank']), MKL_NUM_THREADS=str(plan['cpu_threads_per_rank']),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    stage, world = 'I', 4
    try:
        for stage in ['I','II']:
            while True:
                state=dict(phase='training',stage=stage,world_size=world,run=str(output),time=time.time())
                atomic_json(WORK/'queue_status.json',state)
                atomic_json(output/'pipeline.json',state)
                result=subprocess.run(commands(plan,output,world)[stage],cwd=WORK,env=env)
                if result.returncode == 0:
                    break
                if world == 1 or not is_cuda_oom(output,stage):
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
        state=dict(phase='completed',student_checkpoint=final['path'],world_size=world,automatic_evaluation=False,time=time.time())
        atomic_json(output/'pipeline.json',state)
        atomic_json(WORK/'queue_status.json',state)
    except BaseException:
        state=dict(phase='failed',stage=stage,world_size=world,error=traceback.format_exc(),time=time.time())
        atomic_json(output/'pipeline.json',state)
        atomic_json(WORK/'queue_status.json',state)
        raise
    finally:
        lock.close();queue_lock.close()
