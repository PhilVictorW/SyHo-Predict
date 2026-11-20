"""Persistent simulation process with transactional claims and per-episode results."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback
from common import *

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(REPO))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(CHECKPOINTS), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--smoke-job', type=int)
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero
    assert Path(libero.libero.__file__).resolve().is_relative_to(REPO)
    from examples.LIBERO.eval_files import libero_eval_core as core
    from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter
    from goal_local.eval_client import StudentClient

    folder = RUN / (f'preflight/{args.variant}' if args.smoke_job is not None else args.variant)
    status_path = folder / 'workers' / f'{args.worker}.json'
    current = {}
    def status(phase, **extra):
        current.update(extra)
        atomic(status_path, dict(phase=phase, pid=os.getpid(), worker=args.worker,
                                gpu=args.gpu, variant=args.variant, updated=time.time(), **current))

    class Client(StudentClient):
        def step(self, *a, **kw):
            if kw.get('step', 0) % 8 == 0:
                status('evaluating', action_step=kw.get('step', 0))
            result = super().step(*a, **kw)
            if args.variant == 'plus':
                gripper = np.asarray(result['raw_action']['open_gripper'])
                if not np.isin(gripper, [0., 1.]).all():
                    raise ValueError('Expected binarized Plus gripper')
                result['raw_action']['open_gripper'] = 1. - gripper
            return result

    client = Client(CHECKPOINTS[args.variant], args.url)
    def build_env(task, resolution, seed):
        env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=resolution,
                                 camera_widths=resolution, render_gpu_device_id=args.gpu)
        env.seed(seed)
        actual = ' '.join(str(env.language_instruction).split()).lower()
        if actual != task.instruction:
            env.close()
            raise ValueError(f'BDDL instruction mismatch: {actual!r} vs {task.instruction!r}')
        return env, task.instruction

    adapter = dataclasses.replace(get_benchmark_adapter('libero'), variant_name='libero_pro',
                                  build_env=build_env, get_max_steps=lambda suite: MAX_STEPS[suite])
    from types import SimpleNamespace
    manifest = read(RUN / 'manifest.json')
    db = None
    if args.smoke_job is None:
        db = sqlite3.connect(folder / 'queue.sqlite', timeout=60, isolation_level=None)
        db.execute('PRAGMA busy_timeout=60000')
    status('ready')
    while True:
        if (folder / 'workers' / f'{args.worker}.stop').exists():
            break
        if args.smoke_job is not None:
            job = manifest['jobs'][args.smoke_job]
        else:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id,payload FROM jobs WHERE gpu=? AND state='pending' ORDER BY id LIMIT 1", (args.gpu,)).fetchone()
            if row is None:
                row = db.execute("SELECT id,payload FROM jobs WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                db.execute('COMMIT')
                break
            job_id, payload = row
            job = json.loads(payload)
            db.execute("UPDATE jobs SET state='running',worker=?,gpu=?,attempts=attempts+1,started=? WHERE id=?", (args.worker,args.gpu,time.time(),job_id))
            db.execute('COMMIT')
        status('evaluating', job_id=job['id'], suite=job['suite'], category=job['category'],
               task_id=job['task_id'], episode_idx=job['episode_idx'], action_step=0, started=time.time())
        try:
            states = torch.load(job['init'], map_location='cpu', weights_only=False)
            client.episode_seed = job['noise_seed']
            eval_args = core.EvalArgs(pretrained_path=CHECKPOINTS[args.variant], task_suite_name=job['suite'],
                                     task_id=job['task_id'], num_trials_per_task=1, num_workers=1,
                                     seed=0, num_steps_wait=10, save_videos=False,
                                     video_out_path=str(folder), eval_action_chunk_len=8)
            begin = time.monotonic()
            result = core._run_single_episode(args=eval_args, adapter=adapter, task=SimpleNamespace(**job),
                        task_id=job['task_id'], episode_idx=job['episode_idx'],
                        init_state=states[job['episode_idx']], client_model=client,
                        run_output_dir=folder, env_seed=0)
            if result.error:
                raise RuntimeError(result.error)
            value = dict(dataclasses.asdict(result), job_id=job['id'], suite=job['suite'],
                         category=job['category'], name=job['name'], instruction=job['instruction'],
                         noise_seed=job['noise_seed'], worker=args.worker, gpu=args.gpu,
                         variant=args.variant, checkpoint=CHECKPOINTS[args.variant],
                         elapsed_seconds=time.monotonic()-begin, completed_at=time.time())
            atomic(folder / 'episodes' / f'{job["id"]:05d}.json', value)
            if db:
                db.execute("UPDATE jobs SET state='done',success=?,finished=?,error=NULL WHERE id=?", (int(result.success),time.time(),job['id']))
            print(json.dumps(dict(event='episode', **value)), flush=True)
            status('ready', last_success=result.success)
        except Exception as exc:
            traceback.print_exc()
            atomic(folder / 'errors' / f'{args.worker}_{job["id"]}_{int(time.time())}.json',
                   dict(job=job, error=repr(exc), time=time.time(), worker=args.worker))
            status('error', error=repr(exc))
            if db:
                count = db.execute('SELECT attempts FROM jobs WHERE id=?', (job['id'],)).fetchone()[0]
                db.execute("UPDATE jobs SET state=?,error=? WHERE id=?", ('failed' if count>=3 else 'pending',repr(exc),job['id']))
            else:
                raise
            # Let the controller recover a broken renderer/server in a fresh process.
            return 2
        if args.smoke_job is not None:
            break
    status('finished')
    client.close()
    if db:
        db.close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
