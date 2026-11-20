"""Resume this A1 Stage I with its full state and preserve automatic handoff."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(WORK), str(WORK.parents[3])]
from a1_local.config_runtime import (load_config, physical_gpus, run_root, stage_root,
    checkpoint_path, stage_steps, atomic_json, code_identity)
from a1_local.preflight import plan, check_complete, check_teacher
from ablation.aba1.plus_common.environment import subprocess_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = load_config(WORK / 'config.json', 'A1')
    resume = Path(args.resume).resolve()
    if resume.parent != stage_root(cfg, 'I'):
        raise ValueError('Resume checkpoint must belong to this Stage I')
    meta, marker = check_complete(resume, require_resume=True)
    if meta['config'] != cfg or meta['extra'].get('stage') != 'I':
        raise ValueError('Resume config/stage mismatch')
    if not 0 < meta['step'] < stage_steps(cfg, 'I'):
        raise ValueError('Resume step must be before Stage I completion')
    if any((resume / name).stat().st_size != info['bytes'] for name, info in marker['files'].items()):
        raise ValueError('Resume artifact size mismatch')
    gpu = ','.join(map(str, physical_gpus(cfg)))
    command = [cfg['model_python'], '-B', '-m', 'a1_local.training',
        '--config', str(WORK / 'config.json'), '--stage', 'I', '--gpus', gpu,
        '--workers', '1', '--cpu-threads', '2', '--offload-saved-tensors',
        '--share-frozen-encoders', '--keep-models', '1', '--resume', str(resume), '--execute']
    spec = plan(cfg, 'I', require_sources=True)
    report = dict(plan=spec, command=command, resume_step=meta['step'],
                  resume_checkpoint=str(resume), full_state=True, training_started=args.execute)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    output = run_root(cfg)
    with (output / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = subprocess_environment(cfg, gpu, output / 'runtime_cache')
        env['PYTHONPATH'] = str(WORK) + os.pathsep + env['PYTHONPATH']
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['TORCH_GLOO_LAZY_INIT'] = '1'
        env['MUJOCO_EGL_DEVICE_ID'] = str(physical_gpus(cfg)[0])
        env['TMPDIR'] = '/tmp'
        atomic_json(output / 'pipeline_plan.json', dict(report, config=cfg, code_identity=code_identity()))
        atomic_json(output / 'pipeline_status.json', dict(phase='training', stage='I',
                    time=time.time(), command=command, resume_step=meta['step']))
        try:
            # Each worker verifies all checkpoint hashes inside the serialized loading lock.
            subprocess.run(command, env=env, cwd=WORK, check=True)
            final = checkpoint_path(cfg, 'I', stage_steps(cfg, 'I'))
            check_complete(final, verify_hash=True)
            check_teacher(final, cfg)
        except BaseException as error:
            atomic_json(output / 'pipeline_status.json', dict(phase='failed', stage='I',
                        error=repr(error), time=time.time(), resume_step=meta['step']))
            raise
        atomic_json(output / 'pipeline_status.json', dict(phase='completed', completed_stages=['I'],
                    teacher_step=stage_steps(cfg, 'I'), student_step=None, time=time.time()))
        print('Resumed Stage I completed; Teacher3000 is ready for automatic Stage II.', flush=True)


if __name__ == '__main__':
    main()
