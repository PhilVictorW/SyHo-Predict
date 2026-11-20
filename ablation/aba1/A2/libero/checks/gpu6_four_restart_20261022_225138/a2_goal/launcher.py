"""Run Goal-only Teacher 1000 -> Student 2000 -> LIBERO full 40x50."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from .config import WORK, ROOT, OWNER, load_plan, run_path, resource_report, atomic_json, source_manifest, file_sha256
from goal_local.eval_common import SIM_PYTHON, SUITES, model_metadata


def commands(plan, output):
    return {stage: [plan['model_python'], '-B', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc-per-node=3', '--max-restarts=0',
        '--log-dir', str(output/'logs'/('stage_'+stage)), '--tee', '3',
        '-m', 'a2_goal.training', '--stage', stage, '--experiment-root', str(output)] for stage in ['I','II']}


def evaluation_command(plan, output):
    checkpoint = output/'stage_II/checkpoints/step_002000'
    evaluation = WORK/'evaluation'/output.name/'full_40x50_step2000'
    return [plan['model_python'], '-B', str(WORK/'evaluate.py'), '--checkpoint', str(checkpoint),
        '--protocol', 'full', '--output', str(evaluation), '--device', 'cuda:0', '--sim-python', SIM_PYTHON]


def completed_stage(output, stage, cfg, expected_step):
    folder = output/('stage_'+stage)
    checkpoint = folder/'checkpoints'/f'step_{expected_step:06d}'
    final = json.loads((folder/'final.json').read_text())
    state = json.loads((folder/'status.json').read_text())
    marker = json.loads((checkpoint/'checkpoint_complete.json').read_text())
    mode = 'posterior' if stage == 'I' else 'student'
    if (state.get('phase'), state.get('step')) != ('completed', expected_step):
        raise ValueError('Stage has not completed its requested optimizer steps')
    if Path(final['path']).resolve() != checkpoint or final['step'] != expected_step:
        raise ValueError('Unexpected final checkpoint')
    if (marker.get('owner'), marker.get('stage'), marker.get('step'), marker.get('mode')) != (OWNER, stage, expected_step, mode):
        raise ValueError('Checkpoint ownership/step mismatch')
    checksum = file_sha256(checkpoint/'model.pt')
    if checksum != marker.get('model_sha256') or checksum != final.get('model_sha256'):
        raise ValueError('Checkpoint checksum mismatch')
    meta = model_metadata(checkpoint/'model.pt')
    if meta['mode'] != mode or meta['step'] != expected_step or meta['config'] != cfg:
        raise ValueError('Checkpoint metadata mismatch')
    if stage == 'II':
        teacher = meta['extra'].get('teacher_source') or {}
        expected = output/'stage_I/checkpoints/step_001000'
        teacher_marker = json.loads((expected/'checkpoint_complete.json').read_text())
        if teacher.get('step') != 1000 or Path(teacher.get('path','')).resolve() != expected:
            raise ValueError('Student must inherit this run\'s Teacher step1000')
        if teacher.get('model_sha256') != teacher_marker['model_sha256']:
            raise ValueError('Teacher provenance checksum mismatch')
    return final


def evaluation_complete(evaluation, checksum):
    summary = json.loads((evaluation/'summary.json').read_text())
    tasks = summary.get('tasks', [])
    expected = {(suite, task) for suite in SUITES for task in range(10)}
    if not (summary.get('status') == 'complete' and summary.get('episodes_completed') == 2000
        and summary.get('expected_episodes') == 2000 and summary.get('checkpoint_sha256') == checksum
        and len(tasks) == 40 and {(x['suite'],x['task_id']) for x in tasks} == expected
        and all(x.get('complete') and x.get('episodes') == 50 for x in tasks)):
        raise ValueError('Full evaluation has not completed all 40 x 50 episodes')
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    plan, cfg = load_plan()
    output = run_path(plan, args.output)
    report = dict(plan=plan, config=cfg, output=str(output), resources=resource_report(plan),
        commands=commands(plan,output), evaluation_command=evaluation_command(plan,output),
        starts_training=args.execute, resource_thresholds_enforced=False)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    queue_lock = (WORK/'queue.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'pipeline.json').exists() or any((output/('stage_'+s)).exists() for s in ['I','II']):
        raise FileExistsError('Preserve existing run; model exports cannot restore optimizer state.')
    if not Path(SIM_PYTHON).is_file():
        raise FileNotFoundError(SIM_PYTHON)
    atomic_json(output/'launch.json', report)
    atomic_json(output/'source_manifest.json', source_manifest())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, plan['physical_gpus'])), CUDA_DEVICE_ORDER='PCI_BUS_ID', A2_PIPELINE_RUN=str(output),
        PYTHONPATH=str(WORK)+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS=str(plan['cpu_threads_per_rank']), MKL_NUM_THREADS=str(plan['cpu_threads_per_rank']),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    stage = 'I'
    def status(phase, **details):
        state = dict(phase=phase, stage=stage, run=str(output), world_size=plan['world_size'], physical_gpus=plan['physical_gpus'],
                     time=time.time(), **details)
        atomic_json(output/'pipeline.json', state)
        atomic_json(WORK/'queue_status.json', state)
    try:
        for stage in ['I','II']:
            status('training')
            result = subprocess.run(commands(plan,output)[stage], cwd=WORK, env=env)
            if result.returncode:
                raise RuntimeError(f'Stage {stage} failed (exit {result.returncode}); inspect rank logs')
            final = completed_stage(output, stage, cfg, plan['stage_'+stage]['steps'])
        stage = 'evaluation'
        evaluation = WORK/'evaluation'/output.name/'full_40x50_step2000'
        env.update(CUDA_VISIBLE_DEVICES='1', MUJOCO_EGL_DEVICE_ID='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
        for attempt in range(1, plan['evaluation']['max_attempts']+1):
            status('evaluating', attempt=attempt, student_checkpoint=final['path'], evaluation=str(evaluation))
            with (output/'evaluation.log').open('a') as log:
                result = subprocess.run(evaluation_command(plan,output), cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                summary = evaluation_complete(evaluation, final['model_sha256'])
                break
            if attempt == plan['evaluation']['max_attempts']:
                raise RuntimeError('Full evaluation failed after three resumable attempts; inspect evaluation.log')
        status('completed', student_checkpoint=final['path'], evaluation=str(evaluation),
               episodes=summary['episodes_completed'], successes=summary['successes'], success_rate=summary['success_rate'])
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        lock.close()
        queue_lock.close()
