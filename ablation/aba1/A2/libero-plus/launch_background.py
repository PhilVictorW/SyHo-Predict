"""Launch this A2 training chain under nohup with isolated files and PID receipts."""
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
from a2_local.config_runtime import load_config, physical_gpus, stage_spec, atomic_json
from a2_local.preflight import plan, resources

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = load_config(WORK / 'config.json', 'A2')
    settings = cfg['a2_training']
    if physical_gpus(cfg) != [6] or settings['processes_per_gpu'] != 8:
        raise ValueError('This run requires physical GPU6, eight ranks')
    summary = dict(plans=[plan(cfg, stage) for stage in ('I', 'II')], resources=resources(cfg),
                   evaluation_tasks=10030, evaluation_workers=8, training_started=False)
    if not args.execute:
        print(json.dumps(summary, indent=2)); return
    os.chdir(WORK)
    logs = WORK / 'logs'; logs.mkdir(exist_ok=True)
    with (logs / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (WORK / 'protocol_plus_v1/seed_2026/I/arguments.json').exists() or (logs / 'launch_record.json').exists():
            raise RuntimeError('Existing run found; use explicit recovery, not a duplicate launch')
        for process in Path('/proc').iterdir():
            if not process.name.isdigit(): continue
            try: command = (process / 'cmdline').read_bytes().split(b'\0')
            except OSError: continue
            if b'a2_local.training' in command and str(WORK / 'config.json').encode() in command:
                raise RuntimeError('An A2 training process already exists: ' + process.name)
        if min(g['free_mib'] for g in summary['resources']['gpus']) < 56000:
            raise RuntimeError('Need at least 56000 MiB free on GPU6 for eight ranks')
        if summary['resources']['disk_free_gib'] < 75:
            raise RuntimeError('Need 75 GiB for teacher retention and atomic full checkpoints')
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        def launch(command, name):
            with (logs / name).open('a') as stream, open(os.devnull) as stdin:
                return subprocess.Popen(command, cwd=WORK, env=environment, stdin=stdin,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        training = launch(['nohup', 'bash', str(WORK / 'run_logged.sh')], 'nohup_gpu6.log')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record = dict(time=time.time(), boot_id=boot, launcher_pid=training.pid,
            launch_method='nohup', stage='I', physical_gpus=physical_gpus(cfg), world_size=8,
            processes_per_gpu=8, rank_to_gpu=[6]*8, stage_I_steps=3000, stage_II_steps=2500,
            effective_batch=32, accumulation_per_rank=stage_spec(cfg, 'I')['accumulation_per_rank'],
            save_every=50, keep_checkpoints_per_stage=1, evaluation_tasks=10030)
        atomic_json(logs / 'launch_record.json', record)
        (logs / 'nohup_training.pid').write_text(str(training.pid) + '\n')
        continuation = launch(['nohup', cfg['model_python'], '-B', '-u',
            str(WORK / 'continue_training.py'), '--execute', '--poll-seconds', '10'], 'auto_continuation.log')
        (logs / 'auto_continuation.pid').write_text(str(continuation.pid) + '\n')
        atomic_json(logs / 'background_launch.json', dict(record, continuation_pid=continuation.pid))
        print(json.dumps(dict(record, continuation_pid=continuation.pid), indent=2), flush=True)
if __name__ == '__main__': main()
