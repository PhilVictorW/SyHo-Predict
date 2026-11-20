"""Bounded eight-task/mode pilot against recorded Round10 compatible trajectories."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'robotwin/round11/online_pilot_500'
PY = '/home/wangchaoyang/envs/LaWAM/bin/python'
SIMPY = '/home/wangchaoyang/envs/RoboTwin4LaWAM/bin/python'
SIM = '/mnt/hdd/wangchaoyang/results/LaWAM/runtime/RoboTwin'
PORT = 26811


def read(p):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return {}


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def task_json(job, name):
    paths = list(job.glob('*/*/tasks/*/'+name))
    return read(paths[0]) if paths else {}


def stop_child(p):
    # Only Popen children created by THIS pilot are ever terminated.
    if p is None: return
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: p.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=12)


def reference_jobs():
    old = read(ROOT/'robotwin/round10/candidate_online/status.json')
    rows = {r['key']:r for r in old['results'] if r['status'] == 'completed'}
    modes = ['demo_clean', 'demo_randomized']
    tasks = sorted({r['task'] for r in rows.values()
                    if all(r['task']+'__'+m in rows for m in modes)})[:4]
    assert len(tasks) == 4
    plan = []
    for task in tasks:
        for mode in modes:
            row = rows[task+'__'+mode]
            episodes = sorted(row['summary']['episodes'], key=lambda e:e['episode_id'])[:5]
            requests = [json.loads(x) for x in (Path(row['job'])/'requests.jsonl').read_text().splitlines()]
            first = {r['episode_seed']:r for r in requests if r['query_index'] == 0}
            assert len(episodes) == 5 and all(e['seed'] in first for e in episodes)
            plan.append(dict(task=task, mode=mode, key=row['key'], reference_job=row['job'],
                             episodes=episodes, first_requests={str(e['seed']):first[e['seed']] for e in episodes}))
    return plan


def main():
    global OUT, PORT
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidate', choices=['query_only', 'query_gate', 'action_adapter'], required=True)
    a = parser.parse_args()
    if a.candidate == 'action_adapter':
        OUT = ROOT/'robotwin/round11/online_adapter_200'
        PORT = 26812
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'manifest.json').exists(), 'Existing online pilot must not be overwritten'
    jobs = reference_jobs()
    manifest = dict(candidate=a.candidate, physical_gpu=2, jobs=jobs, target_episodes=40,
        selection='first four alphabetical tasks with completed Round10 results in both modes; first five episode_ids, irrespective of outcome',
        comparator='immutable Round10 compatible fixed-query initialization, NOT original LaWAM',
        pairing='replay exact episode seed, instruction, per-query noise; verify first-observation hashes',
        stall_seconds=1800, timeout='after weights ready, skip job after 30min without completed episode',
        online_limitation='small screening subset; recorded-reference comparison, not contemporaneous paired full benchmark')
    write(OUT/'manifest.json', manifest)
    state = dict(pid=os.getpid(), phase='loading', completed_jobs=0, total_jobs=8, target_episodes=40)
    results = []
    server = child = None
    def update(**kw):
        state.update(kw)
        write(OUT/'status.json', dict(state, updated_at=time.time(), results=results))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('ROBOTWIN_', 'PAIR_')): env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='2', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT), ROBOTWIN_PATH=SIM, ROBOTWIN_NUM_SLOTS='1', ROBOTWIN_TEST_NUM='5',
        ROBOTWIN_SAVE_VIDEO='0', ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN='1',
        ROBOTWIN_SHARED_WARP_CACHE_DIR=str(OUT/'cache/warp'))
    (OUT/'cache/warp').mkdir(parents=True, exist_ok=True)
    cfg = read(ROOT/'robotwin/configs/stage_I.json')
    try:
        with socket.socket() as sock: sock.bind(('127.0.0.1', PORT))
        update()
        with (OUT/'server.log').open('a') as log:
            server = subprocess.Popen([PY, '-u', '-m', 'robotwin.round11.online_server', '--candidate', a.candidate,
                '--port', str(PORT), '--output', str(OUT)], cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while True:
            if server.poll() is not None: raise RuntimeError('Student server exited during loading')
            ready = read(OUT/'ready.json')
            if ready.get('pid') == server.pid and ready.get('ready'):
                try:
                    with socket.create_connection(('127.0.0.1', PORT), timeout=1): break
                except OSError: pass
            if time.time()-started > 1800: raise TimeoutError('Student server startup exceeded 30min')
            update(server_pid=server.pid)
            time.sleep(5)
        for item in jobs:
            job = OUT/'jobs'/item['key']
            job.mkdir(parents=True)
            instructions = {k:r['instruction'] for k,r in item['first_requests'].items()}
            write(job/'instructions.json', instructions)
            ee = dict(env, PAIR_TASK=item['task'], PAIR_MODE=item['mode'], PAIR_ARM='compatible',
                ROUND11_REFERENCE_INSTRUCTIONS=str(job/'instructions.json'), ROBOTWIN_EVAL_ROOT=str(job),
                ROBOTWIN_RUN_TAG='round11_student', ROBOTWIN_FIXED_SEEDS=','.join(str(e['seed']) for e in item['episodes']),
                PYTHONPATH=f'{SIM}:{ROOT}:{ROOT}/examples/Robotwin/eval_files')
            cmd = [SIMPY, str(ROOT/'robotwin/round11/online_bridge.py'), '--config',
                str(ROOT/'examples/Robotwin/eval_files/deploy_policy.yml'), '--overrides',
                '--task_name', item['task'], '--task_config', item['mode'], '--ckpt_setting', 'round11_student',
                '--seed', '0', '--instruction_type', 'seen', '--policy_name', 'model2robotwin_interface',
                '--host', '127.0.0.1', '--port', str(PORT), '--policy_ckpt_path', cfg['baseline_checkpoint'],
                '--replan_steps', '36', '--action_ensemble', '0', '--action_ensemble_alpha', '0']
            with (job/'bridge.log').open('a') as log:
                child = subprocess.Popen(cmd, cwd=SIM, env=ee, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            count, last, timed = 0, time.time(), False
            while child.poll() is None:
                if server.poll() is not None: raise RuntimeError('Student server exited during evaluation')
                progress = task_json(job, 'progress.json')
                if int(progress.get('completed_episodes', 0)) > count:
                    count, last = int(progress['completed_episodes']), time.time()
                update(phase='evaluating', active=item['key'], bridge_pid=child.pid,
                       progress=progress, no_episode_progress_seconds=time.time()-last)
                if time.time()-last >= 1800:
                    timed = True
                    stop_child(child)
                    break
                time.sleep(5)
            sm = task_json(job, 'summary.json') or task_json(job, 'episodes.partial.json')
            rc = child.poll()
            stop_child(child)
            child = None
            reference = {e['seed']:e for e in item['episodes']}
            requests_path = job/'requests.jsonl'
            requests = [json.loads(x) for x in requests_path.read_text().splitlines()] if requests_path.exists() else []
            first = {r['episode_seed']:r for r in requests if r['query_index'] == 0}
            pairs = []
            for e in sm.get('episodes', []):
                seed = e['seed']
                old, new = item['first_requests'].get(str(seed), {}), first.get(seed, {})
                matched = bool(old and new and seed in reference and all(old[k] == new[k]
                    for k in ['instruction', 'noise_seed', 'image_sha256']))
                pairs.append(dict(seed=seed, matched=matched, candidate=bool(e['success']),
                                  reference=bool(reference[seed]['success']) if seed in reference else None))
            row = dict(key=item['key'], exit_code=rc, status='skipped_timeout' if timed else
                       'completed' if rc == 0 and len(sm.get('episodes', [])) == 5 else 'incomplete_error', pairs=pairs)
            write(job/'result.json', row)
            results.append(row)
            matched = [p for r in results for p in r['pairs'] if p['matched']]
            update(completed_jobs=len(results), matched_episodes=len(matched),
                   candidate_successes=sum(p['candidate'] for p in matched),
                   reference_successes=sum(p['reference'] for p in matched))
        write(OUT/'results.json', dict(completed=True, manifest=manifest, results=results, summary=state))
        update(phase='completed', active=None)
    except BaseException as exc:
        update(phase='failed', error=repr(exc))
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        stop_child(child)
        stop_child(server)


if __name__ == '__main__': main()
