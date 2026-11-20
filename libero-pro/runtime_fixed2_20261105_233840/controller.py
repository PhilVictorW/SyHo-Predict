"""Fixed two inference replicas per GPU, with durable queues and pair admission."""
import concurrent.futures
from contextlib import contextmanager
import fcntl
import signal
import sqlite3
import subprocess
import sys
import threading
import traceback
from common import *

RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from goal_local.eval_client import rpc

SLOT_MIB = 7000
HEADROOM_MIB = 1024
PAIR_START_MIB = 2 * SLOT_MIB + HEADROOM_MIB
MAX_LOADERS = 4
POLL_SECONDS = 10
stop = threading.Event()
pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LOADERS)
slots = [dict(name=f'{variant}_g{gpu}_w{rank}', variant=variant, gpu=gpu,
              rank=rank, phase='waiting_memory', generation=0, retry_after=0)
         for variant, gpus in GPUS.items() for gpu in gpus for rank in (1, 2)]


@contextmanager
def db_open(variant):
    db = sqlite3.connect(RUN / variant / 'queue.sqlite', timeout=30, isolation_level=None)
    db.execute('PRAGMA busy_timeout=30000')
    try:
        yield db
    finally:
        db.close()


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, time=time.time(), **kwargs)), flush=True)


def gpu_free():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    p.check_returncode()
    return {int(line.split(',')[0]): int(line.split(',')[1]) for line in p.stdout.splitlines()}


def host_available():
    return next(int(line.split()[1]) / 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=8)
    except ProcessLookupError:
        pass


def recover(slot, reason):
    with db_open(slot['variant']) as db:
        db.execute("UPDATE jobs SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END, "
                   "error=? WHERE state='running' AND worker=?", (reason, slot['name']))


def release(slot, reason, phase='waiting_memory'):
    terminate(slot.get('_sim'))
    terminate(slot.get('_server'))
    recover(slot, reason)
    slot.update(phase=phase, reason=reason, retry_after=time.time() + 60)
    for key in ('_sim', '_server', '_future', 'server_pid', 'sim_pid', 'url'):
        slot.pop(key, None)
    emit('slot_released', worker=slot['name'], reason=reason)


def start_server(slot):
    directory = Path(slot['directory'])
    ready = directory / 'ready.json'
    with (directory / 'server.log').open('a') as log:
        proc = subprocess.Popen([MODEL_PYTHON, '-B', '-u', str(RUNTIME / 'server.py'),
                                 '--ready', str(ready)], cwd=PROJECT, env=environment(slot['gpu']),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    slot['_server'] = proc
    slot['server_pid'] = proc.pid
    for _ in range(180):
        if stop.is_set():
            raise RuntimeError('Controller stopping')
        if proc.poll() is not None:
            raise RuntimeError(f'Model process exited {proc.returncode}')
        info = read(ready, {})
        if info.get('pid') == proc.pid:
            url = f'http://127.0.0.1:{info["port"]}'
            loaded = rpc(url, '/load', dict(checkpoint=CHECKPOINTS[slot['variant']]), timeout=600)
            assert loaded['step'] == 4000 and loaded['checkpoint'] == CHECKPOINTS[slot['variant']]
            slot['url'] = url
            return loaded
        stop.wait(1)
    raise TimeoutError('Model server startup exceeded 180 seconds')


def start_sim(slot):
    directory = Path(slot['directory'])
    with (directory / 'simulation.log').open('a') as log:
        proc = subprocess.Popen([SIM_PYTHON, '-B', '-u', str(RUNTIME / 'worker.py'),
                                 '--variant', slot['variant'], '--gpu', str(slot['gpu']),
                                 '--worker', slot['name'], '--url', slot['url']],
                                cwd=PROJECT, env=environment(slot['gpu']), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    slot.update(_sim=proc, sim_pid=proc.pid, phase='running', sim_started=time.time())
    emit('worker_started', worker=slot['name'], gpu=slot['gpu'], variant=slot['variant'],
         server_pid=slot['server_pid'], sim_pid=proc.pid)


def summarize():
    result = {}
    for variant, gpus in GPUS.items():
        with db_open(variant) as db:
            counts = dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
            successes = db.execute("SELECT COALESCE(SUM(success),0) FROM jobs WHERE state='done'").fetchone()[0]
            per_gpu = []
            for gpu in gpus:
                rows = db.execute('SELECT state,COUNT(*),COALESCE(SUM(success),0) FROM jobs WHERE gpu=? GROUP BY state', (gpu,)).fetchall()
                c = {state: n for state, n, _ in rows}
                n = c.get('done', 0)
                k = sum(k for state, _, k in rows if state == 'done')
                pair = [s for s in slots if s['variant'] == variant and s['gpu'] == gpu]
                per_gpu.append(dict(gpu=gpu, total=sum(c.values()), completed=n, successes=k,
                                    pending=c.get('pending', 0), running=c.get('running', 0),
                                    errors=c.get('failed', 0), success_rate=k/n if n else None,
                                    workers=sum(s['phase']=='running' for s in pair),
                                    loading=sum(s['phase']=='loading' for s in pair)))
            categories = []
            for category in CATEGORIES:
                n, k = db.execute("SELECT COUNT(*),COALESCE(SUM(success),0) FROM jobs WHERE state='done' AND json_extract(payload,'$.category')=?", (category,)).fetchone()
                categories.append(dict(category=category, completed=n, successes=k, total=2000))
        done = counts.get('done', 0)
        finished = not counts.get('pending', 0) and not counts.get('running', 0)
        phase = ('completed' if done == 8000 else 'completed_with_errors') if finished else 'running'
        data = dict(variant=variant, phase=phase, total=8000, completed=done, successes=successes,
                    success_rate=successes/done if done else None, counts=counts,
                    gpus=per_gpu, categories=categories, updated=time.time())
        atomic(RUN / variant / 'summary.json', data)
        result[variant] = data
    return result


def publish(free, summary, phase='running', **extra):
    public = []
    for slot in slots:
        row = {k: v for k, v in slot.copy().items() if not k.startswith('_')}
        if slot.get('directory'):
            row['model'] = read(Path(slot['directory']) / 'ready.json', {})
        status = read(RUN / slot['variant'] / 'workers' / f'{slot["name"]}.json', {})
        if status.get('pid') == slot.get('sim_pid'):
            row['progress'] = status
        public.append(row)
    atomic(RUN / 'controller.json', dict(pid=os.getpid(), updated=time.time(), phase=phase,
           runtime=str(RUNTIME), slots=public, gpu_free_mib=free, required_pair_free_mib=PAIR_START_MIB,
           inference_processes_per_gpu=2, gpu_assignment=GPUS, **extra))


def main():
    lock = (RUN / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    for variant in GPUS:
        with db_open(variant) as db:
            db.execute("UPDATE jobs SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END WHERE state='running'")
    free = {}
    emit('controller_started', pid=os.getpid(), assignment=GPUS, replicas=2,
         pair_start_free_mib=PAIR_START_MIB)
    try:
        while not stop.is_set():
            summary = summarize()
            for slot in slots:
                if slot['phase'] == 'loading':
                    future = slot['_future']
                    if future.done():
                        try:
                            slot['load_info'] = future.result()
                            slot['phase'] = 'ready'
                        except Exception as exc:
                            traceback.print_exc()
                            release(slot, f'load_error: {exc}')
                if slot['phase'] not in ('ready', 'running'):
                    continue
                if slot['_server'].poll() is not None:
                    release(slot, 'model_server_exited')
                    continue
                if slot['phase'] == 'ready':
                    start_sim(slot)
                    continue
                status = read(RUN/slot['variant']/'workers'/f'{slot["name"]}.json', {})
                if slot['_sim'].poll() is not None:
                    code = slot['_sim'].returncode
                    recover(slot, f'simulator_exit_{code}')
                    with db_open(slot['variant']) as db:
                        pending = db.execute("SELECT COUNT(*) FROM jobs WHERE gpu=? AND state='pending'", (slot['gpu'],)).fetchone()[0]
                    if not pending:
                        release(slot, 'gpu_queue_finished', 'completed')
                    elif code == 0:
                        slot['phase'] = 'ready'
                    elif 'out of memory' in status.get('error', '').lower():
                        release(slot, 'cuda_out_of_memory')
                    else:
                        emit('simulator_retry', worker=slot['name'], error=status.get('error'), code=code)
                        slot['phase'] = 'ready'
                elif time.time() - status.get('updated', slot['sim_started']) > 600:
                    release(slot, 'no_progress_600_seconds')
            try:
                free = gpu_free()
            except Exception as exc:
                emit('gpu_monitor_retry', error=repr(exc))
                publish(free, summary, monitor_error=repr(exc))
                stop.wait(POLL_SECONDS)
                continue
            for variant, gpus in GPUS.items():
                for gpu in gpus:
                    pair = [s for s in slots if s['variant'] == variant and s['gpu'] == gpu]
                    waiting = [s for s in pair if s['phase'] == 'waiting_memory']
                    if not waiting or any(s['phase']=='loading' for s in pair):
                        continue
                    if any(s['retry_after'] > time.time() for s in waiting):
                        continue
                    with db_open(variant) as db:
                        pending = db.execute("SELECT COUNT(*) FROM jobs WHERE gpu=? AND state='pending'", (gpu,)).fetchone()[0]
                    if not pending:
                        for s in waiting: s['phase'] = 'completed'
                        continue
                    required = len(waiting) * SLOT_MIB + HEADROOM_MIB
                    for s in waiting:
                        s['required_free_mib'] = required
                        s['reason'] = f'Awaiting {required} MiB free; available {free[gpu]} MiB'
                    loading = sum(s['phase']=='loading' for s in slots)
                    if free[gpu] < required or loading + len(waiting) > MAX_LOADERS or host_available() < 24000:
                        continue
                    emit('gpu_admitted', variant=variant, gpu=gpu, replicas=len(waiting), free_mib=free[gpu])
                    for slot in waiting:
                        slot['generation'] += 1
                        directory = RUN/variant/'processes'/slot['name']/f'attempt_{slot["generation"]:03d}'
                        directory.mkdir(parents=True, exist_ok=True)
                        slot.update(directory=str(directory), phase='loading', created=time.time(), reason='loading checkpoint')
                        slot['_future'] = pool.submit(start_server, slot)
            publish(free, summary)
            if all(row['phase'].startswith('completed') for row in summary.values()):
                break
            stop.wait(POLL_SECONDS)
    finally:
        stop.set()
        for slot in slots:
            terminate(slot.get('_sim'))
            terminate(slot.get('_server'))
        pool.shutdown(wait=True, cancel_futures=True)
        for slot in slots:
            recover(slot, 'controller_exit')
        summary = summarize()
        publish(free, summary, phase='stopped')
        atomic(RUN/'controller_final.json', dict(updated=time.time(), summary=summary))


if __name__ == '__main__':
    main()
