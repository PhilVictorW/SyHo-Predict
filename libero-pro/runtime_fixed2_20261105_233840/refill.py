"""Resume pending LIBERO episodes on the completed GPU 0/1/3 pairs."""
import collections
import fcntl
import signal
import sqlite3
import time
import traceback
import controller as c

TARGETS = (0, 1, 3)
TAG = 'refill_20261106'
STATE = c.RUN / 'refill.json'
AUDIT = c.RUN / TAG
slots = [dict(name=f'libero_g{gpu}_w{rank}_{TAG}', variant='libero', gpu=gpu,
              rank=rank, phase='waiting_memory', generation=0, retry_after=0)
         for gpu in TARGETS for rank in (1, 2)]

def original():
    state = c.read(c.RUN / 'controller.json', {})
    if time.time() - state.get('updated', 0) > 60:
        raise RuntimeError('Original controller heartbeat is stale')
    if not c.Path(f'/proc/{state["pid"]}').exists():
        raise RuntimeError('Original controller is not alive')
    for gpu in TARGETS:
        pair = [s for s in state['slots'] if s['gpu'] == gpu]
        if len(pair) != 2 or any(s['phase'] != 'completed' for s in pair):
            raise RuntimeError(f'GPU {gpu} has not released both original replicas')
    return state

def pending(gpu):
    with c.db_open('libero') as db:
        return db.execute("SELECT COUNT(*) FROM jobs WHERE gpu=? AND state='pending'", (gpu,)).fetchone()[0]

def redistribute():
    original()
    AUDIT.mkdir(exist_ok=True)
    plan_file = AUDIT / 'redistribution.json'
    if plan_file.exists():
        raise RuntimeError('Refill was already initialized; refusing another redistribution')
    with c.db_open('libero') as db:
        with sqlite3.connect(AUDIT / 'queue_before.sqlite') as backup:
            db.backup(backup)
        db.execute('BEGIN IMMEDIATE')
        try:
            before = db.execute('SELECT gpu,state,COUNT(*) FROM jobs GROUP BY gpu,state').fetchall()
            assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 8000
            assert not db.execute("SELECT COUNT(*) FROM jobs WHERE gpu IN (0,1,3) AND state!='done'").fetchone()[0]
            rows = db.execute("SELECT id,gpu FROM jobs WHERE state='pending' ORDER BY id").fetchall()
            assignments = [dict(id=job_id, old_gpu=gpu, new_gpu=(0, 1, 2, 3)[i % 4])
                           for i, (job_id, gpu) in enumerate(rows)]
            c.atomic(plan_file, dict(time=time.time(), before=before, assignments=assignments))
            db.executemany("UPDATE jobs SET gpu=? WHERE id=? AND state='pending'",
                           [(r['new_gpu'], r['id']) for r in assignments])
            db.execute('COMMIT')
        except BaseException:
            db.execute('ROLLBACK')
            raise
    c.emit('pending_redistributed', total=len(rows),
           by_gpu=dict(collections.Counter(r['new_gpu'] for r in assignments)))

def publish(free, phase='running'):
    public = []
    for s in slots:
        row = {k: v for k, v in s.copy().items() if not k.startswith('_')}
        if s.get('directory'):
            row['model'] = c.read(c.Path(s['directory']) / 'ready.json', {})
        progress = c.read(c.RUN / 'libero' / 'workers' / f'{s["name"]}.json', {})
        if progress.get('pid') == s.get('sim_pid'):
            row['progress'] = progress
        public.append(row)
    c.atomic(STATE, dict(pid=c.os.getpid(), updated=time.time(), phase=phase,
                        slots=public, gpu_free_mib=free, targets=TARGETS))

def main():
    lock = (c.RUN / 'refill.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: c.stop.set())
    signal.signal(signal.SIGINT, lambda *_: c.stop.set())
    redistribute()
    c.emit('refill_started', pid=c.os.getpid(), targets=TARGETS, replicas_per_gpu=2)
    free = {}
    try:
        while not c.stop.is_set():
            for s in slots:
                if s['phase'] == 'loading' and s['_future'].done():
                    try:
                        s['load_info'] = s['_future'].result()
                        s['phase'] = 'ready'
                    except Exception as exc:
                        traceback.print_exc()
                        c.release(s, f'load_error: {exc}')
                if s['phase'] not in ('ready', 'running'):
                    continue
                if s['_server'].poll() is not None:
                    c.release(s, 'model_server_exited')
                    continue
                if s['phase'] == 'ready':
                    c.start_sim(s)
                    continue
                progress = c.read(c.RUN / 'libero' / 'workers' / f'{s["name"]}.json', {})
                if s['_sim'].poll() is not None:
                    code = s['_sim'].returncode
                    c.recover(s, f'simulator_exit_{code}')
                    if not pending(s['gpu']):
                        c.release(s, 'gpu_queue_finished', 'completed')
                    elif 'out of memory' in progress.get('error', '').lower():
                        c.release(s, 'cuda_out_of_memory')
                    else:
                        s['phase'] = 'ready'
                        c.emit('simulator_retry', worker=s['name'], code=code)
                elif time.time() - progress.get('updated', s['sim_started']) > 600:
                    c.release(s, 'no_progress_600_seconds')
            if all(s['phase'] == 'completed' for s in slots):
                break
            try:
                original()
                free = c.gpu_free()
                for gpu in TARGETS:
                    pair = [s for s in slots if s['gpu'] == gpu]
                    waiting = [s for s in pair if s['phase'] == 'waiting_memory']
                    if not waiting or any(s['phase'] == 'loading' for s in pair):
                        continue
                    if not pending(gpu):
                        for s in waiting:
                            s['phase'] = 'completed'
                        continue
                    needed = len(waiting) * c.SLOT_MIB + c.HEADROOM_MIB
                    loading = sum(s['phase'] == 'loading' for s in slots)
                    if (free[gpu] < needed or loading + len(waiting) > c.MAX_LOADERS
                            or c.host_available() < 24000
                            or any(s['retry_after'] > time.time() for s in waiting)):
                        continue
                    c.emit('refill_gpu_admitted', gpu=gpu, free_mib=free[gpu], replicas=len(waiting))
                    for s in waiting:
                        s['generation'] += 1
                        directory = c.RUN / 'libero' / 'processes' / s['name'] / f'attempt_{s["generation"]:03d}'
                        directory.mkdir(parents=True, exist_ok=True)
                        s.update(directory=str(directory), phase='loading', created=time.time())
                        s['_future'] = c.pool.submit(c.start_server, s)
            except Exception as exc:
                c.emit('refill_admission_wait', error=repr(exc))
            publish(free)
            c.stop.wait(c.POLL_SECONDS)
    finally:
        c.stop.set()
        for s in slots:
            c.terminate(s.get('_sim'))
            c.terminate(s.get('_server'))
        c.pool.shutdown(wait=True, cancel_futures=True)
        for s in slots:
            c.terminate(s.get('_sim'))
            c.terminate(s.get('_server'))
            c.recover(s, 'refill_controller_exit')
        publish(free, 'completed' if all(s['phase'] == 'completed' for s in slots) else 'stopped')

if __name__ == '__main__':
    main()
