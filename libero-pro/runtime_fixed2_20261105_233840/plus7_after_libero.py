"""After all 8000 LIBERO episodes finish, share remaining PLUS work over seven GPUs."""
import argparse
import collections
import fcntl
import signal
import sqlite3
import time
import traceback

import controller as c

GPUS = (0, 1, 2, 3, 5, 6, 7)
NEW_GPUS = (0, 1, 2, 3)
TAG = 'plus7_20261106'
STATE = c.RUN / 'plus7.json'
AUDIT = c.RUN / 'plus7_handoff'
EVENT = 'plus7_after_libero'
slots = []


def counts(variant):
    with c.db_open(variant) as db:
        return dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))


def processes():
    result = []
    for p in c.Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != c.os.getuid():
                continue
            args = (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            script = next((a for a in args if a.startswith(str(c.BASE) + '/')
                           and c.Path(a).name in ('server.py', 'worker.py')), None)
            if not script or (p / 'stat').read_text().split(') ', 1)[1].split()[0] == 'Z':
                continue
            env = (p / 'environ').read_bytes().split(b'\0')
            visible = next(x.split(b'=', 1)[1] for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
            gpu = int(visible)
            kind = 'server' if c.Path(script).name == 'server.py' else 'sim'
            variant = args[args.index('--variant') + 1] if '--variant' in args else None
            if '--ready' in args:
                ready = c.Path(args[args.index('--ready') + 1])
                variant = ready.relative_to(c.RUN).parts[0] if ready.is_relative_to(c.RUN) else 'other_run'
            result.append(dict(pid=int(p.name), gpu=gpu, kind=kind, variant=variant))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def gate_reason(libero_counts, live):
    if libero_counts != {'done': 8000}:
        return f'Waiting for LIBERO: {libero_counts.get("done", 0)}/8000 done; {libero_counts}'
    old = [p['pid'] for p in live if p['gpu'] in NEW_GPUS]
    if old:
        return f'Waiting for GPU 0/1/2/3 model and simulator processes to exit: {old}'
    return None


def original_state():
    state = c.read(c.RUN / 'controller.json', {})
    if time.time() - state.get('updated', 0) > 60 or state.get('phase') != 'running':
        raise RuntimeError('Original controller is not running with a fresh heartbeat')
    return state


def transfer_pending(db):
    """Caller holds BEGIN IMMEDIATE; only GPU assignments of pending rows change."""
    before = db.execute('SELECT gpu,state,COUNT(*) FROM jobs GROUP BY gpu,state').fetchall()
    if db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] != 8000:
        raise RuntimeError('Expected exactly 8000 PLUS episodes')
    rows = db.execute("SELECT id,gpu FROM jobs WHERE state='pending' ORDER BY id").fetchall()
    plan = [dict(id=job, old_gpu=gpu, new_gpu=GPUS[i % len(GPUS)])
            for i, (job, gpu) in enumerate(rows)]
    db.executemany("UPDATE jobs SET gpu=? WHERE id=? AND state='pending'",
                   [(x['new_gpu'], x['id']) for x in plan])
    return dict(time=time.time(), before=before, assignments=plan,
                pending_per_gpu=dict(collections.Counter(x['new_gpu'] for x in plan)))


def activate():
    reason = gate_reason(counts('libero'), processes())
    if reason:
        raise RuntimeError(reason)
    original_state()
    AUDIT.mkdir(exist_ok=True)
    with c.db_open('plus') as db:
        existing = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduling_events'").fetchone()
        if existing and db.execute('SELECT 1 FROM scheduling_events WHERE event=?', (EVENT,)).fetchone():
            raise RuntimeError('Seven-GPU handoff was already committed; refusing duplicate activation')
        with sqlite3.connect(AUDIT / 'queue_before.sqlite') as backup:
            db.backup(backup)
        db.execute('BEGIN IMMEDIATE')
        try:
            db.execute('CREATE TABLE IF NOT EXISTS scheduling_events(event TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            if db.execute('SELECT 1 FROM scheduling_events WHERE event=?', (EVENT,)).fetchone():
                raise RuntimeError('Seven-GPU handoff was already committed; refusing duplicate activation')
            plan = transfer_pending(db)
            db.execute('INSERT INTO scheduling_events VALUES(?,?)', (EVENT, c.json.dumps(plan)))
            c.atomic(AUDIT / 'redistribution.json', plan)
            db.execute('COMMIT')
        except BaseException:
            db.execute('ROLLBACK')
            raise
    for gpu in NEW_GPUS:
        add_pair(gpu)
    c.emit('plus7_activated', gpus=GPUS, replicas_per_gpu=2,
           pending_per_gpu=plan['pending_per_gpu'], preserved_original_workers=True)


def add_pair(gpu):
    if any(s['gpu'] == gpu for s in slots):
        return
    for rank in (1, 2):
        slots.append(dict(name=f'plus_g{gpu}_w{rank}_{TAG}', variant='plus', gpu=gpu,
                          rank=rank, phase='waiting_memory', generation=0, retry_after=0))


def pending(gpu):
    with c.db_open('plus') as db:
        return db.execute("SELECT COUNT(*) FROM jobs WHERE gpu=? AND state='pending'", (gpu,)).fetchone()[0]


def advance():
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
        progress = c.read(c.RUN / 'plus' / 'workers' / f'{s["name"]}.json', {})
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


def admission_reason(gpu, pair, live, free, loading, host_available):
    waiting = [s for s in pair if s['phase'] == 'waiting_memory']
    if not waiting or any(s['phase'] == 'loading' for s in pair):
        return 'Pair is already loading or has no waiting slots'
    if len(pair) != 2:
        return 'Expected exactly two managed slots'
    own_pids = {s.get(k) for s in pair for k in ('server_pid', 'sim_pid')}
    if any(p['gpu'] == gpu and p['pid'] not in own_pids for p in live):
        return 'Another LIBERO-PRO model or simulator still occupies this GPU'
    models = sum(p['gpu'] == gpu and p['kind'] == 'server' for p in live)
    if models + len(waiting) > 2:
        return 'Two-inference-process limit'
    needed = len(waiting) * c.SLOT_MIB + c.HEADROOM_MIB
    if free < needed:
        return f'Waiting for {needed} MiB free; available {free} MiB'
    if loading + len(waiting) > c.MAX_LOADERS:
        return 'Waiting for checkpoint loading capacity'
    if host_available < 24000:
        return 'Waiting for 24000 MiB available host memory'
    if any(s['retry_after'] > time.time() for s in waiting):
        return 'Waiting for retry cooldown'
    return None


def admit(free):
    old = original_state()
    live = processes()
    # The original controller never restarts a completed pair. Take over only such pairs.
    for gpu in (5, 6, 7):
        pair = [s for s in old['slots'] if s['gpu'] == gpu and s['variant'] == 'plus']
        if len(pair) == 2 and all(s['phase'] == 'completed' for s in pair) and pending(gpu):
            if not any(p['gpu'] == gpu for p in live):
                add_pair(gpu)
    for gpu in sorted({s['gpu'] for s in slots}):
        pair = [s for s in slots if s['gpu'] == gpu]
        waiting = [s for s in pair if s['phase'] == 'waiting_memory']
        if not waiting:
            continue
        if not pending(gpu):
            for s in waiting:
                s['phase'] = 'completed'
            continue
        loading = sum(s['phase'] == 'loading' for s in slots)
        reason = admission_reason(gpu, pair, live, free[gpu], loading, c.host_available())
        for s in waiting:
            s['reason'] = reason or 'Loading PLUS checkpoint'
        if reason:
            continue
        c.emit('plus7_gpu_admitted', gpu=gpu, replicas=len(waiting), free_mib=free[gpu])
        for s in waiting:
            s['generation'] += 1
            directory = c.RUN / 'plus' / 'processes' / s['name'] / f'attempt_{s["generation"]:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            s.update(directory=str(directory), phase='loading', created=time.time())
            s['_future'] = c.pool.submit(c.start_server, s)


def publish(phase, reason='', free=None, activated=False):
    public = []
    for s in slots:
        row = {k: v for k, v in s.copy().items() if not k.startswith('_')}
        if s.get('directory'):
            row['model'] = c.read(c.Path(s['directory']) / 'ready.json', {})
        progress = c.read(c.RUN / 'plus' / 'workers' / f'{s["name"]}.json', {})
        if progress.get('pid') == s.get('sim_pid'):
            row['progress'] = progress
        public.append(row)
    c.atomic(STATE, dict(pid=c.os.getpid(), updated=time.time(), phase=phase, reason=reason,
                        activated=activated, slots=public, gpu_assignment={'plus': GPUS},
                        inference_processes_per_gpu=2, checkpoint=c.CHECKPOINTS['plus'],
                        libero_counts=counts('libero'), plus_counts=counts('plus'),
                        gpu_free_mib=free or {}))


def main():
    lock = (c.RUN / 'plus7.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if c.read(STATE, {}).get('activated'):
        raise RuntimeError('Handoff has already activated; refusing an unreviewed duplicate controller')
    signal.signal(signal.SIGTERM, lambda *_: c.stop.set())
    signal.signal(signal.SIGINT, lambda *_: c.stop.set())
    c.emit('plus7_waiter_started', pid=c.os.getpid(), target_gpus=GPUS,
           trigger='All 8000 LIBERO episodes done and GPU 0/1/2/3 processes released')
    activated = False
    phase = 'waiting_libero'
    free = {}
    try:
        while not c.stop.is_set():
            reason = ''
            try:
                if not activated:
                    reason = gate_reason(counts('libero'), processes())
                    if reason:
                        publish('waiting_libero', reason)
                        c.stop.wait(c.POLL_SECONDS)
                        continue
                    if counts('plus') == {'done': 8000}:
                        phase = 'completed'
                        break
                    activate()
                    activated = True
                phase = 'running'
                advance()
                plus = counts('plus')
                if not plus.get('pending', 0) and not plus.get('running', 0):
                    phase = 'completed' if plus.get('done') == 8000 else 'completed_with_errors'
                    break
                free = c.gpu_free()
                admit(free)
            except Exception as exc:
                traceback.print_exc()
                reason = repr(exc)
                c.emit('plus7_retry', error=reason)
            publish(phase, reason, free, activated)
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
            c.recover(s, 'plus7_controller_exit')
        publish(phase if phase.startswith('completed') else 'stopped', free=free, activated=activated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        live = processes()
        print(c.json.dumps(dict(libero_counts=counts('libero'), plus_counts=counts('plus'),
              waiting_reason=gate_reason(counts('libero'), live), live_processes=live,
              target_gpus=GPUS, inference_processes_per_gpu=2,
              checkpoint=c.CHECKPOINTS['plus']), indent=2))
    else:
        if (c.RUN / 'plus7.disabled').exists():
            raise SystemExit((c.RUN / 'plus7.disabled').read_text())
        main()
