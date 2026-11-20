"""Serialize model materialization across the connected training ranks."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from .config import atomic_json, rank_device


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def check_failures(output):
    failures = list(Path(output).glob('failure_rank_*.json'))
    if failures:
        raise RuntimeError('A rank failed during startup: ' + str(failures[0]))
    for path in Path(output).glob('initialization_*.json'):
        state = read(path)
        if state.get('phase') == 'failed':
            raise RuntimeError('A rank failed during model loading: ' + str(path))


@contextmanager
def load_slot(plan, output, rank, stage):
    if plan.get('model_load_strategy', 'parallel') != 'rank_serial':
        yield
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logical, physical = rank_device(plan, rank)
    state = dict(rank=rank, pid=os.getpid(), stage=stage, physical_gpu=physical,
                 logical_gpu=logical, phase='waiting_turn', model_loaded=False,
                 waiting_for_rank=rank-1 if rank else None, created_at=time.time())
    marker = output/f'initialization_{rank}.json'
    def update(**values):
        state.update(values, updated_at=time.time())
        atomic_json(marker, state)
    update()
    deadline = time.monotonic()+plan['loading_timeout_seconds']
    last_write = time.monotonic()
    with (output/'model_load.lock').open('a') as lock:
        try:
            while True:
                check_failures(output)
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Rank {rank} timed out waiting for its model loading turn')
                previous = read(output/f'initialization_{rank-1}.json') if rank else {}
                ready = rank == 0 or (previous.get('model_loaded') and
                    time.time() >= previous['load_finished_at']+plan['load_gap_seconds'])
                if ready:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                if time.monotonic()-last_write >= 10:
                    update()
                    last_write = time.monotonic()
                time.sleep(.5)
            update(phase='loading_model', load_started_at=time.time(), waiting_for_rank=None)
            print(json.dumps(dict(event='model_load_started', **state)), flush=True)
            yield
            update(phase='waiting_for_peers', model_loaded=True, load_finished_at=time.time())
            print(json.dumps(dict(event='model_load_finished', **state)), flush=True)
        except BaseException as error:
            update(phase='failed', error=repr(error))
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_for_loaded(plan, output):
    if plan.get('model_load_strategy', 'parallel') != 'rank_serial':
        return
    deadline = time.monotonic()+plan['loading_timeout_seconds']
    while True:
        check_failures(output)
        if all(read(Path(output)/f'initialization_{rank}.json').get('model_loaded')
               for rank in range(plan['world_size'])):
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Timed out waiting for all ranks to finish model loading')
        time.sleep(.5)
