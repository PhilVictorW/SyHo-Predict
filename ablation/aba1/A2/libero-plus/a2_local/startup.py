"""Serialize model construction across eight ranks before synchronized training."""
from contextlib import contextmanager
import fcntl
import json
import os
import time

@contextmanager
def model_load_slot(output, rank, status):
    status(phase="waiting_model_load", step=0)
    with (output / ".model_load.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.time()
        record = dict(rank=rank, pid=os.getpid(), started=started, phase="loading")
        path = output / f"model_load_rank{rank}.json"
        path.write_text(json.dumps(record) + "\n")
        status(phase="loading", step=0)
        print(f"[staggered-load] rank={rank} pid={os.getpid()} loading model", flush=True)
        try:
            yield
        except BaseException as error:
            record.update(phase="failed", error=repr(error), ended=time.time())
            path.write_text(json.dumps(record) + "\n")
            raise
        else:
            record.update(phase="loaded", ended=time.time())
            path.write_text(json.dumps(record) + "\n")
            status(phase="loading_data", step=0)
            print(f"[staggered-load] rank={rank} loaded in {record['ended']-started:.1f}s", flush=True)
