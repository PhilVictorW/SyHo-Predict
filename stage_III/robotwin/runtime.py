"""Only explicit --execute callers may spawn owned processes."""
import os
import signal
import subprocess
import time
from pathlib import Path
from .common import atomic_json, read_json

class Progress:
    def __init__(self, path):
        self.path = Path(path)
        self.value = dict(started_at=time.time(), last_progress_at=time.time(), env_steps=0)
    def report(self, phase, meaningful=False, **kw):
        self.value.update(phase=phase, updated_at=time.time(), pid=os.getpid(), **kw)
        if meaningful:
            self.value["last_progress_at"] = time.time()
        atomic_json(self.path,self.value)

def stop_owned(proc):
    if proc.poll() is not None:
        return
    # Popen was created with start_new_session; never target historical processes.
    if os.getpgid(proc.pid) != proc.pid:
        raise RuntimeError("Refusing to stop an unowned process group")
    os.killpg(proc.pid,signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=10)

def wait_owned(proc, progress, stall_seconds=1800, hard_seconds=14400):
    start = time.time()
    try:
        while proc.poll() is None:
            try:
                row = read_json(progress)
            except (OSError,ValueError):
                row = {}
            last = max(start,float(row.get("last_progress_at",start)))
            if time.time()-last > stall_seconds or time.time()-start > hard_seconds:
                raise TimeoutError("No task progress / hard runtime budget exceeded")
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if proc.returncode:
            raise RuntimeError("Owned worker exited with code "+str(proc.returncode))
    except BaseException:
        stop_owned(proc)
        raise

def spawn(cmd, log, cwd, env):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with open(log,"ab",buffering=0) as f:
        return subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,
                                start_new_session=True)

def child_env(cfg, gpu, run):
    from .common import ROOT
    e = os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER="PCI_BUS_ID",
             PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",
             PYTHONPATH=str(ROOT),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
             OPENBLAS_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false",
             HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             ROBOTWIN_SAVE_VIDEO="0",
             ROBOTWIN_SHARED_WARP_CACHE_DIR=str(Path(run)/"cache/warp"),
             WARP_CACHE_PATH=str(Path(run)/"cache/warp"))
    return e
