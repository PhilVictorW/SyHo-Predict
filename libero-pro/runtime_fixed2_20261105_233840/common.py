import json
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parent
REPO = BASE / 'vendor/LIBERO-PRO'
RUN = BASE / 'runs/20261105_233840_fixed2'
MODEL_PYTHON = '/home/wangchaoyang/envs/LaWAM/bin/python'
SIM_PYTHON = '/home/wangchaoyang/envs/LIBERO4LaWAM/bin/python'
CHECKPOINTS = {
    'plus': str(PROJECT / 'libero-plus/runs/stage_II_20260925/step_004000'),
    'libero': str(PROJECT / 'runs/goal_local/stage_II_warm250_20260921/step_004000'),
}
GPUS = {'libero': [0, 1, 2, 3], 'plus': [5, 6, 7]}
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CATEGORIES = ['object', 'swap', 'lan', 'task']
# Horizon from the official LIBERO-PRO evaluation example, plus ten settling steps.
MAX_STEPS = dict(zip(SUITES, [220, 280, 300, 520]))

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default

def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', LIBERO_HOME=str(REPO),
               LIBERO_CONFIG_PATH=str(BASE / 'resources/libero_config'),
               PYTHONPATH=os.pathsep.join([str(REPO), str(PROJECT), str(BASE / 'runtime')]),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env
