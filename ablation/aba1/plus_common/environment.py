"""Reproduce the historical Plus environment with caches isolated per new run."""
import os
from pathlib import Path
from .config import ROOT

BASE=ROOT.parent/"LaWAM_aba2"
SYSROOT=ROOT.parent/"LaWAM_without_subgoal/libero-plus/runtime/sysroot"

def subprocess_environment(cfg,gpu,cache_dir=None):
    env=os.environ.copy()
    for key in ("RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","MASTER_ADDR","MASTER_PORT"):
        env.pop(key,None)
    sysroot=Path(cfg["sim_sysroot"]);libero_config=Path(cfg["libero_config_path"])
    for path in (sysroot/"usr/lib/x86_64-linux-gnu/libMagickWand-6.Q16.so.7",libero_config/"config.yaml"):
        if not path.exists():raise FileNotFoundError(path)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",LIBERO_HOME=cfg["benchmark"],LIBERO_CONFIG_PATH=str(libero_config),
        MAGICK_HOME=str(sysroot/"usr"),WAND_MAGICK_LIBRARY_SUFFIX="-6.Q16",
        MAGICK_CONFIGURE_PATH=str(sysroot/"etc/ImageMagick-6"),
        MAGICK_CODER_MODULE_PATH=str(sysroot/"usr/lib/x86_64-linux-gnu/ImageMagick-6.9.12/modules-Q16/coders"),
        LD_LIBRARY_PATH=str(sysroot/"usr/lib/x86_64-linux-gnu")+os.pathsep+env.get("LD_LIBRARY_PATH",""),
        PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",NO_ALBUMENTATIONS_UPDATE="1",
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
        PYTHONPATH=os.pathsep.join(map(str,[ROOT,BASE,Path(cfg["benchmark"]),BASE/"libero-plus/runtime"])))
    if cache_dir is not None:
        cache=Path(cache_dir)
        for name in ("tmp","matplotlib","numba"):(cache/name).mkdir(parents=True,exist_ok=True)
        env.update(TMPDIR=str(cache/"tmp"),XDG_CACHE_HOME=str(cache),
                   MPLCONFIGDIR=str(cache/"matplotlib"),NUMBA_CACHE_DIR=str(cache/"numba"))
    return env


def load_legacy_eval_core():
    """Pin examples imports to the exact historical Plus rollout implementation."""
    import importlib
    import sys
    sys.path.insert(0,str(BASE))
    try:
        importlib.invalidate_caches()
        core=importlib.import_module("examples.LIBERO.eval_files.libero_eval_core")
        if Path(core.__file__).resolve()!=BASE/"examples/LIBERO/eval_files/libero_eval_core.py":
            raise ValueError("Wrong rollout core imported")
        adapters=importlib.import_module("examples.LIBERO.eval_files.libero_benchmark_adapters")
        if not Path(adapters.__file__).resolve().is_relative_to(BASE):
            raise ValueError("Wrong benchmark adapter imported")
        return core
    finally:
        sys.path.remove(str(BASE))
