from pathlib import Path
import os
import json
import subprocess
import sqlite3
import time
from ablation.aba1.plus_common.config import ROOT,code_identity as common_identity,file_sha256,checkpoint_path,atomic_json
from ablation.aba1.plus_common.environment import subprocess_environment as common_environment

from a2_local.config_runtime import stage_steps

WORK=Path(__file__).resolve().parents[1]
EVAL_BASE=ROOT
LIBERO_CONFIG=WORK/"a2_eval/libero_config"
SIM_RUNTIME=ROOT.parent/"LaWAM_without_subgoal/libero-plus/runtime"

def simulation_config(cfg):
    # Relocated simulation assets do not change the checkpoint's frozen config.
    return dict(cfg,libero_config_path=str(LIBERO_CONFIG))

def subprocess_environment(cfg,gpu,cache_dir=None):
    env=common_environment(simulation_config(cfg),gpu,cache_dir)
    env["PYTHONPATH"]=os.pathsep.join(map(str,[WORK,ROOT,Path(cfg["benchmark"]),SIM_RUNTIME]))
    env["TMPDIR"]="/proc/self/cwd/.t"
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    return env

def code_identity():
    result=common_identity()
    for path in sorted((WORK/"a2_eval").glob("*.py")):
        result[str(path)]=file_sha256(path)
    for path in (WORK/"evaluate.py",WORK/"a2_local/numerics.py"):
        result[str(path)]=file_sha256(path)
    return result

def validate_final_budget(meta,marker,cfg):
    extra=meta["extra"]
    teacher_step=stage_steps(cfg,"I");student_step=stage_steps(cfg,"II")
    if (cfg["ablation"]!="A2" or meta["mode"]!="student" or meta["step"]!=student_step
        or extra.get("stage")!="II" or extra.get("teacher_step")!=teacher_step
        or marker.get("step")!=student_step or marker.get("cumulative_step")!=teacher_step+student_step
        or extra.get("source_paths",{}).get("teacher")!=str(checkpoint_path(cfg,"I",teacher_step))):
        raise ValueError("A2 full evaluation requires its own configured final Teacher -> Student checkpoint")

def wait_for_gpu_memory(gpu,index,folder,minimum_mib=7168):
    """Wait for 7 GiB: measured 4.45 GiB model peak plus CUDA/render headroom."""
    status=Path(folder)/f"admission{index}.json"
    last_print=0
    while True:
        with sqlite3.connect(Path(folder)/"queue.sqlite",timeout=30) as db:
            unfinished=db.execute("SELECT COUNT(*) FROM tasks WHERE state IN ('pending','running')").fetchone()[0]
        if not unfinished:
            atomic_json(status,dict(worker=index,physical_gpu=int(gpu),phase="finished",time=time.time()))
            return False
        result=subprocess.run(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free",
            "--format=csv,noheader,nounits"],check=True,capture_output=True,text=True,timeout=30)
        free=int(result.stdout.strip())
        row=dict(worker=index,physical_gpu=int(gpu),free_mib=free,minimum_free_mib=minimum_mib,
            phase="loading" if free>=minimum_mib else "waiting_gpu_memory",time=time.time())
        atomic_json(status,row)
        if free>=minimum_mib:return True
        if time.monotonic()-last_print>=60:
            print(json.dumps(row),flush=True);last_print=time.monotonic()
        time.sleep(15)
