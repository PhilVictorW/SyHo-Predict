from pathlib import Path
import os
from ablation.aba1.plus_common.config import code_identity as common_identity,file_sha256,checkpoint_path
from ablation.aba1.plus_common.environment import subprocess_environment as common_environment

WORK=Path(__file__).resolve().parents[1]

def subprocess_environment(cfg,gpu,cache_dir=None):
    env=common_environment(cfg,gpu,cache_dir)
    env["PYTHONPATH"]=str(WORK)+os.pathsep+env["PYTHONPATH"]
    env["TMPDIR"]="/tmp"
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    return env

def code_identity():
    result=common_identity()
    for path in sorted((WORK/"a1_eval").glob("*.py")):
        result[str(path)]=file_sha256(path)
    for path in (WORK/"evaluate.py",WORK/"a1_local/numerics.py"):
        result[str(path)]=file_sha256(path)
    return result

def validate_final_budget(meta,marker,cfg):
    extra=meta["extra"]
    if (cfg["ablation"]!="A1" or meta["mode"]!="student" or meta["step"]!=2500
        or extra.get("stage")!="II" or extra.get("teacher_step")!=5000
        or marker.get("step")!=2500 or marker.get("cumulative_step")!=7500
        or extra.get("source_paths",{}).get("teacher")!=str(checkpoint_path(cfg,"I",5000))):
        raise ValueError("A1 full evaluation requires its own Teacher5000 -> Student2500 checkpoint")
