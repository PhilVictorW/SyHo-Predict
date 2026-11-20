from pathlib import Path
import os
from ablation.aba1.plus_common.config import code_identity as common_identity,file_sha256,checkpoint_path
from ablation.aba1.plus_common.environment import subprocess_environment as common_environment

from a1_local.config_runtime import stage_steps

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
    teacher_step=stage_steps(cfg,"I");student_step=stage_steps(cfg,"II")
    if (cfg["ablation"]!="A1" or meta["mode"]!="student" or meta["step"]!=student_step
        or extra.get("stage")!="II" or extra.get("teacher_step")!=teacher_step
        or marker.get("step")!=student_step or marker.get("cumulative_step")!=teacher_step+student_step
        or extra.get("source_paths",{}).get("teacher")!=str(checkpoint_path(cfg,"I",teacher_step))):
        raise ValueError("A1 full evaluation requires its own configured final Teacher -> Student checkpoint")
