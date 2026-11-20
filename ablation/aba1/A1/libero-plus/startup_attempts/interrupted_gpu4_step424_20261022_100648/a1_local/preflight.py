"""Read-only A1 plan; no model initialization, CUDA allocation, or training sampling."""
import argparse,json,shutil,subprocess
from pathlib import Path
from .config_runtime import *
from ablation.aba1.plus_common.preflight import (
    check_complete,check_teacher,inspect_dataset,plan as common_plan)

WORK=Path(__file__).resolve().parents[1]

def plan(cfg,stage,stop=None,require_sources=False):
    spec=stage_spec(cfg,stage)
    if stop is not None and stop!=spec["stop_after_updates"]:
        raise ValueError("Formal A1 budgets are Teacher5000 and Student2500")
    result=common_plan(cfg,stage,None,require_sources=require_sources)
    result.update(spec)
    result.update(final_update_lr=cfg["learning_rate"]*lr_factor(spec["stop_after_updates"]-1,spec["schedule_total_steps"]),
                  cumulative_updates=spec["stop_after_updates"],physical_gpus=[4],
                  ranks_per_gpu=spec["world_size"],save_step_zero=False,save_optimizer=False,
                  checkpoint_retention=1 if stage=="I" else 2)
    return result

def resources():
    proc=subprocess.run(["nvidia-smi","--id=4","--query-gpu=index,memory.free,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],check=True,text=True,capture_output=True)
    gpu,free,total,util=[int(v.strip()) for v in proc.stdout.strip().split(",")]
    return dict(gpu=gpu,gpu_free_mib=free,gpu_total_mib=total,gpu_utilization=util,
                disk_free_gib=shutil.disk_usage(WORK).free/2**30)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(WORK/"config.json"));p.add_argument("--report")
    a=p.parse_args();cfg=load_config(a.config,"A1")
    report=dict(plans=[plan(cfg,s) for s in ("I","II")],data=inspect_dataset(cfg),
                resources=resources(),versions=runtime_environment(),code_sha256=digest_json(code_identity()),
                status="preflight_only",training_started=False)
    if a.report:
        path=Path(a.report).resolve()
        if not path.is_relative_to(WORK):raise ValueError("Report must remain in A1")
        atomic_json(path,report)
    print(json.dumps(report,indent=2))
