"""Read-only A1 plan; no model initialization, CUDA allocation, or training sampling."""
import argparse,json,shutil,subprocess
from pathlib import Path
from .config_runtime import *
from ablation.aba1.plus_common.preflight import (
    check_complete,inspect_dataset)

WORK=Path(__file__).resolve().parents[1]

def check_teacher(path,cfg):
    meta,marker=check_complete(path)
    if (Path(path).resolve()!=checkpoint_path(cfg,"I",stage_steps(cfg,"I"))
        or meta["mode"]!="posterior" or meta["step"]!=stage_steps(cfg,"I")
        or meta["config"]!=cfg or meta["extra"].get("stage")!="I" or marker.get("stage")!="I"):
        raise ValueError("Teacher must be this A1 run's completed configured Stage I")
    return meta

def plan(cfg,stage,stop=None,require_sources=False):
    spec=stage_spec(cfg,stage)
    if stop is not None and stop!=spec["stop_after_updates"]:
        raise ValueError("Requested stop differs from configured A1 budget")
    for key in ("baseline_checkpoint","statistics_path","qwen_path","dino_path","dataset_root",
                "benchmark","simulator_python","model_python"):
        if not Path(cfg[key]).exists():raise FileNotFoundError(f"{key}: {cfg[key]}")
    sources={};pending=[]
    for name,path in source_paths(cfg,stage).items():
        target=path if name=="baseline" else path/"model.pt"
        sources[name]=str(path)
        if not target.exists():pending.append(str(target))
        elif name=="teacher":check_teacher(path,cfg)
    if require_sources and pending:raise FileNotFoundError("Upstream training not yet complete: "+", ".join(pending))
    return dict(variant=cfg["ablation"],protocol=PROTOCOL,**spec,effective_batch=32,
        train_seed=cfg["train_seed"],split_seed=cfg["split_seed"],output=str(stage_root(cfg,stage)),
        initialization_sources=sources,pending_upstream_checkpoints=pending,
        read_future=dict(zip(("local","goal"),future_roles(cfg,stage))),
        first_update_lr=cfg["learning_rate"]*lr_factor(0,spec["schedule_total_steps"]),
        final_update_lr=cfg["learning_rate"]*lr_factor(spec["stop_after_updates"]-1,spec["schedule_total_steps"]),
        cumulative_updates=spec["cumulative_offset"]+spec["stop_after_updates"],physical_gpus=[physical_gpu(cfg)],
        ranks_per_gpu=spec["world_size"],save_step_zero=False,save_optimizer=False,
        checkpoint_retention=1 if stage=="I" else 2,experimental_training_started=False)

def resources(cfg):
    proc=subprocess.run(["nvidia-smi",f"--id={physical_gpu(cfg)}","--query-gpu=index,memory.free,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],check=True,text=True,capture_output=True)
    gpu,free,total,util=[int(v.strip()) for v in proc.stdout.strip().split(",")]
    return dict(gpu=gpu,gpu_free_mib=free,gpu_total_mib=total,gpu_utilization=util,
                disk_free_gib=shutil.disk_usage(WORK).free/2**30)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(WORK/"config.json"));p.add_argument("--report")
    a=p.parse_args();cfg=load_config(a.config,"A1")
    report=dict(plans=[plan(cfg,s) for s in ("I","II")],data=inspect_dataset(cfg),
                resources=resources(cfg),versions=runtime_environment(),code_sha256=digest_json(code_identity()),
                status="preflight_only",training_started=False)
    if a.report:
        path=Path(a.report).resolve()
        if not path.is_relative_to(WORK):raise ValueError("Report must remain in A1")
        atomic_json(path,report)
    print(json.dumps(report,indent=2))
