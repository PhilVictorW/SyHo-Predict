"""Read-only launch planning; does not construct policies, initialize CUDA or run rollouts."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from .config import *
from goal_local.eval_common import model_metadata

def checkpoint_metadata(path):
    path=Path(path)
    meta=model_metadata(path/"model.pt")
    return meta

def check_teacher(path,cfg):
    meta=checkpoint_metadata(path)
    if meta["mode"]!="posterior" or meta["step"]!=5000: raise ValueError("Teacher must be posterior step5000")
    old=meta["config"]
    if cfg["ablation"] in ("A1","A2"):
        if meta["format"]!=FORMAT or old!=cfg: raise ValueError("Single-scale Teacher must match its own experiment")
        check_complete(path,require_resume=False)
    elif meta["format"]!="goal_local_v1" or Path(path).resolve()!=Path(cfg["full_teacher"]).resolve():
        raise ValueError("A3 must use the prescribed full Teacher")
    for key in ("baseline_checkpoint","dataset_root","statistics_path","flow","goal_horizons","horizon","llm_layers","distill_layers"):
        if old[key]!=cfg[key]: raise ValueError("Teacher config mismatch: "+key)
    return meta

def check_complete(path,require_resume=False,verify_hash=False):
    path=Path(path)
    marker=json.loads((path/"checkpoint_complete.json").read_text())
    meta=checkpoint_metadata(path)
    if marker.get("format")!=FORMAT or not marker.get("complete") or marker["step"]!=meta["step"]:
        raise ValueError("Incomplete/incompatible checkpoint: "+str(path))
    if meta["format"]!=FORMAT or digest_json(meta["config"])!=marker["config_sha256"]:
        raise ValueError("Checkpoint config mismatch")
    if require_resume and not marker.get("resumable"): raise ValueError("This snapshot no longer has full recovery state")
    required=[name for name in marker["files"] if name!="trainer.pt" or require_resume]
    if "model.pt" not in required or (require_resume and "trainer.pt" not in required):
        raise ValueError("Missing model/recovery artifact manifest")
    for name in required:
        if not (path/name).is_file(): raise FileNotFoundError(path/name)
        if verify_hash and file_sha256(path/name)!=marker["files"][name]["sha256"]:
            raise ValueError("Checkpoint checksum mismatch: "+name)
    return meta,marker

def check_manifest(cfg):
    path=Path(cfg["full_eval_manifest"]);manifest=json.loads(path.read_text());tasks=manifest["tasks"]
    if len(tasks)!=10030 or len({(t["suite"],t["task_id"]) for t in tasks})!=10030:
        raise ValueError("Expected all 10030 unique task configurations")
    if dict(Counter(t["suite"] for t in tasks))!=SUITE_COUNTS: raise ValueError("Suite counts changed")
    if {t["category"] for t in tasks}!=set(CATEGORIES): raise ValueError("Category mismatch")
    source=Path(manifest["source"])
    if not source.is_relative_to(Path(cfg["benchmark"])) or file_sha256(source)!=manifest["source_sha256"]:
        raise ValueError("Task classification source mismatch")
    return dict(tasks=10030,suites=SUITE_COUNTS,sha256=file_sha256(path),source_sha256=manifest["source_sha256"])

def plan(cfg,stage,stop=None,require_sources=False):
    spec=stage_spec(cfg,stage)
    if stop is not None:
        if not 0<stop<=spec["stop_after_updates"]: raise ValueError("Stop must be within this stage's formal budget")
        spec["stop_after_updates"]=stop
    if cfg["ablation"]=="A3" and cfg["train_seed"]!=2026:
        raise ValueError("A3's historical paired Teacher/step0 supports train_seed=2026 only")
    for key in ("baseline_checkpoint","statistics_path","qwen_path","dino_path","dataset_root",
                "benchmark","simulator_python","model_python"):
        if not Path(cfg[key]).exists(): raise FileNotFoundError(f"{key}: {cfg[key]}")
    pending=[];sources={}
    for name,path in source_paths(cfg,stage).items():
        target=path if name=="baseline" else path/"model.pt"
        sources[name]=str(path)
        if not target.exists():
            pending.append(str(target));continue
        if name=="teacher": check_teacher(path,cfg)
        if name=="previous_action":
            meta,_=check_complete(path)
            if meta["config"]!=cfg or meta["step"]!=5000 or meta["extra"].get("stage")!="action1":
                raise ValueError("A0 action2 must inherit its own action1 step5000")
        if name=="student_init":
            meta=checkpoint_metadata(path)
            marker=json.loads((path/"checkpoint_complete.json").read_text())
            if meta["mode"]!="student" or meta["step"]!=0 or meta["extra"].get("teacher_step")!=5000 or marker.get("step")!=0:
                raise ValueError("A3 paired initialization must be untrained step0 from Teacher5000")
    if require_sources and pending: raise FileNotFoundError("Upstream training not yet complete: "+", ".join(pending))
    return dict(variant=cfg["ablation"],protocol=PROTOCOL,**spec,effective_batch=32,
        train_seed=cfg["train_seed"],split_seed=cfg["split_seed"],output=str(stage_root(cfg,stage)),
        initialization_sources=sources,pending_upstream_checkpoints=pending,
        read_future=dict(zip(("local","goal"),future_roles(cfg,stage))),
        first_update_lr=cfg["learning_rate"]*lr_factor(0,spec["schedule_total_steps"]),
        final_update_lr=cfg["learning_rate"]*lr_factor(spec["stop_after_updates"]-1,spec["schedule_total_steps"]),
        cumulative_updates=spec["cumulative_offset"]+spec["stop_after_updates"],
        experimental_training_started=False)

def inspect_dataset(cfg):
    from .data import IndexedFutureDataset
    stats=json.loads(Path(cfg["statistics_path"]).read_text())
    datasets=[IndexedFutureDataset(cfg["dataset_root"],stats,s,cfg["split_seed"],decode=False,local=False,goal=False)
              for s in ("train","val")]
    train,val=datasets
    actual=(len(train),len(val),len(train.selected),len(val.selected))
    if actual!=(2125407,112629,13630,717): raise ValueError(f"Offline split changed: {actual}")
    if set(e["episode_index"] for e in train.selected)&set(e["episode_index"] for e in val.selected):
        raise ValueError("Episode ID overlap")
    result=dict(train_anchors=len(train),val_anchors=len(val),train_episodes=len(train.selected),
        val_episodes=len(val.selected),manifest_sha256=train.source_digest,
        split_sha256=digest_json(val.validation_episodes),
        historical_trajectory_duplicates=True,validation_use="fit/numerical diagnostic only")
    for ds in datasets: ds.close()
    return result

def main(variant,config,argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(config));parser.add_argument("--report")
    args=parser.parse_args(argv);cfg=load_config(args.config,variant)
    stages=["action1","action2"] if variant=="A0" else ["II"] if variant=="A3" else ["I","II"]
    from .environment import subprocess_environment
    subprocess_environment(cfg,"",None)  # Validate existing runtime assets without making caches or touching CUDA.
    report=dict(variant=variant,plans=[plan(cfg,s) for s in stages],data=inspect_dataset(cfg),
        evaluation=check_manifest(cfg),versions=runtime_environment(),code_sha256=digest_json(code_identity()),
        disk_free_gib=shutil.disk_usage(ABA).free/2**30,
        status="implementation_preflight_only",training_started=False,experimental_inference_started=False,
        budget_note="A0 formal plan is 5000+4000 optimizer updates. Optional 1000 updates is a pilot, not 1000 epochs.")
    if args.report:
        target=Path(args.report).resolve()
        if not target.is_relative_to(ABA/variant/"libero-plus"):raise ValueError("Report must remain in this variant")
        atomic_json(target,report)
    print(json.dumps(report,indent=2,ensure_ascii=False))
