"""A1 runtime: configured GPUs, six ranks and explicit stage budgets."""
import os
from ablation.aba1.plus_common.config import *
from ablation.aba1.plus_common.config import stage_spec as common_stage_spec, code_identity as common_identity

def physical_gpus(cfg):
    settings=cfg["a1_training"]
    gpus=settings["physical_gpus"]
    if not gpus or len(set(gpus))!=len(gpus) or any(not isinstance(g,int) or g<0 for g in gpus):
        raise ValueError("A1 requires distinct nonnegative physical GPU IDs")
    if int(settings["processes_per_gpu"])*len(gpus)!=int(settings["world_size"]):
        raise ValueError("GPU count times processes_per_gpu must equal world_size")
    return list(gpus)

def rank_device(cfg,rank):
    gpus=physical_gpus(cfg)
    if not 0<=rank<int(cfg["a1_training"]["world_size"]):
        raise ValueError("Rank outside configured world")
    return rank//int(cfg["a1_training"]["processes_per_gpu"])

def physical_gpu(cfg):
    return physical_gpus(cfg)[0]

def stage_steps(cfg,stage):
    steps=int(cfg["a1_training"][f"stage_{stage}_steps"])
    if not 0<steps<common_stage_spec(cfg,stage)["schedule_total_steps"]:
        raise ValueError("Invalid A1 stage budget")
    return steps

def source_paths(cfg,stage):
    if stage=="I":return {"baseline":Path(cfg["baseline_checkpoint"])}
    if stage=="II":return {"teacher":checkpoint_path(cfg,"I",stage_steps(cfg,"I"))}
    raise ValueError("A1 requires I/II")

def runtime_placement(cfg, stage, gpus=None, processes_per_gpu=None, rank_to_gpu=None):
    """Keep execution placement separate from immutable model/teacher config."""
    settings=cfg["a1_training"]
    devices=physical_gpus(cfg) if gpus is None else list(gpus)
    per_gpu=int(settings["processes_per_gpu"] if processes_per_gpu is None else processes_per_gpu)
    if not devices or len(set(devices))!=len(devices) or any(type(g) is not int or g<0 for g in devices):
        raise ValueError("Invalid physical GPU list")
    mapping=[gpu for gpu in devices for _ in range(per_gpu)] if rank_to_gpu is None else list(rank_to_gpu)
    if not mapping or any(type(g) is not int or g not in devices for g in mapping) or set(mapping)!=set(devices):
        raise ValueError("Invalid rank-to-GPU mapping")
    counts=[mapping.count(g) for g in devices]
    per_gpu=counts[0] if len(set(counts))==1 else None
    world=len(mapping)
    if not 1<=world<=32:
        raise ValueError("Need 1..32 ranks for the fixed effective batch of 32")
    if stage!="II" and (devices!=physical_gpus(cfg) or mapping!=[g for g in physical_gpus(cfg) for _ in range(int(settings["processes_per_gpu"]))]):
        raise ValueError("Placement overrides apply only to Stage II")
    return dict(physical_gpus=devices,processes_per_gpu=per_gpu,world_size=world,
                rank_to_gpu=mapping)

def stage_spec(cfg, stage, placement=None):
    if cfg["ablation"] != "A1" or stage not in ("I", "II"):
        raise ValueError("This runtime is only for A1 I/II")
    placement=runtime_placement(cfg,stage) if placement is None else runtime_placement(
        cfg,stage,placement["physical_gpus"],placement["processes_per_gpu"],placement["rank_to_gpu"])
    world=placement["world_size"]
    if int(os.environ.get("A1_WORLD_SIZE",str(world)))!=world:
        raise ValueError("Environment world size differs from runtime placement")
    result = common_stage_spec(cfg, stage)
    result.update(**placement)
    result.update(world_size=world, batch_size=1, accumulation=32//world if 32%world==0 else None,
                  accumulation_per_rank=[len(range(rank,32,world)) for rank in range(world)],
                  synchronization="gloo_master", stop_after_updates=stage_steps(cfg,stage),
                  save_every=int(cfg["a1_training"]["teacher_save_every" if stage=="I" else "student_save_every"]),
                  cumulative_offset=0 if stage=="I" else stage_steps(cfg,"I"),
                  eval_batches=128 if stage=="I" else 32)
    if result["save_every"]!=50 or any(cfg["a1_training"][key]!=1 for key in ("teacher_keep_checkpoints","student_keep_checkpoints")) or not cfg["a1_training"]["save_optimizer"]:
        raise ValueError("A1 requires a full resumable checkpoint every 50 updates, keeping one per stage")
    return result

def code_identity():
    result=common_identity()
    work=Path(__file__).resolve().parent
    for p in sorted(work.glob("*.py")):
        result[str(p.relative_to(ROOT))]=file_sha256(p)
    for name in ("train.py","start_training.sh"):
        p=work.parent/name
        result[str(p.relative_to(ROOT))]=file_sha256(p)
    return result
