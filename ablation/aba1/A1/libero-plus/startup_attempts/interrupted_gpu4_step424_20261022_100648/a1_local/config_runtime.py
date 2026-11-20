"""Fixed A1 runtime: GPU 4, six synchronized ranks, 5000 -> 2500 updates."""
import os
from ablation.aba1.plus_common.config import *
from ablation.aba1.plus_common.config import stage_spec as common_stage_spec, code_identity as common_identity

def stage_spec(cfg, stage):
    if cfg["ablation"] != "A1" or stage not in ("I", "II"):
        raise ValueError("This runtime is only for A1 I/II")
    world=int(os.environ.get("A1_WORLD_SIZE", "6"))
    if world not in (1,2,4,6):raise ValueError("A1 supports 6, 4, 2 or 1 processes")
    result = common_stage_spec(cfg, stage)
    result.update(world_size=world, batch_size=1, accumulation=32//world if 32%world==0 else None,
                  accumulation_per_rank=[len(range(rank,32,world)) for rank in range(world)],
                  synchronization="gloo_master", stop_after_updates=5000 if stage=="I" else 2500,
                  save_every=5000 if stage=="I" else 500,
                  eval_batches=128 if stage=="I" else 32)
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
