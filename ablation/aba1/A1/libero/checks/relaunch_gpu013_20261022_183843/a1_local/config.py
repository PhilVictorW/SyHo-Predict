"""Frozen step budget and paths for the user's A1 experiment."""
import json
import math
import shutil
import subprocess
from pathlib import Path

from ablation.aba1.libero_common.config import load_config, atomic_json, file_sha256, digest_json

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
PLAN = WORK / "run_plan.json"
OWNER = "aba1.A1.libero.step_training"


def load_plan(path=PLAN):
    plan = json.loads(Path(path).read_text())
    expected = {
        "backend": "gloo", "global_batch": 32, "micro_batch": 1,
        "save_initial_weights": False, "save_best_weights": False,
        "automatic_stage_II": True, "automatic_evaluation": False,
        "checkpoint_contents": "model_processor_statistics_and_provenance_only",
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Unexpected A1 execution contract: {key}")
    gpu, world = plan.get('physical_gpu'), plan.get('world_size')
    if not isinstance(gpu, int) or gpu < 0 or not isinstance(world, int) or not 1 <= world <= 4:
        raise ValueError('Expected one physical GPU and one to four training ranks')
    if plan.get('rank_anchors_per_update') != [len(range(r, 32, world)) for r in range(world)]:
        raise ValueError('Rank shares must partition the effective batch of 32')
    if plan.get('oom_fallback_world_sizes') != list(range(world, 0, -1)):
        raise ValueError('OOM fallback must start at the requested rank count')
    for stage, steps, interval, keep in [("I", 1000, 1000, 1), ("II", 2000, 500, 2)]:
        spec = plan[f"stage_{stage}"]
        if (spec["steps"], spec["schedule_steps"], spec["save_every"], spec["keep"]) != (steps, steps, interval, keep):
            raise ValueError(f"Unexpected stage {stage} budget/retention")
        if not 0 < spec["warmup_steps"] < steps:
            raise ValueError("Invalid warmup")
    if plan["validation_anchors"] < 1 or plan["validation_every"] < 1:
        raise ValueError("Invalid validation settings")
    if plan["loader_workers_per_rank"] < 0 or plan["cpu_threads_per_rank"] < 1:
        raise ValueError("Invalid worker settings")
    for key in ("minimum_gpu_free_gib", "minimum_host_available_gib", "minimum_disk_free_gib"):
        if plan[key] <= 0:
            raise ValueError(f"Invalid resource floor: {key}")
    cfg = load_config(WORK / plan["model_config"], "A1")
    for key in ("baseline_checkpoint", "statistics_path", "qwen_path", "dino_path", "dataset_root"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key}: {cfg[key]}")
    return plan, cfg


def run_path(plan, output=None):
    result = Path(output).resolve() if output else WORK / "runs" / plan["run_name"]
    if not result.is_relative_to(WORK / "runs") or result == WORK / "runs":
        raise ValueError("Run must remain under A1/libero/runs")
    return result


def learning_rate(completed, spec, base):
    warm, total = spec["warmup_steps"], spec["schedule_steps"]
    if not 0 <= completed < spec["steps"]:
        raise ValueError("LR requested outside this stage")
    if completed < warm:
        return base * (completed + 1) / warm
    progress = min(1., (completed - warm + 1) / (total - warm))
    return base * (.05 + .95 * .5 * (1 + math.cos(math.pi * progress)))


def resource_report(plan):
    raw = subprocess.check_output([
        "nvidia-smi", "-i", str(plan["physical_gpu"]),
        "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20).strip().split(",")
    index, uuid, total, free, utilization = [x.strip() for x in raw]
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host = int(meminfo["MemAvailable"].split()[0]) / 1024**2
    gpu = float(free) / 1024
    disk = shutil.disk_usage(WORK).free / 1024**3
    blockers = []
    for label, available, minimum in [
        (f"GPU {plan['physical_gpu']} free GiB", gpu, plan["minimum_gpu_free_gib"]),
        ("host available GiB", host, plan["minimum_host_available_gib"]),
        ("workspace free GiB", disk, plan["minimum_disk_free_gib"]),
    ]:
        if available < minimum:
            blockers.append(f"{label}: {available:.2f} < {minimum}")
    return dict(gpu_index=int(index), gpu_uuid=uuid, gpu_total_gib=float(total)/1024,
                gpu_free_gib=gpu, gpu_utilization_percent=int(utilization),
                host_available_gib=host, disk_free_gib=disk,
                launch_resource_blockers=blockers,
                gpu_floor_is_conservative_estimate=True)


def source_manifest():
    paths = set((WORK / "a1_local").glob("*.py"))
    paths.update((ROOT / "ablation/aba1/libero_common").glob("*.py"))
    paths.update(ROOT / x for x in [
        "goal_local/optim.py", "goal_local/train.py",
        "ablation/aba1/plus_common/distributed.py",
        "starVLA/model/framework/vlas/flowmatching_expert.py",
        "starVLA/model/framework/vlas/cross_attention_dit.py",
        "starVLA/model/framework/latent_world/vlm_adapter.py",
    ])
    paths.update([WORK / "config.json", PLAN, WORK / "launch.py", WORK / "train.py"])
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}
