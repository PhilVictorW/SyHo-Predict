"""Frozen step budget and paths for the user's A2 experiment."""
import json
import math
import shutil
import subprocess
from pathlib import Path

from ablation.aba1.libero_common.config import load_config, atomic_json, file_sha256, digest_json

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
PLAN = WORK / "run_plan.json"
OWNER = "aba1.A2.libero.step_training"


def load_plan(path=PLAN):
    plan = json.loads(Path(path).read_text())
    expected = {
        "backend": "gloo", "global_batch": 32,
        "micro_batch": 1,
        "save_initial_weights": False, "save_best_weights": False,
        "automatic_stage_II": True, "automatic_evaluation": True,
        "checkpoint_contents": "model_processor_statistics_and_provenance_only",
        "stage_II_frozen_teacher_placement": "cpu_module_streaming_same_dtype",
        "evaluation": {"protocol": "full", "episodes_per_task": 50, "total_tasks": 40,
                       "expected_episodes": 2000, "student_step": 2000,
                       "physical_gpu": plan["evaluation"]["physical_gpu"], "max_attempts": 3},
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Unexpected A2 execution contract: {key}")
    world = plan.get("world_size")
    if type(world) is not int or not 1 <= world <= 32:
        raise ValueError("Invalid rank count")
    if plan.get("rank_anchors_per_update") != [len(range(r,32,world)) for r in range(world)]:
        raise ValueError("Rank batches must partition the global batch")
    if type(plan["evaluation"]["physical_gpu"]) is not int or plan["evaluation"]["physical_gpu"] < 0:
        raise ValueError("Invalid evaluation GPU")
    gpus = plan.get("physical_gpus")
    if (not isinstance(gpus, list) or len(gpus) != plan["world_size"]
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError("One physical GPU mapping is required per rank")
    for stage, steps in [("I", 1000), ("II", 2000)]:
        spec = plan[f"stage_{stage}"]
        if (spec["steps"], spec["schedule_steps"]) != (steps, steps):
            raise ValueError(f"Unexpected stage {stage} budget")
        if (type(spec["save_every"]) is not int or spec["save_every"] <= 0
            or steps % spec["save_every"] or type(spec["keep"]) is not int or spec["keep"] < 1):
            raise ValueError(f"Invalid stage {stage} save schedule/retention")
        if not 0 < spec["warmup_steps"] < steps:
            raise ValueError("Invalid warmup")
    if plan["validation_anchors"] < 1 or plan["validation_every"] < 1:
        raise ValueError("Invalid validation settings")
    if plan["loader_workers_per_rank"] < 0 or plan["cpu_threads_per_rank"] < 1:
        raise ValueError("Invalid worker settings")
    for key in ("minimum_gpu_free_gib", "minimum_host_available_gib", "minimum_disk_free_gib"):
        if plan[key] <= 0:
            raise ValueError(f"Invalid resource floor: {key}")
    cfg = load_config(WORK / plan["model_config"], "A2")
    for key in ("baseline_checkpoint", "statistics_path", "qwen_path", "dino_path", "dataset_root"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key}: {cfg[key]}")
    return plan, cfg


def run_path(plan, output=None):
    result = Path(output).resolve() if output else WORK / "runs" / plan["run_name"]
    if not result.is_relative_to(WORK / "runs") or result == WORK / "runs":
        raise ValueError("Run must remain under A2/libero/runs")
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
    rows = subprocess.check_output([
        "nvidia-smi", "-i", ",".join(map(str, dict.fromkeys(plan["physical_gpus"]))),
        "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20).strip().splitlines()
    devices = []
    for row in rows:
        index, uuid, total, free, utilization = [x.strip() for x in row.split(",")]
        devices.append(dict(index=int(index), uuid=uuid, total_gib=float(total)/1024,
                            free_gib=float(free)/1024, utilization_percent=int(utilization)))
    if [d["index"] for d in devices] != list(dict.fromkeys(plan["physical_gpus"])):
        raise RuntimeError("Unexpected GPU inventory")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host = int(meminfo["MemAvailable"].split()[0]) / 1024**2
    disk = shutil.disk_usage(WORK).free / 1024**3
    return dict(devices=devices, host_available_gib=host, disk_free_gib=disk)


def source_manifest():
    paths = set((WORK / "a2_goal").glob("*.py"))
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
