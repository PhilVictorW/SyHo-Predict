"""Frozen LIBERO-Plus ablation contracts. Import has no runtime side effects."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ABA = ROOT / "ablation/aba1"
FORMAT = "aba1_libero_plus_v1"
PROTOCOL = "libero_plus_A_v1"
LAYOUTS = {"A0": (8, 8, True), "A1": (16, 0, False),
           "A2": (0, 16, False), "A3": (8, 8, False), "A4": (8, 8, True)}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATEGORIES = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
MAX_STEPS = dict(zip(SUITES, [250, 300, 320, 550]))
SUITE_COUNTS = dict(zip(SUITES, [2402, 2518, 2591, 2519]))
ACTION_PROTOCOL = dict(id="libero_plus_open_to_legacy_close_v1", source_gripper="one_is_open",
    client_gripper="one_is_closed", simulator_gripper="minus_one_open_plus_one_closed",
    conversion="client_gripper = 1 - predicted_gripper", motion_dimensions_unchanged=True)

def validate_config(cfg):
    variant = cfg["ablation"]
    local, goal, interaction = LAYOUTS[variant]
    expected = dict(architecture=FORMAT, protocol=PROTOCOL, local_slots=local, goal_slots=goal,
        interaction=interaction, llm_layers=16, plan_tokens=4, gate_start_layer=9,
        distill_layers=[4,8,12,16], goal_horizons=[3,5,7,9,11], horizon=8,
        future_encoder="framewise_qwen_temporal_resampler", repeated_diffusion_steps=2,
        lambda_local=1., lambda_goal=1., gram_alpha=1., split_seed=2026, val_ratio=.05,
        learning_rate=1e-5, warmup_steps=500, weight_decay=1e-8, gradient_clip=1.,
        actor_mode="absent", action_protocol=ACTION_PROTOCOL)
    for k, v in expected.items():
        if cfg.get(k) != v: raise ValueError(f"{variant}: {k} must be {v!r}")
    for k, v in dict(use_state=False,action_dim=32,vlm_dim=2048,vision_dim=768,hidden_dim=1024,
        num_layers=16,attention_heads=16,num_inference_steps=10,horizon_sec=.4,
        num_target_vision_tokens=-1).items():
        if cfg["flow"].get(k) != v: raise ValueError(f"Invalid flow contract: {k}")
    if cfg["seed"] != cfg["train_seed"]: raise ValueError("seed must equal train_seed, never split_seed")
    return cfg

def load_config(path, variant=None):
    cfg = validate_config(json.loads(Path(path).read_text()))
    if variant is not None and cfg["ablation"] != variant: raise ValueError("Entrypoint/config mismatch")
    return cfg

def stage_spec(cfg, stage):
    if cfg["ablation"] == "A0":
        if stage not in ("action1","action2"): raise ValueError("A0 has only action1/action2")
    elif cfg["ablation"] in ("A1","A2"):
        if stage not in ("I","II"): raise ValueError("A1/A2 require I/II")
    elif stage != "II":
        raise ValueError("A3 reuses the fixed full Teacher; only Stage II is supported")
    first = stage in ("I","action1")
    return dict(stage=stage, world_size=2 if first else 4, batch_size=2 if first else 1,
        accumulation=8, schedule_total_steps=16000 if first else 10000,
        stop_after_updates=5000 if first else 4000, save_every=1000 if first else 500,
        eval_every=250 if first else 500, eval_batches=64 if first else 32,
        synchronization="nccl_ddp" if first else "gloo_master",
        cumulative_offset=5000 if stage=="action2" else 0)

def future_roles(cfg, stage):
    stage_spec(cfg, stage)
    return (False,False) if cfg["ablation"]=="A0" else (cfg["local_slots"]>0,cfg["goal_slots"]>0)

def lr_factor(completed_updates, schedule_total_steps, warmup=500):
    """Value used for update completed_updates+1, identical to historical LambdaLR."""
    if not 0 <= completed_updates < schedule_total_steps: raise ValueError("LR index outside schedule")
    warm = min(warmup, max(1, schedule_total_steps//10))
    return min(1.,(completed_updates+1)/warm)*(.05+.95*.5*(1+math.cos(
        math.pi*max(0,completed_updates-warm)/max(1,schedule_total_steps-warm))))

def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    tmp.replace(path)

def run_root(cfg):
    return ABA/cfg["ablation"]/"libero-plus"/"protocol_plus_v1"/f"seed_{cfg['train_seed']}"

def stage_root(cfg,stage):
    stage_spec(cfg,stage)
    return run_root(cfg)/stage

def checkpoint_path(cfg,stage,step):
    return stage_root(cfg,stage)/f"step_{step:06d}"

def source_paths(cfg,stage):
    if stage=="I" or stage=="action1": return {"baseline":Path(cfg["baseline_checkpoint"])}
    if stage=="action2": return {"previous_action":checkpoint_path(cfg,"action1",5000)}
    if cfg["ablation"] in ("A1","A2"): return {"teacher":checkpoint_path(cfg,"I",5000)}
    return {"teacher":Path(cfg["full_teacher"]), "student_init":Path(cfg["paired_student_init"])}

def code_identity():
    files=list(Path(__file__).parent.glob("*.py"))
    files += [ROOT/"goal_local"/n for n in ("data.py","optim.py","eval_client.py","eval_common.py")]
    files += [ROOT/"libero-plus/runtime/indexed_future.py"]
    files += list((ROOT/"starVLA/model/framework/vlas").glob("*.py"))
    files += list((ROOT/"starVLA/model/framework/latent_world").glob("*.py"))
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(set(files))}

def runtime_environment():
    import importlib.metadata
    versions={}
    for package in ("torch","transformers","diffusers","numpy","accelerate"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    return versions


def asset_identity(cfg):
    files=[Path(cfg["statistics_path"]),Path(cfg["dataset_root"])/"manifest.json",
           ROOT/"libero-plus/cache/index/metadata.json",Path(cfg["dino_path"])/"config.json"]
    files += [p for p in Path(cfg["qwen_path"]).iterdir() if p.is_file() and p.suffix in (".json",".txt",".jinja")]
    return {str(p):file_sha256(p) for p in sorted(set(files))}
