import json
from pathlib import Path
from ablation.aba1.plus_common.config import (
    ROOT, ACTION_PROTOCOL, SUITES, CATEGORIES, MAX_STEPS, atomic_json,
    digest_json, file_sha256, runtime_environment,
)

WORK = Path(__file__).resolve().parents[1]
FORMAT = 'a0_plus_single_latent_dino_v2'
PROTOCOL = 'libero_plus_A0_single_latent_dino_1000epochs_v2'

def load_config(path, variant=None):
    cfg = json.loads(Path(path).read_text())
    expected = dict(architecture=FORMAT, protocol=PROTOCOL, ablation='A0',
        latent_slots=16, uses_dino=True, plan_tokens=0, conditioning='latent_only_plus_current_dino',
        epochs=1000, world_size=4, micro_batch=4, accumulation=2,
        physical_gpus=[0,1,3,7], action_protocol=ACTION_PROTOCOL,
        llm_layers=16, horizon=8, repeated_diffusion_steps=2)
    for k,v in expected.items():
        if cfg.get(k)!=v: raise ValueError(f'{k} must be {v!r}')
    for k,v in dict(use_state=False,num_target_vision_tokens=-1,cfg_drop_prob=0.,
        vlm_dim=2048,vision_dim=768,hidden_dim=1024,num_layers=16,num_inference_steps=10).items():
        if cfg['flow'].get(k)!=v: raise ValueError(f'Invalid flow {k}')
    return cfg

def code_identity():
    files = sorted(Path(__file__).parent.glob('*.py'))
    files += [ROOT/'ablation/aba1/plus_common'/n for n in ('model.py','action.py','data.py','roles.py','environment.py','evaluation.py','training.py')]
    files += [ROOT/'libero-plus/runtime/indexed_future.py', ROOT/'goal_local/data.py', ROOT/'goal_local/eval_client.py']
    files += sorted((ROOT/'starVLA/model/framework/vlas').glob('*.py'))
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in files}

def source_identity(path):
    p=Path(path)
    return dict(path=str(p.resolve()),sha256=file_sha256(p),bytes=p.stat().st_size)

def run_root(cfg):
    return WORK/'runs'/cfg['run_name']
