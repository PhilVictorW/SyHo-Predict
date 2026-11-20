"""Experiment contracts, deliberately independent of mutable historical defaults."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ABA = ROOT / 'ablation/aba1'
LAYOUTS = {'A0': (8, 8, True), 'A1': (16, 0, False),
           'A2': (0, 16, False), 'A3': (8, 8, False), 'A4': (8, 8, True)}
FORMAT = 'aba1_libero_v1'


def validate_config(cfg):
    variant = cfg['ablation']
    if variant not in LAYOUTS:
        raise ValueError('Unknown ablation')
    local, goal, interaction = LAYOUTS[variant]
    expected = dict(architecture=FORMAT, local_slots=local, goal_slots=goal, interaction=interaction,
                    llm_layers=16, plan_tokens=4, gate_start_layer=9,
                    distill_layers=[4, 8, 12, 16], goal_horizons=[3, 5, 7, 9, 11],
                    horizon=8, future_encoder='framewise_qwen_temporal_resampler',
                    repeated_diffusion_steps=2, lambda_local=1., lambda_goal=1., gram_alpha=1.)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'{variant}: {key} must be {value!r}')
    flow = cfg['flow']
    for key, value in dict(use_state=False, action_dim=32, vlm_dim=2048,
                           hidden_dim=1024, num_layers=16, num_inference_steps=10,
                           horizon_sec=.4, num_target_vision_tokens=-1).items():
        if flow.get(key) != value:
            raise ValueError(f'Invalid flow contract: {key}')
    if cfg['split_seed'] != 2026 or cfg['val_ratio'] != .05:
        raise ValueError('This protocol fixes the offline episode split')
    if cfg['warmup_steps'] < 1 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid optimizer configuration')
    return cfg


def load_config(path, variant=None):
    cfg = json.loads(Path(path).read_text())
    validate_config(cfg)
    if variant is not None and cfg['ablation'] != variant:
        raise ValueError('Entrypoint/config ablation mismatch')
    return cfg


def future_roles(cfg, stage):
    if cfg['ablation'] == 'A0':
        if stage != 'action':
            raise ValueError('A0 supports action-only training, no posterior stage')
        return False, False
    if stage not in ('I', 'II'):
        raise ValueError('A1-A4 require stage I or II')
    return cfg['local_slots'] > 0, cfg['goal_slots'] > 0


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def source_identity(path, hash_file=True):
    p = Path(path).resolve()
    st = p.stat()
    return dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns,
                sha256=file_sha256(p) if hash_file else None)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)
