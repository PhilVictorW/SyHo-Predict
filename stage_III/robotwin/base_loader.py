"""Load the exact composite Round10 candidate, never a latest checkpoint."""
import json
import torch
from .common import base_files, load_checkpoint, validate_base_snapshot

def load_base(cfg, device="cuda:0"):
    from robotwin.round6.model import RetainedTeacher
    from .policy import InteractionPolicy
    files = base_files(cfg)
    base_cfg = json.loads(files["config"].read_text())
    if base_cfg["baseline_checkpoint"] != str(files["checkpoint"]):
        raise ValueError("Original checkpoint mismatch")
    teacher = RetainedTeacher(base_cfg)
    named = dict(teacher.named_parameters())
    for name in ("source_adapter", "teacher_adapter"):
        p = torch.load(files[name], map_location="cpu", mmap=True, weights_only=False)
        if p["step"] != 200 or p["manifest"]["base_config"] != base_cfg:
            raise ValueError("Adapter identity mismatch: " + name)
        with torch.no_grad():
            for k, v in p["adapter"].items():
                if k not in named or named[k].shape != v.shape:
                    raise ValueError("Unexpected adapter tensor: " + k)
                named[k].copy_(v)
        del p
    teacher = teacher.to(device).eval().requires_grad_(False)
    teacher.policy.checkpoint_blocks = False
    teacher.flow.checkpoint_blocks = False
    teacher.flow.DiT.eval()
    student = InteractionPolicy(teacher).to(device).eval().requires_grad_(False)
    init = torch.load(files["initializers"], map_location="cpu", weights_only=False)
    student.set_queries(*init["queries"]["fixed_queries"])
    student.gate.load_state_dict(init["gate"], strict=True)
    student.remember_baseline_queries()
    if student.gate.scales.count_nonzero() or teacher.flow.config.horizon_sec != 1.2:
        raise ValueError("Not the Round10 zero-gate / 36-step starting point")
    return student

def load_student(cfg, checkpoint, device="cuda:0"):
    p = load_checkpoint(checkpoint)
    if p.get("kind") != "interaction" or p["control_site"] != "vlm_goal_local":
        raise ValueError("Expected an interaction checkpoint")
    validate_base_snapshot(cfg, p["base_snapshot"])
    student = load_base(cfg, device)
    student.load_state_dict(p["student"], strict=True)
    student.eval().requires_grad_(False)
    return student, p
