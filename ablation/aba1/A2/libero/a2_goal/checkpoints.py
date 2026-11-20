"""Atomic model-only exports; no initial/best/optimizer checkpoints."""
import json
import shutil
from pathlib import Path

from .config import OWNER, atomic_json, file_sha256


def retain_latest(folder, stage, keep):
    folder = Path(folder).resolve()
    candidates = []
    for path in folder.glob("step_*/checkpoint_complete.json"):
        record = json.loads(path.read_text())
        if record.get("owner") != OWNER or record.get("stage") != stage:
            raise ValueError(f"Refusing to prune foreign checkpoint: {path}")
        directory = path.parent
        if directory.is_symlink() or directory.resolve().parent != folder:
            raise ValueError("Refusing checkpoint symlink traversal")
        if directory.name != f"step_{record['step']:06d}":
            raise ValueError("Checkpoint name/marker mismatch")
        candidates.append((record["step"], directory))
    candidates.sort()
    for _, path in candidates[:-keep]:
        shutil.rmtree(path)
    return [step for step, _ in candidates[-keep:]]


def save_weights(policy, output, stage, step, spec, extra):
    import torch
    if step % spec["save_every"] or not 0 < step <= spec["steps"]:
        raise ValueError("Checkpoint outside the requested save schedule")
    output = Path(output)
    destination = output / "checkpoints" / f"step_{step:06d}"
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Preserve existing checkpoint: {destination}")
    state_bytes = sum(v.numel()*v.element_size() for v in policy.state_dict().values())
    if shutil.disk_usage(output).free < state_bytes + 2*1024**3:
        raise OSError("Insufficient space for atomic model export; old checkpoints preserved")
    temporary.mkdir(parents=True)
    # Explicit CPU state makes the export portable and avoids retaining CUDA storages.
    state = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    policy.processor.save_pretrained(temporary / "processor")
    from ablation.aba1.libero_common.config import FORMAT
    torch.save(dict(format=FORMAT, mode=policy.mode, config=policy.cfg, step=step,
                    qwen_config=policy.qwen.config.to_dict(), dino_config=policy.dino.config.to_dict(),
                    statistics=policy.statistics, state_dict=state,
                    extra=dict(extra, stage=stage, weights_only=True, optimizer_saved=False)),
               temporary / "model.pt")
    del state
    checksum = file_sha256(temporary / "model.pt")
    marker = dict(owner=OWNER, stage=stage, step=step, mode=policy.mode,
                  model_sha256=checksum, weights_only=True, optimizer_saved=False)
    atomic_json(temporary / "checkpoint_complete.json", marker)
    temporary.rename(destination)
    entry = dict(path=str(destination), **marker)
    atomic_json(output / "latest.json", entry)
    kept = retain_latest(output / "checkpoints", stage, spec["keep"])
    atomic_json(output / "retention.json", dict(stage=stage, steps=kept, keep=spec["keep"]))
    if step == spec["steps"]:
        atomic_json(output / "final.json", entry)
    return entry


def verify_teacher(path, cfg, expected_step=1000):
    from ablation.aba1.libero_common.training import validate_teacher_metadata
    path = Path(path).resolve()
    marker = json.loads((path / "checkpoint_complete.json").read_text())
    if (marker.get("owner"), marker.get("stage"), marker.get("step")) != (OWNER, "I", expected_step):
        raise ValueError("Expected this pipeline's completed A2 Stage-I step1000")
    meta = validate_teacher_metadata(path, cfg)
    if meta["config"] != cfg or meta["step"] != expected_step:
        raise ValueError("Teacher configuration/step mismatch")
    if file_sha256(path / "model.pt") != marker["model_sha256"]:
        raise ValueError("Teacher checksum mismatch")
    return dict(path=str(path), step=expected_step, model_sha256=marker["model_sha256"])
