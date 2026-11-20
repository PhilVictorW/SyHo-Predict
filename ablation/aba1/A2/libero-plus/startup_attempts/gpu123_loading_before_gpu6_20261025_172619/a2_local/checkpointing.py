"""Durable rolling checkpoints: commit and verify the new one before pruning."""
import json
import os
from pathlib import Path
import shutil
from .config_runtime import file_sha256

RECOVERY_FORMAT = "a2_full_recovery_v1"


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def tensor_bytes(value):
    """Count serialized tensor storages, including CPU master and Adam moments."""
    seen = set()
    def count(item):
        if hasattr(item, "untyped_storage"):
            storage = item.untyped_storage()
            identity = (str(item.device), storage.data_ptr())
            if identity in seen:
                return 0
            seen.add(identity)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(count(x) for x in item.values())
        if isinstance(item, (tuple, list)):
            return sum(count(x) for x in item)
        return 0
    return count(value)


def rank_recovery(recovery, rank, world):
    if recovery.get("format") != RECOVERY_FORMAT or recovery["contract"]["world"] != world:
        raise ValueError("Recovery format/world mismatch")
    states = recovery["rank_states"]
    if len(states) != world or [x["rank"] for x in states] != list(range(world)):
        raise ValueError("Missing or misordered per-rank recovery states")
    if not 0 <= rank < world:
        raise ValueError("Invalid recovery rank")
    state = states[rank]
    if state["epoch"] < 0 or state["consumed"] < 0 or not state["rng"]:
        raise ValueError("Invalid per-rank sampling/RNG state")
    return state


def make_recovery(step, contract, optimizer, scheduler, rank_states):
    if step <= 0 or optimizer is None or not optimizer["masters"]:
        raise ValueError("Full recovery requires a completed update and FP32 optimizer masters")
    recovery = dict(format=RECOVERY_FORMAT, step=step, contract=contract,
                    optimizer=optimizer, scheduler=scheduler, rank_states=rank_states)
    for rank in range(contract["world"]):
        rank_recovery(recovery, rank, contract["world"])
    return recovery


def commit_checkpoint(temporary, final, metadata, artifacts, validate):
    """Only complete checkpoints of this stage/config may be deleted after commit."""
    temporary, final = Path(temporary), Path(final)
    output = final.parent
    if temporary.parent != output or final.exists():
        raise ValueError("Checkpoint must be a new directory in the same stage")
    if not metadata.get("complete") or not metadata.get("resumable"):
        raise ValueError("Rolling checkpoint must include full recovery state")
    if not {"model.pt", "trainer.pt"}.issubset(artifacts):
        raise ValueError("Model and trainer artifacts are required")
    files = {}
    for name in artifacts:
        path = temporary / name
        if not path.resolve().is_relative_to(temporary.resolve()) or not path.is_file():
            raise ValueError("Invalid checkpoint artifact: " + name)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        files[name] = dict(bytes=path.stat().st_size, sha256=file_sha256(path))
    durable_json(temporary / "checkpoint_complete.json", dict(metadata, files=files))
    # Validate the serialized model metadata and both model/trainer file hashes.
    validate(temporary, require_resume=True, verify_hash=True)
    for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(temporary)
    temporary.rename(final)
    fsync_directory(output)
    durable_json(output / "latest.json", dict(checkpoint=str(final), step=metadata["step"]))
    # A failed write/validation/rename/latest update above never reaches deletion.
    for marker_path in sorted(output.glob("step_*/checkpoint_complete.json")):
        old = marker_path.parent
        if old == final:
            continue
        marker = json.loads(marker_path.read_text())
        if (marker.get("complete") and marker.get("format") == metadata["format"]
                and marker.get("stage") == metadata["stage"]
                and marker.get("config_sha256") == metadata["config_sha256"]
                and marker.get("step", metadata["step"]) < metadata["step"]):
            shutil.rmtree(old)
    fsync_directory(output)
    durable_json(output / "recovery_retention.json", dict(full_resume=str(final),
                 retained_models=[str(final)], keep_checkpoints=1, save_every=50))
