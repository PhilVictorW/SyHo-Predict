"""Verified local-SSD replicas for read-only Stage-II checkpoint loading."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_CACHE_ROOT = Path("/tmp") / f"lawam_stage3_libero_plus_cache_{os.getuid()}"


def fingerprint(path: Path) -> dict:
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def tree_fingerprints(root: Path) -> dict:
    return {str(p.relative_to(root)): fingerprint(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_cache(cache: Path, source: Path, expected_sha256: str) -> Path:
    receipt = json.loads((cache / "verified.json").read_text())
    if (receipt.get("format") != "stage3_verified_student_cache_v1"
            or receipt.get("source") != str(source.resolve())
            or receipt.get("sha256") != expected_sha256
            or receipt.get("source_model") != fingerprint(source / "model.pt")
            or receipt.get("cache_model") != fingerprint(cache / "model.pt")
            or receipt.get("source_processor") != tree_fingerprints(source / "processor")
            or receipt.get("cache_processor") != tree_fingerprints(cache / "processor")):
        raise RuntimeError("SSD checkpoint replica or source changed; verification required")
    return cache


def prepare_student_cache(source: Path, expected_sha256: str,
                          cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
    source = source.resolve()
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (cache_root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = cache_root / "latest.json"
        if index.exists():
            try:
                cache = Path(json.loads(index.read_text())["cache"])
                validate_cache(cache, source, expected_sha256)
                print(f"[Stage III][SSD缓存] 复用已验证副本：{cache}", flush=True)
                return cache
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
        before = fingerprint(source / "model.pt")
        processor_before = tree_fingerprints(source / "processor")
        cache = cache_root / f"student_{expected_sha256[:16]}_{time.time_ns()}"
        cache.mkdir()
        temp = cache / "model.pt.partial"
        h = hashlib.sha256(); count = 0; last = 0.0
        print("[Stage III][SSD缓存] 从原学生权重只读复制并计算 SHA256；仅向 /tmp SSD 写入。", flush=True)
        with (source / "model.pt").open("rb") as src, temp.open("xb") as dst:
            while True:
                chunk = src.read(16 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk); h.update(chunk); count += len(chunk)
                now = time.monotonic()
                if now - last >= 15:
                    print(f"[Stage III][SSD缓存] 复制校验 {count / before['size']:.1%} "
                          f"({count / 2**30:.2f}/{before['size'] / 2**30:.2f} GiB)", flush=True)
                    last = now
            dst.flush(); os.fsync(dst.fileno())
        if fingerprint(source / "model.pt") != before or h.hexdigest() != expected_sha256:
            raise RuntimeError("source checkpoint changed or checksum mismatch; refusing SSD replica")
        temp.replace(cache / "model.pt")
        shutil.copytree(source / "processor", cache / "processor")
        if tree_fingerprints(source / "processor") != processor_before:
            raise RuntimeError("source processor changed during copying")
        for rel in processor_before:
            if digest(source / "processor" / rel) != digest(cache / "processor" / rel):
                raise RuntimeError(f"processor copy mismatch: {rel}")
        receipt = dict(format="stage3_verified_student_cache_v1", source=str(source),
                       sha256=h.hexdigest(), source_model=before,
                       cache_model=fingerprint(cache / "model.pt"),
                       source_processor=processor_before,
                       cache_processor=tree_fingerprints(cache / "processor"), verified_at=time.time())
        (cache / "verified.json").write_text(json.dumps(receipt, indent=2))
        validate_cache(cache, source, expected_sha256)
        tmp = index.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(cache=str(cache)))); tmp.replace(index)
        print(f"[Stage III][SSD缓存] 副本校验通过：{h.hexdigest()}；路径 {cache}", flush=True)
        return cache


def checkpoint_load_path(source: Path) -> Path:
    value = os.environ.get("STAGE3_STUDENT_CACHE")
    if not value:
        return source
    baseline = json.loads((Path(__file__).parent / "baseline_snapshot.json").read_text())
    return validate_cache(Path(value), source, baseline["base_student_model_sha256"])


if __name__ == "__main__":
    baseline = json.loads((Path(__file__).parent / "baseline_snapshot.json").read_text())
    prepare_student_cache(Path(baseline["base_student"]), baseline["base_student_model_sha256"])
