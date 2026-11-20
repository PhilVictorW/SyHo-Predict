"""Configuration, identities and atomic I/O. No torch/GPU imports here."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DEFAULT_CONFIG = WORK / "configs/workflow.json"
FORMAT = "robotwin_goal_local_stage3_v1"

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def local_output(path):
    p = Path(path).expanduser().resolve()
    if not p.is_relative_to(WORK) or p == WORK:
        raise ValueError("Outputs must be in stage_III/robotwin/<run-directory>")
    return p

def load_config(path=DEFAULT_CONFIG):
    c = read_json(path)
    if c.get("format") != FORMAT or c.get("control_site") != "vlm_goal_local":
        raise ValueError("Only the agreed Goal->Local route is supported")
    expected = dict(action_hz=30, horizon=36, action_dim=16, buffer_steps=50,
                    buffer_dim=32, flow_steps=10, views=3, use_state=False,
                    slots=8, vlm_dim=2048, goal_local_start_layer=9)
    if c.get("model") != expected:
        raise ValueError("RoboTwin model contract changed")
    s, r = c["interaction"], c["rl"]
    for k in ("steps", "save_every", "validate_every", "micro_batch", "accumulation", "validation_samples"):
        if not isinstance(s[k], int) or s[k] <= 0:
            raise ValueError("invalid interaction." + k)
    if s["variant"] not in ("query_gate", "query_only"):
        raise ValueError("invalid supervised variant")
    if r["group_size"] != 4 or r["groups_per_iteration"] < 1 or r["iterations"] < 1:
        raise ValueError("RL needs complete groups of four")
    if not 0 <= r["efficiency_weight"] < 1 or r["max_updates"] not in (1, 2):
        raise ValueError("invalid reward/update configuration")
    if r["control_site"] != "vlm_goal_local" or r["std_floor"] <= 0:
        raise ValueError("invalid GRPO configuration")
    if c["runtime"]["stall_seconds"] != 1800:
        raise ValueError("Preserve the user's 30-minute no-progress rule")
    if any(float(s[k]) < 0 for k in ("lambda_local", "lambda_goal", "lambda_keep")):
        raise ValueError("loss weights must be nonnegative")
    if s["flow_repeats"] != 2 or r["feature_dim"] != 6144:
        raise ValueError("flow repeat / Actor features mismatch")
    if not 0 < r["learning_rate"] or not 0 < s["learning_rate"]:
        raise ValueError("learning rate must be positive")
    return c

def base_files(c):
    b = c["base"]
    return {k: Path(b[k]).resolve() for k in
            ("config", "checkpoint", "source_adapter", "teacher_adapter", "initializers", "statistics")}

def base_snapshot(c, full_hash=False):
    records = {}
    for key, p in base_files(c).items():
        st = p.stat()
        records[key] = dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)
        if full_hash or key in ("config", "statistics"):
            records[key]["sha256"] = sha256(p)
    # Pin code dependencies as well as the composite weights.
    sources = ["robotwin/round7/model.py", "robotwin/round6/model.py",
               "robotwin/round5/model.py", "robotwin/round4/model.py",
               "robotwin/round3/model.py", "robotwin/round2/model.py", "robotwin/round6/train.py",
               "robotwin/model.py", "robotwin/data.py", "goal_local/model.py",
               "goal_local/roles.py", "goal_local/action.py", "goal_local/optim.py",
               "robotwin/round10/bridge.py", "robotwin/round10/server.py",
               "examples/Robotwin/eval_files/model2robotwin_interface.py",
               "examples/Robotwin/eval_files/robotwin_eval_common.py",
               "starVLA/model/framework/vlas/lawam.py"]
    sources += [str(p.relative_to(ROOT)) for p in sorted(WORK.glob("*.py"))]
    code = {p: sha256(ROOT / p) for p in sources}
    return dict(files=records, code=code, full_hash=full_hash)

def validate_base_snapshot(c, snapshot):
    current = base_snapshot(c)
    for k, v in current["files"].items():
        old = snapshot["files"][k]
        if any(old.get(x) != v[x] for x in ("path", "size", "mtime_ns")):
            raise ValueError("Base file changed: " + k)
    if current["code"] != snapshot["code"]:
        raise ValueError("Source changed since checkpoint; create a new run")

def require_execute(args):
    if not args.execute:
        raise SystemExit("No execution requested. Use check.sh for offline validation; --execute is required to run.")
    if getattr(args, "gpu", None) is None:
        raise SystemExit("An explicit physical --gpu is required; no historical GPU assignment is assumed")
    if not str(args.gpu).isdigit():
        raise SystemExit("--gpu must be a single physical GPU index")

def torch_save_atomic(path, payload):
    import torch
    path = local_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
        atomic_json(str(path) + ".complete.json", dict(sha256=sha256(path), format=FORMAT))
    finally:
        if os.path.exists(name):
            os.unlink(name)

def load_checkpoint(path):
    import torch
    path = Path(path).resolve()
    marker = read_json(str(path) + ".complete.json")
    if marker["sha256"] != sha256(path):
        raise ValueError("Checkpoint checksum/commit marker mismatch: " + str(path))
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("format") != FORMAT:
        raise ValueError("Not an isolated RoboTwin Stage III checkpoint")
    return p

def configure_gpu(gpu):
    # Called before importing torch/model modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")


def claim_run(path):
    import fcntl
    out = local_output(path)
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / "run.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another process owns this run")
    handle.seek(0); handle.truncate(); handle.write(str(os.getpid())); handle.flush()
    return handle  # Caller retains the handle for the entire process lifetime.
