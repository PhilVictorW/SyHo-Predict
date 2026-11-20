import hashlib,json
from pathlib import Path
WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parents[3]
ABA=ROOT/"ablation/aba1"
FORMAT="a0_single_latent_no_dino_v1"

def load_config(path,variant=None):
    c=json.loads(Path(path).read_text())
    assert c["architecture"]==FORMAT and c["latent_slots"]==16 and c["uses_dino"] is False and c["plan_tokens"]==0
    assert "epochs" not in c and c["total_steps"]==500 and c["warmup_steps"]==50
    assert c["validation_every_updates"]==250
    assert len(c["physical_gpus"])==3 and len(set(c["physical_gpus"]))==3
    assert all(type(g) is int and 0<=g<8 for g in c["physical_gpus"])
    assert c["world_size"]==3 and c["micro_batch"]==1 and c["accumulation"]==11
    assert c["global_batch"]==32 and c["optimizer_device"]=="cpu"
    assert c["split_seed"]==2026 and c["val_ratio"]==.05
    assert c["flow"]["num_target_vision_tokens"]==-1 and not c["flow"]["use_state"]
    return c

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


def baseline_identity(path):
    """Reuse a verified SHA only while path, inode, size, mtime and ctime match."""
    p=Path(path).resolve();st=p.stat();cache=WORK/'checks/baseline_verified.json'
    if cache.is_file():
        saved=json.loads(cache.read_text())
        now=dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino)
        if all(saved.get(k)==v for k,v in now.items()):return saved
    identity=source_identity(p)
    identity.update(ctime_ns=st.st_ctime_ns,inode=st.st_ino)
    return identity
