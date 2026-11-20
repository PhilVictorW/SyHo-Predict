"""Create candidate seeds only; this command never initializes a simulator."""
import argparse
from .common import DEFAULT_CONFIG,FORMAT,atomic_json,canonical_hash,load_config,local_output

def build(cfg):
    from examples.Robotwin.eval_files.robotwin_eval_common import DEFAULT_TASKS
    if len(DEFAULT_TASKS)!=50:raise ValueError("Expected 50 tasks")
    rows=[];s=cfg["scenes"]
    for cell,(task,mode) in enumerate((t,m) for t in DEFAULT_TASKS for m in ("demo_clean","demo_randomized")):
        for si,split in enumerate(("train","val","test")):
            count=s[split+"_per_cell"]*s["candidates_multiplier"]
            if count>=s["seed_stride"]//3:raise ValueError("Candidate seed ranges overlap")
            for j in range(count):
                seed=s["seed_base"]+cell*s["seed_stride"]+si*(s["seed_stride"]//3)+j
                rows.append(dict(scene_id=f"{task}__{mode}__{split}__{j:04d}",
                                 task=task,mode=mode,split=split,reset_seed=seed,prepared=False))
    return dict(format=FORMAT,kind="scene_candidates",prepared=False,config_hash=canonical_hash(s),rows=rows)

def validate_manifest(m, split=None):
    if m.get("format")!=FORMAT or m.get("kind")!="prepared_scenes" or not m.get("prepared"):
        raise ValueError("RL requires measured prepared scenes, not just different seeds")
    if not m.get("complete"):
        raise ValueError("Prepared scene coverage is incomplete; skipped cells cannot disappear")
    if split and m["split"]!=split:raise ValueError("Wrong online data split")
    ids=set();physical=set()
    for r in m["rows"]:
        if not r.get("prepared") or r["split"]!=m["split"] or not r["instruction"]:
            raise ValueError("Invalid prepared row")
        key=(r["task"],r["initial_state_hash"])
        if r["scene_id"] in ids or key in physical:raise ValueError("Duplicate initial scene")
        ids.add(r["scene_id"]);physical.add(key)
    if not ids:raise ValueError("Empty prepared manifest")
    if "requested_cells" in m:
        from collections import Counter
        counts=Counter((r["task"],r["mode"]) for r in m["rows"])
        expected={tuple(c) for c in m["requested_cells"]}
        if set(counts)!=expected or any(n!=m["quota_per_cell"] for n in counts.values()):
            raise ValueError("Prepared manifest does not match its declared cell quotas")
    return m["rows"]

def assert_disjoint(*manifests):
    seen={}
    for m in manifests:
        for r in validate_manifest(m):
            key=(r["task"],r["initial_state_hash"])
            if key in seen and seen[key]!=m["split"]:raise ValueError("Physical state leaks across online splits")
            seen[key]=m["split"]

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--config",default=str(DEFAULT_CONFIG))
    p.add_argument("--output",required=True);a=p.parse_args()
    out=local_output(a.output)
    if out.exists():raise ValueError("Candidate manifest already exists")
    atomic_json(out,build(load_config(a.config)));print(out)
if __name__=="__main__":main()
