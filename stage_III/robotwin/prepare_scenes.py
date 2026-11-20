"""Explicit simulator-only preparation: validate actual repeated resets."""
import argparse
import time
from collections import defaultdict
from pathlib import Path
from .common import (DEFAULT_CONFIG,FORMAT,ROOT,atomic_json,canonical_hash,load_config,
                     local_output,read_json,require_execute,sha256)
from .runtime import spawn,wait_owned,child_env

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--candidates",required=True)
    p.add_argument("--split",choices=["train","val","test"],required=True);p.add_argument("--output",required=True)
    p.add_argument("--limit-cells",type=int);p.add_argument("--scenes-per-cell",type=int)
    p.add_argument("--gpu");p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);cfg=load_config(a.config)
    source=read_json(a.candidates);out=local_output(a.output)
    if source["kind"]!="scene_candidates":raise ValueError("Wrong candidate manifest")
    rows=[r for r in source["rows"] if r["split"]==a.split]
    cells=list(dict.fromkeys((r["task"],r["mode"]) for r in rows))
    if a.limit_cells:cells=cells[:a.limit_cells]
    count=a.scenes_per_cell or cfg["scenes"][a.split+"_per_cell"]
    if count<1:raise ValueError("Invalid scenes-per-cell")
    accepted=[];events=[];used=set();counts=defaultdict(int)
    if out.exists():
        previous=read_json(out)
        if previous["candidate_sha256"]!=sha256(a.candidates):raise ValueError("Candidate manifest changed")
        accepted=previous["rows"];events=previous["events"]
        for r in accepted:used.add((r["task"],r["initial_state_hash"]));counts[(r["task"],r["mode"])]+=1
    # Any scene present in a different prepared split cannot be accepted here.
    for path in out.parent.glob("prepared_*.json"):
        other=read_json(path)
        if other.get("split")!=a.split:
            for r in other.get("rows",[]):used.add((r["task"],r["initial_state_hash"]))
    attempted={e["scene_id"] for e in events}
    for row in rows:
        cell=(row["task"],row["mode"])
        if cell not in cells or counts[cell]>=count or row["scene_id"] in attempted:continue
        job=out.parent/"scene_jobs"/row["scene_id"];job.mkdir(parents=True,exist_ok=True)
        atomic_json(job/"row.json",row)
        cmd=[cfg["runtime"]["sim_python"],"-m","stage_III.robotwin.rollout","--config",str(Path(a.config).resolve()),
             "--row",str(job/"row.json"),"--output",str(job),"--probe","--gpu",str(a.gpu),"--execute"]
        proc=spawn(cmd,job/"probe.log",cfg["runtime"]["sim_root"],child_env(cfg,a.gpu,job))
        try:
            wait_owned(proc,job/"progress.json",cfg["runtime"]["stall_seconds"],
                       cfg["runtime"]["hard_trajectory_seconds"])
            result=read_json(job/"result.json")
            if result["status"]=="accepted":
                r=result["row"];key=(r["task"],r["initial_state_hash"])
                if key not in used:
                    accepted.append(r);used.add(key);counts[cell]+=1
                else:result=dict(status="rejected",reason="duplicate_actual_scene")
        except (TimeoutError,RuntimeError) as exc:
            result=dict(status="skipped",reason=repr(exc))
        events.append(dict(scene_id=row["scene_id"],status=result["status"],reason=result.get("reason")))
        atomic_json(out,dict(format=FORMAT,kind="prepared_scenes",prepared=True,split=a.split,
                    candidate_sha256=sha256(a.candidates),requested_cells=cells,
                    quota_per_cell=count,complete=all(counts[c]>=count for c in cells),
                    rows=accepted,events=events,updated_at=time.time()))
    if not out.exists():raise RuntimeError("No scene preparation was attempted")
    print(out)
if __name__=="__main__":main()
