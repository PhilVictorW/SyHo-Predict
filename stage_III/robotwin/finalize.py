"""Select using validation only, lock the choice, then evaluate untouched test."""
import argparse
from pathlib import Path
from collections import defaultdict
from .common import (DEFAULT_CONFIG,atomic_json,load_config,local_output,read_json,require_execute,sha256)
from .make_manifests import validate_manifest,assert_disjoint
from .evaluate import evaluate

def macro(rows):
    cells=defaultdict(list)
    for r in rows:cells[(r["task"],r["mode"])].append(float(r["success"]))
    if not cells:raise ValueError("Empty evaluation arm")
    per_mode={}
    for mode in ("demo_clean","demo_randomized"):
        values=[sum(v)/len(v) for (task,m),v in cells.items() if m==mode]
        if not values:raise ValueError("Both modes are required")
        per_mode[mode]=sum(values)/len(values)
    return dict(successes=sum(int(r["success"]) for r in rows),episodes=len(rows),cells=len(cells),
                macro_success=sum(per_mode.values())/2,modes=per_mode)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student",required=True)
    p.add_argument("--rl-run",required=True);p.add_argument("--val-manifest",required=True)
    p.add_argument("--test-manifest",required=True);p.add_argument("--output",required=True)
    p.add_argument("--gpu");p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);cfg=load_config(a.config)
    vm,tm=read_json(a.val_manifest),read_json(a.test_manifest)
    validate_manifest(vm,"val");validate_manifest(tm,"test");assert_disjoint(vm,tm)
    out=local_output(a.output);out.mkdir(parents=True,exist_ok=True)
    candidates=[]
    for path in Path(a.rl_run).glob("validation_*/report.json"):
        e=read_json(path)
        if not e["complete"] or e["manifest_sha256"]!=sha256(a.val_manifest):continue
        if e["student_sha256"]!=sha256(a.student):raise ValueError("Validation Student identity changed")
        iteration=int(path.parent.name.rsplit("_",1)[1])
        actor=Path(a.rl_run)/"checkpoints"/("actor_%04d.pt"%iteration)
        if sha256(actor)!=e["actor_sha256"]:raise ValueError("Validated Actor was overwritten")
        candidates.append((macro(e["arms"]["actor_mean"])["macro_success"],-iteration,actor))
    if not candidates:raise ValueError("No completed validation; refusing test selection")
    _,_,actor=max(candidates,key=lambda x:(x[0],x[1]))
    controls=[dict(name="eta_"+str(v),policy="fixed",eta=v) for v in (0.,.8,1.,1.2)]
    controls.append(dict(name="actor_mean",policy="actor_mean"))
    val=evaluate(cfg,str(Path(a.config).resolve()),a.student,a.val_manifest,out/"val_controls",a.gpu,actor,controls)
    if not val["complete"]:raise ValueError("Validation controls incomplete")
    fixed=max((macro(val["arms"]["eta_"+str(v)])["macro_success"],-abs(v-1),v) for v in (0.,.8,1.,1.2))[2]
    selection=dict(student_sha256=sha256(a.student),actor_path=str(actor.resolve()),actor_sha256=sha256(actor),
        fixed_eta=fixed,val_manifest_sha256=sha256(a.val_manifest),test_manifest_sha256=sha256(a.test_manifest),
        val_metrics={k:macro(v) for k,v in val["arms"].items()},selected_without_test_outcomes=True)
    if (out/"selection.json").exists() and read_json(out/"selection.json")!=selection:
        raise ValueError("Selection was already locked; cannot tune using test results")
    atomic_json(out/"selection.json",selection)
    policies=[dict(name="round10",policy="baseline"),dict(name="interaction",policy="fixed",eta=1.),
              dict(name="best_fixed",policy="fixed",eta=fixed),dict(name="actor_mean",policy="actor_mean")]
    test=evaluate(cfg,str(Path(a.config).resolve()),a.student,a.test_manifest,out/"test",a.gpu,actor,policies)
    atomic_json(out/"summary.json",dict(selection=selection,complete=test["complete"],skips=test["skips"],
        metrics={k:macro(v) for k,v in test["arms"].items() if v},
        scope="Use reported task coverage; a pilot subset is not the full 50x2 benchmark."))
if __name__=="__main__":main()
