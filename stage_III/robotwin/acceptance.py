"""Recompute admission from measured sensitivity and paired held-out results."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import (DEFAULT_CONFIG,FORMAT,atomic_json,canonical_hash,load_checkpoint,
                     load_config,local_output,read_json,sha256)
from .make_manifests import validate_manifest

def paired_statistics(report, manifest, cfg):
    scenes=validate_manifest(manifest,"val")
    expected={r["scene_id"]:r for r in scenes}
    if not report.get("complete") or report.get("skips") or report["split"]!="val":
        raise ValueError("Incomplete/selected evaluation cannot admit RL")
    maps={arm:{r["scene_id"]:r for r in report["arms"][arm]} for arm in ("baseline","interaction")}
    for arm in maps:
        if len(report["arms"][arm])!=len(maps[arm]) or set(maps[arm])!=set(expected):
            raise ValueError("Evaluation omitted or duplicated planned scenes")
    cells=defaultdict(list);tasks=set()
    for sid,scene in expected.items():
        old,new=maps["baseline"][sid],maps["interaction"][sid]
        if old["policy"]!="baseline" or new["policy"]!="fixed" or new["eta"]!=1:
            raise ValueError("Not the required Round10-versus-interaction comparison")
        for r in (old,new):
            if any(r[k]!=scene[k] for k in ("task","mode","instruction","initial_state_hash","env_config_hash")):
                raise ValueError("Paired initial condition mismatch")
            if r["termination"] not in ("success","step_limit") or r["success"]!=(r["termination"]=="success"):
                raise ValueError("Invalid termination record")
        if old["noise_repeat"]!=new["noise_repeat"]:
            raise ValueError("Evaluation noise protocol mismatch")
        cells[(scene["task"],scene["mode"])].append(float(new["success"])-float(old["success"]))
        tasks.add(scene["task"])
    a=cfg["acceptance"]
    if len(cells)<a["minimum_task_mode_cells"] or any(len(v)<a["minimum_scenes_per_cell"] for v in cells.values()):
        raise ValueError("Insufficient predeclared validation coverage")
    modes=("demo_clean","demo_randomized")
    if any((t,m) not in cells for t in tasks for m in modes):
        raise ValueError("Both modes of every validation task are required")
    task_list=sorted(tasks);rng=np.random.default_rng(261014)
    estimate={m:float(np.mean([np.mean(cells[t,m]) for t in task_list])) for m in modes}
    boot={m:[] for m in modes};overall=[]
    for _ in range(a["bootstrap_samples"]):
        chosen=rng.choice(task_list,len(task_list),replace=True)
        sample={}
        for m in modes:
            sample[m]=float(np.mean([np.mean(rng.choice(cells[t,m],len(cells[t,m]),replace=True)) for t in chosen]))
            boot[m].append(sample[m])
        overall.append(float(np.mean(list(sample.values()))))
    ci={m:np.quantile(boot[m],[.025,.975]).tolist() for m in modes}
    total_ci=np.quantile(overall,[.025,.975]).tolist()
    passed=total_ci[0]>=-a["success_margin"] and all(ci[m][0]>=-a["mode_margin"] for m in modes)
    return dict(passed=passed,cells=len(cells),scenes=len(scenes),
                mean_difference=float(np.mean(list(estimate.values()))),mode_difference=estimate,
                confidence_interval=total_ci,mode_confidence_interval=ci,
                scope="paired pilot noninferiority; not a full benchmark claim")

def assess(cfg,student,sensitivity,evaluation):
    digest=sha256(student)
    p=load_checkpoint(student)
    if p["kind"]!="interaction" or p["variant"]!="query_gate" or p["step"]<=0:
        raise ValueError("Need a trained query+gate Student")
    scale=p["student"]["gate.scales"]
    if not scale.isfinite().all() or not scale.count_nonzero():
        raise ValueError("Goal->Local gate is still zero/nonfinite")
    s=read_json(sensitivity);e=read_json(evaluation)
    if s.get("kind")!="interaction_sensitivity" or not s.get("passed") or s["student_sha256"]!=digest:
        raise ValueError("Missing matching, measured gate sensitivity")
    if e.get("kind")!="paired_evaluation" or e["student_sha256"]!=digest:
        raise ValueError("Evaluation belongs to a different Student")
    manifest=read_json(e["manifest_path"])
    if sha256(e["manifest_path"])!=e["manifest_sha256"]:raise ValueError("Evaluation manifest changed")
    stats=paired_statistics(e,manifest,cfg)
    return dict(format=FORMAT,kind="stage3_admission",control_site="vlm_goal_local",
        passed=stats["passed"],student_sha256=digest,student_path=str(Path(student).resolve()),
        sensitivity_path=str(Path(sensitivity).resolve()),sensitivity_sha256=sha256(sensitivity),
        evaluation_path=str(Path(evaluation).resolve()),evaluation_sha256=sha256(evaluation),
        acceptance_config_hash=canonical_hash(cfg["acceptance"]),statistics=stats)

def validate_admission(cfg,path,student):
    saved=read_json(path)
    if saved.get("kind")!="stage3_admission" or saved["student_path"]!=str(Path(student).resolve()):
        raise ValueError("Wrong Stage III admission artifact")
    actual=assess(cfg,student,saved["sensitivity_path"],saved["evaluation_path"])
    if actual!=saved or not actual["passed"]:
        raise ValueError("Admission missing, stale, altered, or inconclusive")
    return actual

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student",required=True)
    p.add_argument("--sensitivity",required=True);p.add_argument("--evaluation",required=True)
    p.add_argument("--output",required=True);a=p.parse_args()
    result=assess(load_config(a.config),a.student,a.sensitivity,a.evaluation)
    atomic_json(local_output(a.output),result)
    print(result["statistics"])
    if not result["passed"]:raise SystemExit("Held-out noninferiority not established; Stage III stays blocked")
if __name__=="__main__":main()
