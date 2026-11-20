"""Offline validation only: no torch import, model construction or simulator."""
import argparse
import ast
import json
from pathlib import Path
from .common import DEFAULT_CONFIG,ROOT,WORK,atomic_json,base_files,load_config,read_json,sha256

def check(config=DEFAULT_CONFIG):
    c=load_config(config);files=base_files(c);checks={}
    for key,p in files.items():
        if not p.is_file() or p.stat().st_size==0:raise ValueError("Missing base asset: "+str(p))
        checks[key]=dict(path=str(p),bytes=p.stat().st_size)
    base=read_json(files["config"])
    if base["baseline_checkpoint"]!=str(files["checkpoint"]) or base["statistics_path"]!=str(files["statistics"]):
        raise ValueError("Base configuration/normalization mismatch")
    recipe=read_json(ROOT/"robotwin/round7/candidate_recipe.json")
    for ck,rk in (("source_adapter","source_checkpoint"),("teacher_adapter","teacher_checkpoint"),("initializers","initializers")):
        if str(files[ck])!=recipe[rk]:raise ValueError("Does not match Round10 recipe: "+ck)
    for key in ("source_adapter","teacher_adapter"):
        if not (files[key].parent/"checkpoint_complete.json").exists():raise ValueError("Adapter completion marker missing")
    stats=read_json(files["statistics"])["agilex"]["action"]
    if any(len(stats[k])!=16 for k in ("min","max")):raise ValueError("Expected RoboTwin EEF statistics")
    meta=read_json(Path(c["base"]["dataset"])/"meta/info.json")
    if meta["fps"]!=30 or meta["features"]["action"]["shape"]!=[16]:raise ValueError("Dataset is not 30Hz 16D EEF")
    for k in ("model_python","sim_python","sim_root"):
        if not Path(c["runtime"][k]).exists():raise ValueError("Missing runtime: "+k)
    rt=Path(c["runtime"]["sim_root"])
    for name in ("demo_clean","demo_randomized"):
        if not (rt/"task_config"/(name+".yml")).exists():raise ValueError("Missing simulator mode")
    from .make_manifests import build
    candidate=build(c)
    if len({r["reset_seed"] for r in candidate["rows"]})!=len(candidate["rows"]):raise ValueError("Candidate seed collision")
    syntax={}
    for p in sorted(WORK.rglob("*.py")):
        if any(k in p.parts for k in ("runs","cache")):continue
        ast.parse(p.read_text(),filename=str(p));syntax[str(p.relative_to(WORK))]=sha256(p)
    checks.update(model_contract=c["model"],candidate_scenes=len(candidate["rows"]),
                  candidate_cells=len({(r["task"],r["mode"]) for r in candidate["rows"]}),
                  syntax_files=syntax,physical_gpu=c["runtime"]["physical_gpu"],
                  disk_weight_hashes="deferred to explicit runtime; large weights were not loaded")
    return dict(status="offline_checks_passed",training_started=False,gpu_model_loaded=False,
                simulator_started=False,checks=checks,
                pending_runtime_checks=["Round10 full-model step0 equality and actual gradients",
                    "posterior target quality screen","interaction sensitivity after training",
                    "RoboTwin repeated-reset identity","paired closed-loop admission",
                    "actual GPU memory/throughput and end-to-end rollout"])
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--config",default=str(DEFAULT_CONFIG))
    p.add_argument("--report",default=str(WORK/"reports/static_check.json"))
    a=p.parse_args();result=check(a.config)
    from .common import local_output
    atomic_json(local_output(a.report),result)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
