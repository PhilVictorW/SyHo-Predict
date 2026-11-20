"""Paired closed-loop baseline, interaction and Actor evaluation."""
import argparse
from pathlib import Path
from .common import (DEFAULT_CONFIG,FORMAT,atomic_json,load_config,local_output,read_json,
                     require_execute,sha256)
from .make_manifests import validate_manifest
from .engine import start_server,collect
from .runtime import stop_owned
from .rpc import call

def evaluate(cfg,config_path,student,manifest,out,gpu,actor=None,policies=None,session=None):
    out=local_output(out);out.mkdir(parents=True,exist_ok=True)
    m=read_json(manifest);rows=validate_manifest(m)
    if m["split"]=="train":raise ValueError("Evaluation requires val or test")
    policies=policies or ([dict(name="baseline",policy="baseline"),
                          dict(name="interaction",policy="fixed",eta=1.)] if not actor else
                         [dict(name="interaction",policy="fixed",eta=1.),
                          dict(name="actor_mean",policy="actor_mean")])
    proc,url,info=session if session is not None else start_server(cfg,config_path,student,gpu,out)
    reports={};skips=[]
    try:
        ah=None
        if actor:
            ah=call(url,"/actor",dict(path=str(Path(actor).resolve())))["actor_sha256"]
        for arm in policies:
            result=[]
            for j,row in enumerate(rows):
                request=dict(url=url,student_sha256=info["student_sha256"],actor_sha256=ah,
                             policy=arm["policy"],eta=arm.get("eta"),noise_repeat=0)
                value=collect(cfg,config_path,row,request,j,1,gpu,out/arm["name"]/row["scene_id"])
                if value["status"]=="completed":result.append(value["trajectories"][0])
                else:skips.append(dict(arm=arm["name"],scene_id=row["scene_id"],reason=value["reason"]))
                atomic_json(out/"progress.json",dict(arm=arm["name"],completed=j+1,total=len(rows),
                                successes=sum(int(r["success"]) for r in result),skipped=len(skips)))
            reports[arm["name"]]=result
        report=dict(format=FORMAT,kind="paired_evaluation",student_sha256=info["student_sha256"],
                actor_sha256=ah,manifest_sha256=sha256(manifest),manifest_path=str(Path(manifest).resolve()),
                split=m["split"],requested_scenes=len(rows),requested_cells=len({(r["task"],r["mode"]) for r in rows}),
                arms=reports,skips=skips,complete=not skips and all(len(v)==len(rows) for v in reports.values()))
        atomic_json(out/"report.json",report)
        return report
    finally:
        if session is None:stop_owned(proc)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student",required=True)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--actor")
    p.add_argument("--fixed-etas",nargs="+",type=float);p.add_argument("--gpu");p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);cfg=load_config(a.config)
    policies=None
    if a.fixed_etas:
        if any(not 0<=v<=2 for v in a.fixed_etas):raise ValueError("eta out of range")
        policies=[dict(name="eta_"+str(v),policy="fixed",eta=v) for v in a.fixed_etas]
        if a.actor:policies += [dict(name="actor_mean",policy="actor_mean"),dict(name="actor_sample",policy="actor_sample")]
    evaluate(cfg,str(Path(a.config).resolve()),a.student,a.manifest,a.output,a.gpu,a.actor,policies)
if __name__=="__main__":main()
