"""Explicit bounded G-route workflow; default prints the plan and runs nothing."""
import argparse
import subprocess
import time
from pathlib import Path
from .common import DEFAULT_CONFIG,ROOT,WORK,atomic_json,load_config,local_output,read_json,require_execute
from .runtime import child_env

def commands(a,c):
    out=Path(a.run_dir).resolve();man=out/"manifests"
    py=c["runtime"]["model_python"];cfg=str(Path(a.config).resolve())
    def module(name,*args):
        return [py,"-m","stage_III.robotwin."+name,"--config",cfg,*map(str,args)]
    gpu=["--gpu",str(a.gpu),"--execute"]
    student=out/"interaction/checkpoints"/("student_%06d.pt"%c["interaction"]["steps"])
    cmds=[
        ("offline_check",module("check","--report",out/"static_check.json")),
        ("candidate_manifest",module("make_manifests","--output",man/"candidates.json")),
        ("interaction",module("prepare_interaction","--run-dir",out/"interaction",*gpu)),
        ("sensitivity",module("sensitivity","--student",student,"--output",out/"sensitivity.json",*gpu))]
    for split,n in (("train",2),("val",3),("test",3)):
        cmds.append(("prepare_"+split,module("prepare_scenes","--candidates",man/"candidates.json",
            "--split",split,"--output",man/("prepared_"+split+".json"),
            "--limit-cells",a.cells,"--scenes-per-cell",n,*gpu)))
    cmds += [
        ("paired_validation",module("evaluate","--student",student,"--manifest",man/"prepared_val.json",
                                    "--output",out/"paired_validation",*gpu)),
        ("admission",module("acceptance","--student",student,"--sensitivity",out/"sensitivity.json",
                             "--evaluation",out/"paired_validation/report.json","--output",out/"admission.json")),
        ("stage_III",module("train","--student",student,"--admission",out/"admission.json",
             "--train-manifest",man/"prepared_train.json","--val-manifest",man/"prepared_val.json",
             "--test-manifest",man/"prepared_test.json","--run-dir",out/"rl",*gpu))]
    if getattr(a,"final_evaluation",False):
        cmds.append(("final_comparison",module("finalize","--student",student,"--rl-run",out/"rl",
            "--val-manifest",man/"prepared_val.json","--test-manifest",man/"prepared_test.json",
            "--output",out/"final_comparison",*gpu)))
    return cmds

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--config",default=str(DEFAULT_CONFIG))
    p.add_argument("--run-dir",default=str(WORK/"runs/goal_local_pilot"))
    p.add_argument("--cells",type=int,default=16);p.add_argument("--gpu");p.add_argument("--execute",action="store_true")
    p.add_argument("--final-evaluation",action="store_true")
    a=p.parse_args();c=load_config(a.config)
    if not 16<=a.cells<=100 or a.cells%2:raise ValueError("Use an even 16..100 task-mode cells")
    out=local_output(a.run_dir);cmds=commands(a,c)
    if not a.execute:
        import shlex
        print("PLAN ONLY: no processes launched. Supply --gpu N --execute only when training is authorized.")
        for name,cmd in cmds:print(name+": "+shlex.join(cmd))
        return
    require_execute(a)
    if out.exists():raise ValueError("Workflow run exists; resume the explicit phase from its completed checkpoint")
    out.mkdir(parents=True)
    for name,cmd in cmds:
        atomic_json(out/"workflow_status.json",dict(phase=name,status="running",time=time.time()))
        with (out/(name+".log")).open("ab") as log:
            result=subprocess.run(cmd,cwd=ROOT,env=child_env(c,a.gpu,out),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(out/"workflow_status.json",dict(phase=name,status="stopped",returncode=result.returncode))
            raise SystemExit("Workflow stopped at "+name+"; inspect its log")
        if name=="interaction":
            status=read_json(out/"interaction/status.json")
            if status["phase"]!="interaction_complete_pending_online_acceptance":raise SystemExit("Interaction stopped early; refusing to advance")
    atomic_json(out/"workflow_status.json",dict(phase="pilot_completed",
                status="comparison_completed" if a.final_evaluation else "pending_final_comparison",
                note="Use complete coverage and matched controls to judge improvement."))
if __name__=="__main__":main()
