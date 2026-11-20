"""Sequential A1 Stage I -> Stage II with an explicit execution boundary."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .config_runtime import *
from .preflight import plan,resources,check_complete,check_teacher,inspect_dataset
from ablation.aba1.plus_common.environment import subprocess_environment

WORK=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(WORK/"config.json"))
    p.add_argument("--execute",action="store_true")
    p.add_argument("--stage",choices=["both","I","II"],default="both")
    p.add_argument("--gpus",default="4")
    a=p.parse_args()
    if a.gpus!="4":p.error("This run is pinned to physical GPU 4, six processes")
    cfg=load_config(a.config,"A1")
    stages=["I","II"] if a.stage=="both" else [a.stage]
    summary=dict(plans=[plan(cfg,s) for s in stages],resources=resources(),
                 training_started=False,automatic_stage_transition=a.stage=="both")
    print(json.dumps(summary,indent=2),flush=True)
    if not a.execute:return
    if summary["resources"]["disk_free_gib"]<28:
        raise OSError("Need at least 28 GiB free for Teacher + two Student models + an atomic save")
    data=inspect_dataset(cfg)
    output=run_root(cfg);output.mkdir(parents=True,exist_ok=True)
    with (output/".pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=subprocess_environment(cfg,"4",output/"runtime_cache")
        env["PYTHONPATH"]=str(WORK)+os.pathsep+env["PYTHONPATH"]
        env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
        env["TORCH_GLOO_LAZY_INIT"]="1"
        env["TMPDIR"]="/tmp"  # AF_UNIX paths must fit Linux sockaddr_un.
        atomic_json(output/"pipeline_plan.json",dict(summary,data=data,config=cfg,code_identity=code_identity()))
        for stage in stages:
            final=checkpoint_path(cfg,stage,stage_spec(cfg,stage)["stop_after_updates"])
            # A completed stage may be reused only if metadata and every file hash agree.
            if final.exists():
                meta,_=check_complete(final,verify_hash=True)
                if meta["config"]!=cfg or meta["extra"].get("stage")!=stage:
                    raise ValueError("Existing completed stage identity differs")
                if stage=="I":check_teacher(final,cfg)
                continue
            if (stage_root(cfg,stage)/"arguments.json").exists():
                raise RuntimeError("Incomplete prior run exists; inspect it before restarting (weights-only snapshots)")
            plan(cfg,stage,require_sources=True)
            cmd=[cfg["model_python"],"-B","-m","a1_local.training","--config",str(Path(a.config).resolve()),
                 "--stage",stage,"--gpus","4","--workers","1","--cpu-threads","2",
                 "--offload-saved-tensors","--share-frozen-encoders",
                 "--keep-models","1" if stage=="I" else "2","--execute"]
            atomic_json(output/"pipeline_status.json",dict(phase="training",stage=stage,time=time.time(),command=cmd))
            try:
                subprocess.run(cmd,env=env,cwd=WORK,check=True)
                check_complete(final,verify_hash=True)
            except BaseException as exc:
                atomic_json(output/"pipeline_status.json",dict(phase="failed",stage=stage,error=repr(exc),time=time.time()))
                raise
        atomic_json(output/"pipeline_status.json",dict(phase="completed",teacher_step=5000,student_step=2500,
                                                       time=time.time()))
        print("A1 completed: Teacher5000 -> Student2500",flush=True)

if __name__=="__main__":main()
