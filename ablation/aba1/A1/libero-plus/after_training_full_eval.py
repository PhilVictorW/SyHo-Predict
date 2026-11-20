"""Monitor the existing A1 run and evaluate only after its successful final exit."""
import argparse,fcntl,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
RUN=WORK/"protocol_plus_v1/seed_2026"
STATE=WORK/"logs/after_training_full_eval_status.json"
LOG=WORK/"logs/after_training_full_eval.log"
SETTINGS=json.loads((WORK/"config.json").read_text())["a1_training"]
TEACHER_STEPS=SETTINGS["stage_I_steps"];STUDENT_STEPS=SETTINGS["stage_II_steps"];GPUS=[int(g) for g in os.environ["A1_EVALUATION_GPUS"].split(",")] if os.environ.get("A1_EVALUATION_GPUS") else SETTINGS["physical_gpus"]
if not GPUS or len(GPUS)!=len(set(GPUS)) or any(g<0 or g>7 for g in GPUS):raise ValueError("Invalid evaluation GPU override")
EVAL=RUN/f"evaluation/full_II_step_{STUDENT_STEPS:06d}"


def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {} if default is None else default

def write_json(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(row,indent=2)+"\n");tmp.replace(path)

def live_process(pid,needle):
    try:
        root=Path("/proc")/str(pid)
        state=(root/"stat").read_text().split(") ",1)[1][0]
        return state not in ("Z","X") and needle.encode() in (root/"cmdline").read_bytes()
    except (OSError,ValueError):return False

def gate(pipeline,teacher,student,exit_code,live,files_ready):
    if pipeline.get("phase")!="completed":
        return "waiting_training_recovery" if pipeline.get("phase")=="failed" else "waiting_training"
    if not files_ready:return "waiting_final_checkpoints"
    for marker,stage,step in ((teacher,"I",TEACHER_STEPS),(student,"II",STUDENT_STEPS)):
        if not marker.get("complete") or marker.get("stage")!=stage or marker.get("step")!=step:
            return "invalid_final_checkpoint"
    if live:return "waiting_training_exit"
    if exit_code!=0:return "waiting_successful_pipeline_exit"
    return "ready"

def snapshot(exit_file=None):
    pipeline=read_json(RUN/"pipeline_status.json")
    teacher=RUN/"I"/f"step_{TEACHER_STEPS:06d}";student=RUN/"II"/f"step_{STUDENT_STEPS:06d}"
    workers=[];steps={}
    for stage in ("I","II"):
        states=[read_json(p) for p in (RUN/stage).glob("status_rank*.json")]
        workers += [x["pid"] for x in states if live_process(x.get("pid",0),"a1_local.training")]
        recorded=[]
        try:
            for line in (RUN/stage/"metrics.jsonl").read_text().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                if row.get("split")=="train":recorded.append(row.get("step",0))
        except OSError:pass
        steps[stage]=max(recorded+[x.get("step",0) for x in states]+[0])
    exit_file=Path(exit_file) if exit_file is not None else WORK/"logs/exit_code.txt"
    try:exit_code=int(exit_file.read_text().strip())
    except (OSError,ValueError):exit_code=None
    phase=gate(pipeline,read_json(teacher/"checkpoint_complete.json"),read_json(student/"checkpoint_complete.json"),
               exit_code,workers,all((p/"model.pt").is_file() for p in (teacher,student)))
    return dict(phase=phase,training_stage=pipeline.get("stage"),training_phase=pipeline.get("phase"),
                completed_steps=steps,training_worker_pids=workers,training_exit_code=exit_code,training_exit_file=str(exit_file),
                target_checkpoint=str(student),evaluation_output=str(EVAL))

_last_print=None
_last_time=0

def record(phase,**values):
    global _last_print,_last_time
    values.pop("phase",None)
    row=dict(phase=phase,pid=os.getpid(),time=time.time(),armed=True,tasks=10030,physical_gpus=GPUS,**values)
    write_json(STATE,row)
    key=(phase,values.get("training_stage"),values.get("error"))
    if key!=_last_print or time.monotonic()-_last_time>=60:
        message=json.dumps(row,ensure_ascii=False)
        print(message,flush=True)
        with LOG.open("a") as f:f.write(message+"\n")
        _last_print=key;_last_time=time.monotonic()

def verify_final():
    sys.path[:0]=[str(WORK),str(WORK.parents[3])]
    from ablation.aba1.plus_common.config import load_config
    from a1_local.preflight import check_complete,check_teacher
    from ablation.aba1.plus_common.deployment import validate_checkpoint
    from a1_eval.runtime import validate_final_budget
    cfg=load_config(WORK/"config.json","A1")
    teacher=RUN/"I"/f"step_{TEACHER_STEPS:06d}";student=RUN/"II"/f"step_{STUDENT_STEPS:06d}"
    tm,marker=check_complete(teacher,verify_hash=True);check_teacher(teacher,cfg)
    sm,student_marker=validate_checkpoint(student,cfg,verify_hash=True)
    validate_final_budget(sm,student_marker,cfg)
    proof=dict(time=time.time(),teacher=str(teacher),teacher_step=tm["step"],
               teacher_model_sha256=marker["files"]["model.pt"]["sha256"],
               student=str(student),student_step=sm["step"],
               student_model_sha256=student_marker["files"]["model.pt"]["sha256"])
    write_json(WORK/"logs/final_checkpoint_handoff.json",proof)
    return cfg

def evaluation_progress():
    result=read_json(EVAL/"results.json")
    queue=EVAL/"queue.sqlite"
    if queue.exists():
        try:
            with sqlite3.connect(queue.as_uri()+"?mode=ro",uri=True,timeout=2) as db:
                states=dict(db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
            result=dict(result,completed=states.get("done",0),states=states)
        except sqlite3.Error:pass
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true",help="Arm the monitor; never stops training")
    p.add_argument("--workers",type=int,choices=[1,2,4,5,6],default=6)
    p.add_argument("--poll-seconds",type=int,default=10)
    p.add_argument("--training-exit-file",type=Path,default=WORK/"logs/exit_code.txt")
    a=p.parse_args()
    if not 2<=a.poll_seconds<=60:p.error("Polling must be 2..60 seconds")
    command=[json.loads((WORK/"config.json").read_text())["model_python"],"-B","-u",str(WORK/"evaluate.py"),
             "--scope","full","--checkpoint",str(RUN/"II"/f"step_{STUDENT_STEPS:06d}"),"--gpus",",".join(map(str,GPUS)),"--workers",str(a.workers),"--execute"]
    if not a.execute:
        print(json.dumps(dict(snapshot(a.training_exit_file),armed=False,tasks=10030,workers=a.workers,command=command),indent=2));return
    with (WORK/"logs/.after_training_full_eval.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read_json(STATE)
        existing_pid=previous.get("eval_pid",0)
        adopting=live_process(existing_pid,str(WORK/"evaluate.py"))
        proc=None
        if not adopting:
            while True:
                current=snapshot(a.training_exit_file);phase=current.pop("phase")
                record(phase,workers=a.workers,**current)
                if phase=="invalid_final_checkpoint":raise ValueError("Final marker has the wrong stage/budget")
                if phase=="ready":break
                time.sleep(a.poll_seconds)
            record("verifying_final_checkpoints",workers=a.workers,**current)
            verify_final()
            # Recheck after hashing; an explicit training restart postpones evaluation.
            if snapshot(a.training_exit_file)["phase"]!="ready":raise RuntimeError("Training state changed during checkpoint verification")
            log=WORK/"logs/full_eval.log"
            with log.open("a") as stream:
                proc=subprocess.Popen(command,cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            existing_pid=proc.pid
        while proc.poll() is None if proc is not None else live_process(existing_pid,str(WORK/"evaluate.py")):
            result=evaluation_progress()
            record("evaluating",eval_pid=existing_pid,workers=a.workers,command=command,
                   evaluation_output=str(EVAL),completed=result.get("completed",0),expected=10030,states=result.get("states",{}))
            time.sleep(a.poll_seconds)
        result=read_json(EVAL/"results.json")
        if (proc is not None and proc.returncode!=0) or not result.get("complete") or result.get("completed")!=10030:
            raise RuntimeError(f"Evaluation incomplete; inspect {WORK/'logs/full_eval.log'}")
        record("completed",workers=a.workers,evaluation_output=str(EVAL),completed=10030,
               successes=result["successes"],success_rate=result["success_rate"])

if __name__=="__main__":
    try:main()
    except BlockingIOError:
        print("Another handoff monitor already owns the lock",file=sys.stderr);raise SystemExit(1)
    except Exception as exc:
        record("failed",error=repr(exc));raise
