"""Restart full A2 evaluation without restarting either training stage."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from after_training_full_eval import GPUS,read_json,write_json

WORK=Path(__file__).resolve().parent
LOGS=WORK/"logs"

def record(phase,**values):
    row=dict(time=time.time(),pid=os.getpid(),phase=phase,armed=True,
        stage_II_steps=2500,stage_II_completed=True,evaluation_workers=8,
        evaluation_tasks=10030,evaluation_physical_gpus=GPUS,
        evaluation_worker_gpus=[GPUS[i%len(GPUS)] for i in range(8)],**values)
    write_json(LOGS/"auto_continuation_status.json",row)
    print(json.dumps(row),flush=True)

def main():
    with (LOGS/".auto_continuation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,"-B","-u",str(WORK/"after_training_full_eval.py"),
            "--execute","--workers","8","--poll-seconds","10",
            "--training-exit-file",str(LOGS/"stage_II_exit_code.txt")]
        record("starting_full_evaluation",command=command)
        child=subprocess.Popen(command,cwd=WORK,stdin=subprocess.DEVNULL,pass_fds=(lock.fileno(),))
        while child.poll() is None:
            state=read_json(LOGS/"after_training_full_eval_status.json")
            if state.get("pid")==child.pid:
                record("full_evaluation",monitor_pid=child.pid,eval_pid=state.get("eval_pid"),
                    evaluation_phase=state.get("phase"),completed=state.get("completed",0))
            time.sleep(15)
        record("completed" if child.returncode==0 else "failed_full_evaluation",exit_code=child.returncode)
        return child.returncode

if __name__=="__main__":raise SystemExit(main())
