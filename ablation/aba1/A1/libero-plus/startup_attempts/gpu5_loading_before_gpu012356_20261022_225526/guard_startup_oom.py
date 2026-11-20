"""Reduce ranks only after a real CUDA OOM before the stage's first recorded update."""
import json,os,subprocess,time
from pathlib import Path

WORK=Path(__file__).resolve().parent
RUN=WORK/"protocol_plus_v1/seed_2026"
LOG=WORK/"logs/train_gpu4.log"
STATE=WORK/"logs/oom_guard_status.json"

def record(**row):
    row["time"]=time.time()
    tmp=STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(row,indent=2)+"\n");tmp.replace(STATE)
    print(json.dumps(row),flush=True)

def main():
    world=6;offset=0
    record(phase="watching",world=world)
    marker=WORK/"logs/exit_code.txt"
    while not marker.exists():time.sleep(10)
    code=int(marker.read_text().strip())
    while code:
        with LOG.open() as f:f.seek(offset);text=f.read()
        if not ("CUDA out of memory" in text or "torch.OutOfMemoryError" in text):
            record(phase="stopped_non_oom",exit_code=code,world=world);return
        stage=json.loads((RUN/"pipeline_status.json").read_text())["stage"]
        folder=RUN/stage
        metrics=folder/"metrics.jsonl"
        rows=[json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
        if any(row.get("split")=="train" for row in rows):
            # Weights-only checkpoints cannot resume Adam/RNG exactly. Never silently reset a trained run.
            record(phase="stopped_after_updates",stage=stage,world=world,
                   reason="Exact optimizer recovery unavailable; preserve this trained run");return
        if world==1:
            record(phase="stopped_oom_at_one_rank",stage=stage);return
        for path in folder.glob("status_rank*.json"):
            pid=json.loads(path.read_text())["pid"]
            proc=Path("/proc")/str(pid)/"cmdline"
            if proc.exists() and b"a1_local.training" in proc.read_bytes():
                record(phase="stopped_worker_still_alive",pid=pid);return
        archived=WORK/"startup_attempts"/f"oom_{stage}_{world}ranks_{int(time.time())}"
        archived.mkdir(parents=True)
        folder.rename(archived/stage)
        world={6:4,4:2,2:1}[world]
        record(phase="retrying",stage=stage,world=world,accumulation=32//world,archive=str(archived))
        env=os.environ.copy();env["A1_WORLD_SIZE"]=str(world)
        offset=LOG.stat().st_size
        code=subprocess.run(["bash",str(WORK/"run_logged.sh")],cwd=WORK,env=env).returncode
    record(phase="completed",world=world)

if __name__=="__main__":main()
