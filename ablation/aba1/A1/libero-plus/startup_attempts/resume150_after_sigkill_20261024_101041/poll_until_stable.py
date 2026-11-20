"""Read-only training observer; retain polling evidence until five stable updates."""
import argparse
import json
import math
import os
from pathlib import Path
import time

WORK=Path(__file__).resolve().parent
RUN=WORK/"protocol_plus_v1/seed_2026"
LOGS=WORK/"logs"

def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def metrics():
    rows=[]
    try:
        for line in (RUN/"I/metrics.jsonl").read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    return rows

def process(state):
    result=dict(alive=False,cpu_seconds=0,read_gib=0,rss_gib=0)
    if not state.get("pid"):return result
    root=Path("/proc")/str(state["pid"])
    try:
        fields=(root/"stat").read_text().rsplit(") ",1)[1].split()
        args=(root/"cmdline").read_bytes().split(b"\0")
        result.update(alive=fields[0] not in ("Z","X") and b"a1_local.training" in args and str(WORK/"config.json").encode() in args,
                      process_state=fields[0],cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK"),
                      rss_gib=int(fields[21])*os.sysconf("SC_PAGE_SIZE")/2**30)
        values=dict(line.split(":",1) for line in (root/"io").read_text().splitlines())
        result["read_gib"]=int(values["read_bytes"])/2**30
    except (OSError,ValueError,IndexError):pass
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval",type=int,default=30)
    parser.add_argument("--steps",type=int,default=5)
    args=parser.parse_args()
    cfg=read_json(WORK/"config.json");launch=read_json(LOGS/"launch_record.json")
    gpus=cfg["a1_training"]["physical_gpus"]
    previous={}
    while True:
        now=time.time();rows=metrics();train=[r for r in rows if r.get("split")=="train"]
        states=[read_json(RUN/"I"/f"status_rank{rank}.json") for rank in range(6)]
        workers=[]
        for rank,state in enumerate(states):
            info=process(state);old=previous.get(rank,{})
            info.update(rank=rank,pid=state.get("pid"),gpu=state.get("physical_gpu"),phase=state.get("phase"),step=state.get("step",0),micro=state.get("micro"),
                        status_age=round(now-state.get("time",now),1),
                        cpu_delta=round(info["cpu_seconds"]-old.get("cpu_seconds",info["cpu_seconds"]),1),
                        read_delta_gib=round(info["read_gib"]-old.get("read_gib",info["read_gib"]),3))
            previous[rank]=dict(info)
            for key in ("cpu_seconds","read_gib","rss_gib"):info[key]=round(info[key],3)
            workers.append(info)
        audit=read_json(RUN/"I/sync_audit.json")
        loaded=[read_json(p) for p in (RUN/"I").glob("model_load_rank*.json")]
        snapshot=dict(time=now,local_time=time.strftime("%Y-%m-%d %H:%M:%S"),step=train[-1]["step"] if train else 0,
                      loaded_ranks=sorted(r["rank"] for r in loaded if r.get("phase")=="loaded"),
                      validation=next((r for r in reversed(rows) if r.get("split")=="validation"),None),
                      latest_train=train[-1] if train else None,workers=workers,sync_audit=audit)
        failure=None
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip()!=launch["boot_id"]:failure="server_rebooted"
        if read_json(RUN/"pipeline_status.json").get("phase")=="failed":failure="pipeline_failed"
        for state,worker in zip(states,workers):
            if state.get("phase")=="failed":failure=state.get("error","rank_failed")
            if state.get("pid") and not worker["alive"]:failure=f"rank{worker['rank']}_exited"
        for row in rows:
            for name in ("loss","grad_norm","flow"):
                if name in row and not math.isfinite(float(row[name])):failure=f"nonfinite_{name}_at_step{row.get('step')}"
        recent=train[-args.steps:]
        stable=(len(recent)>=args.steps and [r["step"] for r in recent]==list(range(recent[0]["step"],recent[0]["step"]+args.steps))
                and all(w["alive"] and w["step"]>=recent[-1]["step"] and w["phase"]=="training" for w in workers)
                and [w["gpu"] for w in workers]==gpus and audit.get("max_difference")==0)
        snapshot.update(stable=stable and not failure,failure=failure)
        with (LOGS/"stage_I_stability_poll.jsonl").open("a") as stream:stream.write(json.dumps(snapshot)+"\n")
        print(json.dumps(snapshot),flush=True)
        if failure or stable:
            result=dict(snapshot,checked_updates=recent,required_consecutive_updates=args.steps)
            (LOGS/"stage_I_stability_result.json").write_text(json.dumps(result,indent=2)+"\n")
            return 2 if failure else 0
        time.sleep(args.interval)

if __name__=="__main__":raise SystemExit(main())
