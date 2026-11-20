"""Read-only A2 full-evaluation progress display."""
import json,os,sqlite3,time
from pathlib import Path
from after_training_full_eval import EVAL,GPUS,STATE,read_json

def live(pid):
    try:return (Path("/proc")/str(pid)/"stat").read_text().split(") ",1)[1][0] not in ("Z","X")
    except OSError:return False

def main():
    episodes={};protocol_id=None
    while True:
        current_protocol=read_json(EVAL/"protocol.json").get("protocol_sha256")
        if current_protocol!=protocol_id:
            episodes={};protocol_id=current_protocol
        states={};monitor=read_json(STATE)
        if (EVAL/"queue.sqlite").exists():
            try:
                with sqlite3.connect((EVAL/"queue.sqlite").as_uri()+"?mode=ro",uri=True,timeout=2) as db:
                    states=dict(db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
            except sqlite3.Error:pass
        for f in (EVAL/"episodes").glob("*.json"):
            if f.name not in episodes:
                row=read_json(f)
                if "success" in row:episodes[f.name]=bool(row["success"])
        done=states.get("done",0);width=40;filled=width*done//10030
        print("\033[2J\033[H",end="")
        print("A2 LIBERO-Plus full evaluation | Stage II step 2500")
        print(time.strftime("%Y-%m-%d %H:%M:%S %Z"),"|",monitor.get("phase","not started"))
        print("["+"#"*filled+"-"*(width-filled)+f"] {done}/10030 ({100*done/10030:.2f}%)")
        success=sum(episodes.values())
        print(f"Successes: {success}/{len(episodes)} | Completed-only rate: "+(f"{100*success/len(episodes):.2f}%" if episodes else "N/A"))
        print("Queue:",states)
        for i in range(8):
            server=read_json(EVAL/f"server{i}.json");worker=read_json(EVAL/f"worker{i}.json");admission=read_json(EVAL/f"admission{i}.json")
            phase=worker.get("phase") if live(worker.get("pid",0)) else admission.get("phase","waiting_load_slot")
            task=worker.get("task",{})
            print(f"worker{i} GPU{GPUS[i%len(GPUS)]} server={server.get('pid','-')} alive={live(server.get('pid',0))} {phase} task={task.get('suite','-')}/{task.get('task_id','-')}")
        print("\nCtrl-b d detaches; evaluation continues in the background.",flush=True)
        time.sleep(10)

if __name__=="__main__":main()
