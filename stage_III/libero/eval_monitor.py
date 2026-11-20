"""Live tmux progress bars for the three fixed Stage III evaluation rounds."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from goal_local.eval_common import read_json, atomic_json, SUITES
from .eval_common import atomic_csv, task_result, task_path

TERMINAL_PHASES={"completed","failed","stopped","incomplete"}
SUITE_NAMES={"libero_spatial":"Spatial","libero_object":"Object","libero_goal":"Goal","libero_10":"Long"}

def alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def update(root):
    manifest=read_json(root/"launch_manifest.json",{});rows=[];now=time.time()
    for item in manifest.get("rounds",[]):
        out=Path(item["output"]);q=read_json(out/"queue_status.json",{})
        protocol=read_json(out/"protocol.json",{})
        tasks=[task_result(out,s,t) for s in SUITES for t in range(10)]
        n=sum(t["episodes"] for t in tasks);k=sum(t["successes"] for t in tasks)
        completed=sum(t["complete"] for t in tasks)
        current=next((t for t in tasks if (t["suite"],t["task_id"])==(q.get("suite"),q.get("task_id"))),{})
        latest=None
        for t in tasks:
            f=task_path(out,t["suite"],t["task_id"])/"episodes.checkpoint.jsonl"
            try:stamp=f.stat().st_mtime
            except FileNotFoundError:continue
            latest=max(latest or stamp,stamp)
        suites={}
        for suite in SUITES:
            group=[t for t in tasks if t["suite"]==suite]
            sn=sum(t["episodes"] for t in group);sk=sum(t["successes"] for t in group)
            suites[suite]=sk/sn if sn else None
            suites[suite+"_episodes"]=sn;suites[suite+"_successes"]=sk
        rows.append(dict(round_index=item["round_index"],gpu=item["gpu"],phase=q.get("phase","queued"),
            heartbeat_age_seconds=round(max(0,now-q["time"]),1) if "time" in q else None,
            driver_alive=alive(q.get("pid")),server_alive=alive(q.get("server_pid")),task_alive=alive(q.get("task_pid")),
            episodes_completed=n,expected_episodes=2000,tasks_completed=completed,successes=k,failures=n-k,
            running_success_rate=k/n if n else None,final_success_rate=k/2000 if completed==40 else None,
            success_videos=sum(bool(t["video"]) for t in tasks),current_suite=q.get("suite"),current_task=q.get("task_id"),
            current_task_episodes=current.get("episodes",0),current_task_successes=current.get("successes",0),
            run_started_at=protocol.get("started_at"),last_episode_age_seconds=round(max(0,now-latest),1) if latest else None,
            **suites,actor_checkpoint=item["actor_checkpoint"],output=str(out)))
    complete=len(rows)==3 and all(r["phase"]=="completed" for r in rows)
    atomic_json(root/"comparison.json",dict(updated_at=now,complete=complete,rounds=rows))
    if rows:atomic_csv(root/"comparison.csv",rows,list(rows[0]))
    return rows,complete

def gpu_stats(gpus):
    try:
        result=subprocess.run(["nvidia-smi","--id="+",".join(map(str,gpus)),
            "--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],
            capture_output=True,text=True,timeout=3,check=True)
        stats={}
        for line in result.stdout.splitlines():
            i,util,used,total=[v.strip() for v in line.split(",")]
            stats[int(i)]=dict(util=util,used=float(used)/1024,total=float(total)/1024)
        return stats
    except (OSError,ValueError,subprocess.SubprocessError):return {}

def bar(n,total,width):
    filled=min(width,max(0,int(width*n/total)))
    return "["+"#"*filled+"-"*(width-filled)+"]"

def percent(value):return "N/A" if value is None else f"{100*value:.2f}%"

def age(seconds):
    if seconds is None:return "N/A"
    seconds=int(seconds)
    if seconds<60:return f"{seconds}s"
    if seconds<3600:return f"{seconds//60}m{seconds%60:02d}s"
    return f"{seconds//3600}h{seconds%3600//60:02d}m"

def render(rows,stats,root,interval,width,height,color):
    def paint(text,code):return f"\033[{code}m{text}\033[0m" if color else text
    compact=height<33
    bar_width=min(42,max(16,width-46))
    total=sum(r["episodes_completed"] for r in rows)
    lines=[paint("LIBERO Stage III | 3 rounds x 40 tasks x 50 episodes", "1;36"),
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  | refresh {interval:g}s | total {total}/6000 episodes",
        "Precision metric: episode SUCCESS RATE (successful / completed episodes).", ""]
    for r in rows:
        phase=r["phase"];hb=r["heartbeat_age_seconds"]
        if phase in TERMINAL_PHASES:
            label=phase.upper();tone="1;32" if phase=="completed" else "1;31"
        elif not r["driver_alive"]:
            label="DRIVER EXITED";tone="1;31"
        elif hb is not None and hb>45:
            label="STALE HEARTBEAT";tone="1;31"
        elif phase=="evaluating" and not r["server_alive"]:
            label="SERVER EXITED";tone="1;31"
        else:
            label="RUNNING" if phase=="evaluating" else phase.upper();tone="1;32" if phase=="evaluating" else "1;33"
        metric=stats.get(r["gpu"])
        gpu=f" | GPU total: {metric['util']}% util, {metric['used']:.1f}/{metric['total']:.1f} GiB" if metric and width>=95 else ""
        lines.append(paint(f"ROUND {r['round_index']}  |  GPU {r['gpu']}  |  {label}",tone)+gpu)
        n=r["episodes_completed"];k=r["successes"]
        lines.append(f"  {bar(n,2000,bar_width)} {n:4d}/2000  ({100*n/2000:6.2f}%)")
        score=percent(r["final_success_rate"] if r["final_success_rate"] is not None else r["running_success_rate"])
        label="FINAL" if r["final_success_rate"] is not None else "RUNNING"
        lines.append(f"  {label} success: {score:>7} ({k}/{n}) | failed {n-k} | tasks {r['tasks_completed']}/40 | videos {r['success_videos']}/40")
        suite=SUITE_NAMES.get(r["current_suite"],r["current_suite"] or "--")
        task="--" if r["current_task"] is None else f"{r['current_task']:02d}"
        cn=r["current_task_episodes"];ck=r["current_task_successes"]
        lines.append(f"  Current: {suite} / task {task} | {cn}/50 episodes | success {ck}/{cn} ({percent(ck/cn if cn else None)})")
        if not compact:
            elapsed=time.time()-r["run_started_at"] if r["run_started_at"] else None
            lines.append(f"  Elapsed {age(elapsed)} | heartbeat {age(hb)} | last result {age(r['last_episode_age_seconds'])} ago")
        lines.append("")
    if not compact:
        lines.append(paint("SUITE SUCCESS RATES   (successes / completed episodes; 500 episodes per suite)","1;36"))
        lines.append(f"{'Suite':10s} {'Round 1 / GPU 0':24s} {'Round 2 / GPU 2':24s} {'Round 3 / GPU 5':24s}")
        for suite in SUITES:
            cells=[f"{percent(r[suite]):>7} ({r[suite+'_successes']:3d}/{r[suite+'_episodes']:<3d})" for r in rows]
            lines.append(f"{SUITE_NAMES[suite]:10s} "+" ".join(f"{cell:24s}" for cell in cells))
        lines.append("")
    lines += ["Partial success rates are provisional; final scores require all 2000 episodes per round.",
              "Detach and keep running: Ctrl-b, then d. | Results: "+root.name]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--once",action="store_true");p.add_argument("--interval",type=float,default=5)
    a=p.parse_args()
    if a.interval<1:p.error("--interval must be at least 1 second")
    if not (a.root/"launch_manifest.json").is_file():p.error("Missing launch_manifest.json")
    interactive=sys.stdout.isatty() and not a.once
    color=interactive and "NO_COLOR" not in os.environ
    if interactive:print("\033[?25l",end="",flush=True)
    try:
        while True:
            try:
                rows,_=update(a.root);stats=gpu_stats([r["gpu"] for r in rows])
                size=shutil.get_terminal_size((120,45))
                content=render(rows,stats,a.root,a.interval,size.columns,size.lines,color)
            except Exception as exc:
                content=f"LIBERO evaluation monitor | {time.strftime('%Y-%m-%d %H:%M:%S')}\nMonitor read error: {exc!r}\nRetrying; evaluation processes are independent."
                if a.once:raise
            if interactive:print("\033[H\033[J",end="")
            print(content,flush=True)
            if a.once:return
            time.sleep(a.interval)
    except KeyboardInterrupt:pass
    finally:
        if interactive:print("\033[?25h\033[0m",end="",flush=True)
if __name__=="__main__":main()
