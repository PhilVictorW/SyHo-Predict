"""Torch-free summaries for the three fixed Stage III evaluation checkpoints."""
from __future__ import annotations
import csv, io, json, time
from pathlib import Path
from goal_local.eval_common import SUITES, atomic_json, read_json
_CACHE = {}

def task_path(output, suite, task):
    return Path(output) / "tasks" / f"{suite}__task{task:02d}"

def read_episodes(path, task):
    file = Path(path) / "episodes.checkpoint.jsonl"
    if not file.exists(): return {}
    stat = file.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if file in _CACHE and _CACHE[file][0] == signature: return _CACHE[file][1]
    rows = {}
    lines = file.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip(): continue
        try: row = json.loads(line)
        except ValueError:
            if i == len(lines)-1: continue  # Writer may still be appending.
            raise ValueError(f"Invalid completed JSONL line in {file}")
        if row["task_id"] != task or not 0 <= row["episode_idx"] < 50:
            raise ValueError(f"Unexpected episode in {file}")
        if row.get("error"): continue
        key = row["episode_idx"]
        if key in rows and rows[key] != row: raise ValueError(f"Conflicting duplicate in {file}")
        rows[key] = row
    _CACHE[file] = (signature, rows)
    return rows

def task_result(output, suite, task):
    path = task_path(output, suite, task)
    rows = read_episodes(path, task)
    n = len(rows); k = sum(bool(r["success"]) for r in rows.values())
    videos = [p for p in path.glob("*_success.mp4") if p.stat().st_size > 0]
    if len(videos) > 1 or list(path.glob("*_failure.mp4")):
        raise ValueError(f"Video policy violated in {path}")
    return dict(suite=suite, task_id=task, episodes=n, successes=k,
                success_rate=k/n if n else None, complete=n==50 and (bool(videos) or k==0),
                video=str(videos[0]) if videos else None,
                video_status="saved" if videos else "no_success" if n==50 and k==0 else "pending",
                task_description=next(iter(rows.values()), {}).get("task_description"))

def atomic_csv(path, rows, fields):
    stream=io.StringIO(); writer=csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    path=Path(path); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(stream.getvalue()); tmp.replace(path)

def summarize(output, round_index, gpu):
    output=Path(output)
    tasks=[task_result(output,s,t) for s in SUITES for t in range(10)]
    n=sum(t["episodes"] for t in tasks); k=sum(t["successes"] for t in tasks)
    completed=sum(t["complete"] for t in tasks)
    suites={}
    for suite in SUITES:
        group=[t for t in tasks if t["suite"]==suite]
        count=sum(t["episodes"] for t in group); passed=sum(t["successes"] for t in group)
        suites[suite]=dict(episodes=count, expected_episodes=500, successes=passed,
                           success_rate=passed/count if count else None)
    result=dict(round_index=round_index,gpu=gpu,status="complete" if completed==40 else "partial" if n else "queued",
                episodes_completed=n,expected_episodes=2000,successes=k,
                running_success_rate=k/n if n else None,final_success_rate=k/2000 if completed==40 else None,
                tasks_completed=completed,expected_tasks=40,success_videos=sum(bool(t["video"]) for t in tasks),
                suites=suites,tasks=tasks,updated_at=time.time())
    atomic_json(output/"summary.json",result)
    atomic_csv(output/"task_results.csv",tasks,list(tasks[0]))
    atomic_json(output/"videos.json",[dict(suite=t["suite"],task_id=t["task_id"],path=t["video"]) for t in tasks if t["video"]])
    return result
