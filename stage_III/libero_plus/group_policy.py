"""Terminal group outcomes: timed-out groups never become GRPO rewards."""
from __future__ import annotations
import json
import time
from pathlib import Path


def is_skipped(value) -> bool:
    return isinstance(value, dict) and value.get("status") == "skipped_timeout"


def archive_partial(run_dir: Path, group_id: int, n: int) -> int:
    target = run_dir / "skipped_trajectories" / f"group_{group_id:04d}"
    for idx in range(n):
        src = run_dir / "trajectories" / f"group_{group_id:04d}_traj_{idx:02d}.json"
        if src.exists():
            target.mkdir(parents=True, exist_ok=True)
            dest = target / src.name
            if dest.exists():
                raise RuntimeError(f"refusing to overwrite archived trajectory: {dest}")
            src.replace(dest)
    return len(list(target.glob("*.json"))) if target.exists() else 0


def load_skip(run_dir: Path, group_id: int, task: dict, actor_hash: str, n: int):
    path = run_dir / "skipped_groups" / f"group_{group_id:04d}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (not is_skipped(record) or record.get("group_id") != group_id
            or record.get("suite") != task["suite"] or record.get("task_id") != task["task_id"]
            or record.get("actor_sha256") != actor_hash):
        raise RuntimeError(f"skip record identity mismatch: {path}")
    archive_partial(run_dir, group_id, n)
    return record


def record_skip(run_dir: Path, group_id: int, task: dict, actor_hash: str,
                n: int, reason: str) -> dict:
    paths = [run_dir / "trajectories" / f"group_{group_id:04d}_traj_{i:02d}.json" for i in range(n)]
    record = dict(status="skipped_timeout", group_id=group_id, suite=task["suite"],
                  task_id=task["task_id"], task_name=task["name"], actor_sha256=actor_hash,
                  reason=reason, time=time.time(), partial_trajectories=sum(p.exists() for p in paths),
                  trajectories_used=0)
    path = run_dir / "skipped_groups" / f"group_{group_id:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    tmp.replace(path)
    archive_partial(run_dir, group_id, n)
    print(f"[Stage III][超时跳过] 组 {group_id}：{reason}；整组不参与更新，"
          f"已隔离保存 {record['partial_trajectories']} 条轨迹，继续下一组。", flush=True)
    return record


def partition_groups(outcomes, n: int):
    valid, skipped = [], []
    for outcome in outcomes:
        if is_skipped(outcome):
            skipped.append(outcome)
        elif isinstance(outcome, list) and len(outcome) == n and all(isinstance(x, dict) for x in outcome):
            valid.append(outcome)
        else:
            raise RuntimeError("missing or incomplete GRPO group result")
    return valid, skipped


def summary_counts(rows):
    skipped = sum(is_skipped(row) for row in rows)
    return dict(groups_processed=len(rows), groups_completed=len(rows)-skipped, groups_skipped=skipped)
