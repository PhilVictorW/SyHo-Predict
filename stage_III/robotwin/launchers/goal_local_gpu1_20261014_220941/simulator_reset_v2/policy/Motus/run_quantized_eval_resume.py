#!/usr/bin/env python3
"""Resume-aware Motus quantized evaluation scheduler with OOM retry."""

import argparse
import getpass
import json
import os
from pathlib import Path
from collections import deque
import subprocess
import time
import signal

from run_quantized_eval import (
    load_config,
    load_tasks,
    write_json,
    query_free_mem_mib,
    start_job,
    finalize_job,
    summarize_results,
    parse_success_from_log,
    parse_success_from_result,
    is_oom_log,
    is_oom_log_since,
    now_string,
    POLICY_NAME,
)

OOM_WAITING_STATUS = "oom_waiting"
TERMINAL_STATUSES = {"completed", "failed", "skipped_timeout", "interrupted"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="policy/Motus/paths_config.yml")
    parser.add_argument("--tasks-file", default="tasks_all.txt")
    parser.add_argument("--gpus", default="0,1,2,3,6")
    parser.add_argument("--min-free-mem-mib", type=int, default=43000)
    parser.add_argument("--stall-timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-num", type=int, default=None)
    parser.add_argument(
        "--retry-oom-only",
        action="store_true",
        help="Only run tasks whose latest status/log indicates CUDA OOM.",
    )
    return parser.parse_args()


def list_eval_processes():
    user = getpass.getuser()
    try:
        out = subprocess.check_output(
            ["ps", "-u", user, "-o", "pid,ppid,pgid,sid,stat,etime,cmd"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    procs = {}
    for line in out.splitlines():
        if "script/eval_policy.py" not in line or "--task_name" not in line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, _ppid, pgid, _sid, stat, etime, cmd = parts
        tokens = cmd.split()
        task = None
        if "--task_name" in tokens:
            i = tokens.index("--task_name")
            if i + 1 < len(tokens):
                task = tokens[i + 1]
        if task:
            procs[task] = {"pid": int(pid), "pgid": int(pgid), "stat": stat, "etime": etime, "cmd": cmd}
    return procs


def terminate_pid_group(pid_or_pgid):
    try:
        os.killpg(int(pid_or_pgid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            os.kill(int(pid_or_pgid), signal.SIGTERM)
        except Exception:
            return
    time.sleep(10)
    try:
        os.killpg(int(pid_or_pgid), signal.SIGKILL)
    except Exception:
        pass


def status_payload(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def status_path_for(quant_root, task):
    return quant_root / task / "status.json"


def normalize_log_stamp(stamp=None):
    return (stamp or now_string()).replace(" ", "_").replace(":", "")


def resolve_log_dirs(result_root, previous_state, retry_oom_only, stamp=None):
    candidate_log_dir = Path(
        previous_state.get("log_dir")
        or result_root / f"logs_quantized_resume_{normalize_log_stamp(stamp)}"
    )
    if retry_oom_only:
        run_log_dir = result_root / f"logs_quantized_oom_retry_{normalize_log_stamp(stamp)}"
        return candidate_log_dir, run_log_dir
    return candidate_log_dir, candidate_log_dir


def log_path_for_status(status, log_dir, task):
    return Path(status.get("log_path") or log_dir / f"{task}.log")


def infer_latest_attempt_offset(log_path, task):
    path = Path(log_path)
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    marker = f"| task={task} |"
    marker_idx = text.rfind(marker)
    if marker_idx < 0:
        return 0
    header_idx = text.rfind("\n===== ", 0, marker_idx)
    if header_idx < 0:
        header_idx = text.rfind("===== ", 0, marker_idx)
    return max(header_idx, 0)


def status_attempt_offset(status, log_path, task):
    if status.get("log_start_offset") is not None:
        try:
            return int(status.get("log_start_offset") or 0)
        except Exception:
            pass
    return infer_latest_attempt_offset(log_path, task)


def status_attempt_has_oom(status, log_path, task):
    return is_oom_log_since(log_path, status_attempt_offset(status, log_path, task))


def job_attempt_has_oom(job):
    return is_oom_log_since(job.log_path, getattr(job, "log_start_offset", 0))


def metrics_path_for(quant_root, task):
    return quant_root / task / "metrics.json"


def remove_stale_metrics(quant_root, task):
    metrics_path = metrics_path_for(quant_root, task)
    if metrics_path.exists():
        try:
            metrics_path.unlink()
        except Exception:
            pass


def is_oom_retry_candidate(task, status, log_path):
    state = status.get("status")
    if state == OOM_WAITING_STATUS:
        return True
    if status.get("oom_detected") and state != "completed":
        return True
    if state == "failed" and status_attempt_has_oom(status, log_path, task):
        return True
    return False


def is_completed_by_outputs(result_root, task, task_config, ckpt_label, log_path):
    result_data = parse_success_from_result(result_root, task, task_config, ckpt_label)
    log_data = parse_success_from_log(log_path)
    if result_data is not None:
        return True
    if log_data is not None and int(log_data.get("episodes", 0)) >= 30:
        return True
    return False


def close_job_log(job):
    try:
        job.log_handle.close()
    except Exception:
        pass


def mark_oom_waiting(status_path, task, gpu, log_path, quant_root, exit_code=None, reason=None, start_time=None, extra=None):
    previous = status_payload(status_path)
    attempts = int(previous.get("oom_attempts") or 0) + 1
    remove_stale_metrics(quant_root, task)
    payload = {
        "task_name": task,
        "status": OOM_WAITING_STATUS,
        "gpu": gpu,
        "exit_code": exit_code,
        "reason": reason or "CUDA OOM detected; task will be retried when a GPU has enough free memory",
        "oom_detected": True,
        "oom_attempts": attempts,
        "last_oom_at": now_string(),
        "log_path": str(log_path),
    }
    if start_time is not None:
        payload["started_at_epoch"] = start_time
        payload["elapsed_seconds"] = round(time.time() - start_time, 3)
    if extra:
        payload.update(extra)
    write_json(status_path, payload)
    return payload


def finalize_oom_job(job, quant_root, exit_code=None, reason=None):
    close_job_log(job)
    return mark_oom_waiting(
        job.status_path,
        job.task,
        job.gpu,
        job.log_path,
        quant_root,
        exit_code=exit_code,
        reason=reason,
        start_time=job.start_time,
    )


def enqueue_task(pending, pending_set, task):
    if task in pending_set:
        return
    pending.append(task)
    pending_set.add(task)


def dequeue_task(pending, pending_set):
    task = pending.popleft()
    pending_set.discard(task)
    return task


def build_initial_pending(tasks, quant_root, log_dir, retry_oom_only):
    pending = deque()
    pending_set = set()
    for task in tasks:
        status_path = status_path_for(quant_root, task)
        status = status_payload(status_path)
        log_path = log_path_for_status(status, log_dir, task)
        state = status.get("status")
        if retry_oom_only:
            if is_oom_retry_candidate(task, status, log_path):
                enqueue_task(pending, pending_set, task)
            continue
        if not state or state == OOM_WAITING_STATUS or is_oom_retry_candidate(task, status, log_path):
            enqueue_task(pending, pending_set, task)
    return pending, pending_set


def inherited_running_exists(quant_root):
    return any(status_payload(p).get("status") == "running" for p in quant_root.glob("*/status.json"))


def write_live_state(run_state_path, tasks, pending, running_new, quant_root, gpu_ids, min_free_mem_mib, log_dir, retry_oom_only):
    status_entries = [status_payload(p) for p in quant_root.glob("*/status.json")]
    running = {
        str(s.get("gpu")): p.parent.name
        for p in quant_root.glob("*/status.json")
        for s in [status_payload(p)]
        if s.get("status") == "running" and s.get("gpu") is not None
    }
    oom_waiting = sorted(
        p.parent.name
        for p in quant_root.glob("*/status.json")
        for s in [status_payload(p)]
        if s.get("status") == OOM_WAITING_STATUS
    )
    finished = sum(1 for s in status_entries if s.get("status") in TERMINAL_STATUSES)
    write_json(run_state_path, {
        "status": "running",
        "updated_at": now_string(),
        "tasks_total": len(tasks),
        "tasks_pending": len(pending),
        "tasks_running": running,
        "tasks_finished": finished,
        "oom_waiting_tasks": oom_waiting,
        "oom_waiting_count": len(oom_waiting),
        "gpus": gpu_ids,
        "min_free_mem_mib": min_free_mem_mib,
        "log_dir": str(log_dir),
        "quantization_dir": str(quant_root),
        "scheduler": "resume_oom_retry",
        "retry_oom_only": retry_oom_only,
        "running_new_count": len(running_new),
    })


def main():
    args = parse_args()
    config_path = Path(args.config_file).resolve()
    policy_dir = config_path.parent
    config = load_config(config_path)
    if args.result_root:
        config["result_root"] = args.result_root

    task_path = Path(args.tasks_file)
    if not task_path.is_absolute():
        task_path = policy_dir / task_path
    tasks = load_tasks(task_path)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    result_root = Path(config.get("result_root") or "/mnt/hdd/wangchaoyang/results/Motus")
    quant_root = result_root / "quantization"
    previous_state_path = result_root / "run_state.json"
    previous_state = status_payload(previous_state_path)
    candidate_log_dir, log_dir = resolve_log_dirs(result_root, previous_state, args.retry_oom_only)
    log_dir.mkdir(parents=True, exist_ok=True)
    quant_root.mkdir(parents=True, exist_ok=True)

    task_config = args.task_config or str(config.get("task_config", "demo_randomized_motus"))
    ckpt_label = Path(str(config["checkpoint_path"])).name
    run_state_path = result_root / "run_state.json"

    running_new = {}
    pending, pending_set = build_initial_pending(tasks, quant_root, candidate_log_dir, args.retry_oom_only)

    while pending or running_new or inherited_running_exists(quant_root):
        eval_procs = list_eval_processes()

        for gpu, job in list(running_new.items()):
            ret = job.proc.poll()
            if ret is not None:
                if ret != 0 and job_attempt_has_oom(job):
                    finalize_oom_job(job, quant_root, ret)
                    enqueue_task(pending, pending_set, job.task)
                else:
                    finalize_job(job, "completed" if ret == 0 else "failed", ret)
                running_new.pop(gpu, None)
                continue

            if job_attempt_has_oom(job):
                terminate_pid_group(job.proc.pid)
                finalize_oom_job(job, quant_root, job.proc.poll(), "CUDA OOM detected in log; process terminated and queued for retry")
                enqueue_task(pending, pending_set, job.task)
                running_new.pop(gpu, None)
                continue

            mtime = job.log_path.stat().st_mtime if job.log_path.exists() else job.start_time
            if mtime > job.last_log_mtime:
                job.last_log_mtime = mtime
                job.last_progress_time = time.time()
            elif time.time() - job.last_progress_time > args.stall_timeout_sec:
                terminate_pid_group(job.proc.pid)
                finalize_job(job, "skipped_timeout", job.proc.poll(), f"no log progress for {args.stall_timeout_sec} seconds")
                running_new.pop(gpu, None)

        for status_file in list(quant_root.glob("*/status.json")):
            task = status_file.parent.name
            status = status_payload(status_file)
            if status.get("status") != "running":
                continue
            if any(job.task == task for job in running_new.values()):
                continue
            log_path = log_path_for_status(status, log_dir, task)
            proc = eval_procs.get(task)
            if status_attempt_has_oom(status, log_path, task):
                if proc is not None:
                    terminate_pid_group(proc["pgid"])
                mark_oom_waiting(
                    status_file,
                    task,
                    status.get("gpu"),
                    log_path,
                    quant_root,
                    exit_code=None,
                    reason="CUDA OOM detected in inherited process; task queued for retry",
                    extra={"external_pid": proc["pid"] if proc else None},
                )
                enqueue_task(pending, pending_set, task)
                continue
            if proc is None:
                if is_completed_by_outputs(result_root, task, task_config, ckpt_label, log_path):
                    payload = dict(status)
                    payload.update({
                        "status": "completed",
                        "exit_code": 0,
                        "finished_at": now_string(),
                        "reason": "completed by existing eval process before resume scheduler observed exit",
                    })
                    write_json(status_file, payload)
                else:
                    payload = dict(status)
                    payload.update({
                        "status": "failed",
                        "exit_code": None,
                        "finished_at": now_string(),
                        "reason": "running process disappeared before completion could be verified",
                    })
                    write_json(status_file, payload)
                continue

            started_at = status.get("started_at_epoch") or (log_path.stat().st_ctime if log_path.exists() else time.time())
            mtime = log_path.stat().st_mtime if log_path.exists() else started_at
            last_progress = status.get("last_progress_epoch") or mtime
            if mtime > float(last_progress):
                payload = dict(status)
                payload["last_progress_epoch"] = mtime
                payload["external_pid"] = proc["pid"]
                write_json(status_file, payload)
            elif time.time() - float(last_progress) > args.stall_timeout_sec:
                terminate_pid_group(proc["pgid"])
                payload = dict(status)
                payload.update({
                    "status": "skipped_timeout",
                    "exit_code": None,
                    "finished_at": now_string(),
                    "reason": f"no log progress for {args.stall_timeout_sec} seconds",
                    "external_pid": proc["pid"],
                })
                write_json(status_file, payload)

        active_gpus = set(running_new.keys())
        for status_file in quant_root.glob("*/status.json"):
            status = status_payload(status_file)
            if status.get("status") == "running" and status.get("gpu") is not None:
                active_gpus.add(str(status.get("gpu")))

        for gpu in gpu_ids:
            if not pending or gpu in active_gpus:
                continue
            if query_free_mem_mib(gpu) < args.min_free_mem_mib:
                continue
            task = dequeue_task(pending, pending_set)
            running_new[gpu] = start_job(gpu, task, config, args, log_dir, quant_root)
            active_gpus.add(gpu)

        write_live_state(run_state_path, tasks, pending, running_new, quant_root, gpu_ids, args.min_free_mem_mib, log_dir, args.retry_oom_only)
        summarize_results(tasks, config, args, log_dir, quant_root)

        if not pending and not running_new and not inherited_running_exists(quant_root):
            break
        time.sleep(args.poll_sec)

    summarize_results(tasks, config, args, log_dir, quant_root)
    write_json(run_state_path, {
        "status": "completed",
        "finished_at": now_string(),
        "tasks_total": len(tasks),
        "log_dir": str(log_dir),
        "quantization_dir": str(quant_root),
        "result_txt": str(result_root / "result.txt"),
        "quantization_summary": str(quant_root / "summary.json"),
        "scheduler": "resume_oom_retry",
        "retry_oom_only": args.retry_oom_only,
    })


if __name__ == "__main__":
    main()
