#!/usr/bin/env python3
"""GPU-aware RoboTwin Motus evaluation runner with quantization metrics."""

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

POLICY_NAME = "Motus"
OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory",
)
OOM_RETRY_STATUSES = {"oom_waiting", "oom_retrying"}


@dataclass
class Job:
    task: str
    gpu: str
    proc: subprocess.Popen
    log_path: Path
    log_handle: object
    start_time: float
    last_log_mtime: float
    last_progress_time: float
    status_path: Path
    log_start_offset: int = 0


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def query_free_mem_mib(gpu_id):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.strip().splitlines()[0])
    except Exception:
        return 0


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def is_oom_text(text):
    clean_text = strip_ansi(text or "")
    return any(pattern in clean_text for pattern in OOM_PATTERNS)


def is_oom_log(log_path):
    return is_oom_log_since(log_path, 0)


def is_oom_log_since(log_path, start_offset=0):
    path = Path(log_path)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            f.seek(max(int(start_offset or 0), 0))
            text = f.read().decode("utf-8", errors="ignore")
        return is_oom_text(text)
    except Exception:
        return False


def status_allows_success_score(status):
    state = status.get("status")
    if state not in {"completed", "skipped_timeout"}:
        return False
    if state in OOM_RETRY_STATUSES:
        return False
    if status.get("oom_detected") and state != "completed":
        return False
    return True


def status_allows_metric_aggregation(status):
    if status.get("status") in OOM_RETRY_STATUSES:
        return False
    if status.get("oom_detected") and status.get("status") != "completed":
        return False
    return True


def parse_success_from_log(log_path):
    if not log_path.exists():
        return None
    try:
        text = strip_ansi(log_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    matches = re.findall(r"Success rate:\s*(\d+)\s*/\s*(\d+).*?=>\s*([0-9.]+)%", text, re.S)
    if not matches:
        return None
    suc, total, pct = matches[-1]
    return {"successes": int(suc), "episodes": int(total), "success_rate_percent": float(pct)}


def parse_success_from_result(result_root, task, task_config, ckpt_label):
    pattern = result_root / "eval_result" / task / POLICY_NAME / task_config / ckpt_label / "*" / "_result.txt"
    files = sorted(glob.glob(str(pattern)), key=lambda p: os.path.getmtime(p), reverse=True)
    for file_name in files:
        path = Path(file_name)
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                ratio = float(line)
            except ValueError:
                continue
            return {"success_rate_percent": ratio * 100.0, "result_file": str(path)}
    return None


def find_success_video(result_root, task, task_config, ckpt_label):
    pattern = result_root / "eval_result" / task / POLICY_NAME / task_config / ckpt_label / "*" / "success_episode*.mp4"
    files = sorted(glob.glob(str(pattern)), key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0] if files else None


def terminate_job(job, timeout_sec=60):
    try:
        os.killpg(job.proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if job.proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(job.proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def build_command(config, task, args, log_dir, quant_root):
    checkpoint_path = config["checkpoint_path"]
    wan_path = config["wan_path"]
    vlm_path = config["vlm_path"]
    task_config = args.task_config or str(config.get("task_config", "demo_randomized_motus"))
    seed = str(args.seed if args.seed is not None else config.get("seed", 42))
    test_num = str(args.test_num if args.test_num is not None else config.get("test_num", 30))
    result_root = args.result_root or config.get("result_root") or "/mnt/hdd/wangchaoyang/results/Motus"

    return [
        sys.executable,
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        "--task_name",
        task,
        "--task_config",
        task_config,
        "--ckpt_setting",
        checkpoint_path,
        "--seed",
        seed,
        "--test_num",
        test_num,
        "--policy_name",
        POLICY_NAME,
        "--log_dir",
        str(log_dir),
        "--result_root",
        str(Path(result_root) / "eval_result"),
        "--wan_path",
        wan_path,
        "--vlm_path",
        vlm_path,
        "--success_video_only",
        "true",
        "--quantization_dir",
        str(quant_root),
    ]


def start_job(gpu, task, config, args, log_dir, quant_root):
    log_path = log_dir / f"{task}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_start_offset = log_path.stat().st_size if log_path.exists() else 0
    status_path = quant_root / task / "status.json"
    stale_metrics_path = quant_root / task / "metrics.json"
    if stale_metrics_path.exists():
        try:
            stale_metrics_path.unlink()
        except Exception:
            pass
    write_json(status_path, {
        "task_name": task,
        "status": "running",
        "gpu": gpu,
        "started_at": now_string(),
        "log_path": str(log_path),
        "log_start_offset": log_start_offset,
    })

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["PYTHONPATH"] = f"{config['robotwin_root']}:{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "8"
    env["HF_HOME"] = "/mnt/hdd/hf_a100/wangchaoyang"
    env["HF_HUB_CACHE"] = "/mnt/hdd/hf_a100/wangchaoyang/hub"
    env["MOTUS_QUANTIZATION_DIR"] = str(quant_root)
    env["MOTUS_QUANTIZATION_TASK"] = task

    command = build_command(config, task, args, log_dir, quant_root)
    log_handle = open(log_path, "ab", buffering=0)
    header = f"\n===== {now_string()} | task={task} | gpu={gpu} =====\nCommand: {' '.join(command)}\n\n"
    log_handle.write(header.encode("utf-8"))

    proc = subprocess.Popen(
        command,
        cwd=config["robotwin_root"],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.time()
    return Job(task, gpu, proc, log_path, log_handle, started, started, started, status_path, log_start_offset)


def finalize_job(job, status, exit_code, reason=None):
    try:
        job.log_handle.close()
    except Exception:
        pass
    payload = {
        "task_name": job.task,
        "status": status,
        "gpu": job.gpu,
        "exit_code": exit_code,
        "reason": reason,
        "started_at_epoch": job.start_time,
        "finished_at": now_string(),
        "elapsed_seconds": round(time.time() - job.start_time, 3),
        "log_path": str(job.log_path),
    }
    write_json(job.status_path, payload)


def summarize_results(tasks, config, args, log_dir, quant_root):
    result_root = Path(args.result_root or config.get("result_root") or "/mnt/hdd/wangchaoyang/results/Motus")
    task_config = args.task_config or str(config.get("task_config", "demo_randomized_motus"))
    ckpt_label = Path(str(config["checkpoint_path"])).name

    rows = []
    scored = []
    total_success = 0
    total_episodes = 0
    for task in tasks:
        status_path = quant_root / task / "status.json"
        status = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {}
        task_log_path = Path(status.get("log_path") or log_dir / f"{task}.log")
        if status_allows_success_score(status):
            result_data = parse_success_from_result(result_root, task, task_config, ckpt_label)
            log_data = parse_success_from_log(task_log_path)
            success_data = result_data or log_data
        else:
            log_data = None
            success_data = None
        success_rate = success_data.get("success_rate_percent") if success_data else None
        if success_rate is not None:
            scored.append(success_rate)
        if log_data and "successes" in log_data and "episodes" in log_data:
            total_success += int(log_data["successes"])
            total_episodes += int(log_data["episodes"])
        rows.append({
            "task": task,
            "status": status.get("status", "missing"),
            "exit_code": status.get("exit_code"),
            "success_rate_percent": success_rate,
            "success_video": find_success_video(result_root, task, task_config, ckpt_label),
            "log_path": str(task_log_path),
        })

    avg_success = sum(scored) / len(scored) if scored else 0.0
    overall_episode_success = (total_success / total_episodes * 100.0) if total_episodes else None

    result_txt = result_root / "result.txt"
    result_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("Motus RoboTwin2.0 Evaluation Summary\n")
        f.write("====================================\n")
        f.write(f"Generated at: {now_string()}\n")
        f.write(f"Result root: {result_root}\n")
        f.write(f"Task config: {task_config}\n")
        f.write(f"Checkpoint: {config['checkpoint_path']}\n")
        f.write(f"Tasks: {len(tasks)}\n")
        f.write(f"Scored tasks: {len(scored)}\n")
        f.write(f"Average task success rate: {avg_success:.2f}%\n")
        if overall_episode_success is not None:
            f.write(f"Overall episode success rate: {overall_episode_success:.2f}% ({total_success}/{total_episodes})\n")
        f.write("\nTask Results:\n")
        for row in rows:
            rate = "NA" if row["success_rate_percent"] is None else f"{row['success_rate_percent']:.2f}%"
            f.write(f"{row['task']}\t{row['status']}\t{rate}\tvideo={row['success_video'] or 'NA'}\tlog={row['log_path']}\n")

    summarize_quantization(tasks, quant_root)


def summarize_quantization(tasks, quant_root):
    task_payloads = {}
    aggregate = {
        "task_count": len(tasks),
        "tasks_with_metrics": 0,
        "inference_call_count": 0,
        "cuda_latency_ms_total": 0.0,
        "wall_time_ms_total": 0.0,
        "action_count_total": 0.0,
        "flops_totals": {
            "video_generation_dit": 0,
            "intention_understanding_expert": 0,
            "action_expert": 0,
            "total": 0,
        },
    }
    for task in tasks:
        metrics_path = quant_root / task / "metrics.json"
        status_path = quant_root / task / "status.json"
        status = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {}
        if not metrics_path.exists():
            task_payloads[task] = {"status": status, "metrics_path": str(metrics_path), "metrics": None}
            continue
        if not status_allows_metric_aggregation(status):
            task_payloads[task] = {
                "status": status,
                "metrics_path": str(metrics_path),
                "metrics": None,
                "ignored": "metrics from an OOM attempt are excluded until the task completes successfully",
            }
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            task_payloads[task] = {"status": status, "metrics_path": str(metrics_path), "error": str(exc)}
            continue
        calls = int(metrics.get("inference_call_count") or 0)
        aggregate["tasks_with_metrics"] += 1
        aggregate["inference_call_count"] += calls
        metric_stats = metrics.get("metrics", {})
        aggregate["cuda_latency_ms_total"] += float(metric_stats.get("cuda_latency_ms", {}).get("total") or 0.0)
        aggregate["wall_time_ms_total"] += float(metric_stats.get("wall_time_ms", {}).get("total") or 0.0)
        aggregate["action_count_total"] += float(metric_stats.get("actions_per_policy_call", {}).get("total") or 0.0)
        for key, value in metrics.get("flops", {}).get("totals", {}).items():
            if key in aggregate["flops_totals"]:
                aggregate["flops_totals"][key] += int(value or 0)
        task_payloads[task] = {
            "status": status,
            "metrics_path": str(metrics_path),
            "inference_call_count": calls,
            "cuda_latency_ms": metric_stats.get("cuda_latency_ms"),
            "action_throughput_hz_effective": metric_stats.get("action_throughput_hz_effective"),
            "flops": metrics.get("flops"),
        }

    calls = max(int(aggregate["inference_call_count"]), 1)
    wall_seconds = max(aggregate["wall_time_ms_total"] / 1000.0, 1e-12)
    cuda_seconds = max(aggregate["cuda_latency_ms_total"] / 1000.0, 1e-12)
    aggregate["cuda_latency_ms_mean_per_policy_call"] = aggregate["cuda_latency_ms_total"] / calls
    aggregate["wall_time_ms_mean_per_policy_call"] = aggregate["wall_time_ms_total"] / calls
    aggregate["action_throughput_hz_effective"] = aggregate["action_count_total"] / wall_seconds
    aggregate["action_throughput_hz_cuda_only"] = aggregate["action_count_total"] / cuda_seconds
    aggregate["flops_mean_per_policy_call"] = {key: value / calls for key, value in aggregate["flops_totals"].items()}

    write_json(quant_root / "summary.json", {
        "generated_at": now_string(),
        "aggregate": aggregate,
        "tasks": task_payloads,
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="policy/Motus/paths_config.yml")
    parser.add_argument("--tasks-file", default="tasks_all.txt")
    parser.add_argument("--gpus", default="1,2,3,6")
    parser.add_argument("--min-free-mem-mib", type=int, default=43000)
    parser.add_argument("--stall-timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-num", type=int, default=None)
    return parser.parse_args()


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
    effective_task_config = args.task_config or str(config.get('task_config', 'demo_randomized_motus'))
    log_dir = result_root / f"logs_quantized_{effective_task_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_root.mkdir(parents=True, exist_ok=True)
    quant_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_state_path = result_root / "run_state.json"
    pending = deque(tasks)
    running = {}

    write_json(run_state_path, {
        "status": "running",
        "started_at": now_string(),
        "tasks_total": len(tasks),
        "tasks_pending": len(pending),
        "gpus": gpu_ids,
        "min_free_mem_mib": args.min_free_mem_mib,
        "stall_timeout_sec": args.stall_timeout_sec,
        "log_dir": str(log_dir),
        "quantization_dir": str(quant_root),
    })

    try:
        while pending or running:
            for gpu, job in list(running.items()):
                ret = job.proc.poll()
                if ret is not None:
                    status = "completed" if ret == 0 else "failed"
                    finalize_job(job, status, ret)
                    running.pop(gpu, None)
                    continue

                mtime = job.log_path.stat().st_mtime if job.log_path.exists() else job.start_time
                if mtime > job.last_log_mtime:
                    job.last_log_mtime = mtime
                    job.last_progress_time = time.time()
                elif time.time() - job.last_progress_time > args.stall_timeout_sec:
                    terminate_job(job)
                    finalize_job(job, "skipped_timeout", job.proc.poll(), f"no log progress for {args.stall_timeout_sec} seconds")
                    running.pop(gpu, None)

            for gpu in gpu_ids:
                if not pending or gpu in running:
                    continue
                free_mem = query_free_mem_mib(gpu)
                if free_mem < args.min_free_mem_mib:
                    continue
                task = pending.popleft()
                running[gpu] = start_job(gpu, task, config, args, log_dir, quant_root)

            write_json(run_state_path, {
                "status": "running",
                "updated_at": now_string(),
                "tasks_total": len(tasks),
                "tasks_pending": len(pending),
                "tasks_running": {gpu: job.task for gpu, job in running.items()},
                "tasks_finished": len(tasks) - len(pending) - len(running),
                "log_dir": str(log_dir),
                "quantization_dir": str(quant_root),
            })

            if pending or running:
                time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        for job in list(running.values()):
            terminate_job(job)
            finalize_job(job, "interrupted", job.proc.poll(), "runner interrupted")
        raise
    finally:
        summarize_results(tasks, config, args, log_dir, quant_root)

    write_json(run_state_path, {
        "status": "completed",
        "finished_at": now_string(),
        "tasks_total": len(tasks),
        "log_dir": str(log_dir),
        "quantization_dir": str(quant_root),
        "result_txt": str(result_root / "result.txt"),
        "quantization_summary": str(quant_root / "summary.json"),
    })


if __name__ == "__main__":
    main()
