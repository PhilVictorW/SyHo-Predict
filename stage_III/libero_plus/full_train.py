"""Full LIBERO-plus Stage-III GRPO pass over all 8380 anchors.

One manifest row is one GRPO anchor.  Each anchor is rolled out four times
from the same initial state.  Collection is streamed in batches so the full
33,520 trajectories never need to stay in RAM or on disk at once.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
from pathlib import Path

import torch

from .gate_actor import GateActor
from .checkpoint_cache import prepare_student_cache
from .group_policy import is_skipped, partition_groups, summary_counts
from .train import (
    BASELINE,
    LIBERO_HOME,
    MODEL_PY,
    ROOT,
    SIMPY,
    STAGE3,
    STUDENT,
    atomic_json,
    collect_group,
    load_or_init_actor,
    make_batch,
    rpc,
    runtime_env,
    save_actor,
    sha256,
    train_iteration,
    verify_baseline,
)

FULL_MANIFEST_SOURCE = ROOT / "libero-plus/evaluation/stage_II_steps2500_4000_parallel_20261008/manifest_same_as_steps1000_1500_2000.json"
FULL_MANIFEST = STAGE3 / "manifest/full_8380.json"
DEFAULT_INITIAL_ACTOR = STAGE3 / "runs/grpo_20261012_232244/checkpoints/actor_iter_0002.pt"
# 两个独立的 Student/GateActor 服务映射到每张允许使用的 GPU。
# 重复 GPU 是有意的：每个服务独立占用一个进程，采样任务可并行推进。
DEFAULT_GPUS = [1, 1, 3, 3, 6, 6, 7, 7]
DEFAULT_PORTS = list(range(29700, 29708))
TRAJECTORIES_PER_GROUP = 4


def ensure_manifest() -> list[dict]:
    if not FULL_MANIFEST.exists():
        data = json.loads(FULL_MANIFEST_SOURCE.read_text())
        if len(data.get("tasks", [])) != 8380:
            raise RuntimeError("the selected source manifest is not the 8380-row subset")
        FULL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        FULL_MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    data = json.loads(FULL_MANIFEST.read_text())
    tasks = data.get("tasks")
    if len(tasks) != 8380:
        raise RuntimeError(f"expected 8380 manifest rows, got {len(tasks)}")
    if len({(int(t["task_id"]), str(t["suite"])) for t in tasks}) != 8380:
        raise RuntimeError("manifest task ids are not unique by suite")
    return tasks


def start_server(run_dir: Path, actor_path: Path, gpu: int, port: int, index: int):
    ready = run_dir / "servers" / f"server_{index}_ready.json"
    log_path = run_dir / "servers" / f"server_{index}.log"
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.unlink(missing_ok=True)
    env = runtime_env(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [str(MODEL_PY), "-m", "stage_III.libero_plus.server",
           "--device", "cuda:0", "--port", str(port), "--ready", str(ready),
           "--checkpoint", str(STUDENT), "--actor-checkpoint", str(actor_path)]
    stream = log_path.open("a", buffering=1)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    begin = time.monotonic()
    last_report = float("-inf")
    while True:
        if proc.poll() is not None:
            stream.close()
            raise RuntimeError(f"Stage III server GPU{gpu} exited: {proc.returncode}; see {log_path}")
        phase = "等待服务监听"
        try:
            info = rpc(url, "/health", timeout=10)
            if info.get("load_state") == "failed":
                raise RuntimeError(f"server{index} model load failed: {info.get('load_error')}; see {log_path}")
            if info.get("checkpoint") == str(STUDENT) and info.get("actor_checkpoint") == str(actor_path.resolve()):
                print(f"[Stage III][全量服务] server{index} 已就绪：GPU{gpu}，端口{port}，Teacher 未加载。", flush=True)
                return proc, url, stream
            if info.get("load_state", "idle") == "idle":
                info = rpc(url, "/start_load", {"checkpoint": str(STUDENT), "actor_checkpoint": str(actor_path)}, timeout=15)
            phase = info.get("load_phase", "模型加载中")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # A slow health/load request does not cancel the loading thread or
            # shut down other servers. The next health query observes its state.
            phase = f"连接等待：{type(exc).__name__}"
        now = time.monotonic()
        if now - last_report >= 30:
            print(f"[Stage III][加载等待] server{index} GPU{gpu}：{phase}，"
                  f"已等待 {now - begin:.0f} 秒；超时继续等待，不结束训练。", flush=True)
            last_report = now
        time.sleep(2)


def attach_server(run_dir: Path, actor_path: Path, gpu: int, port: int, index: int):
    url = f"http://127.0.0.1:{port}"
    info = rpc(url, "/health", timeout=15)
    expected = json.loads((run_dir / "servers" / f"server_{index}_ready.json").read_text())
    if (info.get("pid") != expected.get("pid")
            or info.get("checkpoint") != str(STUDENT)
            or info.get("actor_checkpoint") != str(actor_path.resolve())
            or str(info.get("physical_gpu")) != str(gpu)):
        raise RuntimeError(f"existing server identity mismatch: {info}")
    print(f"[Stage III][恢复服务] server{index} GPU{gpu} 端口{port} 已复用，PID={info['pid']}", flush=True)
    return int(info["pid"]), url, None


def stop_server(item):
    if not item:
        return
    proc, _, stream = item
    if isinstance(proc, int):
        cmdline = Path(f"/proc/{proc}/cmdline")
        if cmdline.exists() and b"stage_III.libero_plus.server" in cmdline.read_bytes():
            os.killpg(proc, signal.SIGTERM)
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    stream.close()


def hot_reload(servers, actor_path: Path) -> None:
    for _, url, _ in servers:
        info = rpc(url, "/reload_actor", {"actor_checkpoint": str(actor_path)}, timeout=180)
        if info.get("actor_checkpoint") != str(actor_path.resolve()):
            raise RuntimeError(f"actor reload mismatch: {info}")


def remove_batch_files(run_dir: Path, group_ids: list[int]) -> None:
    traj_dir = run_dir / "trajectories"
    for group_id in group_ids:
        for idx in range(TRAJECTORIES_PER_GROUP):
            (traj_dir / f"group_{group_id:04d}_traj_{idx:02d}.json").unlink(missing_ok=True)


def update_collected_groups(actor, optimizer, groups, config):
    valid, skipped = partition_groups(groups, TRAJECTORIES_PER_GROUP)
    if valid:
        metrics = train_iteration(actor, optimizer, valid, config)
    else:
        metrics = dict(groups=0, trajectories=0, decisions=0, successes=0,
                       active_groups=0, updated=False, reason="all_groups_skipped_timeout")
        print("[Stage III][跳过更新] 本批全部超时；参数和 Adam 状态保持不变，继续下一批。", flush=True)
    metrics.update(groups_processed=len(groups), groups_skipped=len(skipped),
                   optimizer_step=bool(valid))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(str(x) for x in DEFAULT_GPUS))
    parser.add_argument("--ports", default=",".join(str(x) for x in DEFAULT_PORTS))
    parser.add_argument("--groups-per-update", type=int, default=32)
    parser.add_argument("--limit-groups", type=int, default=None,
                        help="只用于正式启动前的隔离 smoke；完整运行省略此参数")
    parser.add_argument("--run", default=None)
    parser.add_argument("--resume", action="store_true", help="从已提交的更新和当前批次缓存恢复")
    parser.add_argument("--reuse-servers", action="store_true", help="校验后接管本 run 已运行的模型服务")
    parser.add_argument("--initial-actor", default=str(DEFAULT_INITIAL_ACTOR))
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    if not gpus or len(gpus) != len(ports) or 4 in gpus:
        raise ValueError("gpus and ports must have equal nonzero length; GPU4 is forbidden")
    if args.groups_per_update <= 0:
        raise ValueError("groups-per-update must be positive")
    tasks = ensure_manifest()
    total_groups = len(tasks) if args.limit_groups is None else min(int(args.limit_groups), len(tasks))
    if total_groups <= 0:
        raise ValueError("no groups selected")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run) if args.run else STAGE3 / "runs" / f"grpo_full8380_{stamp}"
    if (args.resume or args.reuse_servers) and not args.run:
        raise ValueError("--resume/--reuse-servers requires --run")
    if args.reuse_servers and not args.resume:
        raise ValueError("--reuse-servers requires --resume")
    if not run_dir.resolve().is_relative_to(STAGE3.resolve()):
        raise ValueError("run directory must remain within isolated Stage III")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    lock = (run_dir / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("tasks", "trajectories", "checkpoints", "servers"):
        (run_dir / name).mkdir(exist_ok=args.resume)
    config = {
        "algorithm": "GRPO-style clipped GateActor",
        "manifest": str(FULL_MANIFEST),
        "manifest_rows": 8380,
        "selected_groups": total_groups,
        "trajectories_per_group": TRAJECTORIES_PER_GROUP,
        "planned_online_trajectories": total_groups * TRAJECTORIES_PER_GROUP,
        "groups_per_update": args.groups_per_update,
        "gpus": gpus,
        "ports": ports,
        "base_student": str(STUDENT),
        "initial_actor": str(Path(args.initial_actor).resolve()),
        "efficiency_weight": 0.1,
        "clip_eps": 0.2,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "preservation": "Stage I/II checkpoint and evaluation files are read-only",
    }
    current_actor = run_dir / "checkpoints/actor_current.pt"
    if args.resume:
        saved_config = json.loads((run_dir / "config.json").read_text())
        for key in ("selected_groups", "groups_per_update", "trajectories_per_group", "base_student"):
            if saved_config[key] != config[key]:
                raise ValueError(f"resume config mismatch: {key}")
        if json.loads((run_dir / "manifest.json").read_text())["tasks"] != tasks:
            raise ValueError("resume manifest changed")
        resource_change = saved_config["gpus"] != gpus or saved_config["ports"] != ports
        if resource_change and args.reuse_servers:
            raise ValueError("GPU/port migration requires starting fresh servers")
        config = {**saved_config, "gpus": gpus, "ports": ports}
        status = json.loads((run_dir / "status.json").read_text())
        payload = torch.load(current_actor, map_location="cpu", weights_only=False)
        summaries = [json.loads(x) for x in (run_dir / "group_summary.jsonl").read_text().splitlines()]
        start_group = int(status.get("groups_processed", status["groups_completed"]))
        saved_update = int(status.get("batch", 0))
        if (payload["iteration"] != saved_update
                or start_group != min(saved_update * args.groups_per_update, total_groups)
                or [x["group_id"] for x in summaries] != list(range(start_group))
                or status.get("phase") == "updating"):
            raise RuntimeError("checkpoint/status/summary disagree; refusing to repeat or skip an update")
        counts = summary_counts(summaries)
        if status["groups_completed"] != counts["groups_completed"]:
            raise RuntimeError("valid group count disagrees with summary")
        status.update(counts)
        status.setdefault("optimizer_updates", saved_update)
        actor, optimizer = load_or_init_actor(current_actor, config)
        if resource_change:
            migration = dict(time=time.time(), update=saved_update,
                             old_gpus=saved_config["gpus"], new_gpus=gpus,
                             old_ports=saved_config["ports"], new_ports=ports)
            with (run_dir / "resource_history.jsonl").open("a") as f:
                f.write(json.dumps(migration) + "\n")
            print(f"[Stage III][设备迁移] GPU {saved_config['gpus']} → {gpus}；"
                  "权重、Adam 状态、任务顺序和有效轨迹均保留。", flush=True)
        record = dict(time=time.time(), groups_processed=start_group,
                      groups_completed=status["groups_completed"], update=saved_update,
                      actor_sha256=sha256(current_actor))
        with (run_dir / "resume_history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        status.update(phase="resuming", resumed_at=record["time"],
                      resume_base_trajectories=status["groups_completed"] * TRAJECTORIES_PER_GROUP + sum(
                          1 for p in (run_dir / "trajectories").glob("group_*_traj_*.json")
                          if int(p.stem.split("_")[1]) >= start_group),
                      trainer_pid=os.getpid())
        status.pop("error", None)
        print(f"[Stage III][断点恢复] 已恢复第 {saved_update} 次更新及 Adam 状态；"
              f"从 group {start_group} 继续，当前批次有效轨迹直接复用。", flush=True)
    else:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
        (run_dir / "manifest.json").write_text(json.dumps({"source": str(FULL_MANIFEST_SOURCE), "tasks": tasks}, indent=2, ensure_ascii=False))
        initial = Path(args.initial_actor).resolve()
        if not initial.exists():
            raise FileNotFoundError(f"initial GateActor not found: {initial}")
        actor, optimizer = load_or_init_actor(initial, config)
        save_actor(current_actor, actor, optimizer, 0, config, {"kind": "full8380_initial_actor"})
        start_group = 0
        status = {"phase": "initializing", "run_dir": str(run_dir), "config": config,
                  "groups_completed": 0, "groups_processed": 0, "groups_skipped": 0,
                  "optimizer_updates": 0, "trajectories_completed": 0, "trainer_pid": os.getpid()}
    config.update(timeout_policy="skip_whole_group", stall_timeout_seconds=180,
                  import_timeout_seconds=600, group_hard_timeout_seconds=3600)
    os.environ["STAGE3_STALL_TIMEOUT"] = str(config["stall_timeout_seconds"])
    os.environ["STAGE3_IMPORT_TIMEOUT"] = str(config["import_timeout_seconds"])
    atomic_json(run_dir / "config.json", config)
    status["config"] = config
    atomic_json(run_dir / "status.json", status)
    print("[Stage III][超时策略] 初始化无进展 600 秒 / 采样无进展 180 秒 / 单组最多 3600 秒；"
          "达到任一上限即跳过整组，不赋失败奖励，不结束训练。", flush=True)
    print(f"[Stage III][全量初始化] 8380 个 group，每组 4 条轨迹，计划 {total_groups * 4} 条在线轨迹。", flush=True)
    print(f"[Stage III][全量初始化] 使用 GPU：{gpus}；GPU4 明确禁用。", flush=True)
    servers = []
    try:
        status["phase"] = "preparing_cache"; atomic_json(run_dir / "status.json", status)
        print("[Stage III][恢复准备] 正在准备或等待经完整 SHA256 验证的 SSD 权重副本。", flush=True)
        model_cache = prepare_student_cache(STUDENT, BASELINE["base_student_model_sha256"])
        os.environ["STAGE3_STUDENT_CACHE"] = str(model_cache)
        config["student_load_cache"] = str(model_cache)
        atomic_json(run_dir / "config.json", config)
        verify_baseline("全量训练启动前", model_cache=model_cache)
        status["phase"] = "starting_servers"; atomic_json(run_dir / "status.json", status)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, len(gpus))) as pool:
            launcher = attach_server if args.reuse_servers else start_server
            futures = [pool.submit(launcher, run_dir, current_actor, gpu, port, i)
                       for i, (gpu, port) in enumerate(zip(gpus, ports))]
            servers = [None] * len(futures)
            indices = {future: index for index, future in enumerate(futures)}
            failures = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    servers[indices[future]] = future.result()
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]
        hot_reload(servers, current_actor)
        print(f"[Stage III][全量服务] {len(servers)} 个服务全部就绪，GPU={gpus}。", flush=True)
        summary_path = run_dir / "group_summary.jsonl"
        metrics_path = run_dir / "metrics.jsonl"
        for batch_start in range(start_group, total_groups, args.groups_per_update):
            batch_ids = list(range(batch_start, min(batch_start + args.groups_per_update, total_groups)))
            status.update({"phase": "collecting", "batch_start": batch_start,
                           "batch_end": batch_ids[-1], "groups_processed": batch_start})
            atomic_json(run_dir / "status.json", status)
            print(f"[Stage III][全量采样] group {batch_ids[0]}-{batch_ids[-1]} / {total_groups - 1}，"
                  f"本批 {len(batch_ids)} 组、{len(batch_ids) * 4} 条轨迹。", flush=True)
            task_paths = {}
            for group_id in batch_ids:
                path = run_dir / "tasks" / f"task_{group_id:05d}.json"
                path.write_text(json.dumps(tasks[group_id], ensure_ascii=False))
                task_paths[group_id] = path
            groups = [None] * len(batch_ids)
            def collect_slot(slot):
                # One sequential queue per service; never send two rollouts to
                # the same service just because another thread finished sooner.
                for pos in range(slot, len(batch_ids), len(servers)):
                    group_id = batch_ids[pos]
                    groups[pos] = collect_group(run_dir, task_paths[group_id], group_id,
                                               TRAJECTORIES_PER_GROUP, current_actor,
                                               servers[slot][1], gpus[slot], timeout_policy="skip")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as pool:
                jobs = [pool.submit(collect_slot, slot) for slot in range(len(servers))]
                for job in concurrent.futures.as_completed(jobs):
                    job.result()
            if any(group is None for group in groups):
                raise RuntimeError("missing group result")
            status["phase"] = "updating"; atomic_json(run_dir / "status.json", status)
            metrics = update_collected_groups(actor, optimizer, groups, config)
            update_id = batch_start // args.groups_per_update + 1
            save_actor(current_actor, actor, optimizer, update_id, config, metrics)
            if update_id == 1 or update_id % 10 == 0 or batch_ids[-1] == total_groups - 1:
                archive = run_dir / "checkpoints" / f"actor_update_{update_id:04d}.pt"
                shutil.copy2(current_actor, archive)
            hot_reload(servers, current_actor)
            with summary_path.open("a") as sf:
                for group_id, group in zip(batch_ids, groups):
                    if is_skipped(group):
                        sf.write(json.dumps(group, ensure_ascii=False) + "\n")
                        continue
                    row = group[0]
                    sf.write(json.dumps({"group_id": group_id, "suite": row["suite"],
                        "task_id": row["task_id"], "successes": sum(int(x["success"]) for x in group),
                        "rewards": [float(x["reward"]) for x in group],
                        "decisions": sum(len(x["decisions"]) for x in group)}, ensure_ascii=False) + "\n")
            status.update({"phase": "batch_complete", "batch": update_id,
                           "groups_processed": batch_ids[-1] + 1,
                           "groups_completed": status["groups_completed"] + metrics["groups"],
                           "groups_skipped": status["groups_skipped"] + metrics["groups_skipped"],
                           "optimizer_updates": status["optimizer_updates"] + int(metrics["optimizer_step"]),
                           "trajectories_completed": status["trajectories_completed"] + metrics["trajectories"],
                           "last_metrics": metrics, "actor": str(current_actor)})
            atomic_json(run_dir / "status.json", status)
            with metrics_path.open("a") as mf:
                mf.write(json.dumps({"update": update_id, "time": time.time(), **metrics}) + "\n")
            remove_batch_files(run_dir, batch_ids)
            print(f"[Stage III][全量更新] batch={update_id} 完成，已处理组={batch_ids[-1] + 1}/{total_groups}，"
                  f"有效训练组={status['groups_completed']}，超时跳过组={status['groups_skipped']}，"
                  f"有效轨迹={status['trajectories_completed']}/{total_groups * 4}。", flush=True)
            groups.clear(); gc.collect()
        status.update({"phase": "completed_with_skips" if status["groups_skipped"] else "completed",
                       "groups_processed": total_groups,
                       "actor": str(current_actor), "completed_at": time.time()})
        atomic_json(run_dir / "status.json", status)
        verify_baseline("全量训练结束", model_cache=model_cache)
        print(f"[Stage III][全量完成] 已遍历 {total_groups}/{total_groups} 组；"
              f"有效训练 {status['groups_completed']} 组 / {status['trajectories_completed']} 条轨迹；"
              f"超时跳过 {status['groups_skipped']} 组（未计入训练）。", flush=True)
    except Exception as exc:
        status.update({"phase": "failed", "error": repr(exc), "failed_at": time.time()})
        atomic_json(run_dir / "status.json", status)
        print(f"[Stage III][全量失败] {exc!r}", flush=True)
        raise
    finally:
        for item in servers:
            stop_server(item)
        lock.close()


if __name__ == "__main__":
    main()
