"""Full classic-LIBERO Stage-III GRPO pass.

The online RL-train manifest has one real simulator anchor for each of the
40 classic LIBERO tasks and initial-state indices 10..29: 800 groups total.
Each group produces four trajectories. Updates are streamed in batches of 32
and all Student inference servers and this orchestrator run under tmux.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import ExitStack
import gc
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .train import (
    BASELINE, LIBERO_HOME, MODEL_PY, ROOT, SIMPY, STAGE3, STUDENT,
    atomic_json, collect_group, load_or_init_actor, make_batch, rpc,
    runtime_env, save_actor, train_iteration, verify_baseline,
)

FULL_MANIFEST = STAGE3 / "manifest/full_800.json"
DEFAULT_INITIAL_ACTOR = STAGE3 / "runs/grpo_20261013_002717/checkpoints/actor_iter_0002.pt"
DEFAULT_GPUS = [0, 2, 2, 5, 5, 7, 7]
DEFAULT_PORTS = list(range(29800, 29807))
TRAJECTORIES_PER_GROUP = 4
TOTAL_GROUPS = 800
GROUPS_PER_UPDATE = 32


def ensure_manifest() -> list[dict]:
    data = json.loads(FULL_MANIFEST.read_text())
    tasks = data.get("tasks")
    if data.get("group_count") != TOTAL_GROUPS or len(tasks or []) != TOTAL_GROUPS:
        raise RuntimeError(f"expected classic LIBERO full manifest with {TOTAL_GROUPS} rows")
    keys = {(str(t["suite"]), int(t["task_id"]), int(t["initial_state_idx"])) for t in tasks}
    if len(keys) != TOTAL_GROUPS:
        raise RuntimeError("classic full manifest contains duplicate anchors")
    if any(int(t["initial_state_idx"]) not in range(10, 30) for t in tasks):
        raise RuntimeError("full training manifest leaked outside RL-train state indices 10..29")
    return tasks


def start_server(run_dir: Path, actor_path: Path, gpu: int, port: int, index: int):
    ready = run_dir / "servers" / f"server_{index}_ready.json"
    log_path = run_dir / "servers" / f"server_{index}.log"
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.unlink(missing_ok=True)
    env = runtime_env(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [str(MODEL_PY), "-m", "stage_III.libero.server", "--device", "cuda:0",
           "--port", str(port), "--ready", str(ready), "--checkpoint", str(STUDENT),
           "--actor-checkpoint", str(actor_path)]
    stream = log_path.open("a", buffering=1)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stream.close()
            raise RuntimeError(f"classic Stage III server GPU{gpu} exited: {proc.returncode}; see {log_path}")
        try:
            health = rpc(url, "/health", timeout=10)
            if health.get("checkpoint") is None:
                rpc(url, "/load", {"checkpoint": str(STUDENT), "actor_checkpoint": str(actor_path)}, timeout=1800)
            info = rpc(url, "/health", timeout=10)
            if info.get("checkpoint") == str(STUDENT):
                print(f"[Stage III][全量服务] worker{index} 已就绪：GPU{gpu}，端口{port}，Teacher 未加载。", flush=True)
                return proc, url, stream
        except Exception:
            time.sleep(2)
    proc.terminate(); stream.close()
    raise TimeoutError(f"classic Stage III server GPU{gpu} startup timed out")


def stop_server(item) -> None:
    if not item:
        return
    proc, _, stream = item
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except Exception: pass
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(str(x) for x in DEFAULT_GPUS))
    parser.add_argument("--ports", default=",".join(str(x) for x in DEFAULT_PORTS))
    parser.add_argument("--groups-per-update", type=int, default=GROUPS_PER_UPDATE)
    parser.add_argument("--run", required=True)
    parser.add_argument("--initial-actor", default=str(DEFAULT_INITIAL_ACTOR))
    parser.add_argument("--tmux-session", default="libero_stage3_full")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("gpus must contain at least one nonnegative GPU index")
    if len(ports) != len(gpus):
        raise ValueError("gpus and ports must have equal length")
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("each worker must have a distinct valid TCP port")
    if args.groups_per_update != GROUPS_PER_UPDATE:
        raise ValueError("full classic LIBERO run uses 32 groups per update")
    tasks = ensure_manifest()
    run_dir = Path(args.run).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("tasks", "trajectories", "checkpoints", "servers"):
        (run_dir / sub).mkdir()
    total_updates = (TOTAL_GROUPS + args.groups_per_update - 1) // args.groups_per_update
    config = {
        "algorithm": "GRPO-style clipped GateActor",
        "benchmark": "classic_libero",
        "rounds": 1,
        "manifest": str(FULL_MANIFEST),
        "manifest_rows": TOTAL_GROUPS,
        "selected_groups": TOTAL_GROUPS,
        "trajectories_per_group": TRAJECTORIES_PER_GROUP,
        "planned_online_trajectories": TOTAL_GROUPS * TRAJECTORIES_PER_GROUP,
        "groups_per_update": args.groups_per_update,
        "planned_updates": total_updates,
        "gpus": gpus,
        "gpu_worker_counts": {str(g): gpus.count(g) for g in sorted(set(gpus))},
        "ports": ports,
        "base_student": str(STUDENT),
        "initial_actor": str(Path(args.initial_actor).resolve()),
        "efficiency_weight": 0.1,
        "clip_eps": 0.2,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "tmux_session": args.tmux_session,
        "preservation": "Stage I/II checkpoint and evaluation files are read-only",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (run_dir / "manifest.json").write_text(json.dumps({"source": str(FULL_MANIFEST), "tasks": tasks}, indent=2, ensure_ascii=False))
    initial = Path(args.initial_actor).resolve()
    if not initial.exists():
        raise FileNotFoundError(initial)
    actor, optimizer = load_or_init_actor(initial, config)
    current_actor = run_dir / "checkpoints/actor_current.pt"
    save_actor(current_actor, actor, optimizer, 0, config, {"kind": "classic_full800_initial_actor"})
    status = {"stage": "III", "phase": "initializing", "run_dir": str(run_dir),
              "config": config, "groups_completed": 0, "trajectories_completed": 0,
              "batch": 0, "total_batches": total_updates, "worker_status": {}}
    atomic_json(run_dir / "status.json", status)
    print(f"[Stage III][全量初始化] classic LIBERO：800 groups，3200 trajectories，25 updates。", flush=True)
    print(f"[Stage III][全量初始化] GPU worker layout={gpus}；tmux={args.tmux_session}。", flush=True)
    verify_baseline("classic 全量训练启动前")
    servers = []
    try:
        status["phase"] = "starting_servers"; atomic_json(run_dir / "status.json", status)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(start_server, run_dir, current_actor, gpu, port, i)
                       for i, (gpu, port) in enumerate(zip(gpus, ports))]
            servers = [f.result() for f in futures]
        status["phase"] = "servers_ready"
        status["worker_status"] = {str(i): {"gpu": g, "port": ports[i], "state": "ready"}
                                    for i, g in enumerate(gpus)}
        atomic_json(run_dir / "status.json", status)
        print(f"[Stage III][全量服务] {len(servers)} 个 classic Student 服务全部就绪。", flush=True)
        summary_path = run_dir / "group_summary.jsonl"
        metrics_path = run_dir / "metrics.jsonl"
        for batch_start in range(0, TOTAL_GROUPS, args.groups_per_update):
            batch_ids = list(range(batch_start, min(batch_start + args.groups_per_update, TOTAL_GROUPS)))
            update_id = batch_start // args.groups_per_update + 1
            status.update({"phase": "collecting", "batch": update_id, "batch_start": batch_start,
                           "batch_end": batch_ids[-1], "groups_completed": batch_start,
                           "trajectories_completed": batch_start * TRAJECTORIES_PER_GROUP})
            status["worker_status"] = {str(i): {"gpu": gpus[i], "port": ports[i], "state": "waiting"}
                                        for i in range(len(servers))}
            atomic_json(run_dir / "status.json", status)
            print(f"[Stage III][全量采样] batch {update_id}/{total_updates}：group {batch_ids[0]}-{batch_ids[-1]}，"
                  f"{len(batch_ids)} groups / {len(batch_ids)*TRAJECTORIES_PER_GROUP} trajectories。", flush=True)
            task_paths = {}
            for group_id in batch_ids:
                path = run_dir / "tasks" / f"task_{group_id:04d}.json"
                path.write_text(json.dumps(tasks[group_id], ensure_ascii=False))
                task_paths[group_id] = path
            groups = [None] * len(batch_ids)
            def one_group(pos_group):
                pos, group_id = pos_group
                worker_idx = pos % len(servers)
                server = servers[worker_idx]
                row = tasks[group_id]
                rows = collect_group(run_dir, task_paths[group_id], group_id,
                                     TRAJECTORIES_PER_GROUP, current_actor, server[1],
                                     gpus[worker_idx], initial_state_idx=int(row["initial_state_idx"]))
                return pos, group_id, worker_idx, rows
            # Keep one serial collection queue per model worker. A fast group
            # must not start a second collector on another worker that is busy.
            with ExitStack() as stack:
                pools = [stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=1))
                         for _ in servers]
                jobs = [pools[pos % len(servers)].submit(one_group, (pos, group_id))
                        for pos, group_id in enumerate(batch_ids)]
                for job in concurrent.futures.as_completed(jobs):
                    pos, group_id, worker_idx, rows = job.result()
                    groups[pos] = rows
                    status["worker_status"][str(worker_idx)] = {
                        "gpu": gpus[worker_idx], "port": ports[worker_idx],
                        "state": "group_complete", "last_group": group_id,
                        "successes": sum(int(x["success"]) for x in rows),
                        "trajectories": len(rows),
                    }
                    atomic_json(run_dir / "status.json", status)
            if any(group is None for group in groups):
                raise RuntimeError("missing classic LIBERO group result")
            status["phase"] = "updating"; atomic_json(run_dir / "status.json", status)
            metrics = train_iteration(actor, optimizer, groups, config)
            save_actor(current_actor, actor, optimizer, update_id, config, metrics)
            if update_id == 1 or update_id % 5 == 0 or batch_ids[-1] == TOTAL_GROUPS - 1:
                shutil.copy2(current_actor, run_dir / "checkpoints" / f"actor_update_{update_id:04d}.pt")
            hot_reload(servers, current_actor)
            with summary_path.open("a") as sf:
                for group_id, group in zip(batch_ids, groups):
                    row = group[0]
                    sf.write(json.dumps({"group_id": group_id, "suite": row["suite"],
                        "task_id": row["task_id"], "initial_state_idx": row["initial_state_idx"],
                        "successes": sum(int(x["success"]) for x in group),
                        "rewards": [float(x["reward"]) for x in group],
                        "decisions": sum(len(x["decisions"]) for x in group)}, ensure_ascii=False) + "\n")
            with metrics_path.open("a") as mf:
                mf.write(json.dumps({"batch": update_id, **metrics}, ensure_ascii=False) + "\n")
            remove_batch_files(run_dir, batch_ids)
            status.update({"phase": "batch_complete", "batch": update_id,
                           "groups_completed": batch_ids[-1] + 1,
                           "trajectories_completed": (batch_ids[-1] + 1) * TRAJECTORIES_PER_GROUP,
                           "last_metrics": metrics, "actor": str(current_actor)})
            atomic_json(run_dir / "status.json", status)
            print(f"[Stage III][全量更新] batch={update_id}/{total_updates} 完成，"
                  f"累计 groups={batch_ids[-1]+1}/{TOTAL_GROUPS}，"
                  f"trajectories={(batch_ids[-1]+1)*TRAJECTORIES_PER_GROUP}/{TOTAL_GROUPS*TRAJECTORIES_PER_GROUP}，"
                  f"successes={metrics['successes']}，KL={metrics['kl_old_new']:.6g}，"
                  f"eta_mean={metrics['actor_eta_mean']:.6f}。", flush=True)
            groups.clear(); gc.collect()
        status.update({"phase": "completed", "groups_completed": TOTAL_GROUPS,
                       "trajectories_completed": TOTAL_GROUPS * TRAJECTORIES_PER_GROUP,
                       "actor": str(current_actor), "completed_at": time.time()})
        atomic_json(run_dir / "status.json", status)
        verify_baseline("classic 全量训练结束")
        print("[Stage III][全量完成] classic LIBERO 800/800 groups，3200/3200 trajectories。", flush=True)
    except Exception as exc:
        status.update({"phase": "failed", "error": repr(exc), "failed_at": time.time()})
        atomic_json(run_dir / "status.json", status)
        print(f"[Stage III][全量失败] {exc!r}", flush=True)
        raise
    finally:
        for item in servers:
            stop_server(item)


if __name__ == "__main__":
    main()
