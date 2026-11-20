"""Isolated Stage-III GRPO-style trainer for LIBERO-Plus.

The Stage-II Student is always loaded read-only by ``server.py``.  This
orchestrator stores only GateActor checkpoints and trajectory records below
``stage_III/libero_plus``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

from .gate_actor import FlatGateBatch, GateActor, clipped_gate_loss, group_advantages
from .runtime_safety import cached_trajectory, file_sha256, wait_with_progress
from .group_policy import load_skip, record_skip
from .checkpoint_cache import validate_cache

ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "stage_III/libero_plus"
BASELINE = json.loads((STAGE3 / "baseline_snapshot.json").read_text())
STUDENT = Path(BASELINE["base_student"])
MANIFEST = STAGE3 / "manifest/stage3_seed_tasks.json"
SERVER_PY = ROOT / "stage_III/libero_plus/server.py"
SIMPY = Path("/home/wangchaoyang/envs/libero-plus/bin/python")
MODEL_PY = Path("/home/wangchaoyang/envs/LaWAM/bin/python")
LIBERO_HOME = Path("/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero-plus/inference/LIBERO-plus")
LIBERO_SOURCE = ROOT.parent / "LaWAM_without_subgoal"
LIBERO_PY_SOURCE = LIBERO_SOURCE / "runtime/LIBERO"
LIBERO_RUNTIME = LIBERO_SOURCE / "libero-plus/runtime"
LIBERO_CONFIG = ROOT.parent / "LaWAM_aba2/libero-plus/config/libero"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_baseline(label: str, *, model_cache: Path | None = None) -> None:
    checks = {
        "base_student_model_sha256": STUDENT / "model.pt",
        "base_student_marker_sha256": STUDENT / "checkpoint_complete.json",
        "base_student_action_semantics_sha256": STUDENT / "action_semantics.json",
        "baseline_result_sha256": Path(BASELINE["baseline_result"]),
    }
    print(f"[Stage III][{label}] 正在核对 Stage I/II 基线哈希（只读）...", flush=True)
    for key, path in checks.items():
        expected = BASELINE[key]
        if not path.exists():
            raise FileNotFoundError(path)
        if key == "base_student_model_sha256" and model_cache is not None:
            validate_cache(model_cache, STUDENT, expected)
            actual = expected  # Full SHA256 was checked while copying; both fingerprints still match.
        else:
            actual = sha256(path)
        print(f"[Stage III][{label}] {path.name}: {actual[:16]}...", flush=True)
        if actual != expected:
            raise RuntimeError(f"基线哈希变化：{path} expected={expected} actual={actual}")
    marker = json.loads((STUDENT / "checkpoint_complete.json").read_text())
    if marker != BASELINE["checkpoint_marker"]:
        raise RuntimeError(f"Stage-II marker mismatch: {marker}")
    print(f"[Stage III][{label}] 基线哈希全部一致；不会写入 Stage I/II 目录。", flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def rpc(url: str, endpoint: str, payload: dict | None = None, timeout: float = 1800.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        value = json.load(response)
    if isinstance(value, dict) and "error" in value:
        raise RuntimeError(value["error"])
    return value


def runtime_env(gpu: int) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "LIBERO_HOME": str(LIBERO_HOME),
        "LIBERO_CONFIG_PATH": str(LIBERO_CONFIG),
        "MAGICK_HOME": str(LIBERO_RUNTIME / "sysroot/usr"),
        "MAGICK_CONFIGURE_PATH": str(LIBERO_RUNTIME / "sysroot/etc/ImageMagick-6"),
        "MAGICK_CODER_MODULE_PATH": str(LIBERO_RUNTIME / "sysroot/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.12/modules-Q16/coders"),
        "WAND_MAGICK_LIBRARY_SUFFIX": "-6.Q16",
    })
    ld = str(LIBERO_RUNTIME / "sysroot/usr/lib/x86_64-linux-gnu")
    env["LD_LIBRARY_PATH"] = ld + ":/usr/local/cuda12.8/lib64:" + env.get("LD_LIBRARY_PATH", "")
    paths = [str(ROOT), str(LIBERO_HOME), str(LIBERO_PY_SOURCE), str(LIBERO_RUNTIME)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def save_actor(path: Path, actor: GateActor, optimizer: torch.optim.Optimizer | None,
               iteration: int, config: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "stage3_gate_actor_v1",
        "iteration": iteration,
        "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }, tmp)
    tmp.replace(path)


def load_or_init_actor(path: Path, config: dict) -> tuple[GateActor, torch.optim.Optimizer]:
    actor = GateActor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["learning_rate"]))
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "stage3_gate_actor_v1":
            raise RuntimeError(f"unexpected actor checkpoint: {path}")
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        if payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        print(f"[Stage III][初始化] 恢复隔离 GateActor：{path}", flush=True)
    else:
        print("[Stage III][初始化] 创建 GateActor：6144 → 256(SiLU) → Beta(20,20)，初始 eta 均值为 1。", flush=True)
        save_actor(path, actor, optimizer, 0, config, {"kind": "initialization"})
    return actor, optimizer


def start_server(run_dir: Path, actor_path: Path, gpu: int, port: int) -> tuple[subprocess.Popen, str, Path]:
    ready = run_dir / "server_ready.json"
    ready.unlink(missing_ok=True)
    env = runtime_env(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [str(MODEL_PY), "-m", "stage_III.libero_plus.server",
           "--device", "cuda:0", "--port", str(port), "--ready", str(ready),
           "--checkpoint", str(STUDENT), "--actor-checkpoint", str(actor_path)]
    print(f"[Stage III][服务端] 启动只读 Student 服务：GPU{gpu}，端口 {port}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    url = f"http://127.0.0.1:{port}"
    begin = time.time()
    loaded = False
    while time.time() - begin < 1800:
        if proc.poll() is not None:
            raise RuntimeError(f"Stage III server exited with code {proc.returncode}")
        try:
            if ready.exists():
                health = rpc(url, "/health", timeout=10)
                if health.get("checkpoint") is None:
                    rpc(url, "/load", {"checkpoint": str(STUDENT), "actor_checkpoint": str(actor_path)}, timeout=1800)
                loaded = True
                break
        except Exception as exc:
            if int(time.time() - begin) % 20 == 0:
                print(f"[Stage III][服务端] 等待模型加载：{type(exc).__name__}", flush=True)
        time.sleep(2)
    if not loaded:
        proc.terminate()
        raise TimeoutError("Stage III server did not become ready")
    print("[Stage III][服务端] Student step4000 已加载；Teacher 未加载；GateActor 已加载。", flush=True)
    return proc, url, ready


def collect_group(run_dir: Path, task_path: Path, group_id: int, n: int,
                  actor_path: Path, url: str, gpu: int, *, timeout_policy: str = "raise") -> list[dict] | dict:
    if timeout_policy not in ("raise", "skip"):
        raise ValueError("invalid timeout policy")
    print(f"[Stage III][采样] 组 {group_id}：同一任务、同一初始状态，采样 {n} 条交互轨迹。", flush=True)
    env = runtime_env(gpu)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [str(SIMPY), str(ROOT / "stage_III/libero_plus/rollout.py"),
           "--task-json", str(task_path), "--group", str(group_id),
           "--trajectories", str(n), "--initial-state-idx", "0",
           "--checkpoint", str(STUDENT), "--actor-checkpoint", str(actor_path),
           "--url", url, "--output", str(run_dir)]
    task = json.loads(task_path.read_text())
    actor_hash = file_sha256(actor_path)
    if timeout_policy == "skip":
        previous = load_skip(run_dir, group_id, task, actor_hash, n)
        if previous is not None:
            print(f'[Stage III][恢复] 组 {group_id} 已记录为超时跳过，继续。', flush=True)
            return previous
    def read_cached():
        return [cached_trajectory(
            run_dir / 'trajectories' / f'group_{group_id:04d}_traj_{idx:02d}.json',
            group=group_id, trajectory=idx, task=task, student=STUDENT,
            actor=actor_path, actor_hash=actor_hash) for idx in range(n)]
    rows = read_cached()
    if all(row is not None for row in rows):
        print(f'[Stage III][恢复] 组 {group_id} 的 {n} 条轨迹均已完成，直接复用。', flush=True)
    else:
        if timeout_policy == "skip":
            # Honor already exhausted timeouts from this exact checkpoint when
            # recovering the older trainer, without throwing away complete groups.
            history = run_dir / "watchdog_events.jsonl"
            if history.exists():
                for line in history.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Another sampling thread may be appending a new event.
                    if (event.get('group') == group_id and event.get('attempt', 0) >= 3
                            and event.get('time', 0) >= actor_path.stat().st_mtime
                            and str(event.get('error', '')).startswith('TimeoutError(')):
                        return record_skip(run_dir, group_id, task, actor_hash, n,
                                           '恢复时确认：同一权重下已连续 3 次采样超时')
        timeout = float(env.get('STAGE3_STALL_TIMEOUT', '180'))
        import_timeout = float(env.get('STAGE3_IMPORT_TIMEOUT', '600'))
        for attempt in range(1, 4):
            progress = run_dir / 'progress' / f'group_{group_id:04d}.json'
            atomic_json(progress, dict(group=group_id, phase='启动采样子进程', attempt=attempt,
                                       updated_at=time.time(), timeout_seconds=import_timeout))
            try:
                proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
                code = wait_with_progress(proc, progress, stall_timeout=timeout, hard_timeout=3600,
                                          import_timeout=import_timeout)
                if code:
                    raise RuntimeError(f'rollout exited with code={code}')
                rows = read_cached()
                if any(row is None for row in rows):
                    raise RuntimeError('rollout returned without all trajectory files')
                break
            except (TimeoutError, RuntimeError) as exc:
                event = dict(group=group_id, attempt=attempt, error=repr(exc), time=time.time())
                with (run_dir / 'watchdog_events.jsonl').open('a') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
                if isinstance(exc, TimeoutError) and timeout_policy == "skip":
                    return record_skip(run_dir, group_id, task, actor_hash, n, str(exc))
                print(f'[Stage III][自动恢复] 组 {group_id} 第 {attempt}/3 次采样异常：{exc}；'
                      '保留已完成轨迹，同一初始状态重试。', flush=True)
                if attempt == 3:
                    raise RuntimeError(f'group {group_id} failed after 3 attempts; cached trajectories retained') from exc
    rewards = [float(row["reward"]) for row in rows]
    successes = sum(bool(row["success"]) for row in rows)
    print(f"[Stage III][采样] 组 {group_id} 完成：成功 {successes}/{n}，rewards={rewards}", flush=True)
    return rows


def make_batch(groups: list[list[dict]]) -> tuple[FlatGateBatch, dict]:
    reward_matrix = torch.tensor([[float(row["reward"]) for row in group] for group in groups], dtype=torch.float32)
    advantages, std, active = group_advantages(reward_matrix)
    features, z, old_logp, old_alpha, old_beta, adv, tids = [], [], [], [], [], [], []
    lengths = []
    for gi, group in enumerate(groups):
        g_adv = advantages[gi]
        for ti, row in enumerate(group):
            decisions = row.get("decisions", [])
            if not decisions:
                raise RuntimeError(f"trajectory has no GateActor decisions: group={gi}, trajectory={ti}")
            lengths.append(len(decisions))
            for decision in decisions:
                features.append(decision["features"])
                z.append(float(decision["z"]))
                old_logp.append(float(decision["old_logp"]))
                old_alpha.append(float(decision["old_alpha"]))
                old_beta.append(float(decision["old_beta"]))
                adv.append(float(g_adv[ti]))
                tids.append(len(lengths) - 1)
    batch = FlatGateBatch(
        features=torch.tensor(features, dtype=torch.float32),
        z=torch.tensor(z, dtype=torch.float32),
        old_logp=torch.tensor(old_logp, dtype=torch.float32),
        old_alpha=torch.tensor(old_alpha, dtype=torch.float32),
        old_beta=torch.tensor(old_beta, dtype=torch.float32),
        advantage=torch.tensor(adv, dtype=torch.float32),
        trajectory_id=torch.tensor(tids, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.long),
    )
    batch.validate()
    stats = {
        "groups": len(groups),
        "trajectories": int(reward_matrix.numel()),
        "decisions": int(batch.features.shape[0]),
        "reward_mean": float(reward_matrix.mean()),
        "reward_max": float(reward_matrix.max()),
        "successes": int(sum(bool(row["success"]) for group in groups for row in group)),
        "active_groups": int(active.sum()),
        "group_std": [float(x) for x in std],
    }
    return batch, stats


def train_iteration(actor: GateActor, optimizer: torch.optim.Optimizer,
                     groups: list[list[dict]], config: dict) -> dict:
    batch, stats = make_batch(groups)
    actor.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = clipped_gate_loss(actor, batch, clip_eps=float(config["clip_eps"]))
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), float(config["max_grad_norm"])))
    optimizer.step()
    actor.eval()
    out = {**stats, **metrics, "grad_norm": grad_norm, "updated": bool(stats["active_groups"])}
    print(f"[Stage III][更新] GateActor：decisions={stats['decisions']}，active_groups={stats['active_groups']}，"
          f"loss={metrics['loss']:.6f}，KL={metrics['kl_old_new']:.6f}，eta_mean={metrics['actor_eta_mean']:.4f}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--groups-per-iteration", type=int, default=4)
    parser.add_argument("--trajectories-per-group", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=29650)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    if args.iterations <= 0 or args.groups_per_iteration <= 0 or args.trajectories_per_group < 2:
        raise ValueError("iterations/groups must be positive and trajectories-per-group >= 2")
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    tasks = json.loads(MANIFEST.read_text())
    config = {
        "algorithm": "GRPO-style clipped GateActor",
        "base_student": str(STUDENT),
        "gate_distribution": "Beta(alpha,beta), eta=2*z",
        "initial_concentration": 20.0,
        "efficiency_weight": 0.1,
        "clip_eps": 0.2,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "groups_per_iteration": args.groups_per_iteration,
        "trajectories_per_group": args.trajectories_per_group,
        "iterations": args.iterations,
        "gpu": args.gpu,
        "manifest": str(MANIFEST),
        "preservation": "Stage I/II checkpoint and evaluation files are read-only",
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run) if args.run else STAGE3 / "runs" / f"grpo_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (run_dir / "manifest.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False))
    actor_path = run_dir / "checkpoints/actor_init.pt"
    status = {"stage": "III", "phase": "initializing", "run_dir": str(run_dir), "config": config}
    atomic_json(run_dir / "status.json", status)
    print("[Stage III][初始化] 进入隔离训练目录：", run_dir, flush=True)
    verify_baseline("启动前")
    actor, optimizer = load_or_init_actor(actor_path, config)
    server = None
    metrics_path = run_dir / "metrics.jsonl"
    try:
        status.update({"phase": "server_starting", "student": str(STUDENT), "actor": str(actor_path)})
        atomic_json(run_dir / "status.json", status)
        server, url, _ = start_server(run_dir, actor_path, args.gpu, args.port)
        for iteration in range(1, args.iterations + 1):
            print(f"\n[Stage III][第 {iteration}/{args.iterations} 轮] 开始在线交互采样。", flush=True)
            status.update({"phase": "collecting", "iteration": iteration})
            atomic_json(run_dir / "status.json", status)
            groups = []
            for local_group in range(args.groups_per_iteration):
                group_id = (iteration - 1) * args.groups_per_iteration + local_group
                task_path = run_dir / "tasks" / f"task_{local_group:04d}.json"
                task_path.parent.mkdir(parents=True, exist_ok=True)
                task_path.write_text(json.dumps(tasks[local_group % len(tasks)], ensure_ascii=False, indent=2))
                groups.append(collect_group(run_dir, task_path, group_id,
                                            args.trajectories_per_group, actor_path, url, args.gpu))
            print(f"[Stage III][第 {iteration}/{args.iterations} 轮] 采样完成，开始计算组内相对优势并更新 GateActor。", flush=True)
            status.update({"phase": "updating", "iteration": iteration})
            atomic_json(run_dir / "status.json", status)
            metrics = train_iteration(actor, optimizer, groups, config)
            actor_path = run_dir / "checkpoints" / f"actor_iter_{iteration:04d}.pt"
            save_actor(actor_path, actor, optimizer, iteration, config, metrics)
            rpc(url, "/reload_actor", {"actor_checkpoint": str(actor_path)}, timeout=120)
            with metrics_path.open("a") as f:
                f.write(json.dumps({"iteration": iteration, **metrics}, ensure_ascii=False) + "\n")
            status.update({"phase": "iteration_complete", "iteration": iteration,
                           "actor": str(actor_path), "last_metrics": metrics})
            atomic_json(run_dir / "status.json", status)
            print(f"[Stage III][第 {iteration}/{args.iterations} 轮] 完成；新 GateActor 已热加载，Student 仍未改动。", flush=True)
        status.update({"phase": "completed", "completed_at": time.time(), "actor": str(actor_path)})
        atomic_json(run_dir / "status.json", status)
        verify_baseline("训练结束")
        print("[Stage III][完成] GRPO-style 交互强度训练结束，Stage I/II 基线哈希仍一致。", flush=True)
    except Exception as exc:
        status.update({"phase": "failed", "error": repr(exc), "failed_at": time.time()})
        atomic_json(run_dir / "status.json", status)
        print(f"[Stage III][失败] {exc!r}", flush=True)
        raise
    finally:
        if server is not None and server.poll() is None:
            print("[Stage III][服务端] 正在停止隔离推理服务。", flush=True)
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
