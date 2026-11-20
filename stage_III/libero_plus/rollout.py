"""Online LIBERO-Plus trajectory collector for Stage III.

It reuses the already verified Plus environment adapter and action conversion,
but writes all trajectory data below stage_III/libero_plus only.
"""
from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "LaWAM_without_subgoal"
PLUS_RUNTIME = SOURCE / "libero-plus/runtime"
for path in (SOURCE, PLUS_RUNTIME, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate_actor import success_efficiency_reward  # noqa: E402
from client import Stage3Client  # noqa: E402
from runtime_safety import Progress, cached_trajectory, file_sha256, patch_reset


def _imports():
    from sim_compat import enable
    enable()
    # LIBERO-plus 的环境噪声代码仍使用 np.float_；NumPy 2.0 已移除该别名。
    # 只在 Stage III rollout 子进程内提供兼容别名，不修改全局环境或原始 LIBERO 文件。
    import numpy as np
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    import libero.libero as libero
    from libero.libero.envs.env_wrapper import ControlEnv
    patch_reset(ControlEnv)
    from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter
    import plus_eval_core as core
    return libero, get_benchmark_adapter, core


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _state_hash(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def run_trajectory(task_row: dict, group_id: int, trajectory_idx: int,
                   initial_state_idx: int, checkpoint: str, actor_checkpoint: str,
                   url: str, output: Path, efficiency_weight: float = 0.1) -> dict:
    faulthandler.enable()
    progress = Progress(output, group_id, trajectory_idx)
    progress.report('导入仿真环境', force=True)
    actor_hash = file_sha256(Path(actor_checkpoint))
    libero, get_benchmark_adapter, core = _imports()
    benchmark = libero.benchmark.get_benchmark_dict()
    suite_name = str(task_row["suite"])
    task_id = int(task_row["task_id"])
    suite = benchmark[suite_name]()
    task = suite.get_task(task_id)
    if getattr(task, "name", None) != task_row["name"]:
        raise ValueError(f"task manifest mismatch: {task.name!r} != {task_row['name']!r}")
    initial_states = suite.get_task_init_states(task_id)
    if initial_state_idx >= len(initial_states):
        raise ValueError(f"init_state_idx={initial_state_idx} unavailable for task {task_id}")
    initial_state = initial_states[initial_state_idx]
    state_hash = _state_hash(initial_state)
    adapter = get_benchmark_adapter("libero_plus")
    original_build = adapter.build_env
    def build_with_progress(task, resolution, seed):
        progress.report('创建环境', force=True)
        env, description = original_build(task, resolution, seed)
        for name, label in [('reset', '重置环境'), ('set_init_state', '载入固定初始状态')]:
            original = getattr(env, name)
            def wrapped(*a, _original=original, _label=label, **kw):
                progress.report(_label, force=True)
                result = _original(*a, **kw)
                progress.report(_label + '完成', force=True)
                return result
            setattr(env, name, wrapped)
        original_step = env.step
        def step_with_progress(*a, **kw):
            progress.report('仿真执行动作')
            return original_step(*a, **kw)
        env.step = step_with_progress
        return env, description
    adapter = dataclasses.replace(adapter, build_env=build_with_progress)
    args = core.EvalArgs(
        pretrained_path=checkpoint,
        benchmark_variant="libero_plus",
        task_suite_name=suite_name,
        num_trials_per_task=1,
        num_workers=1,
        seed=group_id,
        num_steps_wait=10,
        save_videos=False,
        eval_action_chunk_len=8,
    )
    seed = group_id * 1_000_000 + trajectory_idx * 10_000 + task_id
    client = Stage3Client(checkpoint, actor_checkpoint, url, seed)
    original_client_step = client.step
    def policy_step(*a, **kw):
        progress.report('模型推理', step=kw.get('step'))
        return original_client_step(*a, **kw)
    client.step = policy_step
    client.configure_episode(suite_index=list(benchmark).index(suite_name), task_id=task_id,
                             episode_idx=trajectory_idx, group_id=group_id,
                             trajectory_idx=trajectory_idx)
    begin = time.time()
    print(f"[Stage III][rollout] 组 {group_id} / 轨迹 {trajectory_idx}: "
          f"{suite_name}:{task_id}，初始状态哈希 {state_hash[:12]}", flush=True)
    result = core._run_single_episode(
        args=args, adapter=adapter, task=task, task_id=task_id,
        episode_idx=trajectory_idx, init_state=initial_state,
        client_model=client, run_output_dir=output, env_seed=group_id,
    )
    if result.error:
        raise RuntimeError(result.error)
    max_steps = adapter.get_max_steps(suite_name)
    reward = success_efficiency_reward(result.success, result.num_actions,
                                        max_steps, efficiency_weight)
    payload = dict(
        group_id=group_id,
        trajectory_idx=trajectory_idx,
        suite=suite_name,
        task_id=task_id,
        task_name=task_row["name"],
        category=task_row.get("category"),
        initial_state_idx=initial_state_idx,
        initial_state_hash=state_hash,
        behavior_policy="Stage3GateActor",
        base_student_checkpoint=str(Path(checkpoint).resolve()),
        actor_checkpoint=str(Path(actor_checkpoint).resolve()),
        actor_sha256=actor_hash,
        success=bool(result.success),
        env_steps=int(result.num_actions),
        max_steps=int(max_steps),
        reward=float(reward),
        termination="success_or_done" if result.success else "timeout_or_failure",
        decisions=client.decision_records,
        collected_seconds=time.time() - begin,
        collected_at=time.time(),
    )
    target = output / "trajectories" / f"group_{group_id:04d}_traj_{trajectory_idx:02d}.json"
    _atomic(target, payload)
    progress.report('轨迹已保存', force=True)
    faulthandler.cancel_dump_traceback_later()
    print(f"[Stage III][rollout] 组 {group_id} / 轨迹 {trajectory_idx} 完成："
          f"success={payload['success']}，环境步数={payload['env_steps']}，"
          f"决策次数={len(payload['decisions'])}，reward={payload['reward']:.4f}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--initial-state-idx", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task = json.loads(Path(args.task_json).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actor_hash = file_sha256(Path(args.actor_checkpoint))
    for trajectory_idx in range(args.trajectories):
        target = output / 'trajectories' / f'group_{args.group:04d}_traj_{trajectory_idx:02d}.json'
        if cached_trajectory(target, group=args.group, trajectory=trajectory_idx,
                             task=task, student=Path(args.checkpoint),
                             actor=Path(args.actor_checkpoint), actor_hash=actor_hash,
                             initial_state_idx=args.initial_state_idx) is not None:
            print(f'[Stage III][恢复] 组 {args.group} / 轨迹 {trajectory_idx} 已存在，校验通过后复用。', flush=True)
            continue
        run_trajectory(task, args.group, trajectory_idx, args.initial_state_idx,
                       args.checkpoint, args.actor_checkpoint, args.url, output)


if __name__ == "__main__":
    main()

