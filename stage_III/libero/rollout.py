"""Online classic-LIBERO trajectory collector for isolated Stage III."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "LaWAM_aba2"
LIBERO_SOURCE = SOURCE / "runtime/LIBERO"
for path in (SOURCE, LIBERO_SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate_actor import success_efficiency_reward  # noqa: E402
from client import Stage3Client  # noqa: E402


def _imports():
    # The classic evaluator imports a websocket client only for its optional
    # remote-policy interface. Stage III calls _run_single_episode directly
    # with the local client, so provide a type-only stub in the LIBERO env.
    try:
        import websockets.sync.client  # noqa: F401
    except Exception:

        ws = types.ModuleType("websockets")
        sync = types.ModuleType("websockets.sync")
        client_mod = types.ModuleType("websockets.sync.client")
        client_mod.ClientConnection = type("ClientConnection", (), {})
        sync.client = client_mod
        ws.sync = sync
        sys.modules.update({"websockets": ws, "websockets.sync": sync,
                            "websockets.sync.client": client_mod})
    # The evaluator's remote ModelClient is imported for typing/API parity but
    # never instantiated by this local Stage3Client. Bypass optional msgpack.
    remote_mod = types.ModuleType("deployment.model_server.tools.websocket_policy_client")
    remote_mod.WebsocketClientPolicy = type("WebsocketClientPolicy", (), {})
    sys.modules["deployment.model_server.tools.websocket_policy_client"] = remote_mod
    from libero.libero import benchmark
    from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter
    from examples.LIBERO.eval_files import libero_eval_core as core
    return benchmark, get_benchmark_adapter, core


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
    benchmark, get_benchmark_adapter, core = _imports()
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_name = str(task_row["suite"])
    task_id = int(task_row["task_id"])
    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)
    manifest_name = task_row.get("name")
    actual_name = str(getattr(task, "name", None) or task.language)
    expected_name = str(manifest_name or actual_name)
    if manifest_name is not None and actual_name != expected_name:
        raise ValueError(f"task manifest mismatch: {actual_name!r} != {expected_name!r}")
    initial_states = suite.get_task_init_states(task_id)
    if not 0 <= initial_state_idx < len(initial_states):
        raise ValueError(f"init_state_idx={initial_state_idx} unavailable for task {task_id}")
    initial_state = initial_states[initial_state_idx]
    state_hash = _state_hash(initial_state)
    adapter = get_benchmark_adapter("libero")
    args = core.EvalArgs(
        pretrained_path=checkpoint,
        benchmark_variant="libero",
        task_suite_name=suite_name,
        num_trials_per_task=1,
        num_workers=1,
        seed=group_id,
        num_steps_wait=10,
        save_videos=False,
        eval_action_chunk_len=8,
    )
    seed = group_id * 1_000_000_000 + trajectory_idx * 10_000_000 + task_id
    client = Stage3Client(checkpoint, actor_checkpoint, url, seed)
    suite_index = list(benchmark_dict).index(suite_name)
    client.configure_episode(suite_index=suite_index, task_id=task_id,
                             episode_idx=trajectory_idx, group_id=group_id,
                             trajectory_idx=trajectory_idx)
    begin = time.time()
    print(f"[Stage III][rollout] 组 {group_id} / 轨迹 {trajectory_idx}: "
          f"{suite_name}:{task_id}，初始状态索引 {initial_state_idx}，状态哈希 {state_hash[:12]}",
          flush=True)
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
        task_name=expected_name,
        category=task_row.get("category"),
        initial_state_idx=initial_state_idx,
        initial_state_hash=state_hash,
        behavior_policy="Stage3GateActor",
        base_student_checkpoint=str(Path(checkpoint).resolve()),
        actor_checkpoint=str(Path(actor_checkpoint).resolve()),
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
    for trajectory_idx in range(args.trajectories):
        run_trajectory(task, args.group, trajectory_idx, args.initial_state_idx,
                       args.checkpoint, args.actor_checkpoint, args.url, output)


if __name__ == "__main__":
    main()
