"""One LIBERO task, 50 episodes, original rollout and first success video."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from goal_local.eval_common import SUITES, atomic_json
from .client import Stage3Client, rpc

class EvaluationClient(Stage3Client):
    def __init__(self, checkpoint, actor_checkpoint, url):
        super().__init__(checkpoint, actor_checkpoint, url, trajectory_seed=0)

    def step(self, example, step=0, return_intermediates=False, **kwargs):
        if return_intermediates:
            raise ValueError("Evaluation does not request similarity features")
        if self.actions is None or self.cursor == len(self.actions):
            # Exactly the historical 98.5% StudentClient action-noise schedule.
            seed = self.trajectory_seed + self.chunk
            result = rpc(self.url, "/infer", dict(
                checkpoint=self.checkpoint, actor_checkpoint=self.actor_checkpoint,
                main=self._encode(example["primary_image"]),
                wrist=self._encode(example["wrist_image"]),
                instruction=str(example["lang"]), noise_seed=int(seed), gate_seed=0))
            actions = np.asarray(result["actions"], dtype=np.float32)
            if result.get("normalized") is not False or actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise ValueError("Invalid Stage III denormalized action chunk")
            self.actions, self.cursor = actions, 0
            self.decision_records.append(dict(chunk=self.chunk, noise_seed=int(seed), **result["gate"]))
            self.chunk += 1
        action = self.actions[self.cursor].copy()
        self.cursor += 1
        return {"raw_action": dict(world_vector=action[:3], rotation_delta=action[3:6], open_gripper=action[6:7])}

def main():
    p=argparse.ArgumentParser()
    for key in ("checkpoint", "actor-checkpoint", "url", "suite", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--episodes", type=int, default=50)
    a=p.parse_args()
    if a.suite not in SUITES or not 0 <= a.task_id < 10:
        raise ValueError("Invalid classic LIBERO task")
    from examples.LIBERO.eval_files import libero_eval_core as core
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()[a.suite]()
    if len(suite.get_task_init_states(a.task_id)) < a.episodes:
        raise ValueError("Need distinct initial states for all requested episodes")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    videos=list(out.glob("*_success.mp4"))
    if len(videos)>1:raise RuntimeError("More than one success video for this task")
    for video in videos:
        try:
            reader=core.imageio.get_reader(video)
            reader.get_data(0);reader.close()
        except Exception:
            video.rename(video.with_suffix(".invalid_"+str(time.time_ns())))
    if not list(out.glob("*_success.mp4")):
        (out/f".task{a.task_id:02d}_first_success_video.claim").unlink(missing_ok=True)
    client=EvaluationClient(a.checkpoint,a.actor_checkpoint,a.url)
    core._build_model_client=lambda args:client
    original=core._run_single_episode
    def episode(**kw):
        kw["args"].save_videos=not any(out.glob("*_success.mp4"))
        kw["client_model"].configure_episode(SUITES.index(a.suite),kw["task_id"],kw["episode_idx"])
        result=original(**kw)
        if result.error:raise RuntimeError(result.error)
        record=dict(suite=a.suite,task_id=kw["task_id"],episode_idx=kw["episode_idx"],
                    success=result.success,num_actions=result.num_actions,decisions=client.decision_records,
                    recorded_at=time.time())
        with (out/"gate_diagnostics.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        print(json.dumps(dict(event="episode_complete",suite=a.suite,task_id=kw["task_id"],
                              episode_idx=kw["episode_idx"],success=result.success,
                              num_actions=result.num_actions,time=time.time())),flush=True)
        return result
    core._run_single_episode=episode
    atomic_json(out/"evaluation_protocol.json",dict(checkpoint=a.checkpoint,actor_checkpoint=a.actor_checkpoint,
        suite=a.suite,task_id=a.task_id,episodes=a.episodes,initial_state_indices=list(range(a.episodes)),
        simulator_seed=0,gate_mode="beta_mean",actor_precision="float32",action_chunk=8,flow_steps=10,
        noise_seed_formula="(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index",
        video_policy="first successful episode only"))
    args=core.EvalArgs(pretrained_path=a.checkpoint,task_suite_name=a.suite,task_id=a.task_id,
        num_trials_per_task=a.episodes,num_workers=1,seed=0,num_steps_wait=10,save_videos=True,
        save_first_success_video=True,resume_incomplete=True,video_out_path=str(out),eval_action_chunk_len=8)
    core.eval_libero(args)
    # If a pre-existing video was corrupt, regenerate only its visualization;
    # never append the replay as a new scored episode.
    if not list(out.glob("*_success.mp4")):
        rows=[json.loads(x) for x in (out/"episodes.checkpoint.jsonl").read_text().splitlines() if x.strip()]
        successes=[x for x in rows if x["success"] and not x.get("error")]
        if successes:
            adapter=core.get_benchmark_adapter("libero")
            for row in successes:
                result=episode(args=args,adapter=adapter,task=suite.get_task(a.task_id),task_id=a.task_id,
                    episode_idx=row["episode_idx"],init_state=suite.get_task_init_states(a.task_id)[row["episode_idx"]],
                    client_model=client,run_output_dir=out,env_seed=0)
                if result.rollout_video_path:
                    atomic_json(out/"video_repair.json",dict(episode_idx=row["episode_idx"],path=result.rollout_video_path))
                    break
            else:raise RuntimeError("Could not regenerate a successful video")
    videos=list(out.glob("*_success.mp4"))
    if len(videos)>1 or list(out.glob("*_failure.mp4")):
        raise RuntimeError("Video policy violated")

if __name__=="__main__":
    main()
