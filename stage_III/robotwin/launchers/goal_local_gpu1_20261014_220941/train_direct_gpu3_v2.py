"""User-authorized training-first Stage III; held-out assessment runs independently."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
import os
import signal
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from stage_III.robotwin.common import (DEFAULT_CONFIG,FORMAT,atomic_json,canonical_hash,load_checkpoint,load_config,
                     local_output,read_json,require_execute,sha256,torch_save_atomic)

def balanced_schedule(rows, count, seed):
    import random
    rng=random.Random(seed);cells=defaultdict(list)
    for r in rows:cells[(r["task"],r["mode"])].append(r)
    keys=sorted(cells)
    for rs in cells.values():rng.shuffle(rs)
    used=defaultdict(int);schedule=[]
    for i in range(count):
        if i%len(keys)==0:rng.shuffle(keys)
        key=keys[i%len(keys)];r=cells[key][used[key]%len(cells[key])];used[key]+=1
        schedule.append(r)
    return schedule

def run(a,cfg):
    import torch
    from stage_III.robotwin.make_manifests import validate_manifest,assert_disjoint
    from stage_III.robotwin.gate_actor import GateActor
    from stage_III.robotwin.grpo import update_actor
    from timeout_runtime import OnlineRuntime, POLICY
    from stage_III.robotwin.runtime import Progress,stop_owned
    manifests=[read_json(a.train_manifest)]
    validate_manifest(manifests[0], "train")
    out=local_output(a.run_dir);out.mkdir(parents=True,exist_ok=True)
    from stage_III.robotwin.common import claim_run
    run_lock=claim_run(out)
    if (out/"status.json").exists() and not a.resume:raise ValueError("Run exists; explicit --resume required")
    startup=verify_startup(a,cfg,out,manifests[0])
    r=cfg["rl"];torch.set_num_threads(2);torch.manual_seed(r["seed"])
    actor=GateActor(r["hidden_dim"],r["concentration"]).float()  # CPU updates, no second large model.
    optimizer=torch.optim.Adam(actor.parameters(),lr=r["learning_rate"])
    manifests_hash={"train":sha256(a.train_manifest)}
    schedule=balanced_schedule(manifests[0]["rows"],r["iterations"]*r["groups_per_iteration"],r["seed"])
    atomic_json(out/"schedule.json",dict(scene_ids=[x["scene_id"] for x in schedule],
                                        manifest_hashes=manifests_hash))
    student_hash=sha256(a.student);iteration=0;total_steps=0;total_episodes=0
    attempted_step_budget=0;update_count=0
    if a.resume:
        saved=load_checkpoint(a.resume)
        if saved["kind"]!="actor" or saved["student_sha256"]!=student_hash or saved["config"]!=cfg or saved["manifest_hashes"]!=manifests_hash:
            raise ValueError("Actor resume identity/config/manifest mismatch")
        if saved.get("startup_sha256") != sha256(out/"startup.json"):
            raise ValueError("Training-first authorization/source identity changed")
        if saved.get("runtime_revision_sha256") not in (None,sha256(out/"runtime_revision_v2.json")):
            raise ValueError("Timeout runtime revision changed")
        actor.load_state_dict(saved["actor"]);optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        iteration=saved["iteration"];total_steps=saved["total_env_steps"]
        total_episodes=saved["total_episodes"];attempted_step_budget=saved["attempted_step_budget"]
        update_count=saved["update_count"]
    def save(i):
        if shutil.disk_usage(out).free<r["minimum_free_gib"]*2**30:raise RuntimeError("Actor disk reserve exhausted")
        path=out/"checkpoints"/("actor_%04d.pt"%i)
        torch_save_atomic(path,dict(format=FORMAT,kind="actor",control_site="vlm_goal_local",
            student_sha256=student_hash,admission_sha256=None,
            evaluation_policy="deferred",startup_sha256=sha256(out/"startup.json"),
            driver_sha256=sha256(__file__),runtime_revision_sha256=sha256(out/"runtime_revision_v2.json"),
            timeout_policy=POLICY,actor=actor.state_dict(),
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),config=cfg,
            iteration=i,update_count=update_count,total_env_steps=total_steps,total_episodes=total_episodes,
            attempted_step_budget=attempted_step_budget,manifest_hashes=manifests_hash))
        atomic_json(out/"last_checkpoint.json",dict(path=str(path),iteration=i))
        return path
    actor_path=Path(a.resume).resolve() if a.resume else save(0)
    progress=Progress(out/"status.json")
    progress.report("loading",meaningful=True,iteration=iteration,total=r["iterations"],gpu=int(a.gpu),
                    evaluation_policy="deferred",update_count=update_count)
    adoption=read_json(a.adopt_runtime) if getattr(a,"adopt_runtime",None) else None
    runtime=OnlineRuntime(cfg,a.config,a.student,a.gpu,out,adoption)
    say("已启用超时约束：采样超时直接跳过；权重加载后40分钟无任务进度跳过；训练不因这些超时退出。")
    try:
        for it in range(iteration,r["iterations"]):
            begin=it*r["groups_per_iteration"]
            iteration_reserve=sum(r["group_size"]*row["max_steps"]*(cfg["runtime"]["max_infrastructure_retries"]+1)
                                  for row in schedule[begin:begin+r["groups_per_iteration"]])
            if attempted_step_budget+iteration_reserve>r["max_env_steps"]:
                progress.report("stopped_at_interaction_budget",meaningful=True,iteration=it)
                return  # Preserve the last committed Actor and its behavior hash.
            current_hash=sha256(actor_path)
            runtime.set_actor(actor_path,current_hash)
            groups=[];skipped=[]
            for j in range(r["groups_per_iteration"]):
                gid=it*r["groups_per_iteration"]+j
                row=schedule[gid]
                cost=r["group_size"]*row["max_steps"]*(cfg["runtime"]["max_infrastructure_retries"]+1)
                # Count all attempted groups conservatively, including discarded partial rollouts.
                request=dict(url=runtime.url,student_sha256=student_hash,actor_sha256=current_hash,
                             policy="actor_sample",noise_repeat=it,eta=None)
                say(f"第{it+1}/{r['iterations']}轮，第{j+1}/{r['groups_per_iteration']}组：{row['task']} / {row['mode']}，4条轨迹。")
                progress.report("collecting",meaningful=True,iteration=it,group=j+1,task=row["task"],
                                mode=row["mode"],env_steps=total_steps,episodes=total_episodes)
                result=runtime.collect(cfg,a.config,row,request,gid,r["group_size"],a.gpu,
                               out/"groups"/("group_%06d"%gid))
                attempted_step_budget+=cost
                if result["status"]=="completed":
                    group=result["trajectories"];groups.append(group)
                    total_steps+=sum(x["env_steps"] for x in group);total_episodes+=len(group)
                    say(f"第{gid+1}组完成：成功{sum(int(x['success']) for x in group)}/4；奖励 {[round(x['reward'],5) for x in group]}")
                else:
                    skipped.append(dict(group_id=gid,reason=result["reason"]))
                    say(f"第{gid+1}组跳过：{result['reason']}")
                progress.report("collecting",meaningful=True,iteration=it,group=j+1,
                                env_steps=total_steps,episodes=total_episodes,skipped=len(skipped))
            metrics=update_actor(actor,optimizer,groups,r) if groups else dict(updated=False,updates=0,reason="no_complete_groups")
            update_count+=metrics["updates"]
            actor_path=save(it+1)
            flat=[x for group in groups for x in group]
            record=dict(iteration=it+1,**metrics,skipped=skipped,update_count=update_count,
                        successes=sum(int(x["success"]) for x in flat),
                        reward_mean=sum(x["reward"] for x in flat)/len(flat) if flat else None,
                        total_env_steps=total_steps,
                        total_episodes=total_episodes,attempted_step_budget=attempted_step_budget,time=time.time())
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
            progress.report("iteration_committed",meaningful=True,iteration=it+1,last_metrics=record,update_count=update_count)
            say(f"第{it+1}轮已保存：Actor更新={metrics['updated']}，累计更新{update_count}，KL={metrics.get('kl')}，累计有效轨迹{total_episodes}。")
            if (it+1)%r["eval_every"]==0 or it+1==r["iterations"]:
                # Queue held-out evaluation without blocking or contaminating training.
                atomic_json(out/"evaluation_requests"/("iteration_%04d.json"%(it+1)),
                    dict(iteration=it+1,actor_path=str(actor_path),actor_sha256=sha256(actor_path),
                         student_sha256=student_hash,status="pending_held_out_evaluation"))
                say(f"第{it+1}轮评估请求已记录；训练继续，完整评估后置。")
        progress.report("completed_pending_final_comparison",meaningful=True,iteration=r["iterations"])
    finally:runtime.close()

def say(message):
    print(time.strftime("[%m-%d %H:%M:%S] ")+message,flush=True)

def verify_startup(a,cfg,out,manifest):
    # Preserve and verify the original startup identity; record this operational revision separately.
    from train_direct_gpu3 import verify_startup as original_verify
    from timeout_runtime import POLICY
    original=original_verify(a,cfg,out,manifest)
    files=("train_direct_gpu3.py","train_direct_gpu3_v2.py","timeout_runtime.py","rollout_timeout_v2.py")
    revision=dict(version=2,authorization="用户要求采样超时跳过，加载后40分钟无进度跳过，训练继续",
        parent_startup_sha256=sha256(out/"startup.json"),config_sha256=canonical_hash(cfg),
        code={name:sha256(Path(__file__).with_name(name)) for name in files},timeout_policy=POLICY,
        optimizer_and_sampling_schedule="unchanged",checks_gpu7="unchanged")
    path=out/"runtime_revision_v2.json"
    if path.exists() and read_json(path)!=revision:raise ValueError("Runtime revision identity changed")
    atomic_json(path,revision)
    return original


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student",required=True)
    p.add_argument("--train-manifest",required=True)
    p.add_argument("--source-run",required=True)
    p.add_argument("--defer-pretraining-evaluation",action="store_true")
    p.add_argument("--run-dir",required=True);p.add_argument("--resume");p.add_argument("--gpu")
    p.add_argument("--adopt-runtime")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);a.config=str(Path(a.config).resolve())
    cfg=load_config(a.config)
    if not a.defer_pretraining_evaluation: p.error("Explicit training-first option is required")
    from stage_III.robotwin.common import configure_gpu
    configure_gpu(a.gpu)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:run(a,cfg)
    except BaseException as exc:
        atomic_json(local_output(a.run_dir)/"failure.json",dict(error=repr(exc),time=time.time()))
        raise
if __name__=="__main__":main()
