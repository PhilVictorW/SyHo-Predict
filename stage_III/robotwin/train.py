"""Reward-only Actor training AFTER measured interaction admission."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from .common import (DEFAULT_CONFIG,FORMAT,atomic_json,canonical_hash,load_checkpoint,load_config,
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
    from .acceptance import validate_admission
    from .make_manifests import validate_manifest,assert_disjoint
    from .gate_actor import GateActor
    from .grpo import update_actor
    from .engine import start_server,collect
    from .runtime import Progress,stop_owned
    from .rpc import call
    admission=validate_admission(cfg,a.admission,a.student)
    manifests=[read_json(p) for p in (a.train_manifest,a.val_manifest,a.test_manifest)]
    for m,s in zip(manifests,("train","val","test")):validate_manifest(m,s)
    assert_disjoint(*manifests)
    if cfg["rl"]["iterations"]>5 and admission["statistics"]["cells"]<100:
        raise ValueError("Beyond the bounded pilot, require full 100-cell admission evidence")
    out=local_output(a.run_dir);out.mkdir(parents=True,exist_ok=True)
    from .common import claim_run
    run_lock=claim_run(out)
    if (out/"status.json").exists() and not a.resume:raise ValueError("Run exists; explicit --resume required")
    r=cfg["rl"];torch.set_num_threads(2);torch.manual_seed(r["seed"])
    actor=GateActor(r["hidden_dim"],r["concentration"]).float()  # CPU updates, no second large model.
    optimizer=torch.optim.Adam(actor.parameters(),lr=r["learning_rate"])
    manifests_hash={s:sha256(p) for s,p in zip(("train","val","test"),(a.train_manifest,a.val_manifest,a.test_manifest))}
    schedule=balanced_schedule(manifests[0]["rows"],r["iterations"]*r["groups_per_iteration"],r["seed"])
    atomic_json(out/"schedule.json",dict(scene_ids=[x["scene_id"] for x in schedule],
                                        manifest_hashes=manifests_hash))
    student_hash=sha256(a.student);iteration=0;total_steps=0;total_episodes=0
    attempted_step_budget=0;update_count=0
    if a.resume:
        saved=load_checkpoint(a.resume)
        if saved["kind"]!="actor" or saved["student_sha256"]!=student_hash or saved["config"]!=cfg or saved["manifest_hashes"]!=manifests_hash:
            raise ValueError("Actor resume identity/config/manifest mismatch")
        actor.load_state_dict(saved["actor"]);optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        iteration=saved["iteration"];total_steps=saved["total_env_steps"]
        total_episodes=saved["total_episodes"];attempted_step_budget=saved["attempted_step_budget"]
        update_count=saved["update_count"]
    def save(i):
        if shutil.disk_usage(out).free<r["minimum_free_gib"]*2**30:raise RuntimeError("Actor disk reserve exhausted")
        path=out/"checkpoints"/("actor_%04d.pt"%i)
        torch_save_atomic(path,dict(format=FORMAT,kind="actor",control_site="vlm_goal_local",
            student_sha256=student_hash,admission_sha256=sha256(a.admission),actor=actor.state_dict(),
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),config=cfg,
            iteration=i,update_count=update_count,total_env_steps=total_steps,total_episodes=total_episodes,
            attempted_step_budget=attempted_step_budget,manifest_hashes=manifests_hash))
        atomic_json(out/"last_checkpoint.json",dict(path=str(path),iteration=i))
        return path
    actor_path=Path(a.resume).resolve() if a.resume else save(0)
    progress=Progress(out/"status.json")
    progress.report("loading",meaningful=True,iteration=iteration,total=r["iterations"])
    proc,url,info=start_server(cfg,a.config,a.student,a.gpu,out)
    try:
        for it in range(iteration,r["iterations"]):
            begin=it*r["groups_per_iteration"]
            iteration_reserve=sum(r["group_size"]*row["max_steps"]*(cfg["runtime"]["max_infrastructure_retries"]+1)
                                  for row in schedule[begin:begin+r["groups_per_iteration"]])
            if attempted_step_budget+iteration_reserve>r["max_env_steps"]:
                progress.report("stopped_at_interaction_budget",meaningful=True,iteration=it)
                return  # Preserve the last committed Actor and its behavior hash.
            current_hash=sha256(actor_path)
            ack=call(url,"/actor",dict(path=str(actor_path)))
            if ack["actor_sha256"]!=current_hash:raise RuntimeError("Actor reload was not acknowledged")
            groups=[];skipped=[]
            for j in range(r["groups_per_iteration"]):
                gid=it*r["groups_per_iteration"]+j
                row=schedule[gid]
                cost=r["group_size"]*row["max_steps"]*(cfg["runtime"]["max_infrastructure_retries"]+1)
                # Count all attempted groups conservatively, including discarded partial rollouts.
                request=dict(url=url,student_sha256=student_hash,actor_sha256=current_hash,
                             policy="actor_sample",noise_repeat=it,eta=None)
                result=collect(cfg,a.config,row,request,gid,r["group_size"],a.gpu,
                               out/"groups"/("group_%06d"%gid))
                attempted_step_budget+=cost
                if result["status"]=="completed":
                    group=result["trajectories"];groups.append(group)
                    total_steps+=sum(x["env_steps"] for x in group);total_episodes+=len(group)
                else:skipped.append(dict(group_id=gid,reason=result["reason"]))
                progress.report("collecting",meaningful=True,iteration=it,group=j+1,
                                env_steps=total_steps,episodes=total_episodes,skipped=len(skipped))
            metrics=update_actor(actor,optimizer,groups,r) if groups else dict(updated=False,updates=0,reason="no_complete_groups")
            update_count+=metrics["updates"]
            actor_path=save(it+1)
            record=dict(iteration=it+1,**metrics,skipped=skipped,total_env_steps=total_steps,
                        total_episodes=total_episodes,attempted_step_budget=attempted_step_budget,time=time.time())
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
            progress.report("iteration_committed",meaningful=True,iteration=it+1,last_metrics=record)
            if (it+1)%r["eval_every"]==0 or it+1==r["iterations"]:
                # Reuse the same frozen server, avoiding a second backbone on this GPU.
                from .evaluate import evaluate
                report=evaluate(cfg,a.config,a.student,a.val_manifest,out/("validation_%04d"%(it+1)),
                                a.gpu,actor_path,session=(proc,url,info))
                atomic_json(out/"last_validation.json",dict(iteration=it+1,
                    actor_sha256=sha256(actor_path),complete=report["complete"],
                    successes={k:sum(int(x["success"]) for x in v) for k,v in report["arms"].items()},
                    episodes={k:len(v) for k,v in report["arms"].items()}))
        progress.report("completed_pending_final_comparison",meaningful=True,iteration=r["iterations"])
    finally:stop_owned(proc)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student",required=True)
    p.add_argument("--admission",required=True);p.add_argument("--train-manifest",required=True)
    p.add_argument("--val-manifest",required=True);p.add_argument("--test-manifest",required=True)
    p.add_argument("--run-dir",required=True);p.add_argument("--resume");p.add_argument("--gpu")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);a.config=str(Path(a.config).resolve())
    cfg=load_config(a.config)
    try:run(a,cfg)
    except BaseException as exc:
        atomic_json(local_output(a.run_dir)/"failure.json",dict(error=repr(exc),time=time.time()))
        raise
if __name__=="__main__":main()
