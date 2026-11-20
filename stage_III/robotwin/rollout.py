"""One isolated scene probe or complete same-state rollout group."""
from __future__ import annotations
import argparse
import hashlib
import time
from pathlib import Path
from .common import (DEFAULT_CONFIG,load_config,read_json,atomic_json,local_output,require_execute,
                     configure_gpu)
from .runtime import Progress
from .rpc import call,array_decode,ModelError

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:4],"little")%(2**31)

def trajectory(cfg,row,group,index,request,out,progress):
    from .env_adapter import (prepare_args,reset_scene,same_initial,config_identity,
                              observation_request,physical_actions,InvalidPolicy)
    from .grpo import reward
    args,(_,_,constructor)=prepare_args(cfg,row,out)
    env,obs,fp=reset_scene(cfg,row,args,constructor)
    decisions=[];steps=0
    try:
        same_initial(row,fp,config_identity(cfg,row,args))
        if int(env.step_lim)!=row["max_steps"]:raise ValueError("step_lim changed")
        progress.report("policy_takeover",meaningful=True,trajectory=index,env_steps=0)
        stats=read_json(cfg["base"]["statistics"])
        while not env.eval_success and env.take_action_cnt<env.step_lim:
            k=len(decisions)
            # Evaluation methods share flow noise; training replicas use independent streams.
            noise_seed=seed_for("flow",row["scene_id"],request["noise_repeat"],index,k)
            req=dict(**observation_request(obs,row["instruction"]),
                     policy=request["policy"],eta=request.get("eta"),
                     student_sha256=request["student_sha256"],actor_sha256=request.get("actor_sha256"),
                     noise_seed=noise_seed,gate_seed=seed_for("gate",group,index,k,request["noise_repeat"]))
            reply=call(request["url"],"/predict",req,timeout=cfg["runtime"]["stall_seconds"])
            if reply["student_sha256"]!=request["student_sha256"] or reply.get("actor_sha256")!=request.get("actor_sha256"):
                raise InvalidPolicy("Policy changed during trajectory")
            actions=physical_actions(array_decode(reply["actions"]),stats)
            start=int(env.take_action_cnt)
            for action in actions:
                if env.eval_success or env.take_action_cnt>=env.step_lim:break
                before=int(env.take_action_cnt)
                env.take_action(action,action_type="ee")
                steps=int(env.take_action_cnt)
                if steps!=before+1:raise InvalidPolicy("Unexpected environment action counter")
                progress.report("executing",meaningful=True,trajectory=index,env_steps=steps)
            used=int(env.take_action_cnt)-start
            if used<=0:raise InvalidPolicy("Generated action chunk was never executed")
            d=reply["gate"] or {}
            d.update(chunk_start_step=start,executed_steps=used,noise_seed=noise_seed)
            decisions.append(d)
            if not env.eval_success and env.take_action_cnt<env.step_lim:obs=env.get_obs()
        success=bool(env.eval_success);steps=int(env.take_action_cnt)
        return dict(group_id=group,trajectory_idx=index,scene_id=row["scene_id"],
            task=row["task"],mode=row["mode"],instruction=row["instruction"],
            initial_state_hash=fp["state_hash"],env_config_hash=row["env_config_hash"],
            student_sha256=request["student_sha256"],actor_sha256=request.get("actor_sha256"),
            policy=request["policy"],eta=request.get("eta"),noise_repeat=request["noise_repeat"],
            success=success,env_steps=steps,max_steps=int(env.step_lim),
            reward=reward(success,steps,int(env.step_lim),cfg["rl"]["efficiency_weight"]),
            termination="success" if success else "step_limit",decisions=decisions)
    finally:
        env.close_env()

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--row",required=True)
    p.add_argument("--output",required=True);p.add_argument("--request")
    p.add_argument("--probe",action="store_true");p.add_argument("--group",type=int,default=0)
    p.add_argument("--trajectories",type=int,default=4);p.add_argument("--gpu")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);configure_gpu(a.gpu);cfg=load_config(a.config)
    out=local_output(a.output);out.mkdir(parents=True,exist_ok=True)
    progress=Progress(out/"progress.json");progress.report("initializing",meaningful=False)
    row=read_json(a.row)
    try:
        if a.probe:
            from .env_adapter import probe_scene
            result=probe_scene(cfg,row,out,progress)
        else:
            if not row.get("prepared"):raise ValueError("Scene has not passed actual reset verification")
            req=read_json(a.request)
            rows=[]
            for index in range(a.trajectories):
                r=trajectory(cfg,row,a.group,index,req,out,progress)
                atomic_json(out/("trajectory_%02d.json"%index),r)
                rows.append(r)
            result=dict(status="completed",trajectories=rows)
        atomic_json(out/"result.json",result)
        progress.report("completed",meaningful=True)
    except BaseException as exc:
        from .env_adapter import InvalidPolicy,SceneMismatch
        kind="model_error" if isinstance(exc,ModelError) else "invalid_policy" if isinstance(exc,InvalidPolicy) else "scene_mismatch" if isinstance(exc,SceneMismatch) else "runtime_error"
        atomic_json(out/"failure.json",dict(kind=kind,error=repr(exc),time=time.time()))
        raise
if __name__=="__main__":main()
