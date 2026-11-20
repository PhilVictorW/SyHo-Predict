"""Process-local RoboTwin setup and measured initial-state identity."""
from __future__ import annotations
import hashlib
import os
import random
import sys
from pathlib import Path
import numpy as np
from .common import ROOT,canonical_hash,sha256
from .rpc import array_encode

class SceneMismatch(RuntimeError):pass
class InvalidPolicy(RuntimeError):pass

def seed_reset(seed):
    import torch
    random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))

def imports(cfg):
    root=Path(cfg["runtime"]["sim_root"])
    os.chdir(root)
    for p in (str(ROOT),str(root),str(ROOT/"examples/Robotwin/eval_files"),
              str(root/"description")):
        if p not in sys.path:sys.path.insert(0,p)
    from robotwin.round10.bridge import prepare_task_args,_try_seed_once,_select_instruction_native
    from script.eval_policy import class_decorator
    return prepare_task_args,_try_seed_once,_select_instruction_native,class_decorator

def config_identity(cfg,row,args):
    clean={k:v for k,v in args.items() if k not in ("save_path","eval_video_save_dir")}
    root=Path(cfg["runtime"]["sim_root"])
    files={str(p.relative_to(root)):sha256(p) for p in
           (root/"task_config"/(row["mode"]+".yml"),root/"envs"/(row["task"]+".py"),
            root/"envs/_base_task.py")}
    return canonical_hash(dict(args=clean,files=files,protocol="fresh_setup_seed_all_rng_v1"))

def fingerprint(env,obs,decimals):
    def values(x):
        a=np.asarray(x,dtype=np.float64)
        if not np.isfinite(a).all():raise SceneMismatch("Nonfinite physical state")
        return np.round(a,decimals).tolist()
    def pose(x):
        p=x.get_root_pose() if hasattr(x,"get_root_pose") else x.get_pose()
        return dict(p=values(p.p),q=values(p.q))
    actors=[]
    for obj in env.scene.get_all_actors():
        r=dict(name=str(obj.get_name()),pose=pose(obj))
        for name in ("get_velocity","get_angular_velocity"):
            if hasattr(obj,name):r[name]=values(getattr(obj,name)())
        actors.append(r)
    arts=[]
    for obj in env.scene.get_all_articulations():
        r=dict(name=str(obj.get_name()),pose=pose(obj),qpos=values(obj.get_qpos()),qvel=values(obj.get_qvel()))
        targets=[]
        for joint in obj.get_active_joints():
            entry=dict(name=str(joint.get_name()))
            for name in ("get_drive_target","get_drive_velocity_target"):
                if hasattr(joint,name):entry[name]=values(getattr(joint,name)())
            targets.append(entry)
        r["targets"]=targets;arts.append(r)
    if not actors or not arts:raise SceneMismatch("Cannot identify physical scene/robot")
    cameras=obs["observation"]
    image_hash={k:hashlib.sha256(np.asarray(cameras[k]["rgb"]).tobytes()).hexdigest()
                for k in ("head_camera","left_camera","right_camera")}
    state=dict(actors=sorted(actors,key=canonical_hash),articulations=sorted(arts,key=canonical_hash),
               endpose=obs.get("endpose"),scene_metadata=getattr(env,"info",{}),take_action_cnt=int(env.take_action_cnt),step_lim=int(env.step_lim))
    # endpose contains NumPy arrays; normalize before canonical JSON.
    def native(x):
        if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [native(v) for v in x]
        if isinstance(x,np.ndarray):return values(x)
        if isinstance(x,np.generic):return x.item()
        return x
    return dict(state_hash=canonical_hash(native(state)),first_rgb_hash=image_hash,physical_state=native(state))

def prepare_args(cfg,row,out):
    prepare,probe,instruction,constructor=imports(cfg)
    user=dict(task_name=row["task"],task_config=row["mode"],ckpt_setting="stage3_goal_local",
              eval_video_log=False,instruction_type=cfg["scenes"]["instruction_type"])
    args,_=prepare(user,save_dir=Path(out),robotwin_path=cfg["runtime"]["sim_root"])
    args["eval_video_log"]=False;args["eval_video_save_dir"]=None
    return args,(probe,instruction,constructor)

def reset_scene(cfg,row,args,constructor):
    seed_reset(row["reset_seed"])
    env=constructor(row["task"])
    try:
        env.setup_demo(now_ep_num=0,seed=int(row["reset_seed"]),is_test=True,**args)
        env.set_instruction(instruction=row["instruction"])
        obs=env.get_obs()
        fp=fingerprint(env,obs,cfg["scenes"]["state_round_decimals"])
        if int(env.take_action_cnt)!=0:raise SceneMismatch("Policy takeover count must be zero")
        return env,obs,fp
    except BaseException:
        try:env.close_env()
        except Exception:pass
        raise

def same_initial(row,fp,identity):
    if row["env_config_hash"]!=identity or row["initial_state_hash"]!=fp["state_hash"]:
        raise SceneMismatch("Actual physical initial state differs from manifest")
    if row["first_rgb_hash"]!=fp["first_rgb_hash"]:
        raise SceneMismatch("Camera/randomization differs from manifest")

def probe_scene(cfg,row,out,progress):
    args,(probe,choose,constructor)=prepare_args(cfg,row,out)
    env=constructor(row["task"]);seed_reset(row["reset_seed"])
    progress.report("scene_probe",meaningful=False)
    ok,info,error=probe(env,args,seed=row["reset_seed"],episode_probe_id=0)
    if not ok:
        return dict(status="rejected",row=row,error=error,reason="expert_probe_or_setup_invalid")
    random.seed(row["reset_seed"])
    text=choose(task_name=row["task"],instruction_type=cfg["scenes"]["instruction_type"],
                episode_info=dict(info.get("info",{})),seed=row["reset_seed"])
    if not text:return dict(status="rejected",row=row,reason="empty_instruction")
    result=dict(row,instruction=text)
    first=None
    for repeat in range(2):
        env,obs,fp=reset_scene(cfg,result,args,constructor)
        try:
            if first is not None and (fp["state_hash"]!=first["state_hash"] or fp["first_rgb_hash"]!=first["first_rgb_hash"]):
                raise SceneMismatch("Two fresh resets disagree")
            first=fp
        finally:env.close_env()
        progress.report("scene_reset_verified",meaningful=True,repeat=repeat+1)
    result.update(initial_state_hash=first["state_hash"],first_rgb_hash=first["first_rgb_hash"],
                  env_config_hash=config_identity(cfg,row,args),max_steps=int(first["physical_state"]["step_lim"]),
                  prepared=True,reset_protocol="fresh_setup_seed_all_rng_v1")
    return dict(status="accepted",row=result)

def observation_request(obs,instruction):
    import cv2
    images=[obs["observation"][k]["rgb"] for k in ("head_camera","left_camera","right_camera")]
    images=[cv2.resize(np.asarray(x),(256,256),interpolation=cv2.INTER_AREA) for x in images]
    return dict(instruction=instruction,**{k:array_encode(v) for k,v in zip(("main","left","right"),images)})

def physical_actions(actions,statistics):
    from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient
    if actions.shape!=(36,16) or not np.isfinite(actions).all():
        raise InvalidPolicy("Invalid normalized actions")
    return ModelClient.unnormalize_actions(actions,statistics["agilex"]["action"],
                gripper_indices=(),passthrough_indices=(3,4,5,6,11,12,13,14),invert_indices=(7,15))
