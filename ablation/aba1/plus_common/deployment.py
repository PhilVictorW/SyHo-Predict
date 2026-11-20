"""Strict ablation checkpoint boundary and the one-time Plus gripper conversion."""
import json
from pathlib import Path
import numpy as np
from .config import ACTION_PROTOCOL,checkpoint_path
from .preflight import check_complete
from goal_local.eval_client import StudentClient

def validate_checkpoint(path,cfg=None,verify_hash=False):
    path=Path(path).resolve()
    meta,marker=check_complete(path,verify_hash=verify_hash)
    actual=meta["config"]
    if cfg is not None and cfg!=actual:raise ValueError("Deployment config mismatch")
    stage=meta["extra"]["stage"]
    if (meta["mode"]!="student" or meta["step"]<=0 or stage not in ("action1","action2","II")
        or path!=checkpoint_path(actual,stage,meta["step"])):
        raise ValueError("Expected a completed trained ablation policy in its own Plus stage directory")
    if meta["extra"].get("action_protocol")!=ACTION_PROTOCOL:
        raise ValueError("Model action protocol mismatch")
    if json.loads((path/"action_semantics.json").read_text())!=ACTION_PROTOCOL:
        raise ValueError("Checkpoint action semantics mismatch")
    return meta,marker

class PlusStudentClient(StudentClient):
    def __init__(self,checkpoint,url,action_protocol):
        validate_checkpoint(checkpoint)
        if action_protocol!=ACTION_PROTOCOL:raise ValueError("Evaluation action protocol mismatch")
        super().__init__(checkpoint,url)
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        raw=result["raw_action"];g=np.asarray(raw["open_gripper"])
        if g.shape!=(1,) or not np.isin(g,[0.,1.]).all():raise ValueError("Expected binary model gripper")
        raw["open_gripper"]=1.-g
        return result
