"""Explicit Plus checkpoint adapter; never changes legacy LIBERO clients or training."""
import json
from pathlib import Path
import numpy as np
from goal_local.eval_client import StudentClient
from student_eval_common import TRAIN, ACTION_PROTOCOL, atomic


def validate_checkpoint(path):
    path=Path(path).resolve()
    if path.parent!=TRAIN.resolve():
        raise ValueError('Gripper conversion is restricted to this LIBERO-Plus Stage-II run')
    marker=json.loads((path/'checkpoint_complete.json').read_text())
    step=marker.get('step',-1)
    if (marker.get('stage')!='II' or marker.get('mode')!='student' or
        not isinstance(step,int) or step<500 or step>10000 or step%500 or path.name!=f'step_{step:06d}'):
        raise ValueError('Expected a completed Plus student checkpoint at a 500-step interval')
    return path


def ensure_checkpoint_protocol(path):
    path=validate_checkpoint(path);sidecar=path/'action_semantics.json'
    if sidecar.exists():
        if json.loads(sidecar.read_text())!=ACTION_PROTOCOL:
            raise ValueError(f'Conflicting checkpoint action semantics: {sidecar}')
    else:
        atomic(sidecar,ACTION_PROTOCOL)
    return ACTION_PROTOCOL


class PlusStudentClient(StudentClient):
    def __init__(self,checkpoint,url,*,action_protocol):
        path=validate_checkpoint(checkpoint)
        if action_protocol!=ACTION_PROTOCOL or json.loads((path/'action_semantics.json').read_text())!=ACTION_PROTOCOL:
            raise ValueError('Explicit matching Plus action semantics are required')
        super().__init__(str(path),url)

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        raw=result['raw_action'];gripper=np.asarray(raw['open_gripper'])
        if gripper.shape!=(1,) or not np.isin(gripper,[0.,1.]).all():
            raise ValueError('Expected one binarized Plus gripper prediction')
        # Legacy core maps its misleadingly named open_gripper to 2*g-1:
        # it expects 1=closed, while these Plus weights learned 1=open.
        # Replace the returned copy; never invert the cached action chunk in place.
        raw['open_gripper']=1.0-gripper
        return result
