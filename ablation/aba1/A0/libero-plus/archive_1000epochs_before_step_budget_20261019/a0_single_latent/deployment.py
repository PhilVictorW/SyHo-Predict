"""Strict A0 architecture boundary and one-time Plus gripper conversion."""
import json
from pathlib import Path
import numpy as np
from goal_local.eval_common import model_metadata
from goal_local.eval_client import StudentClient
from .config import FORMAT,WORK,ACTION_PROTOCOL,file_sha256

def validate_checkpoint(path,cfg=None,verify_hash=False):
    path=Path(path).resolve()
    if not path.is_relative_to(WORK/'runs'):raise ValueError('Wrong experiment directory')
    meta=model_metadata(path/'model.pt');marker=json.loads((path/'checkpoint_complete.json').read_text())
    if meta['format']!=FORMAT or marker.get('format')!=FORMAT or not marker.get('complete'):
        raise ValueError('Incomplete or incompatible A0 checkpoint')
    if cfg is not None and meta['config']!=cfg:raise ValueError('Deployment config mismatch')
    if meta['step']<=0 or marker['step']!=meta['step'] or meta['mode']!='student':raise ValueError('Untrained checkpoint')
    if meta['extra'].get('action_protocol')!=ACTION_PROTOCOL or json.loads((path/'action_semantics.json').read_text())!=ACTION_PROTOCOL:
        raise ValueError('Action semantics mismatch')
    if verify_hash and file_sha256(path/'model.pt')!=marker['model_sha256']:raise ValueError('Model hash mismatch')
    marker=dict(marker,cumulative_step=meta['step'],files=dict(marker['files']))
    marker['files']['model.pt']=dict(sha256=marker['model_sha256'])
    return meta,marker

class PlusStudentClient(StudentClient):
    def __init__(self,checkpoint,url,action_protocol):
        validate_checkpoint(checkpoint)
        if action_protocol!=ACTION_PROTOCOL:raise ValueError('Wrong action protocol')
        super().__init__(checkpoint,url)
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs);raw=result['raw_action'];g=np.asarray(raw['open_gripper'])
        if g.shape!=(1,) or not np.isin(g,[0.,1.]).all():raise ValueError('Expected binary model gripper')
        raw['open_gripper']=1.-g
        return result
