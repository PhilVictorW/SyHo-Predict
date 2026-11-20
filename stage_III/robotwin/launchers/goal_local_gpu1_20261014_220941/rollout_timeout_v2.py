"""Use the unchanged trajectory implementation with the explicit 40-minute RPC policy."""
import argparse
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from stage_III.robotwin.common import load_config,configure_gpu,require_execute,local_output,read_json,atomic_json
from stage_III.robotwin.runtime import Progress
from stage_III.robotwin.rpc import ModelError
from timeout_runtime import POLICY,is_timeout

def main():
    p=argparse.ArgumentParser(__doc__)
    for arg in ('config','row','request','output','gpu'):p.add_argument('--'+arg,required=True)
    p.add_argument('--group',type=int,required=True);p.add_argument('--trajectories',type=int,default=4)
    p.add_argument('--execute',action='store_true');a=p.parse_args();require_execute(a)
    configure_gpu(a.gpu);cfg=load_config(a.config)
    # An operational override only; checkpoint/model/config identities stay immutable.
    cfg['runtime']['stall_seconds']=POLICY['rpc_timeout_seconds']
    out=local_output(a.output);out.mkdir(parents=True,exist_ok=True)
    progress=Progress(out/'progress.json');progress.report('initializing',meaningful=False)
    try:
        from stage_III.robotwin.rollout import trajectory
        row=read_json(a.row);request=read_json(a.request)
        if not row.get('prepared'):raise ValueError('Scene has not passed reset checks')
        rows=[]
        for index in range(a.trajectories):
            value=trajectory(cfg,row,a.group,index,request,out,progress)
            atomic_json(out/f'trajectory_{index:02d}.json',value);rows.append(value)
        atomic_json(out/'result.json',dict(status='completed',trajectories=rows))
        progress.report('completed',meaningful=True)
    except BaseException as exc:
        from stage_III.robotwin.env_adapter import InvalidPolicy,SceneMismatch
        kind='sampling_timeout' if is_timeout(exc) else 'model_error' if isinstance(exc,ModelError) else (
            'invalid_policy' if isinstance(exc,InvalidPolicy) else 'scene_mismatch' if isinstance(exc,SceneMismatch) else 'runtime_error')
        atomic_json(out/'failure.json',dict(kind=kind,error=repr(exc),time=time.time()))
        raise

if __name__=='__main__':main()
