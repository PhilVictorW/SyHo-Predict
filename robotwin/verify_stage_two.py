"""Real-data student initialization, future isolation and backward preflight."""
import json,os,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .model import RobotwinPolicy
from .data import FutureDataset
from goal_local.model import make_stage_two
from .stage_two_utils import finite_clip

def main():
    torch.set_num_threads(4);torch.manual_seed(2026)
    torch.cuda.set_per_process_memory_fraction(.36)
    root=Path(__file__).resolve().parents[1];source=root/'robotwin/runs/stage_I_20260924/step_001000'
    teacher=RobotwinPolicy.load(source);system=make_stage_two(teacher).to('cuda');system.train()
    assert isinstance(system.student,RobotwinPolicy) and system.student.mode=='student'
    assert system.teacher.flow is None and not system.teacher.training
    tp=dict(teacher.named_parameters());versions={k:p._version for k,p in tp.items()}
    sp=list(system.student.parameters());assert not ({p.data_ptr() for p in tp.values()} & {p.data_ptr() for p in sp})
    ds=FutureDataset(teacher.cfg['dataset_root'],teacher.statistics,'train',2026,.05)
    batch=next(iter(DataLoader(ds,batch_size=1,num_workers=0)))
    with torch.autocast('cuda',dtype=torch.bfloat16):result=system(batch)
    result['loss'].backward();parameters=[p for p in sp if p.requires_grad];norm=finite_clip(parameters,1.)
    assert torch.isfinite(norm) and all(torch.isfinite(x) for x in result.values())
    assert all(p.grad is None and not p.requires_grad and p._version==versions[k] for k,p in tp.items())
    assert system.student.q_goal.grad is not None and system.student.q_local.grad is not None
    gate_grad=system.student.gate.proj_g.weight.grad
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    system.zero_grad(set_to_none=True);system.eval()
    # Online student must run with no future fields whatsoever.
    online={k:batch[k] for k in ('current_main','current_wrist','instruction')}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=system.student.encode(online)
        cond=system.student.action_conditions(online,encoded)
        actions=system.student.flow.sample(cond,torch.tensor([30.],device='cuda'),torch.tensor([1],device='cuda'))
    assert actions.shape==(1,8,16) and torch.isfinite(actions).all()
    report=dict(source=str(source),teacher_step=1000,student_type=type(system.student).__name__,
        losses={k:float(v) for k,v in result.items()},grad_norm=float(norm),teacher_frozen=True,independent_storage=True,
        student_future_free=True,gate_gradient_finite=True,action_shape=list(actions.shape),
        roles={str(k):[list(v[0].shape),list(v[1].shape)] for k,v in encoded['layers'].items()},
        peak_gib=torch.cuda.max_memory_allocated()/2**30,time=time.time(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    path=root/'robotwin/verification/stage_II_preflight.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
