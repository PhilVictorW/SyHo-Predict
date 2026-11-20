import json,math,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.round4.model import GroundedTeacher,FutureActionDataset
ROOT=Path(__file__).resolve().parents[2]
assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.16)
torch.manual_seed(260926)
cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
m=GroundedTeacher(cfg,auxiliary=True)
src=ROOT/'robotwin/round3/runs/teacher_terminal_hold36_20260926/step_000200/adapter.pt'
s=torch.load(src,map_location='cpu',mmap=True,weights_only=False)
params=dict(m.named_parameters())
with torch.no_grad():
    for n,v in s['adapter'].items():params[n].copy_(v)
del s,params
m=m.cuda().train()
ds=FutureActionDataset(cfg['dataset_root'],m.policy.statistics,'train')
i=int(np.random.default_rng(260926).integers(0,len(ds)))
b=default_collate([ds[i]])
with torch.autocast('cuda',dtype=torch.bfloat16):result=m(b)
probes=[m.policy.future.local.query,m.policy.future.goal.query]
a=torch.autograd.grad(result['flow'],probes,retain_graph=True,allow_unused=True)
d=torch.autograd.grad(result['aux'],probes,retain_graph=True,allow_unused=True)
an=math.sqrt(sum(float(g.float().square().sum()) for g in a if g is not None));dn=math.sqrt(sum(float(g.float().square().sum()) for g in d if g is not None))
assert math.isfinite(an) and math.isfinite(dn) and an>0 and dn>0
weight=min(.001/20,an/dn)
loss=result['flow']+weight*result['aux'];loss.backward()
assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.local_head.parameters())
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.goal_head.parameters())
report=dict(passed=True,no_optimizer_updates=True,index=i,losses={k:float(v.detach()) for k,v in result.items()},
 action_query_gradient_norm=an,aux_query_gradient_norm=dn,effective_initial_aux_weight=weight,
 weighted_aux_to_action_ratio=weight*dn/an,backbone_has_no_grad=True,local_and_goal_heads_have_grad=True,
 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
(ROOT/'robotwin/round4/gpu_aux_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
