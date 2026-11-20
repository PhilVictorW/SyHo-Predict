import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from robotwin.round4.model import GroundedTeacher,FutureActionDataset
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
heads=SimpleNamespace(local_head=nn.Linear(2048,16),goal_head=nn.Linear(2048,80))
with torch.no_grad():
    for head,n in [(heads.local_head,1),(heads.goal_head,5)]:
        head.weight.zero_();head.bias.zero_();head.bias.view(n,16)[:,[3,11]]=1.
encoded={k:torch.zeros(2,8,2048) for k in ['goal','local']}
target=torch.zeros(2,6,16);target[:,:,[3,11]]=1.
b={'future_action_target':target,'future_action_valid':torch.ones(2,6,dtype=torch.bool)}
x=GroundedTeacher.auxiliary_loss(heads,b,encoded)
assert float(x['aux'])==0
b['future_action_target']=target.clone();b['future_action_target'][:,:,[3,11]]=-1
assert float(GroundedTeacher.auxiliary_loss(heads,b,encoded)['aux'])==0
b['future_action_valid'].zero_();b['future_action_target'].fill_(12.)
z=GroundedTeacher.auxiliary_loss(heads,b,encoded)['aux'];assert torch.isfinite(z) and float(z)==0
z.backward();assert all(p.grad is not None and not p.grad.any() for h in [heads.local_head,heads.goal_head] for p in h.parameters())
cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text());stats=json.loads(Path(cfg['statistics_path']).read_text())
ds=FutureActionDataset(cfg['dataset_root'],stats,'train',decode=False)
checks=0
for e in ds.selected[:3]:
    for t in [0,e['length']-1,max(0,e['length']-30)]:
        b=ds.sample(e,t)
        for j,offset in enumerate([8,24,40,56,72,88]):
            if t+offset<e['length']:
                expected=ds.normalizer.normalize(torch.from_numpy(ds.actions[e['dataset_from_index']+t+offset].copy()))
                assert torch.equal(expected,b['future_action_target'][j])
            else:assert not b['future_action_valid'][j] and not b['future_action_target'][j].any()
        assert int(b['actions_mask'].sum())==36*16
        checks+=1
out=dict(passed=True,quaternion_sign_invariant=True,all_invalid_loss_and_gradients_zero=True,
         future_labels_match_episode_local_actions=True,dataset_boundary_cases=checks)
(Path(__file__).resolve().parent/'cpu_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
