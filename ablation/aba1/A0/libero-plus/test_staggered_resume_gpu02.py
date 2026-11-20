"""Small actual two-GPU test of serial FSDP construction and gradient averaging."""
import json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
import torch.distributed as dist
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parents[3]))
from a0_single_latent.distributed import initialize,StaggeredStartup
rank,world,device=initialize()
folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
startup=StaggeredStartup(rank,world)
startup.begin()
begin=time.time()
if rank==1:
    first=json.loads((folder/'rank0.json').read_text())
    assert begin>=first['end']
torch.manual_seed(2026)
model=FSDP(nn.Linear(4,4,bias=False),device_id=device,sync_module_states=False,
           mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
with torch.no_grad():
    for p in model.parameters():p.fill_(0.1)
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
(folder/f'rank{rank}.json').write_text(json.dumps(dict(begin=begin,end=time.time(),gpu=[0,2][rank],pid=__import__('os').getpid())))
startup.finish()
with torch.autocast('cuda',dtype=torch.bfloat16):
    loss=model(torch.ones(1,4,device=device)*(rank+1)).sum()
loss.backward()
for p in model.parameters():torch.testing.assert_close(p.grad,torch.full_like(p.grad,1.5))
optimizer.step()
for p in model.parameters():torch.testing.assert_close(p,torch.full_like(p,-0.05),atol=1e-6,rtol=1e-6)
dist.barrier()
if rank==0:
    result=dict(passed=True,ordered_load=True,fsdp_gradient_average=1.5,gpus=[0,2])
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
dist.destroy_process_group()
