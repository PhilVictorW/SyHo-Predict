"""Check NCCL averaging/conditional-unused accumulation and disjoint resume sampling."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train_ddp import DistributedBatches

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Linear(3,2,bias=False)
        self.future=torch.nn.Linear(3,2,bias=False)
    def forward(self,x,use_future):
        return self.base(x)+(self.future(x) if use_future else 0)

def main():
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(2); torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('nccl')
    assert world==2
    data=list(range(101))
    batches=DistributedBatches(data,2,world,rank,2026)
    for epoch in [0,1]:
        batches.reset(epoch); full=list(batches)
        for offset in [1,7,24]:
            batches.reset(epoch,offset); assert list(batches)==full[offset:]
        flat=[x for row in full for x in row]
        both=[None,None]; dist.all_gather_object(both,flat)
        assert set(both[0]).isdisjoint(both[1]) and len(set(both[0]+both[1]))==100
    torch.manual_seed(2026)
    model=Toy().to(device); reference=Toy().to(device); reference.load_state_dict(model.state_dict())
    ddp=DDP(model,device_ids=[local],find_unused_parameters=True)
    with ddp.no_sync():
        ddp(torch.ones(1,3,device=device)*(rank+1),rank==0).square().mean().div(2).backward()
    ddp(torch.ones(1,3,device=device)*(rank+3),False).square().mean().div(2).backward()
    for r in range(world):
        reference(torch.ones(1,3,device=device)*(r+1),r==0).square().mean().div(4).backward()
        reference(torch.ones(1,3,device=device)*(r+3),False).square().mean().div(4).backward()
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4); opt.step()
    for p in model.parameters():
        values=[torch.empty_like(p) for _ in range(world)]; dist.all_gather(values,p)
        torch.testing.assert_close(values[0],values[1],atol=0,rtol=0)
    if rank==0:
        report=dict(passed=True,physical_gpus=[1,3],backend='nccl',world_size=world,
            disjoint_sampler=True,consumed_batch_resume=True,conditional_unused_accumulation=True,
            ddp_gradients_equal_global_reference=True,parameters_equal_after_update=True)
        (Path(__file__).resolve().parents[1]/'verification/ddp.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':
    main()
