"""CPU/gloo check: conditional gradients, rank-zero AdamW, broadcast and resume.

Run with torchrun --standalone --nproc_per_node=2 --module robotwin.verify_distributed.
This does not access GPUs, the dataset, or any live training output.
"""
import copy
from contextlib import nullcontext
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from goal_local.optim import MasterAdamW
from .train_distributed import broadcast_parameters


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Linear(3,2)
        self.role=torch.nn.Parameter(torch.randn(2))

    def forward(self,x,target,role):
        y=self.base(x)
        if role:
            y=y+self.role
        loss=((y-target)**2).mean()
        return {'loss':loss,'flow':loss}


def main():
    torch.set_num_threads(1)
    rank=int(os.environ['RANK'])
    dist.init_process_group('gloo',timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        policy=TinyPolicy()
        reference=copy.deepcopy(policy)
        model=DDP(policy,find_unused_parameters=True,gradient_as_bucket_view=True,broadcast_buffers=False)
        parameters=list(policy.parameters())
        optim=MasterAdamW(parameters,lr=.001,weight_decay=1e-8) if rank==0 else None
        ref_optim=MasterAdamW(reference.parameters(),lr=.001,weight_decay=1e-8) if rank==0 else None
        generator=torch.Generator().manual_seed(777)
        inputs=torch.randn(3,4,3,generator=generator)
        targets=torch.randn(3,4,2,generator=generator)
        errors=[]
        for step in range(3):
            policy.zero_grad(set_to_none=True)
            if rank==0:
                optim.zero_grad()
                ref_optim.zero_grad()
                for sample in range(4):
                    (reference(inputs[step,sample],targets[step,sample],sample==0)['loss']/4).backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
                ref_optim.step()
            for micro in range(2):
                sample=rank*2+micro
                with model.no_sync() if micro==0 else nullcontext():
                    (model(inputs[step,sample],targets[step,sample],sample==0)['loss']/2).backward()
            if rank==0:
                torch.nn.utils.clip_grad_norm_(parameters,1.)
                optim.step()
            broadcast_parameters(parameters,limit=16)
            replicas=[torch.empty(sum(p.numel() for p in parameters)) for _ in range(2)]
            dist.all_gather(replicas,torch.cat([p.detach().flatten() for p in parameters]))
            assert torch.equal(replicas[0],replicas[1]),'Parameters diverged across ranks'
            if rank==0:
                error=max(float((p-q).abs().max()) for p,q in zip(parameters,reference.parameters()))
                errors.append(error)
                assert error<2e-7,('DDP versus single-batch optimizer mismatch',step,error)
                if step==1:
                    saved=copy.deepcopy(optim.state_dict())
                    optim=MasterAdamW(parameters,lr=.001,weight_decay=1e-8)
                    optim.load_state_dict(saved)
        if rank==0:
            print('PASS: conditional/no_sync gradients, averaged loss, exact replica broadcast, FP32 AdamW resume; max errors',errors,flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
