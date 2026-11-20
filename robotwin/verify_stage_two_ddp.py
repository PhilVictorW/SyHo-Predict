"""Inject one nonfinite rank, verify synchronized skip and subsequent recovery."""
import copy,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from goal_local.optim import MasterAdamW
from .train_distributed import broadcast_parameters
from .stage_two_utils import finite_clip

def main():
    rank=int(os.environ['RANK']);torch.set_num_threads(1);dist.init_process_group('gloo')
    try:
        torch.manual_seed(2);net=torch.nn.Linear(2,1);model=DDP(net)
        params=list(net.parameters());optim=MasterAdamW(params,.001,0.) if rank==0 else None
        skipped=0;updates=0
        for attempt in range(3):
            net.zero_grad(set_to_none=True)
            if optim:optim.zero_grad()
            previous=[p.detach().clone() for p in params]
            x=torch.ones(1,2)
            if attempt==1 and rank==1:x[0,0]=float('inf')
            loss=model(x).square().mean();loss.backward()
            norm=finite_clip(params,1.) if rank==0 else torch.tensor(0.)
            finite=torch.tensor(int(torch.isfinite(loss) and torch.isfinite(norm)))
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if finite:
                if optim:optim.step()
                broadcast_parameters(params,limit=16);updates+=1
            else:
                skipped+=1
                assert all(torch.equal(p,q) for p,q in zip(params,previous))
                net.zero_grad(set_to_none=True)
            flat=torch.cat([p.detach().flatten() for p in params]);replicas=[torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(replicas,flat);assert torch.equal(*replicas)
            assert torch.isfinite(flat).all()
        assert updates==2 and skipped==1
        if rank==0:
            report=dict(two_rank_nonfinite_injection=True,synchronized_skip=True,weights_unchanged_on_skip=True,
                subsequent_update_succeeded=True,replicas_identical=True,updates=updates,skipped=skipped)
            (Path(__file__).resolve().parent/'verification/stage_II_ddp_controls.json').write_text(json.dumps(report,indent=2));print(report)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
