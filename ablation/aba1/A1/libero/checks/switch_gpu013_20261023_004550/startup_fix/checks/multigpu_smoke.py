"""Exercise the configured GPU mapping and staggered loading and CPU Gloo optimizer on tiny CUDA tensors."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
w=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(w),str(w.parents[3])]
import torch
import torch.distributed as dist
from a1_local.config import load_plan, rank_device, physical_gpus
from a1_local.sampling import backward_weight
from a1_local.startup import load_slot, wait_for_loaded, read
from ablation.aba1.plus_common.distributed import DistributedMaster
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
plan,_=load_plan()
rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==plan['world_size'] and os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,physical_gpus(plan)))
logical,physical=rank_device(plan,rank)
torch.set_num_threads(1);torch.cuda.set_device(logical)
startup_plan = dict(plan, load_gap_seconds=.1, loading_timeout_seconds=60)
startup = args.output.parent/'smoke_initialization'
with load_slot(startup_plan,startup,rank,'I'):
    value=torch.nn.Parameter(torch.tensor([.2,-.1],device=f'cuda:{logical}'))
wait_for_loaded(startup_plan,startup)
dist.init_process_group('gloo',timeout=timedelta(minutes=5))
try:
    opt=DistributedMaster([value],.01,.0001,bucket_elements=1);opt.broadcast_weights()
    reference=value.detach().clone().requires_grad_(True)
    ref_opt=torch.optim.AdamW([reference],lr=.01,weight_decay=.0001,betas=(.9,.95),eps=1e-8,foreach=False)
    x=torch.arange(64,dtype=torch.float32,device=value.device).reshape(32,2)/64
    y=torch.linspace(-.3,.4,32,device=value.device)
    norms=[]
    for step in range(3):
        opt.zero_grad();ref_opt.zero_grad()
        for i in range(rank,32,world):
            ((value@x[i]-y[i]).square()*backward_weight(world,32)).backward()
        norms.append(opt.step(.01,1.))
        ((x@reference-y).square().mean()).backward()
        torch.nn.utils.clip_grad_norm_([reference],1.,foreach=False);ref_opt.step()
        torch.testing.assert_close(value,reference,atol=1e-7,rtol=1e-6)
    row=dict(rank=rank,pid=os.getpid(),logical_gpu=logical,physical_gpu=physical,steps=3,norms=norms,final=value.detach().cpu().tolist())
    gathered=[None]*world;dist.all_gather_object(gathered,row)
    if rank==0:
        loads=[read(startup/f'initialization_{i}.json') for i in range(world)]
        assert all(loads[i]['load_started_at'] >= loads[i-1]['load_finished_at']+.1 for i in range(1,world))
        result=dict(passed=True,workers=gathered,equivalent_to_full_batch_adamw=True,serial_loading_verified=True,loading_records=loads)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:
    dist.destroy_process_group()
