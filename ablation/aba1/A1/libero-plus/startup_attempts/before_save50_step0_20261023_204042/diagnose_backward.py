"""Reproduce initial A1 backward on real data; never updates or saves weights."""
import os,sys,json,random,argparse,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[3]
sys.path.insert(0,str(ROOT))
from a1_local.config_runtime import load_config
from ablation.aba1.plus_common.environment import subprocess_environment
cfg=load_config(WORK/"config.json","A1")
os.environ.update(subprocess_environment(cfg,"4",WORK/"diagnostics_cache"))
os.environ["TMPDIR"]="/tmp"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch,numpy as np
from torch.utils.data import DataLoader
from a1_local.training import DistributedBatches
from ablation.aba1.plus_common.model import GoalLocalPolicy
from ablation.aba1.plus_common.data import IndexedFutureDataset
p=argparse.ArgumentParser();p.add_argument("--rank",type=int,default=0);p.add_argument("--micros",type=int,default=6)
p.add_argument("--anomaly",action="store_true");p.add_argument("--stable-attention",action="store_true")
a=p.parse_args()
torch.set_num_threads(2)
if a.stable_attention:
    from a1_local.numerics import configure_attention_backends
    print(configure_attention_backends(),flush=True)
seed=cfg["train_seed"]
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
print("Loading diagnostic teacher",flush=True)
model=GoalLocalPolicy.from_baseline(cfg).to("cuda:0").train()
print("Teacher loaded",flush=True)
train=IndexedFutureDataset(cfg["dataset_root"],model.statistics,"train",cfg["split_seed"],local=True,goal=False)
batches=DistributedBatches(train,1,6,a.rank,seed);batches.reset(0,0)
loader=DataLoader(train,batch_sampler=batches,num_workers=0,generator=torch.Generator().manual_seed(seed+a.rank))
random.seed(seed+a.rank);np.random.seed(seed+a.rank);torch.manual_seed(seed+a.rank)
bad=[];max_grad={}
def hook(name):
    def inspect(param):
        grad=param.grad.detach().float().cpu()
        if not torch.isfinite(grad).all():bad.append(name)
        max_grad[name]=float(grad.abs().max())
        param.grad=None
    return inspect
handles=[p.register_post_accumulate_grad_hook(hook(n)) for n,p in model.named_parameters() if p.requires_grad]
if a.anomaly:torch.autograd.set_detect_anomaly(True)
for i,batch in enumerate(loader):
    if i>=a.micros:break
    begin=time.monotonic()
    with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
        losses=model(batch);loss=losses["loss"]*(6/32)
    print(json.dumps(dict(rank=a.rank,micro=i+1,loss=float(loss),phase="forward")),flush=True)
    loss.backward()
    result=dict(rank=a.rank,micro=i+1,loss=float(loss),bad=bad,largest=sorted(max_grad.items(),key=lambda x:x[1],reverse=True)[:12],seconds=time.monotonic()-begin)
    print(json.dumps(result),flush=True)
    if bad:raise FloatingPointError(str(bad[:20]))
    bad.clear();max_grad.clear()
print("All diagnostic gradients finite",flush=True)
