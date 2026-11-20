"""Read-only bounded numerical probe; set CUDA_VISIBLE_DEVICES externally."""
import json,os
import torch
from torch.nn import functional as F
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.015)
torch.use_deterministic_algorithms(True);torch.manual_seed(123)
a=torch.randn(1,280,2048,device="cuda",dtype=torch.bfloat16)
w=torch.randn(2048,2048,device="cuda",dtype=torch.bfloat16)
def repeat(a,w):
    y=F.linear(a,w);result=[]
    for _ in range(6):
        d=(F.linear(a,w).float()-y.float()).abs()
        result.append(dict(max_abs=float(d.max()),changed_elements=int((d!=0).sum()),mean_abs=float(d.mean())))
    return result
report=dict(physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,cuda=torch.version.cuda)
report["bf16"]=repeat(a,w)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
report["bf16_without_reduced_reduction"]=repeat(a,w)
report["fp32"]=repeat(a.float(),w.float())
print(json.dumps(report,indent=2))
