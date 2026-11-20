"""Resolve BF16 Qwen parity and separate Plan insertion from horizon changes."""
import json,time
import torch
from torch.utils.data import default_collate
from audit import ROOT,OUT,RUN,qwen_reference,sample,difference
from robotwin.model import RobotwinPolicy
from robotwin.data import FutureDataset

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.16);torch.manual_seed(2026)
    cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
    ds=FutureDataset(cfg['dataset_root'],json.loads(__import__('pathlib').Path(cfg['statistics_path']).read_text()),'val')
    ids=json.loads((RUN/'validation_samples.json').read_text())['indices'][:4]
    report={'started_at':time.time(),'scope':'offline numerical parity and Plan/horizon controls; no optimizer','rows':[]}
    def save():
        p=OUT/'precision_plan_controls.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(OUT/'precision_plan_controls.json')
    print('Loading baseline',flush=True)
    model=RobotwinPolicy.from_baseline(cfg).cuda().eval()
    model.qwen.float()
    b=default_collate([ds[int(ids[0])]])
    report['qwen_fp32']=qwen_reference(model,b)[0];save();print('QWEN_FP32',report['qwen_fp32'],flush=True)
    model.qwen.to(torch.bfloat16);torch.cuda.empty_cache()
    for i,idx in enumerate(ids):
        b=default_collate([ds[int(idx)]])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            enc=model.encode(b);cond=model.action_conditions(b,enc)
            torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
            pred={}
            for name,h,variant in [('native8',8,'native'),('native36',36,'native'),('plan8',8,'roles_off'),('plan36',36,'roles_off'),('full36',36,'full')]:
                pred[name]=sample(model.flow,cond,noise,h,variant)[:,:8]
            own=sample(model.flow,cond,noise,8,'full')
            hz=b['action_hz'].cuda();emb=b['embodiment_id'].cuda()
            actual=model.flow.sample(cond,hz,emb,noise=noise)
            bf16_noise=noise.to(torch.bfloat16)
            accum32=sample(model.flow,cond,bf16_noise.float(),8,'full')
            accum16=model.flow.sample(cond,hz,emb,noise=bf16_noise)
            target=b['actions'][:,:8,:16].cuda();mask=b['actions_mask'][:,:8,:16].cuda()
            mse=lambda y:float(((y-target).square()*mask).sum()/mask.sum())
            row=dict(sample=i,action_mse={k:mse(y) for k,y in pred.items()},plan_at36=difference(pred['plan36'],pred['native36']),
                actual_sampler_vs_control=difference(actual,own),bf16_vs_fp32_accumulation=difference(accum16,accum32))
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
    report['completed']=True;report['finished_at']=time.time();report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;save()

if __name__=='__main__':main()
