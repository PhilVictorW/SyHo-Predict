"""Fixed real anchors, isolated GPU6 only. Never modifies live policies or runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='6'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,time,gc,copy
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; WORK=ROOT/'libero-plus'
sys.path[:0]=[str(ROOT),str(WORK/'runtime')]
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(.34,0)
from accelerate import init_empty_weights
from torch.utils.data import default_collate
from goal_local.model import GoalLocalPolicy
from indexed_future import IndexedFutureDataset
cfg=json.loads((WORK/'configs/stage_I.json').read_text())
stats=json.loads(Path(cfg['statistics_path']).read_text())
ds=IndexedFutureDataset(cfg['dataset_root'],stats,'val',cfg['seed'],cfg['val_ratio'])
groups=defaultdict(list)
for e in ds.selected:
    bits=Path(e['origin']).parts
    if e['length']>=100:groups[(bits[-3],bits[-2])].append(e)
rng=np.random.default_rng(20260925); anchors=[];batches=[]
for group,episodes in sorted(groups.items()):
    e=episodes[int(rng.integers(len(episodes)))];t=min(int(e['length']*.35),e['length']-89)
    anchors.append(dict(group=group,episode=e['episode_index'],t=t,length=e['length'],instruction=e['tasks'][0],origin=e['origin']))
    batches.append(default_collate([ds.sample(e,t)]))
report=dict(started=time.time(),physical_gpu=6,anchors=anchors,models={},sample_count=len(batches),online_rollouts=False)
def save():
    p=HERE/'offline_ablation.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2));q.replace(p)
def log(k,v):print(k,json.dumps(v),flush=True)
save();log('anchors',dict(n=len(anchors),groups=list(groups)))

def load(path):
    with init_empty_weights(include_buffers=False):m=GoalLocalPolicy.load(path)
    p=torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=False)
    m.load_state_dict(p['state_dict'],strict=True,assign=True);del p
    assert not any(p.is_meta for p in m.parameters()) and not any(p.is_meta for p in m.buffers())
    return m.to('cuda').eval().requires_grad_(False)
def autocast(enabled=True):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled)
def cp(x):return x.detach().float().cpu() if x.is_floating_point() else x.detach().cpu()
def gpu(x):return x.to('cuda')
def condition(flow,enc,dino,valid):
    return flow.conditions(gpu(enc['current']),gpu(enc['current_valid']),gpu(dino),gpu(enc['goal']),gpu(enc['local']),gpu(valid))
hz=torch.tensor([20.],device='cuda');emb=torch.tensor([25],device='cuda');valid=torch.arange(50,device='cuda')[None]<8
predictions={};teacher_cache=[];baseline_cache=[]
teacher_flow=None;baseline_flow=None

@torch.inference_mode()
def evaluate(flow,cond,batch,index,bypass=False,bf16=False):
    gen=torch.Generator(device='cuda').manual_seed(9000+index)
    noise=torch.randn((1,50,32),device='cuda',generator=gen,dtype=torch.float32)
    target=batch['actions'].to('cuda'); mask=batch['actions_mask'].to('cuda')
    losses=[]
    with autocast(bf16):
        for tau in (.1,.4,.7):
            x=((1-tau)*noise+tau*target)*valid[...,None]
            v=flow.velocity(x,torch.tensor([int(tau*1000)],device='cuda'),hz,emb,valid,cond,bypass_roles=bypass)
            losses.append(float((((v.float()-(target-noise))**2)*mask).sum()/mask.sum()))
        x=(noise.to(torch.bfloat16) if bf16 else noise.clone())*valid[...,None]
        for j in range(10):
            v=flow.velocity(x,torch.tensor([j*100],device='cuda'),hz,emb,valid,cond,bypass_roles=bypass)
            x=(x+v/10)*valid[...,None]
    act=x[:,:8,:7].float();truth=target[:,:8,:7]
    return dict(flow_mse=float(np.mean(losses)),flow_by_tau=losses,
        action_mse=float((act-truth).square().mean()),xyz_mse=float((act[...,:3]-truth[...,:3]).square().mean()),
        rot_mse=float((act[...,3:6]-truth[...,3:6]).square().mean()),
        gripper_accuracy=float(((act[...,6]>.5)==(truth[...,6]>.5)).float().mean()),
        saturation_fraction=float((act[...,:6].abs()>1).float().mean()),
        gripper_exact_half=float((act[...,6]==.5).float().mean()),actions=act[0].cpu().tolist())

def put(label,rows):
    keys=['flow_mse','action_mse','xyz_mse','rot_mse','gripper_accuracy','saturation_fraction','gripper_exact_half']
    report['models'][label]=dict(mean={k:float(np.mean([r[k] for r in rows])) for k in keys},per_anchor=rows)
    save();log(label,report['models'][label]['mean'])

for label,path in [('baseline',None),('teacher5000',WORK/'runs/stage_I_20260923/retained_checkpoints/step_005000'),('student0',WORK/'runs/stage_II_20260925/step_000000'),('student500',WORK/'runs/stage_II_20260925/step_000500')]:
    torch.manual_seed(2026);start=time.time()
    model=(GoalLocalPolicy.from_baseline(cfg).to('cuda').eval().requires_grad_(False) if path is None else load(path))
    model.flow.float();log('loaded',dict(label=label,seconds=time.time()-start,memory_gib=torch.cuda.memory_allocated()/2**30))
    variants=defaultdict(list)
    for i,batch in enumerate(batches):
        with torch.inference_mode(),autocast():
            if label=='baseline':
                cur=model.current_embeddings(model.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction']))
                encoded=model.language(cur,torch.zeros(1,8,2048,device='cuda'),torch.zeros(1,8,2048,device='cuda'))
            else:encoded=model.encode(batch)
            dino=model.current_dino(batch['current_main'])
        enc={k:cp(encoded[k]) for k in ('current','current_valid','goal','local')};dn=cp(dino)
        role_valid=torch.stack((batch['goal_frame_valid'].any(-1),batch['local_valid']),-1)
        with torch.inference_mode():
            cond=condition(model.flow,enc,dn,role_valid)
            if label=='baseline':
                baseline_cache.append((enc,dn))
                variants['baseline_fp32'].append(evaluate(model.flow,cond,batch,i,bypass=True))
                variants['baseline_plus_random_plans'].append(evaluate(model.flow,tuple(list(cond[:-1])+[torch.zeros_like(cond[-1])]),batch,i))
            else:
                variants[label+'_fp32'].append(evaluate(model.flow,cond,batch,i))
                if label=='teacher5000':
                    teacher_cache.append((enc,dn))
                    variants['teacher5000_no_roles'].append(evaluate(model.flow,tuple(list(cond[:-1])+[torch.zeros_like(cond[-1])]),batch,i))
                    variants['teacher5000_bypass'].append(evaluate(model.flow,cond,batch,i,bypass=True))
                if label=='student500':
                    with autocast():bcond=condition(model.flow,enc,dn,role_valid)
                    variants['student500_bf16_flow'].append(evaluate(model.flow,bcond,batch,i,bf16=True))
                    variants['student500_bypass'].append(evaluate(model.flow,cond,batch,i,bypass=True))
                    te,td=teacher_cache[i]
                    mixed=dict(enc,goal=te['goal'],local=te['local'])
                    variants['student500_teacher_roles'].append(evaluate(model.flow,condition(model.flow,mixed,dn,role_valid),batch,i))
                    mixed=dict(te,goal=enc['goal'],local=enc['local'])
                    variants['teacher5000_student_roles'].append(evaluate(teacher_flow,condition(teacher_flow,mixed,td,role_valid),batch,i))
                    variants['baseline_flow_student_current'].append(evaluate(baseline_flow,condition(baseline_flow,enc,dn,role_valid),batch,i,bypass=True))
                    be,bd=baseline_cache[i]
                    mixed=dict(enc,current=be['current'],current_valid=be['current_valid'])
                    variants['student500_baseline_current'].append(evaluate(model.flow,condition(model.flow,mixed,bd,role_valid),batch,i))
        del encoded,cond
        log('anchor_done',dict(model=label,index=i+1,total=len(batches)))
    for k,v in variants.items():put(k,v)
    if label=='baseline':baseline_flow=model.flow;model.flow=None
    if label=='teacher5000':teacher_flow=model.flow;model.flow=None
    del model;gc.collect();torch.cuda.empty_cache()
report['finished']=dict(time=time.time(),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
save();log('finished',report['finished'])
