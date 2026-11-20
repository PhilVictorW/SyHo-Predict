"""Read-only model experiments; isolated from production training and eval.

Run as a script from LaWAM_aba2_change with PYTHONPATH set to that root.
No optimizer, process signals, online rollout or production-code edits.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from robotwin.data import FutureDataset
from robotwin.model import RobotwinPolicy
from goal_local.roles import remove_old_queries
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'robotwin/runs/stage_II_20260925'
REPORT = {'started_at':time.time(), 'pid':os.getpid(), 'scope':'offline, fixed inputs/noise; not online success rates', 'experiments':{}}

def save():
    p=OUT/'audit_results.tmp'
    p.write_text(json.dumps(REPORT,indent=2)+'\n');p.replace(OUT/'audit_results.json')

def emit(name, value):
    REPORT['experiments'][name]=value;save()
    print(name, json.dumps(value), flush=True)

def difference(a,b):
    d=a.float()-b.float()
    return dict(max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),
                relative_l2=float(d.norm()/b.float().norm().clamp_min(1e-12)))

def no_roles(cond):
    return (*cond[:-1],torch.zeros_like(cond[-1]))

def native_velocity(flow,x,t,hz,emb,valid,cond):
    base,im,vm,*_=cond
    grid,_,_=build_time_grid(flow.config.horizon_sec,hz,x.shape[1])
    a=flow.action_encoder(x,emb)*valid[...,None]
    pos=flow.time_encoder(grid).to(a.dtype)*valid[...,None]
    h=flow.DiT(hidden_states=a,encoder_hidden_states=base,timestep=t,
               hidden_attention_mask=valid,encoder_attention_mask=im|vm,
               image_mask=im,vlm_mask=vm,hidden_positional_embeddings=pos)
    return flow.action_decoder(h,emb)*valid[...,None]

def sample(flow,cond,noise,horizon=8,variant='full'):
    old=flow.config.horizon_sec;flow.config.horizon_sec=horizon/30
    try:
        hz=torch.full((noise.shape[0],),30.,device=noise.device)
        emb=torch.ones(noise.shape[0],dtype=torch.long,device=noise.device)
        _,valid,_=build_time_grid(horizon/30,hz,50)
        x=noise.clone()*valid[...,None]
        for i in range(10):
            t=torch.full((len(x),),i*100,dtype=torch.long,device=x.device)
            if variant=='native': v=native_velocity(flow,x,t,hz,emb,valid,cond)
            else:v=flow.velocity(x,t,hz,emb,valid,no_roles(cond) if variant=='roles_off' else cond,bypass_roles=variant=='no_plan')
            x=(x+v/10)*valid[...,None]
        return x[:,:horizon,:16].float()
    finally:flow.config.horizon_sec=old

def fm_loss(model,batch,cond,seed=783):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    return float(model.flow.loss(batch['actions'].cuda(),batch['actions_mask'].cuda(),
        batch['action_hz'].cuda(),batch['embodiment_id'].cuda(),batch['remaining_actions'].cuda(),cond,repeats=2))

def slot_stats(x):
    x=F.normalize(x.float(),dim=-1)
    within=x@x.transpose(-1,-2);mask=~torch.eye(8,dtype=torch.bool,device=x.device)
    flat=F.normalize(x.flatten(1),dim=-1);across=flat@flat.T
    amask=~torch.eye(len(x),dtype=torch.bool,device=x.device)
    return dict(within_sample_offdiagonal_cosine=float(within[:,mask].mean()),
                across_sample_cosine=float(across[amask].mean()) if len(x)>1 else None)

@torch.no_grad()
def qwen_reference(model,batch):
    inputs=model.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction'])
    cur=model.current_embeddings(inputs)
    ids=inputs['input_ids'];hidden=model.qwen.get_input_embeddings()(ids).clone()
    hidden[cur[2]]=model.act_query[None].expand(len(ids),-1,-1).reshape(-1,2048).to(hidden.dtype)
    hidden[cur[3]]=model.flow_action_query[None].expand(len(ids),-1,-1).reshape(-1,2048).to(hidden.dtype)
    model.qwen.rope_deltas=None
    ref=model.qwen(inputs_embeds=hidden,attention_mask=inputs['attention_mask'],
        pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw'],use_cache=False).last_hidden_state
    roles=torch.zeros(len(ids),8,2048,device=model.device,dtype=hidden.dtype)
    out=model.language(cur,roles,roles)
    expected,_=remove_old_queries(ref,cur[1],cur[2],cur[3])
    perturb=model.language(cur,roles+torch.randn_like(roles)*5,roles+torch.randn_like(roles)*5)
    valid=out['current_valid']
    return dict(current_matches_native=difference(out['current'][valid],expected[valid]),
                current_future_isolation=difference(out['current'][valid],perturb['current'][valid]),
                input_lengths=inputs['attention_mask'].sum(-1).tolist()),out,ref,cur

def release(model):
    del model;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.16)
    torch.manual_seed(2026);np.random.seed(2026)
    cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
    stats=json.loads(Path(cfg['statistics_path']).read_text())
    ds=FutureDataset(cfg['dataset_root'],stats,'val')
    fixed=json.loads((RUN/'validation_samples.json').read_text())['indices'][:a.samples]
    batches=[default_collate([ds[int(i)]]) for i in fixed]
    emit('samples',[dict(index=int(i),episode=int(b['episode_id'][0]),t=int(b['t'][0]),instruction=b['instruction'][0]) for i,b in zip(fixed,batches)])
    hashes={}
    for f in ['starVLA/model/framework/vlas/flowmatching_expert.py','starVLA/model/framework/vlas/cross_attention_dit.py','starVLA/model/framework/latent_world/vlm_adapter.py','goal_local/model.py','goal_local/action.py','robotwin/model.py','robotwin/data.py']:
        hashes[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    emit('source_sha256',hashes)
    print('Loading fresh inherited model',flush=True)
    model=RobotwinPolicy.from_baseline(cfg).eval()
    state=torch.load(cfg['baseline_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    mismatch=[];count=0
    for k,v in model.state_dict().items():
        old=('policy_backend.vlm.model.'+k[5:] if k.startswith('qwen.') else
             'policy_backend.lam.vision_encoder.model.'+k[5:] if k.startswith('dino.') else
             'policy_backend.'+k)
        if old in state:
            count+=1
            if not torch.equal(v,state[old].to(v.dtype)):mismatch.append(k)
    emit('retained_weights',dict(compared_tensors=count,mismatched=mismatch,initialization=model.initialization_report))
    del state
    model=model.cuda();model.checkpoint_blocks=False
    with torch.autocast('cuda',dtype=torch.bfloat16):
        qr,enc,ref,cur=qwen_reference(model,batches[0]);emit('qwen_single',qr)
        b2=default_collate([ds[int(fixed[0])],ds[int(fixed[-1])]])
        qr2,*_=qwen_reference(model,b2);emit('qwen_variable_length_batch',qr2)
        # Real current visual/text features; role inputs isolated from current.
        enc=model.encode(batches[0]);cond=model.action_conditions(batches[0],enc)
    # Disable autocast for this precise kernel parity test; both use same FP32 head.
    cond=tuple(x.float() if x.is_floating_point() else x for x in cond)
    noise=torch.randn(1,50,32,device='cuda');hz=torch.tensor([30.],device='cuda');emb=torch.ones(1,dtype=torch.long,device='cuda')
    _,valid,_=build_time_grid(8/30,hz,50);t=torch.tensor([400],device='cuda');x=noise*valid[...,None]
    native=native_velocity(model.flow,x,t,hz,emb,valid,cond)
    bypass=model.flow.velocity(x,t,hz,emb,valid,cond,bypass_roles=True)
    off=model.flow.velocity(x,t,hz,emb,valid,no_roles(cond))
    emit('flow_fp32_parity',dict(bypass_vs_native=difference(bypass[:,:8,:16],native[:,:8,:16]),
        plan_present_roles_off_vs_native=difference(off[:,:8,:16],native[:,:8,:16])))
    controls=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for i,b in enumerate(batches[:4]):
            encoded=model.encode(b);cond=model.action_conditions(b,encoded)
            torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
            native8=sample(model.flow,cond,noise,8,'native')
            no_plan=sample(model.flow,cond,noise,8,'no_plan')
            off=sample(model.flow,cond,noise,8,'roles_off');full=sample(model.flow,cond,noise,8,'full')
            native36=sample(model.flow,cond,noise,36,'native')
            # Restore old query outputs while still omitting LaWM, one change only.
            _,_,allcur,rawcur=qwen_reference(model,b)
            allcond=model.flow.conditions(allcur,rawcur[1],model.current_dino(b['current_main']),encoded['goal'],encoded['local'])
            oldqueries8=sample(model.flow,allcond,noise,8,'native')
            target=b['actions'][:,:8,:16].cuda();mask=b['actions_mask'][:,:8,:16].cuda()
            mse=lambda y:float(((y-target).square()*mask).sum()/mask.sum())
            controls.append(dict(sample=i,bypass_vs_native=difference(no_plan,native8),
                plan_only_effect=difference(off,native8),role_injection_effect=difference(full,off),
                horizon_36_vs_8=difference(native36[:,:8],native8),query_removal_effect=difference(oldqueries8,native8),
                normalized_action_mse=dict(native_aba2_8=mse(native8),native_aba2_36_first8=mse(native36[:,:8]),
                    plan_only=mse(off),new_full=mse(full),old_queries_restored=mse(oldqueries8))))
            emit('fresh_action_controls',controls)
    del model,enc,cond,ref,cur,encoded,allcur,allcond,rawcur;gc.collect();torch.cuda.empty_cache()
    for label,path in [('teacher1000',ROOT/'robotwin/runs/stage_I_20260924/step_001000'),('student2000',RUN/'step_002000')]:
        print('Loading',label,flush=True)
        model=RobotwinPolicy.load(path).cuda().eval();model.checkpoint_blocks=False
        rows=[];goals=[];locals_=[]
        for i,b in enumerate(batches):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                enc=model.encode(b);cond=model.action_conditions(b,enc)
                goals.append(enc['goal'].float().cpu());locals_.append(enc['local'].float().cpu())
                full_loss=fm_loss(model,b,cond,seed=10000+i)
                off_loss=fm_loss(model,b,no_roles(cond),seed=10000+i)
                torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
                full=sample(model.flow,cond,noise);off=sample(model.flow,cond,noise,variant='roles_off')
                noplan=sample(model.flow,cond,noise,variant='no_plan')
                target=b['actions'][:,:8,:16].cuda();mask=b['actions_mask'][:,:8,:16].cuda()
                mse=lambda y:float(((y-target).square()*mask).sum()/mask.sum())
                row=dict(sample=i,flow_loss=full_loss,roles_off_flow_loss=off_loss,
                    action_mse=mse(full),roles_off_action_mse=mse(off),no_plan_action_mse=mse(noplan),
                    role_action_effect=difference(full,off),plan_action_effect=difference(off,noplan))
                if model.mode=='posterior':
                    cur=model.current_embeddings(model.encode_current_inputs(b['current_main'],b['current_wrist'],b['instruction']))
                    lv,gv=model.encode_future_images(b)
                    g,l=model.future(lv,gv,b['local_valid'].cuda(),b['goal_frame_valid'].cuda())
                    zero=model.language(cur,torch.zeros_like(g),torch.zeros_like(l))
                    zero_cond=model.action_conditions(b,zero)
                    zero_action=sample(model.flow,zero_cond,noise)
                    row['zero_future_seed_action_effect']=difference(zero_action,full)
                    row['zero_future_seed_action_mse']=mse(zero_action)
                else:
                    scales=model.gate.scales.clone();model.gate.scales.zero_()
                    gated_off=model.encode(b);model.gate.scales.copy_(scales)
                    gateoff=sample(model.flow,model.action_conditions(b,gated_off),noise)
                    row['gate_action_effect']=difference(gateoff,full);row['gate_off_action_mse']=mse(gateoff)
                rows.append(row)
                emit(label,dict(rows=rows,goal_slots=slot_stats(torch.cat(goals)),local_slots=slot_stats(torch.cat(locals_)),
                    goal_scale=model.flow.goal_scale.float().tolist(),local_scale=model.flow.local_scale.float().tolist()))
        del model,enc,cond;gc.collect();torch.cuda.empty_cache()
    REPORT['finished_at']=time.time();REPORT['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    REPORT['completed']=True;save();print('AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
