"""Isolated GPU0 investigation. No optimizer updates or production mutations."""
import gc,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(OUT.parent/'migration_20260925'))
from audit import qwen_reference,sample,difference,no_roles,native_velocity
from robotwin.model import RobotwinPolicy
from robotwin.data import FutureDataset
from goal_local.model import make_stage_two
from starVLA.model.framework.vlas.lawam import VLMToLAMQFormer
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid
from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient

RUN=ROOT/'robotwin/runs/stage_II_20260925'
report=dict(started_at=time.time(),pid=os.getpid(),physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
            scope='32 offline samples plus 8 fresh-student gradient probes; no optimizer or online rollout',phases={})

def save():
    p=OUT/'results.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(OUT/'results.json')

def event(name,value):
    report['phase']=name;report['updated_at']=time.time();save();print(name,value,flush=True)

def action_metrics(pred,batch,stats):
    pred=pred[:,:8,:16].float().cpu();target=batch['actions'][:,:8,:16];mask=batch['actions_mask'][:,:8,:16]
    mse=float(((pred-target).square()*mask).sum()/mask.sum())
    def physical(x):
        return ModelClient.unnormalize_actions(x[0].numpy(),stats['agilex']['action'],gripper_indices=(),
            passthrough_indices=(3,4,5,6,11,12,13,14),invert_indices=(7,15))
    y,t=physical(pred),physical(target);valid=mask[0,:,0].numpy();y,t=y[valid],t[valid]
    xyz=np.stack([np.linalg.norm(y[:,i:i+3]-t[:,i:i+3],axis=-1) for i in (0,8)],-1)
    angular=[]
    for i in (3,11):
        dot=np.abs((y[:,i:i+4]*t[:,i:i+4]).sum(-1));angular.append(2*np.arccos(np.clip(dot,0,1))*180/np.pi)
    return dict(normalized_mse=mse,mean_position_error_cm=float(xyz.mean()*100),
        mean_rotation_error_deg=float(np.stack(angular,-1).mean()),
        mean_gripper_error=float(np.abs(y[:,[7,15]]-t[:,[7,15]]).mean()))

@torch.no_grad()
def original_phase(cfg,batches):
    event('loading_original',len(batches))
    model=RobotwinPolicy.from_baseline(cfg).eval()
    decoder=LAMDecoder_v2(context_dim=1024,input_dim=768,num_queries=1,num_layers=12,num_heads=16,
        dropout=0.,grid_hw=(16,16),train_in_latent=True,ffn_expansion_factor=4,num_embodiments=32,code_dim=32,last_ln=True)
    mapper=VLMToLAMQFormer(vlm_hidden_dim=2048,lam_code_dim=32,num_layers=1,num_heads=8,ffn_expansion_factor=4.,dropout=0.)
    state=torch.load(cfg['baseline_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    for module,prefix in [(decoder,'policy_backend.lam.decoder.'),(mapper,'policy_backend.vlm_to_lam.')]:
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True,assign=True)
    del state
    model=model.cuda();decoder=decoder.cuda().eval();mapper=mapper.cuda().eval()
    rows=[];report['phases']['original']=rows
    for i,b in enumerate(batches):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,enc,allcur,cur=qwen_reference(model,b)
            dino=model.current_dino(b['current_main']);future=decoder(dino,mapper(allcur[cur[2]].reshape(1,8,2048)))[:,0]
            def cond(lawm,queries):
                c,v=(allcur,cur[1]) if queries else (enc['current'],enc['current_valid'])
                vis=torch.cat((dino,future),1) if lawm else dino
                base=torch.cat((vis,model.flow.enc_vlm(c)),1)
                im=torch.cat((torch.ones(1,vis.shape[1],device='cuda',dtype=torch.bool),torch.zeros_like(v)),1)
                vm=torch.cat((torch.zeros(1,vis.shape[1],device='cuda',dtype=torch.bool),v),1)
                return base,im,vm,None,None,None
            torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
            outputs={}
            for name,h,l,q in [('original36',36,True,True),('original8',8,True,True),
                                ('no_lawm36',36,False,True),('aba2_36',36,False,False)]:
                outputs[name]=sample(model.flow,cond(l,q),noise,h,'native')
            rows.append(dict(sample=i,metrics={k:action_metrics(v,b,model.statistics) for k,v in outputs.items()}))
            # Compare every inherited DiT block on exactly matching first-8 input tokens.
            if i==0:
                traces={}
                for h in (36,8):
                    seen={};handles=[]
                    for n,block in enumerate(model.flow.DiT.transformer_blocks):
                        def capture(module,args,value,n=n):seen[n]=value[:,:8].float().cpu()
                        handles.append(block.register_forward_hook(capture))
                    model.flow.config.horizon_sec=h/30
                    hz=torch.tensor([30.],device='cuda');emb=torch.ones(1,dtype=torch.long,device='cuda')
                    _,valid,_=build_time_grid(h/30,hz,50)
                    native_velocity(model.flow,noise*valid[...,None],torch.zeros(1,dtype=torch.long,device='cuda'),hz,emb,valid,cond(True,True))
                    for handle in handles:handle.remove()
                    traces[h]=seen
                model.flow.config.horizon_sec=8/30
                report['block_trace']=[dict(block=n+1,type='self_attention' if n%2 else 'cross_attention',
                    **difference(traces[8][n],traces[36][n])) for n in range(16)]
        event('original_controls',f'{i+1}/{len(batches)}')
    del model,decoder,mapper;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def teacher_phase(batches):
    event('loading_teacher1000','retained Stage I source')
    model=RobotwinPolicy.load(ROOT/'robotwin/runs/stage_I_20260924/step_001000').cuda().eval()
    model.checkpoint_blocks=False
    donor=next(b for b in batches if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all()))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        dl,dg=model.encode_future_images(donor)
        wrong_g,wrong_l=model.future(dl,dg,donor['local_valid'].cuda(),donor['goal_frame_valid'].cuda())
    rows=[];report['phases']['teacher1000']=rows
    for i,b in enumerate(batches):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cur=model.current_embeddings(model.encode_current_inputs(b['current_main'],b['current_wrist'],b['instruction']))
            lv,gv=model.encode_future_images(b);g,l=model.future(lv,gv,b['local_valid'].cuda(),b['goal_frame_valid'].cuda())
            encoded=model.language(cur,g,l);cond=model.action_conditions(b,encoded)
            wrong=model.language(cur,wrong_g,wrong_l);wc=model.action_conditions(b,wrong)
            torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
            full=sample(model.flow,cond,noise);off=sample(model.flow,cond,noise,variant='roles_off');mismatch=sample(model.flow,wc,noise)
            rows.append(dict(sample=i,donor_same_episode=bool(b['episode_id'][0]==donor['episode_id'][0]),
                local_valid=bool(b['local_valid'][0]),goal_valid=bool(b['goal_frame_valid'][0].any()),
                metrics={k:action_metrics(v,b,model.statistics) for k,v in [('full',full),('roles_off',off),('wrong_future',mismatch)]},
                roles_off_effect=difference(off,full),wrong_future_effect=difference(mismatch,full)))
        event('teacher_controls',f'{i+1}/{len(batches)}')
    return model

def gradient_phase(teacher,batches):
    event('building_fresh_student','Stage I 1000 initialization, zero updates')
    torch.manual_seed(2026)
    system=make_stage_two(teacher).cuda().eval();student=system.student
    student.requires_grad_(False)
    names=['q_goal','q_local','act_query','flow_action_query',
           'qwen.language_model.layers.0.self_attn.q_proj.weight',
           'qwen.language_model.layers.15.self_attn.q_proj.weight']
    params=dict(student.named_parameters());selected=[params[n] for n in names]
    for p in selected:p.requires_grad_(True)
    report['gradient_scope']=dict(initialization='new student from Stage I 1000, zero optimizer updates',
        selected_parameters=names,not_whole_model_gradient=True)
    rows=[];report['phases']['initial_gradient_probes']=rows
    for i,b in enumerate(batches[:8]):
        torch.manual_seed(12026+i)
        with torch.autocast('cuda',dtype=torch.bfloat16):losses=system(b)
        values={k:float(v) for k,v in losses.items()};grads={}
        for key in ['flow','local_distill','goal_distill']:
            gs=torch.autograd.grad(losses[key],selected,retain_graph=key!='goal_distill',allow_unused=True)
            grads[key]=[torch.zeros_like(p,dtype=torch.float32,device='cpu') if g is None else g.detach().float().cpu() for p,g in zip(selected,gs)]
        comparisons={}
        for n,af,al,ag in zip(names,grads['flow'],grads['local_distill'],grads['goal_distill']):
            d=al+ag;an=float(af.norm());dn=float(d.norm());cos=float((af.flatten()@d.flatten())/(af.norm()*d.norm()).clamp_min(1e-20))
            comparisons[n]=dict(action_gradient_norm=an,local_gradient_norm=float(al.norm()),goal_gradient_norm=float(ag.norm()),
                combined_distill_norm=dn,distill_to_action_norm_ratio=dn/max(an,1e-20),action_vs_distill_cosine=cos)
        rows.append(dict(sample=i,losses=values,parameters=comparisons));event('gradient_probes',f'{i+1}/8')
        del losses,grads;gc.collect()
    del system,student;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25);torch.manual_seed(2026);np.random.seed(2026)
    cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
    ds=FutureDataset(cfg['dataset_root'],json.loads(Path(cfg['statistics_path']).read_text()),'val')
    indices=json.loads((RUN/'validation_samples.json').read_text())['indices'][:32]
    event('decoding_samples',len(indices));batches=[default_collate([ds[int(i)]]) for i in indices]
    report['samples']=[dict(index=int(i),episode=int(b['episode_id'][0]),t=int(b['t'][0]),instruction=b['instruction'][0]) for i,b in zip(indices,batches)];save()
    original_phase(cfg,batches)
    teacher=teacher_phase(batches)
    gradient_phase(teacher,batches)
    report['completed']=True;report['finished_at']=time.time();report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    event('completed','no optimizer updates')

if __name__=='__main__':main()
