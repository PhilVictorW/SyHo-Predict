"""Read-only CPU audit; only this audit directory receives new files."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WORK=ROOT/'libero-plus'
BASE=ROOT.parent/'LaWAM_aba2'
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
report=dict(started=time.time(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cpu_threads=2)

def record(key,value):
    report[key]=value
    (HERE/'audit_cpu.json').write_text(json.dumps(report,indent=2))
    print(key,json.dumps(value),flush=True)

def load(path):return torch.load(path,map_location='cpu',mmap=True,weights_only=False)

teacher=load(WORK/'runs/stage_I_20260923/retained_checkpoints/step_005000/model.pt')
student0=load(WORK/'runs/stage_II_20260925/step_000000/model.pt')
student500=load(WORK/'runs/stage_II_20260925/step_000500/model.pt')
original=load(Path(teacher['config']['baseline_checkpoint']))
t=teacher['state_dict'];s=student0['state_dict'];s500=student500['state_dict']
shared={k for k in t if not k.startswith('future.')}
missing=sorted(shared-set(s));extra=sorted(set(s)-shared)
mismatch=[k for k in sorted(shared & set(s)) if not torch.equal(t[k],s[k])]
assert not missing and not mismatch
assert all(k.startswith(('q_goal','q_local','gate.')) for k in extra)
assert teacher['config']==student0['config']==student500['config']
assert teacher['statistics']==student0['statistics']==student500['statistics']
record('teacher_to_student_exact_copy',dict(teacher_step=teacher['step'],student_step=student0['step'],
    tensors=len(shared),mismatched=mismatch,missing=missing,new_keys=extra,
    config_identical=True,normalizer_identical=True,future_resampler_absent=not any(k.startswith('future.') for k in s)))

frozen={}
for oldprefix,newprefix in [('policy_backend.vlm.model.visual.','qwen.visual.'),('policy_backend.lam.vision_encoder.model.','dino.')]:
    keys=[k for k in original if k.startswith(oldprefix)]
    bad=[]
    for k in keys:
        nk=newprefix+k[len(oldprefix):]
        if nk not in t or not torch.equal(original[k],t[nk]) or not torch.equal(t[nk],s500[nk]):bad.append(k)
    frozen[newprefix]=dict(tensors=len(keys),mismatch=bad)
record('frozen_visual_exact_from_original_to_student500',frozen)

files=['starVLA/model/framework/vlas/flowmatching_expert.py','starVLA/model/framework/vlas/cross_attention_dit.py',
       'starVLA/model/framework/latent_world/vlm_adapter.py','starVLA/model/framework/latent_world/batch_utils.py']
record('shared_source_file_identity',{f:(BASE/f).read_bytes()==(ROOT/f).read_bytes() for f in files})
scales={}
for label,st in [('teacher5000',t),('student0',s),('student500',s500)]:
    scales[label]={k:st[k].float().tolist() for k in ['flow.goal_scale','flow.local_scale','gate.scales'] if k in st}
record('learned_scales',scales)

# Small exact architecture tests, including native Qwen vs manual role execution.
spec=importlib.util.spec_from_file_location('audit_contracts',ROOT/'tests/goal_local/test_contracts.py')
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
os.chdir(ROOT)
from goal_local.roles import role_distillation,remove_old_queries
from goal_local.model import make_stage_two
torch.manual_seed(2026)
model=tests.tiny_policy().eval();cur=tests.current()
g=torch.randn(2,8,64);l=torch.randn(2,8,64)
with torch.no_grad():
    manual=model.language(cur,g,l)
    native=model.qwen.language_model(inputs_embeds=cur[0],attention_mask=cur[1],position_ids=cur[4],use_cache=False,return_dict=True).last_hidden_state
    native,valid=remove_old_queries(native,cur[1],cur[2],cur[3])
    delta=(manual['current'][valid]-native[valid]).abs().max().item()
    torch.testing.assert_close(manual['current'][valid],native[valid],rtol=2e-5,atol=2e-5)
record('native_qwen_current_path_parity_tiny',dict(max_abs_difference=delta,passed=True))

# Checkpoint recomputation must preserve BOTH forward and all parameter gradients.
torch.manual_seed(1)
base_system=make_stage_two(tests.tiny_policy());plain=base_system.student.train();ckpt=copy.deepcopy(plain).train();ckpt.checkpoint_blocks=True
cur=tests.current();cur=tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in cur)
def run(m):
    out=m.language(cur,m.q_goal[None].expand(2,-1,-1),m.q_local[None].expand(2,-1,-1))
    loss=out['goal'][...,:5].square().mean()+out['local'][...,:5].square().mean()
    loss.backward();return loss.detach()
torch.manual_seed(123);lp=run(plain);torch.manual_seed(123);lc=run(ckpt)
bad=[];max_diff=0.
for (name,a),(name2,b) in zip(plain.named_parameters(),ckpt.named_parameters()):
    assert name==name2
    if (a.grad is None)!=(b.grad is None):bad.append(name);continue
    if a.grad is not None:
        d=(a.grad-b.grad).abs().max().item();max_diff=max(max_diff,d)
        if not torch.allclose(a.grad,b.grad,rtol=1e-5,atol=1e-6):bad.append(name)
record('language_checkpoint_forward_backward_parity',dict(loss_difference=float((lp-lc).abs()),max_gradient_difference=max_diff,mismatch=bad))
assert not bad

# Normalized feature distillation does not constrain raw feature magnitude.
features=torch.randn(2,8,64);loss=role_distillation(features*3,features,torch.ones(2,dtype=torch.bool))
record('distillation_scale_invariance',dict(student_feature_scale=3,distillation_loss=float(loss),raw_relative_error=2.0))

# Real inherited action weights: separate loop correctness from new Plan topology.
from goal_local.action import HierarchicalFlow
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig,build_time_grid
cfg=ConditionalFlowMatchingConfig(**teacher['config']['flow'])
torch.manual_seed(2026);flow=HierarchicalFlow(cfg).eval()
old={k[len('policy_backend.flow.'):]:v for k,v in original.items() if k.startswith('policy_backend.flow.')}
missing,unexpected=flow.load_state_dict(old,strict=False)
assert not unexpected and all(k.startswith(('plan','goal_proj.','local_proj.','goal_attn.','local_attn.','goal_scale','local_scale')) for k in missing)
record('real_action_weight_loading',dict(inherited_tensors=len(old),new_tensors=len(missing),unexpected=unexpected))
torch.manual_seed(98)
b=1;current=torch.randn(b,8,2048);valid=torch.ones(b,8,dtype=torch.bool);dino=torch.randn(b,256,768)
goal=torch.randn(b,8,2048);local=torch.randn(b,8,2048)
x=torch.randn(b,50,32);hz=torch.full((b,),20.);eid=torch.full((b,),25,dtype=torch.long);tv=torch.arange(50)[None]<8;tm=torch.tensor([300])
with torch.no_grad():
    cond=flow.conditions(current,valid,dino,goal,local)
    base,im,vm,*_=cond
    grid,_,_=build_time_grid(.4,hz,50)
    pos=flow.time_encoder(grid)*tv[...,None]
    h=flow.action_encoder(x,eid)*tv[...,None]
    native=flow.action_decoder(flow.DiT(hidden_states=h,encoder_hidden_states=base,timestep=tm,hidden_attention_mask=tv,
        image_mask=im,vlm_mask=vm,encoder_attention_mask=im|vm,hidden_positional_embeddings=pos),eid)*tv[...,None]
    bypass=flow.velocity(x,tm,hz,eid,tv,cond,bypass_roles=True)
    normal=flow.velocity(x,tm,hz,eid,tv,cond)
    flow.goal_scale.zero_();flow.local_scale.zero_()
    plans_only=flow.velocity(x,tm,hz,eid,tv,cond)
    def error(a,b):
        a=a[:,:8,:7].float();b=b[:,:8,:7].float()
        return dict(max_abs=float((a-b).abs().max()),mean_abs=float((a-b).abs().mean()),relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-10)))
    record('real_action_loop_and_plan_insertion',dict(condition='synthetic fixed features, original SFT action weights; not an online success test',
        bypass_vs_original=error(bypass,native),new_architecture_vs_original=error(normal,native),zero_role_scales_but_plans_present_vs_original=error(plans_only,native)))
    torch.testing.assert_close(bypass,native,atol=2e-5,rtol=2e-5)
    # Show genuine FP32/BF16 numerical difference independently of new role topology.
    with torch.autocast('cpu',dtype=torch.bfloat16):
        cond_bf16=flow.conditions(current,valid,dino,goal,local)
        bf16=flow.velocity(x,tm,hz,eid,tv,cond_bf16,bypass_roles=True)
    record('action_precision_cpu_probe',dict(fp32_vs_bf16=error(bf16,bypass),note='CPU numerical probe, not proof of CUDA accuracy impact'))

record('finished',dict(time=time.time(),gpu_used=False,training_or_evaluation_modified=False))
