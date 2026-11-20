"""CPU checks for inference integration and unchanged training, without loading checkpoints."""
import ast, contextlib, json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Optional, Union, Dict, cast
import torch

ROOT=Path(__file__).resolve().parents[1]
rel='starVLA/model/framework/vlas/lawam.py'
original=ast.parse((ROOT.parent/'LaWAM_aba1'/rel).read_text())
modified=ast.parse((ROOT/rel).read_text())
get_cls=lambda tree: next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LatentWorldPolicyBackend')
old=get_cls(original); new=get_cls(modified)
old_methods={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
new_methods={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for name,method in old_methods.items():
    if name!='predict_action': assert ast.dump(method)==ast.dump(new_methods[name]), name
assert ast.dump(original.body[0])==ast.dump(modified.body[0])
# Execute the actual modified inference method with fake VLM/Flow stages.
ns=dict(torch=torch,Tuple=Tuple,Optional=Optional,Union=Union,Dict=Dict,cast=cast,
        LatentWorldPolicyInferBatch=dict,_cuda_autocast=lambda _:contextlib.nullcontext())
cls=ast.ClassDef(name='Backend',bases=[],keywords=[],decorator_list=[],body=[new_methods['_remove_flow_context_queries'],new_methods['predict_action']])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(ROOT/rel),'exec'),ns)
Backend=ns['Backend']; backend=Backend()
b,l,d=2,40,2048
h=torch.arange(b*l*d,dtype=torch.float32).reshape(b,l,d)
attn=torch.ones(b,l,dtype=torch.bool);attn[0,-3:]=False;attn[1,:3]=False
am=torch.zeros(b,l,dtype=torch.bool);fm=am.clone()
am[0,1:9]=True;fm[0,13:21]=True;am[1,3:11]=True;fm[1,24:32]=True
keep=~(am|fm)
expected_h=torch.stack([h[i,keep[i]] for i in range(b)])
expected_mask=torch.stack([attn[i,keep[i]] for i in range(b)])
shared=SimpleNamespace(h_vlm=h,h_t=torch.randn(b,256,768),h_t1_pred=torch.empty(b,0,768))
calls=[]
backend.flow=SimpleNamespace(config=SimpleNamespace(cfg_guidance_scale=1.,num_inference_steps=10),
    sample_actions_cfg=lambda **kw: calls.append(kw) or torch.zeros(b,8,32))
backend._prepare_infer_batch=lambda *,batch: batch
backend._run_shared_encoding_infer=lambda **kw: shared
backend.num_action_queries=8;backend.flow_action_num_queries=8
backend.inference_disable_lawm=True
batch=dict(attention_mask=attn,act_placeholder_mask=am,flow_placeholder_mask=fm,state=torch.zeros(b,32),
           state_mask=torch.zeros(b,32,dtype=torch.bool),action_hz=torch.full((b,),20.),embodiment_id=torch.zeros(b,dtype=torch.long))
backend.inference_remove_query_tokens=True
out=backend.predict_action(batch=batch)
assert out.shape==(b,8,32)
assert torch.equal(calls[-1]['h_vlm'],expected_h)
assert torch.equal(calls[-1]['attention_mask'],expected_mask)
assert calls[-1]['h_t'] is shared.h_t and calls[-1]['h_t1_star'] is shared.h_t1_pred
assert calls[-1]['num_inference_steps']==10 and calls[-1]['cfg_scale']==1.
# Large changes only at the deleted outputs cannot affect the memory handed to Flow.
h2=h.clone();h2[am|fm]=-99999
filtered,mask=backend._remove_flow_context_queries(h2,attn,am,fm,8,8)
assert torch.equal(filtered,expected_h) and torch.equal(mask,expected_mask)
backend.inference_remove_query_tokens=False
backend.predict_action(batch=batch)
assert calls[-1]['h_vlm'] is h and torch.equal(calls[-1]['attention_mask'],attn)
for bad_act,bad_flow in [(am,am),(am & False,fm)]:
    try: backend._remove_flow_context_queries(h,attn,bad_act,bad_flow,8,8)
    except ValueError: pass
    else: raise AssertionError('Invalid query masks accepted')
# Compile every changed executable and ensure runtime passes no video-saving switches.
for name in [rel,'deployment/model_server/server_policy.py','runtime/run_task_queue.py','runtime/summarize_ablation.py']:
    compile((ROOT/name).read_text(),str(ROOT/name),'exec')
runner=(ROOT/'runtime/run_task_queue.py').read_text()
assert "'--args.save-videos'" not in runner and "'--args.save-first-success-video'" not in runner
print(json.dumps(dict(status='PASS',unchanged_backend_methods=len(old_methods)-1,context_before=[b,l,d],context_after=list(expected_h.shape),
    checks=['batch-specific query positions','padding and order preservation','unchanged remaining values','inference wiring','disabled-switch identity','invalid-mask rejection','no video flags','syntax']),indent=2))
