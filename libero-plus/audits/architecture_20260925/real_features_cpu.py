"""One real validation anchor, offline CPU teacher/student latent diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];WORK=ROOT/'libero-plus'
sys.path[:0]=[str(ROOT),str(WORK/'runtime')]
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
from accelerate import init_empty_weights
from goal_local.model import GoalLocalPolicy
from indexed_future import IndexedFutureDataset
from torch.utils.data import default_collate
cfg=json.loads((WORK/'configs/stage_I.json').read_text())
stats=json.loads(Path(cfg['statistics_path']).read_text())
ds=IndexedFutureDataset(cfg['dataset_root'],stats,'val',cfg['seed'],cfg['val_ratio'])
episode=next(e for e in ds.selected if e['length']>100)
batch=default_collate([ds.sample(episode,0)])
report=dict(anchor=dict(episode=episode['episode_index'],t=0,length=episode['length'],instruction=episode['tasks'][0]),cpu_only=True)
def record(k,v):
    report[k]=v;(HERE/'real_features_cpu.json').write_text(json.dumps(report,indent=2));print(k,json.dumps(v),flush=True)
def stats_of(x):
    x=x.float();n=torch.nn.functional.normalize(x,dim=-1);gram=n@n.transpose(-1,-2);mask=~torch.eye(x.shape[1],dtype=torch.bool)
    return dict(rms=float(x.square().mean().sqrt()),mean_slot_cosine=float(gram[:,mask].mean()),slot_variation_rms=float(x.std(dim=1).square().mean().sqrt()))
outputs={}
for label,path in [('teacher5000',WORK/'runs/stage_I_20260923/retained_checkpoints/step_005000'),('student500',WORK/'runs/stage_II_20260925/step_000500')]:
    begin=time.time()
    with init_empty_weights(include_buffers=False):model=GoalLocalPolicy.load(path,with_action=False)
    # Assign outside Accelerate's temporary meta-registration context.
    payload=torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict({k:v for k,v in payload['state_dict'].items() if not k.startswith('flow.')},strict=True,assign=True)
    del payload
    assert not any(p.is_meta for p in model.parameters())
    assert not any(p.is_meta for p in model.buffers())
    model.eval().requires_grad_(False)
    record(label+'_loaded',dict(seconds=time.time()-begin,mode=model.mode,step=model.source_step))
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
        if model.mode=='posterior':
            fl,fg=model.encode_future_images(batch);g,l=model.future(fl,fg,batch['local_valid'],batch['goal_frame_valid'])
            record('teacher_seed_stats',dict(goal=stats_of(g),local=stats_of(l)))
        else:
            record('student_seed_stats',dict(goal=stats_of(model.q_goal[None]),local=stats_of(model.q_local[None])))
        out=model.encode(batch)
    outputs[label]={k:(g.float().clone(),l.float().clone()) for k,(g,l) in out['layers'].items()}
    record(label+'_layers',{k:dict(goal=stats_of(g),local=stats_of(l)) for k,(g,l) in outputs[label].items()})
    if model.mode=='posterior':
        # Actual images, real vision DeepStack features: native current-path parity.
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            inp=model.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction'])
            cur=model.current_embeddings(inp)
            native=model.qwen.language_model(inputs_embeds=cur[0],attention_mask=cur[1],position_ids=cur[4],
                visual_pos_masks=cur[5],deepstack_visual_embeds=cur[6],use_cache=False,return_dict=True).last_hidden_state
            from goal_local.roles import remove_old_queries
            native,valid=remove_old_queries(native,cur[1],cur[2],cur[3])
            delta=(native[valid].float()-out['current'][valid].float())
            record('real_current_native_parity',dict(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/native[valid].float().norm()),deepstack_layers=len(cur[6])))
    del model,out;gc.collect()
comparison={}
for k in outputs['teacher5000']:
    comparison[k]={}
    for i,name in enumerate(['goal','local']):
        t=outputs['teacher5000'][k][i];s=outputs['student500'][k][i]
        comparison[k][name]=dict(mean_cosine=float(torch.nn.functional.cosine_similarity(t,s,dim=-1).mean()),
            student_teacher_rms_ratio=float(s.norm()/t.norm()),relative_l2=float((s-t).norm()/t.norm()))
record('teacher_student_comparison',comparison)
record('finished',dict(time=time.time(),sample_count=1,not_an_online_success_test=True))
