import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];WORK=ROOT/'libero-plus';sys.path[:0]=[str(ROOT),str(WORK/'runtime')]
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
from accelerate import init_empty_weights
from goal_local.model import GoalLocalPolicy
from goal_local.roles import remove_old_queries
from indexed_future import IndexedFutureDataset
from torch.utils.data import default_collate
path=WORK/'runs/stage_I_20260923/retained_checkpoints/step_005000'
with init_empty_weights(include_buffers=False):m=GoalLocalPolicy.load(path,with_action=False)
p=torch.load(path/'model.pt',mmap=True,map_location='cpu',weights_only=False);m.load_state_dict({k:v for k,v in p['state_dict'].items() if not k.startswith('flow.')},strict=True,assign=True);del p
m.float().eval().requires_grad_(False)
cfg=m.cfg;ds=IndexedFutureDataset(cfg['dataset_root'],m.statistics,'val',cfg['seed'],cfg['val_ratio']);e=next(e for e in ds.selected if e['length']>100);b=default_collate([ds.sample(e,0)])
print('loaded FP32 CPU',flush=True)
with torch.inference_mode():
 inp=m.encode_current_inputs(b['current_main'],b['current_wrist'],b['instruction']);cur=m.current_embeddings(inp);g=torch.zeros(1,8,2048);l=g.clone()
 manual=m.language(cur,g,l)['current']
 native=m.qwen.language_model(inputs_embeds=cur[0],attention_mask=cur[1],position_ids=cur[4],visual_pos_masks=cur[5],deepstack_visual_embeds=cur[6],use_cache=False,return_dict=True).last_hidden_state
 native,valid=remove_old_queries(native,cur[1],cur[2],cur[3]);d=manual[valid]-native[valid]
 r=dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/native[valid].norm()),episode=e['episode_index'],dtype='float32',device='cpu',deepstack_layers=len(cur[6]),time=time.time())
 (HERE/'current_fp32_cpu.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
