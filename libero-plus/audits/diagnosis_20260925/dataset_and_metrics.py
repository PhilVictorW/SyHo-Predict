import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib,collections,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];WORK=ROOT/'libero-plus'
sys.path[:0]=[str(ROOT),str(WORK/'runtime')]
import torch
torch.set_num_threads(1)
from indexed_future import IndexedFutureDataset
cfg=json.loads((WORK/'configs/stage_I.json').read_text());stats=json.loads(Path(cfg['statistics_path']).read_text())
ds=IndexedFutureDataset(cfg['dataset_root'],stats,'all',decode=False)
val=set(ds.validation_episodes);groups=collections.defaultdict(list)
for e in ds.episodes:
    a=ds.actions[e['dataset_from_index']:e['dataset_to_index']]
    h=hashlib.sha256(e['tasks'][0].encode()+a.tobytes()).hexdigest();groups[h].append(e)
overlap=[]
for h,es in groups.items():
    tr=[e for e in es if e['episode_index'] not in val];va=[e for e in es if e['episode_index'] in val]
    if tr and va:overlap.extend(e['episode_index'] for e in va)
report=dict(time=time.time(),same_full_action_sequence_and_instruction=dict(total_episodes=len(ds.episodes),unique_sequences=len(groups),
    validation_episodes=len(val),validation_with_training_duplicate=len(overlap),fraction=len(overlap)/len(val),
    note='Hash of full float32 action sequence plus instruction; images may differ. Not a random-frame overlap test.'),training={})
for stage,name in [('I','stage_I_20260923'),('II','stage_II_20260925')]:
    rows=[json.loads(x) for x in (WORK/'runs'/name/'metrics.jsonl').read_text().splitlines()]
    tr=[r for r in rows if r.get('split')=='train'];va=[r for r in rows if r.get('split')=='validation']
    windows=[]
    for lo,hi in [(1,100),(401,500),(901,1000),(1901,2000),(4901,5000),(max(1,tr[-1]['step']-99),tr[-1]['step'])]:
        rr=[r for r in tr if lo<=r['step']<=hi]
        if rr:windows.append(dict(lo=lo,hi=hi,n=len(rr),mean={k:float(np.mean([r[k] for r in rr])) for k in ['flow','loss','local_distill','goal_distill','grad_norm','update_seconds'] if all(k in r for r in rr)}))
    report['training'][stage]=dict(last_step=tr[-1]['step'],effective_batch=tr[-1].get('effective_batch',32),windows=windows,validation=va)
    if stage=='II':report['training'][stage]['observed_fraction_epoch_at_500']=500*32/(len(ds)-sum(e['length'] for e in ds.episodes if e['episode_index'] in val))
offline=json.loads((HERE/'offline_ablation.json').read_text());corrected={}
for label,model in offline['models'].items():
    rows=[]
    for anchor,pred in zip(offline['anchors'],model['per_anchor']):
        b=ds.sample(ds.episodes[anchor['episode']],anchor['t']);target=b['actions'][:8,:7].numpy();target[:,6]=1-target[:,6]
        a=np.asarray(pred['actions']);flipped=a.copy();flipped[:,6]=1-flipped[:,6]
        rows.append(dict(unmodified_physical_mse=float(np.mean((a-target)**2)),flipped_physical_mse=float(np.mean((flipped-target)**2)),
            unmodified_physical_gripper_accuracy=float(((a[:,6]>.5)==(target[:,6]>.5)).mean()),
            flipped_physical_gripper_accuracy=float(((flipped[:,6]>.5)==(target[:,6]>.5)).mean())))
    corrected[label]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
report['offline_using_original_physical_gripper_convention']=corrected
(HERE/'dataset_and_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
