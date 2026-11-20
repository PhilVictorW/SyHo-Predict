"""Compare actual gripper semantics, read-only datasets and CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
cfg=json.loads((WORK/'configs/stage_I.json').read_text())
root=Path(cfg['dataset_root']);base=Path('/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero_merged_no_noops_20hz')
report=dict(time=time.time(),source_shards=32,read_only=True)

def describe(a,s,ep):
    width=np.abs(s[:,6])+np.abs(s[:,7])
    result=dict(frames=len(a),action_labels=np.unique(a[:,-1]).tolist())
    for label in (0.,1.):
        valid=(a[:-5,-1]==label)&(a[5:,-1]==label)&(ep[:-5]==ep[5:])
        result[str(label)]=dict(stable_frames=int(valid.sum()),current_width_median=float(np.median(width[:-5][valid])),
            next5_width_median=float(np.median(width[5:][valid])),next5_width_mean=float(width[5:][valid].mean()),
            open_fraction_next5=float((width[5:][valid]>.065).mean()),closed_fraction_next5=float((width[5:][valid]<.035).mean()))
    return result

table=pq.read_table(base/'data/chunk-000/file-000.parquet',columns=['action','observation.state','episode_index','frame_index'])
oa=np.stack(table['action'].to_pylist());os_=np.stack(table['observation.state'].to_pylist());oe=table['episode_index'].to_numpy()
report['original_libero']=describe(oa,os_,oe)
actions=[];states=[];episodes=[]
for sid in np.linspace(0,1023,32,dtype=int):
    with np.load(root/f'shard_{sid:04d}'/'arrays.npz',allow_pickle=False) as z:
        actions.append(z['actions']);states.append(z['states']);episodes.append(z['episode']+int(sid)*1000)
pa=np.concatenate(actions);ps=np.concatenate(states);pe=np.concatenate(episodes)
report['plus_indexed']=describe(pa,ps,pe)
# Same continuous state and motion, excluding ambiguous repeated keys.
index={};ambiguous=set()
for a,s in zip(oa,os_):
    key=np.round(np.r_[s,a[:6]],4).astype(np.float32).tobytes()
    if key in index and index[key]!=a[-1]:ambiguous.add(key)
    else:index[key]=float(a[-1])
same=opposite=0;examples=[]
for a,s in zip(pa,ps):
    key=np.round(np.r_[s,a[:6]],4).astype(np.float32).tobytes()
    if key not in index or key in ambiguous:continue
    old=index[key];same+=int(old==a[-1]);opposite+=int(old!=a[-1])
    if len(examples)<8:examples.append(dict(original_label=old,plus_label=float(a[-1]),state=s.tolist(),motion=a[:6].tolist()))
report['matched_same_state_and_motion_rounded_4dp']=dict(same_label=same,opposite_label=opposite,examples=examples)
meta=json.loads((WORK/'cache/index/metadata.json').read_text());groups=defaultdict(lambda:dict(episodes=0,frames=0))
for e in meta['episodes']:
    p=Path(e['origin']).parts;k='/'.join(p[-3:-1]);groups[k]['episodes']+=1;groups[k]['frames']+=e['length']
report['training_source_groups']=groups
report['unique_instructions']=len({e['tasks'][0] for e in meta['episodes']})
for label,v in report.items():print(label,json.dumps(v),flush=True)
(HERE/'gripper_data_audit.json').write_text(json.dumps(report,indent=2))
