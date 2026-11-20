"""Check whether short absolute-action supervision rewards holding current pose."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
import torch
from robotwin.data import ActionNormalizer

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
norm=ActionNormalizer(json.loads(Path(cfg['statistics_path']).read_text()))
episodes=json.loads((ROOT/'robotwin/cache/index.json').read_text())['episodes']
samples=json.loads((OUT/'audit_results.json').read_text())['experiments']['samples']
actions=np.load(ROOT/'robotwin/cache/actions.npy',mmap_mode='r')
needed={episodes[s['episode']]['dataset_from_index']+s['t']:s for s in samples}
states={}
for f in sorted((Path(cfg['dataset_root'])/'data').rglob('*.parquet')):
    p=pq.ParquetFile(f);names=p.schema.names;ix=names.index('index')
    for rg in range(p.metadata.num_row_groups):
        st=p.metadata.row_group(rg).column(ix).statistics
        if not st or not st.has_min_max:raise ValueError('Index row-group statistics required')
        wanted=[i for i in needed if i not in states and st.min<=i<=st.max]
        if not wanted:continue
        table=p.read_row_group(rg,columns=['index','observation.state'])
        idx=table['index'].to_numpy();values=np.asarray(table['observation.state'].combine_chunks().values).reshape(-1,16)
        for i in wanted:
            hit=np.flatnonzero(idx==i)
            if len(hit):states[i]=values[hit[0]].copy()
    if len(states)==len(needed):break
assert len(states)==len(needed)
rows=[]
for i,s in needed.items():
    e=episodes[s['episode']];state=torch.tensor(states[i]);sn=norm.normalize(state)
    row=dict(sample=s,index=i,first_action_vs_current_state_max_abs=float(np.abs(actions[i]-states[i]).max()))
    for h in [8,36]:
        target=torch.tensor(actions[i:min(i+h,e['dataset_to_index'])].copy())
        error=norm.normalize(target)-sn
        row[f'hold_current_pose_mse_{h}']=float(error.square().mean())
        pos=[0,1,2,8,9,10]
        row[f'mean_xyz_displacement_m_{h}']=float((target[:,pos]-state[pos]).reshape(-1,2,3).norm(dim=-1).mean())
    rows.append(row)
report=dict(scope='8 fixed offline samples; zero-order hold baseline, no model involved',rows=rows)
(OUT/'data_hold_control.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
