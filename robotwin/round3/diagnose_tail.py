"""Matched original-weight velocity tests. No parameter updates."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.round2.model import Dataset36,CompatibilityTeacher
from robotwin.round3.model import TerminalHoldDataset
from robotwin.round2.train import atomic_json
from robotwin.diagnostics.migration_20260925.audit import native_velocity
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.model.framework.latent_world.batch_utils import sample_or_pad_sequence_with_mask
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'diagnostics'

def main():
    OUT.mkdir(exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.manual_seed(92026)
    report=dict(pid=os.getpid(),phase='loading',rows=[],started_at=time.time())
    def save():report['updated_at']=time.time();atomic_json(OUT/'tail_control.json',report)
    save()
    cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text());model=CompatibilityTeacher(cfg).cuda().eval()
    old=Dataset36(cfg['dataset_root'],model.policy.statistics,'train')
    fixed=TerminalHoldDataset(cfg['dataset_root'],model.policy.statistics,'train')
    # Source-level padding function and actual reference collator utility, no recreated oracle.
    dry=TerminalHoldDataset(cfg['dataset_root'],model.policy.statistics,'train',decode=False)
    checked=0
    for e in dry.selected[:5]:
        raw=np.array(dry.actions[e['dataset_from_index']:e['dataset_to_index']])
        for t in [0,e['length']-1,max(0,e['length']-8),max(0,e['length']-36)]:
            original=LeRobotSingleDataset.retrieve_data_and_pad(None,raw,np.arange(t,t+36),len(raw),'first_last')
            b=dry.sample(e,t)
            assert np.array_equal(original.astype(np.float32),b['action_chunk'].numpy())
            assert b['actions_mask'].sum().item()==36*16
            assert b['observed_actions_mask'].sum().item()==min(36,len(raw)-t)*16
            padded,mask=sample_or_pad_sequence_with_mask(b['actions'][:36],50)
            assert torch.equal(padded,b['actions']) and torch.equal(mask,b['actions_mask'].any(-1))
            checked+=1
    report['reference_padding_checks']=checked;del dry
    ids=json.loads((ROOT/'robotwin/round2/runs/teacher_compat36_20260925/samples.json').read_text())['train_indices']
    selected=[]
    for idx in ids:
        j=int(np.searchsorted(old.offsets,idx,side='right')-1);e=old.selected[j];t=int(idx-old.offsets[j]);remain=e['length']-t
        kind='tail' if remain<36 else 'interior'
        if sum(s[0]==kind for s in selected)<12:selected.append((kind,idx,e,t))
        if len(selected)==24:break
    report['phase']='matched_velocity';save()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,(kind,idx,e,t) in enumerate(selected):
            b=default_collate([old.sample(e,t)])
            # Same decoded images/current/future, change only action window preparation.
            f={k:v.clone() if torch.is_tensor(v) else v for k,v in b.items()}
            n=min(36,e['length']-t);f['actions'][:,n:36,:16]=f['actions'][:,n-1:n,:16];f['actions_mask'][:,:36,:16]=True
            prep=model.prepare(b);cond=model.conditions(b,prep)
            torch.manual_seed(1826+i);noise=torch.randn(1,50,32,device='cuda')
            hz=b['action_hz'].cuda();emb=b['embodiment_id'].cuda();mask=b['actions_mask'].cuda()
            targets={k:v['actions'].cuda() for k,v in [('old',b),('fixed',f)]}
            row=dict(kind=kind,index=idx,episode=e['episode_index'],t=t,remaining=e['length']-t,taus=[])
            for tau in [0.,.25,.5,.75]:
                vals={}
                for k,item in [('old',b),('fixed',f)]:
                    valid=item['actions_mask'].any(-1).cuda()
                    x=((1-tau)*noise+tau*targets[k])*valid[...,None]
                    bucket=torch.tensor([int(tau*1000)],device='cuda')
                    pred=native_velocity(model.flow,x,bucket,hz,emb,valid,cond)
                    error=(pred.float()-(targets[k]-noise)).square()
                    vals[k]=float((error*mask).sum()/mask.sum())
                row['taus'].append(dict(tau=tau,**vals))
            report['rows'].append(row);save();print(i+1,kind,row['remaining'],row['taus'],flush=True)
    report['phase']='completed';report['completed']=True
    report['summary']={kind:{k:float(np.mean([z[k] for r in report['rows'] if r['kind']==kind for z in r['taus']])) for k in ['old','fixed']} for kind in ['tail','interior']}
    save();print(report['summary'],flush=True)

if __name__=='__main__':main()
