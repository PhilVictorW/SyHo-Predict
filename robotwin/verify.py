"""Data, inherited-weight and real-batch gradient checks before starting Stage I."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate, Subset
from .data import FutureDataset, ActionNormalizer, CAMERAS

WORK = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    cfg=json.loads((WORK/'configs/stage_I.json').read_text())
    stats=json.loads(Path(cfg['statistics_path']).read_text())
    out=WORK/'verification';out.mkdir(exist_ok=True)
    ds=FutureDataset(cfg['dataset_root'],stats,'train')
    val=FutureDataset(cfg['dataset_root'],stats,'val',decode=False)
    a={e['episode_index'] for e in ds.selected};b=set(val.validation_episodes)
    assert a.isdisjoint(b) and len(a|b)==27500
    audit=ds.audit();audit.update(train_episodes=len(a),validation_episodes_count=len(b),train_val_disjoint=True)
    print('DATA_AUDIT',json.dumps({k:v for k,v in audit.items() if k!='validation_episodes'}),flush=True)
    (out/'dataset_audit.json').write_text(json.dumps(audit,indent=2))
    # Independent reference: the original per-modality normalization implementation.
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
    sample=torch.from_numpy(ds.actions[::10000].copy())
    norm=ds.normalizer.normalize(sample)
    reference=sample.clone()
    s=stats['agilex']['action']
    for ids in ([0,1,2],[7],[8,9,10],[15]):
        n=Normalizer(mode='min_max',statistics={k:torch.tensor(s[k])[ids] for k in ('min','max')})
        reference[:,ids]=n.forward(sample[:,ids])
    reference[:,[7,15]] *= -1
    assert torch.equal(norm,reference)
    assert torch.allclose(ds.normalizer.denormalize(norm),sample,atol=2e-7)
    assert torch.equal(norm[:,[3,4,5,6,11,12,13,14]],sample[:,[3,4,5,6,11,12,13,14]])
    checked=[]
    for i in np.linspace(0,len(ds.selected)-1,8,dtype=int):
        e=ds.selected[int(i)]
        for t in (0,e['length']-1):
            row=ds.sample(e,t)
            assert row['current_main'].shape==(3,256,256) and row['current_wrist'].shape==(2,3,256,256)
            assert torch.isfinite(row['actions']).all()
            assert row['actions_mask'].sum()==min(8,e['length']-t)*16
            if t==e['length']-1:
                assert not row['local_valid'] and not row['goal_frame_valid'].any()
                assert row['local_main'].count_nonzero()==0 and row['local_wrist'].count_nonzero()==0
            checked.append(dict(episode=e['episode_index'],t=t))
    print('FRAMES_VERIFIED',len(checked),flush=True)
    loader=DataLoader(Subset(ds,[0,len(ds)//2,len(ds)-1]),batch_size=1,num_workers=2,multiprocessing_context='spawn')
    assert sum(len(batch['episode_id']) for batch in loader)==3
    report=dict(data_samples=checked,spawn_loader=True,normalization_matches_reference=True,
                quaternion_passthrough=True,normalization_roundtrip=True,three_cameras=True)
    if args.gpu:
        from .model import RobotwinPolicy
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        torch.cuda.set_per_process_memory_fraction(.40)
        print('LOADING_ROBOTWIN_BASELINE',flush=True)
        model=RobotwinPolicy.from_baseline(cfg).cuda()
        report['initialization']=model.initialization_report
        report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert all(not p.requires_grad for p in model.qwen.visual.parameters())
        assert all(not p.requires_grad for p in model.dino.parameters())
        assert not any(k.startswith(('lam.','vlm_to_lam')) for k in model.state_dict())
        model.train()
        batch=default_collate([ds.sample(ds.selected[0],0)])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            enc=model.encode(batch)
            assert enc['goal'].shape==enc['local'].shape==(1,8,2048)
            loss=model.action_loss(batch,enc)
        assert torch.isfinite(loss)
        loss.backward()
        probes=['future.local.query','future.goal.query','future.view','future.horizon','flow.plan',
                'flow.goal_proj.weight','flow.local_proj.weight','flow.goal_attn.0.in_proj_weight',
                'flow.local_attn.0.in_proj_weight','qwen.language_model.layers.0.self_attn.q_proj.weight',
                'act_query','flow_action_query']
        params=dict(model.named_parameters());grads={}
        for key in probes:
            g=params[key].grad
            assert g is not None and torch.isfinite(g).all() and g.abs().max()>0,key
            grads[key]=float(g.float().norm())
        assert all(p.grad is None for p in model.qwen.visual.parameters())
        assert all(p.grad is None for p in model.dino.parameters())
        report.update(real_batch_flow=float(loss),gradient_norms=grads,
                      frozen_vision_no_grad=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        del loss,enc,batch
        model.zero_grad(set_to_none=True)
        tail=default_collate([ds.sample(ds.selected[0],ds.selected[0]['length']-1)])
        with torch.autocast('cuda',dtype=torch.bfloat16): result=model(tail)
        assert torch.isfinite(result['loss']);result['loss'].backward()
        report['tail_loss']=float(result['loss']);report['tail_backward_finite']=True
    report['passed']=True;report['time']=time.time()
    (out/('gpu.json' if args.gpu else 'data.json')).write_text(json.dumps(report,indent=2))
    print('VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':main()
