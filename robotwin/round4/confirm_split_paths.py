"""Independent episode confirmation of the fixed Plan0/Role3 candidate."""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from robotwin.round2.train import atomic_json
from robotwin.round4.audit_split_paths import sample
from robotwin.round4.model import FutureActionDataset, GroundedTeacher
from robotwin.round4.train import physical_metrics

ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT/'robotwin/round4'
OUT = ROUND/'split_path_confirmation'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():
        raise RuntimeError('Existing confirmation results must not be overwritten')
    status=dict(pid=os.getpid(),physical_gpu=0,no_optimizer_updates=True)

    def update(**kw):
        status.update(kw,updated_at=time.time())
        atomic_json(OUT/'status.json',status)
        print(json.dumps(status),flush=True)

    try:
        update(phase='loading')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.16)
        torch.manual_seed(260926)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        model=GroundedTeacher(cfg,auxiliary=True)
        ckpt=ROUND/'runs/future_aux/step_000200/adapter.pt'
        saved=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
        assert saved['format']=='robotwin_future_aux36_adapter_v1'
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert list(params)==list(saved['adapter'])
        with torch.no_grad():
            for n,p in params.items():p.copy_(saved['adapter'][n])
        del saved,params
        model=model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks=False;model.flow.checkpoint_blocks=False
        versions={n:p._version for n,p in model.named_parameters()}
        val=FutureActionDataset(cfg['dataset_root'],model.policy.statistics,'val')
        previous=json.loads((ROUND/'assessment/future_aux/samples.json').read_text())
        excluded=set(previous['episodes'])|set(previous['excluded_tuning_episodes'])
        rng=np.random.default_rng(260927);indices=[];episodes=set()
        while len(indices)<32:
            i=int(rng.integers(len(val)))
            e=val.selected[int(np.searchsorted(val.offsets,i,side='right')-1)]['episode_index']
            if e in excluded or e in episodes:continue
            episodes.add(e);indices.append(i)
        assert episodes.isdisjoint(excluded)
        manifest=dict(checkpoint=str(ckpt),candidate={'plan_multiplier':0,'role_multiplier':3},
            control={'plan_multiplier':1,'role_multiplier':1},indices=indices,episodes=sorted(episodes),
            excluded_episodes=sorted(excluded),selection_seed=260927,noise_seeds=[3800,4800],
            screen='All physical errors <=1.05x base AND improvement-over-control CI lower >0 AND future-benefit CI lower >0',
            no_optimizer_updates=True)
        atomic_json(OUT/'manifest.json',manifest)
        batches=[default_collate([val[i]]) for i in indices];prepared=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,b in enumerate(batches):
                prepared.append(model.prepare(b));update(phase='prepare',completed=i+1,total=32)
        donors=[i for i,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors)>=2
        report=dict(manifest=manifest,rows=[])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,(b,prep) in enumerate(zip(batches,prepared)):
                full=model.conditions(b,prep)
                j=next(j for j in donors[i%len(donors):]+donors[:i%len(donors)] if j!=i)
                wrong=model.conditions(b,prep,donor=(batches[j],prepared[j]))
                for seed in [3800,4800]:
                    torch.manual_seed(seed+i);noise=torch.randn(1,50,32,device='cuda')
                    metrics={}
                    for name,cond,gp,gr in [('base',full,0,0),('plan_only',full,1,0),
                                           ('control',full,1,1),('control_wrong',wrong,1,1),
                                           ('candidate',full,0,3),('candidate_wrong',wrong,0,3)]:
                        pred=sample(model.flow,cond,noise,gp,gr)
                        assert bool(torch.isfinite(pred).all())
                        metrics[name]=physical_metrics(pred,b,model.policy.statistics)
                    report['rows'].append(dict(sample=i,episode=int(b['episode_id'][0]),seed=seed+i,metrics=metrics))
                    atomic_json(OUT/'partial.json',report)
                    update(phase='independent_confirmation',completed=len(report['rows']),total=64)
        keys=['position_cm','rotation_deg','gripper_mae']
        vals={v:np.array([[np.mean([r['metrics'][v][k] for r in report['rows'] if r['sample']==i]) for k in keys] for i in range(32)]) for v in report['rows'][0]['metrics']}
        means={v:dict(zip(keys,a.mean(0).tolist())) for v,a in vals.items()}
        scale=vals['base'].mean(0).clip(1e-10)
        draws=np.random.default_rng(29326).integers(0,32,size=(4000,32))
        summary={}
        for name in ['control','candidate']:
            delta=((vals[name+'_wrong']-vals[name])/scale).mean(1)
            summary[name]=dict(full_to_base_ratio=dict(zip(keys,(vals[name].mean(0)/scale).tolist())),
                              future_benefit_mean=float(delta.mean()),
                              future_benefit_bootstrap_95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
        gain=((vals['control']-vals['candidate'])/scale).mean(1)
        summary['candidate_vs_control']=dict(mean=float(gain.mean()),bootstrap_95=np.quantile(gain[draws].mean(1),[.025,.975]).tolist())
        screen=bool(all(v<=1.05 for v in summary['candidate']['full_to_base_ratio'].values())
                    and summary['candidate']['future_benefit_bootstrap_95'][0]>0
                    and summary['candidate_vs_control']['bootstrap_95'][0]>0)
        assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
        report.update(completed=True,mean=means,summary=summary,exploratory_screen_passed=screen,
                      all_parameters_unchanged=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(OUT/'results.json',report)
        lines=['# 分离门控候选的独立轨迹复核','','固定候选Plan0/Role3，不再筛选倍率。32条新轨迹×2个噪声；排除之前32条诊断轨迹及16条调参轨迹。',
               '按轨迹平均两个噪声后bootstrap；无权重更新，无闭环测试。','',
               '|条件|位置cm|旋转°|夹爪MAE|','|---|---:|---:|---:|']
        for name,m in means.items():lines.append(f"|{name}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
        lines+=['',f"候选相对原门控的标准化收益95%区间：{summary['candidate_vs_control']['bootstrap_95']}。",'',
                f"候选正确未来相对错误未来的收益95%区间：{summary['candidate']['future_benefit_bootstrap_95']}。",'',
                f'探索筛查通过：{screen}。判据预先写入manifest.json。',
                '原始SFT可能见过这些示范；本检查仍不代表闭环泛化能力。未启动学生训练或在线评测。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',completed=64,total=64,exploratory_screen_passed=screen)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
