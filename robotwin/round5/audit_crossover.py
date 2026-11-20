"""Read-only 2x2 swap of future resamplers and action-injection parameters."""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from robotwin.round2.train import atomic_json
from robotwin.round5.model import BalancedTeacher, FutureActionDataset
from robotwin.round5.train import physical_metrics

ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT/'robotwin/round5'
OUT = ROUND/'crossover_audit'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():
        raise RuntimeError('Existing audit results must not be overwritten')
    status=dict(pid=os.getpid(),physical_gpu=0,no_optimizer_updates=True)

    def update(**kw):
        status.update(kw,updated_at=time.time())
        atomic_json(OUT/'status.json',status)
        print(json.dumps(status),flush=True)

    try:
        update(phase='loading')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.18)
        torch.manual_seed(260929)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        model=BalancedTeacher(cfg)
        source=ROOT/'robotwin/round4/runs/future_aux/step_000200/adapter.pt'
        final=ROUND/'runs/equal/step_000200/adapter.pt'
        initial=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        trained=torch.load(final,map_location='cpu',mmap=True,weights_only=False)
        assert initial['format']=='robotwin_future_aux36_adapter_v1'
        assert trained['format']=='robotwin_balanced_centered36_adapter_v1'
        named=dict(model.named_parameters())
        assert list(trained['adapter'])==[n for n,p in named.items() if p.requires_grad]
        with torch.no_grad():
            for n,v in initial['adapter'].items():named[n].copy_(v)
        groups={
            'representation':[n for n in trained['adapter'] if n.startswith('policy.future.')],
            'injection':[n for n in trained['adapter'] if n.startswith('policy.flow.')],
        }
        assert groups['representation'] and groups['injection']
        assert all(n.startswith(('policy.flow.goal_','policy.flow.local_')) for n in groups['injection'])
        ignored=set(trained['adapter'])-set(groups['representation'])-set(groups['injection'])
        assert all(n.startswith(('local_head.','goal_head.')) for n in ignored)
        # Auxiliary heads are not used to execute actions and remain outside this swap.
        model=model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks=False;model.flow.checkpoint_blocks=False
        named=dict(model.named_parameters())
        banks={key:{n:state['adapter'][n].to('cuda').clone() for ns in groups.values() for n in ns}
               for key,state in [('initial',initial),('trained',trained)]}
        del initial,trained
        fixed={n:p._version for n,p in named.items() if n not in banks['initial']}

        def install(group,which):
            for n in groups[group]:named[n].copy_(banks[which][n])

        val=FutureActionDataset(cfg['dataset_root'],model.policy.statistics,'val')
        manifest=json.loads((ROUND/'assessment/equal/samples.json').read_text())
        prior=json.loads((ROUND/'assessment/equal/results.json').read_text())
        prior_rows={r['sample']:r for r in prior['rows'] if r['noise_seed']==5800+r['sample']}
        batches=[default_collate([val[i]]) for i in manifest['indices']]
        prepared=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for i,b in enumerate(batches):
                prepared.append(model.prepare(b));update(phase='prepare',completed=i+1,total=32)
        donors=[i for i,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors)>=2
        report=dict(initial_checkpoint=str(source),trained_checkpoint=str(final),samples=manifest,
                    groups=groups,ignored_auxiliary_head_parameters=sorted(ignored),rows=[],
                    scope='32 round5 assessment episodes x one matched noise; exploratory crossover, no training')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for i,(b,prep) in enumerate(zip(batches,prepared)):
                donor=next(j for j in donors[i%len(donors):]+donors[:i%len(donors)] if j!=i)
                torch.manual_seed(5800+i);noise=torch.randn(1,50,32,device='cuda')
                encoded={}
                for rep in ['initial','trained']:
                    install('representation',rep)
                    _,encoded[(rep,'full')]=model.conditions_encoded(b,prep)
                    _,encoded[(rep,'wrong')]=model.conditions_encoded(b,prep,donor=(batches[donor],prepared[donor]))
                valid=torch.stack((b['goal_frame_valid'].any(-1),b['local_valid']),-1).cuda()
                metrics={}
                for rep in ['initial','trained']:
                    for inj in ['initial','trained']:
                        install('injection',inj)
                        case=f'rep_{rep}__inj_{inj}'
                        for future in ['full','wrong']:
                            enc=encoded[(rep,future)]
                            cond=(*prep['base'],model.flow.goal_proj(enc['goal']),model.flow.local_proj(enc['local']),valid)
                            pred=model.flow.sample(cond,noise)
                            assert bool(torch.isfinite(pred).all())
                            metrics[case+'__'+future]=physical_metrics(pred,b,model.policy.statistics)
                        update(phase='crossover',sample=i+1,completed=i*4+len(metrics)//2,total=128)
                cond=(*prep['base'],model.flow.goal_proj(encoded[('trained','full')]['goal']),
                      model.flow.local_proj(encoded[('trained','full')]['local']),valid)
                metrics['base']=physical_metrics(model.flow.sample(cond,noise,bypass=True),b,model.policy.statistics)
                for actual,old in [('base','base'),('rep_trained__inj_trained__full','full'),('rep_trained__inj_trained__wrong','wrong_future')]:
                    assert all(abs(metrics[actual][k]-prior_rows[i]['metrics'][old][k])<1e-6 for k in metrics[actual]),(i,actual,metrics[actual],prior_rows[i]['metrics'][old])
                report['rows'].append(dict(sample=i,episode=int(b['episode_id'][0]),metrics=metrics))
                atomic_json(OUT/'partial.json',report)
            install('representation','initial');install('injection','initial')
        keys=['position_cm','rotation_deg','gripper_mae']
        vals={v:np.array([[r['metrics'][v][k] for k in keys] for r in report['rows']]) for v in report['rows'][0]['metrics']}
        means={v:dict(zip(keys,a.mean(0).tolist())) for v,a in vals.items()}
        scale=vals['base'].mean(0).clip(1e-10)
        draws=np.random.default_rng(5027).integers(0,32,(4000,32))

        def benefit(old,new):
            d=((vals[old]-vals[new])/scale).mean(1)
            return dict(mean=float(d.mean()),bootstrap_95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())

        summary={}
        for rep in ['initial','trained']:
            for inj in ['initial','trained']:
                name=f'rep_{rep}__inj_{inj}'
                summary[name]=dict(future_benefit=benefit(name+'__wrong',name+'__full'),
                    improvement_over_initial=benefit('rep_initial__inj_initial__full',name+'__full'),
                    full_to_base_ratio=dict(zip(keys,(vals[name+'__full'].mean(0)/scale).tolist())))
        comparisons={
            'representation_update_with_initial_injection':benefit('rep_initial__inj_initial__full','rep_trained__inj_initial__full'),
            'injection_update_with_initial_representation':benefit('rep_initial__inj_initial__full','rep_initial__inj_trained__full'),
            'injection_update_with_trained_representation':benefit('rep_trained__inj_initial__full','rep_trained__inj_trained__full'),
            'representation_update_with_trained_injection':benefit('rep_initial__inj_trained__full','rep_trained__inj_trained__full')}
        assert all(p.grad is None for p in model.parameters())
        assert all(named[n]._version==version for n,version in fixed.items())
        report.update(completed=True,mean=means,summary=summary,comparisons=comparisons,
                      frozen_backbone_unchanged=True,original_assessment_reproduced=True,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(OUT/'results.json',report)
        lines=['# 第五轮：表示与动作注入参数交叉替换','','固定等权组及其初始权重，32条相同轨迹、一个噪声；只读、不更新优化器。',
               'representation为future resampler；injection包括Goal/Local投影、注意力与逐层scale。原始骨干、Plan及总门控固定，辅助头不参与动作输出。','',
               '|组合|位置cm|旋转°|夹爪MAE|','|---|---:|---:|---:|']
        for name,m in means.items():lines.append(f"|{name}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
        lines+=['','正收益表示参数更新降低误差；区间为按轨迹bootstrap的探索性区间，未校正多重比较。','']
        for name,x in comparisons.items():lines.extend([f"{name}: {x['mean']:.6f}, 95% {x['bootstrap_95']}",''])
        lines+=['最终模型与base/wrong逐样本重现原评估（容差1e-6）。交叉组合可能改变模块间配合，只能诊断这组权重，不能证明某个模块在所有训练配置下均有害。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',completed=128,total=128)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
