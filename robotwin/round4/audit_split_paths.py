"""Read-only separation of Plan scaffold and Goal/Local-conditioned velocity.

The diagnostic does not alter production model classes or checkpoint weights.
"""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from goal_local.action import HierarchicalFlow
from robotwin.round2.train import atomic_json
from robotwin.round4.model import FutureActionDataset, GroundedTeacher
from robotwin.round4.train import physical_metrics
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid

ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / 'robotwin/round4'
OUT = ROUND / 'split_path_audit'
MODES = {'tied': (1, 1), 'role_only': (0, 1), 'role_only_3': (0, 3), 'plan1_role3': (1, 3)}


def velocity(flow, x, bucket, hz, emb, valid, cond, plan_multiplier, role_multiplier):
    ref = HierarchicalFlow.velocity(flow, x, bucket, hz, emb, valid, cond, bypass_roles=True)
    gate = flow.residual_gate.tanh()
    if plan_multiplier == 0 and role_multiplier == 0:
        return ref
    if role_multiplier:
        aug = HierarchicalFlow.velocity(flow, x, bucket, hz, emb, valid, cond, bypass_roles=False)
        # Equal gates follow the original arithmetic order exactly.
        result = ref + (gate * role_multiplier) * (aug - ref)
    else:
        result = ref
    if plan_multiplier != role_multiplier:
        off_cond = (*cond[:-1], torch.zeros_like(cond[-1]))
        off = HierarchicalFlow.velocity(flow, x, bucket, hz, emb, valid, off_cond, bypass_roles=False)
        result = result + (gate * (plan_multiplier-role_multiplier)) * (off-ref)
    return result


@torch.no_grad()
def sample(flow, cond, noise, gp, gr):
    hz = torch.full((len(noise),), 30., device=noise.device)
    emb = torch.ones(len(noise), dtype=torch.long, device=noise.device)
    _, valid, count = build_time_grid(flow.config.horizon_sec, hz, noise.shape[1])
    assert torch.all(count == 36)
    x = noise.clone() * valid[..., None]
    for t in range(10):
        bucket = torch.full((len(x),), t*100, device=x.device, dtype=torch.long)
        x = (x + velocity(flow, x, bucket, hz, emb, valid, cond, gp, gr)/10) * valid[..., None]
    return x[:, :36, :16].float()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():
        raise RuntimeError('Existing results must not be overwritten')
    status = dict(pid=os.getpid(), physical_gpu=0, no_optimizer_updates=True)

    def update(**kw):
        status.update(kw, updated_at=time.time())
        atomic_json(OUT/'status.json', status)
        print(json.dumps(status), flush=True)

    try:
        update(phase='loading')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.16)
        torch.manual_seed(260926)
        cfg = json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        model = GroundedTeacher(cfg, auxiliary=True)
        ckpt = ROUND/'runs/future_aux/step_000200/adapter.pt'
        saved = torch.load(ckpt, map_location='cpu', mmap=True, weights_only=False)
        assert saved['format'] == 'robotwin_future_aux36_adapter_v1'
        params = {n:p for n,p in model.named_parameters() if p.requires_grad}
        assert list(params) == list(saved['adapter'])
        with torch.no_grad():
            for n,p in params.items():
                p.copy_(saved['adapter'][n])
        del saved, params
        model = model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks = False
        model.flow.checkpoint_blocks = False
        frozen_versions = {n:p._version for n,p in model.named_parameters()}
        val = FutureActionDataset(cfg['dataset_root'], model.policy.statistics, 'val')
        samples = json.loads((ROUND/'assessment/future_aux/samples.json').read_text())
        prior = json.loads((ROUND/'assessment/future_aux/results.json').read_text())
        prior_rows = {r['sample']:r for r in prior['rows'] if r['noise_seed']==1800+r['sample']}
        batches = [default_collate([val[i]]) for i in samples['indices']]
        prepared = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i,b in enumerate(batches):
                prepared.append(model.prepare(b))
                update(phase='prepare', completed=i+1, total=32)
        donors = [i for i,b in enumerate(batches)
                  if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors)>=2
        report = dict(checkpoint=str(ckpt), gate=float(model.flow.residual_gate.tanh()), samples=samples,
                      modes=MODES, rows=[], no_optimizer_updates=True,
                      scope='32 fixed validation episodes, one noise seed, exploratory only',
                      equation='v_base + g_plan*(v_plan_off-v_base) + g_role*(v_plan_role-v_plan_off)')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i,(b,prep) in enumerate(zip(batches,prepared)):
                full = model.conditions(b,prep)
                donor = next(j for j in donors[i%len(donors):]+donors[:i%len(donors)] if j!=i)
                wrong = model.conditions(b,prep,donor=(batches[donor],prepared[donor]))
                torch.manual_seed(1800+i)
                noise = torch.randn(1,50,32,device='cuda')
                if i==0:
                    update(phase='compatibility_preflight')
                    original = model.flow.sample(full,noise)
                    tied = sample(model.flow,full,noise,1,1)
                    reference = model.flow.sample(full,noise,bypass=True)
                    zero = sample(model.flow,full,noise,0,0)
                    off_cond = (*full[:-1],torch.zeros_like(full[-1]))
                    plan = sample(model.flow,full,noise,1,0)
                    old_off = model.flow.sample(off_cond,noise)
                    hz=torch.tensor([30.],device='cuda');emb=torch.ones(1,dtype=torch.long,device='cuda')
                    _,valid,_=build_time_grid(1.2,hz,50)
                    dtype = str(velocity(model.flow,noise*valid[...,None],torch.zeros(1,dtype=torch.long,device='cuda'),hz,emb,valid,full,1,1).dtype)
                    checks=dict(tied_max_abs=float((original-tied).abs().max()),
                                zero_max_abs=float((reference-zero).abs().max()),
                                plan_only_max_abs=float((plan-old_off).abs().max()),velocity_dtype=dtype)
                    assert all(checks[k]==0 for k in ['tied_max_abs','zero_max_abs','plan_only_max_abs']),checks
                    atomic_json(OUT/'preflight.json',dict(passed=True,**checks))
                predictions = {'base':sample(model.flow,full,noise,0,0),
                               'plan_only':sample(model.flow,full,noise,1,0)}
                for name,(gp,gr) in MODES.items():
                    predictions[name] = sample(model.flow,full,noise,gp,gr)
                    predictions[name+'_wrong'] = sample(model.flow,wrong,noise,gp,gr)
                    update(phase='split_path_evaluation', sample=i+1, completed=i*4+list(MODES).index(name)+1, total=128)
                metrics = {name:physical_metrics(pred,b,model.policy.statistics) for name,pred in predictions.items()}
                for name,old_name in [('base','base'),('tied','full'),('tied_wrong','wrong_future'),('plan_only','roles_off')]:
                    assert all(abs(metrics[name][k]-prior_rows[i]['metrics'][old_name][k])<1e-6 for k in metrics[name]),(i,name)
                report['rows'].append(dict(sample=i,episode=int(b['episode_id'][0]),metrics=metrics,
                    wrong_action_relative={name:float((predictions[name]-predictions[name+'_wrong']).norm()/predictions[name].norm().clamp_min(1e-12)) for name in MODES}))
                atomic_json(OUT/'partial.json',report)
        keys=['position_cm','rotation_deg','gripper_mae']
        vals={v:np.array([[r['metrics'][v][k] for k in keys] for r in report['rows']]) for v in report['rows'][0]['metrics']}
        scale=vals['base'].mean(0).clip(1e-10)
        draws=np.random.default_rng(29326).integers(0,32,size=(4000,32))
        report['mean']={v:dict(zip(keys,a.mean(0).tolist())) for v,a in vals.items()}
        report['summary']={}
        for mode in MODES:
            delta=((vals[mode+'_wrong']-vals[mode])/scale).mean(1)
            benefit=((vals['tied']-vals[mode])/scale).mean(1)
            ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
            ratios=vals[mode].mean(0)/scale
            report['summary'][mode]=dict(full_to_base_ratio=dict(zip(keys,ratios.tolist())),
                future_benefit_mean=float(delta.mean()),future_benefit_bootstrap_95=ci,
                improvement_over_tied_mean=float(benefit.mean()),
                improvement_over_tied_bootstrap_95=np.quantile(benefit[draws].mean(1),[.025,.975]).tolist(),
                exploratory_screen=bool(all(ratios<=1.05) and ci[0]>0))
        assert all(p.grad is None and p._version==frozen_versions[n] for n,p in model.named_parameters())
        report.update(completed=True,all_parameters_unchanged=True,prior_metrics_reproduced=True,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(OUT/'results.json',report)
        lines=['# Plan与Goal/Local分离对照','','固定教师、32轨迹、一个固定噪声；只读、无训练，无在线评估。',
               '',report['equation'],'','|模式|位置cm|旋转°|夹爪MAE|','|---|---:|---:|---:|']
        for name,m in report['mean'].items():
            lines.append(f"|{name}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
        lines+=['','模式中的倍率相对于已有tanh门控；tied=Plan1/Role1，role_only=Plan0/Role1，role_only_3=Plan0/Role3，plan1_role3=Plan1/Role3。','']
        for name,s in report['summary'].items():
            lines += [f"{name}：正确未来收益95%区间{s['future_benefit_bootstrap_95']}；探索筛查通过={s['exploratory_screen']}。",'']
        lines+=['已有tied/base/plan_only/wrong逐样本重现原评估，最大容差1e-6；初始等价性见preflight.json。',
                '本结果用于诊断现有权重，不能代替分离门控重新训练、独立轨迹验证或闭环成功率。多重比较未经校正。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',completed=128,total=128)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
