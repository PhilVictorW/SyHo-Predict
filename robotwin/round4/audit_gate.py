"""Read-only gate sweep: does stronger future injection improve actions?"""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from robotwin.round2.train import atomic_json
from robotwin.round4.model import FutureActionDataset, GroundedTeacher
from robotwin.round4.train import physical_metrics

ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / 'robotwin/round4'
OUT = ROUND / 'gate_audit'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise RuntimeError('Existing results must not be overwritten')
    status = dict(pid=os.getpid(), physical_gpu=0, no_optimizer_updates=True)

    def update(**kw):
        status.update(kw, updated_at=time.time())
        atomic_json(OUT / 'status.json', status)
        print(json.dumps(status), flush=True)

    try:
        update(phase='loading')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.16)
        torch.manual_seed(260926)
        cfg = json.loads((ROOT / 'robotwin/configs/stage_I.json').read_text())
        model = GroundedTeacher(cfg, auxiliary=True)
        ckpt = ROUND / 'runs/future_aux/step_000200/adapter.pt'
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
        original_gate = model.flow.residual_gate.detach().clone()
        gate = float(original_gate.tanh())
        val = FutureActionDataset(cfg['dataset_root'], model.policy.statistics, 'val')
        samples = json.loads((ROUND / 'assessment/future_aux/samples.json').read_text())
        # First 16 previously selected episodes, without selection by performance.
        batches = [default_collate([val[i]]) for i in samples['indices'][:16]]
        prepared = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i,b in enumerate(batches):
                prepared.append(model.prepare(b))
                update(phase='prepare', completed=i+1, total=16)
        donors = [i for i,b in enumerate(batches)
                  if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors) >= 2
        report = dict(checkpoint=str(ckpt), original_gate=gate, rows=[], no_optimizer_updates=True,
                      scope='Exploratory first 16 held-out episodes, one matched noise seed; no online rollout',
                      indices=samples['indices'][:16], donors='Fully valid donor, recipient validity unchanged')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i,(b,prep) in enumerate(zip(batches,prepared)):
                donor = next(j for j in donors[i % len(donors):] + donors[:i % len(donors)] if j != i)
                full = model.conditions(b,prep)
                wrong = model.conditions(b,prep,donor=(batches[donor],prepared[donor]))
                off = (*full[:-1],torch.zeros_like(full[-1]))
                torch.manual_seed(1800+i)
                noise = torch.randn(1,50,32,device='cuda')
                base = physical_metrics(model.flow.sample(full,noise,bypass=True),b,model.policy.statistics)
                for multiplier in [1,3,10]:
                    model.flow.residual_gate.copy_(torch.atanh(original_gate.tanh()*multiplier))
                    metrics = {'base':base}
                    for variant,cond in [('full',full),('wrong_future',wrong),('roles_off',off)]:
                        pred = model.flow.sample(cond,noise)
                        assert bool(torch.isfinite(pred).all())
                        metrics[variant] = physical_metrics(pred,b,model.policy.statistics)
                    report['rows'].append(dict(sample=i,episode=int(b['episode_id'][0]),multiplier=multiplier,
                                               gate=gate*multiplier,metrics=metrics))
                    atomic_json(OUT / 'partial.json',report)
                    update(phase='gate_sweep',completed=len(report['rows']),total=48)
            model.flow.residual_gate.copy_(original_gate)
        summary = {}
        keys = ['position_cm','rotation_deg','gripper_mae']
        for mult in [1,3,10]:
            rows = [r for r in report['rows'] if r['multiplier']==mult]
            values = {v:np.array([[r['metrics'][v][k] for k in keys] for r in rows])
                      for v in ['base','full','wrong_future','roles_off']}
            scale = values['base'].mean(0).clip(1e-10)
            delta = ((values['wrong_future']-values['full'])/scale).mean(1)
            draws = np.random.default_rng(29326).integers(0,len(rows),size=(4000,len(rows)))
            summary[str(mult)] = dict(mean={v:dict(zip(keys,a.mean(0).tolist())) for v,a in values.items()},
                full_to_base_ratio=dict(zip(keys,(values['full'].mean(0)/scale).tolist())),
                future_benefit_mean=float(delta.mean()),
                future_benefit_bootstrap_95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
        report.update(summary=summary,completed=True,original_gate_restored=True,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(OUT / 'results.json',report)
        lines = ['# 动作端门控只读对照','','固定200步教师；前16条验证轨迹、同一噪声，门控仅在内存中改动，完成后还原。','',
                 '|门控倍数|条件|位置cm|旋转°|夹爪MAE|','|---:|---|---:|---:|---:|']
        for mult,s in summary.items():
            for variant,m in s['mean'].items():
                lines.append(f"|{mult}|{variant}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
            lines += ['',f"{mult}倍门控：正确未来收益95%区间 {s['future_benefit_bootstrap_95']}。",'']
        lines += ['此实验只能诊断现有权重对门控的响应，不能替代重新训练后的架构比较或在线成功率。',
                  '门控同时放大Plan路径和Goal/Local增量；roles_off保留Plan路径，因此两者不能混为一谈。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',completed=48,total=48)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
