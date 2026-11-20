"""Read-only auxiliary-head audit on the existing 32 held-out episodes.

No optimizer, student training, online rollout, or checkpoint writes.
"""
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

ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / 'robotwin/round4'
OUT = ROUND / 'auxiliary_audit'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise RuntimeError('Audit results already exist; do not overwrite')
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
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

        def load_adapter(path, expected_format):
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            assert saved['format'] == expected_format
            if expected_format == 'robotwin_future_aux36_adapter_v1':
                assert list(saved['adapter']) == trainable_names
            named = dict(model.named_parameters())
            with torch.no_grad():
                for n, v in saved['adapter'].items():
                    named[n].copy_(v)

        source = ROOT / 'robotwin/round3/runs/teacher_terminal_hold36_20260926/step_000200/adapter.pt'
        load_adapter(source, 'robotwin_terminal_hold36_adapter_v1')
        model = model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks = False
        model.flow.checkpoint_blocks = False
        val = FutureActionDataset(cfg['dataset_root'], model.policy.statistics, 'val')
        manifest = json.loads((ROUND / 'assessment/future_aux/samples.json').read_text())
        batches = [default_collate([val[i]]) for i in manifest['indices']]
        prepared = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i, b in enumerate(batches):
                prepared.append(model.prepare(b))
                update(phase='prepare', completed=i+1, total=len(batches))
        donors = [i for i, b in enumerate(batches)
                  if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors) >= 2
        report = dict(scope='Same 32 held-out episodes as paired action assessment; auxiliary losses only',
                      no_optimizer_updates=True, samples=manifest, checkpoints={},
                      limitation='Training auxiliary loss, not physical rollout success; checkpoints evaluated on the same episodes.')
        for step in [0, 50, 100, 200]:
            if step:
                ckpt = ROUND / f'runs/future_aux/step_{step:06d}/adapter.pt'
                assert (ckpt.parent / 'checkpoint_complete.json').exists()
                load_adapter(ckpt, 'robotwin_future_aux36_adapter_v1')
            rows = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for i, (b, prep) in enumerate(zip(batches, prepared)):
                    donor = next(j for j in donors[i % len(donors):] + donors[:i % len(donors)] if j != i)
                    assert int(b['episode_id'][0]) != int(batches[donor]['episode_id'][0])
                    _, full = model.conditions_encoded(b, prep)
                    _, wrong = model.conditions_encoded(b, prep, donor=(batches[donor], prepared[donor]))
                    row = dict(sample=i, episode=int(b['episode_id'][0]), metrics={})
                    for branch in ['all', 'local', 'goal']:
                        mask = b['future_action_valid'].clone()
                        if branch == 'local':
                            mask[:, 1:] = False
                        elif branch == 'goal':
                            mask[:, :1] = False
                        if not bool(mask.any()):
                            continue
                        masked = dict(b, future_action_valid=mask)
                        row['metrics'][branch] = dict(valid_targets=int(mask.sum()))
                        for name, enc in [('full', full), ('wrong_future', wrong)]:
                            losses = model.auxiliary_loss(masked, enc)
                            assert all(bool(torch.isfinite(v)) for v in losses.values())
                            row['metrics'][branch][name] = {k: float(v) for k, v in losses.items()}
                    rows.append(row)
                    update(phase='evaluate_auxiliary', checkpoint_step=step, completed=i+1, total=len(batches))
            summary = {}
            for branch in ['all', 'local', 'goal']:
                available = [r['metrics'][branch] for r in rows if branch in r['metrics']]
                full = np.array([r['full']['aux'] for r in available])
                wrong = np.array([r['wrong_future']['aux'] for r in available])
                delta = wrong - full
                draws = np.random.default_rng(29326).integers(0, len(delta), size=(4000, len(delta)))
                summary[branch] = dict(episodes=len(available), valid_targets=sum(r['valid_targets'] for r in available),
                    full_mean=float(full.mean()), wrong_future_mean=float(wrong.mean()),
                    future_benefit_mean=float(delta.mean()),
                    future_benefit_bootstrap_95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                    component_means={v: {k: float(np.mean([r[v][k] for r in available]))
                        for k in available[0][v]} for v in ['full', 'wrong_future']})
            report['checkpoints'][str(step)] = dict(rows=rows, summary=summary)
            atomic_json(OUT / 'partial.json', report)
        logs = [json.loads(s) for s in (ROUND / 'runs/future_aux/metrics.jsonl').read_text().splitlines()]
        train = [r for r in logs if r.get('split') == 'train']
        report['gradient_probes'] = [dict(step=r['step'], **r['aux_gradient_probe'])
                                     for r in train if r.get('aux_gradient_probe')]
        report['train_windows'] = [dict(start=a+1, end=a+50,
            auxiliary_mean=float(np.mean([r['auxiliary'] for r in train[a:a+50]])),
            flow_mean=float(np.mean([r['flow'] for r in train[a:a+50]])),
            lambda_mean=float(np.mean([r['lambda_aux'] for r in train[a:a+50]]))) for a in [0, 50, 100, 150]]
        assert all(p.grad is None for p in model.parameters())
        report['completed'] = True
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        atomic_json(OUT / 'results.json', report)
        lines = ['# 辅助头独立审计', '', '只读评估同一批32条验证轨迹，不更新模型。', '',
                 '|追加步数|分支|有效轨迹|正确未来辅助loss|错误未来辅助loss|正确未来收益95%区间|',
                 '|---:|---|---:|---:|---:|---|']
        for step, data in report['checkpoints'].items():
            for branch, s in data['summary'].items():
                lines.append(f"|{step}|{branch}|{s['episodes']}|{s['full_mean']:.6f}|{s['wrong_future_mean']:.6f}|{s['future_benefit_bootstrap_95']}|")
        lines += ['', '收益为错误未来loss减正确未来loss，正数有利于正确未来；按轨迹bootstrap，未做多重比较校正。',
                  '辅助loss下降不代表动作头利用未来，也不等于在线成功率提升。所有Stage II和在线评估保持原状态。']
        (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        update(phase='completed', checkpoint_step=200, completed=32, total=32)
    except BaseException as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
