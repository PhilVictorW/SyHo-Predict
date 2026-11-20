"""Held-out endpoint confirmation of the 200-step current-only action adapter."""
import fcntl
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.round11.action_adapter import CurrentDataset
from robotwin.round6.model import RetainedTeacher
from robotwin.round6.train import atomic_json, physical_metrics
from robotwin.round7.model import CurrentQueryStudent, OBSERVATION_KEYS

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'robotwin/round11/action_adapter_200'
OUT = ROOT/'robotwin/round11/adapter_confirmation_200'
KEYS = ['position_cm', 'rotation_deg', 'gripper_mae']
VARIANTS = ['initial', 'adapter200', 'goal_off', 'local_off', 'roles_off']


def summarize(rows):
    groups = {c:[r for r in rows if r['cohort'] == c] for c in ['fresh', 'tail']}
    for remain in [1, 8, 24, 36]:
        groups[f'tail_remaining_{remain}'] = [r for r in groups['tail'] if r['remaining'] == remain]
    output = {}
    for name, selected in groups.items():
        episodes = sorted({r['episode'] for r in selected})
        values = {v:np.array([[np.mean([r['metrics'][v][k] for r in selected if r['episode'] == e])
                   for k in KEYS] for e in episodes]) for v in VARIANTS}
        draws = np.random.default_rng(281204).integers(0, len(episodes), (4000, len(episodes)))
        comparisons = {}
        for a,b in [('adapter200','initial'), ('goal_off','adapter200'),
                    ('local_off','adapter200'), ('roles_off','adapter200')]:
            scale = np.maximum(values[b].mean(0), 1e-10)
            ratios = values[a].mean(0)/scale
            delta = ((values[a]-values[b])/scale).mean(1)
            ci = np.quantile(delta[draws].mean(1), [.025,.975]).tolist()
            comparisons[a+'_vs_'+b] = dict(ratios=dict(zip(KEYS, ratios.tolist())),
                normalized_excess=float(delta.mean()), bootstrap_95=ci,
                nonregression_passed=bool(max(ratios) <= 1.05 and ci[1] <= .05),
                improvement_supported=bool(ci[1] < 0 and max(ratios) <= 1.05))
        output[name] = dict(episodes=len(episodes), pairs=len(selected),
            means={v:dict(zip(KEYS, x.mean(0).tolist())) for v,x in values.items()}, comparisons=comparisons,
            normalized_prediction_drift={v:float(np.mean([r['drift_mse'][v] for r in selected]))
                                         for v in VARIANTS if v != 'initial'})
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    assert not (OUT/'results.json').exists(), 'Completed result is immutable'
    state = dict(pid=os.getpid(), phase='loading', physical_gpu=2, completed=0, total=128,
                 started_at=time.time(), optimizer_updates=0, online_evaluation=False)
    def update(**kw):
        state.update(kw)
        state['updated_at'] = time.time()
        atomic_json(OUT/'status.json', state)
        print(json.dumps(state), flush=True)
    update()
    try:
        assert json.loads((RUN/'status.json').read_text())['phase'] == 'completed'
        assert json.loads((RUN/'frozen_audit.json').read_text())['passed']
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(281201)
        cfg = json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        recipe = json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        teacher = RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'], recipe['teacher_checkpoint']]:
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            assert saved['manifest']['base_config'] == cfg
            named = dict(teacher.named_parameters())
            with torch.no_grad():
                for n,v in saved['adapter'].items(): named[n].copy_(v)
            del named, saved
        teacher = teacher.cuda().eval().requires_grad_(False)
        teacher.policy.checkpoint_blocks = teacher.flow.checkpoint_blocks = False
        student = CurrentQueryStudent(teacher).cuda().eval().requires_grad_(False)
        init = torch.load(recipe['initializers'], map_location='cpu', weights_only=False)
        student.set_queries(*init['queries']['fixed_queries'])
        student.gate.load_state_dict(init['gate'])
        assert not student.gate.scales.any()
        del init
        named = dict(teacher.named_parameters())
        weights, checkpoints = {}, {}
        for variant,step in [('initial',0), ('adapter200',200)]:
            path = RUN/f'step_{step:06d}/adapter.pt'
            assert (path.parent/'checkpoint_complete.json').exists()
            payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            assert payload['format'] == 'robotwin_current_action_adapter_v1' and payload['step'] == step
            assert payload['manifest']['base_config'] == cfg and payload['manifest']['recipe'] == recipe
            assert set(payload['adapter']) == set(payload['manifest']['trainable_names'])
            assert all(n.startswith(('policy.flow.goal_', 'policy.flow.local_')) for n in payload['adapter'])
            weights[variant] = {n:v.to(device='cuda', dtype=named[n].dtype).clone() for n,v in payload['adapter'].items()}
            checkpoints[variant] = dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            del payload
        assert weights['initial'].keys() == weights['adapter200'].keys()
        assert all(torch.equal(named[n], v) for n,v in weights['initial'].items())
        immutable = {n:p._version for n,p in teacher.named_parameters() if n not in weights['initial']}
        student_versions = {n:p._version for n,p in student.named_parameters()}
        def select(variant):
            for n,v in weights[variant].items(): named[n].copy_(v)
        def audit():
            assert all(p._version == immutable[n] and p.grad is None
                       for n,p in teacher.named_parameters() if n in immutable)
            assert all(p._version == student_versions[n] and p.grad is None for n,p in student.named_parameters())
            assert not any(p.requires_grad for p in teacher.parameters())
        def forbidden(*a, **kw): raise AssertionError('Deployed student tried to read true future')
        val = CurrentDataset(cfg['dataset_root'], teacher.policy.statistics, 'val')
        train = CurrentDataset(cfg['dataset_root'], teacher.policy.statistics, 'train')
        by_id = {e['episode_index']:e for e in val.selected}
        assert set(by_id).isdisjoint(e['episode_index'] for e in train.selected)
        del train
        prior = json.loads((ROOT/'robotwin/round11/confirmation_500/manifest.json').read_text())
        excluded = set(prior['excluded_prior_episodes'])
        for cohort in prior['cohorts'].values(): excluded.update(x['episode'] for x in cohort['anchors'])
        for row in json.loads((RUN/'validation_000200.json').read_text())['rows']: excluded.add(row['episode'])
        available = sorted(set(by_id)-excluded)
        assert len(available) >= 40
        rng = np.random.default_rng(281202)
        fresh_ids = [int(x) for x in rng.choice(available, 32, replace=False)]
        fresh = [dict(episode=e, t=int(rng.integers(by_id[e]['length']))) for e in fresh_ids]
        eligible = [e for e in available if e not in fresh_ids and by_id[e]['length'] >= 100]
        tail_ids = [int(x) for x in np.random.default_rng(281203).choice(eligible, 8, replace=False)]
        tails = [dict(episode=e, t=by_id[e]['length']-n, remaining=n) for e in tail_ids for n in [1,8,24,36]]
        cohorts = dict(fresh=dict(anchors=fresh, noise_starts=[31800,32800]),
                       tail=dict(anchors=tails, noise_starts=[33800,34800]))
        manifest = dict(primary='adapter200 vs fixed-query compatible initialization on 32 fresh episodes',
            endpoint=200, checkpoint_selection='fixed final endpoint; no best-checkpoint selection',
            variants=VARIANTS, checkpoints=checkpoints, source_recipe=recipe, cohorts=cohorts,
            excluded_prior_episodes=sorted(excluded), train_validation_episode_disjoint=True,
            bootstrap='4000 resamples of independent episodes; average noise and anchors within episode',
            acceptance='fresh improvement upper95<0 and no metric>1.05; fresh/tail/each tail subgroup nonregression upper95<=.05 and no metric>1.05',
            ablations='trained200 fixed; independently disable Goal, Local, or both injection flags; diagnostic only',
            physical_gpu=2, model_inputs=list(OBSERVATION_KEYS), action_horizon=36,
            limitations=['Offline action error is not closed-loop success', 'Original SFT may have seen these episodes',
                        'Compatible architecture retains original LaWM and queries; this is not a completed StageII distillation experiment'],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if (OUT/'manifest.json').exists():
            assert json.loads((OUT/'manifest.json').read_text()) == manifest, 'Resume manifest differs'
        else: atomic_json(OUT/'manifest.json', manifest)
        report = json.loads((OUT/'partial.json').read_text()) if (OUT/'partial.json').exists() else dict(manifest=manifest, rows=[])
        assert report['manifest'] == manifest
        report.update(completed=False, preflight_max_abs=0.)
        completed_keys = {(r['cohort'], r['sample'], r['noise_seed']) for r in report['rows']}
        assert len(completed_keys) == len(report['rows'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False), \
             patch.object(teacher.policy, 'encode_future_images', side_effect=forbidden), \
             patch.object(teacher.policy.future, 'forward', side_effect=forbidden):
            update(phase='preflight')
            for variant,step in [('initial',0), ('adapter200',200)]:
                select(variant)
                recorded = json.loads((RUN/f'validation_{step:06d}.json').read_text())
                for i in [0,15]:
                    row = recorded['rows'][i]
                    b = default_collate([val.sample(by_id[row['episode']], row['t'])])
                    cond = student.conditions(student.prepare({k:b[k] for k in OBSERVATION_KEYS}))
                    torch.manual_seed(800+i)
                    pred = teacher.flow.sample(cond, torch.randn(1,50,32,device='cuda'))
                    metric = physical_metrics(pred, b, teacher.policy.statistics)
                    error = max(abs(metric[k]-row['metrics'][k]) for k in KEYS)
                    report['preflight_max_abs'] = max(report['preflight_max_abs'], error)
                    assert error < 1e-6, (variant, i, error)
            atomic_json(OUT/'preflight.json', dict(passed=True, reproduction_max_abs=report['preflight_max_abs'],
                        posterior_access_guard=True, initial_weights_match_recipe=True))
            for cohort,spec in cohorts.items():
                for i,anchor in enumerate(spec['anchors']):
                    seeds = [start+i for start in spec['noise_starts']]
                    if all((cohort,i,s) in completed_keys for s in seeds): continue
                    update(phase='decoding', cohort=cohort, anchor=i+1, anchors_total=32, completed=len(report['rows']))
                    b = default_collate([val.sample(by_id[anchor['episode']], anchor['t'])])
                    if cohort == 'tail': assert int(b['remaining_actions'][0]) == anchor['remaining']
                    prep = student.prepare({k:b[k] for k in OBSERVATION_KEYS})
                    predictions = {s:{} for s in seeds}
                    for variant in ['initial', 'adapter200']:
                        select(variant)
                        cond = student.conditions(prep)
                        for seed in seeds:
                            if (cohort,i,seed) in completed_keys: continue
                            update(phase='sampling', variant=variant, noise_seed=seed)
                            torch.manual_seed(seed)
                            noise = torch.randn(1,50,32,device='cuda')
                            predictions[seed][variant] = teacher.flow.sample(cond, noise)
                            if variant == 'adapter200':
                                for ablation,flag in [('goal_off',[False,True]), ('local_off',[True,False]), ('roles_off',[False,False])]:
                                    ablated = (*cond[:-1], torch.tensor([flag], dtype=torch.bool, device='cuda'))
                                    predictions[seed][ablation] = teacher.flow.sample(ablated, noise)
                    for seed in seeds:
                        if (cohort,i,seed) in completed_keys: continue
                        pred = predictions[seed]
                        metrics = {v:physical_metrics(p,b,teacher.policy.statistics) for v,p in pred.items()}
                        assert all(np.isfinite(x) for m in metrics.values() for x in m.values())
                        report['rows'].append(dict(cohort=cohort, sample=i, episode=anchor['episode'], t=anchor['t'],
                            task=by_id[anchor['episode']]['tasks'], remaining=int(b['remaining_actions'][0]), noise_seed=seed,
                            metrics=metrics, drift_mse={v:float((p-pred['initial']).square().mean()) for v,p in pred.items() if v!='initial'}))
                        completed_keys.add((cohort,i,seed))
                        atomic_json(OUT/'partial.json', report)
                        update(completed=len(report['rows']))
                    audit()
            select('initial')
            assert all(torch.equal(named[n],v) for n,v in weights['initial'].items())
            audit()
        assert len(report['rows']) == 128
        result = summarize(report['rows'])
        primary = result['fresh']['comparisons']['adapter200_vs_initial']
        safe = all(g['comparisons']['adapter200_vs_initial']['nonregression_passed'] for g in result.values())
        report.update(completed=True, summary=result, primary_improvement_supported=primary['improvement_supported'],
            all_nonregression_passed=safe, online_eligible=bool(primary['improvement_supported'] and safe),
            future_access_guard_anchors=64, frozen_audit=dict(passed=True, restored_initial_weights=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(OUT/'results.json', report)
        lines = ['200-step action adapter independent confirmation',
                 f'Online eligible: {report["online_eligible"]}', 'Offline errors; not a closed-loop success rate.']
        for name,g in result.items():
            lines += ['', name+': '+str(g['episodes'])+' independent episodes',
                      json.dumps(g['comparisons'], indent=2)]
        (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        update(phase='completed', completed=128, online_eligible=report['online_eligible'],
               primary_improvement_supported=primary['improvement_supported'], all_nonregression_passed=safe)
    except BaseException as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__': main()
