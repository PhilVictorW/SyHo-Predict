"""Prespecified offline confirmation of the completed query-only pilot."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate

from robotwin.data import CAMERAS, read_frames
from robotwin.round6.model import RetainedTeacher, FutureActionDataset
from robotwin.round6.train import atomic_json, physical_metrics
from robotwin.round7.model import CurrentQueryStudent, OBSERVATION_KEYS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'robotwin/round7/postpilot_eval'
PILOT = ROOT / 'robotwin/round7/query_pilot'
KEYS = ['position_cm', 'rotation_deg', 'gripper_mae']
STEPS = [0, 50, 100, 150, 200]


def summarize(rows):
    summaries = {}
    groups = {'fresh': [r for r in rows if r['cohort'] == 'fresh'],
              'tail': [r for r in rows if r['cohort'] == 'tail']}
    for remain in [1, 8, 24, 36]:
        groups[f'tail_remaining_{remain}'] = [r for r in groups['tail'] if r['remaining'] == remain]
    for name, group in groups.items():
        episodes = sorted({r['episode'] for r in group})
        variants = [f'step_{s:06d}' for s in STEPS] + ['base']
        values = {v: np.array([[np.mean([r['metrics'][v][k] for r in group if r['episode'] == e])
                                for k in KEYS] for e in episodes]) for v in variants}
        initial = values['step_000000']; base = values['base']
        scale = np.maximum(base.mean(0), 1e-10)
        draws = np.random.default_rng(280908).integers(0, len(episodes), (4000, len(episodes)))
        comparisons = {}
        for variant in variants[:-1]:
            value = values[variant]
            excess = ((value-initial)/scale).mean(1)
            ci = np.quantile(excess[draws].mean(1), [.025, .975]).tolist()
            ri = value.mean(0)/np.maximum(initial.mean(0), 1e-10)
            rb = value.mean(0)/scale
            comparisons[variant] = dict(ratio_to_initial=dict(zip(KEYS, ri.tolist())),
                ratio_to_base=dict(zip(KEYS, rb.tolist())), normalized_excess_vs_initial=float(excess.mean()),
                bootstrap_95=ci, screen_passed=bool(max(ri) <= 1.05 and max(rb) <= 1.05 and ci[1] <= .05))
        summaries[name] = dict(episodes=len(episodes), pairs=len(group),
            means={v: dict(zip(KEYS, x.mean(0).tolist())) for v, x in values.items()}, comparisons=comparisons)
    return summaries


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Do not overwrite an existing confirmation experiment')
    status = dict(pid=os.getpid(), phase='loading', physical_gpu=0, completed=0, total=128,
                  started_at=time.time(), online_evaluation=False, optimizer_updates=0)
    def update(**kwargs):
        status.update(kwargs); status['updated_at'] = time.time(); atomic_json(OUT/'status.json', status)
    update()
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        assert json.loads((PILOT/'status.json').read_text())['phase'] == 'completed_query_pilot'
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.25); torch.manual_seed(280906)
        cfg = json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        recipe = json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        teacher = RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'], recipe['teacher_checkpoint']]:
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            assert saved['manifest']['base_config'] == cfg
            named = dict(teacher.named_parameters())
            with torch.no_grad():
                for n, v in saved['adapter'].items(): named[n].copy_(v)
            del saved, named
        teacher = teacher.cuda().eval().requires_grad_(False)
        teacher.policy.checkpoint_blocks = False; teacher.flow.checkpoint_blocks = False
        frozen = {n: p._version for n, p in teacher.named_parameters()}
        student = CurrentQueryStudent(teacher).cuda().eval().requires_grad_(False)
        init = torch.load(recipe['initializers'], map_location='cpu', weights_only=False)
        student.gate.load_state_dict(init['gate']); assert not student.gate.scales.any()
        gate_versions = {n: p._version for n, p in student.gate.named_parameters()}
        queries = {}
        for step in STEPS:
            payload = torch.load(PILOT/f'step_{step:06d}/student.pt', map_location='cpu', weights_only=False)
            assert payload['step'] == step and payload['format'] == 'robotwin_current_query_pilot_v1'
            assert payload['manifest']['gate_frozen_at_zero'] and payload['manifest']['backbone_and_actor_frozen']
            queries[step] = tuple(payload['queries'][key][None] for key in ['q_goal', 'q_local'])
        assert all(torch.equal(x, y) for x, y in zip(queries[0], init['queries']['fixed_queries']))
        del init, payload
        # Decode only the three current images. Future/remaining metadata stays outside the model API.
        val = FutureActionDataset(cfg['dataset_root'], teacher.policy.statistics, 'val', decode=False)
        by_id = {e['episode_index']: e for e in val.selected}
        old = json.loads((ROOT/'robotwin/round7/audit/manifest.json').read_text())['cohorts']['fresh']
        excluded = set(old['episodes']) | set(old['excluded_episodes'])
        rng = np.random.default_rng(280906); fresh = []; chosen = set()
        assert len(set(by_id)-excluded) >= 40
        while len(fresh) < 32:
            index = int(rng.integers(len(val)))
            ei = int(np.searchsorted(val.offsets, index, side='right')-1); e = val.selected[ei]
            if e['episode_index'] in excluded | chosen: continue
            chosen.add(e['episode_index'])
            fresh.append(dict(episode=e['episode_index'], t=int(index-val.offsets[ei])))
        eligible = [e for e in val.selected if e['episode_index'] not in excluded | chosen and e['length'] >= 100]
        assert len(eligible) >= 8
        tail_episodes = [eligible[i] for i in np.random.default_rng(280907).choice(len(eligible), 8, replace=False)]
        tail = [dict(episode=e['episode_index'], t=e['length']-remaining, remaining=remaining)
                for e in tail_episodes for remaining in [1, 8, 24, 36]]
        cohorts = dict(fresh=dict(anchors=fresh, noise_starts=[11800, 12800]),
                       tail=dict(anchors=tail, noise_starts=[13800, 14800]))
        manifest = dict(primary_endpoint='step_000200', intermediate_checkpoints='exploratory; do not select best on confirmation set',
            excluded_prior_episodes=sorted(excluded), cohorts=cohorts, variants=STEPS,
            fresh_screen='each metric mean <=1.05 x initial AND base; upper95 normalized excess CI <=.05',
            tail_screen='same screen per remaining-length subgroup; 8 episode clusters, exploratory uncertainty',
            bootstrap='4000 resamples of episodes; average noises and tail anchors within episode',
            inputs=list(OBSERVATION_KEYS), role_valid='constant all true', action_horizon=36,
            source_recipe=recipe, query_checkpoints={str(s):str(PILOT/f'step_{s:06d}/student.pt') for s in STEPS},
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=['Offline action error, not closed-loop success rate', 'Original SFT may have seen these demonstrations',
                        'Compatible repaired model retains original LaWM and queries; not pure aba2 ablation'],
            next_phase='review evidence; no automatic training or online rollout')
        atomic_json(OUT/'manifest.json', manifest)

        def batch_for(anchor):
            e = by_id[anchor['episode']]; t = anchor['t']; batch = val.sample(e, t)
            images = []
            for cam in CAMERAS:
                pre = 'videos/'+cam
                path = val.root / val.info['video_path'].format(video_key=cam, chunk_index=e[pre+'/chunk_index'], file_index=e[pre+'/file_index'])
                timestamp = float(val.timestamps[e['dataset_from_index']+t]+e[pre+'/from_timestamp'])
                frame = read_frames(path, [timestamp])[0]
                images.append(torch.from_numpy(frame.copy()).permute(2, 0, 1))
            batch['current_main'] = images[0]; batch['current_wrist'] = torch.stack(images[1:])
            return default_collate([batch])

        def forbidden(*args, **kwargs): raise AssertionError('Student accessed posterior future path')
        def check_frozen():
            assert all(p._version == frozen[n] and p.grad is None for n, p in teacher.named_parameters())
            assert all(p._version == gate_versions[n] and p.grad is None for n, p in student.gate.named_parameters())

        report = dict(manifest=manifest, completed=False, rows=[], preflight_max_abs=0., future_access_guard_anchors=0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), \
                patch.object(teacher.policy, 'encode_future_images', side_effect=forbidden), \
                patch.object(teacher.policy.future, 'forward', side_effect=forbidden):
            # Reproduce stored outputs with the new current-frame-only decoder before judging new data.
            update(phase='preflight')
            for step in [0, 200]:
                previous = json.loads((PILOT/f'validation_{step:06d}.json').read_text())
                student.set_queries(*queries[step])
                for i in [0, 15]:
                    row = previous['rows'][i]; b = batch_for(row)
                    cond = student.conditions(student.prepare({k:b[k] for k in OBSERVATION_KEYS}))
                    torch.manual_seed(800+i); noise = torch.randn(1, 50, 32, device='cuda')
                    metrics = physical_metrics(teacher.flow.sample(cond, noise), b, teacher.policy.statistics)
                    error = max(abs(metrics[k]-row['metrics'][k]) for k in KEYS)
                    report['preflight_max_abs'] = max(report['preflight_max_abs'], error)
                    assert error < 1e-6, (step, i, error)
            atomic_json(OUT/'preflight.json', dict(passed=True, reproduction_max_abs=report['preflight_max_abs'],
                current_frames_only=True, initial_queries_exact=True, student_gate_zero=True, frozen_dit_eval=not teacher.flow.DiT.training))
            for cohort, specification in cohorts.items():
                for i, anchor in enumerate(specification['anchors']):
                    update(phase='decoding', cohort=cohort, anchor=i+1, anchors_total=32)
                    b = batch_for(anchor)
                    assert int(b['episode_id'][0]) == anchor['episode'] and int(b['t'][0]) == anchor['t']
                    if cohort == 'tail': assert int(b['remaining_actions'][0]) == anchor['remaining']
                    prepared = student.prepare({k:b[k] for k in OBSERVATION_KEYS})
                    conditions = {}
                    for step in STEPS:
                        student.set_queries(*queries[step]); conditions[f'step_{step:06d}'] = student.conditions(prepared)
                        assert conditions[f'step_{step:06d}'][-1].all()
                    for noise_start in specification['noise_starts']:
                        update(phase='sampling', noise_seed=noise_start+i)
                        torch.manual_seed(noise_start+i); noise = torch.randn(1, 50, 32, device='cuda')
                        metrics = {name:physical_metrics(teacher.flow.sample(cond, noise), b, teacher.policy.statistics)
                                   for name, cond in conditions.items()}
                        metrics['base'] = physical_metrics(teacher.flow.sample(conditions['step_000000'], noise, bypass=True), b, teacher.policy.statistics)
                        report['rows'].append(dict(cohort=cohort, sample=i, episode=anchor['episode'], t=anchor['t'],
                            remaining=int(b['remaining_actions'][0]), noise_seed=noise_start+i, metrics=metrics))
                        atomic_json(OUT/'partial.json', report); update(completed=len(report['rows']))
                    report['future_access_guard_anchors'] += 1; check_frozen()
        summary = summarize(report['rows'])
        fresh_passed = summary['fresh']['comparisons']['step_000200']['screen_passed']
        tail_passed = all(summary[f'tail_remaining_{r}']['comparisons']['step_000200']['screen_passed'] for r in [1, 8, 24, 36])
        report.update(completed=True, summary=summary, primary_fresh_passed=fresh_passed,
            primary_tail_passed=tail_passed, frozen_audit=dict(passed=True, tensors=len(frozen),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(OUT/'results.json', report)
        lines = ['# Query-only pilot: independent offline confirmation', '',
            'Step 200 is the prespecified primary endpoint. Intermediate checkpoints are exploratory.',
            'Each table averages within episode first; confidence intervals resample episodes.',
            'Lower error is better. These are offline action errors, not task success rates.']
        for name, result in summary.items():
            lines += ['', f'## {name}: {result["episodes"]} episodes / {result["pairs"]} noise pairs', '',
                      '|Checkpoint|Position cm|Rotation deg|Gripper MAE|', '|---|---:|---:|---:|']
            for variant, m in result['means'].items():
                lines.append(f'|{variant}|{m["position_cm"]:.6f}|{m["rotation_deg"]:.6f}|{m["gripper_mae"]:.7f}|')
            primary = result['comparisons']['step_000200']
            lines += ['', f'Primary ratios vs initialization: {primary["ratio_to_initial"]}',
                      f'Normalized excess 95% CI: {primary["bootstrap_95"]}; screen passed: {primary["screen_passed"]}.']
        lines += ['', f'Fresh screen: {fresh_passed}; all tail subgroup screens: {tail_passed}.',
                  'Passing a non-regression screen does not establish improvement. Tail cohort has only 8 independent episodes.',
                  'No full Stage II training or online rollout was started.']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed', primary_fresh_passed=fresh_passed, primary_tail_passed=tail_passed)
    except BaseException as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc()); raise


if __name__ == '__main__': main()
