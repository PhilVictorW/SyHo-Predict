"""Summarize completed, read-only model probes; does not load models."""
import json
from pathlib import Path
from statistics import mean, median

root = Path(__file__).resolve().parent
d = json.loads((root / 'results.json').read_text())
assert d['completed'] and len(d['phases']['original']) == 32
assert len(d['phases']['teacher1000']) == 32
assert len(d['phases']['initial_gradient_probes']) == 8
summary = dict(completed=True, peak_allocated_gib=d['peak_allocated_gib'])
for name in ('original', 'teacher1000'):
    rows = d['phases'][name]
    summary[name] = {
        variant: {metric: dict(mean=mean(r['metrics'][variant][metric] for r in rows),
                               median=median(r['metrics'][variant][metric] for r in rows))
                  for metric in rows[0]['metrics'][variant]}
        for variant in rows[0]['metrics']}
rows = [r for r in d['phases']['teacher1000'] if not r['donor_same_episode']]
summary['teacher_donor_control'] = dict(
    samples=len(rows),
    mean_wrong_future_relative_l2=mean(r['wrong_future_effect']['relative_l2'] for r in rows),
    mean_roles_off_relative_l2=mean(r['roles_off_effect']['relative_l2'] for r in rows))
rows = d['phases']['initial_gradient_probes']
summary['gradient_probes'] = {
    name: dict(median_distill_to_action_ratio=median(r['parameters'][name]['distill_to_action_norm_ratio'] for r in rows),
               median_cosine=median(r['parameters'][name]['action_vs_distill_cosine'] for r in rows),
               negative_cosine_samples=sum(r['parameters'][name]['action_vs_distill_cosine'] < 0 for r in rows),
               samples=len(rows))
    for name in rows[0]['parameters']}
summary['gradient_scope'] = d['gradient_scope']
summary['first_two_blocks'] = d['block_trace'][:2]
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['teacher_donor_control']))
print('Wrote', root / 'summary.json')
