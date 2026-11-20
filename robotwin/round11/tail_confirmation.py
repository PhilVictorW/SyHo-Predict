"""Three-way held-out confirmation for bounded current-only adapter continuations."""
import fcntl, hashlib, json, os, time, traceback
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
OUT = ROOT/'robotwin/round11/tail_confirmation_200'
BASE = ROOT/'robotwin/round11/action_adapter_200'
RUNS = {
    'base': BASE/'step_000200/adapter.pt',
    'uniform': ROOT/'robotwin/round11/tail_adapter_200_uniform/step_000200/adapter.pt',
    'tail': ROOT/'robotwin/round11/tail_adapter_200_tail/step_000200/adapter.pt',
}
VARIANTS = list(RUNS)
KEYS = ['position_cm','rotation_deg','gripper_mae']

def summarize(rows):
    names=['fresh','tail']+[f'tail_remaining_{x}' for x in [1,8,24,36]]
    groups={n:[r for r in rows if (r['cohort']==n or (n.startswith('tail_remaining_') and r['cohort']=='tail' and r['remaining']==int(n.rsplit('_',1)[1])))] for n in names}
    out={}
    for name,selected in groups.items():
        episodes=sorted({r['episode'] for r in selected})
        values={v:np.array([[np.mean([r['metrics'][v][k] for r in selected if r['episode']==e]) for k in KEYS] for e in episodes]) for v in VARIANTS}
        draws=np.random.default_rng(281204).integers(0,len(episodes),(4000,len(episodes)))
        comps={}
        for cand in ['uniform','tail']:
            scale=np.maximum(values['base'].mean(0),1e-10)
            ratios=values[cand].mean(0)/scale
            delta=((values[cand]-values['base'])/scale).mean(1)
            ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
            comps[cand+'_vs_base']=dict(ratios=dict(zip(KEYS,ratios.tolist())),normalized_excess=float(delta.mean()),bootstrap_95=ci,
                nonregression_passed=bool(max(ratios)<=1.05 and ci[1]<=.05),improvement_supported=bool(ci[1]<0 and max(ratios)<=1.05))
        out[name]=dict(episodes=len(episodes),pairs=len(selected),means={v:dict(zip(KEYS,x.mean(0).tolist())) for v,x in values.items()},comparisons=comps)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2': raise RuntimeError('must run on physical GPU2')
    if (OUT/'results.json').exists(): raise RuntimeError('completed result is immutable')
    state=dict(pid=os.getpid(),phase='loading',physical_gpu=2,completed=0,total=128,started_at=time.time(),online_evaluation=False)
    def update(**kw):
        state.update(kw); state['updated_at']=time.time(); atomic_json(OUT/'status.json',state); print(json.dumps(state),flush=True)
    update()
    try:
        for p in RUNS.values():
            assert (p.parent/'checkpoint_complete.json').exists(), p
        assert json.loads((BASE/'status.json').read_text())['phase']=='completed'
        assert json.loads((BASE/'frozen_audit.json').read_text())['passed']
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_per_process_memory_fraction(.25)
        torch.use_deterministic_algorithms(True); torch.manual_seed(281301)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        recipe=json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        teacher=RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'],recipe['teacher_checkpoint']]:
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert saved['manifest']['base_config']==cfg
            named=dict(teacher.named_parameters())
            with torch.no_grad():
                for n,v in saved['adapter'].items(): named[n].copy_(v)
            del named,saved
        teacher=teacher.cuda().eval().requires_grad_(False)
        teacher.policy.checkpoint_blocks=teacher.flow.checkpoint_blocks=False
        student=CurrentQueryStudent(teacher).cuda().eval().requires_grad_(False)
        init=torch.load(recipe['initializers'],map_location='cpu',weights_only=False)
        student.set_queries(*init['queries']['fixed_queries']); student.gate.load_state_dict(init['gate']); del init
        assert not student.gate.scales.any()
        named=dict(teacher.named_parameters()); weights={}; checkpoints={}
        for variant,path in RUNS.items():
            payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert payload['format']=='robotwin_current_action_adapter_v1' and payload['step']==200
            assert payload['manifest']['base_config']==cfg
            assert all(n.startswith(('policy.flow.goal_','policy.flow.local_')) for n in payload['adapter'])
            assert set(payload['adapter'])==set(payload['manifest']['trainable_names'])
            weights[variant]={n:v.to(device='cuda',dtype=named[n].dtype).clone() for n,v in payload['adapter'].items()}
            checkpoints[variant]=dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            del payload
        assert all(weights['base'].keys()==weights[v].keys() for v in ['uniform','tail'])
        immutable={n:p._version for n,p in teacher.named_parameters() if n not in weights['base']}
        student_versions={n:p._version for n,p in student.named_parameters()}
        def select(v):
            with torch.no_grad():
                for n,x in weights[v].items(): named[n].copy_(x)
        def audit():
            assert all(p._version==immutable[n] and p.grad is None for n,p in teacher.named_parameters() if n in immutable)
            assert all(p._version==student_versions[n] and p.grad is None for n,p in student.named_parameters())
            assert not any(p.requires_grad for p in teacher.parameters())
        def forbidden(*a,**kw): raise AssertionError('deployed student tried to read true future')
        val=CurrentDataset(cfg['dataset_root'],teacher.policy.statistics,'val')
        train=CurrentDataset(cfg['dataset_root'],teacher.policy.statistics,'train')
        by_id={e['episode_index']:e for e in val.selected}
        assert set(by_id).isdisjoint(e['episode_index'] for e in train.selected); del train
        prior=json.loads((ROOT/'robotwin/round11/adapter_confirmation_200/manifest.json').read_text())
        cohorts=prior['cohorts']
        manifest=dict(kind='three-way held-out continuation confirmation',endpoint=200,variants=VARIANTS,checkpoints=checkpoints,source_recipe=recipe,cohorts=cohorts,
            excluded_prior_episodes=prior['excluded_prior_episodes'],train_validation_episode_disjoint=True,
            bootstrap='4000 resamples of independent episodes; average noise and anchors within episode',
            acceptance='candidate fresh improvement upper95<0 and fresh/tail/each tail subgroup nonregression upper95<=.05 with no metric ratio>1.05',
            physical_gpu=2,model_inputs=list(OBSERVATION_KEYS),action_horizon=36,
            limitations=['Offline action error is not closed-loop success','Compatible current-only adapter confirmation, not full StageII distillation'],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text())==manifest
        else: atomic_json(OUT/'manifest.json',manifest)
        report=json.loads((OUT/'partial.json').read_text()) if (OUT/'partial.json').exists() else dict(manifest=manifest,rows=[])
        assert report['manifest']==manifest
        completed={(r['cohort'],r['sample'],r['noise_seed']) for r in report['rows']}
        assert len(completed)==len(report['rows'])
        # exact reproduction of the already completed base diagnostic endpoint
        update(phase='preflight')
        recorded=json.loads((BASE/'validation_000200.json').read_text())
        for i in [0,15]:
            row=recorded['rows'][i]; b=default_collate([val.sample(by_id[row['episode']],row['t'])])
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                prep=student.prepare({k:b[k] for k in OBSERVATION_KEYS}); select('base'); cond=student.conditions(prep)
                torch.manual_seed(800+i); pred=teacher.flow.sample(cond,torch.randn(1,50,32,device='cuda'))
                metric=physical_metrics(pred,b,teacher.policy.statistics); err=max(abs(metric[k]-row['metrics'][k]) for k in KEYS)
            assert err<1e-6,(i,err)
        atomic_json(OUT/'preflight.json',dict(passed=True,base_reproduction=True,posterior_access_guard=True))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False),patch.object(teacher.policy,'encode_future_images',side_effect=forbidden),patch.object(teacher.policy.future,'forward',side_effect=forbidden):
            for cohort,spec in cohorts.items():
                for i,anchor in enumerate(spec['anchors']):
                    seeds=[int(x)+i for x in spec['noise_starts']]
                    if all((cohort,i,s) in completed for s in seeds): continue
                    update(phase='decoding',cohort=cohort,anchor=i+1,anchors_total=len(spec['anchors']),completed=len(report['rows']))
                    b=default_collate([val.sample(by_id[anchor['episode']],anchor['t'])])
                    if cohort=='tail': assert int(b['remaining_actions'][0])==int(anchor['remaining'])
                    prep=student.prepare({k:b[k] for k in OBSERVATION_KEYS}); predictions={s:{} for s in seeds}
                    for v in VARIANTS:
                        select(v); cond=student.conditions(prep)
                        for seed in seeds:
                            update(phase='sampling',variant=v,noise_seed=seed)
                            torch.manual_seed(seed); predictions[seed][v]=teacher.flow.sample(cond,torch.randn(1,50,32,device='cuda'))
                    for seed in seeds:
                        if (cohort,i,seed) in completed: continue
                        metrics={v:physical_metrics(p,b,teacher.policy.statistics) for v,p in predictions[seed].items()}
                        assert all(np.isfinite(x) for m in metrics.values() for x in m.values())
                        report['rows'].append(dict(cohort=cohort,sample=i,episode=int(anchor['episode']),t=int(anchor['t']),task=by_id[anchor['episode']]['tasks'],remaining=int(b['remaining_actions'][0]),noise_seed=seed,metrics=metrics,drift_mse={v:float((p-predictions[seed]['base']).square().mean()) for v,p in predictions[seed].items() if v!='base'}))
                        completed.add((cohort,i,seed)); atomic_json(OUT/'partial.json',report); update(completed=len(report['rows']))
                    audit()
            select('base'); audit()
        assert len(report['rows'])==128
        result=summarize(report['rows'])
        candidate_flags={v: all(result[g]['comparisons'][v+'_vs_base']['nonregression_passed'] for g in result) for v in ['uniform','tail']}
        fresh_flags={v:result['fresh']['comparisons'][v+'_vs_base']['improvement_supported'] for v in ['uniform','tail']}
        eligible={v:bool(candidate_flags[v] and fresh_flags[v]) for v in ['uniform','tail']}
        report.update(completed=True,summary=result,online_eligible=any(eligible.values()),candidate_online_eligible=eligible,
                      frozen_audit=dict(passed=True,restored_base=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(OUT/'results.json',report)
        lines=['Three-way held-out continuation confirmation','Offline action errors; not a closed-loop success rate.',f'Online eligible candidates: {eligible}']
        for g in result: lines += ['',g, json.dumps(result[g]['comparisons'],indent=2)]
        (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        update(phase='completed',completed=128,online_eligible=report['online_eligible'],candidate_online_eligible=eligible)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}'); (OUT/'failure.txt').write_text(traceback.format_exc()); raise

if __name__=='__main__': main()
