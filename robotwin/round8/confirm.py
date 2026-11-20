"""Fresh paired confirmation and same-checkpoint gate ablation, offline only."""
import hashlib,json,os,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.data import CAMERAS,read_frames
from robotwin.round6.model import RetainedTeacher,FutureActionDataset
from robotwin.round6.train import atomic_json,physical_metrics
from robotwin.round7.model import CurrentQueryStudent,OBSERVATION_KEYS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'robotwin/round8/confirmation';PILOT=ROOT/'robotwin/round8/paired_pilot'
KEYS=['position_cm','rotation_deg','gripper_mae'];VARIANTS=['initial','query_only','query_gate','gate_off','base']

def summary(rows):
    groups={c:[r for r in rows if r['cohort']==c] for c in ['fresh','tail']}
    for remain in [1,8,24,36]:groups[f'tail_remaining_{remain}']=[r for r in groups['tail'] if r['remaining']==remain]
    result={}
    for group,rr in groups.items():
        episodes=sorted({r['episode'] for r in rr})
        values={v:np.array([[np.mean([r['metrics'][v][k] for r in rr if r['episode']==e]) for k in KEYS] for e in episodes]) for v in VARIANTS}
        draws=np.random.default_rng(280933).integers(0,len(episodes),(4000,len(episodes)))
        contrasts={}
        for a,b in [('query_gate','query_only'),('query_gate','gate_off'),('query_gate','initial'),('query_only','initial')]:
            scale=np.maximum(values[b].mean(0),1e-10);delta=((values[a]-values[b])/scale).mean(1)
            ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist();ratios=values[a].mean(0)/scale
            contrasts[a+'_vs_'+b]=dict(ratios=dict(zip(KEYS,ratios.tolist())),normalized_excess=float(delta.mean()),bootstrap_95=ci,
                nonregression_passed=bool(max(ratios)<=1.05 and ci[1]<=.05),improvement_supported=bool(ci[1]<0 and max(ratios)<=1.05))
        result[group]=dict(episodes=len(episodes),pairs=len(rr),means={v:dict(zip(KEYS,x.mean(0).tolist())) for v,x in values.items()},comparisons=contrasts,
            gate_output_drift_mse=float(np.mean([np.mean([r['gate_output_drift_mse'] for r in rr if r['episode']==e]) for e in episodes])))
    return result

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Existing confirmation must not be overwritten')
    state=dict(pid=os.getpid(),phase='loading',physical_gpu=0,completed=0,total=128,started_at=time.time(),optimizer_updates=0,online_evaluation=False)
    def update(**kw):state.update(kw);state['updated_at']=time.time();atomic_json(OUT/'status.json',state)
    update()
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        assert json.loads((PILOT/'status.json').read_text())['phase']=='completed'
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
        torch.use_deterministic_algorithms(True);torch.manual_seed(280930)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        recipe=json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        teacher=RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'],recipe['teacher_checkpoint']]:
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False);assert saved['manifest']['base_config']==cfg
            named=dict(teacher.named_parameters())
            with torch.no_grad():
                for n,v in saved['adapter'].items():named[n].copy_(v)
            del named,saved
        teacher=teacher.cuda().eval().requires_grad_(False);teacher.policy.checkpoint_blocks=False;teacher.flow.checkpoint_blocks=False
        frozen={n:p._version for n,p in teacher.named_parameters()}
        init=torch.load(recipe['initializers'],map_location='cpu',weights_only=False)
        initial_queries=tuple(x.clone() for x in init['queries']['fixed_queries']);students={};checkpoints={}
        for arm in ['query_only','query_gate']:
            student=CurrentQueryStudent(teacher).cuda().eval().requires_grad_(False);student.gate.load_state_dict(init['gate'])
            path=PILOT/arm/'step_000200/student.pt';assert (path.parent/'checkpoint_complete.json').exists()
            payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert payload['format']=='robotwin_paired_query_gate_v1' and payload['step']==200 and payload['arm']==arm
            named=dict(student.named_parameters());assert set(payload['trainables'])==set(payload['manifest']['arms'][arm]['trainable_names'])
            with torch.no_grad():
                for n,v in payload['trainables'].items():named[n].copy_(v)
            students[arm]=student;checkpoints[arm]=str(path);del payload,named
        del init
        control_queries=(students['query_only'].q_goal.detach().clone()[None],students['query_only'].q_local.detach().clone()[None])
        trained_scales=students['query_gate'].gate.scales.detach().clone();assert trained_scales.abs().sum()>0
        student_versions={a:{n:p._version for n,p in s.named_parameters() if n not in ('q_goal','q_local','gate.scales')} for a,s in students.items()}
        val=FutureActionDataset(cfg['dataset_root'],teacher.policy.statistics,'val',decode=False);by_id={e['episode_index']:e for e in val.selected}
        old=json.loads((ROOT/'robotwin/round7/postpilot_eval/manifest.json').read_text())
        excluded=set(old['excluded_prior_episodes'])
        for c in old['cohorts'].values():excluded.update(a['episode'] for a in c['anchors'])
        excluded.update(a['episode'] for a in json.loads((PILOT/'validation_anchors.json').read_text()))
        rng=np.random.default_rng(280931);fresh=[];chosen=set();assert len(set(by_id)-excluded)>=40
        while len(fresh)<32:
            idx=int(rng.integers(len(val)));ei=int(np.searchsorted(val.offsets,idx,side='right')-1);e=val.selected[ei]
            if e['episode_index'] in excluded|chosen:continue
            chosen.add(e['episode_index']);fresh.append(dict(episode=e['episode_index'],t=int(idx-val.offsets[ei])))
        eligible=[e for e in val.selected if e['episode_index'] not in excluded|chosen and e['length']>=100];assert len(eligible)>=8
        tails=[eligible[i] for i in np.random.default_rng(280932).choice(len(eligible),8,replace=False)]
        tail=[dict(episode=e['episode_index'],t=e['length']-n,remaining=n) for e in tails for n in [1,8,24,36]]
        cohorts=dict(fresh=dict(anchors=fresh,noise_starts=[15800,16800]),tail=dict(anchors=tail,noise_starts=[17800,18800]))
        manifest=dict(primary='query_gate200 vs query_only200 on fresh32 episodes; no intermediate checkpoint selection',
            secondary='query_gate200 vs same query_gate200 with scales zeroed; tail subgroup tests exploratory',cohorts=cohorts,
            excluded_prior_episodes=sorted(excluded),checkpoints=checkpoints,source_recipe=recipe,variants=VARIANTS,
            bootstrap='4000 resamples of independent episodes; average noise and anchors within episode',
            nonregression='each mean <=1.05x comparator and upper95 normalized excess <=.05',
            improvement='upper95 normalized excess <0 and no metric mean >1.05x comparator',
            gate_ablation='only set gate scales to zero, preserve trained queries/message/relevance weights, restore scales exactly after each anchor',
            model_inputs=list(OBSERVATION_KEYS),role_valid='constant true',action_horizon=36,decoder='current frames only',
            limitations=['Offline action error, not closed-loop success','Original SFT may have seen these demonstrations','Retains original LaWM and old queries; not pure aba2'],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_json(OUT/'manifest.json',manifest)
        def batch(anchor):
            e=by_id[anchor['episode']];t=anchor['t'];b=val.sample(e,t);images=[]
            for cam in CAMERAS:
                pre='videos/'+cam;path=val.root/val.info['video_path'].format(video_key=cam,chunk_index=e[pre+'/chunk_index'],file_index=e[pre+'/file_index'])
                timestamp=float(val.timestamps[e['dataset_from_index']+t]+e[pre+'/from_timestamp'])
                images.append(torch.from_numpy(read_frames(path,[timestamp])[0].copy()).permute(2,0,1))
            b['current_main']=images[0];b['current_wrist']=torch.stack(images[1:]);return default_collate([b])
        def forbidden(*a,**kw):raise AssertionError('Student tried to access posterior future')
        def check():
            assert all(p._version==frozen[n] and p.grad is None for n,p in teacher.named_parameters())
            for a,s in students.items():
                assert all(p._version==student_versions[a][n] for n,p in s.named_parameters() if n in student_versions[a])
            assert torch.equal(students['query_gate'].gate.scales,trained_scales)
        report=dict(manifest=manifest,completed=False,rows=[],preflight_max_abs=0.,future_access_guard_anchors=0)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),patch.object(teacher.policy,'encode_future_images',side_effect=forbidden),patch.object(teacher.policy.future,'forward',side_effect=forbidden):
            update(phase='preflight')
            prior=json.loads((PILOT/'validation_000200.json').read_text())
            for arm,s in students.items():
                for i in [0,15]:
                    row=prior['rows'][arm][i];b=batch(row);cond=s.conditions(s.prepare({k:b[k] for k in OBSERVATION_KEYS}))
                    torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda');metrics=physical_metrics(teacher.flow.sample(cond,noise),b,teacher.policy.statistics)
                    err=max(abs(metrics[k]-row['metrics'][k]) for k in KEYS);report['preflight_max_abs']=max(report['preflight_max_abs'],err);assert err<1e-6,(arm,i,err)
            atomic_json(OUT/'preflight.json',dict(passed=True,reproduction_max_abs=report['preflight_max_abs'],current_only=True))
            for cohort,spec in cohorts.items():
                for i,anchor in enumerate(spec['anchors']):
                    update(phase='decoding',cohort=cohort,anchor=i+1,anchors_total=32);b=batch(anchor)
                    if cohort=='tail':assert int(b['remaining_actions'][0])==anchor['remaining']
                    prep=students['query_only'].prepare({k:b[k] for k in OBSERVATION_KEYS});conds={}
                    students['query_only'].set_queries(*initial_queries);conds['initial']=students['query_only'].conditions(prep)
                    students['query_only'].set_queries(*control_queries);conds['query_only']=students['query_only'].conditions(prep)
                    conds['query_gate']=students['query_gate'].conditions(prep)
                    try:
                        students['query_gate'].gate.scales.zero_();conds['gate_off']=students['query_gate'].conditions(prep)
                    finally:students['query_gate'].gate.scales.copy_(trained_scales)
                    assert all(c[-1].all() for c in conds.values())
                    for start in spec['noise_starts']:
                        update(phase='sampling',noise_seed=start+i);torch.manual_seed(start+i);noise=torch.randn(1,50,32,device='cuda')
                        pred={v:teacher.flow.sample(c,noise) for v,c in conds.items()};pred['base']=teacher.flow.sample(conds['initial'],noise,bypass=True)
                        metrics={v:physical_metrics(p,b,teacher.policy.statistics) for v,p in pred.items()}
                        report['rows'].append(dict(cohort=cohort,sample=i,episode=anchor['episode'],t=anchor['t'],remaining=int(b['remaining_actions'][0]),noise_seed=start+i,metrics=metrics,
                            gate_output_drift_mse=float((pred['query_gate']-pred['gate_off']).square().mean())))
                        atomic_json(OUT/'partial.json',report);update(completed=len(report['rows']))
                    report['future_access_guard_anchors']+=1;check()
        result=summary(report['rows']);primary=result['fresh']['comparisons']['query_gate_vs_query_only']
        report.update(completed=True,summary=result,primary_improvement_supported=primary['improvement_supported'],primary_nonregression_passed=primary['nonregression_passed'],
            frozen_audit=dict(passed=True,teacher_tensors=len(frozen),scales_restored=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(OUT/'results.json',report)
        lines=['# Round8 paired confirmation and gate ablation','','Primary endpoint: fresh query_gate200 vs query_only200; lower errors are better.','Offline errors do not establish closed-loop success.']
        for name,g in result.items():
            lines+=['',f'## {name}: {g["episodes"]} episodes / {g["pairs"]} noise pairs','', '|Variant|Position cm|Rotation deg|Gripper MAE|','|---|---:|---:|---:|']
            for v,m in g['means'].items():lines.append(f'|{v}|{m["position_cm"]:.6f}|{m["rotation_deg"]:.6f}|{m["gripper_mae"]:.7f}|')
            for v,c in g['comparisons'].items():lines.append(f'\n{v}: normalized excess 95%CI {c["bootstrap_95"]}; ratios {c["ratios"]}; improvement supported {c["improvement_supported"]}.')
            lines.append(f'\nSame-checkpoint gate output drift MSE: {g["gate_output_drift_mse"]}.')
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');update(phase='completed',primary_improvement_supported=primary['improvement_supported'],primary_nonregression_passed=primary['nonregression_passed'])
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');(OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
