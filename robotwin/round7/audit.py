"""Zero-update student transfer: old diagnostic cohort and fresh confirmation."""
import hashlib,json,os,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.round6.model import RetainedTeacher,FutureActionDataset
from robotwin.round6.train import atomic_json,physical_metrics
from robotwin.round7.model import CurrentQueryStudent,OBSERVATION_KEYS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'robotwin/round7/audit'
KEYS=['position_cm','rotation_deg','gripper_mae']

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('Completed audit exists')
    state=dict(pid=os.getpid(),phase='loading_model',physical_gpu=0,completed=0,total=128,started_at=time.time())
    def update(**kw):state.update(kw);state['updated_at']=time.time();atomic_json(OUT/'status.json',state)
    update()
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
        torch.manual_seed(92026)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        model=RetainedTeacher(cfg)
        source=ROOT/'robotwin/round4/runs/future_aux/step_000200/adapter.pt'
        trained=ROOT/'robotwin/round6/runs/control/step_000200/adapter.pt'
        for path in [source,trained]:
            assert (path.parent/'checkpoint_complete.json').exists()
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert saved['step']==200 and saved['manifest']['base_config']==cfg
            named=dict(model.named_parameters())
            with torch.no_grad():
                for n,v in saved['adapter'].items():named[n].copy_(v)
            del saved,named
        model=model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks=False;model.flow.checkpoint_blocks=False
        frozen={n:p._version for n,p in model.named_parameters()}
        student=CurrentQueryStudent(model).cuda().eval().requires_grad_(False)
        assert torch.count_nonzero(student.gate.scales)==0
        gate_versions={n:p._version for n,p in student.gate.named_parameters()}
        train=FutureActionDataset(cfg['dataset_root'],model.policy.statistics,'train')
        val=FutureActionDataset(cfg['dataset_root'],model.policy.statistics,'val')
        old_samples=json.loads((ROOT/'robotwin/round6/assessment/control/samples.json').read_text())
        old_results=json.loads((ROOT/'robotwin/round6/assessment/control/results.json').read_text())
        prior={(r['sample'],r['noise_seed']):r for r in old_results['rows']}
        excluded=set(old_samples['episodes'])|set(old_samples['excluded_tuning_episodes'])
        rng=np.random.default_rng(280902);fresh=[];episodes=set()
        while len(fresh)<32:
            i=int(rng.integers(len(val)));e=val.selected[int(np.searchsorted(val.offsets,i,side='right')-1)]['episode_index']
            if e in excluded or e in episodes:continue
            episodes.add(e);fresh.append(i)
        # Fix train-only initializers before any evaluation outcome is computed.
        eligible=[e for e in train.selected if e['length']>88]
        fixed=eligible[0];assert fixed['episode_index'] not in excluded|episodes
        rng=np.random.default_rng(280903)
        chosen=[eligible[i] for i in rng.choice(len(eligible),32,replace=False)]
        calibration=[(e,int(rng.integers(0,e['length']-88))) for e in chosen]
        manifest=dict(scope='Current-only student initialization audit; zero optimizer updates',teacher=str(trained),source=str(source),
            primary_candidate='fixed_queries',secondary_candidate='mean_queries',negative_control='zero_queries',
            calibration=[dict(episode=e['episode_index'],t=t) for e,t in calibration],fixed_donor=dict(episode=fixed['episode_index'],t=0),
            cohorts={'diagnostic':dict(indices=old_samples['indices'],episodes=old_samples['episodes'],noise_seeds=[7800,8800]),
                     'fresh':dict(indices=fresh,episodes=sorted(episodes),noise_seeds=[9800,10800],excluded_episodes=sorted(excluded))},
            inference_inputs=list(OBSERVATION_KEYS),student_role_valid='constant all true, never episode-derived',student_gate_initial_scale=0,
            original_LaWM_and_queries_retained=True,action_horizon=36,action_dimensions=16,
            screen='fresh: every mean metric <=1.05 x teacher AND <=1.05 x base; upper 95% CI of normalized excess vs teacher <=0.05',
            limitations=['Exploratory offline screen, not deployment approval','Training-derived global queries are fixed across inference episodes',
                        'Original SFT may have seen validation demonstrations','No old Stage II resume or online rollout'],
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py')]})
        atomic_json(OUT/'manifest.json',manifest)
        update(phase='calibrating_train_only_queries')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            def seeds(e,t):
                b=default_collate([train.sample(e,t)])
                assert b['local_valid'].all() and b['goal_frame_valid'].all()
                lv,gv=model.policy.encode_future_images(b)
                return tuple(x.detach().float() for x in model.policy.future(lv,gv,b['local_valid'].cuda(),b['goal_frame_valid'].cuda()))
            fixed_seeds=seeds(fixed,0);gs=[];ls=[]
            for j,(e,t) in enumerate(calibration):
                g,l=seeds(e,t);gs.append(g);ls.append(l);update(calibration_completed=j+1,calibration_total=32)
            choices={'fixed_queries':fixed_seeds,'mean_queries':(torch.stack(gs).mean(0),torch.stack(ls).mean(0)),
                     'zero_queries':(torch.zeros_like(gs[0]),torch.zeros_like(ls[0]))}
        del gs,ls
        torch.save(dict(format='robotwin_current_query_initializers_v1',teacher=str(trained),queries={k:tuple(x.cpu() for x in v) for k,v in choices.items()},
                        gate={n:p.detach().cpu() for n,p in student.gate.state_dict().items()},manifest=manifest),OUT/'initializers.pt')
        report=dict(manifest=manifest,rows=[],completed=False,reproduction_max_abs=0.,current_base_max_abs=0.,zero_gate_equivalence_max_abs=0.,future_access_guard_passed=0)
        def fail_future(*args,**kwargs):raise AssertionError('Student attempted to access posterior future path')
        for cohort,c in manifest['cohorts'].items():
            for i,index in enumerate(c['indices']):
                update(phase='decoding',cohort=cohort,sample=i+1)
                b=default_collate([val[index]])
                obs={k:b[k] for k in OBSERVATION_KEYS}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    teacher_prep=model.prepare(b);teacher_cond=model.conditions(b,teacher_prep)
                    with patch.object(model.policy,'encode_future_images',side_effect=fail_future),patch.object(model.policy.future,'forward',side_effect=fail_future):
                        current=student.prepare(obs)
                        errors=[float((x.float()-y.float()).abs().max()) for x,y in zip(current['base'],teacher_prep['base'])]
                        report['current_base_max_abs']=max(report['current_base_max_abs'],max(errors));assert max(errors)==0
                        conds={}
                        for name,(g,l) in choices.items():
                            student.set_queries(g,l);conds[name]=student.conditions(current)
                            assert conds[name][-1].all()
                        # Zero bridge must exactly preserve the teacher language path for equal seed inputs.
                        enc=model.policy.language(current['current'],*fixed_seeds)
                        expected=(model.flow.goal_proj(enc['goal']),model.flow.local_proj(enc['local']))
                        err=max(float((x.float()-y.float()).abs().max()) for x,y in zip(conds['fixed_queries'][3:5],expected))
                        report['zero_gate_equivalence_max_abs']=max(report['zero_gate_equivalence_max_abs'],err);assert err==0,err
                        report['future_access_guard_passed']+=1
                        for noise_start in c['noise_seeds']:
                            torch.manual_seed(noise_start+i);noise=torch.randn(1,50,32,device='cuda')
                            pred={k:model.flow.sample(cond,noise) for k,cond in conds.items()}
                            # Posterior reference uses previously encoded tensors; guard stays active.
                            pred['teacher']=model.flow.sample(teacher_cond,noise);pred['base']=model.flow.sample(teacher_cond,noise,bypass=True)
                            metrics={k:physical_metrics(v,b,model.policy.statistics) for k,v in pred.items()}
                            if cohort=='diagnostic':
                                old=prior[(i,noise_start+i)];assert old['episode']==int(b['episode_id'][0])
                                err=max(abs(metrics[k][m]-old['metrics'][v][m]) for k,v in [('teacher','full'),('base','base')] for m in KEYS)
                                report['reproduction_max_abs']=max(report['reproduction_max_abs'],err);assert err<1e-6,err
                            report['rows'].append(dict(cohort=cohort,sample=i,episode=int(b['episode_id'][0]),t=int(b['t'][0]),noise_seed=noise_start+i,
                                remaining=int(b['remaining_actions'][0]),all_future_valid=bool(b['local_valid'][0] and b['goal_frame_valid'][0].all()),metrics=metrics,
                                student_role_valid_all_true=True))
                            atomic_json(OUT/'partial.json',report);update(phase='sampling',completed=len(report['rows']))
                assert all(p._version==frozen[n] and p.grad is None for n,p in model.named_parameters())
                assert all(p._version==gate_versions[n] for n,p in student.gate.named_parameters())
        summaries={}
        for cohort in manifest['cohorts']:
            def array(k):return np.array([[np.mean([r['metrics'][k][m] for r in report['rows'] if r['cohort']==cohort and r['sample']==i]) for m in KEYS] for i in range(32)])
            vals={k:array(k) for k in [*choices,'teacher','base']};scale=np.maximum(vals['base'].mean(0),1e-10)
            draws=np.random.default_rng(280904).integers(0,32,(4000,32));comparisons={}
            for k in choices:
                excess=((vals[k]-vals['teacher'])/scale).mean(1);ci=np.quantile(excess[draws].mean(1),[.025,.975]).tolist()
                ratio_t=vals[k].mean(0)/np.maximum(vals['teacher'].mean(0),1e-10);ratio_b=vals[k].mean(0)/scale
                comparisons[k]=dict(normalized_excess_vs_teacher=float(excess.mean()),bootstrap_95=ci,ratio_to_teacher=dict(zip(KEYS,ratio_t.tolist())),
                    ratio_to_base=dict(zip(KEYS,ratio_b.tolist())),exploratory_screen_passed=bool(max(ratio_t)<=1.05 and max(ratio_b)<=1.05 and ci[1]<=.05))
            summaries[cohort]=dict(means={k:dict(zip(KEYS,x.mean(0).tolist())) for k,x in vals.items()},comparisons=comparisons,
                samples_with_missing_future=sum(not next(r for r in report['rows'] if r['cohort']==cohort and r['sample']==i)['all_future_valid'] for i in range(32)))
        report.update(completed=True,summary=summaries,frozen_audit=dict(passed=True,tensors=len(frozen),gate_frozen=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(OUT/'results.json',report)
        lines=['# 第七轮：无真实未来输入的学生初始化验证','',
            '学生只读取当前图像和指令；角色有效性恒为真。固定/平均query均由训练集离线生成，推理时不读取未来图像、未来掩码、剩余步数或动作标签。原始LaWM预测条件和旧query保留。','',
            '新增学生RelevanceGate位于后8层，初始scale=0；预检要求同query时与教师语言路径逐元素相同。本轮不更新任何权重。']
        for cohort,s in summaries.items():
            lines+=['',f'## {cohort}：32条轨迹×2噪声','', '|配置|位置cm|姿态°|夹爪MAE|','|---|---:|---:|---:|']
            for k,m in s['means'].items():lines.append(f"|{k}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
            lines+=['','|候选|相对教师标准化误差增量95%区间|探索性筛查|','|---|---|---|']
            for k,v in s['comparisons'].items():lines.append(f"|{k}|{v['bootstrap_95']}|{v['exploratory_screen_passed']}|")
        lines+=['',f"旧结果复现最大差值：{report['reproduction_max_abs']}；当前条件一致性：{report['current_base_max_abs']}；零门控一致性：{report['zero_gate_equivalence_max_abs']}。",
            f"未来路径阻断检查通过{report['future_access_guard_passed']}个样本；全部教师参数与新门控版本检查通过。",
            'fixed_queries为预先指定的主候选；mean_queries为次要候选，不能在看到fresh结果后择优并把同一数据当新验收。',
            '离线筛查不等于闭环成功率。是否开展新学生小规模训练需依据本结果另行判断；没有恢复旧Stage II或在线评测。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');update(phase='completed',completed=len(report['rows']),primary_screen_passed=summaries['fresh']['comparisons']['fixed_queries']['exploratory_screen_passed'])
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');(OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
