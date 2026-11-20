"""Fresh held-out, paired seed teacher checks. Runs after the pilot, without training."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.round3.model import TerminalHoldTeacher,TerminalHoldDataset
from robotwin.round3.train import physical_metrics
from robotwin.round2.train import atomic_json
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'robotwin/round3/runs/teacher_terminal_hold36_20260926'
OUT=ROOT/'robotwin/round3/assessment'

def main():
    OUT.mkdir(exist_ok=True)
    status=dict(pid=os.getpid(),phase='waiting_for_teacher_200',updated_at=time.time(),physical_gpu=0)
    def update(**kw):status.update(kw);status['updated_at']=time.time();atomic_json(OUT/'status.json',status)
    update()
    try:
        while True:
            if (RUN/'status.json').exists():
                train=json.loads((RUN/'status.json').read_text())
                if train['phase']=='failed':raise RuntimeError('Teacher pilot failed: '+train.get('error',''))
                if train['phase']=='completed_teacher_pilot':break
                if time.time()-train['updated_at']>2700:raise RuntimeError('Teacher status stale >45 minutes; no training processes stopped')
            time.sleep(30)
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
        torch.manual_seed(92026)
        update(phase='loading_teacher_200')
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text());model=TerminalHoldTeacher(cfg)
        ckpt=RUN/'step_000200/adapter.pt'
        assert (ckpt.parent/'checkpoint_complete.json').exists()
        saved=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
        assert saved['format']=='robotwin_terminal_hold36_adapter_v1' and saved['step']==200
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert list(params)==list(saved['adapter'])
        with torch.no_grad():
            for n,p in params.items():p.copy_(saved['adapter'][n])
        del saved,params
        model=model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks=False;model.flow.checkpoint_blocks=False
        val=TerminalHoldDataset(cfg['dataset_root'],model.policy.statistics,'val')
        old_samples=json.loads((RUN/'samples.json').read_text())['validation_indices']
        old_episodes={val.selected[int(np.searchsorted(val.offsets,i,side='right')-1)]['episode_index'] for i in old_samples}
        # Select one anchor per fresh validation episode; no overlap with the tuning episodes.
        rng=np.random.default_rng(260926);indices=[];selected_episodes=set()
        while len(indices)<32:
            i=int(rng.integers(0,len(val)));e=val.selected[int(np.searchsorted(val.offsets,i,side='right')-1)]['episode_index']
            if e in old_episodes or e in selected_episodes:continue
            selected_episodes.add(e);indices.append(i)
        update(phase='prepare_fresh_validation',samples=32,seeds=2)
        batches=[default_collate([val[i]]) for i in indices]
        atomic_json(OUT/'samples.json',dict(seed=260926,indices=indices,episodes=sorted(selected_episodes),
            excluded_tuning_episodes=sorted(old_episodes),noise_seeds=[1800,2800]))
        prepared=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,b in enumerate(batches):
                prepared.append(model.prepare(b));update(prepared=i+1)
        donors=[i for i,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors)>=2
        report=dict(scope='32 fresh validation episodes x 2 noise seeds; offline only',checkpoint=str(ckpt),rows=[],
                    future_donor_mask='donor has all future frames; resampler validity matched to recipient',role_sensitivity=[])
        def encode(b,prep,source):
            # Change future values only; preserve receiver frame validity exactly.
            g,l=model.policy.future(*source['future'],b['local_valid'].cuda(),b['goal_frame_valid'].cuda())
            enc=model.policy.language(prep['current'],g,l)
            valid=torch.stack((b['goal_frame_valid'].any(-1),b['local_valid']),-1).cuda()
            cond=(*prep['base'],model.flow.goal_proj(enc['goal']),model.flow.local_proj(enc['local']),valid)
            return cond,enc,(g,l)
        def relative(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,(b,prep) in enumerate(zip(batches,prepared)):
                full,enc,seeds=encode(b,prep,prep)
                donor=next(j for j in donors[i%len(donors):]+donors[:i%len(donors)] if j!=i)
                wrong,we,ws=encode(b,prep,prepared[donor])
                off=(*full[:-1],torch.zeros_like(full[-1]))
                if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].any()):
                    report['role_sensitivity'].append(dict(sample=i,
                      resampler_goal_relative=relative(ws[0],seeds[0]),resampler_local_relative=relative(ws[1],seeds[1]),
                      layers={str(k):dict(goal_relative=relative(we['layers'][k][0],enc['layers'][k][0]),
                                           local_relative=relative(we['layers'][k][1],enc['layers'][k][1])) for k in enc['layers']}))
                for noise_seed in [1800,2800]:
                    torch.manual_seed(noise_seed+i);noise=torch.randn(1,50,32,device='cuda')
                    predictions={k:model.flow.sample(c,noise,bypass=k=='base') for k,c in [('full',full),('base',full),('roles_off',off),('wrong_future',wrong)]}
                    row=dict(sample=i,episode=int(b['episode_id'][0]),t=int(b['t'][0]),remaining=int(b['remaining_actions'][0]),
                             donor_episode=int(batches[donor]['episode_id'][0]),noise_seed=noise_seed+i,
                             metrics={k:physical_metrics(v,b,model.policy.statistics) for k,v in predictions.items()},
                             wrong_future_relative_l2=relative(predictions['wrong_future'],predictions['full']))
                    report['rows'].append(row);atomic_json(OUT/'results.json',report)
                    update(phase='paired_assessment',completed_pairs=len(report['rows']),total_pairs=64)
        keys=['position_cm','rotation_deg','gripper_mae'];variants=['full','base','roles_off','wrong_future']
        # Bootstrap episodes, averaging seeds within an episode; do not treat seeds as independent episodes.
        values={v:np.array([[np.mean([r['metrics'][v][k] for r in report['rows'] if r['sample']==i]) for k in keys] for i in range(32)]) for v in variants}
        means={v:dict(zip(keys,values[v].mean(0).tolist())) for v in variants}
        scale=np.maximum(values['base'].mean(0),1e-10)
        delta=((values['wrong_future']-values['full'])/scale).mean(1)
        rng=np.random.default_rng(29326);draws=rng.integers(0,32,size=(4000,32));ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        nonregression={k:float(values['full'][:,j].mean()/scale[j]) for j,k in enumerate(keys)}
        screen=all(x<=1.05 for x in nonregression.values()) and ci[0]>0
        summary=dict(mean=means,future_benefit_normalized_mean=float(delta.mean()),future_benefit_bootstrap_95=ci,
                     full_to_base_ratio=nonregression,exploratory_screen_passed=screen,
                     screen_definition='all physical mean errors <=1.05 x baseline AND paired future-benefit CI lower bound >0; exploratory, not deployment acceptance',
                     mean_wrong_future_relative_l2=float(np.mean([r['wrong_future_relative_l2'] for r in report['rows']])))
        report['summary']=summary;report['completed']=True;atomic_json(OUT/'results.json',report)
        lines=['# 修正后教师：新验证轨迹配对评估','',
               '32条新验证轨迹，每条两个噪声种子；不与16个调参验证样本的轨迹重合。所有方案只在真实帧上统计误差。','',
               '|条件|位置cm|姿态°|夹爪MAE|','|---|---:|---:|---:|']
        for v,m in means.items():lines.append(f"|{v}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
        lines += ['',f'真实未来相对错配未来的标准化收益95% bootstrap区间：{ci}。',
                  f'探索性筛查通过：{screen}。此筛查不能代替闭环成功率。',
                  '学生训练和在线评测未自动启动；根据动作保持及未来收益决定下一步。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',exploratory_screen_passed=screen)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
