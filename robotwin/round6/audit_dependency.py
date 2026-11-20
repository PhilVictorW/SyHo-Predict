"""Read-only matched-cohort diagnosis; never updates weights or launches rollouts."""
import json, os, time, traceback, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from robotwin.round6.model import RetainedTeacher, FutureActionDataset
from robotwin.round6.train import atomic_json, physical_metrics
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'robotwin/round6/dependency_audit'
KEYS=['position_cm','rotation_deg','gripper_mae']

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('Completed output already exists')
    status=dict(pid=os.getpid(),phase='loading_model',physical_gpu=0,completed=0,total=128,started_at=time.time())
    def update(**kw):status.update(kw);status['updated_at']=time.time();atomic_json(OUT/'status.json',status)
    update()
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
        torch.manual_seed(92026)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        model=RetainedTeacher(cfg)
        source=ROOT/'robotwin/round4/runs/future_aux/step_000200/adapter.pt'
        saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        named=dict(model.named_parameters())
        with torch.no_grad():
            for n,v in saved['adapter'].items():named[n].copy_(v)
        del saved,named
        adapter_names=[n for n,p in model.named_parameters() if p.requires_grad]
        model=model.cuda().eval().requires_grad_(False)
        model.policy.checkpoint_blocks=False;model.flow.checkpoint_blocks=False
        frozen={n:p._version for n,p in model.named_parameters() if n not in adapter_names}
        val=FutureActionDataset(cfg['dataset_root'],model.policy.statistics,'val')
        train=FutureActionDataset(cfg['dataset_root'],model.policy.statistics,'train')
        samples=json.loads((ROOT/'robotwin/round6/assessment/control/samples.json').read_text())
        prior=json.loads((ROOT/'robotwin/round6/assessment/control/results.json').read_text())
        prior_map={(x['sample'],x['noise_seed']):x for x in prior['rows']}
        assert prior['completed'] and len(prior_map)==64
        indices=samples['indices'];assert len(indices)==32
        update(phase='decoding_samples')
        batches=[default_collate([val[i]]) for i in indices]
        # One fixed TRAIN donor is chosen before outcomes, independently of validation.
        donor_episode=next(e for e in train.selected if e['length']>88)
        fixed_batch=default_collate([train.sample(donor_episode,0)])
        assert fixed_batch['local_valid'].all() and fixed_batch['goal_frame_valid'].all()
        fixed_id=int(fixed_batch['episode_id'][0]);assert fixed_id not in samples['episodes']
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            fixed_future=tuple(x.detach() for x in model.policy.encode_future_images(fixed_batch))
            prepared=[]
            for i,b in enumerate(batches):
                prepared.append(model.prepare(b));update(phase='preparing_frozen_features',prepared=i+1)
        donors=[i for i,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
        assert len(donors)>=2
        manifest=dict(scope='Diagnostic reuse of round6 32 episodes x 2 noises; not a fresh acceptance set',indices=indices,
            episodes=samples['episodes'],noise_seeds=[7800,8800],fixed_training_donor=dict(episode=fixed_id,t=0),
            checkpoints={'round6_control':'robotwin/round6/runs/control/step_000200/adapter.pt','round5_equal':'robotwin/round5/runs/equal/step_000200/adapter.pt'},
            conditions={'full':'True future; all original recipient conditions unchanged',
                'wrong_future':'Same donor selection and recipient validity as round6 published assessment',
                'fixed_train_future':'Same training-episode future images for every recipient; recipient validity preserved',
                'zero_role_seed':'Zero appended Goal/Local seeds before frozen Qwen; role attention and recipient validity preserved'},
            limitations=['Neutral interventions are distribution shifts, not deployable students',
                'Original LaWM prediction and original queries are retained in every condition',
                'Recipient future validity is held fixed; this is a future-value ablation, not a no-future student',
                'Cross-round comparison includes changed dropout/RNG behavior, not a seeded isolated causal trial'],
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_optimizer=True,online_evaluation_paused=True)
        atomic_json(OUT/'manifest.json',manifest)
        report=dict(manifest=manifest,rows=[],reproduction_max_abs=0.,completed=False)
        def encode(b,prep,kind,donor):
            if kind=='zero_role_seed':
                g=prep['current'][0].new_zeros((1,8,2048));l=torch.zeros_like(g)
            else:
                future=prep['future'] if kind=='full' else fixed_future if kind=='fixed_train_future' else prepared[donor]['future']
                g,l=model.policy.future(*future,b['local_valid'].cuda(),b['goal_frame_valid'].cuda())
            enc=model.policy.language(prep['current'],g,l)
            valid=torch.stack((b['goal_frame_valid'].any(-1),b['local_valid']),-1).cuda()
            return (*prep['base'],model.flow.goal_proj(enc['goal']),model.flow.local_proj(enc['local']),valid)
        for ckpt_name,relative in manifest['checkpoints'].items():
            update(phase='loading_adapter',checkpoint=ckpt_name)
            path=ROOT/relative;assert (path.parent/'checkpoint_complete.json').exists()
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert saved['step']==200 and list(saved['adapter'])==adapter_names
            assert saved['manifest']['base_config']==cfg
            named=dict(model.named_parameters())
            with torch.no_grad():
                for n,v in saved['adapter'].items():named[n].copy_(v)
            del named,saved
            conditions=['full','wrong_future','fixed_train_future','zero_role_seed'] if ckpt_name=='round6_control' else ['full']
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i,(b,prep) in enumerate(zip(batches,prepared)):
                    donor=next(j for j in donors[i%len(donors):]+donors[:i%len(donors)] if j!=i)
                    encoded={k:encode(b,prep,k,donor) for k in conditions}
                    for noise_seed in [7800,8800]:
                        torch.manual_seed(noise_seed+i);noise=torch.randn(1,50,32,device='cuda')
                        preds={k:model.flow.sample(c,noise) for k,c in encoded.items()}
                        metrics={k:physical_metrics(v,b,model.policy.statistics) for k,v in preds.items()}
                        old=prior_map[(i,noise_seed+i)]
                        assert old['episode']==int(b['episode_id'][0])
                        if ckpt_name=='round6_control':
                            errors=[abs(metrics[k][m]-old['metrics'][k][m]) for k in ['full','wrong_future'] for m in KEYS]
                            err=max(errors);report['reproduction_max_abs']=max(report['reproduction_max_abs'],err)
                            assert err<1e-6,(i,noise_seed,err)
                        row=dict(checkpoint=ckpt_name,sample=i,episode=int(b['episode_id'][0]),t=int(b['t'][0]),noise_seed=noise_seed+i,
                            all_future_valid=bool(b['local_valid'][0] and b['goal_frame_valid'][0].all()),metrics=metrics,
                            base=old['metrics']['base'],anchor=old['metrics']['anchor'],
                            relative_l2={k:float((v-preds['full']).norm()/preds['full'].norm().clamp_min(1e-12)) for k,v in preds.items()})
                        report['rows'].append(row);atomic_json(OUT/'partial.json',report)
                        update(phase='sampling',checkpoint=ckpt_name,sample=i+1,completed=len(report['rows']))
            assert all(p._version==frozen[n] and p.grad is None for n,p in model.named_parameters() if n in frozen)
        def array(checkpoint,condition,subset=None):
            ids=list(range(32)) if subset is None else subset
            return np.array([[np.mean([(r[condition] if condition in ['base','anchor'] else r['metrics'][condition])[k]
                for r in report['rows'] if r['checkpoint']==checkpoint and r['sample']==i]) for k in KEYS] for i in ids])
        base=array('round6_control','base');scale=np.maximum(base.mean(0),1e-10)
        arrays={k:array('round6_control',k) for k in ['full','wrong_future','fixed_train_future','zero_role_seed','base','anchor']}
        arrays['round5_equal']=array('round5_equal','full')
        def benefit(a,b):
            # positive: b has lower error than a; episode-level paired bootstrap.
            delta=((a-b)/scale).mean(1);draws=np.random.default_rng(271002).integers(0,len(delta),(4000,len(delta)))
            return dict(mean=float(delta.mean()),bootstrap_95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
        comparisons={f'full_over_{k}':benefit(a,arrays['full']) for k,a in arrays.items() if k!='full'}
        valid_ids=[i for i,b in enumerate(batches) if bool(b['local_valid'][0] and b['goal_frame_valid'][0].all())]
        valid_comparisons={f'full_over_{k}':benefit(a[valid_ids],arrays['full'][valid_ids]) for k,a in arrays.items() if k not in ['full','round5_equal']}
        report.update(completed=True,summary=dict(means={k:dict(zip(KEYS,v.mean(0).tolist())) for k,v in arrays.items()},
            paired_benefits=comparisons,all_future_valid_samples=len(valid_ids),all_future_valid_benefits=valid_comparisons),
            frozen_audit=dict(passed=True,tensors=len(frozen),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(OUT/'results.json',report)
        lines=['# 第六轮：未来条件依赖性与跨轮同样本复核','',
            '复用第六轮32条轨迹×2噪声，只读诊断；没有优化器更新，也不是新的独立验收集。','',
            '|条件|位置cm|姿态°|夹爪MAE|','|---|---:|---:|---:|']
        for k,m in report['summary']['means'].items():lines.append(f"|{k}|{m['position_cm']:.6f}|{m['rotation_deg']:.6f}|{m['gripper_mae']:.7f}|")
        lines+=['','正收益表示第六轮真实未来 full 的误差更低；区间按轨迹bootstrap，未校正多重比较。','',
            '|配对对照|标准化收益|95%区间|','|---|---:|---|']
        for k,v in comparisons.items():lines.append(f"|{k}|{v['mean']:.6f}|{v['bootstrap_95']}|")
        lines+=['',f"原 full/wrong_future 结果复现最大差值：{report['reproduction_max_abs']}。",'冻结骨干版本检查通过。',
            '固定训练供体与零角色种子只移除新增分支中的真实未来值；原始LaWM、当前观测和接收样本的未来有效性掩码仍保留。中性输入可能分布外，不能当作已验证的学生。',
            '第五轮/第六轮共同样本比较只确认检查点差异，不单独证明dropout的因果贡献；修正训练模式会改变随机数轨迹。',
            '旧Stage II与在线评测继续暂停。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',completed=len(report['rows']))
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');(OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
