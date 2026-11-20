"""Bounded current-only query pilot, with action supervision and KD diagnostics."""
import hashlib,json,os,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import default_collate
from goal_local.model import GoalLocalPolicy
from goal_local.roles import role_distillation
from goal_local.optim import MasterAdamW
from robotwin.round6.model import RetainedTeacher,FutureActionDataset
from robotwin.round6.train import atomic_json,physical_metrics
from robotwin.round7.model import CurrentQueryStudent,OBSERVATION_KEYS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'robotwin/round7/query_pilot'
KEYS=['position_cm','rotation_deg','gripper_mae']

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'metrics.jsonl').exists():raise RuntimeError('Existing pilot is not overwritten; explicit reviewed resume required')
    status=dict(pid=os.getpid(),phase='loading',physical_gpu=0,step=0,total_steps=200,started_at=time.time(),online_evaluation=False)
    def update(**kw):status.update(kw);status['updated_at']=time.time();atomic_json(OUT/'status.json',status)
    def record(row):
        row['at']=time.time()
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def ac():return torch.autocast('cuda',dtype=torch.bfloat16)
    update()
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        import shutil
        assert shutil.disk_usage(OUT).free>20*2**30
        recipe=json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        assert recipe['status']=='eligible_for_bounded_student_pilot' and recipe['primary_candidate']=='fixed_queries'
        audit=json.loads((ROOT/'robotwin/round7/audit/results.json').read_text())
        assert audit['completed'] and audit['summary']['fresh']['comparisons']['fixed_queries']['exploratory_screen_passed']
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
        torch.manual_seed(280905);np.random.seed(280905)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        teacher=RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'],recipe['teacher_checkpoint']]:
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False);named=dict(teacher.named_parameters())
            assert saved['manifest']['base_config']==cfg
            with torch.no_grad():
                for n,v in saved['adapter'].items():named[n].copy_(v)
            del named,saved
        teacher=teacher.cuda().eval().requires_grad_(False)
        teacher.policy.checkpoint_blocks=False;teacher.flow.checkpoint_blocks=True
        student=CurrentQueryStudent(teacher).cuda().requires_grad_(False)
        init=torch.load(recipe['initializers'],map_location='cpu',weights_only=False)
        student.set_queries(*init['queries']['fixed_queries']);student.gate.load_state_dict(init['gate']);del init
        student.q_goal.requires_grad_(True);student.q_local.requires_grad_(True);student.checkpoint_blocks=True
        params={n:p for n,p in student.named_parameters() if p.requires_grad};assert set(params)=={'q_goal','q_local'}
        assert sum(p.numel() for p in params.values())==32768 and not student.gate.scales.any()
        frozen={n:p._version for n,p in teacher.named_parameters()};gate_versions={n:p._version for n,p in student.gate.named_parameters()}
        def check_frozen():
            assert all(p._version==frozen[n] and p.grad is None for n,p in teacher.named_parameters())
            assert all(p._version==gate_versions[n] and p.grad is None for n,p in student.gate.named_parameters())
        def mode(training):
            student.train(training);student.gate.eval()
            teacher.flow.train(training);teacher.flow.DiT.eval()
        def fail_future(*a,**kw):raise AssertionError('Student called future input path')
        def encode(batch):
            obs={k:batch[k] for k in OBSERVATION_KEYS}
            with patch.object(teacher.policy,'encode_future_images',side_effect=fail_future),patch.object(teacher.policy.future,'forward',side_effect=fail_future):
                prep=student.prepare(obs)
                enc=GoalLocalPolicy.language(student,prep['current'],student.q_goal[None],student.q_local[None])
                valid=torch.ones(1,2,device='cuda',dtype=torch.bool)
                cond=(*prep['base'],student.flow.goal_proj(enc['goal']),student.flow.local_proj(enc['local']),valid)
            return cond,enc,prep
        def action_loss(b,cond):return student.flow.loss(b['actions'].cuda(),b['actions_mask'].cuda(),b['action_hz'].cuda(),b['embodiment_id'].cuda(),b['remaining_actions'].cuda(),cond,repeats=2)
        manifest=dict(kind='current-query-only student pilot, not old Stage II resume',recipe=recipe,seed=280905,total_steps=200,effective_batch=4,
            learning_rate=1e-4,warmup_steps=20,trainable_names=list(params),trainable_count=32768,gate_frozen_at_zero=True,backbone_and_actor_frozen=True,
            objective='action flow-matching loss only',lambda_local_kd=0.,lambda_goal_kd=0.,lambda_action_retention=0.,
            kd_diagnostics='first microbatch each step; valid future targets only; monitoring without gradient',
            saves_every=50,validation_samples=16,regression_guard='any validation mean >1.10 x initial for 2 consecutive validations stops this pilot after saving',
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'robotwin/round7/model.py',ROOT/'robotwin/round6/model.py']})
        atomic_json(OUT/'manifest.json',manifest)
        train=FutureActionDataset(cfg['dataset_root'],teacher.policy.statistics,'train');val=FutureActionDataset(cfg['dataset_root'],teacher.policy.statistics,'val')
        assert set(e['episode_index'] for e in train.selected).isdisjoint(e['episode_index'] for e in val.selected)
        indices=np.random.default_rng(280905).integers(0,len(train),800).tolist()
        vi=json.loads((ROOT/'robotwin/runs/stage_II_20260925/validation_samples.json').read_text())['indices'][:16]
        atomic_json(OUT/'samples.json',dict(train_indices=indices,validation_indices=vi,round7_fresh_not_used_for_tuning=True))
        update(phase='decoding_validation');batches=[default_collate([val[i]]) for i in vi]
        # Numerical and gradient checks must pass before any optimizer update.
        update(phase='preflight');mode(False)
        with torch.no_grad(),ac():
            c,e,p=encode(batches[0]);original=student.conditions(p)
            assert all(torch.equal(x,y) for x,y in zip(c,original))
        del c,e,p,original
        mode(True)
        with ac():c,e,p=encode(batches[0]);loss=action_loss(batches[0],c)
        assert torch.isfinite(loss);loss.backward()
        grads={n:float(p.grad.float().norm()) for n,p in params.items()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in params.values())
        check_frozen()
        for p in params.values():p.grad=None
        atomic_json(OUT/'preflight.json',dict(passed=True,action_loss=float(loss.detach()),query_gradient_norms=grads,frozen_actor_no_grad=True,gate_no_grad=True,
            current_only_guard_passed=True,forward_equivalence_max_abs=0.,frozen_dit_eval=not teacher.flow.DiT.training))
        del c,e,p,loss
        optim=MasterAdamW(params.values(),lr=1e-4,weight_decay=1e-8,offload=True)
        initial_predictions={};initial_means=None
        def save(step):
            assert shutil.disk_usage(OUT).free>16*2**30
            update(phase='saving',step=step);dest=OUT/f'step_{step:06d}';dest.mkdir(exist_ok=True)
            payload=dict(format='robotwin_current_query_pilot_v1',step=step,queries={n:p.detach().cpu() for n,p in params.items()},optimizer=optim.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),manifest=manifest,requires_initializers=recipe['initializers'])
            torch.save(payload,dest/'student.tmp');(dest/'student.tmp').replace(dest/'student.pt')
            atomic_json(dest/'checkpoint_complete.json',dict(step=step,contains_optimizer=True,external_frozen_actor=True))
        @torch.no_grad()
        def evaluate(step):
            mode(False);update(phase='validation',step=step);rows=[]
            with torch.random.fork_rng(devices=[0]),ac():
                for i,b in enumerate(batches):
                    update(validation_sample=i+1,validation_total=len(batches))
                    cond,_,_=encode(b);torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
                    pred=student.flow.sample(cond,noise)
                    if step==0:initial_predictions[i]=pred.detach().cpu()
                    drift=float((pred-initial_predictions[i].to(pred)).square().mean())
                    rows.append(dict(sample=i,episode=int(b['episode_id'][0]),t=int(b['t'][0]),metrics=physical_metrics(pred,b,teacher.policy.statistics),drift_mse=drift))
            means={k:float(np.mean([x['metrics'][k] for x in rows])) for k in KEYS}
            atomic_json(OUT/f'validation_{step:06d}.json',dict(step=step,rows=rows,mean=means))
            record(dict(step=step,split='validation',mean=means,initial_action_drift_mse=float(np.mean([x['drift_mse'] for x in rows]))))
            return means
        initial_means=evaluate(0);save(0);duration=[];regressions=0
        for step in range(200):
            started=time.time();mode(True);optim.zero_grad();mean_loss=0.;kd=None
            for micro in range(4):
                update(phase='training',step=step,micro=micro+1)
                b=default_collate([train[indices[step*4+micro]]])
                with ac():cond,enc,prep=encode(b);loss=action_loss(b,cond)/4
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite action loss; no optimizer update')
                if micro==0:
                    with torch.no_grad(),ac():
                        # Teacher targets are diagnostic only, outside the student's strict input path.
                        future=teacher.policy.encode_future_images(b)
                        _,te=teacher.conditions_encoded(b,dict(current=prep['current'],base=prep['base'],future=future))
                        kd={name:float(role_distillation(enc[name].detach(),te[name],valid.cuda())) for name,valid in [('local',b['local_valid']),('goal',b['goal_frame_valid'].any(-1))]}
                        kd.update(local_valid=bool(b['local_valid'][0]),goal_valid=bool(b['goal_frame_valid'][0].any()))
                    del future,te
                loss.backward();mean_loss+=float(loss.detach())
                del loss,cond,enc,prep,b
            norm=torch.stack([p.grad.detach().double().square().sum() for p in params.values()]).sum().sqrt()
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite query gradient; no optimizer update')
            scale=min(1.,1./max(float(norm),1e-12))
            for p in params.values():p.grad.mul_(scale)
            lr=1e-4*min(1.,(step+1)/20)
            for group in optim.optimizer.param_groups:group['lr']=lr
            optim.step();check_frozen();duration.append(time.time()-started)
            record(dict(step=step+1,split='train',action_loss=mean_loss,kd_diagnostic=kd,kd_weight=0.,grad_norm=float(norm),lr=lr,seconds=duration[-1]))
            update(step=step+1,micro=0,eta_seconds=(199-step)*float(np.mean(duration[-20:])))
            if (step+1)%50==0:
                save(step+1);measured=evaluate(step+1)
                ratios={k:measured[k]/max(initial_means[k],1e-10) for k in KEYS};bad=any(x>1.10 for x in ratios.values())
                regressions=regressions+1 if bad else 0
                atomic_json(OUT/f'guard_{step+1:06d}.json',dict(ratio_to_initial=ratios,consecutive_regressions=regressions))
                if regressions>=2:
                    update(phase='stopped_by_regression_guard',step=step+1,reason='Two consecutive validations exceeded initial error by >10%');return
        check_frozen();atomic_json(OUT/'frozen_audit.json',dict(passed=True,teacher_tensors=len(frozen),gate_tensors=len(gate_versions),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        update(phase='completed_query_pilot',step=200,eta_seconds=0,next_phase='Review fixed validation and run fresh confirmation; no automatic full Stage II or online rollout')
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');(OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
