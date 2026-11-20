"""Paired action-only query vs query+gate trial; current-only student API."""
import hashlib,json,os,shutil,time,traceback
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
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'robotwin/round11/paired_query_pilot_500'
KEYS=['position_cm','rotation_deg','gripper_mae'];ARMS=['query_only','query_gate']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Existing experiment must not be overwritten')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu==2
    state=dict(pid=os.getpid(),physical_gpu=gpu,phase='loading',step=0,total_steps=500,started_at=time.time(),online_evaluation=False)
    def update(**kw):state.update(kw);state['updated_at']=time.time();atomic_json(OUT/'status.json',state)
    def record(row):
        row['at']=time.time()
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def ac():return torch.autocast('cuda',dtype=torch.bfloat16)
    update()
    try:
        assert shutil.disk_usage(OUT).free>32*2**30
        assert json.loads((ROOT/'robotwin/round7/postpilot_eval/results.json').read_text())['primary_tail_passed']
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
        torch.manual_seed(280920);np.random.seed(280920);torch.use_deterministic_algorithms(True)
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        recipe=json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        teacher=RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'],recipe['teacher_checkpoint']]:
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False);named=dict(teacher.named_parameters())
            assert saved['manifest']['base_config']==cfg
            with torch.no_grad():
                for n,v in saved['adapter'].items():named[n].copy_(v)
            del saved,named
        teacher=teacher.cuda().eval().requires_grad_(False)
        teacher.policy.checkpoint_blocks=False;teacher.flow.checkpoint_blocks=True
        frozen={n:p._version for n,p in teacher.named_parameters()}
        init=torch.load(recipe['initializers'],map_location='cpu',weights_only=False)
        students={};params={}
        for arm in ARMS:
            student=CurrentQueryStudent(teacher).cuda().requires_grad_(False)
            student.set_queries(*init['queries']['fixed_queries']);student.gate.load_state_dict(init['gate'])
            student.q_goal.requires_grad_(True);student.q_local.requires_grad_(True)
            if arm=='query_gate':student.gate.requires_grad_(True)
            student.checkpoint_blocks=True;students[arm]=student
            params[arm]={n:p for n,p in student.named_parameters() if p.requires_grad}
        control_versions={n:p._version for n,p in students['query_only'].gate.named_parameters()}
        initial_queries=tuple(x.clone() for x in init['queries']['fixed_queries']);del init
        assert len(params['query_only'])==2 and len(params['query_gate'])==44
        def frozen_check():
            assert all(p._version==frozen[n] and p.grad is None for n,p in teacher.named_parameters())
            assert all(p._version==control_versions[n] and p.grad is None for n,p in students['query_only'].gate.named_parameters())
        def mode(training):
            for student in students.values():student.train(training);student.gate.eval()
            teacher.flow.train(training);teacher.flow.DiT.eval()
        def forbidden(*a,**kw):raise AssertionError('Student called future input path')
        def prepare(b):
            with patch.object(teacher.policy,'encode_future_images',side_effect=forbidden),patch.object(teacher.policy.future,'forward',side_effect=forbidden):
                return students['query_only'].prepare({k:b[k] for k in OBSERVATION_KEYS})
        def encode(arm,prep):
            student=students[arm]
            with patch.object(teacher.policy,'encode_future_images',side_effect=forbidden),patch.object(teacher.policy.future,'forward',side_effect=forbidden):
                enc=GoalLocalPolicy.language(student,prep['current'],student.q_goal[None],student.q_local[None])
                cond=(*prep['base'],teacher.flow.goal_proj(enc['goal']),teacher.flow.local_proj(enc['local']),torch.ones(1,2,device='cuda',dtype=torch.bool))
            return cond,enc
        def action(b,cond):
            return teacher.flow.loss(b['actions'].cuda(),b['actions_mask'].cuda(),b['action_hz'].cuda(),b['embodiment_id'].cuda(),b['remaining_actions'].cuda(),cond,repeats=2)
        def gradient_groups(arm):
            groups={'query':0.,'scales':0.,'messages':0.,'relevance':0.}
            for name,p in params[arm].items():
                if p.grad is None:raise AssertionError(f'Missing gradient: {arm}/{name}')
                assert torch.isfinite(p.grad).all(),name
                k='query' if name.startswith('q_') else 'scales' if name=='gate.scales' else 'messages' if name.startswith('gate.messages.') else 'relevance'
                groups[k]+=float(p.grad.detach().double().square().sum())
            return {k:v**.5 for k,v in groups.items()}
        train=FutureActionDataset(cfg['dataset_root'],teacher.policy.statistics,'train')
        val=FutureActionDataset(cfg['dataset_root'],teacher.policy.statistics,'val')
        samples=json.loads((ROOT/'robotwin/round7/query_pilot/samples.json').read_text())
        rng=np.random.default_rng(281010)
        indices=rng.integers(0,len(train),500*4).tolist();vi=samples['validation_indices'];assert len(indices)==500*4 and len(vi)==16
        assert set(e['episode_index'] for e in train.selected).isdisjoint(e['episode_index'] for e in val.selected)
        manifest=dict(kind='paired query-only vs query+RelevanceGate action-only 500-step pilot',physical_gpu=gpu,recipe=recipe,
            initialization='same round7 fixed-query initialization, not trained student200 or old StageII3000',
            arms={a:dict(trainable_names=list(params[a]),trainable_count=sum(p.numel() for p in params[a].values())) for a in ARMS},
            total_steps=500,effective_batch=4,learning_rate=1e-4,warmup_steps=20,weight_decay=1e-8,
            train_indices=indices,validation_indices=vi,action_noise_seed='280920 + 4*zero_based_step + micro',
            same_prepared_current_features_and_teacher_diagnostic_targets=True,lambda_local_kd=0.,lambda_goal_kd=0.,
            objective='action flow matching only',backbone_and_actor_frozen=True,frozen_dit_eval=True,
            save_every=50,validation='same old16 diagnostic anchors; not independent confirmation',
            regression_guard='any metric >1.10x its step0 for two successive validations stops BOTH arms after saving',
            role_valid='constant true',student_inputs=list(OBSERVATION_KEYS),
            checkpoint_dependencies='frozen actor in recipe plus initializers for frozen query-only gate; gate arm saves all trainables and optimizer',
            next_phase='review paired fixed validation before fresh confirmation; no automatic long training or online evaluation',
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'robotwin/round7/model.py',ROOT/'goal_local/roles.py',ROOT/'goal_local/model.py']})
        atomic_json(OUT/'manifest.json',manifest)
        update(phase='decoding_validation');batches=[default_collate([val[i]]) for i in vi]
        atomic_json(OUT/'validation_anchors.json',[dict(episode=int(b['episode_id'][0]),t=int(b['t'][0])) for b in batches])
        # Preflight probes are restored and use forked RNG; no optimizer updates.
        update(phase='preflight');mode(False)
        with torch.no_grad(),ac():
            prep=prepare(batches[0]);rng0=torch.cuda.get_rng_state().clone();c0,enc0=encode('query_only',prep);rng1=torch.cuda.get_rng_state().clone();cg,encg=encode('query_gate',prep)
            diff=[float((x.float()-y.float()).abs().max()) for x,y in zip(c0,cg)]
            repeat,_=encode('query_only',prep)
            query_equal=all(torch.equal(x,y) for x,y in zip(students['query_only'].parameters(),students['query_gate'].parameters()))
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                nc0,_=encode('query_only',prep);ncg,_=encode('query_gate',prep)
            no_cache=[float((x.float()-y.float()).abs().max()) for x,y in zip(nc0,ncg)]
            diag=dict(condition_max_abs=diff,repeat_control_max_abs=[float((x.float()-y.float()).abs().max()) for x,y in zip(c0,repeat)],all_initial_parameters_equal=query_equal,no_cache_max_abs=no_cache)
            diag['cuda_rng_consumed']=not torch.equal(rng0,rng1)
            diag['teacher_training_modules']=[n for n,m in teacher.named_modules() if m.training]
            diag['encoded_layer_differences']={str(k):[float((x.float()-y.float()).abs().max()) for x,y in zip(enc0['layers'][k],encg['layers'][k])] for k in enc0['layers']}
            torch.manual_seed(280919);seed0,_=encode('query_only',prep)
            torch.manual_seed(280919);seedg,_=encode('query_gate',prep)
            diag['same_seed_max_abs']=[float((x.float()-y.float()).abs().max()) for x,y in zip(seed0,seedg)]
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                math0,_=encode('query_only',prep);mathg,_=encode('query_gate',prep)
            diag['math_backend_max_abs']=[float((x.float()-y.float()).abs().max()) for x,y in zip(math0,mathg)]
            atomic_json(OUT/'zero_gate_diagnostic.json',diag)
            assert all(torch.equal(x,y) for x,y in zip(c0,cg)),diag
            reference=students['query_gate'].conditions(prep);assert all(torch.equal(x,y) for x,y in zip(cg,reference))
        del c0,cg,reference,prep
        checks={}
        with torch.random.fork_rng(devices=[0]):
            mode(True)
            for name,scale in [('zero_scale',0.),('opened_scale',.001)]:
                for p in params['query_gate'].values():p.grad=None
                with torch.no_grad():students['query_gate'].gate.scales.fill_(scale)
                torch.manual_seed(280919)
                with ac():
                    prep=prepare(batches[0]);c,_=encode('query_gate',prep);loss=action(batches[0],c)
                assert torch.isfinite(loss);loss.backward();g=gradient_groups('query_gate');checks[name]=g
                assert g['query']>0 and g['scales']>0
                if scale==0:assert g['messages']==0 and g['relevance']==0
                else:assert g['messages']>0 and g['relevance']>0
                frozen_check();del prep,c,loss
            with torch.no_grad():students['query_gate'].gate.scales.zero_()
            for p in params['query_gate'].values():p.grad=None
        atomic_json(OUT/'preflight.json',dict(passed=True,zero_gate_equivalence_max_abs=0.,gradient_groups=checks,
            interpretation='At scale0 only scale/query gradients are nonzero; after opening, message and relevance gradients propagate.',
            teacher_frozen=True,current_only_guard=True))
        optim={a:MasterAdamW(params[a].values(),lr=1e-4,weight_decay=1e-8,offload=True) for a in ARMS}
        initial_predictions={};initial_means={};regressions={a:0 for a in ARMS}
        def save(step):
            assert shutil.disk_usage(OUT).free>24*2**30
            update(phase='saving',step=step)
            for arm in ARMS:
                dest=OUT/arm/f'step_{step:06d}';dest.mkdir(parents=True,exist_ok=True)
                payload=dict(format='robotwin_paired_query_gate_v1',arm=arm,step=step,
                    trainables={n:p.detach().cpu() for n,p in params[arm].items()},optimizer=optim[arm].state_dict(),
                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),manifest=manifest)
                torch.save(payload,dest/'student.tmp');(dest/'student.tmp').replace(dest/'student.pt')
                atomic_json(dest/'checkpoint_complete.json',dict(arm=arm,step=step,contains_optimizer=True))
            atomic_json(OUT/f'pair_{step:06d}_complete.json',dict(step=step,arms=ARMS))
        @torch.no_grad()
        def evaluate(step):
            mode(False);update(phase='validation',step=step);rows={a:[] for a in ARMS}
            with torch.random.fork_rng(devices=[0]),ac():
                for i,b in enumerate(batches):
                    update(validation_sample=i+1,validation_total=16);prep=prepare(b)
                    torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda');pred={}
                    for arm in ARMS:
                        cond,_=encode(arm,prep);pred[arm]=teacher.flow.sample(cond,noise)
                        if step==0:initial_predictions[arm,i]=pred[arm].cpu()
                        rows[arm].append(dict(sample=i,episode=int(b['episode_id'][0]),t=int(b['t'][0]),metrics=physical_metrics(pred[arm],b,teacher.policy.statistics),
                            drift_mse=float((pred[arm]-initial_predictions[arm,i].to(pred[arm])).square().mean())))
                    if step==0:assert torch.equal(pred['query_only'],pred['query_gate'])
            means={a:{k:float(np.mean([r['metrics'][k] for r in rows[a]])) for k in KEYS} for a in ARMS}
            if step==0:
                previous=json.loads((ROOT/'robotwin/round7/query_pilot/validation_000000.json').read_text())
                error=max(abs(means['query_only'][k]-previous['mean'][k]) for k in KEYS)
                assert error<1e-6,error
                atomic_json(OUT/'initial_reproduction.json',dict(passed=True,mean_metric_max_abs=error,paired_predictions_equal=True))
            atomic_json(OUT/f'validation_{step:06d}.json',dict(step=step,rows=rows,mean=means))
            record(dict(step=step,split='validation',mean=means));frozen_check();return means
        initial_means=evaluate(0);save(0)
        for step in range(500):
            started=time.time();mode(True)
            for opt in optim.values():opt.zero_grad()
            losses={a:0. for a in ARMS};kd={};anchors=[]
            for micro in range(4):
                update(phase='training',step=step,micro=micro+1)
                b=default_collate([train[indices[step*4+micro]]]);anchors.append(dict(episode=int(b['episode_id'][0]),t=int(b['t'][0])))
                with ac():prep=prepare(b)
                target=None
                if micro==0:
                    with torch.no_grad(),ac():
                        future=teacher.policy.encode_future_images(b)
                        _,target=teacher.conditions_encoded(b,dict(current=prep['current'],base=prep['base'],future=future))
                    del future
                for arm in ARMS:
                    update(arm=arm);torch.manual_seed(280920+step*4+micro)
                    with ac():cond,enc=encode(arm,prep);loss=action(b,cond)/4
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite action loss: {arm}')
                    if micro==0:
                        with torch.no_grad():
                            kd[arm]={name:float(role_distillation(enc[name].detach(),target[name],valid.cuda())) for name,valid in [('local',b['local_valid']),('goal',b['goal_frame_valid'].any(-1))]}
                    loss.backward();losses[arm]+=float(loss.detach());del cond,enc,loss
                del prep,target,b
            groups={a:gradient_groups(a) for a in ARMS}
            if step>=1:assert groups['query_gate']['messages']>0 and groups['query_gate']['relevance']>0
            lr=1e-4*min(1.,(step+1)/20)
            for arm in ARMS:
                norm=sum(v*v for v in groups[arm].values())**.5;clip=min(1.,1./max(norm,1e-12))
                for p in params[arm].values():p.grad.mul_(clip)
                for g in optim[arm].optimizer.param_groups:g['lr']=lr
                optim[arm].step()
            frozen_check();scales=students['query_gate'].gate.scales.detach().cpu().tolist()
            assert any(x!=0 for x in scales)
            if step==1:
                named=dict(students['query_gate'].named_parameters())
                init_check=torch.load(recipe['initializers'],map_location='cpu',weights_only=False)
                changes={n:float((p.detach().cpu()-init_check['gate'][n]).abs().max()) for n,p in students['query_gate'].gate.named_parameters()}
                assert changes['scales']>0 and any(v>0 for n,v in changes.items() if n.startswith('messages.'))
                atomic_json(OUT/'first_updates.json',dict(passed=True,step=2,gate_max_abs_changes=changes,gradient_groups=groups));del init_check,named
            record(dict(step=step+1,split='train',action_loss=losses,kd_diagnostic=kd,kd_weight=0.,gradient_groups=groups,
                gate_scales=scales,lr=lr,seconds=time.time()-started,anchors=anchors))
            update(step=step+1,micro=0)
            if (step+1)%50==0:
                save(step+1);measured=evaluate(step+1);guards={}
                for arm in ARMS:
                    ratios={k:measured[arm][k]/max(initial_means[arm][k],1e-10) for k in KEYS}
                    regressions[arm]=regressions[arm]+1 if any(v>1.10 for v in ratios.values()) else 0
                    guards[arm]=dict(ratio_to_initial=ratios,consecutive_regressions=regressions[arm])
                atomic_json(OUT/f'guard_{step+1:06d}.json',guards)
                if max(regressions.values())>=2:
                    update(phase='stopped_by_regression_guard',step=step+1,reason='Both arms stopped at matched step after sustained action regression');return
        frozen_check()
        atomic_json(OUT/'frozen_audit.json',dict(passed=True,teacher_tensors=len(frozen),control_gate_frozen=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        update(phase='completed',step=500,next_phase='Review paired fixed validation; no automatic fresh confirmation, long training or online rollout')
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');(OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
