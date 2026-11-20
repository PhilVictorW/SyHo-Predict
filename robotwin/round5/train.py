"""Small Stage I experiment, with fail-closed compatibility and teacher checks."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from goal_local.optim import MasterAdamW
from robotwin.round5.model import BalancedTeacher as CompatibilityTeacher, FutureActionDataset as Dataset36, COMPONENTS, balanced_weights
from robotwin.round4.audit_split_paths import sample as audited_split_sample
from robotwin.diagnostics.migration_20260925.audit import native_velocity
from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def physical_metrics(pred, batch, stats):
    valid = batch['observed_actions_mask'][0, :36, 0].numpy()
    def convert(x):
        return ModelClient.unnormalize_actions(x[0, :36, :16].float().cpu().numpy(), stats['agilex']['action'],
                gripper_indices=(), passthrough_indices=(3,4,5,6,11,12,13,14), invert_indices=(7,15))[valid]
    a, b = convert(pred), convert(batch['actions'])
    position = np.stack([np.linalg.norm(a[:, i:i+3]-b[:, i:i+3], axis=-1) for i in (0,8)], -1).mean()*100
    rotation = np.stack([2*np.arccos(np.clip(np.abs((a[:, i:i+4]*b[:, i:i+4]).sum(-1)), 0,1))*180/np.pi for i in (3,11)], -1).mean()
    return dict(position_cm=float(position), rotation_deg=float(rotation), gripper_mae=float(np.abs(a[:, [7,15]]-b[:, [7,15]]).mean()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant',choices=['equal','balanced'],required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--validate-every', type=int, default=50)
    parser.add_argument('--validation-samples', type=int, default=16)
    parser.add_argument('--resume', help='Complete adapter checkpoint, including optimizer and RNG')
    args = parser.parse_args()
    if min(args.steps,args.accumulation,args.validate_every,args.validation_samples) < 1:
        parser.error('All counts must be positive')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('This isolated pilot is authorized on physical GPU0 only')
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'robotwin/round5/runs'):
        raise ValueError('Output must be under robotwin/round5/runs')
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.disk_usage(out).free < 40*2**30:raise RuntimeError('Need at least 40GiB free before starting an arm')
    if (out/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Existing run needs explicit --resume')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    seed = 260929
    torch.manual_seed(seed); np.random.seed(seed)
    status = dict(pid=os.getpid(), physical_gpu=0, phase='loading_model', step=0, total_steps=args.steps,
                  micro=0, started_at=time.time(), automatic_online_evaluation=False)
    def update(**values):
        status.update(values); status['updated_at']=time.time(); atomic_json(out/'status.json', status)
    def record(row):
        row['at']=time.time()
        with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def autocast(): return torch.autocast('cuda',dtype=torch.bfloat16)
    update()
    try:
        cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        manifest=dict(base_config=cfg, seed=seed, prediction_steps=36, action_dimensions=16, raw_buffer=[50,32],
                      future_offsets=[8,24,40,56,72,88], original_condition_paths_restored=True,
                      gate='fixed Plan0/Role3, Plan scaffold and saved residual gate frozen',
                      backbone_frozen=True, learning_rate=1e-4, warmup=10,
                      inherited_teacher_step=400,auxiliary=True,variant=args.variant,
                      aux_max_weight=.001,aux_warmup=20,aux_gradient_cap='weighted auxiliary norm <= action norm on sampled resampler query parameters',
                      terminal_padding='repeat last absolute action, keep 36 attention/supervision tokens',
                      comparison='same round4 teacher200, fixed Plan0/Role3, seed, anchors, LR and steps; only auxiliary component weighting differs',
                      arguments=vars(args), next_phase='Teacher evidence review; no automatic Stage II or online tests',
                      source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), Path(__file__).with_name('model.py'),ROOT/'robotwin/round4/model.py',ROOT/'robotwin/round3/model.py',ROOT/'robotwin/round2/model.py',ROOT/'robotwin/model.py',ROOT/'robotwin/data.py',ROOT/'goal_local/model.py',ROOT/'goal_local/action.py']})
        if args.resume:
            old=json.loads((out/'manifest.json').read_text())
            if old['source_sha256']!=manifest['source_sha256']: raise ValueError('Resume source mismatch')
        else: atomic_json(out/'manifest.json',manifest)
        diagnosis=json.loads((ROOT/'robotwin/round3/diagnostics/tail_control.json').read_text())
        assert diagnosis.get('completed') and diagnosis['reference_padding_checks']>=20
        assert diagnosis['summary']['tail']['fixed'] < diagnosis['summary']['tail']['old']
        assert diagnosis['summary']['interior']['fixed'] == diagnosis['summary']['interior']['old']
        model=CompatibilityTeacher(cfg)
        source=ROOT/'robotwin/round4/runs/future_aux/step_000200/adapter.pt'
        payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        assert payload['format']=='robotwin_future_aux36_adapter_v1' and payload['step']==200
        assert payload['manifest']['base_config']==cfg
        named=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in payload['adapter'].items():named[name].copy_(value)
        del payload,named
        model=model.cuda()
        atomic_json(out/'source.json',dict(path=str(source),step=200,variant=args.variant))
        train=Dataset36(cfg['dataset_root'],model.policy.statistics,'train')
        val=Dataset36(cfg['dataset_root'],model.policy.statistics,'val')
        assert set(e['episode_index'] for e in train.selected).isdisjoint(e['episode_index'] for e in val.selected)
        # CPU boundary assertions: action padding must not cross episodes.
        dry=Dataset36(cfg['dataset_root'],model.policy.statistics,'val',decode=False)
        for episode in dry.selected[:3]:
            for t in (0,max(0,episode['length']-36),episode['length']-1):
                b=dry.sample(episode,t);n=min(36,episode['length']-t)
                assert b['actions_mask'].sum().item()==36*16 and not b['actions'][36:].any()
                assert b['observed_actions_mask'].sum().item()==n*16
                assert torch.equal(b['action_chunk'][n:],b['action_chunk'][n-1:n].expand(36-n,-1))
                assert torch.equal(b['future_action_valid'],b['future_indices']<episode['length'])
                assert not b['future_action_target'][~b['future_action_valid']].any()
        del dry
        atomic_json(out/'dataset_audit.json',train.audit())
        train_indices=np.random.default_rng(seed).integers(0,len(train),args.steps*args.accumulation).tolist()
        vi=json.loads((ROOT/'robotwin/runs/stage_II_20260925/validation_samples.json').read_text())['indices'][:args.validation_samples]
        atomic_json(out/'samples.json',dict(train_indices=train_indices,validation_indices=vi))
        update(phase='decoding_validation')
        batches=[default_collate([val[i]]) for i in vi]
        donor_index=next(i for i,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all()))
        # Calibrate relative component scales on TRAIN episodes only, with no updates.
        common=ROOT/'robotwin/round5/calibration.json'
        calibration_indices=[]
        if common.exists():
            calibration=json.loads(common.read_text())
            assert calibration['source']==str(source) and calibration['seed']==260928
        else:
            update(phase='calibrating_auxiliary')
            rng=np.random.default_rng(260928);collected=[];model.eval()
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),autocast():
                while len(collected)<32:
                    index=int(rng.integers(len(train)))
                    batch=default_collate([train[index]])
                    if not bool(batch['future_action_valid'].any()):continue
                    prep=model.prepare(batch);_,enc=model.conditions_encoded(batch,prep)
                    losses=model.auxiliary_loss(batch,enc)
                    collected.append({k:float(losses[k]) for k in COMPONENTS})
                    calibration_indices.append(index)
                    update(calibration_completed=len(collected),calibration_total=32)
            means={k:float(np.mean([x[k] for x in collected])) for k in COMPONENTS}
            calibration=dict(source=str(source),seed=260928,split='train',indices=calibration_indices,
                means=means,weights=balanced_weights(means),rule='inverse initial component mean, normalized to sum 3; floor 1e-6')
            atomic_json(common,calibration)
        model.component_weights=tuple(calibration['weights'] if args.variant=='balanced' else [1.,1.,1.])
        atomic_json(out/'component_weights.json',dict(variant=args.variant,weights=model.component_weights,calibration=calibration))
        params={name:p for name,p in model.named_parameters() if p.requires_grad}
        frozen={name:p._version for name,p in model.named_parameters() if not p.requires_grad}
        atomic_json(out/'parameters.json',dict(trainable=sum(p.numel() for p in params.values()),
              names=list(params), frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)))

        # Compare the trainable implementation with the already verified read-only candidate.
        update(phase='compatibility_preflight');model.eval()
        with torch.no_grad(),autocast():
            prepared=model.prepare(batches[0]);cond=model.conditions(batches[0],prepared)
            torch.manual_seed(800);noise=torch.randn(1,50,32,device='cuda')
            actual=model.flow.sample(cond,noise)
            reference=audited_split_sample(model.flow,cond,noise,0,3)
            assert torch.equal(actual,reference),float((actual-reference).abs().max())
            measured=physical_metrics(actual,batches[0],model.policy.statistics)
            model.flow.role_multiplier=0.
            zero=model.flow.sample(cond,noise);base=model.flow.sample(cond,noise,bypass=True)
            assert torch.equal(zero,base)
            model.flow.role_multiplier=3.
        model.train()
        with autocast():result=model(batches[0]);loss=result['flow']+1e-5*result['aux']
        loss.backward()
        assert all(p.grad is None for name,p in model.named_parameters() if name in frozen)
        assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.local_head.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.goal_head.parameters())
        for p in model.parameters():p.grad=None
        atomic_json(out/'preflight.json',dict(passed=True,audited_candidate_max_abs=0.,
            zero_role_base_max_abs=0.,frozen_backbone_has_no_grad=True,plan_and_gate_frozen=True,
            boundary_masks_passed=True,future_label_masks_passed=True,source_physical_metrics=measured,
            sampled_shape=list(actual.shape),loss=float(loss.detach()),component_weights=model.component_weights))
        del prepared,cond,result,loss,actual,reference,zero,base
        optim=MasterAdamW(params.values(),lr=1e-4,weight_decay=1e-8,offload=True)
        start=0;aux_cap=.001
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu',weights_only=False)
            assert saved['format']=='robotwin_balanced_centered36_adapter_v1'
            assert tuple(saved['component_weights'])==model.component_weights
            aux_cap=saved['aux_cap']
            assert list(saved['adapter'])==list(params)
            for name,p in params.items():p.data.copy_(saved['adapter'][name].to(p))
            optim.load_state_dict(saved['optimizer']);start=saved['step']
            torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng']);del saved

        @torch.no_grad()
        def evaluate(step):
            update(phase='validation',step=step)
            model.eval();rows=[]
            with torch.random.fork_rng(devices=[0]),autocast():
                donors=[j for j,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all())]
                assert len(donors)>=2
                for i,batch in enumerate(batches):
                    update(validation_sample=i+1,validation_total=len(batches))
                    prepared=model.prepare(batch)
                    donor_index=next(j for j in donors if int(batches[j]['episode_id'][0])!=int(batch['episode_id'][0]))
                    donor_batch=batches[donor_index];donor_prepared=model.prepare(donor_batch)
                    cond,encoded=model.conditions_encoded(batch,prepared)
                    wrong_cond,wrong_encoded=model.conditions_encoded(batch,prepared,donor=(donor_batch,donor_prepared))
                    aux_metrics={name:{k:float(v) for k,v in model.auxiliary_loss(batch,enc).items()} for name,enc in [('full',encoded),('wrong',wrong_encoded)]} if bool(batch['future_action_valid'].any()) else None
                    torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
                    pred=model.flow.sample(cond,noise)
                    base=model.flow.sample(cond,noise,bypass=True)
                    off=model.flow.sample(model.conditions(batch,prepared,roles_off=True),noise)
                    wrong=model.flow.sample(wrong_cond,noise)
                    rows.append(dict(sample=i,episode=int(batch['episode_id'][0]),t=int(batch['t'][0]),auxiliary=aux_metrics,
                        donor_same_episode=int(batch['episode_id'][0])==int(donor_batch['episode_id'][0]),
                        metrics={k:physical_metrics(v,batch,model.policy.statistics) for k,v in [('full',pred),('base',base),('roles_off',off),('wrong_future',wrong)]},
                        wrong_future_relative_l2=float((pred-wrong).norm()/pred.norm().clamp_min(1e-12))))
            aggregate={variant:{key:float(np.mean([r['metrics'][variant][key] for r in rows]))
                       for key in rows[0]['metrics'][variant]} for variant in rows[0]['metrics']}
            atomic_json(out/f'validation_{step:06d}.json',dict(step=step,rows=rows,mean=aggregate,gate=float(model.flow.residual_gate.tanh())))
            record(dict(step=step,split='validation',mean=aggregate,gate=float(model.flow.residual_gate.tanh())))
            return aggregate

        def save(step):
            import shutil
            if shutil.disk_usage(out).free < 32*2**30:
                raise RuntimeError('Disk reserve below 32GiB; last complete checkpoint retained')
            update(phase='saving',step=step)
            dest=out/f'step_{step:06d}';dest.mkdir(exist_ok=True)
            bundle=dict(format='robotwin_balanced_centered36_adapter_v1',step=step,source=str(source),component_weights=model.component_weights,aux_cap=aux_cap,baseline=cfg['baseline_checkpoint'],
                adapter={name:p.detach().cpu() for name,p in params.items()},optimizer=optim.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),manifest=manifest)
            torch.save(bundle,dest/'adapter.tmp');(dest/'adapter.tmp').replace(dest/'adapter.pt')
            atomic_json(dest/'checkpoint_complete.json',dict(step=step,contains_optimizer=True,requires_frozen_original_baseline=True))

        evaluate(start)
        duration=[];skipped=0
        probes=[model.policy.future.local.query,model.policy.future.goal.query]
        step=start
        while step<args.steps:
            begin=time.time();model.train();optim.zero_grad();value=0.;aux_value=0.;objective=0.;gradient_probe=None;components={k:0. for k in COMPONENTS}
            for micro in range(args.accumulation):
                update(phase='training',step=step,micro=micro+1)
                index=train_indices[step*args.accumulation+micro]
                batch=default_collate([train[index]])
                with autocast():result=model(batch)
                if model.auxiliary and micro==0 and step%50==0:
                    action_grad=torch.autograd.grad(result['flow'],probes,retain_graph=True,allow_unused=True)
                    aux_grad=torch.autograd.grad(result['aux'],probes,retain_graph=True,allow_unused=True)
                    component_grads={k:torch.autograd.grad(result[k],probes,retain_graph=True,allow_unused=True) for k in COMPONENTS}
                    an=math.sqrt(sum(float(g.float().square().sum()) for g in action_grad if g is not None))
                    dn=math.sqrt(sum(float(g.float().square().sum()) for g in aux_grad if g is not None))
                    if dn>0:aux_cap=min(.001,an/dn)
                    gradient_probe=dict(action_norm=an,auxiliary_norm=dn,cap=aux_cap,scope='local/goal resampler query parameters only',component_norms={k:math.sqrt(sum(float(g.float().square().sum()) for g in gs if g is not None)) for k,gs in component_grads.items()})
                weight=min(.001*min(1.,(step+1)/20),aux_cap) if model.auxiliary else 0.
                loss=(result['flow']+weight*result['aux'])/args.accumulation
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no optimizer update performed')
                loss.backward();value+=float(result['flow'].detach())/args.accumulation
                aux_value+=float(result['aux'].detach())/args.accumulation
                objective+=float(loss.detach())
                for k in COMPONENTS:components[k]+=float(result[k].detach())/args.accumulation
            # FP64 accumulation prevents a finite FP32 gradient from overflowing the norm.
            norm=torch.stack([p.grad.detach().double().square().sum() for p in params.values() if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(norm):
                optim.zero_grad();skipped+=1
                record(dict(step=step,split='skipped',reason='nonfinite_gradient',skipped=skipped))
                if skipped>=3:raise FloatingPointError('Three nonfinite updates; review required')
                continue
            scale=min(1.,1./max(float(norm),1e-12))
            for p in params.values():
                if p.grad is not None:p.grad.mul_(scale)
            lr=1e-4*min(1.,(step+1)/10)
            for group in optim.optimizer.param_groups:group['lr']=lr
            update(phase='optimizer_step');optim.step()
            assert all(p._version==frozen[name] and p.grad is None for name,p in model.named_parameters() if name in frozen)
            duration.append(time.time()-begin)
            record(dict(step=step+1,split='train',flow=value,auxiliary=aux_value,loss=objective,lambda_aux=weight,aux_gradient_probe=gradient_probe,components=components,component_weights=model.component_weights,cumulative_teacher_step=401+step,grad_norm=float(norm),lr=lr,
                        gate=float(model.flow.residual_gate.tanh()),seconds=duration[-1]))
            update(step=step+1,micro=0,eta_seconds=(args.steps-step-1)*float(np.mean(duration[-20:])))
            if (step+1)%args.validate_every==0 or step+1==args.steps:
                save(step+1);evaluate(step+1)
            step+=1
        update(phase='completed_teacher_pilot',step=args.steps,eta_seconds=0,
               next_phase='Matched arm assessment; no student or online rollout automatically')
        atomic_json(out/'frozen_audit.json',dict(passed=True,checked_step=args.steps,frozen_tensor_count=len(frozen),
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
