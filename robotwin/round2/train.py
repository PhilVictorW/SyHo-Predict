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
from robotwin.round2.model import CompatibilityTeacher, Dataset36
from robotwin.diagnostics.migration_20260925.audit import native_velocity
from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def physical_metrics(pred, batch, stats):
    valid = batch['actions_mask'][0, :36, 0].numpy()
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
    if not out.is_relative_to(ROOT/'robotwin/round2/runs'):
        raise ValueError('Output must be under robotwin/round2/runs')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Existing run needs explicit --resume')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    seed = 92026
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
                      gate='v_original + tanh(g)*(v_goal_local_plan - v_original), g starts at 0',
                      backbone_frozen=True, learning_rate=1e-4, warmup=10,
                      arguments=vars(args), next_phase='Teacher evidence review; no automatic Stage II or online tests',
                      source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), Path(__file__).with_name('model.py'),ROOT/'robotwin/model.py',ROOT/'robotwin/data.py',ROOT/'goal_local/model.py',ROOT/'goal_local/action.py']})
        if args.resume:
            old=json.loads((out/'manifest.json').read_text())
            if old['source_sha256']!=manifest['source_sha256']: raise ValueError('Resume source mismatch')
        else: atomic_json(out/'manifest.json',manifest)
        model=CompatibilityTeacher(cfg).cuda()
        train=Dataset36(cfg['dataset_root'],model.policy.statistics,'train')
        val=Dataset36(cfg['dataset_root'],model.policy.statistics,'val')
        assert set(e['episode_index'] for e in train.selected).isdisjoint(e['episode_index'] for e in val.selected)
        # CPU boundary assertions: action padding must not cross episodes.
        dry=Dataset36(cfg['dataset_root'],model.policy.statistics,'val',decode=False)
        for episode in dry.selected[:3]:
            for t in (0,max(0,episode['length']-36),episode['length']-1):
                b=dry.sample(episode,t);n=min(36,episode['length']-t)
                assert b['actions_mask'].sum().item()==n*16 and not b['actions'][n:].any()
        del dry
        atomic_json(out/'dataset_audit.json',train.audit())
        train_indices=np.random.default_rng(seed).integers(0,len(train),args.steps*args.accumulation).tolist()
        vi=json.loads((ROOT/'robotwin/runs/stage_II_20260925/validation_samples.json').read_text())['indices'][:args.validation_samples]
        atomic_json(out/'samples.json',dict(train_indices=train_indices,validation_indices=vi))
        update(phase='decoding_validation')
        batches=[default_collate([val[i]]) for i in vi]
        donor_index=next(i for i,b in enumerate(batches) if bool(b['local_valid'][0]) and bool(b['goal_frame_valid'][0].all()))
        params={name:p for name,p in model.named_parameters() if p.requires_grad}
        frozen={name:p._version for name,p in model.named_parameters() if not p.requires_grad}
        atomic_json(out/'parameters.json',dict(trainable=sum(p.numel() for p in params.values()),
              names=list(params), frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)))

        # Preflight uses the native inherited action head, matching conditions/noise.
        update(phase='compatibility_preflight')
        model.eval()
        with torch.no_grad(),autocast():
            prepared=model.prepare(batches[0]);cond=model.conditions(batches[0],prepared)
        fpcond=tuple(x.float() if x.is_floating_point() else x for x in cond)
        hz=torch.tensor([30.],device='cuda');emb=torch.ones(1,dtype=torch.long,device='cuda')
        _,valid,_=build_time_grid(1.2,hz,50)
        noise=torch.randn(1,50,32,device='cuda');t=torch.tensor([400],device='cuda')
        with torch.no_grad():
            actual=model.flow.velocity(noise*valid[...,None],t,hz,emb,valid,fpcond)
            reference=native_velocity(model.flow,noise*valid[...,None],t,hz,emb,valid,fpcond)
            max_abs=float((actual-reference).abs().max())
            assert torch.equal(actual,reference),max_abs
        with torch.no_grad(),autocast():
            a=model.flow.sample(cond,noise);b=model.flow.sample(cond,noise,bypass=True)
            assert torch.equal(a,b)
        model.train()
        with autocast():loss=model(batches[0])
        loss.backward()
        gate_grad=float(model.flow.residual_gate.grad)
        assert math.isfinite(gate_grad) and gate_grad!=0
        assert all(p.grad is None for name,p in model.named_parameters() if name in frozen)
        for p in model.parameters():p.grad=None
        atomic_json(out/'preflight.json',dict(passed=True,native_fp32_max_abs=max_abs,
              zero_gate_sample_max_abs=0.,gate_gradient=gate_grad,frozen_backbone_has_no_grad=True,
              boundary_masks_passed=True,sampled_shape=list(a.shape),loss=float(loss.detach())))
        del prepared,cond,fpcond,loss,a,b,actual,reference
        optim=MasterAdamW(params.values(),lr=1e-4,weight_decay=1e-8,offload=True)
        start=0
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu',weights_only=False)
            assert saved['format']=='robotwin_compat36_adapter_v1'
            assert list(saved['adapter'])==list(params)
            for name,p in params.items():p.data.copy_(saved['adapter'][name].to(p))
            optim.load_state_dict(saved['optimizer']);start=saved['step']
            torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng']);del saved

        @torch.no_grad()
        def evaluate(step):
            update(phase='validation',step=step)
            model.eval();rows=[]
            with torch.random.fork_rng(devices=[0]),autocast():
                donor_batch=batches[donor_index];donor_prepared=model.prepare(donor_batch)
                for i,batch in enumerate(batches):
                    update(validation_sample=i+1,validation_total=len(batches))
                    prepared=model.prepare(batch)
                    cond=model.conditions(batch,prepared)
                    torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
                    pred=model.flow.sample(cond,noise)
                    base=model.flow.sample(cond,noise,bypass=True)
                    off=model.flow.sample(model.conditions(batch,prepared,roles_off=True),noise)
                    wrong=model.flow.sample(model.conditions(batch,prepared,donor=(donor_batch,donor_prepared)),noise)
                    rows.append(dict(sample=i,episode=int(batch['episode_id'][0]),t=int(batch['t'][0]),
                        donor_same_episode=int(batch['episode_id'][0])==int(donor_batch['episode_id'][0]),
                        metrics={k:physical_metrics(v,batch,model.policy.statistics) for k,v in [('full',pred),('base',base),('roles_off',off),('wrong_future',wrong)]},
                        wrong_future_relative_l2=float((pred-wrong).norm()/pred.norm().clamp_min(1e-12))))
            aggregate={variant:{key:float(np.mean([r['metrics'][variant][key] for r in rows]))
                       for key in rows[0]['metrics'][variant]} for variant in rows[0]['metrics']}
            atomic_json(out/f'validation_{step:06d}.json',dict(step=step,rows=rows,mean=aggregate,gate=float(model.flow.residual_gate.tanh())))
            record(dict(step=step,split='validation',mean=aggregate,gate=float(model.flow.residual_gate.tanh())))
            return aggregate

        def save(step):
            update(phase='saving',step=step)
            dest=out/f'step_{step:06d}';dest.mkdir(exist_ok=True)
            bundle=dict(format='robotwin_compat36_adapter_v1',step=step,baseline=cfg['baseline_checkpoint'],
                adapter={name:p.detach().cpu() for name,p in params.items()},optimizer=optim.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),manifest=manifest)
            torch.save(bundle,dest/'adapter.tmp');(dest/'adapter.tmp').replace(dest/'adapter.pt')
            atomic_json(dest/'checkpoint_complete.json',dict(step=step,contains_optimizer=True,requires_frozen_original_baseline=True))

        evaluate(start)
        duration=[];skipped=0
        step=start
        while step<args.steps:
            begin=time.time();model.train();optim.zero_grad();value=0.
            for micro in range(args.accumulation):
                update(phase='training',step=step,micro=micro+1)
                index=train_indices[step*args.accumulation+micro]
                batch=default_collate([train[index]])
                with autocast():loss=model(batch)/args.accumulation
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no optimizer update performed')
                loss.backward();value+=float(loss.detach())
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
            record(dict(step=step+1,split='train',flow=value,grad_norm=float(norm),lr=lr,
                        gate=float(model.flow.residual_gate.tanh()),seconds=duration[-1]))
            update(step=step+1,micro=0,eta_seconds=(args.steps-step-1)*float(np.mean(duration[-20:])))
            if (step+1)%args.validate_every==0 or step+1==args.steps:
                save(step+1);evaluate(step+1)
            step+=1
        update(phase='completed_teacher_pilot',step=args.steps,eta_seconds=0,
               next_phase='Review teacher full/base/roles_off/wrong_future; do not start student or online eval automatically')
        atomic_json(out/'frozen_audit.json',dict(passed=True,checked_step=args.steps,frozen_tensor_count=len(frozen),
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
