"""RoboTwin Stage II DDP: frozen posterior teacher, independent student, CPU AdamW."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from goal_local.optim import MasterAdamW
from .train_distributed import atomic_json, broadcast_parameters, rng_state, restore_rng, OffsetSampler, validation_indices
from .stage_two_utils import finite_clip, write_eval_sidecars

KEYS = ('loss','flow','local_distill','goal_distill')

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stage-one',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--optimizer-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--eval-batches',type=int,default=64)
    p.add_argument('--warmup',type=int,default=400)
    p.add_argument('--gpu-memory-fraction',type=float,default=.36)
    p.add_argument('--max-consecutive-skips',type=int,default=10)
    a=p.parse_args()
    rank,world,local_rank=map(lambda k:int(os.environ[k]),('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=2 or a.accumulation*world!=32: p.error('Expected two GPUs with global batch 32')
    root=Path(__file__).resolve().parents[1]; out=a.output.resolve(); source=a.stage_one.resolve()
    if not out.is_relative_to(root/'robotwin/runs'):p.error('Output must be in robotwin/runs')
    if min(a.steps,a.accumulation,a.save_every,a.eval_every,a.eval_batches)<1:p.error('Positive counts required')
    if rank==0 and (out/'metrics.jsonl').exists() and not a.resume:p.error('Existing metrics require --resume')
    torch.set_num_threads(a.cpu_threads)
    device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=45),device_id=device)
    # Some framework imports initialize accelerate.PartialState, so import after NCCL.
    from .model import RobotwinPolicy
    from .data import FutureDataset
    from goal_local.model import make_stage_two
    status=dict(stage='II',phase='loading_teacher',step=0,micro=0,total_steps=a.steps,
        started_at=time.time(),pid=os.getpid(),gpu_ids=[2,5],world_size=world,effective_batch=32,
        accumulation_per_gpu=a.accumulation,skipped_updates=0,teacher_source=str(source))
    def update(**kw):
        if rank==0:
            status.update(kw);status['updated_at']=time.time();atomic_json(out/'status.json',status)
    def record(row,filename='metrics.jsonl'):
        if rank==0:
            row['time']=time.time();print(json.dumps(row),flush=True)
            with (out/filename).open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        if rank==0:
            out.mkdir(parents=True,exist_ok=True)
            atomic_json(out/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()});update()
        seed=2026;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        posterior=RobotwinPolicy.load(source)
        if posterior.source_step!=1000 or posterior.mode!='posterior':raise ValueError('Expected Stage I step 1000 posterior')
        system=make_stage_two(posterior);student,teacher=system.student,system.teacher
        cfg=student.cfg;cfg['warmup_steps']=a.warmup
        teacher_step=posterior.source_step
        start=attempt=skipped=0;best=float('inf');saved=None
        if a.resume:
            ckpt=a.resume.absolute();saved=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
            if saved['stage']!='II' or saved['teacher_source']!=str(source) or saved['config']!=cfg:raise ValueError('Resume provenance mismatch')
            resumed=RobotwinPolicy.load(ckpt.parent)
            student.load_state_dict(resumed.state_dict());start=saved['step'];del resumed
            attempt=saved['attempt'];skipped=saved.get('skipped_updates',0);best=saved['best_validation']
            if rank==0 and (out/'metrics.jsonl').exists():
                text=(out/'metrics.jsonl').read_text()
                (out/f'metrics_before_resume_{int(time.time())}.jsonl').write_text(text)
                keep=[line for line in text.splitlines() if json.loads(line)['step']<=start]
                (out/'metrics.jsonl').write_text('\n'.join(keep)+'\n')
        system.to(device)
        parameters=[p for p in student.parameters() if p.requires_grad]
        teacher_params=dict(teacher.named_parameters())
        teacher_ptrs={p.data_ptr() for p in teacher_params.values()}
        assert all(not p.requires_grad for p in teacher_params.values())
        assert not any(p.data_ptr() in teacher_ptrs for p in student.parameters())
        model=DDP(system,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,
            find_unused_parameters=True,gradient_as_bucket_view=True,bucket_cap_mb=16)
        versions={k:p._version for k,p in teacher_params.items()}
        def audit(step):
            assert not teacher.training
            assert all(not p.requires_grad and p.grad is None and p._version==versions[k] for k,p in teacher_params.items())
            if rank==0:atomic_json(out/'teacher_freeze_audit.json',dict(teacher_step=teacher_step,
                all_frozen=True,excluded_from_optimizer=True,independent_storage=True,
                no_teacher_gradients=True,parameter_versions_unchanged=True,checked_after_student_step=step))
        ds=FutureDataset(cfg['dataset_root'],student.statistics,'train',seed,cfg['val_ratio'])
        val=FutureDataset(cfg['dataset_root'],student.statistics,'val',seed,cfg['val_ratio'])
        assert {e['episode_index'] for e in ds.selected}.isdisjoint({e['episode_index'] for e in val.selected})
        sampler=DistributedSampler(ds,num_replicas=world,rank=rank,seed=seed,drop_last=True)
        epoch,offset=divmod(attempt*a.accumulation,len(sampler));sampler.set_epoch(epoch)
        options=dict(num_workers=a.workers,pin_memory=True)
        if a.workers:options.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
        def loader_at(offset):
            return DataLoader(ds,batch_size=1,sampler=OffsetSampler(sampler,offset),
                generator=torch.Generator().manual_seed(seed+rank+200),**options)
        loader=loader_at(offset)
        vi=validation_indices(val,a.eval_batches,seed+1)
        validation=DataLoader(Subset(val,vi),batch_size=1,generator=torch.Generator().manual_seed(seed+2),**options) if rank==0 else None
        optim=scheduler=None
        if rank==0:
            update(phase='initializing_fp32_optimizer')
            optim=MasterAdamW(parameters,cfg['learning_rate'],cfg['weight_decay'],offload=True)
            def lr(step):
                w=a.warmup
                return min(1,(step+1)/w)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-w)/max(1,a.steps-w))))
            scheduler=torch.optim.lr_scheduler.LambdaLR(optim.optimizer,lr)
            if saved is not None:
                optim.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            write_eval_sidecars(out,student.statistics)
            atomic_json(out/'config.json',cfg)
            atomic_json(out/'teacher_source.json',dict(path=str(source),step=teacher_step,optimizer_reset=not bool(a.resume),
                source_architecture='RobotwinPolicy',student_architecture='RobotwinPolicy',losses=list(KEYS)))
            atomic_json(out/'validation_samples.json',dict(indices=vi,seed=seed+1,episodes=val.validation_episodes))
            atomic_json(out/'dataset_audit.json',{**ds.audit(),'train_episodes':len(ds.selected),'validation_frames':len(val),'effective_batch':32})
            atomic_json(out/'parameter_audit.json',dict(trainable=sum(p.numel() for p in parameters),
                teacher_frozen=sum(p.numel() for p in teacher.parameters()),optimizer='FP32 master AdamW on rank zero CPU',
                gpu_ids=[2,5],memory_fraction=a.gpu_memory_fraction))
        random.seed(seed+rank+start);np.random.seed(seed+rank+start);torch.manual_seed(seed+rank+start)
        if saved is not None:restore_rng(saved['distributed_rng'][rank],device)
        del saved
        broadcast_parameters(parameters);audit(start)
        def evaluate(step):
            if rank!=0:return None
            system.eval();values=[]
            with torch.random.fork_rng(devices=[local_rank]),torch.no_grad():
                torch.manual_seed(seed+10000)
                for i,batch in enumerate(validation):
                    update(phase=f'validation {i+1}/{a.eval_batches}',step=step,micro=0)
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=system(batch)
                    values.append({k:float(result[k]) for k in KEYS})
            row={k:sum(v[k] for v in values)/len(values) for k in KEYS}
            if not all(math.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite fixed validation')
            record(dict(step=step,split='validation',**row));return row['flow']
        score=evaluate(start)
        if rank==0:best=min(best,score)
        dist.barrier();iterator=iter(loader);step=start;consecutive=0;durations=[]
        while step<a.steps:
            begin=time.time();model.train();system.zero_grad(set_to_none=True)
            if rank==0:optim.zero_grad()
            metrics=torch.zeros(len(KEYS)+2,device=device,dtype=torch.float64);samples=[]
            for micro in range(a.accumulation):
                update(phase='training',step=step,micro=micro+1,attempt=attempt)
                try:batch=next(iterator)
                except StopIteration:
                    epoch+=1;sampler.set_epoch(epoch);loader=loader_at(0);iterator=iter(loader);batch=next(iterator)
                samples.append(dict(episode=int(batch['episode_id'][0]),t=int(batch['t'][0])))
                metrics[-2]+=batch['local_valid'].sum().to(device);metrics[-1]+=batch['goal_frame_valid'].any(-1).sum().to(device)
                with model.no_sync() if micro+1<a.accumulation else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model(batch);loss=result['loss']/a.accumulation
                    loss.backward()
                for i,k in enumerate(KEYS):metrics[i]+=result[k].detach().double()/a.accumulation
            attempt+=1;dist.all_reduce(metrics)
            norm=finite_clip(parameters,cfg['gradient_clip']) if rank==0 else torch.zeros((),device=device,dtype=torch.float64)
            finite=torch.tensor(int(torch.isfinite(metrics).all() and (rank!=0 or torch.isfinite(norm))),device=device)
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():
                skipped+=1;consecutive+=1
                details=[None]*world if rank==0 else None
                dist.gather_object(dict(rank=rank,samples=samples),details,dst=0)
                if rank==0:
                    bad=[n for n,p in student.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                    record(dict(step=step,attempt=attempt,kind='nonfinite_update_skipped',grad_norm=str(float(norm)),
                        metrics={k:str(float(metrics[i]/world)) for i,k in enumerate(KEYS)},bad_parameters=bad,batches=details),'anomalies.jsonl')
                system.zero_grad(set_to_none=True)
                update(phase='skipping_nonfinite_update',skipped_updates=skipped)
                if consecutive>=a.max_consecutive_skips:raise FloatingPointError('Repeated nonfinite updates; inspect anomalies.jsonl')
                continue
            consecutive=0
            if rank==0:update(phase='optimizer_step');optim.step();scheduler.step()
            broadcast_parameters(parameters);torch.cuda.synchronize(device);step+=1;elapsed=time.time()-begin;durations.append(elapsed)
            audit(step)
            if rank==0:
                record(dict(step=step,split='train',**{k:float(metrics[i]/world) for i,k in enumerate(KEYS)},
                    grad_norm=float(norm),lr=optim.optimizer.param_groups[0]['lr'],update_seconds=elapsed,
                    world_size=world,local_valid_fraction=float(metrics[-2])/32,goal_valid_fraction=float(metrics[-1])/32,
                    skipped_updates=skipped,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30))
                update(step=step,micro=0,skipped_updates=skipped,eta_seconds=(a.steps-step)*sum(durations[-20:])/len(durations[-20:]))
            if step==1 and not a.resume:
                if rank==0:
                    update(phase='saving_first_update');student.save(out/'step_000001',step,extra=dict(stage='II',teacher_source=str(source),smoke_only=True))
                    atomic_json(out/'step_000001/model_complete.json',dict(step=step,smoke_only=True))
                dist.barrier()
            if step%a.eval_every==0:
                score=evaluate(step)
                if rank==0 and score<best:
                    best=score;update(phase='saving_best');student.save(out/'best',step,extra=dict(stage='II',validation_flow=score))
                dist.barrier()
            if step%a.save_every==0 or step==a.steps:
                states=[None]*world if rank==0 else None;dist.gather_object(rng_state(device),states,dst=0)
                if rank==0:
                    dest=out/f'step_{step:06d}';update(phase='saving_checkpoint')
                    student.save(dest,step,extra=dict(stage='II',teacher_source=str(source),teacher_step=teacher_step))
                    trainer=dict(stage='II',config=cfg,teacher_source=str(source),step=step,attempt=attempt,skipped_updates=skipped,
                        optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),best_validation=best,distributed_rng=states,world_size=world)
                    opt_dir=a.optimizer_dir.resolve()/dest.name;opt_dir.mkdir(parents=True,exist_ok=True)
                    torch.save(trainer,opt_dir/'trainer.tmp');(opt_dir/'trainer.tmp').replace(opt_dir/'trainer.pt');del trainer
                    (dest/'trainer.pt').symlink_to(opt_dir/'trainer.pt')
                    atomic_json(dest/'checkpoint_complete.json',dict(step=step,stage='II',owner='robotwin.train_stage_two',eval_episodes=1000))
                    for marker in sorted(out.glob('step_*/checkpoint_complete.json'))[:-2]:
                        old=marker.parent/'trainer.pt'
                        if old.exists():
                            target=old.resolve();old.unlink()
                            if target.is_relative_to(a.optimizer_dir.resolve()):target.unlink()
                    update(phase='training',checkpoint=str(dest))
                dist.barrier()
        update(phase='completed',step=step,eta_seconds=0)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
