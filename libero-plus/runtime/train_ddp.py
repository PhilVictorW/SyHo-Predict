"""Stage-I DDP orchestration; shares the original GoalLocalPolicy/loss/master AdamW."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent))
from indexed_future import IndexedFutureDataset
from goal_local.optim import MasterAdamW


def atomic_json(path,payload):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2))
    temporary.replace(path)


def validation_indices(dataset,count,seed):
    from collections import defaultdict
    groups=defaultdict(list)
    for i,e in enumerate(dataset.selected):
        groups[tuple(e['tasks'])].append(i)
    rng=np.random.default_rng(seed); out=[]; keys=sorted(groups)
    for k in range(count):
        ei=int(rng.choice(groups[keys[k%len(keys)]])); e=dataset.selected[ei]
        t=int(rng.integers(e['length'])); out.append(int(dataset.offsets[ei]+t))
    return out


class DistributedBatches:
    """Resume consumed batches without decoding skipped examples or replaying prefetch."""
    def __init__(self, dataset, batch_size, world, rank, seed):
        self.sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=seed, shuffle=True, drop_last=True)
        self.batches = BatchSampler(self.sampler, batch_size, drop_last=True)
        self.start = 0

    def reset(self, epoch, start=0):
        self.sampler.set_epoch(epoch)
        self.start = start

    def __iter__(self):
        return itertools.islice(iter(self.batches), self.start, None)

    def __len__(self):
        return len(self.batches)-self.start


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/stage_I.json')
    p.add_argument('--output', default='runs/stage_I_20260923')
    p.add_argument('--steps', type=int, default=16000)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--eval-batches', type=int, default=64, help='Global validation batch count, split across ranks')
    p.add_argument('--diagnostic-batches', type=int, default=8, help='Global diagnostic batch count')
    p.add_argument('--keep-checkpoints', type=int, default=2)
    p.add_argument('--resume')
    args = p.parse_args()
    local_rank, rank, world = (int(os.environ[k]) for k in ('LOCAL_RANK','RANK','WORLD_SIZE'))
    if world != 2:
        raise ValueError('This experiment is configured and checked for two DDP ranks')
    if min(args.steps,args.batch_size,args.accumulation,args.save_every,args.eval_every,args.eval_batches,args.keep_checkpoints) < 1:
        raise ValueError('Counts must be positive')
    if args.eval_batches % world:
        raise ValueError('Validation batches must be divisible by world size')
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    output = Path(args.output).resolve()
    if not output.is_relative_to(WORK/'runs'):
        raise ValueError('Output must stay inside this Plus workspace runs directory')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Existing metrics require explicit resume')
    status = dict(phase='initializing_distributed',step=0,micro=0,total_steps=args.steps,
        started_at=time.time(),pid=os.getpid(),rank=rank,world_size=world,physical_gpus=[1,3])
    def update(**kwargs):
        status.update(kwargs); status['updated_at']=time.time()
        atomic_json(output/f'status_rank{rank}.json',status)
        if rank == 0:
            atomic_json(output/'status.json',status)
    def record(row):
        if rank == 0:
            row['time']=time.time()
            print(json.dumps(row),flush=True)
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
    update()
    try:
        dist.init_process_group('nccl',timeout=timedelta(minutes=60))
        # starVLA logging constructs Accelerate PartialState at import; import only
        # after explicit setup, and never in spawned DataLoader worker imports.
        from goal_local.model import GoalLocalPolicy
        cfg=json.loads(Path(args.config).read_text()); seed=cfg['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if rank == 0:
            atomic_json(output/'arguments.json',dict(vars(args),stage='I',world_size=world,physical_gpus=[1,3],
                effective_batch=world*args.batch_size*args.accumulation,optimizer_device='cpu'))
            atomic_json(output/'launch.json',dict(stage='I',physical_gpu='1,3 (DDP)',steps=args.steps,
                batch_size=args.batch_size,accumulation=args.accumulation,world_size=world,effective_batch=32))
        update(phase='loading_model')
        policy=GoalLocalPolicy.load(Path(args.resume).parent) if args.resume else GoalLocalPolicy.from_baseline(cfg)
        if policy.mode!='posterior' or policy.cfg!=cfg:
            raise ValueError('Stage-I model/config mismatch')
        if rank==0 and hasattr(policy,'initialization_report'):
            atomic_json(output/'initialization.json',policy.initialization_report)
        policy.to(device)
        update(phase='dataset_audit')
        ds=IndexedFutureDataset(cfg['dataset_root'],policy.statistics,'train',seed,cfg['val_ratio'])
        val=IndexedFutureDataset(cfg['dataset_root'],policy.statistics,'val',seed,cfg['val_ratio'])
        assert {e['episode_index'] for e in ds.selected}.isdisjoint(e['episode_index'] for e in val.selected)
        if rank==0:
            audit=ds.audit(); audit.update(train_episodes=len(ds.selected),validation_episodes_count=len(val.selected),
                validation_frames=len(val),effective_batch=world*args.batch_size*args.accumulation,
                nominal_epochs=args.steps*world*args.batch_size*args.accumulation/len(ds),
                sampling='DistributedSampler shuffled train anchors; disjoint ranks; drop incomplete epoch batches')
            atomic_json(output/'dataset_audit.json',audit); atomic_json(output/'config.json',cfg)
        batches=DistributedBatches(ds,args.batch_size,world,rank,seed)
        opts=dict(num_workers=args.workers,pin_memory=True)
        if args.workers:
            opts.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
        generator=torch.Generator().manual_seed(seed+rank)
        loader=DataLoader(ds,batch_sampler=batches,generator=generator,**opts)
        vi=validation_indices(val,args.eval_batches*args.batch_size,seed+1)
        owned_batches=list(range(rank,args.eval_batches,world))
        owned_indices=[vi[i] for b in owned_batches for i in range(b*args.batch_size,(b+1)*args.batch_size)]
        validation=DataLoader(Subset(val,owned_indices),batch_size=args.batch_size,
            generator=torch.Generator().manual_seed(seed+2+rank),**opts)
        if rank==0:
            atomic_json(output/'validation_samples.json',dict(seed=seed+1,indices=vi,episodes=val.validation_episodes,
                global_batches=args.eval_batches,assignment='whole batch j belongs to rank j % world_size'))
        update(phase='synchronizing_ddp_model')
        # Future roles may be absent at episode tails, so used parameters vary by batch.
        model=DDP(policy,device_ids=[local_rank],output_device=local_rank,
            broadcast_buffers=False,find_unused_parameters=True)
        parameters=[x for x in policy.parameters() if x.requires_grad]
        if rank==0:
            atomic_json(output/'parameter_audit.json',dict(trainable=sum(x.numel() for x in parameters),
                frozen=sum(x.numel() for x in policy.parameters() if not x.requires_grad),
                optimizer='AdamW with FP32 master weights',optimizer_device='cpu',vlm_layers=16,
                ddp_world_size=world,find_unused_parameters=True,physical_gpus=[1,3]))
        update(phase='initializing_fp32_optimizer')
        optim=MasterAdamW(parameters,cfg['learning_rate'],cfg['weight_decay'],offload=True)
        def lr(step):
            warm=min(cfg['warmup_steps'],max(1,args.steps//10))
            return min(1,(step+1)/warm)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-warm)/max(1,args.steps-warm))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optim.optimizer,lr)
        start=0; best=float('inf'); epoch=0; consumed=0
        random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
            if saved['stage']!='I' or saved['config']!=cfg or saved['world_size']!=world or saved['total_steps']!=args.steps:
                raise ValueError('Resume configuration/world size/schedule mismatch')
            if saved['batch_size']!=args.batch_size or saved['accumulation']!=args.accumulation:
                raise ValueError('Resume batch/accumulation mismatch')
            optim.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            start=saved['step']; best=saved['best_validation']; epoch=saved['epoch']; consumed=saved['batches_consumed']
            rs=saved['rng_by_rank'][rank]
            random.setstate(rs['python']); np.random.set_state(rs['numpy']); torch.set_rng_state(rs['torch'])
            torch.cuda.set_rng_state(rs['cuda'],device); generator.set_state(rs['loader'])
            if policy.source_step!=start:
                raise ValueError('Model/trainer step mismatch')
            del saved
        batches.reset(epoch,consumed)
        update(step=start,recovery_mode='full_ddp' if args.resume else 'fresh',epoch=epoch)

        def evaluate(step):
            update(phase='validation',step=step); model.eval()
            values=np.zeros(5,dtype=np.float64)
            with torch.random.fork_rng(devices=[local_rank]),torch.no_grad():
                torch.manual_seed(seed+10000+rank)
                for i,batch in enumerate(validation):
                    update(phase=f'validation {i+1}/{len(owned_batches)} per rank')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        enc=policy.encode(batch); cond=policy.action_conditions(batch,enc)
                        def flow(c):
                            return policy.flow.loss(batch['actions'].to(device),batch['actions_mask'].to(device),
                                batch['action_hz'].to(device),batch['embodiment_id'].to(device),
                                batch['remaining_actions'].to(device),c,repeats=cfg['repeated_diffusion_steps'])
                        rng=torch.get_rng_state(); crng=torch.cuda.get_rng_state(device)
                        value=flow(cond); values[0]+=float(value); values[1]+=1
                        if owned_batches[i]<args.diagnostic_batches:
                            after=torch.get_rng_state(); ca=torch.cuda.get_rng_state(device)
                            torch.set_rng_state(rng); torch.cuda.set_rng_state(crng,device)
                            disabled=flow(tuple(list(cond[:-1])+[torch.zeros_like(cond[-1])]))
                            values[2]+=float(value); values[3]+=float(disabled); values[4]+=1
                            torch.set_rng_state(after); torch.cuda.set_rng_state(ca,device)
            total=torch.tensor(values,device=device); dist.all_reduce(total)
            a=total.cpu().tolist()
            result=dict(step=step,split='validation',loss=a[0]/a[1],flow=a[0]/a[1],global_batches=int(a[1]))
            if a[4]:
                result.update(diagnostic_flow=a[2]/a[4],without_roles_flow=a[3]/a[4])
            if not all(math.isfinite(v) for v in result.values() if isinstance(v,float)):
                raise FloatingPointError('Nonfinite validation')
            record(result)
            return result['flow']

        def save(step):
            update(phase='saving_checkpoint',step=step)
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),python=random.getstate(),
                numpy=np.random.get_state(),loader=generator.get_state())
            gathered=[None]*world if rank==0 else None
            dist.gather_object(rng,gathered,dst=0)
            destination=output/f'step_{step:06d}'
            if rank==0:
                policy.save(destination,step,extra={'stage':'I','ddp_world_size':world})
                trainer=dict(format='goal_local_plus_ddp_v1',stage='I',config=cfg,step=step,
                    optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),best_validation=best,
                    rng_by_rank=gathered,world_size=world,total_steps=args.steps,batch_size=args.batch_size,
                    accumulation=args.accumulation,epoch=epoch,batches_consumed=consumed)
                torch.save(trainer,destination/'trainer.tmp'); (destination/'trainer.tmp').replace(destination/'trainer.pt')
                atomic_json(destination/'checkpoint_complete.json',dict(step=step,owner='libero_plus.train_ddp',world_size=world))
                del trainer
                # Preserve all model weights before pruning optimizer-bearing checkpoints.
                from retain_weights import retain_one
                retain_one(destination)
                complete=sorted(x.parent for x in output.glob('step_*/checkpoint_complete.json'))
                for old in complete[:-args.keep_checkpoints]:
                    shutil.rmtree(old)
            dist.barrier()
            update(phase='training',checkpoint=str(destination))

        baseline=evaluate(start); best=min(best,baseline)
        iterator=iter(loader); durations=[]
        for step in range(start,args.steps):
            begin=time.time(); model.train(); optim.zero_grad(); values=np.zeros(4,dtype=np.float64)
            for micro in range(args.accumulation):
                update(phase='training',step=step,micro=micro+1,epoch=epoch)
                try:
                    batch=next(iterator)
                except StopIteration:
                    epoch+=1; consumed=0; batches.reset(epoch); iterator=iter(loader); batch=next(iterator)
                consumed+=1
                sync = nullcontext() if micro==args.accumulation-1 else model.no_sync()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        losses=model(batch); loss=losses['loss']/args.accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss rank {rank} step {step}')
                    loss.backward()
                values[0]+=float(losses['flow'].detach())/args.accumulation
                values[1]+=int(batch['local_valid'].sum()); values[2]+=int(batch['goal_frame_valid'].any(-1).sum()); values[3]+=len(batch['local_valid'])
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
            update(phase='optimizer_step'); optim.step(); scheduler.step()
            totals=torch.tensor(values,device=device); dist.all_reduce(totals)
            a=totals.cpu().tolist(); elapsed=time.time()-begin; durations.append(elapsed)
            if step==start or (step+1)%args.eval_every==0:
                named=dict(policy.named_parameters())
                selected=['act_query','flow.plan','future.goal.query','future.local.query','flow.goal_proj.weight']
                check=torch.cat([named[n].detach().reshape(-1)[:32].float() for n in selected])
                low=check.clone(); high=check.clone(); dist.all_reduce(low,op=dist.ReduceOp.MIN); dist.all_reduce(high,op=dist.ReduceOp.MAX)
                difference=float((high-low).abs().max())
                if difference>1e-6:
                    raise RuntimeError(f'DDP parameter divergence {difference}')
                if rank==0:
                    atomic_json(output/'ddp_sync_audit.json',dict(step=step+1,world_size=world,physical_gpus=[1,3],
                        sampled_parameters=selected,max_abs_difference=difference,effective_batch=int(a[3]),finite_gradients=True))
            record(dict(step=step+1,split='train',loss=a[0]/world,flow=a[0]/world,grad_norm=float(norm),
                lr=optim.optimizer.param_groups[0]['lr'],update_seconds=elapsed,
                local_valid_fraction=a[1]/a[3],goal_valid_fraction=a[2]/a[3],effective_batch=int(a[3]),world_size=world))
            update(step=step+1,micro=0,eta_seconds=(args.steps-step-1)*sum(durations[-20:])/len(durations[-20:]))
            if (step+1)%args.eval_every==0:
                score=evaluate(step+1)
                if score<best:
                    best=score
                    if rank==0:
                        update(phase='saving_best'); policy.save(output/'best',step+1,extra={'stage':'I','validation_flow':score})
                    dist.barrier()
            if step==start or (step+1)%args.save_every==0 or step+1==args.steps:
                save(step+1)
        update(phase='completed',step=args.steps,eta_seconds=0)
        dist.destroy_process_group()
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main()
