"""Two-GPU Stage I continuation with averaged gradients and one CPU AdamW owner.

Every rank runs the posterior policy and contributes gradients. DDP averages
them before rank zero applies the restored FP32 AdamW update. Updated parameters
are broadcast in bounded buckets. Only rank zero writes the run/checkpoints.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset

from goal_local.optim import MasterAdamW
def atomic_json(path, payload):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2))
    temporary.replace(path)


def validation_indices(dataset, count, seed):
    return np.random.default_rng(seed).integers(0,len(dataset),size=count).tolist()


class OffsetSampler(Sampler):
    def __init__(self, sampler, offset):
        self.sampler, self.offset = sampler, offset

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return max(0, len(self.sampler)-self.offset)


@torch.no_grad()
def broadcast_parameters(parameters, limit=16*1024*1024):
    """At most 16 MiB of temporary GPU storage; support mixed parameter dtypes."""
    bucket, size, dtype = [], 0, None

    def send(parts):
        flat = torch.cat([part.detach().reshape(-1) for part in parts])
        dist.broadcast(flat, src=0)
        offset = 0
        for part in parts:
            part.copy_(flat[offset:offset+part.numel()].view_as(part))
            offset += part.numel()

    for parameter in parameters:
        chunk_elements = max(1, limit//parameter.element_size())
        for part in parameter.view(-1).split(chunk_elements):
            nbytes = part.numel()*part.element_size()
            if bucket and (dtype != part.dtype or size+nbytes > limit):
                send(bucket)
                bucket, size = [], 0
            bucket.append(part)
            size += nbytes
            dtype = part.dtype
    if bucket:
        send(bucket)


def rng_state(device):
    return dict(rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state(device))


def restore_rng(state, device):
    torch.set_rng_state(state['rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    value = state['cuda_rng']
    torch.cuda.set_rng_state(value[0] if isinstance(value, list) else value, device)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--resume', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--eval-batches', type=int, default=64)
    p.add_argument('--diagnostic-batches', type=int, default=8)
    p.add_argument('--gpu-memory-fraction', type=float, default=.30)
    p.add_argument('--save-first-update', action='store_true')
    args = p.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 2 or world*args.batch_size*args.accumulation != 32:
        p.error('Requires two ranks and effective batch size 32')
    if args.batch_size != 1:
        p.error('Keep microbatch 1 to preserve the original 64-sample validation protocol')
    root = Path(__file__).resolve().parents[1]
    output, source = args.output.resolve(), args.resume.resolve()
    if not output.is_relative_to(root/'robotwin/runs'):
        p.error('Output must be under robotwin/runs')
    if source.name != 'trainer.pt' or not (source.parent/'checkpoint_complete.json').is_file():
        p.error('A completed full model/optimizer checkpoint is required')
    if output == source.parent.parent:
        p.error('Use a new output directory so previous single-GPU history remains intact')
    if rank == 0 and (output/'metrics.jsonl').exists():
        p.error('Use a fresh output directory; history is copied through the recovery step')
    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45), device_id=device)
    # starVLA logger creates PartialState on import; initialize our group first.
    from .data import FutureDataset
    from .model import RobotwinPolicy
    status = dict(phase='loading_model', step=int(json.loads((source.parent/'checkpoint_complete.json').read_text())['step']), micro=0, total_steps=args.steps,
                  started_at=time.time(), pid=os.getpid(), gpu_ids=[2,5], world_size=world,
                  effective_batch=32, accumulation_per_gpu=args.accumulation)

    def update(**values):
        if rank == 0:
            status.update(values)
            status['updated_at'] = time.time()
            atomic_json(output/'status.json', status)

    def record(row):
        if rank == 0:
            row['time'] = time.time()
            print(json.dumps(row), flush=True)
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')

    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output/'arguments.json', {**vars(args), 'resume':str(source), 'output':str(output), 'stage':'I', 'world_size':world})
            update()
            initial_rows=[]
            for line in (source.parent.parent/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['step'] <= status['step']:
                    initial_rows.append(row)
            (output/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in initial_rows))
        dist.barrier()
        policy = RobotwinPolicy.load(source.parent)
        if policy.mode != 'posterior':
            raise ValueError('Stage I must use the posterior teacher')
        cfg, start = policy.cfg, policy.source_step
        if not 0 < start < args.steps:
            raise ValueError('Invalid checkpoint/global target step')
        policy.to(device)
        seed = cfg['seed']
        ds = FutureDataset(cfg['dataset_root'], policy.statistics, 'train', seed, cfg['val_ratio'])
        val = FutureDataset(cfg['dataset_root'], policy.statistics, 'val', seed, cfg['val_ratio'])
        if args.steps*32 > len(ds):
            raise ValueError('This migration sampler expects the run to remain within its first data epoch')
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, seed=seed, drop_last=True)
        sampler.set_epoch(0)
        offset = start*args.batch_size*args.accumulation
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=OffsetSampler(sampler, offset),
                            num_workers=args.workers, pin_memory=True,
                            **({'multiprocessing_context':'spawn', 'persistent_workers':True, 'prefetch_factor':2} if args.workers else {}),
                            generator=torch.Generator().manual_seed(seed+rank+200))
        vi = validation_indices(val, args.eval_batches, seed+1)
        validation = DataLoader(Subset(val, vi), batch_size=1, num_workers=args.workers, pin_memory=True,
                                **({'multiprocessing_context':'spawn', 'persistent_workers':True, 'prefetch_factor':2} if args.workers else {}),
                                generator=torch.Generator().manual_seed(seed+2)) if rank == 0 else None
        parameters = [x for x in policy.parameters() if x.requires_grad]
        model = DDP(policy, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=True,
                    gradient_as_bucket_view=True, bucket_cap_mb=16)
        optim, scheduler, best = None, None, float('inf')
        if rank == 0:
            update(phase='restoring_fp32_optimizer', step=start)
            saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
            if saved['step'] != start or saved['stage'] != 'I' or saved['config'] != cfg:
                raise ValueError('Model/trainer configuration mismatch')
            optim = MasterAdamW(parameters, cfg['learning_rate'], cfg['weight_decay'], offload=True)
            warm = min(cfg['warmup_steps'], max(1,args.steps//10))
            def lr(step):
                return min(1,(step+1)/warm)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-warm)/max(1,args.steps-warm))))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim.optimizer, lr)
            optim.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            best = saved.get('best_validation', best)
            restore_rng(saved, device)
            rank_rng = saved.get('distributed_rng')
            if rank_rng:
                restore_rng(rank_rng[0], device)
            # Carry every pre-migration measurement through the chosen optimizer step.
            rows = []
            for line in (source.parent.parent/'metrics.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row['step'] <= start:
                    rows.append(row)
            (output/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            atomic_json(output/'config.json', cfg)
            atomic_json(output/'validation_samples.json',dict(seed=seed+1,indices=vi,episodes=val.validation_episodes))
            atomic_json(output/'migration.json',dict(source=str(source),step=start,global_batch=32,
                optimizer_restored=True,scheduler_restored=True,master_weights_restored=True,
                history_copied_through_step=start,source_history_preserved=True,
                sampler='seeded DistributedSampler, epoch 0, skip consumed global steps; single-GPU sample order changes',
                cpu_optimizer_rank=0,gpu_ids=[2,5]))
            audit=ds.audit()
            audit.update(effective_batch=32,world_size=2,train_episodes=len(ds.selected),validation_frames=len(val))
            atomic_json(output/'dataset_audit.json',audit)
            atomic_json(output/'parameter_audit.json',dict(trainable=sum(x.numel() for x in parameters),
                frozen=sum(x.numel() for x in policy.parameters() if not x.requires_grad),
                optimizer='restored FP32 AdamW on rank zero CPU',world_size=2,gpu_ids=[2,5]))
            del saved
        else:
            random.seed(seed+rank+start)
            np.random.seed(seed+rank+start)
            torch.manual_seed(seed+rank+start)
        # These rank RNG states are tiny; optimizer tensors are never sent as Python objects.
        rng_payload = [rank_rng if rank == 0 else None]
        dist.broadcast_object_list(rng_payload, src=0, device=device)
        if rng_payload[0] is not None:
            restore_rng(rng_payload[0][rank],device)
        broadcast_parameters(parameters)
        update(step=start,recovery_source=str(source.parent),recovery_mode='full',phase='validating_recovery')

        def evaluate(step, append=True):
            if rank != 0:
                return None
            policy.eval()
            values, diagnostic = [], []
            with torch.random.fork_rng(devices=[local_rank]), torch.no_grad():
                torch.manual_seed(seed+10000)
                for i,batch in enumerate(validation):
                    update(phase=f'validation {i+1}/{args.eval_batches}',step=step,micro=0)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        encoded=policy.encode(batch)
                        conditions=policy.action_conditions(batch,encoded)
                        def flow(cond):
                            return policy.flow.loss(batch['actions'].to(device),batch['actions_mask'].to(device),
                                batch['action_hz'].to(device),batch['embodiment_id'].to(device),
                                batch['remaining_actions'].to(device),cond,repeats=cfg['repeated_diffusion_steps'])
                        before=rng_state(device)
                        value=float(flow(conditions))
                        values.append(value)
                        if i<args.diagnostic_batches:
                            after=rng_state(device)
                            restore_rng(before,device)
                            disabled=float(flow((*conditions[:-1],torch.zeros_like(conditions[-1]))))
                            restore_rng(after,device)
                            diagnostic.append((value,disabled))
            row=dict(step=step,split='validation',loss=sum(values)/len(values),flow=sum(values)/len(values))
            if diagnostic:
                row.update(diagnostic_flow=sum(x[0] for x in diagnostic)/len(diagnostic),
                           without_roles_flow=sum(x[1] for x in diagnostic)/len(diagnostic))
            if not math.isfinite(row['flow']):
                raise FloatingPointError('Nonfinite validation')
            if append:
                record(row)
            else:
                atomic_json(output/'resume_validation.json',row)
            return row['flow']

        score=evaluate(start,append=False)
        if rank == 0:
            best=min(best,score)
        dist.barrier()
        iterator=iter(loader)
        durations=[]
        for step in range(start,args.steps):
            begin=time.time()
            model.train()
            if rank == 0:
                optim.zero_grad()
            else:
                policy.zero_grad(set_to_none=True)
            losses=[]
            local_count=goal_count=0
            for micro in range(args.accumulation):
                update(phase='training',step=step,micro=micro+1)
                batch=next(iterator)
                local_count+=int(batch['local_valid'].sum())
                goal_count+=int(batch['goal_frame_valid'].any(-1).sum())
                sync=model.no_sync() if micro+1<args.accumulation else nullcontext()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        result=model(batch)
                        loss=result['loss']/args.accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss rank={rank} step={step}')
                    loss.backward()
                losses.append(float(result['flow'].detach()))
            statistics=torch.tensor([sum(losses)/args.accumulation,local_count,goal_count],device=device,dtype=torch.float64)
            dist.all_reduce(statistics)
            norm=None
            if rank == 0:
                norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
                update(phase='optimizer_step')
                optim.step()
                scheduler.step()
            broadcast_parameters(parameters)
            torch.cuda.synchronize(device)
            elapsed=time.time()-begin
            durations.append(elapsed)
            if rank == 0:
                flow=float(statistics[0])/world
                record(dict(step=step+1,split='train',loss=flow,flow=flow,grad_norm=float(norm),
                    lr=optim.optimizer.param_groups[0]['lr'],update_seconds=elapsed,world_size=world,
                    local_valid_fraction=float(statistics[1])/32,goal_valid_fraction=float(statistics[2])/32,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
                update(step=step+1,micro=0,eta_seconds=(args.steps-step-1)*sum(durations[-20:])/len(durations[-20:]))
            if (step+1)%args.eval_every == 0:
                score=evaluate(step+1)
                if rank == 0 and score<best:
                    best=score
                    update(phase='saving_best')
                    policy.save(output/'best',step+1,extra=dict(stage='I',validation_flow=score))
                dist.barrier()
            if (step+1)%args.save_every == 0 or step+1 == args.steps or (args.save_first_update and step == start):
                states=[None]*world if rank == 0 else None
                dist.gather_object(rng_state(device),states,dst=0)
                if rank == 0:
                    destination=output/f'step_{step+1:06d}'
                    update(phase='saving_checkpoint')
                    policy.save(destination,step+1,extra=dict(stage='I',world_size=world,gpu_ids=[2,5]))
                    trainer=dict(stage='I',config=cfg,teacher_source=None,step=step+1,
                        optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),best_validation=best,
                        **states[0],distributed_rng=states,world_size=world,
                        loader_rng=torch.Generator().manual_seed(seed).get_state())
                    # Keep compatibility with the single-GPU trainer's CUDA RNG list.
                    trainer['cuda_rng']=[states[0]['cuda_rng']]
                    torch.save(trainer,destination/'trainer.tmp')
                    (destination/'trainer.tmp').replace(destination/'trainer.pt')
                    atomic_json(destination/'checkpoint_complete.json',dict(step=step+1,owner='robotwin.train_distributed'))
                    del trainer
                    complete=sorted(output.glob('step_*/checkpoint_complete.json'))
                    for marker in complete[:-2]:
                        old=marker.parent/'trainer.pt'
                        if old.exists():
                            old.unlink()
                            atomic_json(marker,dict(step=int(marker.parent.name.split('_')[1]),owner='robotwin.train_distributed',model_only=True))
                    update(phase='training',checkpoint=str(destination))
                dist.barrier()
        update(phase='completed',step=args.steps,eta_seconds=0)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
