"""Four synchronous student ranks on GPU 1,1,3,3, fixed step-5000 teacher."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent))
from train_ddp import DistributedBatches, validation_indices, atomic_json
from indexed_future import IndexedFutureDataset


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--optimizer-dir', required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-batches', type=int, default=32)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or os.environ['CUDA_VISIBLE_DEVICES'] != '1,3':
        raise ValueError('Requires four ranks on physical GPU 1,1,3,3')
    local_device = rank//2
    os.environ['LOCAL_RANK'] = str(local_device)
    torch.set_num_threads(2)
    torch.cuda.set_device(local_device)
    device = torch.device('cuda', local_device)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    optimizer_dir = Path(args.optimizer_dir).resolve(); optimizer_dir.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Existing run requires --resume')
    status = dict(stage='II', rank=rank, world_size=world, physical_gpu=[1,1,3,3][rank],
                  pid=os.getpid(), step=0, total_steps=args.steps, micro=0, phase='initializing', started_at=time.time())
    def update(**values):
        status.update(values); status['updated_at']=time.time()
        atomic_json(output/f'status_rank{rank}.json', status)
        if rank == 0: atomic_json(output/'status.json', status)
    def record(row):
        if rank == 0:
            row['time']=time.time()
            with (output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
    update()
    try:
        dist.init_process_group('gloo', timeout=timedelta(minutes=90))
        from goal_local.model import GoalLocalPolicy, make_stage_two, DistillationSystem
        from distributed_master import DistributedMaster
        random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
        update(phase='loading_fixed_teacher')
        teacher_path = Path(args.teacher).resolve()
        teacher = GoalLocalPolicy.load(teacher_path)
        if teacher.mode != 'posterior' or teacher.source_step != 5000:
            raise ValueError('Expected trained posterior step 5000')
        cfg = teacher.cfg
        if args.resume:
            recovery = json.loads((optimizer_dir/'latest.json').read_text())
            student = GoalLocalPolicy.load(recovery['checkpoint'])
            teacher.flow=None; teacher.requires_grad_(False); teacher.frozen_teacher=True; teacher.eval()
            system = DistillationSystem(student, teacher)
        else:
            system = make_stage_two(teacher); student=system.student
        system.to(device)
        params = [p for p in student.parameters() if p.requires_grad]
        assert not any(p.requires_grad for p in teacher.parameters())
        assert not {p.data_ptr() for p in params} & {p.data_ptr() for p in teacher.parameters()}
        teacher_versions = [p._version for p in teacher.parameters()]
        if rank == 0:
            atomic_json(output/'arguments.json', dict(vars(args), physical_gpus=[1,1,3,3], world_size=4,
                effective_batch=world*args.batch_size*args.accumulation, synchronization='CPU Gloo averaged gradients, rank0 FP32 AdamW, broadcast weights'))
            atomic_json(output/'teacher_source.json', dict(path=str(teacher_path), source_step=5000, fixed=True,
                teacher_frozen=True, teacher_action_head_removed=True, shared_parameter_storage=False))
            atomic_json(output/'config.json', cfg)
            atomic_json(output/'parameter_audit.json', dict(trainable=sum(p.numel() for p in params),
                teacher_frozen=sum(p.numel() for p in teacher.parameters()), teacher_in_optimizer=False,
                visual_frozen=not any(p.requires_grad for p in student.qwen.visual.parameters()),
                dino_frozen=not any(p.requires_grad for p in student.dino.parameters())))
        update(phase='dataset_setup')
        seed=cfg['seed']
        ds=IndexedFutureDataset(cfg['dataset_root'],student.statistics,'train',seed,cfg['val_ratio'])
        val=IndexedFutureDataset(cfg['dataset_root'],student.statistics,'val',seed,cfg['val_ratio'])
        assert {e['episode_index'] for e in ds.selected}.isdisjoint(e['episode_index'] for e in val.selected)
        batches=DistributedBatches(ds,args.batch_size,world,rank,seed)
        generator=torch.Generator().manual_seed(seed+rank)
        opts=dict(num_workers=1,pin_memory=True,multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=1)
        loader=DataLoader(ds,batch_sampler=batches,generator=generator,**opts)
        vi=validation_indices(val,args.eval_batches*args.batch_size,seed+1)
        own=[vi[i] for b in range(rank,args.eval_batches,world) for i in range(b*args.batch_size,(b+1)*args.batch_size)]
        validation=DataLoader(Subset(val,own),batch_size=args.batch_size,num_workers=0)
        if rank == 0:
            atomic_json(output/'dataset_audit.json',dict(source=cfg['dataset_root'],train_frames=len(ds),validation_frames=len(val),
                train_episodes=len(ds.selected),validation_episodes=len(val.selected),disjoint=True,seed=seed,
                validation_indices=vi,action_hz=20,state_used=False))
        update(phase='initializing_optimizer')
        optimizer=DistributedMaster(params,cfg['learning_rate'],cfg['weight_decay'])
        start=0;epoch=0;consumed=0
        random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
        if args.resume:
            payload=torch.load(recovery['optimizer'],map_location='cpu',mmap=True,weights_only=False)
            for key in ('steps','batch_size','accumulation'):
                assert payload['arguments'][key]==getattr(args,key)
            assert payload['teacher']==str(teacher_path) and payload['world_size']==world
            optimizer.load_state_dict(payload['optimizer'] if rank==0 else None)
            start=payload['step'];epoch=payload['epoch'];consumed=payload['consumed']
            rng=payload['rng'][rank]
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'],device);generator.set_state(rng['loader'])
            assert student.source_step==start
            del payload
        batches.reset(epoch,consumed)
        loss_keys=('loss','flow','local_distill','goal_distill')

        def validate(step):
            system.eval(); values=torch.zeros(5,dtype=torch.float64)
            with torch.random.fork_rng(devices=[local_device]),torch.no_grad():
                torch.manual_seed(seed+10000+rank)
                for i,batch in enumerate(validation):
                    update(phase='offline_validation',step=step,micro=i+1)
                    with torch.autocast('cuda',dtype=torch.bfloat16): result=system(batch)
                    values[:4]+=torch.tensor([float(result[k]) for k in loss_keys]);values[4]+=1
            dist.all_reduce(values)
            record(dict(step=step,split='validation',**{k:float(values[i]/values[4]) for i,k in enumerate(loss_keys)}))

        def save(step, optimizer_state=True):
            update(phase='saving_checkpoint',step=step)
            checkpoint=output/f'step_{step:06d}'
            if rank==0:
                student.save(checkpoint,step,extra=dict(stage='II',teacher=str(teacher_path),teacher_step=5000))
                atomic_json(checkpoint/'checkpoint_complete.json',dict(step=step,stage='II',teacher_step=5000,mode='student'))
            dist.barrier()
            if optimizer_state:
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),python=random.getstate(),
                         numpy=np.random.get_state(),loader=generator.get_state())
                all_rng=[None]*world if rank==0 else None
                dist.gather_object(rng,all_rng,dst=0)
                if rank==0:
                    # Keep previous full state until the new state and manifest commit.
                    state=optimizer_dir/f'trainer_{step:06d}.pt'
                    torch.save(dict(step=step,optimizer=optimizer.state_dict(),rng=all_rng,epoch=epoch,consumed=consumed,
                        teacher=str(teacher_path),world_size=world,arguments=vars(args)),state.with_suffix('.tmp'))
                    state.with_suffix('.tmp').replace(state)
                    atomic_json(optimizer_dir/'latest.json',dict(step=step,checkpoint=str(checkpoint),optimizer=str(state)))
                    atomic_json(checkpoint/'resume.json',dict(optimizer=str(state),latest_manifest=str(optimizer_dir/'latest.json')))
                    for previous in optimizer_dir.glob('trainer_*.pt'):
                        if previous!=state: previous.unlink()
                dist.barrier()
            update(phase='training',checkpoint=str(checkpoint))

        # Step zero is explicitly a protocol smoke-test snapshot, not a trained evaluation.
        if not args.resume: save(0,False)
        validate(start)
        iterator=iter(loader);durations=[]
        for step in range(start,args.steps):
            begin=time.time();system.train();optimizer.zero_grad();values=torch.zeros(4,dtype=torch.float64)
            for micro in range(args.accumulation):
                update(phase='training',step=step,micro=micro+1)
                try: batch=next(iterator)
                except StopIteration:
                    epoch+=1;consumed=0;batches.reset(epoch);iterator=iter(loader);batch=next(iterator)
                consumed+=1
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=system(batch);loss=result['loss']/args.accumulation
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite rank {rank} step {step}')
                loss.backward()
                values+=torch.tensor([float(result[k].detach()) for k in loss_keys])/args.accumulation
            update(phase='gradient_sync_optimizer')
            warm=min(cfg['warmup_steps'],max(1,args.steps//10))
            scale=min(1,(step+1)/warm)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-warm)/max(1,args.steps-warm))))
            lr=cfg['learning_rate']*scale
            norm=optimizer.step(lr,cfg['gradient_clip'])
            assert teacher_versions==[p._version for p in teacher.parameters()]
            dist.all_reduce(values);values/=world
            elapsed=time.time()-begin;durations.append(elapsed)
            record(dict(step=step+1,split='train',**{k:float(values[i]) for i,k in enumerate(loss_keys)},
                grad_norm=norm,lr=lr,update_seconds=elapsed,effective_batch=world*args.batch_size*args.accumulation))
            update(step=step+1,micro=0,phase='training',cuda_peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
                   eta_seconds=(args.steps-step-1)*sum(durations[-20:])/len(durations[-20:]))
            if step==start or (step+1)%args.save_every==0:
                named=dict(student.named_parameters());keys=['q_goal','q_local','gate.scales','act_query','flow.goal_proj.weight']
                sample=torch.cat([named[k].detach().reshape(-1)[:32].float().cpu() for k in keys])
                low=sample.clone();high=sample.clone();dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
                delta=float((high-low).abs().max());assert delta==0,delta
                if rank==0:atomic_json(output/'sync_audit.json',dict(step=step+1,world_size=4,max_difference=delta,
                    physical_gpus=[1,1,3,3],teacher_unchanged=True,parameters=keys))
            if (step+1)%args.eval_every==0:validate(step+1)
            if (step+1)%args.save_every==0 or step+1==args.steps:save(step+1)
        update(phase='completed',step=args.steps,micro=0)
        dist.destroy_process_group()
    except BaseException as exc:
        update(phase='failed',error=repr(exc));raise


if __name__=='__main__':main()
