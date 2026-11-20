"""Exact 1000-epoch LIBERO-Plus training, four FSDP ranks, retained DINO."""
import argparse,copy,fcntl,json,math,os,random,shutil,time
from pathlib import Path
from contextlib import nullcontext
from statistics import median
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
from .config import WORK,ROOT,load_config,atomic_json,file_sha256,digest_json,source_identity,code_identity
from .data import CurrentDataset
from .distributed import initialize,wrap,ExactRankBatchSampler,WeightedDataset


def rng_state(device):
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state(device))


def restore_rng(state,device):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python']);torch.cuda.set_rng_state(state['cuda'],device)


def lr_factor(step,total,warmup):
    if step<warmup:return (step+1)/warmup
    progress=min(1.,(step-warmup+1)/max(1,total-warmup))
    return .05+.95*.5*(1+math.cos(math.pi*progress))


def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu_tree(v) for v in obj)
    return obj


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(WORK/'config.json'))
    parser.add_argument('--resume')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--save-every-updates',type=int,default=1000)
    parser.add_argument('--smoke',action='store_true',help='Real updates plus checkpoint/restore and current-only inference; no formal budget')
    parser.add_argument('--smoke-updates',type=int,default=12)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();cfg=load_config(args.config)
    output=Path(args.output).resolve()
    if not output.is_relative_to(WORK/'runs') or output==WORK/'runs':raise ValueError('Output must be under A0/libero-plus/runs')
    if args.resume and Path(args.resume).resolve().parent!=output/'checkpoints':raise ValueError('Resume must belong to this run')
    if args.workers<0 or args.save_every_updates<0 or args.smoke_updates<2:raise ValueError('Invalid loader/checkpoint options')
    if args.dry_run:
        print(json.dumps(dict(config=cfg,train_anchors=2125407,epochs=1000,updates_per_epoch=66419,total_updates=66419000,global_batch=32,output=str(output)),indent=2));return
    rank,world,device=initialize();lock=None
    try:
        if rank==0:
            output.mkdir(parents=True,exist_ok=True)
            lock=(output/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (output/'status.json').exists() and not args.resume:raise ValueError('Existing run requires full resume')
            atomic_json(output/'status.json',dict(phase='loading',pid=os.getpid(),time=time.time()))
        dist.barrier()
        run(args,cfg,output,rank,world,device)
    except BaseException as error:
        if rank==0:atomic_json(output/'status.json',dict(phase='failed',error=repr(error),time=time.time()))
        raise
    finally:
        if lock:lock.close()
        dist.destroy_process_group()


def run(args,cfg,output,rank,world,device):
    print(f"rank {rank}: importing policy",flush=True)
    from .model import A0Policy
    from ablation.aba1.plus_common.training import validation_indices
    torch.set_num_threads(2)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    print(f'rank {rank}: loading W0',flush=True)
    policy=A0Policy.from_baseline(cfg)
    trainable=[n for n,p in policy.named_parameters() if p.requires_grad]
    assert not any(n.startswith(('future.','gate.')) for n,_ in policy.named_parameters())
    assert all(not p.requires_grad for p in policy.dino.parameters())
    assert not hasattr(policy.flow,'plan') and policy.latent_query.shape==(16,2048)
    if rank==0:
        atomic_json(output/'config.json',cfg);atomic_json(output/'arguments.json',vars(args))
        atomic_json(output/'initialization.json',policy.initialization_report)
        atomic_json(output/'parameter_audit.json',dict(trainable=sum(p.numel() for p in policy.parameters() if p.requires_grad),
            frozen=sum(p.numel() for p in policy.parameters() if not p.requires_grad),trainable_names=trainable,
            frozen_names=[n for n,p in policy.named_parameters() if not p.requires_grad]))
        atomic_json(output/'implementation.json',{str(p.relative_to(WORK)):file_sha256(p) for p in sorted((WORK/'a0_single_latent').glob('*.py'))})
        atomic_json(output/'baseline_source.json',source_identity(cfg['baseline_checkpoint']))
    print(json.dumps(dict(phase='model_loaded',rank=rank,pid=os.getpid(),physical_gpu=cfg['physical_gpus'][rank])),flush=True)
    model=wrap(policy,device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=cfg['weight_decay'],foreach=False)
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    train=CurrentDataset(cfg['dataset_root'],policy.statistics,'train',cfg['split_seed'],cfg['val_ratio'])
    val=copy.copy(train);val.split='val';valid_ids=set(train.validation_episodes)
    val.selected=[e for e in train.episodes if e['episode_index'] in valid_ids]
    val.offsets=np.cumsum([0]+[e['length'] for e in val.selected])
    if (len(train),len(val),len(train.selected),len(val.selected))!=(2125407,112629,13630,717):raise ValueError('Frozen dataset split mismatch')
    split=dict(train=[e['episode_index'] for e in train.selected],validation=train.validation_episodes,
               lengths=[(e['episode_index'],e['length']) for e in train.episodes])
    vi=validation_indices(val,128,cfg['split_seed']+1)
    if rank==0:
        atomic_json(output/'offline_split.json',split)
        atomic_json(output/'dataset_audit.json',dict(train_anchors=len(train),validation_anchors=len(val),train_episodes=len(train.selected),validation_episodes=len(val.selected),future_images_read=False,source_manifest_sha256=train.source_digest))
        atomic_json(output/'validation_anchors.json',dict(indices=vi,noise_seed=cfg['split_seed']+10000))
        atomic_json(output/'dataset_version.json',dict(manifest=source_identity(Path(cfg['dataset_root'])/'manifest.json'),cache_metadata=source_identity(ROOT/'libero-plus/cache/index/metadata.json')))
        atomic_json(output/'code_identity.json',code_identity())
    dist.barrier()
    global_batch=cfg['world_size']*cfg['micro_batch']*cfg['accumulation']
    total=cfg['epochs']*math.ceil(len(train)/global_batch)
    contract=dict(config=cfg,split_sha256=digest_json(split),dataset_version=digest_json(json.loads((output/'dataset_version.json').read_text())),
        baseline_sha256=json.loads((output/'baseline_source.json').read_text())['sha256'],total_updates=total,world_size=world,global_batch=global_batch,smoke=args.smoke,code_sha256=digest_json(code_identity()))
    if rank==0:atomic_json(output/'training_contract.json',contract)
    epoch=cursor=step=seen=0;best=float('inf');best_name=None
    opts=dict(num_workers=args.workers,pin_memory=True)
    if args.workers:opts.update(multiprocessing_context='spawn',prefetch_factor=2)
    # Dedicated generator prevents worker seeding from perturbing policy RNG on resume.
    validation=DataLoader(Subset(val,vi[rank::world]),batch_size=1,shuffle=False,
        generator=torch.Generator().manual_seed(seed+40000+rank),**opts)

    def record(row):
        if rank:return
        row.update(time=time.time(),step=step,epoch=epoch,cursor=cursor,seen_anchors=seen)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    def status(phase,**kw):
        if rank==0:atomic_json(output/'status.json',dict(phase=phase,step=step,epoch=epoch,cursor=cursor,seen_anchors=seen,
            total_updates=total,world_size=world,physical_gpus=cfg['physical_gpus'],time=time.time(),**kw))

    def evaluate(limit=None):
        status('validation');state=rng_state(device);model.eval();summ=torch.zeros(2,device=device,dtype=torch.float64)
        try:
            torch.manual_seed(cfg['split_seed']+10000+rank)
            with torch.no_grad():
                for i,batch in enumerate(validation):
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=model(batch)['flow']
                    summ+=torch.stack((value.double(),torch.ones((),device=device,dtype=torch.float64)))
                    if limit and i+1>=limit:break
            dist.all_reduce(summ);score=float(summ[0]/summ[1])
            if not math.isfinite(score):raise FloatingPointError('Nonfinite validation')
            record(dict(split='validation',flow=score,anchors=int(summ[1])));return score
        finally:restore_rng(state,device)

    def load_checkpoint(path):
        nonlocal epoch,cursor,step,seen,best,best_name
        path=Path(path);marker=json.loads((path/'checkpoint_complete.json').read_text())
        if marker['world_size']!=world:raise ValueError('Resume world size mismatch')
        # Each file is an exact FP32 shard, not the BF16 deployment export.
        shard=path/f'rank_{rank}.pt'
        info=marker['files'][shard.name]
        if shard.stat().st_size!=info['bytes'] or file_sha256(shard)!=info['sha256']:raise ValueError('Corrupt recovery shard')
        saved=torch.load(path/f'rank_{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
        if saved['contract']!=contract:raise ValueError('Resume training contract mismatch')
        current=dict((n,p) for n,p in model.named_parameters() if p.requires_grad)
        if current.keys()!=saved['parameters'].keys():raise ValueError('FSDP shard layout mismatch')
        with torch.no_grad():
            for name,p in current.items():
                v=saved['parameters'][name]
                if p.shape!=v.shape:raise ValueError('Shard shape mismatch')
                p.copy_(v)
        optimizer.load_state_dict(saved['optimizer'])
        epoch,cursor,step,seen=(saved[k] for k in ('epoch','cursor','step','seen_anchors'))
        best,best_name=saved['best_flow'],saved['best_name']
        if seen!=epoch*len(train)+cursor or step!=epoch*math.ceil(len(train)/global_batch)+math.ceil(cursor/global_batch):raise ValueError('Invalid resume counters')
        if marker['step']!=step:raise ValueError('Checkpoint marker mismatch')
        restore_rng(saved['rng'],device);dist.barrier()

    def save_checkpoint(improved=False):
        nonlocal best_name
        status('checkpointing');state=rng_state(device)
        name=f'epoch_{epoch:04d}_step_{step:09d}';dest=output/'checkpoints'/name;temp=dest.with_name(name+'.incomplete')
        if improved:best_name=name
        if rank==0:
            if dest.exists() or temp.exists():raise FileExistsError(str(dest))
            if shutil.disk_usage(output).free<45*1024**3:raise OSError('Insufficient disk space for complete recovery checkpoint')
            temp.mkdir(parents=True)
        dist.barrier()
        torch.save(dict(contract=contract,epoch=epoch,cursor=cursor,step=step,seen_anchors=seen,
            best_flow=best,best_name=best_name,rng=state,optimizer=cpu_tree(optimizer.state_dict()),
            parameters={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}),temp/f'rank_{rank}.pt')
        dist.barrier()
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            full=model.state_dict()
        if rank==0:
            # Deployment uses the same BF16 parameter precision as training compute.
            full={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in full.items()}
            policy.processor.save_pretrained(temp/'processor')
            torch.save(policy.payload(full,step,dict(completed_epochs=epoch,cursor=cursor,seen_anchors=seen,smoke=args.smoke,action_protocol=cfg['action_protocol'])),temp/'model.pt')
            sha=file_sha256(temp/'model.pt')
            atomic_json(temp/'action_semantics.json',cfg['action_protocol'])
            files={f.name:source_identity(f) for f in temp.glob('rank_*.pt')}
            atomic_json(temp/'checkpoint_complete.json',dict(format=cfg['architecture'],complete=True,step=step,epoch=epoch,cursor=cursor,world_size=world,model_sha256=sha,files=files))
            temp.rename(dest)
            entry=dict(path=str(dest),step=step,epoch=epoch,model_sha256=sha)
            atomic_json(output/'latest.json',entry)
            if improved:atomic_json(output/'best_flow.json',dict(entry,flow=best))
            completed=sorted((output/'checkpoints').glob('epoch_*/checkpoint_complete.json'))
            keep={p.parent.name for p in completed[-2:]}
            for p in completed:
                if p.parent.name not in keep:shutil.rmtree(p.parent)
            del full
        dist.barrier();restore_rng(state,device);return dest

    if args.resume:load_checkpoint(args.resume)
    else:evaluate(limit=1 if args.smoke else None)
    smoke_batch=None;timings=[]
    while epoch<cfg['epochs']:
        sampler=ExactRankBatchSampler(len(train),rank,seed,epoch,cursor,global_batch=global_batch,world=world,micro_batch=cfg['micro_batch'])
        loader=DataLoader(WeightedDataset(train),batch_sampler=sampler,
            generator=torch.Generator().manual_seed(seed+epoch+20000+rank),**opts)
        model.train();optimizer.zero_grad(set_to_none=True);metric=0.;begin=time.monotonic();data_seconds=0.;compute_seconds=0.;last_end=begin
        for batch in loader:
            data_seconds+=time.monotonic()-last_end
            if args.smoke and smoke_batch is None:
                smoke_batch={k:v[:1] for k,v in batch.items()}
            compute_begin=time.monotonic()
            lr=cfg['learning_rate']*lr_factor(step,total,cfg['warmup_steps'])
            for pg in optimizer.param_groups:pg['lr']=lr
            weights=batch['sample_weight'].to(device=device,dtype=torch.float32);n=int(batch['update_count'][0])
            last=bool(batch['last_microstep'][0])
            with nullcontext() if last else model.no_sync():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    values=model(batch);loss=(values['per_example_flow']*weights).sum()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite action loss')
                loss.backward()
            metric+=float((values['per_example_flow'].detach()*weights).sum())/world
            if args.smoke:torch.cuda.synchronize(device)
            compute_seconds+=time.monotonic()-compute_begin
            last_end=time.monotonic()
            if not last:continue
            norm=model.clip_grad_norm_(cfg['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            step+=1;cursor+=n;seen+=n
            stats=torch.tensor([metric,torch.cuda.max_memory_allocated(device)/1024**3],device=device)
            dist.all_reduce(stats[:1]);dist.all_reduce(stats[1:],op=dist.ReduceOp.MAX)
            elapsed=time.monotonic()-begin;timings.append(elapsed)
            record(dict(split='train',flow=float(stats[0]),loss=float(stats[0]),lr=lr,grad_norm=float(norm),
                update_anchors=n,update_seconds=elapsed,anchors_per_second=n/elapsed,
                rank0_data_seconds=data_seconds,rank0_forward_backward_seconds=compute_seconds,max_rank_peak_gib=float(stats[1])))
            if args.smoke and step==args.smoke_updates:
                stable=timings[2:] or timings
                if rank==0:atomic_json(output/'throughput.json',dict(gpus=cfg['physical_gpus'],global_batch=global_batch,micro_batch=cfg['micro_batch'],accumulation=cfg['accumulation'],warmup_updates_excluded=min(2,len(timings)-1),measured_updates=len(stable),median_update_seconds=median(stable),min_update_seconds=min(stable),max_update_seconds=max(stable),median_anchors_per_second=global_batch/median(stable),excludes_checkpoint_and_validation=True))
                saved=save_checkpoint()
                parameter=parameters[0];before=parameter.detach().clone()
                moment=optimizer.state[parameter]['exp_avg'];before_moment=moment.clone()
                saved_rng=rng_state(device)
                with torch.no_grad():parameter.add_(1);moment.zero_()
                torch.rand(9,device=device)
                load_checkpoint(saved)
                if not torch.equal(parameter,before) or not torch.equal(optimizer.state[parameter]['exp_avg'],before_moment):
                    raise ValueError('Exact optimizer/parameter restore failed')
                if not torch.equal(torch.cuda.get_rng_state(device),saved_rng['cuda']):raise ValueError('RNG restore failed')
                del before,before_moment
                model.eval();noise=torch.zeros((1,50,32),device=device,dtype=torch.bfloat16)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    actions=model(smoke_batch,predict=True,noise=noise)
                if actions.shape!=(1,8,7) or not torch.isfinite(actions).all():raise ValueError('Inference smoke failure')
                torch.save(dict(main=smoke_batch['current_main'],wrist=smoke_batch['current_wrist'],instruction=smoke_batch['instruction'],
                    noise=noise.cpu(),actions=actions.cpu()),output/f'inference_reference_rank_{rank}.pt')
                status('smoke_passed',checkpoint=str(saved),inference_shape=list(actions.shape));return
            at_end=cursor==len(train)
            if cursor>len(train):raise RuntimeError('Epoch oversampling')
            if at_end:
                epoch+=1;cursor=0;score=evaluate();improved=score<best
                if improved:best=score
                save_checkpoint(improved);model.train()
            elif args.save_every_updates and step%args.save_every_updates==0:save_checkpoint()
            status('training')
            if at_end:break
            metric=0.;begin=time.monotonic();data_seconds=0.;compute_seconds=0.;last_end=begin
    if step!=total or seen!=cfg['epochs']*len(train):raise RuntimeError('Final epoch budget mismatch')
    if rank==0:atomic_json(output/'final.json',json.loads((output/'latest.json').read_text()))
    status('completed')

if __name__=='__main__':main()
