"""Audited A2 training loop with an already measured, resident Stage II model."""
import contextlib,json,os,random,shutil,time
from pathlib import Path
from datetime import timedelta
from .runtime import *
from a2_local.preflight import check_complete
from a2_local.training import DistributedBatches,validation_indices,capture_rng,restore_rng

def run(args,cfg,spec,prepared):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,Subset
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world!=spec["world_size"]:raise ValueError("World size differs from fixed stage protocol")
    first=False;local=rank_device(cfg,rank)  # Each worker exposes exactly one physical GPU.
    if worker_physical_gpu()!=spec["rank_to_gpu"][rank]:raise ValueError("Worker GPU differs from selected mapping")
    os.environ["LOCAL_RANK"]=str(local);torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    from a2_local.numerics import configure_attention_backends
    attention_backends=configure_attention_backends()
    dist.init_process_group("gloo",init_method=args.rendezvous,rank=rank,world_size=world,timeout=timedelta(minutes=90))
    # Heavy imports only after explicit process-group setup (Accelerate initialization).
    from torch.nn.parallel import DistributedDataParallel as DDP
    from goal_local.optim import MasterAdamW
    from a2_local.checkpointing import make_recovery,rank_recovery,tensor_bytes
    from .checkpointing import commit_checkpoint,wait_for_space
    from a2_local.distributed import DistributedMaster
    from ablation.aba1.plus_common.model import GoalLocalPolicy,DistillationSystem
    from a2_local.model import make_stage_two
    from ablation.aba1.plus_common.data import IndexedFutureDataset
    output=stage_root(cfg,args.stage);seed=cfg["train_seed"]
    def status(**values):
        row=dict(rank=rank,pid=os.getpid(),stage=args.stage,time=time.time(),physical_gpu=spec["rank_to_gpu"][rank],cuda_device=local,**values)
        atomic_json(output/f"status_rank{rank}.json",row)
        if rank==0:atomic_json(output/"status.json",row)
    def record(row):
        if rank==0:
            row["time"]=time.time()
            with (output/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
    def reduce(t):
        x=t.to(device) if first else t
        dist.all_reduce(x);return x.cpu()
    def barrier():dist.barrier()
    try:
        sources=source_paths(cfg,args.stage)
        system=prepared.pop('system');policy=prepared.pop('policy');teacher=prepared.pop('teacher')
        del prepared['padding']
        params=[p for p in policy.parameters() if p.requires_grad]
        teacher_versions=[p._version for p in teacher.parameters()]
        recovery=None
        if args.resume:
            check_complete(args.resume,require_resume=True,verify_hash=True)
            recovery=torch.load(Path(args.resume)/'trainer.pt',map_location='cpu',mmap=True,weights_only=False)
        local_future,goal_future=future_roles(cfg,args.stage)
        train=IndexedFutureDataset(cfg["dataset_root"],policy.statistics,"train",cfg["split_seed"],
            local=local_future,goal=goal_future)
        val=IndexedFutureDataset(cfg["dataset_root"],policy.statistics,"val",cfg["split_seed"],
            local=local_future,goal=goal_future)
        if (len(train),len(val))!=(2125407,112629):raise ValueError("Dataset anchors changed")
        batches=DistributedBatches(train,spec["batch_size"],world,rank,seed)
        generator=torch.Generator().manual_seed(seed+rank)
        options=dict(num_workers=args.workers,pin_memory=True)
        if args.workers:options.update(multiprocessing_context="spawn",persistent_workers=True,prefetch_factor=1)
        loader=DataLoader(train,batch_sampler=batches,generator=generator,**options)
        vi=validation_indices(val,spec["eval_batches"]*spec["batch_size"],seed+1)
        own=[vi[i] for j in range(rank,spec["eval_batches"],world)
             for i in range(j*spec["batch_size"],(j+1)*spec["batch_size"])]
        validation=DataLoader(Subset(val,own),batch_size=spec["batch_size"],num_workers=0,
                              generator=torch.Generator().manual_seed(seed+2+rank))
        model=DDP(system,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True) if first else system
        optimizer=MasterAdamW(params,cfg["learning_rate"],cfg["weight_decay"]) if first else DistributedMaster(
            params,cfg["learning_rate"],cfg["weight_decay"])
        if not first:optimizer.broadcast_weights()
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer,
            lambda n:lr_factor(n,spec["schedule_total_steps"],cfg["warmup_steps"])) if first else None
        identity=code_identity()
        source_hashes=[None]
        if rank==0:
            source_hashes[0]={}
            for key,path in sources.items():
                file=path if key=="baseline" else path/"model.pt"
                source_hashes[0][key]=dict(path=str(path),sha256=file_sha256(file),bytes=file.stat().st_size)
        dist.broadcast_object_list(source_hashes,src=0)
        contract=dict(model_loading="sequential_measured_admission",physical_gpus=spec["physical_gpus"],rank_to_gpu=spec["rank_to_gpu"],config=cfg,stage=args.stage,world=world,batch_size=spec["batch_size"],
            accumulation=spec["accumulation"],accumulation_per_rank=spec["accumulation_per_rank"],gradient_cpu_offload=True,attention_backends=attention_backends,schedule_total_steps=spec["schedule_total_steps"],
            offload_saved_tensors=args.offload_saved_tensors,share_frozen_encoders=args.share_frozen_encoders,
            checkpoint_policy=dict(CHECKPOINT_POLICY),rank_policy=POLICY["rank_policy"],
            code_sha256=digest_json(identity),versions=runtime_environment(),asset_sha256=asset_identity(cfg),
            source_paths={k:str(v.resolve()) for k,v in sources.items()},source_hashes=source_hashes[0],
            data_manifest_sha256=train.source_digest,split_sha256=digest_json(val.validation_episodes))
        epoch=0;consumed=0;start=0
        random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
        if recovery is not None:
            if recovery["contract"]!=contract:raise ValueError("Resume runtime/source/schedule contract changed")
            optimizer.load_state_dict(recovery["optimizer"] if first or rank==0 else None)
            if scheduler is not None:scheduler.load_state_dict(recovery["scheduler"])
            start=recovery["step"];own_state=rank_recovery(recovery,rank,world)
            epoch=own_state["epoch"];consumed=own_state["consumed"]
            if policy.source_step!=start:raise ValueError("Model/optimizer step mismatch")
            restore_rng(own_state["rng"],device,generator);del recovery,own_state
        batches.reset(epoch,consumed)
        if rank==0:
            atomic_json(output/"arguments.json",dict(vars(args),plan=spec))
            atomic_json(output/"contract.json",contract);atomic_json(output/"code_identity.json",identity)
            atomic_json(output/"initialization.json",getattr(policy,"initialization_report",dict(resume=args.resume,
                inherited_action1=args.stage=="action2",reset_optimizer=args.stage=="action2" and not args.resume)))
            atomic_json(output/"parameter_audit.json",dict(trainable=sum(p.numel() for p in params),
                frozen=sum(p.numel() for p in policy.parameters() if not p.requires_grad),
                teacher_frozen=teacher is not None,teacher_in_optimizer=False,
                role_tokens=16,local_slots=policy.kl,goal_slots=policy.kg,plan_tokens=4))
            atomic_json(output/"dataset_audit.json",dict(train_frames=len(train),validation_frames=len(val),
                train_episodes=len(train.selected),validation_episodes=len(val.selected),
                validation_indices=vi,split_seed=cfg["split_seed"],train_seed=seed,
                source_manifest_sha256=train.source_digest,
                validation_episodes_ids=val.validation_episodes,
                validation_is_independent_trajectory_split=False))
            atomic_json(output/"sources.json",source_hashes[0])
        barrier()
        loss_keys=["loss","flow"]+([] if teacher is None else
            (["local_distill"] if policy.kl else [])+(["goal_distill"] if policy.kg else []))
        def evaluate(step):
            system.eval();values=torch.zeros(len(loss_keys)+1,dtype=torch.float64)
            with torch.random.fork_rng(devices=[local]),torch.no_grad():
                torch.manual_seed(seed+10000+rank)
                for index,batch in enumerate(validation,1):
                    status(phase="validation",step=step,validation_batch=index,validation_batches=len(validation))
                    with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):losses=system(batch)
                    values[:-1]+=torch.tensor([float(losses[k]) for k in loss_keys]);values[-1]+=1
            values=reduce(values)
            if not torch.isfinite(values).all():raise FloatingPointError("Nonfinite validation")
            record(dict(step=step,cumulative_step=step+spec["cumulative_offset"],split="validation",
                **{k:float(values[i]/values[-1]) for i,k in enumerate(loss_keys)}))
        def save(step):
            status(phase="checkpoint",step=step)
            own_state=dict(rank=rank,epoch=epoch,consumed=consumed,rng=capture_rng(device,generator))
            rank_states=[None]*world if rank==0 else None
            dist.gather_object(own_state,rank_states,dst=0)
            required=None
            if rank==0:
                final=checkpoint_path(cfg,args.stage,step);temp=output/f".step_{step:06d}.incomplete"
                if final.exists():raise FileExistsError(final)
                if temp.exists():shutil.rmtree(temp)
                recovery=make_recovery(step,contract,optimizer.state_dict(),
                    scheduler.state_dict() if scheduler is not None else None,rank_states)
                required=(tensor_bytes(policy.state_dict())+tensor_bytes(recovery)+2**30
                          +CHECKPOINT_POLICY['shared_disk_reserve_gib']*2**30)
            wait_for_space(output,required,rank,dist,status,step)
            if rank==0:
                temp.mkdir()
                extra=dict(stage=args.stage,ablation=cfg["ablation"],protocol=PROTOCOL,
                    cumulative_step=step+spec["cumulative_offset"],teacher_step=stage_steps(cfg,"I") if teacher else None,
                    action_protocol=ACTION_PROTOCOL,source_paths=contract["source_paths"])
                policy.save(temp,step,extra)
                torch.save(recovery,temp/"trainer.pt")
                del recovery
                atomic_json(temp/"action_semantics.json",ACTION_PROTOCOL)
                artifacts=["model.pt","trainer.pt","action_semantics.json"]+[
                    str(p.relative_to(temp)) for p in sorted((temp/"processor").rglob("*")) if p.is_file()]
                metadata=dict(format=FORMAT,complete=True,resumable=True,
                    step=step,cumulative_step=extra["cumulative_step"],stage=args.stage,mode=policy.mode,
                    config_sha256=digest_json(cfg))
                commit_checkpoint(temp,final,metadata,artifacts,check_complete)
                record(dict(split="checkpoint",step=step,resumable=True,retained_checkpoint=str(final)))
            barrier();torch.cuda.empty_cache();status(phase="training",step=step)
        # Stage II saves every 100 updates, never at step zero.
        evaluate(start);torch.cuda.empty_cache();iterator=iter(loader)
        for completed in range(start,spec["stop_after_updates"]):
            begin=time.monotonic();model.train();optimizer.zero_grad()
            values=torch.zeros(len(loss_keys),dtype=torch.float64)
            for micro in range(spec["accumulation_per_rank"][rank]):
                status(phase="training",step=completed,micro=micro+1,epoch=epoch)
                try:batch=next(iterator)
                except StopIteration:
                    epoch+=1;consumed=0;batches.reset(epoch);iterator=iter(loader);batch=next(iterator)
                consumed+=1
                sync=model.no_sync() if first and micro+1<spec["accumulation"] else contextlib.nullcontext()
                saved=torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload_saved_tensors else contextlib.nullcontext()
                with sync:
                    with saved,torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
                        losses=model(batch);loss=losses["loss"]*(world/32)
                        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                values+=torch.tensor([float(losses[k].detach()) for k in loss_keys])*(world/32)
            used_lr=cfg["learning_rate"]*lr_factor(completed,spec["schedule_total_steps"],cfg["warmup_steps"])
            if first:
                if abs(optimizer.optimizer.param_groups[0]["lr"]-used_lr)>1e-15:raise ValueError("LR trace mismatch")
                norm=torch.nn.utils.clip_grad_norm_(params,cfg["gradient_clip"],error_if_nonfinite=True,foreach=False)
                optimizer.step();scheduler.step()
            else:norm=optimizer.step(used_lr,cfg["gradient_clip"])
            if teacher is not None and teacher_versions!=[p._version for p in teacher.parameters()]:
                raise ValueError("Frozen teacher mutated")
            values=reduce(values)/world;step=completed+1
            record(dict(step=step,cumulative_step=step+spec["cumulative_offset"],split="train",
                **{k:float(values[i]) for i,k in enumerate(loss_keys)},lr_used=used_lr,
                next_lr=cfg["learning_rate"]*lr_factor(step,spec["schedule_total_steps"]),
                grad_norm=float(norm),anchors=step*32,effective_batch=32,update_seconds=time.monotonic()-begin))
            if step==start+1 or step%spec["save_every"]==0:
                selected=[(n,p) for n,p in policy.named_parameters() if p.requires_grad and
                    (n in ("act_query","q_goal","q_local","flow.plan","gate.scales") or n.endswith(".query"))]
                sample=torch.cat([p.detach().flatten()[:32].float().cpu() for _,p in selected])
                low=sample.to(device) if first else sample.clone();high=low.clone()
                dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
                if not torch.equal(low,high):raise ValueError("Ranks diverged")
                if rank==0:atomic_json(output/"sync_audit.json",dict(step=step,max_difference=0,
                    names=[n for n,_ in selected],teacher_unchanged=True))
            if step%spec["eval_every"]==0:
                evaluate(step);torch.cuda.empty_cache()
            if step%spec["save_every"]==0 or step==spec["stop_after_updates"]:save(step)
        status(phase="completed",step=spec["stop_after_updates"],
            cumulative_step=spec["cumulative_offset"]+spec["stop_after_updates"])
        train.close();val.close()
    except BaseException as exc:
        status(phase="failed",error=repr(exc));raise
    finally:dist.destroy_process_group()

