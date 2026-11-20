"""A1 six-rank runtime, based on the audited Plus trainer. Rolling model and full recovery checkpoints."""
import argparse
import contextlib
from datetime import timedelta
import fcntl
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from .config_runtime import *
from .preflight import plan,check_complete,check_teacher,inspect_dataset

class DistributedBatches:
    def __init__(self,dataset,batch_size,world,rank,seed):
        from torch.utils.data import DistributedSampler,BatchSampler
        self.sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,seed=seed,shuffle=True,drop_last=True)
        self.batches=BatchSampler(self.sampler,batch_size,drop_last=True);self.start=0
    def reset(self,epoch,start=0):
        if not 0<=start<=len(self.batches):raise ValueError("Invalid consumed batch cursor")
        self.sampler.set_epoch(epoch);self.start=start
    def __iter__(self):return itertools.islice(iter(self.batches),self.start,None)
    def __len__(self):return len(self.batches)-self.start

def validation_indices(dataset,count,seed):
    from collections import defaultdict
    import numpy as np
    groups=defaultdict(list)
    for i,e in enumerate(dataset.selected):groups[tuple(e["tasks"])].append(i)
    rng=np.random.default_rng(seed);keys=sorted(groups);out=[]
    for k in range(count):
        ei=int(rng.choice(groups[keys[k%len(keys)]]));e=dataset.selected[ei]
        out.append(int(dataset.offsets[ei]+rng.integers(e["length"])))
    return out

def capture_rng(device,generator):
    import torch
    import numpy as np
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),
        python=random.getstate(),numpy=np.random.get_state(),loader=generator.get_state())

def restore_rng(state,device,generator):
    import torch
    import numpy as np
    torch.set_rng_state(state["torch"]);torch.cuda.set_rng_state(state["cuda"],device)
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);generator.set_state(state["loader"])

def parser_for(variant=None,config=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(config) if config else None,required=config is None)
    p.add_argument("--stage",required=True,choices=["I","II"])
    p.add_argument("--gpus",help="Physical GPUs; Stage II may override execution placement")
    p.add_argument("--rank-to-gpu",help="Explicit physical GPU for each synchronized rank")
    p.add_argument("--elastic-runtime",action="store_true",help="Checkpoint-boundary placement changes with a global sampler")
    p.add_argument("--processes-per-gpu",type=int,help="Stage II synchronized ranks per GPU")
    p.add_argument("--profile-updates",type=int,default=0,help="Isolated real-update memory measurement; no formal checkpoint")
    p.add_argument("--profile-output",type=Path)
    p.add_argument("--minimum-free-mib",type=int,default=0,help="Admission floor before each serialized model load")
    p.add_argument("--stop-after-updates",type=int,help="Explicit optimizer update limit, never epochs")
    p.add_argument("--resume",help="Complete checkpoint directory in this same stage")
    p.add_argument("--workers",type=int,default=1);p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--offload-saved-tensors",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--share-frozen-encoders",action=argparse.BooleanOptionalAction,default=False)
    p.add_argument("--keep-models",type=int,default=1)
    p.add_argument("--execute",action="store_true",help="Actually launch training; absent means plan only")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    return p

def main(variant=None,config=None,argv=None):
    p=parser_for(variant,config);args=p.parse_args(argv);cfg=load_config(args.config,variant)
    if args.execute and args.dry_run:p.error("--execute and --dry-run are mutually exclusive")
    if args.workers<0 or args.cpu_threads<1 or args.keep_models!=1:p.error("Invalid runtime/checkpoint retention")
    placement=runtime_placement(cfg,args.stage,
        [int(x) for x in args.gpus.split(",")] if args.gpus else None,args.processes_per_gpu,
        [int(x) for x in args.rank_to_gpu.split(",")] if args.rank_to_gpu else None)
    if args.elastic_runtime and args.stage!="II":p.error("Elastic runtime is Stage II only")
    spec=plan(cfg,args.stage,args.stop_after_updates,require_sources=args.execute,placement=placement)
    if args.minimum_free_mib<0:p.error("Memory admission floor must be nonnegative")
    if args.profile_updates:
        if args.stage!="II" or args.resume or not 1<=args.profile_updates<=6 or args.profile_output is None:
            p.error("Profiling requires Stage II, 1..6 updates, a separate output, and no resume")
        args.profile_output=args.profile_output.resolve()
        base=Path(__file__).resolve().parents[1]/"capacity_profiles"
        if not args.profile_output.is_relative_to(base):p.error("Profile output must be under capacity_profiles")
        spec["output"]=str(args.profile_output)
        args.profile_output=spec["output"]
    elif args.profile_output is not None:p.error("Profile output requires --profile-updates")
    json.dumps(vars(args))  # Fail before launching CUDA if arguments cannot be recorded.
    if args.resume:
        path=Path(args.resume).resolve()
        if path.parent!=stage_root(cfg,args.stage):p.error("Resume must belong to this stage directory")
        meta,_=check_complete(path,require_resume=True)
        if meta["config"]!=cfg or meta["extra"].get("stage")!=args.stage:p.error("Resume identity mismatch")
        if meta["step"]>=spec["stop_after_updates"]:p.error("Resume checkpoint already reached requested stop")
    if not args.execute:
        print(json.dumps(dict(spec,launch_boundary="not executed"),indent=2));return
    if not args.gpus:p.error("--execute requires the configured --gpus")
    gpus=args.gpus.split(",")
    if gpus!=list(map(str,spec["physical_gpus"])):
        p.error("GPU list must match the configured rank placement")
    if args.worker:
        run(args,cfg,spec);return
    # Own one run lock for the lifetime of our child; never adopt another process.
    output=Path(spec["output"]);output.mkdir(parents=True,exist_ok=True)
    with (output/".training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/"arguments.json").exists() and not args.resume:
            raise ValueError("Run already exists; use its full --resume checkpoint")
        if shutil.disk_usage(output).free<10*2**30:
            raise OSError("Need at least 10 GiB free for the next atomic model checkpoint")
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=args.gpus
        env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
        env["PYTHONDONTWRITEBYTECODE"]="1"
        command=[cfg["model_python"],"-B","-m","torch.distributed.run","--standalone",
            "--nnodes=1",f"--nproc-per-node={spec['world_size']}","-m",
            "a1_local.training","--config",str(Path(args.config).resolve()),
            "--stage",args.stage,"--gpus",args.gpus,"--execute","--worker",
            "--workers",str(args.workers),"--cpu-threads",str(args.cpu_threads),
            "--keep-models",str(args.keep_models)]
        if args.stop_after_updates is not None:command+=["--stop-after-updates",str(args.stop_after_updates)]
        if args.resume:command+=["--resume",str(Path(args.resume).resolve())]
        command += ["--rank-to-gpu",",".join(map(str,spec["rank_to_gpu"])),
                    "--minimum-free-mib",str(args.minimum_free_mib)]
        if args.elastic_runtime:command += ["--elastic-runtime"]
        if args.profile_updates:
            command += ["--profile-updates",str(args.profile_updates),"--profile-output",str(args.profile_output)]
        for name in ("offload_saved_tensors","share_frozen_encoders"):
            command+=["--"+("" if getattr(args,name) else "no-")+name.replace("_","-")]
        atomic_json(output/"launch.json",dict(command=command,physical_gpus=gpus,plan=spec))
        subprocess.run(command,env=env,cwd=Path(__file__).resolve().parents[1],check=True)

def run(args,cfg,spec):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,Subset
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world!=spec["world_size"]:raise ValueError("World size differs from fixed stage protocol")
    first=False;local=spec["physical_gpus"].index(spec["rank_to_gpu"][rank])  # CUDA_VISIBLE_DEVICES index, shared for colocated ranks.
    os.environ["LOCAL_RANK"]=str(local);torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    from .numerics import configure_attention_backends
    attention_backends=configure_attention_backends()
    dist.init_process_group("nccl" if first else "gloo",timeout=timedelta(minutes=90))
    # Heavy imports only after explicit process-group setup (Accelerate initialization).
    from torch.nn.parallel import DistributedDataParallel as DDP
    from goal_local.optim import MasterAdamW
    from .checkpointing import commit_checkpoint,make_recovery,rank_recovery,tensor_bytes
    from .distributed import DistributedMaster
    from ablation.aba1.plus_common.model import GoalLocalPolicy,DistillationSystem
    from .model import make_stage_two
    from ablation.aba1.plus_common.data import IndexedFutureDataset
    output=Path(spec["output"]);seed=cfg["train_seed"]
    def status(**values):
        row=dict(rank=rank,pid=os.getpid(),stage=args.stage,time=time.time(),physical_gpu=spec["physical_gpus"][local],cuda_device=local,**values)
        atomic_json(output/f"status_rank{rank}.json",row)
        if rank==0:atomic_json(output/"status.json",row)
    def memory_record(phase,step=0):
        import resource
        free,total=torch.cuda.mem_get_info(device)
        row=dict(time=time.time(),pid=os.getpid(),rank=rank,physical_gpu=spec["physical_gpus"][local],
            phase=phase,step=step,allocated_mib=torch.cuda.memory_allocated(device)/2**20,
            reserved_mib=torch.cuda.memory_reserved(device)/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
            free_mib=free/2**20,total_mib=total/2**20,
            host_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic_json(output/f"memory_rank{rank}.json",row)
        return row
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
        from .startup import model_load_slot
        with model_load_slot(output,rank,status):
            admission_started=time.monotonic()
            while torch.cuda.mem_get_info(device)[0]/2**20<args.minimum_free_mib:
                status(phase="waiting_memory",step=0,required_free_mib=args.minimum_free_mib)
                memory_record("waiting_memory")
                if time.monotonic()-admission_started>600:
                    raise RuntimeError("GPU memory admission floor not met; stopped before allocating models")
                time.sleep(5)
            status(phase="loading")
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            sources=source_paths(cfg,args.stage)
            recovery=None
            if args.resume:
                check_complete(args.resume,require_resume=True,verify_hash=True)
                recovery=torch.load(Path(args.resume)/"trainer.pt",map_location="cpu",mmap=True,weights_only=False)
            teacher=None
            if args.stage=="II":
                check_teacher(sources["teacher"],cfg)
                teacher=GoalLocalPolicy.load(sources["teacher"],with_action=not bool(args.resume),legacy_config=cfg)
                if args.resume:
                    policy=GoalLocalPolicy.load(args.resume)
                    teacher.requires_grad_(False);teacher.frozen_teacher=True;teacher.eval()
                    system=DistillationSystem(policy,teacher)
                else:
                    system=make_stage_two(teacher,cfg,sources.get("student_init"));policy=system.student
                if args.share_frozen_encoders:
                    for a,b in ((policy.qwen.visual,teacher.qwen.visual),(policy.dino,teacher.dino)):
                        if any(p.requires_grad for p in a.parameters()) or any(p.requires_grad for p in b.parameters()):
                            raise ValueError("Only frozen modules may share storage")
                        left,right=a.state_dict(),b.state_dict()
                        if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):
                            raise ValueError("Frozen encoder mismatch")
                    teacher.qwen.visual=policy.qwen.visual;teacher.dino=policy.dino
            else:
                if args.resume:policy=GoalLocalPolicy.load(args.resume)
                elif args.stage=="action2":
                    check_complete(sources["previous_action"],verify_hash=True)
                    policy=GoalLocalPolicy.load(sources["previous_action"])
                else:policy=GoalLocalPolicy.from_baseline(cfg)
                system=policy
            if policy.statistics!=json.loads(Path(cfg["statistics_path"]).read_text()):
                raise ValueError("Policy/action normalization statistics mismatch")
            expected_mode="posterior" if args.stage=="I" else "student"
            if policy.cfg!=cfg or policy.mode!=expected_mode:raise ValueError("Policy contract mismatch")
            system.to(device)
            memory_record("model_loaded")
            params=[p for p in policy.parameters() if p.requires_grad]
            if teacher is not None:
                if any(p.requires_grad for p in teacher.parameters()):raise ValueError("Teacher is trainable")
                if {p.data_ptr() for p in params}&{p.data_ptr() for p in teacher.parameters()}:
                    raise ValueError("Teacher shares trainable storage")
            teacher_versions=[p._version for p in teacher.parameters()] if teacher is not None else None
        local_future,goal_future=future_roles(cfg,args.stage)
        train=IndexedFutureDataset(cfg["dataset_root"],policy.statistics,"train",cfg["split_seed"],
            local=local_future,goal=goal_future)
        val=IndexedFutureDataset(cfg["dataset_root"],policy.statistics,"val",cfg["split_seed"],
            local=local_future,goal=goal_future)
        if (len(train),len(val))!=(2125407,112629):raise ValueError("Dataset anchors changed")
        from .elastic import GlobalBatches,validate_resume_contract
        batches=GlobalBatches(train,world,rank,seed) if args.elastic_runtime else DistributedBatches(train,spec["batch_size"],world,rank,seed)
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
                if key=="teacher":
                    marker=json.loads((path/"checkpoint_complete.json").read_text())
                    if source_hashes[0][key]["sha256"]!=marker["files"]["model.pt"]["sha256"]:
                        raise ValueError("Teacher model checksum differs from committed checkpoint")
        dist.broadcast_object_list(source_hashes,src=0)
        contract=dict(model_loading="serialized_file_lock",physical_gpus=spec["physical_gpus"],rank_to_gpu=spec["rank_to_gpu"],config=cfg,stage=args.stage,world=world,batch_size=spec["batch_size"],
            accumulation=spec["accumulation"],accumulation_per_rank=spec["accumulation_per_rank"],gradient_cpu_offload=True,attention_backends=attention_backends,schedule_total_steps=spec["schedule_total_steps"],
            offload_saved_tensors=args.offload_saved_tensors,share_frozen_encoders=args.share_frozen_encoders,
            code_sha256=digest_json(identity),versions=runtime_environment(),asset_sha256=asset_identity(cfg),
            source_paths={k:str(v.resolve()) for k,v in sources.items()},source_hashes=source_hashes[0],
            data_manifest_sha256=train.source_digest,split_sha256=digest_json(val.validation_episodes))
        if args.elastic_runtime:contract["sampler"]="global_batch32_v1"
        epoch=0;consumed=0;start=0
        random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
        if recovery is not None:
            resized=validate_resume_contract(recovery["contract"],contract,args.elastic_runtime)
            optimizer.load_state_dict(recovery["optimizer"] if first or rank==0 else None)
            if scheduler is not None:scheduler.load_state_dict(recovery["scheduler"])
            start=recovery["step"]
            if policy.source_step!=start:raise ValueError("Model/optimizer step mismatch")
            if resized:
                epoch,consumed=batches.resume_step(start)
                reseed=seed+rank+1009*start
                random.seed(reseed);np.random.seed(reseed);torch.manual_seed(reseed);generator.manual_seed(reseed)
                if rank==0:record(dict(split="elastic_resume",step=start,previous_world=recovery["contract"]["world"],world=world,
                    rng_policy="reseed on rank-count change; global data cursor and full optimizer preserved"))
            else:
                own_state=rank_recovery(recovery,rank,world)
                epoch=own_state["epoch"];consumed=own_state["consumed"]
                restore_rng(own_state["rng"],device,generator);del own_state
            del recovery
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
                for batch in validation:
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
            if rank==0:
                final=checkpoint_path(cfg,args.stage,step);temp=output/f".step_{step:06d}.incomplete"
                if final.exists():raise FileExistsError(final)
                if temp.exists():shutil.rmtree(temp)
                recovery=make_recovery(step,contract,optimizer.state_dict(),
                    scheduler.state_dict() if scheduler is not None else None,rank_states)
                required=tensor_bytes(policy.state_dict())+tensor_bytes(recovery)+2**30
                if shutil.disk_usage(output).free<required:
                    raise OSError(f"Insufficient checkpoint space: need {required/2**30:.1f} GiB; previous checkpoint retained")
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
        # Both stages save at 50-update intervals, never at step zero.
        evaluate(start);memory_record("validation",start);torch.cuda.empty_cache();iterator=iter(loader)
        update_limit=args.profile_updates or spec["stop_after_updates"]
        reconfigured=False
        for completed in range(start,update_limit):
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
                memory_record("backward",completed)
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
            memory_record("optimizer_update",step)
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
            if not args.profile_updates and (step%spec["save_every"]==0 or step==spec["stop_after_updates"]):
                save(step)
                request=[None]
                if rank==0 and args.elastic_runtime:
                    if step<update_limit and (output/"autoscale_control.json").exists():
                        from .autoscale import checkpoint_decision
                        try:checkpoint_decision(output,step,spec["rank_to_gpu"])
                        except Exception as error:
                            record(dict(split="autoscale_check_failed",step=step,error=repr(error)))
                    try:request[0]=json.loads((output/"elastic_reconfigure.json").read_text())
                    except FileNotFoundError:pass
                dist.broadcast_object_list(request,src=0)
                if request[0] and step<update_limit and step>=request[0].get("minimum_step",step):
                    status(phase="reconfigure_ready",step=step)
                    if rank==0:atomic_json(output/"elastic_ready.json",dict(step=step,request=request[0],time=time.time()))
                    reconfigured=True;break
        if args.profile_updates:
            status(phase="profile_completed",step=update_limit)
            atomic_json(output/f"profile_completed_rank{rank}.json",dict(updates=update_limit,
                formal_training=False,memory=memory_record("profile_completed",update_limit)))
        elif not reconfigured:
            status(phase="completed",step=spec["stop_after_updates"],
                cumulative_step=spec["cumulative_offset"]+spec["stop_after_updates"])
        train.close();val.close()
    except BaseException as exc:
        status(phase="failed",error=repr(exc));raise
    finally:dist.destroy_process_group()

if __name__=="__main__":main()
