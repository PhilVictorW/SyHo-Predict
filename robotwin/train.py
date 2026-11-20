"""Two-stage training with auditable sampling, fixed validation and tmux telemetry."""
import argparse,json,math,random,time,os,shutil,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .data import FutureDataset
from .model import RobotwinPolicy as GoalLocalPolicy
from goal_local.model import make_stage_two
from goal_local.optim import MasterAdamW


def atomic_json(path,payload):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(path)


def validation_indices(dataset,count,seed):
    rng=np.random.default_rng(seed)
    return rng.integers(0,len(dataset),size=count).tolist()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',default='configs/goal_local/default.json');p.add_argument('--stage',choices=['I'],required=True)
    p.add_argument('--stage-one');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=32)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--device',default='cuda:0')
    p.add_argument('--save-every',type=int,default=1000);p.add_argument('--eval-every',type=int,default=250);p.add_argument('--eval-batches',type=int,default=128)
    p.add_argument('--diagnostic-batches',type=int,default=8);p.add_argument('--keep-checkpoints',type=int,default=2)
    recovery=p.add_mutually_exclusive_group()
    recovery.add_argument('--resume')
    recovery.add_argument('--warm-start',help='Model directory: retain saved step/weights, reset optimizer and RNG')
    p.add_argument('--save-on-first-update',action='store_true')
    p.add_argument('--optimizer-device',choices=['cpu','gpu'],default='cpu');p.add_argument('--save-initial',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.batch_size,args.accumulation,args.save_every,args.eval_every,args.eval_batches,args.keep_checkpoints)<1:p.error('Counts must be positive')
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.40, torch.device(args.device))
    output=Path(args.output).resolve();root=Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root) or output.is_relative_to((root/'results/Checkpoints').resolve()):p.error('Output must stay inside this independent workspace, outside shared Checkpoints')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.jsonl').exists() and not args.resume:p.error('Existing run requires --resume or a new output directory')
    status=dict(phase='loading_model',step=0,micro=0,total_steps=args.steps,started_at=time.time(),pid=os.getpid())
    def update(**kwargs):
        status.update(kwargs);status['updated_at']=time.time();atomic_json(output/'status.json',status)
    def record(row):
        row['time']=time.time();print(json.dumps(row),flush=True)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    update();atomic_json(output/'arguments.json',vars(args))
    try:
        cfg=json.loads(Path(args.config).read_text());seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        recovery_path=Path(args.resume).parent if args.resume else Path(args.warm_start) if args.warm_start else None
        if args.stage=='I':
            policy=GoalLocalPolicy.load(recovery_path) if recovery_path else GoalLocalPolicy.from_baseline(cfg)
            if policy.mode!='posterior':raise ValueError('Stage I requires posterior model')
            model=policy
            if hasattr(policy,'initialization_report'):atomic_json(output/'initialization.json',policy.initialization_report)
        else:
            if not args.stage_one:raise ValueError('--stage-one is required')
            posterior=GoalLocalPolicy.load(args.stage_one)
            if posterior.source_step<1:raise ValueError('Stage II requires trained new Stage-I teacher')
            teacher_step=posterior.source_step
            model=make_stage_two(posterior);policy=model.student;cfg=policy.cfg
            atomic_json(output/'teacher_source.json',dict(path=str(Path(args.stage_one).resolve()),step=teacher_step,
                losses=['student_action_flow','local_distillation','goal_distillation'],lambda_local=cfg['lambda_local'],lambda_goal=cfg['lambda_goal']))
            if recovery_path:
                saved_policy=GoalLocalPolicy.load(recovery_path)
                if saved_policy.mode!='student' or saved_policy.cfg!=cfg:raise ValueError('Student recovery model/configuration mismatch')
                if args.warm_start:
                    source=json.loads((recovery_path.parent/'teacher_source.json').read_text())
                    if Path(source['path']).resolve()!=Path(args.stage_one).resolve() or source['step']!=teacher_step:
                        raise ValueError('Warm-start teacher mismatch')
                policy.load_state_dict(saved_policy.state_dict());policy.source_step=saved_policy.source_step
                del saved_policy
        if policy.cfg!=cfg:raise ValueError('Recovery model/configuration mismatch')
        update(phase='dataset_audit');model.to(args.device)
        ds=FutureDataset(cfg['dataset_root'],policy.statistics,'train',cfg['seed'],cfg['val_ratio'])
        val=FutureDataset(cfg['dataset_root'],policy.statistics,'val',cfg['seed'],cfg['val_ratio'])
        train_ids={e['episode_index'] for e in ds.selected};val_ids={e['episode_index'] for e in val.selected}
        assert train_ids.isdisjoint(val_ids)
        audit=ds.audit();audit.update(train_episodes=len(train_ids),validation_episodes_count=len(val_ids),validation_frames=len(val),
            sampling='uniform shuffled anchors over train frames, no future-validity filtering',effective_batch=args.batch_size*args.accumulation,
            nominal_epochs=args.steps*args.batch_size*args.accumulation/len(ds))
        atomic_json(output/'dataset_audit.json',audit);atomic_json(output/'config.json',cfg)
        vi=validation_indices(val,args.eval_batches*args.batch_size,seed+1)
        atomic_json(output/'validation_samples.json',dict(seed=seed+1,indices=vi,episodes=val.validation_episodes))
        opts=dict(num_workers=args.workers,pin_memory=True)
        if args.workers:opts.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
        generator=torch.Generator().manual_seed(seed)
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=generator,**opts)
        validation=DataLoader(Subset(val,vi),batch_size=args.batch_size,shuffle=False,
            generator=torch.Generator().manual_seed(seed+2),**opts)
        parameters=[x for x in model.parameters() if x.requires_grad]
        teacher_versions={}
        if args.stage=='II':
            teacher_parameters=dict(model.teacher.named_parameters())
            teacher_ids={id(p) for p in teacher_parameters.values()}
            if any(p.requires_grad for p in teacher_parameters.values()) or any(id(p) in teacher_ids for p in parameters):
                raise RuntimeError('Frozen teacher must be absent from optimizer')
            teacher_storage={p.data_ptr() for p in teacher_parameters.values()}
            if any(p.data_ptr() in teacher_storage for p in model.student.parameters()):
                raise RuntimeError('Teacher and student parameter storage must be independent')
            teacher_versions={k:p._version for k,p in teacher_parameters.items()}
            atomic_json(output/'teacher_freeze_audit.json',dict(teacher_step=teacher_step,all_frozen=True,
                excluded_from_optimizer=True,independent_parameter_storage=True,teacher_parameter_tensors=len(teacher_parameters),checked_after_student_step=0))

        atomic_json(output/'parameter_audit.json',dict(trainable=sum(x.numel() for x in parameters),frozen=sum(x.numel() for x in model.parameters() if not x.requires_grad),
            optimizer='AdamW with FP32 master weights',optimizer_device=args.optimizer_device,vlm_layers=len(policy.qwen.language_model.layers),gpu_allocator_limit_fraction=.40))
        update(phase='initializing_fp32_optimizer')
        optim=MasterAdamW(parameters,cfg['learning_rate'],cfg['weight_decay'],offload=args.optimizer_device=='cpu')
        def lr(step):
            warm=min(cfg['warmup_steps'],max(1,args.steps//10))
            return min(1,(step+1)/warm)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-warm)/max(1,args.steps-warm))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optim.optimizer,lr);start=0;best=float('inf')
        if args.resume:
            resume=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
            if resume['stage']!=args.stage or resume['config']!=cfg:raise ValueError('Resume configuration/stage mismatch')
            if args.stage=='II' and resume['teacher_source']!=str(Path(args.stage_one).resolve()):raise ValueError('Resume teacher mismatch')
            optim.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler']);start=resume['step'];best=resume.get('best_validation',best)
            torch.set_rng_state(resume['rng']);random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng'])
            generator.set_state(resume['loader_rng'])
            if 'cuda_rng' in resume:torch.cuda.set_rng_state_all(resume['cuda_rng'])
            if policy.source_step!=start:raise ValueError('Model/trainer step mismatch')
            del resume
        elif args.warm_start:
            start=policy.source_step
            if start<1:raise ValueError('Warm start requires trained weights')
            # Continue the original global LR schedule; Adam moments and FP32 masters
            # cannot be recovered from a BF16 model-only checkpoint.
            scheduler.last_epoch=start
            scheduler._step_count=start+1
            for group,base_lr in zip(optim.optimizer.param_groups,scheduler.base_lrs):group['lr']=base_lr*lr(start)
            scheduler._last_lr=[group['lr'] for group in optim.optimizer.param_groups]
            atomic_json(output/'warm_start.json',dict(source=str(recovery_path.resolve()),step=start,
                optimizer_reset=True,rng_reset=True,scheduler_global_step=start))
        if start>=args.steps:raise ValueError('Recovery step must be below total steps')
        update(step=start,recovery_source=str(recovery_path) if recovery_path else None,
            recovery_mode='full' if args.resume else 'weights_only' if args.warm_start else 'fresh')
        device=torch.device(args.device);cuda_indices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        def evaluate(step):
            update(phase='validation',step=step);model.eval();rows=[];diag=[]
            with torch.random.fork_rng(devices=cuda_indices),torch.no_grad():
                torch.manual_seed(seed+10000)
                for i,batch in enumerate(validation):
                    update(phase=f'validation {i+1}/{args.eval_batches}')
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        if args.stage=='I':
                            encoded=policy.encode(batch);cond=policy.action_conditions(batch,encoded)
                            def flow(conditions):return policy.flow.loss(batch['actions'].to(device),batch['actions_mask'].to(device),batch['action_hz'].to(device),batch['embodiment_id'].to(device),batch['remaining_actions'].to(device),conditions,repeats=cfg['repeated_diffusion_steps'])
                            rng=torch.get_rng_state();crng=torch.cuda.get_rng_state(device) if cuda_indices else None
                            value=flow(cond);rows.append(dict(loss=float(value),flow=float(value)))
                            if i<args.diagnostic_batches:
                                after=torch.get_rng_state();cafter=torch.cuda.get_rng_state(device) if cuda_indices else None
                                torch.set_rng_state(rng)
                                if crng is not None:torch.cuda.set_rng_state(crng,device)
                                ablated=tuple(list(cond[:-1])+[torch.zeros_like(cond[-1])]);disabled=flow(ablated)
                                diag.append((float(value),float(disabled)))
                                torch.set_rng_state(after)
                                if cafter is not None:torch.cuda.set_rng_state(cafter,device)
                        else:rows.append({k:float(v) for k,v in model(batch).items()})
            result=dict(step=step,split='validation',**{k:sum(x[k] for x in rows)/len(rows) for k in rows[0]})
            if diag:result.update(diagnostic_flow=sum(x[0] for x in diag)/len(diag),without_roles_flow=sum(x[1] for x in diag)/len(diag))
            if not all(math.isfinite(v) for k,v in result.items() if isinstance(v,float)):raise FloatingPointError('Nonfinite validation')
            record(result);return result['flow']
        if args.save_initial and not recovery_path:
            update(phase='saving_initial_weights');policy.save(output/'initial',step=0,extra={'stage':args.stage,'initialization_only':True})
        baseline=evaluate(start)
        if baseline<best:best=baseline
        iterator=iter(loader);durations=[]
        for step in range(start,args.steps):
            begin=time.time();model.train();optim.zero_grad();metrics={};local_count=goal_count=sample_count=0
            for micro in range(args.accumulation):
                update(phase='training',step=step,micro=micro+1)
                try:batch=next(iterator)
                except StopIteration:iterator=iter(loader);batch=next(iterator)
                local_count+=int(batch['local_valid'].sum());goal_count+=int(batch['goal_frame_valid'].any(-1).sum());sample_count+=len(batch['local_valid'])
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):losses=model(batch);loss=losses['loss']/args.accumulation
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
                loss.backward()
                for k,v in losses.items():metrics[k]=metrics.get(k,0)+float(v.detach())/args.accumulation
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
            update(phase='optimizer_step');optim.step();scheduler.step();duration=time.time()-begin;durations.append(duration)
            if args.stage=='II':
                if model.teacher.training or any(p.requires_grad or p.grad is not None or p._version!=teacher_versions[k] for k,p in teacher_parameters.items()):
                    raise RuntimeError('Teacher changed during student training')
                if step==start or (step+1)%args.eval_every==0:
                    atomic_json(output/'teacher_freeze_audit.json',dict(teacher_step=teacher_step,all_frozen=True,
                        excluded_from_optimizer=True,independent_parameter_storage=True,no_teacher_gradients=True,
                        parameter_versions_unchanged=True,checked_after_student_step=step+1))

            record(dict(step=step+1,split='train',grad_norm=float(norm),lr=optim.optimizer.param_groups[0]['lr'],update_seconds=duration,
                local_valid_fraction=local_count/sample_count,goal_valid_fraction=goal_count/sample_count,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**metrics))
            eta=(args.steps-step-1)*sum(durations[-20:])/len(durations[-20:]);update(step=step+1,micro=0,eta_seconds=eta)
            if (step+1)%args.eval_every==0:
                score=evaluate(step+1)
                if score<best:
                    best=score;update(phase='saving_best');policy.save(output/'best',step+1,extra={'stage':args.stage,'validation_flow':score})
            if (step+1)%args.save_every==0 or step+1==args.steps or (args.save_on_first_update and step==start):
                destination=output/f'step_{step+1:06d}';update(phase='saving_checkpoint')
                policy.save(destination,step+1,extra={'stage':args.stage,'teacher_source':args.stage_one})
                trainer=dict(stage=args.stage,config=cfg,teacher_source=str(Path(args.stage_one).resolve()) if args.stage_one else None,step=step+1,
                    optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),best_validation=best,rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),loader_rng=generator.get_state())
                if cuda_indices:trainer['cuda_rng']=torch.cuda.get_rng_state_all()
                torch.save(trainer,destination/'trainer.tmp');(destination/'trainer.tmp').replace(destination/'trainer.pt')
                atomic_json(destination/'checkpoint_complete.json',dict(step=step+1,owner='goal_local.train'))
                del trainer
                completed=sorted(x.parent for x in output.glob('step_*/checkpoint_complete.json'))
                # Keep every 500-step model; retain only the newest two large optimizer states.
                for old in completed[:-2]:
                    trainer_file=old/'trainer.pt'
                    if trainer_file.exists():
                        trainer_file.unlink()
                        atomic_json(old/'checkpoint_complete.json',dict(step=int(old.name.split('_')[1]),owner='robotwin.train',model_only=True))
                update(checkpoint=str(destination),phase='training')
        update(phase='completed',step=args.steps,eta_seconds=0)
    except BaseException as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':main()
