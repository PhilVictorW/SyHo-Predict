"""Exact-epoch, single-process training. No work starts without an explicit CLI call."""
import argparse
import contextlib
import fcntl
import itertools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

from .config import (ABA, ROOT, load_config, future_roles, atomic_json,
                     source_identity, digest_json, file_sha256)


def epoch_updates(size, batch_size, accumulation):
    return math.ceil(math.ceil(size / batch_size) / accumulation)


def learning_rate_factor(completed, total, warmup):
    warm = min(warmup, max(1, total // 10))
    if completed < warm:
        return (completed + 1) / warm
    progress = min(1., (completed - warm + 1) / max(1, total - warm))
    return .05 + .95 * .5 * (1 + math.cos(math.pi * progress))


class EpochBatchSampler:
    """Cursor is consumed anchors, not prefetched batches; independent of global RNG."""
    def __init__(self, size, batch_size, seed, epoch, cursor=0):
        if not 0 <= cursor <= size or (cursor != size and cursor % batch_size):
            raise ValueError('Invalid sampler cursor')
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.cursor = seed, epoch, cursor

    def __iter__(self):
        import torch
        g = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        order = torch.randperm(self.size, generator=g).tolist()
        for start in range(self.cursor, self.size, self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self):
        return math.ceil((self.size - self.cursor) / self.batch_size)


def accumulation_groups(iterator, accumulation):
    iterator = iter(iterator)
    while True:
        group = list(itertools.islice(iterator, accumulation))
        if not group:
            return
        yield group


def capture_rng(device=None):
    import torch
    import numpy as np
    state = dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())
    if device is not None and torch.device(device).type == 'cuda':
        index=torch.device(device).index
        if index is None: index=torch.cuda.current_device()
        state['cuda'] = {index:torch.cuda.get_rng_state(index)}
    return state


def restore_rng(state):
    import torch
    import numpy as np
    torch.set_rng_state(state['torch']); random.setstate(state['python']); np.random.set_state(state['numpy'])
    if 'cuda' in state:
        for index,value in state['cuda'].items(): torch.cuda.set_rng_state(value,index)


def validate_paths(cfg):
    for key in ('baseline_checkpoint', 'statistics_path', 'qwen_path', 'dino_path', 'dataset_root'):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f'{key}: {cfg[key]}')


def validate_teacher_metadata(path, cfg):
    from goal_local.eval_common import model_metadata
    from .config import FORMAT
    path=Path(path)
    meta=model_metadata(path/'model.pt')
    if meta['mode']!='posterior' or meta['step']<1:
        raise ValueError('Stage II requires trained posterior weights, never another Student')
    if meta['format']==FORMAT:
        teacher=meta['config']
        allowed={cfg['ablation']} if cfg['ablation'] in ('A1','A2') else {'A3','A4'}
        if teacher['ablation'] not in allowed:
            raise ValueError('Single-scale and full teachers cannot be interchanged')
        for key in ('local_slots','goal_slots','baseline_checkpoint','dataset_root','flow','split_seed','val_ratio'):
            if teacher[key]!=cfg[key]:raise ValueError(f'Teacher contract mismatch: {key}')
        marker=json.loads((path/'checkpoint_complete.json').read_text())
        if marker['step']!=meta['step']:raise ValueError('Teacher completion marker mismatch')
    elif meta['format']=='goal_local_v1' and cfg['ablation'] in ('A3','A4'):
        teacher=meta['config']
        if teacher['slots']!=8:raise ValueError('Expected the historical full teacher')
        if teacher['seed']!=cfg['split_seed'] or teacher['val_ratio']!=cfg['val_ratio']:
            raise ValueError('Historical teacher used a different episode split')
        for key in ('baseline_checkpoint','dataset_root','statistics_path','flow','goal_horizons','horizon','llm_layers','distill_layers'):
            if teacher[key]!=cfg[key]:raise ValueError(f'Historical teacher contract mismatch: {key}')
    else:
        raise ValueError('Unsupported teacher; A1/A2 must train their own single-scale Stage I')
    return meta


def build_parser(variant, config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(config))
    parser.add_argument('--stage', choices=['action'] if variant == 'A0' else ['I', 'II'],
                        default='action' if variant == 'A0' else None, required=variant != 'A0')
    parser.add_argument('--epochs', type=int, default=500 if variant == 'A0' else None,
                        required=variant != 'A0')
    parser.add_argument('--stage-one', help='Explicit trained teacher directory; required for stage II')
    parser.add_argument('--student-init', help='Optional zero-update paired initialization directory')
    parser.add_argument('--resume', help='Complete checkpoint directory; restores all training state')
    parser.add_argument('--output', required=True, help='New run directory under this variant/libero/runs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, help='Model/data-order seed; does not alter split_seed')
    parser.add_argument('--optimizer-device', choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--save-every-updates', type=int, default=0, help='Optional recovery saves between epochs')
    parser.add_argument('--keep-checkpoints', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='Validate plan and paths only; no model, GPU or output run')
    return parser


def main(variant, config, argv=None):
    parser = build_parser(variant, config)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, variant)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if min(args.epochs, args.batch_size, args.accumulation, args.keep_checkpoints, args.cpu_threads) < 1:
        parser.error('Epochs, batches, accumulation and retention must be positive')
    if args.workers < 0 or args.save_every_updates < 0:
        parser.error('Worker/save counts must be nonnegative')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This implementation is single-process; DDP would change the exact-epoch contract')
    use_local, use_goal = future_roles(cfg, args.stage)
    if args.stage == 'II' and not args.stage_one:
        parser.error('--stage-one is required; no implicit historical teacher')
    if args.stage != 'II' and (args.stage_one or args.student_init):
        parser.error('Only Stage II accepts a teacher or paired initialization')
    if args.resume and args.student_init:
        parser.error('Full resume and fresh paired initialization are mutually exclusive')
    output = Path(args.output).resolve()
    allowed = (ABA / variant / 'libero/runs').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        parser.error(f'Output must be a run subdirectory of {allowed}')
    if args.resume and Path(args.resume).resolve().parent != output/'checkpoints':
        parser.error('Resume must use a checkpoint from this same output run')
    validate_paths(cfg)
    for path in (args.stage_one, args.student_init, args.resume):
        if path and not (Path(path) / 'model.pt').is_file():
            parser.error(f'Checkpoint does not exist: {path}')
    if args.stage_one:
        validate_teacher_metadata(args.stage_one,cfg)
    plan = dict(variant=variant, stage=args.stage, epochs=args.epochs,
                local_slots=cfg['local_slots'], goal_slots=cfg['goal_slots'],
                interaction=cfg['interaction'], decode_local=use_local, decode_goal=use_goal,
                batch_size=args.batch_size, accumulation=args.accumulation,
                expected_train_anchors=260726,
                expected_updates=args.epochs*epoch_updates(260726,args.batch_size,args.accumulation),
                teacher=args.stage_one, output=str(output), dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'status.json').exists() and not args.resume:
            raise ValueError('Run exists; use full --resume or a new output directory')
        atomic_json(output/'plan.json',plan)
        try:
            run(args,cfg,output,use_local,use_goal)
        except BaseException as error:
            atomic_json(output/'status.json',dict(phase='failed',error=repr(error),time=time.time()))
            raise


def run(args, cfg, output, use_local, use_goal):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from .data import FutureDataset
    from .model import GoalLocalPolicy, make_stage_two
    from goal_local.optim import MasterAdamW
    from goal_local.train import validation_indices

    torch.set_num_threads(args.cpu_threads)
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    atomic_json(output/'status.json',dict(phase='loading',pid=os.getpid(),time=time.time()))
    atomic_json(output/'arguments.json',vars(args)); atomic_json(output/'config.json',cfg)
    atomic_json(output/'implementation.json',{
        str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    teacher_source = None
    if args.stage == 'II':
        teacher_path = Path(args.stage_one).resolve()
        teacher_source = source_identity(teacher_path/'model.pt')
        from goal_local.eval_common import model_metadata
        from .config import FORMAT
        metadata=model_metadata(teacher_path/'model.pt')
        if metadata['format']==FORMAT:
            marker=json.loads((teacher_path/'checkpoint_complete.json').read_text())
            if teacher_source['sha256']!=marker['model_sha256']:
                raise ValueError('Teacher checksum mismatch')
        teacher = GoalLocalPolicy.load(teacher_path, legacy_config=cfg)
        teacher_source.update(step=teacher.source_step, extra=teacher.source_extra)
        initial = GoalLocalPolicy.load(args.student_init) if args.student_init else None
        model = make_stage_two(teacher,cfg,initial)
        del initial
        policy = model.student
        atomic_json(output/'teacher_source.json',teacher_source)
    elif args.resume:
        policy = GoalLocalPolicy.load(args.resume); model=policy
    else:
        policy = GoalLocalPolicy.from_baseline(cfg); model=policy
        atomic_json(output/'initialization.json',policy.initialization_report)
        atomic_json(output/'baseline_source.json',source_identity(cfg['baseline_checkpoint']))
    expected_mode = 'posterior' if args.stage == 'I' else 'student'
    if policy.cfg != cfg or policy.mode != expected_mode:
        raise ValueError('Model configuration/mode mismatch')
    resume = None
    if args.resume:
        path = Path(args.resume)
        if not (path/'checkpoint_complete.json').exists():
            raise ValueError('Full resume requires a complete training checkpoint')
        marker=json.loads((path/'checkpoint_complete.json').read_text())
        if file_sha256(path/'model.pt') != marker['model_sha256']:
            raise ValueError('Resume model checksum mismatch')
        resume = torch.load(path/'trainer.pt',map_location='cpu',weights_only=False,mmap=True)
        if args.stage == 'II':
            saved = GoalLocalPolicy.load(path)
            if saved.cfg != cfg or saved.mode != expected_mode:
                raise ValueError('Student resume configuration mismatch')
            policy.load_state_dict(saved.state_dict()); policy.source_step=saved.source_step
            del saved
    model.to(device)
    statistics = policy.statistics
    train = FutureDataset(cfg['dataset_root'],statistics,'train',cfg['split_seed'],cfg['val_ratio'],local=use_local,goal=use_goal)
    # Reuse immutable metadata and arrays; do not read the entire Parquet dataset twice.
    import copy
    val = copy.copy(train)
    val.split='val'
    valid_ids=set(train.validation_episodes)
    val.selected=[e for e in train.episodes if e['episode_index'] in valid_ids]
    val.offsets=np.cumsum([0]+[e['length'] for e in val.selected])
    audit=train.audit()
    if len(train)!=260726 or len(val)!=12739 or len(train.selected)!=1614 or len(val.selected)!=79:
        raise ValueError('Dataset split differs from the frozen LIBERO protocol')
    data_files=[]
    data_root=Path(cfg['dataset_root'])
    for folder in ('meta','data'):
        for path in sorted((data_root/folder).rglob('*.parquet')):
            data_files.append(source_identity(path))
    dataset_version=dict(info=source_identity(data_root/'meta/info.json'),parquet=data_files,
                         video_integrity='Timestamp/episode checks on decoded frames; video bytes not hashed')
    atomic_json(output/'dataset_version.json',dataset_version)
    split=dict(train=[e['episode_index'] for e in train.selected],validation=train.validation_episodes,
               lengths=[(e['episode_index'],e['length']) for e in train.episodes])
    if set(split['train']) & set(split['validation']): raise ValueError('Episode leakage')
    atomic_json(output/'offline_split.json',split)
    vi=validation_indices(val,128,cfg['split_seed']+1)
    atomic_json(output/'validation_anchors.json',dict(indices=vi,noise_seed=cfg['split_seed']+10000))
    atomic_json(output/'dataset_audit.json',dict(audit,validation_frames=len(val),split_sha256=digest_json(split)))
    total=args.epochs*epoch_updates(len(train),args.batch_size,args.accumulation)
    contract=dict(config=cfg,stage=args.stage,epochs=args.epochs,batch_size=args.batch_size,
                  accumulation=args.accumulation,total_updates=total,teacher_source=teacher_source,
                  split_sha256=digest_json(split),dataset_version=digest_json(dataset_version),validation_indices=vi)
    atomic_json(output/'training_contract.json',contract)
    parameters=[x for x in model.parameters() if x.requires_grad]
    teacher_versions={}
    if args.stage=='II':
        teacher_tensors=dict(model.teacher.named_parameters()) | dict(model.teacher.named_buffers())
        teacher_versions={k:v._version for k,v in teacher_tensors.items()}
        storage={p.data_ptr() for p in model.teacher.parameters()}
        if any(p.requires_grad for p in model.teacher.parameters()) or any(p.data_ptr() in storage for p in policy.parameters()):
            raise RuntimeError('Teacher must be frozen and independently stored')
    atomic_json(output/'parameter_audit.json',dict(
        trainable=sum(p.numel() for p in parameters), frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        trainable_names=[n for n,p in policy.named_parameters() if p.requires_grad],
        frozen_names=[n for n,p in policy.named_parameters() if not p.requires_grad]))
    optim=MasterAdamW(parameters,cfg['learning_rate'],cfg['weight_decay'],offload=args.optimizer_device=='cpu')
    epoch=cursor=step=seen=0; best=float('inf'); best_checkpoint=None
    if resume is not None:
        if resume['contract'] != contract: raise ValueError('Resume contract differs; no implicit schedule/teacher changes')
        optim.load_state_dict(resume['optimizer'])
        epoch,cursor,step,seen=(resume[k] for k in ('epoch','cursor','step','seen_anchors'))
        best=resume['best_flow']; best_checkpoint=resume['best_checkpoint']
        if policy.source_step!=step or seen!=epoch*len(train)+cursor:
            raise ValueError('Inconsistent resume counters')
        expected_step=epoch*epoch_updates(len(train),args.batch_size,args.accumulation)
        if cursor:
            if cursor % (args.batch_size*args.accumulation): raise ValueError('Resume is not at an optimizer boundary')
            expected_step+=cursor//(args.batch_size*args.accumulation)
        if expected_step!=step or step>=total: raise ValueError('Invalid or completed resume')
        restore_rng(resume['rng']); del resume
    elif args.stage=='II':
        policy.save(output/'initial',0,extra=dict(initialization_only=True,teacher_source=teacher_source))
        atomic_json(output/'initial/initialization_complete.json',dict(step=0,teacher_source=teacher_source))
    opts=dict(num_workers=args.workers,pin_memory=device.type=='cuda')
    if args.workers: opts.update(multiprocessing_context='spawn',prefetch_factor=2)
    validation=DataLoader(Subset(val,vi),batch_size=args.batch_size,shuffle=False,
        generator=torch.Generator().manual_seed(cfg['split_seed']+2),**opts)
    def record(row):
        row.update(time=time.time(),step=step,epoch=epoch,cursor=cursor,seen_anchors=seen)
        with (output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def evaluate():
        state=capture_rng(device); model.eval(); sums={}; count=0
        try:
            torch.manual_seed(cfg['split_seed']+10000)
            with torch.no_grad():
                for batch in validation:
                    n=len(batch['instruction'])
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        values=model(batch)
                    for key,value in values.items(): sums[key]=sums.get(key,0.)+float(value)*n
                    count+=n
            values={k:v/count for k,v in sums.items()}
            if not all(math.isfinite(v) for v in values.values()): raise FloatingPointError('Nonfinite validation')
            record(dict(split='validation',**values)); return values['flow']
        finally:
            restore_rng(state)
    def save_checkpoint(is_best=False):
        nonlocal best_checkpoint
        name=f'epoch_{epoch:04d}_step_{step:09d}'
        destination=output/'checkpoints'/name
        temp=destination.with_name(name+'.incomplete')
        if destination.exists() or temp.exists():
            raise FileExistsError(f'Checkpoint already exists: {destination}; preserve or move the incomplete directory explicitly')
        if is_best: best_checkpoint=name
        policy.save(temp,step,extra=dict(stage=args.stage,completed_epochs=epoch,cursor=cursor,
                    seen_anchors=seen,teacher_source=teacher_source))
        torch.save(dict(contract=contract,epoch=epoch,cursor=cursor,step=step,seen_anchors=seen,
                        optimizer=optim.state_dict(),best_flow=best,best_checkpoint=best_checkpoint,
                        rng=capture_rng(device),lr_schedule=dict(completed=step,total=total,warmup=cfg['warmup_steps'])),temp/'trainer.pt')
        sha=file_sha256(temp/'model.pt')
        atomic_json(temp/'checkpoint_complete.json',dict(step=step,epoch=epoch,cursor=cursor,model_sha256=sha))
        temp.rename(destination)
        atomic_json(output/'latest.json',dict(path=str(destination),step=step,epoch=epoch,model_sha256=sha))
        if is_best: atomic_json(output/'best_flow.json',dict(path=str(destination),flow=best))
        completed=sorted((output/'checkpoints').glob('epoch_*/checkpoint_complete.json'),key=lambda x:x.parent.name)
        keep={x.parent.name for x in completed[-args.keep_checkpoints:]} | {best_checkpoint}
        for marker in completed:
            if marker.parent.name not in keep: shutil.rmtree(marker.parent)
        return destination
    if not args.resume:
        evaluate()  # Initial validation is diagnostic; best selects trained checkpoints.
    while epoch<args.epochs:
        sampler=EpochBatchSampler(len(train),args.batch_size,seed,epoch,cursor)
        loader=DataLoader(train,batch_sampler=sampler,
            generator=torch.Generator().manual_seed(seed+epoch+20000),**opts)
        for group in accumulation_groups(loader,args.accumulation):
            begin=time.monotonic(); model.train(); optim.zero_grad()
            n_total=sum(len(batch['instruction']) for batch in group); metrics={}; valid_counts={}
            lr=cfg['learning_rate']*learning_rate_factor(step,total,cfg['warmup_steps'])
            for pg in optim.optimizer.param_groups: pg['lr']=lr
            for batch in group:
                n=len(batch['instruction'])
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    values=model(batch); loss=values['loss']*(n/n_total)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                for key,value in values.items(): metrics[key]=metrics.get(key,0.)+float(value.detach())*n/n_total
                for key in ('local_valid','goal_frame_valid'):
                    if key in batch:
                        mask=batch[key]; mask=mask.any(-1) if mask.ndim==2 else mask
                        valid_counts[key]=valid_counts.get(key,0)+int(mask.sum())
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
            optim.step(); step+=1; cursor+=n_total; seen+=n_total
            if args.stage=='II':
                if model.teacher.training or any(p.requires_grad or p.grad is not None for p in model.teacher.parameters()):
                    raise RuntimeError('Teacher freeze violated')
                if any(v._version!=teacher_versions[k] for k,v in teacher_tensors.items()):
                    raise RuntimeError('Teacher parameter/buffer modified')
            record(dict(split='train',lr=lr,grad_norm=float(norm),update_seconds=time.monotonic()-begin,
                        update_anchors=n_total,micro_batches=len(group),valid_counts=valid_counts,**metrics))
            at_end=cursor==len(train)
            if cursor>len(train): raise RuntimeError('Epoch oversampling')
            if at_end:
                epoch+=1; cursor=0
                score=evaluate(); improved=score<best
                if improved: best=score
                save_checkpoint(improved)
            elif args.save_every_updates and step%args.save_every_updates==0:
                save_checkpoint()
            atomic_json(output/'status.json',dict(phase='training',pid=os.getpid(),step=step,
                        epoch=epoch,cursor=cursor,total_updates=total,seen_anchors=seen,time=time.time()))
            if at_end: break
    if step!=total or seen!=len(train)*args.epochs: raise RuntimeError('Final epoch budget mismatch')
    latest=json.loads((output/'latest.json').read_text())
    atomic_json(output/'final.json',latest)
    atomic_json(output/'status.json',dict(phase='completed',epoch=epoch,step=step,seen_anchors=seen,time=time.time()))
