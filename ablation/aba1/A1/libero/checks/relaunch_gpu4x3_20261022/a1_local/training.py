"""Four synchronous ranks sharing physical GPU 5; CPU Gloo and rank-zero AdamW."""
import argparse
import copy
import gc
import json
import os
import random
import time
import traceback
from datetime import timedelta
from pathlib import Path
from .config import load_plan, run_path, atomic_json, learning_rate, source_manifest
from .sampling import RankSampler, backward_weight
from .checkpoints import save_weights, verify_teacher


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['I', 'II'], required=True)
    p.add_argument('--experiment-root', required=True)
    args = p.parse_args()
    plan, cfg = load_plan()
    root = run_path(plan, args.experiment_root)
    if os.environ.get('A1_PIPELINE_RUN') != str(root):
        raise RuntimeError('Use launch.py --execute to run the complete pipeline')
    world = int(os.environ.get('WORLD_SIZE', '0'))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '5' or not 1 <= world <= 4:
        raise RuntimeError('One to four ranks on physical GPU 5 are required')
    plan = dict(plan, world_size=world, rank_anchors_per_update=[len(range(r,32,world)) for r in range(world)])
    import torch
    import torch.distributed as dist
    torch.set_num_threads(plan['cpu_threads_per_rank'])
    torch.cuda.set_device(0)
    dist.init_process_group('gloo', timeout=timedelta(minutes=45))
    rank = dist.get_rank()
    output = root / ('stage_' + args.stage)
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(plan, cfg, args.stage, root, output, rank)
    except BaseException:
        atomic_json(output / f'failure_rank_{rank}.json', dict(error=traceback.format_exc(), time=time.time()))
        if rank == 0:
            atomic_json(output / 'status.json', dict(phase='failed', stage=args.stage, error=traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


def run(plan, cfg, stage, root, output, rank):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from ablation.aba1.libero_common.data import FutureDataset
    from ablation.aba1.libero_common.model import GoalLocalPolicy, make_stage_two
    from ablation.aba1.libero_common.training import capture_rng, restore_rng
    from ablation.aba1.plus_common.distributed import DistributedMaster
    from goal_local.train import validation_indices
    spec, device = plan['stage_' + stage], torch.device('cuda:0')
    def seed(value):
        random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed(value)
    teacher_source = None
    if stage == 'II':
        reference = [verify_teacher(root / 'stage_I/checkpoints/step_001000', cfg) if rank == 0 else None]
        dist.broadcast_object_list(reference, src=0)
        teacher_source = reference[0]
    # All Qwen weights are strictly supplied by W0/the completed teacher.
    # Skip random initialization of this fully overwritten HF module only.
    import ablation.aba1.libero_common.model as model_module
    from transformers.initialization import no_init_weights
    original_qwen = model_module.Qwen3VLModel
    def pretrained_qwen(*args, **kwargs):
        with no_init_weights():
            return original_qwen(*args, **kwargs)
    model_module.Qwen3VLModel = pretrained_qwen
    seed(cfg['seed'])
    if stage == 'I':
        model = policy = GoalLocalPolicy.from_baseline(cfg)
    else:
        model = make_stage_two(GoalLocalPolicy.load(teacher_source['path']), cfg)
        policy = model.student
        assert tuple(policy.q_local.shape) == (16, 2048)
        assert not any(p.requires_grad for p in model.teacher.parameters())
        assert not {p.data_ptr() for p in policy.parameters()} & {p.data_ptr() for p in model.teacher.parameters()}
    assert policy.kl == 16 and policy.kg == 0
    assert not hasattr(policy, 'gate') and not hasattr(policy, 'q_goal')
    model.to(device)
    atomic_json(output / f'worker_{rank}.json', dict(rank=rank, pid=os.getpid(), physical_gpu=5,
                logical_gpu=0, stage=stage, phase='model_loaded', cuda_allocated_gib=torch.cuda.memory_allocated()/1024**3))
    gc.collect(); torch.cuda.empty_cache()
    dist.barrier()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = DistributedMaster(parameters, cfg['learning_rate'], cfg['weight_decay'])
    optimizer.broadcast_weights()
    seed(cfg['seed'] + rank)
    frozen = {n: (p, p._version) for n, p in model.named_parameters() if not p.requires_grad}
    if rank == 0:
        atomic_json(output / 'config.json', cfg)
        atomic_json(output / 'run_plan.json', plan)
        atomic_json(output / 'source_manifest.json', source_manifest())
        atomic_json(output / 'parameter_audit.json', dict(trainable=sum(p.numel() for p in parameters),
            frozen=sum(p.numel() for p, _ in frozen.values()), local_slots=16, goal_slots=0, interaction=False,
            distill_layers=cfg['distill_layers'] if stage == 'II' else [], physical_gpu=5, ranks=plan['world_size']))
        atomic_json(output / ('teacher_source.json' if stage == 'II' else 'initialization.json'),
                    teacher_source if stage == 'II' else policy.initialization_report)
    data = FutureDataset(cfg['dataset_root'], policy.statistics, split='train', seed=cfg['split_seed'],
                         val_ratio=cfg['val_ratio'], local=True, goal=False)
    validation = copy.copy(data)
    validation.split = 'val'
    val_ids = set(data.validation_episodes)
    validation.selected = [e for e in data.episodes if e['episode_index'] in val_ids]
    validation.offsets = np.cumsum([0] + [e['length'] for e in validation.selected])
    assert not {e['episode_index'] for e in data.selected} & val_ids
    chosen = validation_indices(validation, plan['validation_anchors'], cfg['split_seed'] + 1)
    if rank == 0:
        atomic_json(output / 'data_audit.json', dict(**data.audit(), local=True, goal=False,
                    future_offsets=[8], train_anchors=len(data), validation_anchors=len(validation),
                    consumed_train_anchors=spec['steps']*plan['global_batch']))
        atomic_json(output / 'validation_indices.json', chosen)
    workers = plan['loader_workers_per_rank']
    options = dict(num_workers=workers, pin_memory=True)
    if workers:
        options.update(multiprocessing_context='spawn', prefetch_factor=1, persistent_workers=True)
    sampler = RankSampler(len(data), spec['steps'], rank, cfg['seed'], plan['world_size'], plan['global_batch'])
    loader = DataLoader(data, batch_size=1, sampler=sampler,
        generator=torch.Generator().manual_seed(cfg['seed']+rank), **options)
    # Validation uses the main process, avoiding a second set of video decoder workers.
    val_loader = DataLoader(Subset(validation, chosen[rank::plan['world_size']]), batch_size=1, num_workers=0)
    fields = ['loss', 'flow'] + (['local_distill'] if stage == 'II' else [])
    step = 0
    started = time.time()
    def status(phase):
        if rank == 0:
            atomic_json(output / 'status.json', dict(phase=phase, stage=stage, step=step, target_steps=spec['steps'],
                seen_train_anchors=step*plan['global_batch'], elapsed_seconds=time.time()-started, updated_at=time.time()))
    def record(row):
        if rank == 0:
            row = dict(row, step=step, stage=stage, time=time.time())
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)
    def evaluate():
        state = capture_rng(device)
        model.eval(); seed(cfg['seed']+10000+rank)
        values = torch.zeros(len(fields)+1, dtype=torch.float64)
        try:
            with torch.no_grad():
                for batch in val_loader:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        losses = model(batch)
                    for i, name in enumerate(fields): values[i] += float(losses[name])
                    values[-1] += 1
            dist.all_reduce(values)
            assert int(values[-1]) == len(chosen) and torch.isfinite(values).all()
            record(dict(split='validation', **{k:float(values[i]/values[-1]) for i,k in enumerate(fields)}))
        finally:
            restore_rng(state); model.train()
    iterator = iter(loader)
    status('training')
    for step in range(1, spec['steps']+1):
        tick = time.monotonic()
        model.train(); optimizer.zero_grad()
        values = torch.zeros(len(fields)+1, dtype=torch.float64)
        for micro in range(plan['rank_anchors_per_update'][rank]):
            batch = next(iterator)
            assert not any(k.startswith('goal_') for k in batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = model(batch)
            assert set(losses) == set(fields) and torch.isfinite(losses['loss'])
            (losses['loss']*backward_weight(plan['world_size'], plan['global_batch'])).backward()
            for i, name in enumerate(fields): values[i] += float(losses[name].detach())
            values[-1] += 1
            atomic_json(output / f'heartbeat_{rank}.json', dict(stage=stage, step=step, micro=micro+1,
                        phase='backward', time=time.time(), peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))
        lr = learning_rate(step-1, spec, cfg['learning_rate'])
        norm = optimizer.step(lr, cfg['gradient_clip'])
        dist.all_reduce(values)
        assert int(values[-1]) == plan['global_batch']
        for name, (parameter, version) in frozen.items():
            if parameter._version != version or parameter.grad is not None:
                raise RuntimeError('Frozen parameter changed: ' + name)
        record(dict(split='train', **{k:float(values[i]/values[-1]) for i,k in enumerate(fields)},
                    learning_rate=lr, grad_norm=norm, seconds=time.monotonic()-tick))
        status('training')
        if step % plan['validation_every'] == 0:
            status('validating'); evaluate(); status('training')
        if step % spec['save_every'] == 0:
            status('saving'); dist.barrier()
            if rank == 0:
                save_weights(policy, output, stage, step, spec, dict(protocol=plan['protocol'],
                    teacher_source=teacher_source, global_batch=plan['global_batch'], world_size=plan['world_size'],
                    run_plan=plan, source_manifest=source_manifest(), frozen_parameters_unchanged=True))
            dist.barrier(); status('training')
    assert next(iterator, None) is None
    dist.barrier(); status('completed')


if __name__ == '__main__':
    main()
