"""Load once, measure real forward/backward memory, and retain the same model."""
import contextlib
import gc
import json
import math
import os
import random
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from a2_local.model import make_stage_two
from a2_local.numerics import configure_attention_backends
from a2_local.preflight import check_teacher, check_complete
from a2_local.config_runtime import source_paths
from ablation.aba1.plus_common.model import GoalLocalPolicy, DistillationSystem
from ablation.aba1.plus_common.data import IndexedFutureDataset
from .runtime import POLICY, CapacityError, resources, process_gpu_mib, worker_physical_gpu


class ProbeGradients:
    """The same CPU gradient offload used by training, without an optimizer step."""
    def __init__(self, parameters):
        self.params = list(parameters)
        self.grads = [None] * len(self.params)
        self.hooks = [p.register_post_accumulate_grad_hook(self.hook(i))
                      for i, p in enumerate(self.params)]

    def hook(self, i):
        def move(param):
            grad = param.grad.detach().to('cpu')
            if self.grads[i] is None:
                self.grads[i] = grad.clone()
            else:
                self.grads[i].add_(grad)
            param.grad = None
        return move

    def close(self):
        for hook in self.hooks:
            hook.remove()
        for param in self.params:
            param.grad = None
        self.grads.clear()


def unique_bytes(module):
    storages = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def prepare(cfg, rank, report, resume=None):
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    configure_attention_backends()
    seed = cfg['train_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    report('loading_cpu')
    sources = source_paths(cfg, 'II')
    check_teacher(sources['teacher'], cfg)
    teacher = GoalLocalPolicy.load(sources['teacher'], with_action=not bool(resume), legacy_config=cfg)
    if resume:
        check_complete(resume, require_resume=True, verify_hash=True)
        student = GoalLocalPolicy.load(resume)
        teacher.requires_grad_(False); teacher.frozen_teacher = True; teacher.eval()
        system = DistillationSystem(student, teacher)
    else:
        system = make_stage_two(teacher, cfg); student = system.student
    for left, right in ((student.qwen.visual, teacher.qwen.visual), (student.dino, teacher.dino)):
        if any(p.requires_grad for p in left.parameters()) or any(p.requires_grad for p in right.parameters()):
            raise ValueError('Only frozen encoders may share storage')
        a, b = left.state_dict(), right.state_dict()
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError('Frozen encoder mismatch')
    teacher.qwen.visual = student.qwen.visual; teacher.dino = student.dino
    if student.cfg != cfg or student.mode != 'student':
        raise ValueError('Student identity changed')
    if student.statistics != json.loads(Path(cfg['statistics_path']).read_text()):
        raise ValueError('Action statistics changed')
    static_mib = math.ceil(unique_bytes(system) / 2**20)
    before = resources()
    if before['gpu_free_mib'] < static_mib + POLICY['gpu_headroom_mib']:
        raise CapacityError('Insufficient GPU memory for model plus headroom before CUDA transfer')
    report('loading_gpu', static_mib=static_mib, resources=before)
    torch.cuda.reset_peak_memory_stats()
    system.to('cuda:0')
    params = [p for p in student.parameters() if p.requires_grad]
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must stay frozen')
    if {p.data_ptr() for p in params} & {p.data_ptr() for p in teacher.parameters()}:
        raise ValueError('Trainable student storage is shared with teacher')
    versions = [p._version for p in system.parameters()]
    dataset = IndexedFutureDataset(cfg['dataset_root'], student.statistics, 'train', cfg['split_seed'], local=False, goal=True)
    # Full future horizon and long instructions exercise the largest fixed image inputs.
    episodes = [e for e in dataset.selected if e['length'] > 88]
    by_instruction = {}
    for episode in episodes:
        by_instruction.setdefault(episode['tasks'][0], episode)
    longest = sorted(by_instruction, key=lambda s: len(student.processor.tokenizer.encode(s)), reverse=True)[:4]
    if len(longest) != 4:
        raise ValueError('Missing representative probe instructions')
    hooks = ProbeGradients(params)
    losses = []
    try:
        for index, instruction in enumerate(longest):
            batch = default_collate([dataset.sample(by_instruction[instruction], 0)])
            if index == 0:
                system.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    validation = system(batch)
                    if not torch.isfinite(validation['loss']):
                        raise FloatingPointError('Nonfinite probe validation')
                del validation
            system.train()
            report('probing_backward', probe=index + 1, probes=len(longest))
            with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                output = system(batch)
                loss = output['loss'] / 32
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite probe loss')
            loss.backward()
            losses.append(float(output['loss'].detach()))
            del batch, output, loss
        if any(g is not None and not torch.isfinite(g).all() for g in hooks.grads):
            raise FloatingPointError('Nonfinite probe gradients')
    finally:
        hooks.close(); dataset.close()
    if versions != [p._version for p in system.parameters()]:
        raise ValueError('Probe changed model parameters')
    del dataset, hooks
    gc.collect()
    # Retain allocator peak plus per-rank workspace while admitting later ranks.
    padding = torch.empty(POLICY['per_rank_padding_mib'] * 2**20, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    current = resources()
    if current['gpu_free_mib'] < POLICY['gpu_headroom_mib']:
        raise CapacityError('Shared GPU no longer has the required headroom')
    trainable = sum(p.numel() for p in params)
    profile = dict(rank=rank, pid=os.getpid(), losses=losses, optimizer_steps=0,
                   process_gpu_mib=process_gpu_mib(os.getpid()),
                   peak_allocated_mib=math.ceil(torch.cuda.max_memory_allocated() / 2**20),
                   peak_reserved_mib=math.ceil(torch.cuda.max_memory_reserved() / 2**20),
                   host_peak_mib=math.ceil(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                   optimizer_headroom_mib=math.ceil(trainable * 16 / 2**20) + POLICY['host_floor_mib'],
                   physical_gpu=worker_physical_gpu(), resources=current, probed_at=time.time())
    return dict(system=system, policy=student, teacher=teacher, padding=padding), profile
