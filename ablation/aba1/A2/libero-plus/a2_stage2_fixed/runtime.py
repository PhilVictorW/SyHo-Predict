"""Resource admission and the selected Stage II runtime contract."""
import json
import math
import os
from pathlib import Path
import subprocess
from a2_local.config_runtime import *
from a2_local.config_runtime import code_identity as original_identity
from a2_local.preflight import plan as original_plan

WORK = Path(__file__).resolve().parents[1]
POLICY = dict(physical_gpus=[2, 5, 6, 7], max_ranks=10, effective_batch=32,
              base_rank_to_gpu=[2, 5, 6, 7, 2, 5, 6, 7, 7, 7], additional_gpu=7,
              rank_policy='fixed_gpu256_two_gpu7_four_10ranks_v2',
              gpu_headroom_mib=2048, per_rank_padding_mib=512,
              memory_factor=1.05, first_gpu_min_mib=16384,
              first_host_min_mib=49152, host_floor_mib=8192,
              observe_seconds=20, poll_seconds=2, worker_timeout_seconds=7200)
CHECKPOINT_POLICY = dict(save_every=100, keep_checkpoints=1,
                         insufficient_space='wait_all_ranks', poll_seconds=30,
                         shared_disk_reserve_gib=24)


class CapacityError(RuntimeError):
    pass


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def worker_physical_gpu():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in {str(gpu) for gpu in POLICY['physical_gpus']}:
        raise ValueError('Each worker must see exactly one allowed GPU: 2,5,6,7')
    return int(visible)


def validate_mapping(mapping):
    if list(mapping) != POLICY['base_rank_to_gpu']:
        raise ValueError('Fixed mapping required: GPUs2,5,6 two ranks each; GPU7 four ranks')


def selected_spec(cfg, world, rank_to_gpu=None):
    if not isinstance(world, int) or not len(POLICY['base_rank_to_gpu']) <= world <= POLICY['max_ranks']:
        raise ValueError('Fixed Stage II requires exactly ten ranks')
    mapping = list(rank_to_gpu) if rank_to_gpu is not None else (
        POLICY['base_rank_to_gpu'] + [POLICY['additional_gpu']] * (world-len(POLICY['base_rank_to_gpu'])))
    validate_mapping(mapping)
    if len(mapping) != world:
        raise ValueError('Selected world size and GPU mapping differ')
    spec = original_plan(cfg, 'II')
    counts = {gpu: mapping.count(gpu) for gpu in dict.fromkeys(mapping)}
    spec.update(world_size=world, ranks_per_gpu=counts,
                physical_gpus=list(counts), rank_to_gpu=mapping,
                save_every=CHECKPOINT_POLICY['save_every'],
                checkpoint_policy=dict(CHECKPOINT_POLICY),
                accumulation=32 // world if 32 % world == 0 else None,
                accumulation_per_rank=[len(range(rank, 32, world)) for rank in range(world)],
                model_loading='sequential_measured_admission',
                rank_policy=POLICY['rank_policy'],
                configured_world_size=cfg['a2_training']['world_size'])
    return spec


def rank_device(cfg, rank):
    if not 0 <= rank < int(os.environ['WORLD_SIZE']):
        raise ValueError('Invalid adaptive Stage II rank/device')
    worker_physical_gpu()
    return 0


def code_identity():
    identity = original_identity()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        identity[str(path.relative_to(ROOT))] = file_sha256(path)
    for name in ('start_stage_ii_adaptive.sh', 'run_stage_ii_logged.sh', 'start_stage_ii_fixed10.sh'):
        identity[str((WORK / name).relative_to(ROOT))] = file_sha256(WORK / name)
    return identity


def resources_many(physical_gpus):
    gpus = list(physical_gpus)
    if not gpus or any(g not in POLICY['physical_gpus'] for g in gpus):
        raise ValueError('Resource query outside allowed GPUs')
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    host = int(memory['MemAvailable'].split()[0]) // 1024
    result = {}
    for line in output.splitlines():
        gpu, free, total = map(int, line.split(','))
        if gpu in gpus:
            result[gpu] = dict(physical_gpu=gpu, gpu_free_mib=free,
                              gpu_total_mib=total, host_available_mib=host)
    if set(result) != set(gpus):
        raise RuntimeError('Missing requested GPU resource snapshot')
    return result


def resources(physical_gpu=None):
    gpu = worker_physical_gpu() if physical_gpu is None else physical_gpu
    return resources_many([gpu])[gpu]


def process_gpu_mib(pid):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(worker_physical_gpu()), '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        fields = line.split(',')
        if len(fields) == 2 and fields[0].strip() == str(pid):
            return int(fields[1])
    raise RuntimeError('Selected GPU process memory is unavailable')


def admission(resources, profiles):
    if not profiles:
        gpu_needed = POLICY['first_gpu_min_mib']
        host_needed = POLICY['first_host_min_mib']
    else:
        gpu_needed = math.ceil(max(p['process_gpu_mib'] for p in profiles)
                               * POLICY['memory_factor']) + POLICY['gpu_headroom_mib']
        host_needed = math.ceil(max(p['host_peak_mib'] for p in profiles)
                                * POLICY['memory_factor']) + max(p['optimizer_headroom_mib'] for p in profiles)
    return dict(allowed=resources['gpu_free_mib'] >= gpu_needed
                and resources['host_available_mib'] >= host_needed,
                next_gpu_required_mib=gpu_needed, next_host_required_mib=host_needed,
                **resources)
