"""Three unique GPUs, FSDP FP32 parameter shards and BF16 computation."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl


def initialize(expected_gpus):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != 3 or local != rank:
        raise ValueError('This single-node run requires exactly three ranks')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str,expected_gpus)):
        raise ValueError('CUDA_VISIBLE_DEVICES must match the three GPUs in the run configuration')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    dist.barrier(device_ids=[local])
    return rank, world, torch.device('cuda', local)


def wrap(policy, device):
    # FP32 resident shards preserve small Adam updates; kernels run in BF16.
    policy.float()
    vision = policy.qwen.visual
    vision.to(device=device, dtype=torch.bfloat16)
    options = dict(device_id=device, mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16, cast_root_forward_inputs=False),
        use_orig_params=False, sync_module_states=True, limit_all_gathers=True)
    for blocks in (policy.qwen.language_model.layers, policy.flow.DiT.transformer_blocks):
        for i, block in enumerate(blocks):
            if policy.cfg['checkpoint_blocks']:
                block = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                                           preserve_rng_state=True)
            blocks[i] = FSDP(block, **options)
    policy.qwen.language_model.embed_tokens = FSDP(policy.qwen.language_model.embed_tokens, **options)
    return FSDP(policy, ignored_modules=[vision], **options)


class ExactRankBatchSampler:
    """No padded sample is counted. Ranks execute equal numbers of collectives.

    Batch size is one per rank. For each full 32-anchor update, ranks 0/1
    process 11 real anchors and rank 2 processes 10 plus one zero-weight
    dummy. A 22-anchor epoch tail uses 8/7/7 real anchors and two dummies.
    Each real loss is weighted world_size/update_count before FSDP averaging.
    """
    def __init__(self, size, rank, seed, epoch, cursor=0, global_batch=32, world=3):
        if not 0 <= rank < world or not 0 <= cursor <= size:
            raise ValueError('Invalid sampler state')
        if cursor != size and cursor % global_batch:
            raise ValueError('Resume cursor must be an optimizer boundary')
        self.size,self.rank,self.seed,self.epoch,self.cursor=size,rank,seed,epoch,cursor
        self.global_batch,self.world=global_batch,world

    def __iter__(self):
        order=torch.randperm(self.size,generator=torch.Generator().manual_seed(
            self.seed+1_000_003*self.epoch)).tolist()
        for start in range(self.cursor,self.size,self.global_batch):
            n=min(self.global_batch,self.size-start)
            for offset in range(0,n,self.world):
                j=offset+self.rank
                real=j<n
                # Dataset wrapper receives (index, weight, global update size, last microstep).
                yield [(order[start+j] if real else order[start],
                        self.world/n if real else 0.,n,offset+self.world>=n)]

    def __len__(self):
        n=self.size-self.cursor
        full,tail=divmod(n,self.global_batch)
        return full*((self.global_batch+self.world-1)//self.world)+(tail+self.world-1)//self.world


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,item):
        index,weight,count,last=item
        batch=self.dataset[index]
        batch.update(sample_weight=weight,update_count=count,last_microstep=last,anchor_index=index)
        return batch
