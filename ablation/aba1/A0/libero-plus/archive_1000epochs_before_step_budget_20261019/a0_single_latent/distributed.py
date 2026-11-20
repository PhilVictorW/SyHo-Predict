"""Four unique GPUs, FSDP FP32 parameter shards and BF16 computation."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl


def initialize():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != 4 or local != rank:
        raise ValueError('This single-node run requires exactly four ranks')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,3,7':
        raise ValueError('Expected the authorized physical GPUs 0,1,3,7')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    dist.barrier(device_ids=[local])
    return rank, world, torch.device('cuda', local)


def wrap(policy, device):
    # FP32 resident shards preserve small Adam updates; kernels run in BF16.
    policy.float()
    vision = policy.qwen.visual
    dino = policy.dino
    vision.to(device=device, dtype=torch.bfloat16)
    dino.to(device=device, dtype=torch.bfloat16)
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
    return FSDP(policy, ignored_modules=[vision,dino], **options)


class ExactRankBatchSampler:
    """Exactly one permutation per epoch, zero-weight padding at the last microstep.

    All ranks execute the same number of collectives. Each real anchor has
    weight world / real_update_size, compensating for FSDP gradient averaging.
    """
    def __init__(self, size, rank, seed, epoch, cursor=0, global_batch=32, world=4, micro_batch=4):
        if not 0 <= rank < world or not 0 <= cursor <= size:
            raise ValueError('Invalid sampler state')
        if cursor != size and cursor % global_batch:
            raise ValueError('Resume cursor must be an optimizer boundary')
        self.size,self.rank,self.seed,self.epoch,self.cursor=size,rank,seed,epoch,cursor
        if micro_batch<1 or global_batch%(world*micro_batch):raise ValueError('Invalid microbatch size')
        self.global_batch,self.world,self.micro_batch=global_batch,world,micro_batch

    def __iter__(self):
        order=torch.randperm(self.size,generator=torch.Generator().manual_seed(
            self.seed+1_000_003*self.epoch)).tolist()
        for start in range(self.cursor,self.size,self.global_batch):
            n=min(self.global_batch,self.size-start)
            width=self.world*self.micro_batch
            for offset in range(0,n,width):
                batch=[]
                for k in range(self.micro_batch):
                    j=offset+self.rank*self.micro_batch+k
                    real=j<n
                    batch.append((order[start+j] if real else order[start],
                                  self.world/n if real else 0.,n,offset+width>=n))
                yield batch

    def __len__(self):
        n=self.size-self.cursor
        full,tail=divmod(n,self.global_batch)
        width=self.world*self.micro_batch
        return full*(self.global_batch//width)+(tail+width-1)//width


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,item):
        index,weight,count,last=item
        batch=self.dataset[index]
        batch.update(sample_weight=weight,update_count=count,last_microstep=last,anchor_index=index)
        return batch
