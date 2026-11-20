"""Checkpoint-boundary resizing with a world-size independent sample stream."""
import copy

class GlobalBatches:
    def __init__(self,dataset,world,rank,seed):
        self.size=len(dataset);self.world=world;self.rank=rank;self.seed=seed
        self.updates_per_epoch=self.size//32
        self.local_count=len(range(rank,32,world));self.epoch=0;self.start=0
    def reset(self,epoch,start=0):
        if epoch<0 or not 0<=start<=self.updates_per_epoch*self.local_count:
            raise ValueError("Invalid global sampler cursor")
        self.epoch=epoch;self.start=start
    def resume_step(self,step):
        epoch,update=divmod(step,self.updates_per_epoch)
        self.reset(epoch,update*self.local_count)
        return self.epoch,self.start
    def __len__(self):return self.updates_per_epoch*self.local_count-self.start
    def __iter__(self):
        import torch
        order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch))
        update,micro=divmod(self.start,self.local_count)
        for u in range(update,self.updates_per_epoch):
            for j,slot in enumerate(range(self.rank,32,self.world)):
                if u==update and j<micro:continue
                yield [int(order[u*32+slot])]
