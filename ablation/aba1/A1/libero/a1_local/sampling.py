"""Exact global batches, distributed 8/8/8/8 across four same-GPU ranks."""


class RankSampler:
    def __init__(self, size, steps, rank, seed, world=4, batch=32, start_step=0):
        if not 0 <= rank < world or steps * batch > size or not 0 <= start_step <= steps:
            raise ValueError("This run must fit within one shuffled epoch")
        self.size, self.steps, self.rank = size, steps, rank
        self.seed, self.world, self.batch = seed, world, batch
        self.start_step = start_step

    def __iter__(self):
        import torch
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed)).tolist()
        for step in range(self.start_step, self.steps):
            indices = order[step*self.batch:(step+1)*self.batch]
            yield from indices[self.rank::self.world]

    def __len__(self):
        return (self.steps-self.start_step) * len(range(self.rank, self.batch, self.world))


def backward_weight(world=4, global_batch=32):
    # DistributedMaster divides summed gradients by world; this produces 1/B.
    return world / global_batch
