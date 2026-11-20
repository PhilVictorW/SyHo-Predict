"""Exact global batches, distributed 8/8/8/8 across four same-GPU ranks."""


class RankSampler:
    def __init__(self, size, steps, rank, seed, world=4, batch=32):
        if not 0 <= rank < world or steps * batch > size:
            raise ValueError("This run must fit within one shuffled epoch")
        self.size, self.steps, self.rank = size, steps, rank
        self.seed, self.world, self.batch = seed, world, batch

    def __iter__(self):
        import torch
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed)).tolist()
        for step in range(self.steps):
            indices = order[step*self.batch:(step+1)*self.batch]
            yield from indices[self.rank::self.world]

    def __len__(self):
        return self.steps * len(range(self.rank, self.batch, self.world))


def backward_weight(world=4, global_batch=32):
    # DistributedMaster divides summed gradients by world; this produces 1/B.
    return world / global_batch
